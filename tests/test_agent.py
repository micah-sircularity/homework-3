import httpx
import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import UsageLimits

from wikipedia_agent import WikipediaClient, WikipediaDeps, create_agent
from wikipedia_agent.cli import verification_footer


def wikipedia_handler(request):
    if request.url.path == "/w/api.php":
        return httpx.Response(
            200,
            json={
                "query": {"search": [{"title": "Capybara", "pageid": 123, "snippet": "A rodent."}]}
            },
        )
    return httpx.Response(200, text="The capybara is a rodent.")


def scripted_model(responses):
    iterator = iter(responses)

    def respond(messages, info):
        assert {tool.name for tool in info.function_tools} == {"search", "get_page"}
        return ModelResponse(next(iterator))

    return FunctionModel(respond)


async def test_search_fetch_answer_and_deduplicated_evidence():
    model = scripted_model(
        [
            [ToolCallPart("search", {"query": "capybara habitat"})],
            [ToolCallPart("get_page", {"title": "Capybara"})],
            [ToolCallPart("get_page", {"title": "Capybara"})],
            [TextPart("It is a rodent. https://en.wikipedia.org/wiki/Capybara")],
        ]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(wikipedia_handler)) as http:
        deps = WikipediaDeps(WikipediaClient(http))
        result = await create_agent(model).run("What is a capybara?", deps=deps)
    assert "rodent" in result.output
    assert deps.usage.successful_searches == ["capybara habitat"]
    assert len(deps.usage.fetched_pages) == 1
    footer = verification_footer(deps.usage)
    assert "1 pages fetched" in footer
    assert footer.count("https://en.wikipedia.org/wiki/Capybara") == 1


async def test_output_retry_can_recover_by_fetching_page():
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse([TextPart("An unsupported answer")])
        if calls == 2:
            assert any(
                isinstance(part, RetryPromptPart) and "Call get_page" in part.content
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
            return ModelResponse([ToolCallPart("get_page", {"title": "Capybara"})])
        return ModelResponse([TextPart("A rodent, according to Wikipedia.")])

    async with httpx.AsyncClient(transport=httpx.MockTransport(wikipedia_handler)) as http:
        deps = WikipediaDeps(WikipediaClient(http))
        result = await create_agent(FunctionModel(respond)).run("What is it?", deps=deps)
    assert calls == 3
    assert result.output == "A rodent, according to Wikipedia."
    assert deps.usage.fetched_pages


@pytest.mark.parametrize("mode", ["no_tools", "search_only", "failed_fetch", "empty_page"])
async def test_unverified_answers_are_rejected(mode):
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1 and mode == "search_only":
            return ModelResponse([ToolCallPart("search", {"query": "capybara"})])
        if calls == 1 and mode in {"failed_fetch", "empty_page"}:
            return ModelResponse([ToolCallPart("get_page", {"title": "Capybara"})])
        return ModelResponse([TextPart("Wikipedia use verified: I know the answer.")])

    def handler(request):
        if mode == "failed_fetch":
            return httpx.Response(404)
        if mode == "empty_page":
            return httpx.Response(200, text="")
        return wikipedia_handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        deps = WikipediaDeps(WikipediaClient(http))
        with pytest.raises(UnexpectedModelBehavior, match="output retries"):
            await create_agent(FunctionModel(respond)).run("What is it?", deps=deps)
    assert not deps.usage.fetched_pages
    assert calls == (3 if mode == "no_tools" else 4)
    if mode == "search_only":
        assert deps.usage.successful_searches == ["capybara"]


async def test_evidence_does_not_leak_when_agent_and_dependencies_are_reused():
    model = scripted_model(
        [
            [ToolCallPart("get_page", {"title": "Capybara"})],
            [TextPart("A verified first answer")],
            [TextPart("Unverified second answer")],
            [TextPart("Still unverified")],
            [TextPart("Still unverified")],
        ]
    )
    agent = create_agent(model)
    async with httpx.AsyncClient(transport=httpx.MockTransport(wikipedia_handler)) as http:
        deps = WikipediaDeps(WikipediaClient(http))
        await agent.run("First question", deps=deps)
        first_usage = deps.usage
        with pytest.raises(UnexpectedModelBehavior):
            await agent.run("Second question", deps=deps)
    assert first_usage.fetched_pages
    assert first_usage is not deps.usage
    assert not deps.usage.fetched_pages


async def test_tool_error_retry_can_recover():
    requests = 0

    def handler(request):
        nonlocal requests
        requests += 1
        return httpx.Response(503) if requests == 1 else wikipedia_handler(request)

    model = scripted_model(
        [
            [ToolCallPart("get_page", {"title": "Capybara"})],
            [ToolCallPart("get_page", {"title": "Capybara"})],
            [TextPart("Recovered answer")],
        ]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        deps = WikipediaDeps(WikipediaClient(http))
        result = await create_agent(model).run("Question", deps=deps)
    assert requests == 2
    assert result.output == "Recovered answer"
    assert deps.usage.fetched_pages
    assert any(
        isinstance(part, RetryPromptPart) and part.tool_name == "get_page"
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_model_request_limit_stops_repeated_searches():
    def respond(messages, info):
        return ModelResponse([ToolCallPart("search", {"query": "capybara"})])

    async with httpx.AsyncClient(transport=httpx.MockTransport(wikipedia_handler)) as http:
        deps = WikipediaDeps(WikipediaClient(http))
        with pytest.raises(UsageLimitExceeded):
            await create_agent(FunctionModel(respond)).run(
                "Question", deps=deps, usage_limits=UsageLimits(request_limit=10)
            )
    assert len(deps.usage.successful_searches) == 10


async def test_tool_retries_are_bounded():
    def respond(messages, info):
        return ModelResponse([ToolCallPart("get_page", {"title": "Missing"})])

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(404))
    ) as http:
        deps = WikipediaDeps(WikipediaClient(http))
        with pytest.raises(UnexpectedModelBehavior, match="retries"):
            await create_agent(FunctionModel(respond)).run("Question", deps=deps)
    assert not deps.usage.fetched_pages
