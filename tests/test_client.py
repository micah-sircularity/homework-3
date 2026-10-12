import httpx
import pytest

from wikipedia_agent.helpers.client import USER_AGENT, WikipediaClient, WikipediaError


async def test_search_encoding_headers_and_parsing():
    def handler(request):
        assert request.url.path == "/w/api.php"
        assert dict(request.url.params) == {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": "capybara & lesser + café",
        }
        assert request.headers["User-Agent"] == USER_AGENT
        assert request.extensions["timeout"]["read"] == 20.0
        return httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {
                            "title": "Lesser capybara",
                            "pageid": 123,
                            "snippet": (
                                'A <span class="searchmatch">small</span> &amp; rare rodent.'
                            ),
                        }
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        results = await WikipediaClient(http).search(" capybara & lesser + café ")
    assert results[0].model_dump() == {
        "title": "Lesser capybara",
        "page_id": 123,
        "snippet": "A small & rare rodent.",
        "url": "https://en.wikipedia.org/wiki/Lesser_capybara",
    }


async def test_raw_page_encoding_and_headers():
    def handler(request):
        assert request.url.path == "/w/index.php"
        assert dict(request.url.params) == {"title": "Café & A+B", "action": "raw"}
        assert request.headers["User-Agent"] == USER_AGENT
        return httpx.Response(200, text="== Overview ==\nA [[rodent]].")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        page = await WikipediaClient(http).get_page("Café & A+B")
    assert page.wikitext == "== Overview ==\nA [[rodent]]."
    assert page.url == "https://en.wikipedia.org/wiki/Caf%C3%A9_%26_A%2BB"


async def test_empty_search_is_successful():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"query": {"search": []}}))
    ) as http:
        assert await WikipediaClient(http).search("no matches") == []


async def test_search_uses_default_batch_and_does_not_follow_continuation():
    requests = 0

    def handler(request):
        nonlocal requests
        requests += 1
        assert "srlimit" not in request.url.params
        assert "sroffset" not in request.url.params
        return httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {"title": f"Page {i}", "pageid": i, "snippet": "A match."}
                        for i in range(10)
                    ]
                },
                "continue": {"sroffset": 10, "continue": "-||"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        results = await WikipediaClient(http).search("capybara")
    assert requests == 1
    assert len(results) == 10


@pytest.mark.parametrize("method", ["search", "get_page"])
async def test_blank_input_never_makes_request(method):
    def handler(_):
        pytest.fail("Blank input must not reach Wikipedia")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(WikipediaError, match="must not be blank"):
            await getattr(WikipediaClient(http), method)(" \n ")


@pytest.mark.parametrize(
    "response,expected",
    [
        (httpx.Response(200, text="not JSON"), "malformed search"),
        (httpx.Response(200, json=[]), "malformed search"),
        (httpx.Response(200, json={"query": {"search": {}}}), "malformed search"),
        (httpx.Response(200, json={"query": {"search": [{"title": 3}]}}), "malformed search"),
        (httpx.Response(200, json={"error": {"code": "badquery"}}), "API error: badquery"),
    ],
)
async def test_bad_search_responses(response, expected):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as http:
        with pytest.raises(WikipediaError, match=expected):
            await WikipediaClient(http).search("capybara")


@pytest.mark.parametrize(
    "response,expected",
    [
        (httpx.Response(404), "not found"),
        (httpx.Response(429), "HTTP 429"),
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, text=" \n"), "empty page"),
        (
            httpx.Response(
                200, text="<html>challenge</html>", headers={"content-type": "text/html"}
            ),
            "unexpected raw page",
        ),
    ],
)
async def test_bad_page_responses(response, expected):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as http:
        with pytest.raises(WikipediaError, match=expected):
            await WikipediaClient(http).get_page("Capybara")


@pytest.mark.parametrize(
    "error,expected",
    [
        (httpx.ReadTimeout("timeout"), "timed out"),
        (httpx.ConnectError("unreachable"), "connect to Wikipedia"),
    ],
)
async def test_transport_errors(error, expected):
    def handler(_):
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(WikipediaError, match=expected):
            await WikipediaClient(http).get_page("Capybara")
