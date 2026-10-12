"""Thin agent adapters around the reusable Wikipedia client."""

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from wikipedia_agent.helpers.client import SearchResult, WikipediaError, WikipediaPage
from wikipedia_agent.helpers.dependencies import WikipediaDeps

wikipedia_toolset = FunctionToolset[WikipediaDeps](max_retries=2)


@wikipedia_toolset.tool
async def search(ctx: RunContext[WikipediaDeps], query: str) -> list[SearchResult]:
    """Search English Wikipedia using its default batch of up to ten results.

    Args:
        query: A search query you formulate for the user's question.
    """
    try:
        results = await ctx.deps.client.search(query)
    except WikipediaError as exc:
        raise ModelRetry(str(exc)) from exc
    ctx.deps.usage.successful_searches.append(query.strip())
    return results


@wikipedia_toolset.tool
async def get_page(ctx: RunContext[WikipediaDeps], title: str) -> WikipediaPage:
    """Fetch the raw wikitext of an English Wikipedia page.

    Args:
        title: The exact page title, usually obtained from search results.
    """
    try:
        page = await ctx.deps.client.get_page(title)
    except WikipediaError as exc:
        raise ModelRetry(str(exc)) from exc
    ctx.deps.usage.fetched_pages[page.url] = page
    return page
