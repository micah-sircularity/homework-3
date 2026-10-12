"""Wikipedia HTTP access, independent of Pydantic AI."""

from html.parser import HTMLParser
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

USER_AGENT = "ai-engineering-buildcamp/1.0 (educational project)"
SEARCH_URL = "https://en.wikipedia.org/w/api.php"
PAGE_URL = "https://en.wikipedia.org/w/index.php"


class WikipediaError(Exception):
    """A Wikipedia request could not produce usable data."""


class SearchResult(BaseModel):
    title: str
    page_id: int
    snippet: str
    url: str


class WikipediaPage(BaseModel):
    title: str
    url: str
    wikitext: str


class _SnippetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def article_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='')}"


def _nonblank(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise WikipediaError(f"{label} must not be blank.")
    return value


class WikipediaClient:
    """Uses a caller-owned HTTP client; always sends Wikipedia's required User-Agent."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self.http = http

    async def _get(self, url: str, params: dict[str, str | int]) -> httpx.Response:
        try:
            response = await self.http.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20.0
            )
            response.raise_for_status()
            return response
        except httpx.TimeoutException as exc:
            raise WikipediaError("Wikipedia request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            message = (
                "Wikipedia page was not found."
                if status == 404
                else (f"Wikipedia returned HTTP {status}.")
            )
            raise WikipediaError(message) from exc
        except httpx.RequestError as exc:
            raise WikipediaError("Could not connect to Wikipedia.") from exc

    async def search(self, query: str) -> list[SearchResult]:
        query = _nonblank(query, "Search query")
        response = await self._get(
            SEARCH_URL,
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
            },
        )
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Expected an object")
            if "error" in data:
                error = data["error"]
                code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
                raise WikipediaError(f"Wikipedia API error: {code}.")
            matches = data["query"]["search"]
            if not isinstance(matches, list):
                raise ValueError("Expected a result list")
            results = []
            for match in matches:
                title = _nonblank(match["title"], "Page title")
                parser = _SnippetParser()
                parser.feed(match["snippet"])
                results.append(
                    SearchResult(
                        title=title,
                        page_id=match["pageid"],
                        snippet="".join(parser.parts),
                        url=article_url(title),
                    )
                )
            return results
        except (ValueError, KeyError, TypeError, AttributeError, ValidationError) as exc:
            raise WikipediaError("Wikipedia returned a malformed search response.") from exc

    async def get_page(self, title: str) -> WikipediaPage:
        title = _nonblank(title, "Page title")
        response = await self._get(PAGE_URL, {"title": title, "action": "raw"})
        # Reject HTML error/challenge pages returned with HTTP 200.
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type or "application/json" in content_type:
            raise WikipediaError("Wikipedia returned an unexpected raw page response.")
        if not response.text.strip():
            raise WikipediaError("Wikipedia returned an empty page.")
        return WikipediaPage(title=title, url=article_url(title), wikitext=response.text)
