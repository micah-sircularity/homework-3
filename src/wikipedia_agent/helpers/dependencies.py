"""Dependencies and successful tool-use evidence for a single run."""

from dataclasses import dataclass, field

from wikipedia_agent.helpers.client import WikipediaClient, WikipediaPage


@dataclass
class WikipediaUsage:
    successful_searches: list[str] = field(default_factory=list)
    fetched_pages: dict[str, WikipediaPage] = field(default_factory=dict)


@dataclass
class WikipediaDeps:
    client: WikipediaClient
    usage: WikipediaUsage = field(default_factory=WikipediaUsage)
