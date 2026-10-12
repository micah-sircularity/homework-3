"""Reusable Wikipedia research agent and capability."""

from wikipedia_agent.agent import create_agent
from wikipedia_agent.capabilities import WikipediaCapability
from wikipedia_agent.helpers.client import WikipediaClient
from wikipedia_agent.helpers.dependencies import WikipediaDeps, WikipediaUsage

__all__ = [
    "WikipediaCapability",
    "WikipediaClient",
    "WikipediaDeps",
    "WikipediaUsage",
    "create_agent",
]
