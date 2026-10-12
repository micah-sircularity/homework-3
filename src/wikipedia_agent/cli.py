"""One-question command-line entry point."""

import argparse
import asyncio
import sys

import httpx
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from wikipedia_agent.agent import create_agent
from wikipedia_agent.helpers.client import WikipediaClient
from wikipedia_agent.helpers.config import Settings
from wikipedia_agent.helpers.dependencies import WikipediaDeps, WikipediaUsage


def verification_footer(usage: WikipediaUsage) -> str:
    sources = list(usage.fetched_pages)
    return (
        f"Wikipedia use verified: {len(sources)} pages fetched\n"
        + "Sources:\n"
        + "\n".join(f"- {url}" for url in sources)
    )


async def answer_question(question: str, settings: Settings) -> tuple[str, WikipediaUsage]:
    agent = create_agent(model_name=settings.model, api_key=settings.api_key)
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http:
        deps = WikipediaDeps(client=WikipediaClient(http))
        result = await agent.run(question, deps=deps, usage_limits=UsageLimits(request_limit=10))
        return result.output, deps.usage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer a question using verified Wikipedia retrieval."
    )
    parser.add_argument("question", help="One question, enclosed in quotes")
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("question must not be blank")
    try:
        settings = Settings.from_env()
        output, usage = asyncio.run(answer_question(args.question.strip(), settings))
    except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
        print(f"Error: No verified answer could be completed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(output)
    print()
    print(verification_footer(usage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
