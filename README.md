# Wikipedia agent

A uv-managed Python CLI using Pydantic AI and OpenRouter. The agent formulates Wikipedia
search queries, retrieves raw pages, and answers your question with source links.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer. uv can install the
project's pinned Python automatically.

```bash
uv sync --locked
cp .env.example .env
```

Set both values in `.env` (or export them in your shell):

```dotenv
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=provider/model-name
```

Replace `provider/model-name` with an actual OpenRouter model ID that supports function
tools. There is no default model. Exported environment variables take precedence over
`.env`; `.env` is ignored by Git. Model requests use your OpenRouter account and may incur costs.

## Ask a question

```bash
uv run wikipedia-agent "How does the lesser capybara differ from the capybara?"
```

One quoted question is accepted per invocation. Conversation history is not persisted.
The final answer is followed by a footer generated from successful tool calls:

```text
Wikipedia use verified: 2 pages fetched
Sources:
- https://en.wikipedia.org/wiki/Lesser_capybara
- https://en.wikipedia.org/wiki/Capybara
```

The count is the number of distinct fetched article URLs, not total fetch calls. Sources
in the footer list all successfully fetched pages; the agent's answer is instructed to
cite the pages it uses.

## How verification works

`get_page` records evidence only after a successful, nonempty raw-page response. A
final-output validator rejects answers without a successful page fetch in that run and
asks the agent to retrieve a relevant page. Search calls alone do not qualify, and the
model cannot satisfy the check by writing a verification claim in its answer.

Output validation allows two retries, tool failures allow two retries per tool, and the
CLI allows at most ten model requests per run. If no verified answer can be completed,
the command exits nonzero, writes an error to stderr, and prints no answer or footer.
Argument errors use exit code 2; configuration and run failures use exit code 1.

Verification proves a Wikipedia page was retrieved during the run. It does **not** prove
that the page was relevant or that every answer claim is supported. No second model is
used to review claims.

## Reusable structure

```text
src/wikipedia_agent/
  helpers/
    client.py        # Shared async HTTP access and typed results
    config.py        # Environment/.env configuration
    dependencies.py  # Caller-owned client and per-run usage evidence
  tools/             # search(query) and get_page(title) agent adapters
  capabilities/      # WikipediaCapability: toolset, instructions, evidence reset hook
  agent.py           # OpenRouter agent factory and final-output validator
  cli.py             # One-shot CLI, request budget, and evidence footer
```

`WikipediaClient` works without an agent. Search returns Wikipedia’s default batch of up to ten matches with titles, page IDs,
plain-text snippets, and URLs. It makes one request and does not follow pagination. Get page returns the requested title, article URL, and
raw wikitext. Both endpoints are English Wikipedia; requests use a 20-second timeout,
HTTPX parameter encoding, and this User-Agent:

```text
ai-engineering-buildcamp/1.0 (educational project)
```

Use the verified agent from another async application:

```python
import asyncio

import httpx
from pydantic_ai.usage import UsageLimits

from wikipedia_agent import WikipediaClient, WikipediaDeps, create_agent
from wikipedia_agent.helpers.config import Settings


async def main():
    settings = Settings.from_env()
    agent = create_agent(model_name=settings.model, api_key=settings.api_key)
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http:
        deps = WikipediaDeps(WikipediaClient(http))
        result = await agent.run(
            "Where do capybaras live?",
            deps=deps,
            usage_limits=UsageLimits(request_limit=10),
        )
        print(result.output)
        print(list(deps.usage.fetched_pages))


asyncio.run(main())
```

Pass a Pydantic AI `Model` directly to `create_agent(model=...)` for other providers or
offline tests. Create fresh `WikipediaDeps` for each concurrent run. The capability
resets evidence at the start of every run, including sequential reuse of dependencies.
The HTTP client belongs to the caller and must stay open throughout the run.

`WikipediaCapability` can also be attached to your own `Agent` with
`deps_type=WikipediaDeps`. It supplies tools, research instructions, and the reset hook;
the final-output verification validator belongs to `create_agent`. Use that factory
when you need the verification gate. Library callers should explicitly supply their
desired `UsageLimits`; the ten-request limit is enforced by the CLI.

## Tests and checks

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Tests use mocked HTTP and Pydantic AI function models, with live model requests disabled.
They cover endpoint parameters and headers, result parsing, request errors, tool and
output retries, the ten-request limit, evidence isolation, and CLI errors/output.

For an optional live smoke test, configure `.env` and run the example question above.
Confirm the answer includes the generated verification footer and open the source links
to inspect the retrieved articles. Live success depends on your model supporting tool
calls, OpenRouter credentials/credits, and Wikipedia availability.

API references: [Pydantic AI OpenRouter](https://pydantic.dev/docs/ai/models/openrouter/),
[custom capabilities](https://pydantic.dev/docs/ai/capabilities/custom/).
