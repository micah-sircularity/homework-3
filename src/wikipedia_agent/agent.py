"""Agent factory and final-output verification."""

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from wikipedia_agent.capabilities import WikipediaCapability
from wikipedia_agent.helpers.dependencies import WikipediaDeps


def create_agent(
    model: Model | None = None, *, model_name: str | None = None, api_key: str | None = None
) -> Agent[WikipediaDeps, str]:
    """Build a verified agent; pass a Model directly for offline tests or embedding."""
    if model is None:
        if not model_name or not api_key:
            raise ValueError("An OpenRouter model name and API key are required.")
        model = OpenRouterModel(model_name, provider=OpenRouterProvider(api_key=api_key))
    agent = Agent(
        model,
        deps_type=WikipediaDeps,
        capabilities=[WikipediaCapability()],
        retries={"tools": 2, "output": 2},
    )

    @agent.output_validator
    def verify_wikipedia(ctx: RunContext[WikipediaDeps], output: str) -> str:
        if not ctx.deps.usage.fetched_pages:
            raise ModelRetry(
                "Wikipedia use is not verified. Call get_page for a relevant Wikipedia "
                "page and retrieve nonempty content before giving your final answer."
            )
        return output

    return agent
