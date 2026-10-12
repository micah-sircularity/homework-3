"""Explicit OpenRouter configuration, loaded only when the CLI runs."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        model = os.environ.get("OPENROUTER_MODEL", "").strip()
        missing = [
            name
            for name, value in (("OPENROUTER_API_KEY", key), ("OPENROUTER_MODEL", model))
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing configuration: {', '.join(missing)}. Set it in .env or the environment."
            )
        return cls(api_key=key, model=model)
