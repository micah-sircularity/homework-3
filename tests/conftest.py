import pytest
from pydantic_ai import models


@pytest.fixture(autouse=True)
def disable_live_model_requests(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
