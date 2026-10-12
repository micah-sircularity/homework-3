import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from wikipedia_agent import cli
from wikipedia_agent.helpers.client import WikipediaPage
from wikipedia_agent.helpers.config import Settings
from wikipedia_agent.helpers.dependencies import WikipediaUsage


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    monkeypatch.setattr("wikipedia_agent.helpers.config.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)


def test_missing_configuration_exits_without_output(capsys):
    assert cli.main(["Question"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "OPENROUTER_API_KEY" in captured.err
    assert "OPENROUTER_MODEL" in captured.err


@pytest.mark.parametrize("args", [[], ["   "], ["two", "arguments"]])
def test_invalid_cli_arguments(args, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""


def test_successful_cli_output_includes_recorded_sources(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "provider/model")
    url = "https://en.wikipedia.org/wiki/Capybara"

    async def answer(question, settings):
        assert question == "What is a capybara?"
        assert settings == Settings("test-secret", "provider/model")
        page = WikipediaPage(title="Capybara", url=url, wikitext="A rodent.")
        return "A rodent.", WikipediaUsage(fetched_pages={url: page})

    monkeypatch.setattr(cli, "answer_question", answer)
    assert cli.main([" What is a capybara? "]) == 0
    captured = capsys.readouterr()
    assert captured.out == (
        f"A rodent.\n\nWikipedia use verified: 1 pages fetched\nSources:\n- {url}\n"
    )
    assert captured.err == ""


@pytest.mark.parametrize(
    "error",
    [UnexpectedModelBehavior("Exceeded output retries"), RuntimeError("Provider unavailable")],
)
def test_failed_run_prints_no_answer_or_verification(error, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "provider/model")

    async def answer(question, settings):
        raise error

    monkeypatch.setattr(cli, "answer_question", answer)
    assert cli.main(["Question"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err


def test_environment_values_are_trimmed_and_key_is_not_in_repr(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", " test-secret ")
    monkeypatch.setenv("OPENROUTER_MODEL", " provider/model ")
    settings = Settings.from_env()
    assert settings.api_key == "test-secret"
    assert settings.model == "provider/model"
    assert "test-secret" not in repr(settings)
