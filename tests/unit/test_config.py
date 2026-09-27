import pytest
from pydantic import ValidationError

from harness.config import Settings


def test_settings_are_read_from_prefixed_env_vars(monkeypatch, tmp_path):
    monkeypatch.setenv("HARNESS_KB_TOP_K", "1")
    monkeypatch.setenv("HARNESS_MODEL_TEMPERATURE", "0")
    monkeypatch.setenv("HARNESS_DATA_DIR", str(tmp_path))

    settings = Settings(_env_file=None)

    assert settings.kb_top_k == 1
    assert settings.model_temperature == 0.0
    assert settings.data_dir == tmp_path


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("HARNESS_TOOL_TIMEOUT_S", "0"),
        ("HARNESS_TOOL_MAX_ATTEMPTS", "0"),
        ("HARNESS_MODEL_TEMPERATURE", "3"),
        ("HARNESS_LLM_BACKEND", "openai"),
        ("HARNESS_LOG_LEVEL", "LOUD"),
    ],
)
def test_invalid_settings_fail_fast(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
