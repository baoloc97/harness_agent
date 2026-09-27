"""Runtime configuration, read from environment variables (prefix HARNESS_) or .env.

Values are validated at startup, so a bad setting fails fast instead of misbehaving at run time.
Google credentials (GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, ...) are read by ADK directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HARNESS_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://harness:harness@localhost:5432/harness"

    # LLM
    llm_backend: Literal["gemini", "rule_based"] = "gemini"
    model: str = "gemini-2.5-flash"
    model_temperature: float = Field(0.2, ge=0.0, le=2.0)
    model_http_retry_attempts: int = Field(3, ge=1, le=10)
    model_http_retry_initial_delay_s: float = Field(1.0, ge=0.0)

    # Execution limits
    max_llm_calls: int = Field(12, ge=1)
    max_tool_calls: int = Field(10, ge=1)
    max_identical_tool_calls: int = Field(2, ge=1)
    run_time_budget_s: float = Field(60.0, gt=0)
    max_model_retries: int = Field(2, ge=0)

    # Tool resilience
    tool_timeout_s: float = Field(5.0, gt=0)
    tool_max_attempts: int = Field(3, ge=1, le=10)
    tool_backoff_base_s: float = Field(0.5, ge=0.0)
    tool_reflection_retries: int = Field(1, ge=0, le=5)

    # Mock tools
    data_dir: Path = PROJECT_ROOT / "data"
    kb_top_k: int = Field(3, ge=1, le=20)
    mock_faults: str = ""

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    trace_max_str_len: int = Field(2000, ge=100)


@lru_cache
def get_settings() -> Settings:
    return Settings()
