"""
Application settings and configuration.

Design decision: Centralize all configuration in a Pydantic Settings model
so that environment variables, defaults, and overrides are type-checked.
This avoids magic strings scattered across the codebase.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings for the recruitment system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RECRUITMENT_",
        extra="ignore",
    )

    # LLM configuration
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1  # Low temperature for consistent structured output
    llm_max_tokens: int = 4096

    # Workflow limits
    max_revision_count: int = 3
    max_step_budget: int = 50

    # Prompt injection detection
    injection_threshold: float = 0.7

    # Scoring
    passing_score: float = 70.0
    borderline_lower: float = 50.0
    borderline_upper: float = 75.0

    # Logging
    log_level: str = "INFO"
