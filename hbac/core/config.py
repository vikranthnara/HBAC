from __future__ import annotations

import os
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from hbac.dotenv_loader import load_project_env
from hbac.freellmapi_config import bootstrap_freellmapi_env, freellmapi_configured

load_project_env()


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HBAC_LLM_", extra="ignore")

    provider: str = "auto"  # auto | openai | freellmapi | anthropic | vllm
    model: str = "auto"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    lora_path: str | None = None

    @model_validator(mode="after")
    def resolve_provider_and_credentials(self) -> Self:
        bootstrap_freellmapi_env()

        provider = self.provider.lower()
        if provider == "auto":
            env_provider = os.environ.get("HBAC_LLM_PROVIDER", "").strip().lower()
            if env_provider:
                provider = env_provider
            else:
                provider = "freellmapi" if freellmapi_configured() else "openai"
            object.__setattr__(self, "provider", provider)

        if provider == "freellmapi":
            base = os.environ.get("FREELLMAPI_BASE_URL", "").strip().rstrip("/")
            key = os.environ.get("FREELLMAPI_API_KEY", "").strip()
            if base and not self.base_url:
                object.__setattr__(self, "base_url", base)
            if key and not self.api_key:
                object.__setattr__(self, "api_key", key)
            if not self.model or self.model == "auto":
                object.__setattr__(
                    self,
                    "model",
                    os.environ.get("LLM_MODEL", "auto").strip() or "auto",
                )
            return self

        if self.api_key:
            return self

        if provider in {"openai", "vllm"}:
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                object.__setattr__(self, "api_key", key)
            if not self.model or self.model == "auto":
                object.__setattr__(self, "model", "gpt-4o-mini")
        elif provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if key:
                object.__setattr__(self, "api_key", key)
            if not self.model or self.model == "auto":
                object.__setattr__(self, "model", "claude-3-5-sonnet-latest")
        return self


class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HBAC_", extra="ignore")

    budget_tokens: int = 50_000
    max_steps: int = 100
    lambda_penalty: float = 0.001
    output_dir: str = "results"
    limit: int | None = None
    seed: int = 42


class SWEBenchConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HBAC_SWE_", extra="ignore")

    dataset_name: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    observation_max_chars: int = 10_000
    command_timeout: int = 60
    docker_cache_level: str = "env"


class LiveCodeBenchConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HBAC_LCB_", extra="ignore")

    release_version: str = "release_v5"
    max_repair_turns: int = 3
    eval_timeout: int = 6
    start_date: str | None = None
    end_date: str | None = None


class TABConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HBAC_TAB_", extra="ignore")

    mode: str = "heuristic"  # static | heuristic | learned
    budget_options: list[int] = Field(default=[256, 512, 1024, 2048, 4096])
    checkpoint_path: str | None = None


class ReFORCConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HBAC_REFORC_", extra="ignore")

    mode: str = "heuristic"  # heuristic | learned
    lambda_cost: float = 0.001
    stop_threshold: float = 0.05
    checkpoint_path: str | None = None
