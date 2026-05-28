from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _clean_env(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


@dataclass(frozen=True)
class LLMRuntimeConfig:
    backend: Literal["openai", "openrouter", "ollama"] = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class ContextWindowConfig:
    max_turns: int = 6


@dataclass(frozen=True)
class CSCLRuntimeConfig:
    llm: LLMRuntimeConfig = LLMRuntimeConfig()
    context_window: ContextWindowConfig = ContextWindowConfig()


def load_runtime_config_from_env() -> CSCLRuntimeConfig:
    llm_backend = _clean_env(os.getenv("CSCL_LLM_BACKEND")) or "openai"
    llm_model = _clean_env(os.getenv("CSCL_LLM_MODEL")) or "gpt-4o-mini"
    llm_base_url = _clean_env(os.getenv("CSCL_LLM_BASE_URL"))
    llm_api_key = _clean_env(os.getenv("CSCL_LLM_API_KEY"))
    context_max_turns = int(os.getenv("CSCL_CONTEXT_MAX_TURNS", "6"))

    return CSCLRuntimeConfig(
        llm=LLMRuntimeConfig(
            backend=llm_backend,
            model=llm_model,
            base_url=llm_base_url,
            api_key=llm_api_key,
        ),
        context_window=ContextWindowConfig(max_turns=context_max_turns),
    )
