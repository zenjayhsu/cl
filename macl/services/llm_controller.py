from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
import json
import os
from urllib.parse import urlparse

import requests

from services.runtime_logging import get_logger


logger = get_logger("llm")


class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.2) -> str:
        raise NotImplementedError

    def _generate_empty_value(self, schema_type: str) -> Any:
        if schema_type == "array":
            return []
        if schema_type == "string":
            return ""
        if schema_type == "object":
            return {}
        if schema_type in {"number", "integer"}:
            return 0
        if schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: Optional[dict]) -> dict:
        if not response_format or "json_schema" not in response_format:
            return {}
        schema = response_format["json_schema"]["schema"]
        result: Dict[str, Any] = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            result[prop_name] = self._generate_empty_value(prop_schema.get("type", "string"))
        return result


class DisabledLLMController(BaseLLMController):
    def get_completion(self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.2) -> str:
        raise RuntimeError("LLM backend is disabled. A working LLM backend is required.")


class OpenAICompatibleController(BaseLLMController):
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")

    def _provider_name(self) -> str:
        hostname = (urlparse(self.base_url).hostname or "").lower()
        if hostname.endswith("deepseek.com"):
            return "deepseek"
        if hostname.endswith("openrouter.ai"):
            return "openrouter"
        if hostname.endswith("openai.com"):
            return "openai"
        return "openai_compatible"

    def _adapt_response_format(self, response_format: Optional[dict]) -> Optional[dict]:
        if not response_format:
            return None
        if self._provider_name() == "deepseek" and response_format.get("type") == "json_schema":
            return {"type": "json_object"}
        return response_format

    def _build_messages(self, prompt: str, response_format: Optional[dict]) -> list[dict[str, str]]:
        system_content = "Return valid JSON only. Do not include markdown or explanation outside the JSON object."
        if self._provider_name() == "deepseek" and response_format:
            system_content += " The output must be valid json."
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    def _extract_error_payload(self, response: requests.Response | None) -> str:
        if response is None:
            return ""
        try:
            return json.dumps(response.json(), ensure_ascii=False)
        except Exception:
            return response.text.strip()

    def get_completion(self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.2) -> str:
        provider = self._provider_name()
        adapted_response_format = self._adapt_response_format(response_format)
        logger.info(
            "LLM request start | backend=openai_compatible | provider=%s | model=%s | temperature=%s | prompt_chars=%s",
            provider,
            self.model,
            temperature,
            len(prompt),
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt=prompt, response_format=adapted_response_format),
            "temperature": temperature,
        }
        if adapted_response_format:
            payload["response_format"] = adapted_response_format
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Empty completion returned by chat-completions backend.")
            logger.info(
                "LLM request end | backend=openai_compatible | provider=%s | model=%s | response_chars=%s",
                provider,
                self.model,
                len(content),
            )
            return content
        except requests.HTTPError as exc:
            error_payload = self._extract_error_payload(exc.response)
            logger.exception(
                "LLM request failed | backend=openai_compatible | provider=%s | model=%s | status=%s | error_body=%s",
                provider,
                self.model,
                exc.response.status_code if exc.response is not None else "unknown",
                error_payload[:2000],
            )
            raise RuntimeError(f"OpenAI-compatible completion failed: {exc}. response={error_payload[:500]}") from exc
        except Exception as exc:
            logger.exception(
                "LLM request failed | backend=openai_compatible | provider=%s | model=%s",
                provider,
                self.model,
            )
            raise RuntimeError(f"OpenAI-compatible completion failed: {exc}") from exc


class OllamaController(BaseLLMController):
    def __init__(self, model: str, base_url: Optional[str] = None) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

    def get_completion(self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.2) -> str:
        logger.info(
            "LLM request start | backend=ollama | model=%s | temperature=%s | prompt_chars=%s",
            self.model,
            temperature,
            len(prompt),
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": response_format["json_schema"]["schema"] if response_format and "json_schema" in response_format else "json",
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            response.raise_for_status()
            content = response.json().get("response", "")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Empty completion returned by Ollama backend.")
            logger.info("LLM request end | backend=ollama | model=%s | response_chars=%s", self.model, len(content))
            return content
        except Exception as exc:
            logger.exception("LLM request failed | backend=ollama | model=%s", self.model)
            raise RuntimeError(f"Ollama completion failed: {exc}") from exc


@dataclass
class LLMController:
    backend: Literal["disabled", "openai", "openrouter", "ollama"] = "disabled"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.backend == "openai":
            key = self.api_key or os.getenv("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required when CSCL_LLM_BACKEND=openai.")
            self._client = OpenAICompatibleController(
                model=self.model,
                api_key=key,
                base_url=self.base_url or "https://api.openai.com/v1",
            )
        elif self.backend == "openrouter":
            key = self.api_key or os.getenv("OPENROUTER_API_KEY", "")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is required when CSCL_LLM_BACKEND=openrouter.")
            self._client = OpenAICompatibleController(
                model=self.model,
                api_key=key,
                base_url=self.base_url or "https://openrouter.ai/api/v1",
            )
        elif self.backend == "ollama":
            self._client = OllamaController(model=self.model, base_url=self.base_url)
        elif self.backend == "disabled":
            self._client = DisabledLLMController()
        else:
            raise RuntimeError(f"Unsupported LLM backend: {self.backend}")

    def get_completion(self, prompt: str, response_format: Optional[dict] = None, temperature: float = 0.2) -> str:
        return self._client.get_completion(prompt=prompt, response_format=response_format, temperature=temperature)
