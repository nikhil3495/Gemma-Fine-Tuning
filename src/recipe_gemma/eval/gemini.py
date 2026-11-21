"""Zero-shot Gemini baseline + Gemini-as-judge client (google-genai SDK).

The client is injectable so tests can run with a mock and no network or API key.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

TextFn = Callable[[str], str]


def api_key() -> str | None:
    """Read the Gemini key from the environment (never from files)."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


class GeminiClient:
    """Minimal wrapper: ``client(prompt) -> text`` with retries and fixed decoding."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        top_k: int = 40,
        max_output_tokens: int = 512,
        seed: int | None = None,
        sdk_client: Any = None,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.max_retries = max_retries
        if sdk_client is None:
            key = api_key()
            if not key:
                raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
            from google import genai

            sdk_client = genai.Client(api_key=key)
        self._client = sdk_client

    def _config(self, max_output_tokens: int | None) -> Any:
        try:
            from google.genai import types

            kwargs: dict[str, Any] = {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "max_output_tokens": max_output_tokens or self.max_output_tokens,
            }
            if self.seed is not None:
                kwargs["seed"] = self.seed
            try:  # keep the token budget for the answer (no hidden "thinking" tokens)
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                return types.GenerateContentConfig(**kwargs)
            except Exception:
                kwargs.pop("thinking_config", None)
                return types.GenerateContentConfig(**kwargs)
        except ImportError:  # mocked client in tests
            return {"temperature": self.temperature, "max_output_tokens": max_output_tokens}

    def __call__(self, prompt: str, max_output_tokens: int | None = None) -> str:
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config(max_output_tokens)
                )
                return (getattr(resp, "text", None) or "").strip()
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                log.warning("gemini call failed (%s); retry in %.0fs", exc, delay)
                time.sleep(delay)
                delay *= 2
        return ""
