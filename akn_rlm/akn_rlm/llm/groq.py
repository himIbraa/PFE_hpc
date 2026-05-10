"""Groq LLM client (optional fast inference)."""
from __future__ import annotations

import logging
import os

from akn_rlm.llm.client import LLMClient

log = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class GroqClient(LLMClient):
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY", "")
        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:
            raise ImportError("groq package required: pip install groq") from exc
        self._client = Groq(api_key=key)

    def chat(
        self,
        messages: list[dict],
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
