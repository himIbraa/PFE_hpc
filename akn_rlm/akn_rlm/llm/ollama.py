"""Ollama local LLM client (offline fallback)."""
from __future__ import annotations

import json
import logging
import os
import urllib.request

log = logging.getLogger(__name__)

_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


class OllamaClient:
    def __init__(self, base_url: str = _BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: list[dict],
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]

    def complete(
        self,
        prompt: str,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 4096,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=model, max_tokens=max_tokens, temperature=temperature)
