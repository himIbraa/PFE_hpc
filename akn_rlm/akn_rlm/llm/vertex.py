"""Vertex AI / Gemini 2.5 Flash client (hard-case fallback)."""
from __future__ import annotations

import logging
import os

from akn_rlm.config import FALLBACK_MODEL, VERTEX_LOCATION, VERTEX_PROJECT
from akn_rlm.llm.client import LLMClient

log = logging.getLogger(__name__)


class VertexClient(LLMClient):
    """Calls Gemini 2.5 Flash via google-generativeai SDK.

    Authentication: GOOGLE_APPLICATION_CREDENTIALS env var (auto-set
    by config.py if the SA key file exists at the project root).
    """

    def __init__(self, model: str = FALLBACK_MODEL) -> None:
        self._model_name = model
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure()  # picks up GOOGLE_APPLICATION_CREDENTIALS / ADC
            self._client = genai.GenerativeModel(self._model_name)
        except ImportError:
            # Fall back to vertexai SDK
            try:
                import vertexai  # type: ignore
                from vertexai.generative_models import GenerativeModel  # type: ignore
                vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
                self._client = GenerativeModel(self._model_name)
            except ImportError as exc:
                raise ImportError(
                    "Install google-generativeai or vertexai: "
                    "pip install google-generativeai"
                ) from exc
        return self._client

    def chat(
        self,
        messages: list[dict],
        model: str = FALLBACK_MODEL,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        # Convert OpenAI-style messages to Gemini contents
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
            else:
                parts.append(content)
        prompt = "\n".join(parts)
        try:
            resp = client.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
            )
            return resp.text
        except Exception as exc:
            log.error("Vertex call failed (model=%s): %s", model, exc)
            raise
