"""AI Grid LLM client (OpenAI-compatible endpoint)."""
from __future__ import annotations

import logging
from typing import Optional

from akn_rlm.config import AI_GRID_API_KEY, AI_GRID_BASE_URL, ROOT_LLM_MODEL
from akn_rlm.llm.client import LLMClient

log = logging.getLogger(__name__)


class AIGridClient(LLMClient):
    """Wraps the AI Grid OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str = AI_GRID_API_KEY,
        base_url: str = AI_GRID_BASE_URL,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: list[dict],
        model: str = ROOT_LLM_MODEL,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            # Some thinking models (DeepSeek, Qwen-thinking) put visible output
            # in 'reasoning_content' and leave 'content' empty
            if not content:
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    log.debug("model=%s: content empty, using reasoning_content (%d chars)",
                              model, len(reasoning))
                    content = reasoning
            if not content:
                log.warning("model=%s: empty response (round may be skipped)", model)
            return content
        except Exception as exc:
            log.error("AI Grid call failed (model=%s): %s", model, exc)
            raise
