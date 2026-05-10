"""Provider-agnostic LLM chat wrapper.

Each provider subclass implements chat() and is registered by name so
LegalEnv.llm_call() can route by model string.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

log = logging.getLogger(__name__)

# Registry: model-name-prefix -> provider class
_REGISTRY: dict[str, type["LLMClient"]] = {}


def register(prefix: str):
    def _dec(cls):
        _REGISTRY[prefix.lower()] = cls
        return cls
    return _dec


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant text reply."""

    def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Convenience: single-turn prompt -> text."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, model=model, max_tokens=max_tokens, temperature=temperature)


class LLMPool:
    """Routes calls to the appropriate provider based on model name prefix."""

    def __init__(self, clients: dict[str, LLMClient]) -> None:
        # e.g. {"ai_grid": AIGridClient(), "vertex": VertexClient()}
        self._clients = clients
        # R9.7: per-handler-run call counter. ``start_recording`` zeroes
        # this; every ``call`` increments the bucket for its ``model``;
        # ``stop_recording`` returns the dict. The default state is
        # ``None`` so handlers that don't opt in pay no overhead.
        self._call_counts: dict[str, int] | None = None

    def start_recording(self) -> None:
        """Begin recording per-model call counts for the next batch of
        ``.call(...)`` invocations. Idempotent — starting again resets
        the counter."""
        self._call_counts = {}

    def stop_recording(self) -> dict[str, int]:
        """Stop recording and return a copy of the accumulated counts."""
        counts = dict(self._call_counts or {})
        self._call_counts = None
        return counts

    def snapshot_calls(self) -> dict[str, int]:
        """Return a copy of the current call-count buffer without
        stopping the recording. Returns an empty dict if recording is
        not active."""
        return dict(self._call_counts or {})

    def call(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        context_articles: list[dict] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        client = self._route(model)
        ctx = ""
        if context_articles:
            ctx = "\n\n".join(
                f"[{a.get('doc_id', '')} art.{a.get('article_ref', '')}]\n{a.get('text', '')}"
                for a in context_articles
            )
            prompt = ctx + "\n\n---\n\n" + prompt
        # R9.7: bookkeeping. Only fires when ``start_recording`` is
        # active so unrelated callers (e.g. unit tests using a mock
        # pool) pay no overhead.
        if self._call_counts is not None:
            self._call_counts[model] = self._call_counts.get(model, 0) + 1
        return client.complete(
            prompt,
            model=model,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
        )

    def _route(self, model: str) -> LLMClient:
        m = model.lower()
        for prefix, client in self._clients.items():
            if m.startswith(prefix) or prefix in m:
                return client
        # fallback: first registered
        if self._clients:
            return next(iter(self._clients.values()))
        raise ValueError(f"No LLM client available for model '{model}'")

    @classmethod
    def default(cls) -> "LLMPool":
        """Build pool from environment config.

        Each model family uses its own scoped API key so AI Grid's per-key
        model restrictions don't cause 401 errors on sub-LM calls.
        """
        from akn_rlm.config import AI_GRID_API_KEY, AI_GRID_BASE_URL, QWEN_API_KEY, GEMMA_API_KEY
        from akn_rlm.llm.ai_grid import AIGridClient

        clients: dict[str, LLMClient] = {}
        try:
            # Root LLM — gpt-oss-120b key
            gpt_client = AIGridClient(api_key=AI_GRID_API_KEY, base_url=AI_GRID_BASE_URL)
            clients["gpt"] = gpt_client

            # Sub-LM / RAG verifier — Qwen3-30B-A3B-Thinking key
            qwen_client = AIGridClient(api_key=QWEN_API_KEY, base_url=AI_GRID_BASE_URL)
            clients["qwen"] = qwen_client
            clients["qwen3"] = qwen_client

            # Jurisdiction classifier — google/gemma key
            if GEMMA_API_KEY and GEMMA_API_KEY != AI_GRID_API_KEY:
                gemma_client = AIGridClient(api_key=GEMMA_API_KEY, base_url=AI_GRID_BASE_URL)
                clients["gemma"] = gemma_client
                clients["google"] = gemma_client

            # Default fallback for unknown prefixes: use root key
            clients["ai_grid"] = gpt_client

            log.info(
                "LLMPool ready: %d client slots (%s)",
                len(clients), list(clients.keys()),
            )
        except Exception as exc:
            log.warning("AI Grid client unavailable: %s", exc)
        return cls(clients)
