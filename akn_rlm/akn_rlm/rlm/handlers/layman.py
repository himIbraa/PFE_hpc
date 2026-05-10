"""Layman / Darja query handler — Phase 2 / R6.3.

Pipeline (per HANDOFF §3):

  1. **Mandatory** Darja → Modern Standard Arabic rewrite via a small
     LLM (default: Gemma routed via the LLM pool's ``"gemma"`` slot).
     The rewriter normalises colloquial vocabulary
     (``طلقت / نرجعها / الجدارمية / الكوميساريا / وقتاش نخلص``) into
     legal Arabic that the BM25 + Dense indices can match. The
     rewriter is the discriminator that distinguishes ``layman`` from
     other types — without it, the doc-router and BM25 tokenizer
     under-recall on Darja queries (see HANDOFF §4.995 R1 ablation —
     the doc-router was 67% on layman vs ≥83% on other types).

  2. Rewriter result is sanity-checked:
        - non-empty after strip
        - not literally identical to the original query (case folded)
     If either check fails, fall back to the **original** query so the
     handler can never *worsen* recall.

  3. Run the rewritten query through the same pipeline as the
     :class:`RuleApplicationHandler` — doc-route → RRF(BM25, Dense)
     restricted to routed → top-K=8 → mandatory verifier → answer with
     surviving cited.

  4. The summariser (final step inside ``RuleApplicationHandler``)
     receives the **rewritten** query (so the answer phrasing tracks
     the legal terminology, not the colloquial one).

The handler is implemented as a thin shell on top of
:class:`RuleApplicationHandler`. Telemetry adds ``rewrite_input`` /
``rewrite_output`` so an evaluator can verify the rewrite happened and
inspect what was rewritten.

Sub-LM call budget: 1 (rewriter) + ``RuleApplicationHandler``'s budget
(top-K verifier + 1 summariser ≤ 9) = ≤ 10 calls per query, inside the
project ``max_sub_calls=12`` envelope.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from akn_rlm.config import SUB_LLM_MODEL
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Index
from akn_rlm.indexers.dense import DenseIndex
from akn_rlm.rlm.handlers.rule_application import (
    DEFAULT_VERIFY_THRESHOLD as _RA_DEFAULT_VERIFY_THRESHOLD,
    RuleApplicationHandler,
    build_rule_application_handler,
)
from akn_rlm.rlm.routing import DocRouter, build_doc_router

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Gemma is routed via the LLM pool's "gemma" / "google" slot — see
# `akn_rlm.llm.client.LLMPool.default`. The exact model name is the
# AI Grid catalogue identifier; if the slot isn't configured the pool
# falls back to its first registered client (typically gpt-oss-120b).
DEFAULT_REWRITE_MODEL: str = "google/gemma-4-31B"
DEFAULT_REWRITE_MAX_TOKENS: int = 256

#: Mirrors the rule_application handler's default verify threshold so the
#: layman path (which delegates to ``RuleApplicationHandler``) is locked at
#: the same value. Updated in R9.1 from 0.5 → 0.3 to recover Cite F1 by
#: keeping more low-confidence-but-correct verifier verdicts.
DEFAULT_VERIFY_THRESHOLD: float = _RA_DEFAULT_VERIFY_THRESHOLD

TELEMETRY_BASELINE: str = "rlm_layman"


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """\
أنت مترجم لغوي. حوّل السؤال التالي من الدارجة الجزائرية إلى العربية الفصحى \
القانونية، مع الحفاظ على المعنى الأصلي ومصطلحات الحياة اليومية المحلية فقط \
عند الحاجة لفهم القصد القانوني.

قواعد:
- استخدم مصطلحات قانونية عربية رسمية (مثلاً: "زوجتي" بدلاً من "مرتي"، \
"الدرك الوطني" بدلاً من "الجدارمية"، "أدفع" بدلاً من "نخلص").
- لا تضف معلومات غير موجودة في السؤال الأصلي.
- لا تجب على السؤال — فقط أعد صياغته.
- أعد فقط النص المُعاد صياغته بالعربية الفصحى، بدون أي نص إضافي.

السؤال (دارجة): {query}

السؤال (فصحى):"""


def call_darja_rewriter(
    llm_pool,
    query: str,
    model: str = DEFAULT_REWRITE_MODEL,
    *,
    max_tokens: int = DEFAULT_REWRITE_MAX_TOKENS,
) -> str:
    """Default Darja → MSA rewriter using the LLM pool.

    Returns the rewritten query (stripped). On any failure returns ``""``
    (caller falls back to the original query so the handler never
    worsens recall). Designed to be injected with ``rewriter_fn=`` in
    unit tests so the suite never hits a real LLM.
    """
    if not query or not query.strip():
        return ""
    prompt = _REWRITE_PROMPT.format(query=query.strip())
    try:
        raw = llm_pool.call(prompt, model=model, max_tokens=max_tokens, temperature=0.0)
    except Exception as exc:
        log.warning("layman Darja rewriter failed (%s) — falling back to original", exc)
        return ""

    out = (raw or "").strip()
    if not out:
        return ""
    # Some LLMs prepend a label like "السؤال (فصحى):" — strip the prefix.
    for prefix in (
        "السؤال (فصحى):",
        "السؤال:",
        "الفصحى:",
        "MSA:",
    ):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
            break
    # Drop wrapping quotes the LLM sometimes adds.
    if len(out) >= 2 and out[0] in '"«' and out[-1] in '"»':
        out = out[1:-1].strip()
    return out


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

RewriterFn = Callable[[Any, str, str], str]


class LaymanHandler:
    """Typed layman handler: Darja → MSA rewrite → rule_application path."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        llm_pool,
        *,
        router: Optional[DocRouter] = None,
        sub_model: str = SUB_LLM_MODEL,
        rewrite_model: str = DEFAULT_REWRITE_MODEL,
        rewriter_fn: Optional[RewriterFn] = None,
        rule_handler: Optional[RuleApplicationHandler] = None,
        **rule_handler_kwargs: Any,
    ) -> None:
        self._llm_pool = llm_pool
        self._rewrite_model = rewrite_model
        self._rewriter_fn = rewriter_fn or call_darja_rewriter

        # Build a child rule_application handler if not injected. Reuse
        # the same router so we don't pay the startup cost twice.
        router = router or build_doc_router(registry=registry, bm25=bm25)
        self._rule_handler = rule_handler or build_rule_application_handler(
            bm25=bm25,
            dense=dense,
            registry=registry,
            llm_pool=llm_pool,
            router=router,
            sub_model=sub_model,
            **rule_handler_kwargs,
        )

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain(
                "empty_query", original=query or "", rewritten="",
                rewrite_used=False, sub_calls=0,
            )

        original = query.strip()

        # 1. Darja → MSA rewrite (mandatory, but guarded with fallback).
        sub_calls = 0
        try:
            rewritten_raw = self._rewriter_fn(
                self._llm_pool, original, self._rewrite_model
            )
            sub_calls += 1
        except Exception as exc:
            log.warning("layman rewriter exception (%s) — falling back to original", exc)
            rewritten_raw = ""

        rewritten = (rewritten_raw or "").strip()
        # Sanity-check the rewrite. Empty / identical / collapsed-to-whitespace
        # → fall back to the original query so we never worsen recall.
        if not rewritten or rewritten.lower() == original.lower():
            rewritten = original
            rewrite_used = False
        else:
            rewrite_used = True

        # 2. Delegate to the rule_application handler.
        try:
            inner = self._rule_handler.run(rewritten)
        except Exception as exc:
            log.error("layman inner rule_application failed: %s", exc)
            return self._abstain(
                "inner_pipeline_error",
                original=original, rewritten=rewritten,
                rewrite_used=rewrite_used, sub_calls=sub_calls,
            )

        # 3. Tag telemetry. Replace the inner ``rlm_rule_application``
        # baseline with ``rlm_layman`` so compare_baselines.py picks up
        # the right column. Carry inner sub_call_count forward.
        inner_telemetry = inner.get("_telemetry", {})
        inner_sub_calls = int(inner_telemetry.get("sub_call_count", 0))

        merged_telemetry = dict(inner_telemetry)
        merged_telemetry.update({
            "baseline":         TELEMETRY_BASELINE,
            "sub_call_count":   sub_calls + inner_sub_calls,
            "rewrite_input":    original,
            "rewrite_output":   rewritten,
            "rewrite_used":     rewrite_used,
            "inner_baseline":   inner_telemetry.get("baseline", ""),
        })

        out = dict(inner)
        out["_telemetry"] = merged_telemetry
        # Prepend a reasoning_chain note so the answer-explainer sees the
        # rewrite happened.
        chain = list(out.get("reasoning_chain") or [])
        chain.insert(0, f"darja_rewrite_used={rewrite_used}")
        if rewrite_used:
            chain.insert(1, f"rewritten_query={rewritten}")
        out["reasoning_chain"] = chain
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _abstain(
        reason: str,
        *,
        original: str,
        rewritten: str,
        rewrite_used: bool,
        sub_calls: int,
    ) -> dict[str, Any]:
        return {
            "answer_text":       "",
            "abstention":        True,
            "abstention_reason": reason,
            "citations":         [],
            "reasoning_chain":   [
                f"darja_rewrite_used={rewrite_used}",
                f"rewritten_query={rewritten}",
                f"reason={reason}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count":    0,
                "gate_results":   {},
                "baseline":       TELEMETRY_BASELINE,
                "sub_call_count": sub_calls,
                "rewrite_input":  original,
                "rewrite_output": rewritten,
                "rewrite_used":   rewrite_used,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_layman_handler(
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    llm_pool,
    *,
    router: Optional[DocRouter] = None,
    **kwargs: Any,
) -> LaymanHandler:
    """Factory mirroring the baseline ``build_*_pipeline`` helpers."""
    return LaymanHandler(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        **kwargs,
    )
