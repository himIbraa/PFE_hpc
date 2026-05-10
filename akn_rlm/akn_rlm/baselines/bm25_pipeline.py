"""BM25-only baseline pipeline.

Retrieves the top-K articles via BM25 and returns a deterministic Arabic
template answer. The answer is built from corpus text only (no LLM call),
so HCR / JIR are trivially zero — the comparison versus RLM is on retrieval
and citation metrics.

Output shape matches ``akn_rlm.rlm.pipeline.build_pipeline().run(query)``
so :func:`akn_rlm.eval.runner._answer_to_result` can consume it unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, List

from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.normalizers import canonical_article_ref

log = logging.getLogger(__name__)

DEFAULT_TOP_K: int = 5
SUPPORT_SPAN_LEN: int = 280


class BM25BaselinePipeline:
    """Top-K BM25 retrieval + deterministic Arabic template answer."""

    def __init__(
        self,
        bm25: BM25Index,
        registry: ArticleRegistry,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._bm25 = bm25
        self._registry = registry
        self._top_k = top_k

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain("empty_query")

        hits: List[BM25Hit] = self._bm25.search(query, k=self._top_k)
        if not hits:
            return self._abstain("no_hits")

        citations = self._hits_to_citations(hits)
        if not citations:
            return self._abstain("no_hits")

        return {
            "answer_text":       self._template_answer(citations),
            "abstention":        False,
            "abstention_reason": None,
            "citations":         citations,
            "reasoning_chain":   [],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count": 0,
                "gate_results": {},
                "baseline":    "bm25",
            },
        }

    # ------------------------------------------------------------------
    def _hits_to_citations(self, hits: List[BM25Hit]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for h in hits:
            art_ref_canon = canonical_article_ref(h.article_ref) or h.article_ref
            key = (h.doc_id, art_ref_canon)
            if key in seen:
                continue
            seen.add(key)
            text = h.text or ""
            citations.append({
                "doc_id":          h.doc_id,
                "article_ref":     art_ref_canon,
                "doc_title":       self._doc_title(h.doc_id),
                "supporting_span": text[:SUPPORT_SPAN_LEN],
                "text":            text,
                "confidence":      float(h.score),
            })
        return citations

    def _doc_title(self, doc_id: str) -> str:
        entry = self._registry.get_doc(doc_id)
        return entry.doc_title if entry and entry.doc_title else doc_id

    @staticmethod
    def _template_answer(citations: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for c in citations:
            doc_title = c.get("doc_title") or c.get("doc_id", "")
            ref = c.get("article_ref", "")
            text = c.get("supporting_span") or c.get("text", "")
            parts.append(f"وفقًا لـ {doc_title}، المادة {ref}: {text}")
        return "\n\n".join(parts)

    @staticmethod
    def _abstain(reason: str) -> dict[str, Any]:
        return {
            "answer_text":       "",
            "abstention":        True,
            "abstention_reason": reason,
            "citations":         [],
            "reasoning_chain":   [],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count": 0,
                "gate_results": {},
                "baseline":    "bm25",
            },
        }


def build_bm25_pipeline(
    bm25: BM25Index,
    registry: ArticleRegistry,
    top_k: int = DEFAULT_TOP_K,
) -> BM25BaselinePipeline:
    """Factory mirroring :func:`akn_rlm.rlm.pipeline.build_pipeline`."""
    return BM25BaselinePipeline(bm25=bm25, registry=registry, top_k=top_k)
