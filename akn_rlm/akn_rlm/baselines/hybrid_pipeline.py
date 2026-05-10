"""Hybrid (BM25 + Dense, RRF-fused) baseline pipeline.

Runs each retriever to a deeper ``k_each`` (default 20), fuses the two
ranked lists via Reciprocal Rank Fusion (``RRF_K=60`` from
:mod:`akn_rlm.config`) keyed on ``(doc_id, canonical article_ref)``, and
returns the top-``top_k`` (default 5) as a deterministic Arabic template
answer. The fusion key choice matches
:func:`akn_rlm.rlm.legal_env.LegalEnv.search_hybrid`, so the upcoming
rerank baseline (B4) can re-use the same candidate pool.

The answer is built from corpus text only (no LLM call), so HCR / JIR are
trivially zero — the comparison versus RLM is on retrieval and citation
metrics.

Output shape matches ``akn_rlm.rlm.pipeline.build_pipeline().run(query)``
so :func:`akn_rlm.eval.runner._answer_to_result` can consume it unchanged.
"""
from __future__ import annotations

import logging
from typing import Any

from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.indexers.dense import DenseHit, DenseIndex
from akn_rlm.normalizers import canonical_article_ref
from akn_rlm.retrievers.hybrid_fusion import rrf_fuse

log = logging.getLogger(__name__)

DEFAULT_TOP_K: int = 5
DEFAULT_K_EACH: int = 20
SUPPORT_SPAN_LEN: int = 280


class HybridBaselinePipeline:
    """RRF(BM25, Dense) top-K retrieval + deterministic Arabic template answer."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        top_k: int = DEFAULT_TOP_K,
        k_each: int = DEFAULT_K_EACH,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._registry = registry
        self._top_k = top_k
        self._k_each = k_each

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain("empty_query")

        bm25_hits: list[BM25Hit] = self._bm25.search(query, k=self._k_each)
        dense_hits: list[DenseHit] = self._dense.search(query, k=self._k_each)

        bm25_dicts = self._hits_to_dicts(bm25_hits, retriever="bm25")
        dense_dicts = self._hits_to_dicts(dense_hits, retriever="dense")

        if not bm25_dicts and not dense_dicts:
            return self._abstain("no_hits")

        fused = rrf_fuse([bm25_dicts, dense_dicts])
        if not fused:
            return self._abstain("no_hits")

        citations = self._fused_to_citations(fused[: self._top_k])
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
                "baseline":    "hybrid",
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _hits_to_dicts(
        hits: list[BM25Hit] | list[DenseHit],
        *,
        retriever: str,
    ) -> list[dict[str, Any]]:
        """Convert retriever Hit dataclasses to RRF-friendly dicts.

        Canonicalises ``article_ref`` so the fusion key
        ``(doc_id, article_ref)`` collapses surface variants (``9 مكرر``,
        ``الأولى``, …) coming from either retriever.
        """
        out: list[dict[str, Any]] = []
        for h in hits:
            art_ref_canon = canonical_article_ref(h.article_ref) or h.article_ref
            out.append({
                "chunk_id":    h.chunk_id,
                "doc_id":      h.doc_id,
                "article_ref": art_ref_canon,
                "text":        h.text or "",
                "score":       float(h.score),
                "retriever":   retriever,
            })
        return out

    def _fused_to_citations(
        self, fused: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Take top-K fused hits → dedup by (doc_id, canonical ref) → citations.

        ``rrf_fuse`` already collapses by ``(doc_id, article_ref)``, but we
        re-dedup defensively in case the same key appears twice (shouldn't
        with the current implementation, but keeps the contract identical
        to B1/B2).
        """
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for h in fused:
            art_ref_canon = canonical_article_ref(h.get("article_ref", "")) or h.get("article_ref", "")
            doc_id = h.get("doc_id", "")
            key = (doc_id, art_ref_canon)
            if key in seen:
                continue
            seen.add(key)
            text = h.get("text", "") or ""
            citations.append({
                "doc_id":          doc_id,
                "article_ref":     art_ref_canon,
                "doc_title":       self._doc_title(doc_id),
                "supporting_span": text[:SUPPORT_SPAN_LEN],
                "text":            text,
                "confidence":      float(h.get("score", 0.0)),
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
                "baseline":    "hybrid",
            },
        }


def build_hybrid_pipeline(
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    top_k: int = DEFAULT_TOP_K,
    k_each: int = DEFAULT_K_EACH,
) -> HybridBaselinePipeline:
    """Factory mirroring :func:`akn_rlm.rlm.pipeline.build_pipeline`."""
    return HybridBaselinePipeline(
        bm25=bm25,
        dense=dense,
        registry=registry,
        top_k=top_k,
        k_each=k_each,
    )
