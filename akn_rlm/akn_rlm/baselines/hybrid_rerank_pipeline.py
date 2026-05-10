"""Hybrid (BM25 + Dense, RRF-fused) + cross-encoder rerank baseline.

Builds the same fused candidate pool as
:class:`akn_rlm.baselines.hybrid_pipeline.HybridBaselinePipeline` (RRF over
BM25 and Dense, each retrieved at depth ``k_each`` and fused on
``(doc_id, canonical article_ref)``), takes the top-``rerank_pool_size``
fused entries, hands them to an injectable cross-encoder reranker, and
emits the same deterministic Arabic template answer as B1/B2/B3 from the
reranked top-``top_k``.

The reranker is injected so unit tests can mock it without loading the
sentence-transformers model. The default points at
:func:`akn_rlm.reranker.rerank` which already wraps the cross-encoder
configured by :data:`akn_rlm.config.RERANKER_MODEL`.

The answer is built from corpus text only (no LLM call), so HCR / JIR
are trivially zero — the comparison versus RLM is on retrieval and
citation metrics. Output shape matches
``akn_rlm.rlm.pipeline.build_pipeline().run(query)`` so
:func:`akn_rlm.eval.runner._answer_to_result` can consume it unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.indexers.dense import DenseHit, DenseIndex
from akn_rlm.normalizers import canonical_article_ref
from akn_rlm.retrievers.hybrid_fusion import rrf_fuse

log = logging.getLogger(__name__)

DEFAULT_TOP_K: int = 5
DEFAULT_K_EACH: int = 20
DEFAULT_RERANK_POOL_SIZE: int = 50
SUPPORT_SPAN_LEN: int = 280

RerankerFn = Callable[[str, list[dict], int], list[dict]]


def _default_reranker(query: str, candidates: list[dict], k: int) -> list[dict]:
    """Lazy import so tests can mock without loading sentence-transformers."""
    from akn_rlm import reranker
    return reranker.rerank(query, candidates, k=k)


class HybridRerankBaselinePipeline:
    """RRF(BM25, Dense) → cross-encoder rerank → top-K + Arabic template answer."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        reranker: RerankerFn | None = None,
        top_k: int = DEFAULT_TOP_K,
        k_each: int = DEFAULT_K_EACH,
        rerank_pool_size: int = DEFAULT_RERANK_POOL_SIZE,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._registry = registry
        self._reranker: RerankerFn = reranker or _default_reranker
        self._top_k = top_k
        self._k_each = k_each
        self._rerank_pool_size = rerank_pool_size

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

        pool = fused[: self._rerank_pool_size]
        reranked = self._reranker(query, pool, self._top_k) or []
        if not reranked:
            return self._abstain("no_hits")

        citations = self._reranked_to_citations(reranked[: self._top_k])
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
                "baseline":    "hybrid_rerank",
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

    def _reranked_to_citations(
        self, reranked: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Take reranker output → dedup by (doc_id, canonical ref) → citations.

        Each citation's ``confidence`` is the cross-encoder ``rerank_score``
        when present, otherwise it falls back to the RRF ``score`` (so the
        contract still holds when a degraded reranker passes candidates
        through unchanged).
        """
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for h in reranked:
            art_ref_canon = canonical_article_ref(h.get("article_ref", "")) or h.get("article_ref", "")
            doc_id = h.get("doc_id", "")
            key = (doc_id, art_ref_canon)
            if key in seen:
                continue
            seen.add(key)
            text = h.get("text", "") or ""
            confidence = h.get("rerank_score", h.get("score", 0.0))
            citations.append({
                "doc_id":          doc_id,
                "article_ref":     art_ref_canon,
                "doc_title":       self._doc_title(doc_id),
                "supporting_span": text[:SUPPORT_SPAN_LEN],
                "text":            text,
                "confidence":      float(confidence),
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
                "baseline":    "hybrid_rerank",
            },
        }


def build_hybrid_rerank_pipeline(
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    reranker: RerankerFn | None = None,
    top_k: int = DEFAULT_TOP_K,
    k_each: int = DEFAULT_K_EACH,
    rerank_pool_size: int = DEFAULT_RERANK_POOL_SIZE,
) -> HybridRerankBaselinePipeline:
    """Factory mirroring :func:`akn_rlm.rlm.pipeline.build_pipeline`."""
    return HybridRerankBaselinePipeline(
        bm25=bm25,
        dense=dense,
        registry=registry,
        reranker=reranker,
        top_k=top_k,
        k_each=k_each,
        rerank_pool_size=rerank_pool_size,
    )
