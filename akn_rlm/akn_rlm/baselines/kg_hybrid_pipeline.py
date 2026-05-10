"""KG-augmented hybrid baseline (B6).

The pipeline runs the same SPARQL token-coverage retrieval as B5 over the
Algerian legal KG (``data/kg/algerian_legal_kg.ttl``), uses the matched
text spans as a *query expansion* signal (most distinctive content terms
are appended to the query), then runs the same RRF(BM25, Dense) hybrid
as B3 on the rewritten query. As a final step, any fused candidate whose
``(doc_id, article_ref)`` was already surfaced by the KG receives a small
RRF score bias so KG-confirmed articles float to the top of the
candidate pool. The top-``top_k`` (default 5) feeds the same
deterministic Arabic template answer used by B1-B5. No LLM call → HCR /
JIR are zero by construction.

Output shape matches ``akn_rlm.rlm.pipeline.build_pipeline().run(query)``
so :func:`akn_rlm.eval.runner._answer_to_result` consumes it unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from akn_rlm.baselines.kg_pipeline import (
    KGBaselinePipeline,
    MAX_TOKENS_PER_QUERY,
    MIN_TOKEN_LEN,
    SPARQL_LIMIT,
    _ART_SUFFIX_RE,
    _SPARQL_CONCEPT_SEARCH,
    _STOPWORDS,
    _TOKEN_SPLIT_RE,
    _URI_RE,
)
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.indexers.dense import DenseHit, DenseIndex
from akn_rlm.normalizers import canonical_article_ref, normalize_arabic
from akn_rlm.retrievers.hybrid_fusion import rrf_fuse

log = logging.getLogger(__name__)

DEFAULT_TOP_K: int = 5
DEFAULT_K_EACH: int = 20
DEFAULT_EXPANSION_TERMS: int = 5
# Small enough that retrieval ranking still dominates, large enough to
# float a KG-confirmed article a few RRF positions. RRF scores at rank 1
# from a single retriever are ~1/61 ≈ 0.0164, so 0.01 is roughly two
# rank-positions of bias.
DEFAULT_KG_BOOST: float = 0.01
SUPPORT_SPAN_LEN: int = 280


class KGHybridBaselinePipeline:
    """KG entity lookup → query expansion → RRF(BM25, Dense) → KG-biased top-K."""

    def __init__(
        self,
        kg: Any,                                  # rdflib.Graph (passed to sparql_fn)
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        top_k: int = DEFAULT_TOP_K,
        k_each: int = DEFAULT_K_EACH,
        expansion_terms_max: int = DEFAULT_EXPANSION_TERMS,
        kg_boost: float = DEFAULT_KG_BOOST,
        sparql_fn: Optional[Callable[[Any, str], list[dict[str, Any]]]] = None,
    ) -> None:
        self._kg = kg
        self._bm25 = bm25
        self._dense = dense
        self._registry = registry
        self._top_k = top_k
        self._k_each = k_each
        self._expansion_terms_max = expansion_terms_max
        self._kg_boost = kg_boost
        if sparql_fn is None:
            from akn_rlm.retrievers import graphrag as _gr
            sparql_fn = _gr.sparql_query
        self._sparql_fn = sparql_fn

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain("empty_query")

        # 1. Tokenize the original query and run the KG SPARQL per token.
        query_tokens = KGBaselinePipeline._tokenize(query)
        kg_uris: set[tuple[str, str]] = set()
        kg_spans: list[str] = []
        for tok in query_tokens:
            for row in self._sparql_for_token(tok):
                uri = (row.get("article") or "").strip()
                if uri:
                    doc_id, art_ref = self._uri_to_doc_ref(uri)
                    if doc_id and art_ref:
                        canon_ref = canonical_article_ref(art_ref) or art_ref
                        kg_uris.add((doc_id, canon_ref))
                span = (row.get("text") or "").strip()
                if span:
                    kg_spans.append(span)

        # 2. Build the rewritten query by appending up to N expansion terms.
        expansion = self._expansion_terms(query_tokens, kg_spans)
        rewritten = query if not expansion else f"{query} {' '.join(expansion)}"

        # 3. Hybrid retrieval on the rewritten query.
        bm25_hits: list[BM25Hit] = self._bm25.search(rewritten, k=self._k_each)
        dense_hits: list[DenseHit] = self._dense.search(rewritten, k=self._k_each)

        bm25_dicts = self._hits_to_dicts(bm25_hits, retriever="bm25")
        dense_dicts = self._hits_to_dicts(dense_hits, retriever="dense")

        if not bm25_dicts and not dense_dicts:
            return self._abstain("no_hits")

        fused = rrf_fuse([bm25_dicts, dense_dicts])
        if not fused:
            return self._abstain("no_hits")

        # 4. Bias the fused list with the KG-derived URIs.
        if kg_uris and self._kg_boost:
            for entry in fused:
                key = (entry.get("doc_id", ""), entry.get("article_ref", ""))
                if key in kg_uris:
                    boosted = float(entry.get("score", 0.0)) + self._kg_boost
                    entry["score"] = boosted
                    entry["rrf_score"] = boosted
                    entry["kg_boosted"] = True
            fused.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)

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
                "baseline":    "kg_hybrid",
            },
        }

    # ------------------------------------------------------------------
    # SPARQL plumbing — mirrors KGBaselinePipeline._sparql_for_token but
    # kept inline so the rerank/hybrid baselines don't need to instantiate
    # a KGBaselinePipeline just to share two helpers.
    # ------------------------------------------------------------------
    def _sparql_for_token(self, token: str) -> list[dict[str, Any]]:
        safe = token.replace('"', "'").replace("\\", "")
        q = _SPARQL_CONCEPT_SEARCH.format(token=safe, limit=SPARQL_LIMIT)
        try:
            return self._sparql_fn(self._kg, q) or []
        except Exception as exc:                                    # noqa: BLE001
            log.warning("KG SPARQL failed for token %r: %s", token, exc)
            return []

    def _uri_to_doc_ref(
        self, uri: str
    ) -> tuple[Optional[str], Optional[str]]:
        m = _URI_RE.match(uri)
        if not m:
            return (None, None)
        date     = m.group("date")
        num      = m.group("num")
        frag     = m.group("frag")
        art_eid  = _ART_SUFFIX_RE.sub("", frag)
        if not art_eid.startswith("art_"):
            return (None, None)
        article_ref = art_eid[len("art_"):]
        canonical = (
            self._registry.resolve_alias(num)
            or self._registry.resolve_alias(f"{num}_{date}")
            or num
        )
        return (canonical, article_ref)

    # ------------------------------------------------------------------
    # Query expansion — pick the most frequent content tokens that appear
    # in the KG-matched spans but are NOT already in the query. Deterministic.
    # ------------------------------------------------------------------
    def _expansion_terms(
        self, query_tokens: list[str], spans: list[str],
    ) -> list[str]:
        if not spans or self._expansion_terms_max <= 0:
            return []
        already = set(query_tokens)
        counts: dict[str, int] = {}
        for span in spans:
            seen_in_span: set[str] = set()
            for tok in _content_tokens(span):
                if tok in already or tok in seen_in_span:
                    continue
                seen_in_span.add(tok)
                counts[tok] = counts.get(tok, 0) + 1
        if not counts:
            return []
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [tok for tok, _ in ranked[: self._expansion_terms_max]]

    # ------------------------------------------------------------------
    # Hybrid plumbing — duplicated from HybridBaselinePipeline so the
    # baselines stay self-contained (matches the B3/B4 pattern).
    # ------------------------------------------------------------------
    @staticmethod
    def _hits_to_dicts(
        hits: list[BM25Hit] | list[DenseHit],
        *,
        retriever: str,
    ) -> list[dict[str, Any]]:
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
                "kg_boosted":      bool(h.get("kg_boosted", False)),
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
                "baseline":    "kg_hybrid",
            },
        }


def _content_tokens(text: str) -> list[str]:
    """Tokenise corpus text the same way as ``KGBaselinePipeline._tokenize``
    but without the per-query token cap. Used only for expansion-term
    selection over KG-matched spans.
    """
    out: list[str] = []
    for raw in _TOKEN_SPLIT_RE.split(text):
        tok = raw.strip()
        if not tok or len(tok) < MIN_TOKEN_LEN:
            continue
        if tok.startswith("ال") and len(tok) > 4:
            tok_stripped = tok[2:]
        else:
            tok_stripped = tok
        if (
            tok_stripped in _STOPWORDS
            or normalize_arabic(tok_stripped) in _STOPWORDS
        ):
            continue
        out.append(tok_stripped)
    return out


def build_kg_hybrid_pipeline(
    kg: Any,
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    top_k: int = DEFAULT_TOP_K,
    k_each: int = DEFAULT_K_EACH,
    expansion_terms_max: int = DEFAULT_EXPANSION_TERMS,
    kg_boost: float = DEFAULT_KG_BOOST,
    sparql_fn: Optional[Callable[[Any, str], list[dict[str, Any]]]] = None,
) -> KGHybridBaselinePipeline:
    """Factory mirroring :func:`akn_rlm.rlm.pipeline.build_pipeline`."""
    return KGHybridBaselinePipeline(
        kg=kg,
        bm25=bm25,
        dense=dense,
        registry=registry,
        top_k=top_k,
        k_each=k_each,
        expansion_terms_max=expansion_terms_max,
        kg_boost=kg_boost,
        sparql_fn=sparql_fn,
    )


# Silence unused-import warnings for re-exported helpers (used in tests)
_ = MAX_TOKENS_PER_QUERY
