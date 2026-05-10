"""Rule-application query handler — Phase 2 / R6.1.

Pipeline (per HANDOFF §3):

  1. Doc-route the query via :class:`DocRouter` (alias + numeric-id +
     BM25 channels) to get 1-3 likely ``doc_id`` predictions. Empty
     prediction → fall back to corpus-wide retrieval.

  2. RRF-fuse(BM25, Dense) — same fusion as B3/B4 — restricted to
     the routed docs (with full-pool fallback when filtering wipes
     the candidate list).

  3. Take top-``top_k_candidates=8`` from the fused list.

  4. **Mandatory** sub-LM verifier (:func:`call_verifier`) on every
     top-K candidate. Survivors are those with ``relevant=True`` AND
     ``confidence >= verify_threshold``. Per HANDOFF §3 — this is
     the discriminator that distinguishes rule-application from
     simple lookup. No "answer from search alone" path.

  5. Answer with **all** surviving cited articles (no top-1
     truncation — rule-application questions typically combine 2-4
     articles).

  6. Synthesise the answer via :func:`call_summarizer` over the
     surviving citations. On null summary or summariser failure the
     handler falls back to the deterministic Arabic template used by
     B1-B6 and the other handlers.

Sub-LM call budget per query: ``top_k_candidates`` verifier calls + 1
summariser = ≤ 8 + 1 = 9 calls. Inside the project ``max_sub_calls=12``
envelope.

The handler is intentionally self-contained (no LangGraph, no
``RootController``) and is callable as a baseline-shaped pipeline so
the existing scripts can run it through the same evaluation harness as
B1-B6 + R2-R5.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from akn_rlm.config import SUB_LLM_MODEL
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.indexers.dense import DenseHit, DenseIndex
from akn_rlm.normalizers import canonical_article_ref
from akn_rlm.retrievers.hybrid_fusion import rrf_fuse
from akn_rlm.rlm.routing import DocRouter, build_doc_router
from akn_rlm.rlm.sub_worker import call_summarizer, call_verifier
from akn_rlm.rlm.supervisor import SupervisorFn, should_supervise

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# HANDOFF §3 says rule_application uses top-K=8.
DEFAULT_TOP_K_CANDIDATES: int = 8
# F5: reverted F4's 8→4 back to 8 — F4 evidence shows tightening
# was net-zero for RA itself (0.220→0.225) but caused a −0.045
# regression on layman (which delegates to RA). Layman's Darja-
# rewritten queries surface gold deeper in the ranked list and need
# the wider top-K window. RA precision lift now comes from R9.5
# supervisor re-ranking, not from raw truncation.
DEFAULT_FINAL_TOP_K: int = 8
DEFAULT_K_EACH: int = 30
DEFAULT_VERIFY_THRESHOLD: float = 0.5
DEFAULT_ROUTE_TOP_N: int = 3
SUPPORT_SPAN_LEN: int = 280

TELEMETRY_BASELINE: str = "rlm_rule_application"


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

VerifierFn = Callable[[Any, str, dict, str], dict]
SummarizerFn = Callable[[Any, str, list[dict], str], dict]


class RuleApplicationHandler:
    """Typed rule-application handler.

    Pipeline: route -> hybrid k_each -> top-K=8 -> mandatory verify ->
    surviving citations -> synth.
    """

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        llm_pool,
        *,
        router: Optional[DocRouter] = None,
        sub_model: str = SUB_LLM_MODEL,
        top_k_candidates: int = DEFAULT_TOP_K_CANDIDATES,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        k_each: int = DEFAULT_K_EACH,
        verify_threshold: float = DEFAULT_VERIFY_THRESHOLD,
        route_top_n: int = DEFAULT_ROUTE_TOP_N,
        verifier_fn: Optional[VerifierFn] = None,
        summarizer_fn: Optional[SummarizerFn] = None,
        supervisor_fn: Optional[SupervisorFn] = None,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._registry = registry
        self._llm_pool = llm_pool
        self._router = router or build_doc_router(registry=registry, bm25=bm25)
        self._sub_model = sub_model
        self._top_k_candidates = top_k_candidates
        self._final_top_k = final_top_k
        self._k_each = k_each
        self._verify_threshold = verify_threshold
        self._route_top_n = route_top_n
        self._verifier_fn = verifier_fn or call_verifier
        self._summarizer_fn = summarizer_fn or call_summarizer
        # R9.5: optional gpt-oss-120b per-citation re-ranker. Fires only
        # when ``should_supervise`` returns True (uncertainty band).
        self._supervisor_fn = supervisor_fn

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain(
                "empty_query", routed=[], top_score=0.0, candidates=0, sub_calls=0,
            )

        # 1. Doc-route
        route = self._router.route(query, top_n=self._route_top_n)
        routed_ids = list(route.doc_ids)

        # 2. RRF-fuse(BM25, Dense) restricted to routed docs (with fallback)
        candidates = self._fused_candidates(query, routed_ids)
        if not candidates:
            return self._abstain(
                "no_hits",
                routed=routed_ids,
                top_score=0.0,
                candidates=0,
                sub_calls=0,
            )

        # 3. Take top-K
        top_k_pool = candidates[: self._top_k_candidates]
        top_score = float(top_k_pool[0].get("score", 0.0))

        # 4. Mandatory sub-LM verifier on every top-K candidate
        sub_calls = 0
        accumulated: dict[tuple[str, str], dict[str, Any]] = {}
        for cand in top_k_pool:
            cand_article = self._candidate_to_article(cand)
            try:
                verdict = self._verifier_fn(
                    self._llm_pool, query, cand_article, self._sub_model
                )
                sub_calls += 1
            except Exception as exc:
                log.warning("rule_application verifier failed (%s) — skipping", exc)
                continue

            if not verdict.get("relevant"):
                continue
            conf = float(verdict.get("confidence", 0.0) or 0.0)
            if conf < self._verify_threshold:
                continue

            key = (cand_article["doc_id"], cand_article["article_ref"])
            supporting_quote = verdict.get("supporting_span") or ""
            citation = self._build_citation(
                cand_article, supporting_quote=supporting_quote, confidence=conf
            )
            # Keep highest-confidence verdict if duplicates appear.
            prior = accumulated.get(key)
            if prior is None or conf > float(prior.get("confidence", 0.0)):
                accumulated[key] = citation

        if not accumulated:
            return self._abstain(
                "no_verified_articles",
                routed=routed_ids,
                top_score=top_score,
                candidates=len(top_k_pool),
                sub_calls=sub_calls,
            )

        # 5. Rank by confidence, truncate to final_top_k
        ranked = sorted(
            accumulated.values(),
            key=lambda c: float(c.get("confidence", 0.0)),
            reverse=True,
        )
        final_citations = ranked[: self._final_top_k]

        # 5b. R9.5 supervisor (smart-trigger). Re-rank with the strong
        # model when the verifier's confidence is in the uncertainty
        # band — drops cross-doc adjacencies that the small verifier
        # let through, lifts Cite F1 and MRR art.
        supervisor_used = False
        if self._supervisor_fn is not None and should_supervise(final_citations):
            try:
                supervised = self._supervisor_fn(
                    self._llm_pool, query, final_citations,
                )
                sub_calls += 1
                if supervised:
                    final_citations = supervised
                supervisor_used = True
            except Exception as exc:
                log.warning(
                    "rule_application supervisor failed (%s) — "
                    "keeping pre-supervisor citations", exc,
                )

        if not final_citations:
            return self._abstain(
                "supervisor_dropped_all",
                routed=routed_ids,
                top_score=top_score,
                candidates=len(top_k_pool),
                sub_calls=sub_calls,
            )

        # 6. Synthesise — fall back to deterministic template on null/exc
        answer_text = self._template_answer(final_citations)
        try:
            synth = self._summarizer_fn(
                self._llm_pool, query, final_citations, self._sub_model
            )
            sub_calls += 1
            summary = synth.get("summary")
            if isinstance(summary, str) and summary.strip():
                answer_text = summary.strip()
        except Exception as exc:
            log.warning(
                "rule_application summariser failed (%s) — using template", exc
            )

        return {
            "answer_text":       answer_text,
            "abstention":        False,
            "abstention_reason": None,
            "citations":         final_citations,
            "reasoning_chain":   [
                f"routed_doc_ids={routed_ids}",
                f"verified_candidates={len(final_citations)}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 1,
            "_telemetry": {
                "retry_count":     0,
                "gate_results":    {},
                "baseline":        TELEMETRY_BASELINE,
                "routed_doc_ids":  routed_ids,
                "top_score":       top_score,
                "candidate_count": len(top_k_pool),
                "verified_count":  len(final_citations),
                "sub_call_count":  sub_calls,
                "supervisor_used": supervisor_used,
            },
        }

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _fused_candidates(
        self, query: str, routed_ids: list[str]
    ) -> list[dict[str, Any]]:
        """RRF(BM25, Dense) restricted to routed docs (with full-pool fallback)."""
        try:
            bm25_hits: list[BM25Hit] = self._bm25.search(query, k=self._k_each)
        except Exception as exc:
            log.warning("rule_application BM25 failed: %s", exc)
            bm25_hits = []
        try:
            dense_hits: list[DenseHit] = self._dense.search(query, k=self._k_each)
        except Exception as exc:
            log.warning("rule_application dense failed: %s", exc)
            dense_hits = []

        bm25_dicts = self._hits_to_dicts(bm25_hits, retriever="bm25")
        dense_dicts = self._hits_to_dicts(dense_hits, retriever="dense")
        if not bm25_dicts and not dense_dicts:
            return []

        fused = rrf_fuse([bm25_dicts, dense_dicts])
        if routed_ids:
            allowed = set(routed_ids)
            filtered = [h for h in fused if h.get("doc_id") in allowed]
            # If filtering wipes everything, fall back to the unrestricted
            # fused list — better to retrieve from the whole corpus than to
            # return nothing.
            if filtered:
                fused = filtered
        return fused

    @staticmethod
    def _hits_to_dicts(
        hits: list[BM25Hit] | list[DenseHit],
        *,
        retriever: str,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for h in hits:
            ref_canon = canonical_article_ref(h.article_ref) or h.article_ref
            out.append({
                "chunk_id":    h.chunk_id,
                "doc_id":      h.doc_id,
                "article_ref": ref_canon,
                "text":        h.text or "",
                "score":       float(h.score),
                "retriever":   retriever,
            })
        return out

    def _candidate_to_article(self, cand: dict[str, Any]) -> dict[str, Any]:
        doc_id = cand.get("doc_id", "")
        ref_canon = canonical_article_ref(cand.get("article_ref", "")) or cand.get(
            "article_ref", ""
        )
        return {
            "doc_id":      doc_id,
            "article_ref": ref_canon,
            "text":        cand.get("text", "") or "",
            "doc_title":   self._doc_title(doc_id),
            "score":       float(cand.get("score", 0.0)),
        }

    # ------------------------------------------------------------------
    # Citation / answer assembly
    # ------------------------------------------------------------------

    def _build_citation(
        self,
        article: dict[str, Any],
        *,
        supporting_quote: str,
        confidence: float,
    ) -> dict[str, Any]:
        text = article.get("text", "") or ""
        # Prefer the verifier's exact supporting quote when it really is a
        # substring of the article text (the span-existence gate later in the
        # pipeline demands that). Fall back to text[:280] otherwise.
        if supporting_quote and supporting_quote in text:
            span = supporting_quote[:SUPPORT_SPAN_LEN]
        else:
            span = text[:SUPPORT_SPAN_LEN]
        return {
            "doc_id":            article["doc_id"],
            "article_ref":       article["article_ref"],
            "doc_title":         article.get("doc_title", "") or article["doc_id"],
            "supporting_span":   span,
            "text":              text,
            "confidence":        float(confidence),
            "verifier_relevant": True,
        }

    def _doc_title(self, doc_id: str) -> str:
        try:
            entry = self._registry.get_doc(doc_id)
        except Exception:
            entry = None
        return getattr(entry, "doc_title", "") or doc_id

    @staticmethod
    def _template_answer(citations: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for c in citations:
            doc_title = c.get("doc_title") or c.get("doc_id", "")
            ref = c.get("article_ref", "")
            text = c.get("supporting_span") or c.get("text", "")
            parts.append(f"وفقًا لـ {doc_title}، المادة {ref}: {text}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    @staticmethod
    def _abstain(
        reason: str,
        *,
        routed: list[str],
        top_score: float,
        candidates: int,
        sub_calls: int,
    ) -> dict[str, Any]:
        return {
            "answer_text":       "",
            "abstention":        True,
            "abstention_reason": reason,
            "citations":         [],
            "reasoning_chain":   [
                f"routed_doc_ids={routed}",
                f"reason={reason}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count":     0,
                "gate_results":    {},
                "baseline":        TELEMETRY_BASELINE,
                "routed_doc_ids":  routed,
                "top_score":       top_score,
                "candidate_count": candidates,
                "verified_count":  0,
                "sub_call_count":  sub_calls,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rule_application_handler(
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    llm_pool,
    *,
    router: Optional[DocRouter] = None,
    **kwargs: Any,
) -> RuleApplicationHandler:
    """Factory mirroring the baseline ``build_*_pipeline`` helpers."""
    return RuleApplicationHandler(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        **kwargs,
    )
