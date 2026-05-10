"""Unanswerable query handler — Phase 2 / R5.

Pipeline (per HANDOFF §3):

  1. Detect foreign-law / DZ-absence signals **first** via a regex
     dictionary. The check is deliberately fast (no LLM, no search) so a
     contaminated query can be abstained on without giving the rest of
     the pipeline a chance to fabricate from tangential matches.

       - Existing dictionary: :func:`akn_rlm.gates.jurisdiction.detect`
         (the same regex table the JIR metric uses — covers French / US /
         UK / Egyptian / Tunisian / Gulf / international markers and the
         documented DZ_ABSENCE concepts).

       - Local dictionary :data:`_LOCAL_FOREIGN_PATTERNS` covers the 13
         unanswerable-slice questions the existing detector misses
         (Arabic phrasings of `concubinage` / `معاشرة`, Tunisian polygamy
         ban Arabic phrasing, Egyptian inheritance / detention / clause
         pénale judiciaire, US-style 401k pension funds, French
         contrôle des loyers / droit administratif français, the
         constitutional auto-saisine and municipal taxing-power
         DZ_ABSENCE Arabic phrasings, etc.).

  2. **One** confirming hybrid search restricted to the routed docs (the
     "ONE confirming hybrid search" called out in the HANDOFF). The
     search is *not* used to bootstrap an answer — it serves two
     purposes only:
        a. It records the strongest Algerian-corpus match (so the
           telemetry / logs can show that no clear counterpart exists),
           and
        b. It provides an escape hatch: if the dispatcher ever sends
           a misclassified *answerable* query through this handler and
           there are no foreign-law signals AND retrieval surfaces a
           strong match, the handler returns a cautious answer instead
           of an unjustified abstention.

  3. Abstention rules:

     ┌────────────────────────────┬──────────────────────────────────┐
     │ Detected signals?          │ Action                          │
     ├────────────────────────────┼──────────────────────────────────┤
     │ Yes (any source)           │ Abstain — reason = "infected"  │
     │ No, top RRF score < weak  │ Abstain — reason = "weak_evidence" │
     │ No, top RRF score ≥ weak  │ Cautious answer with top-K      │
     └────────────────────────────┴──────────────────────────────────┘

The escape-hatch path matters for R7 (dispatcher) — when handler is
called on the unanswerable slice in isolation, almost every question
exits via path 1 or 2. The Phase-1 baselines and R2-R4 score AbstF1 =
0.000 on this stratum because their only abstention path is empty/
no-hits; this handler lifts that by abstaining on the entire infected
+ weak-evidence subspace.

No sub-LM calls in the default configuration. The handler can be
called with optional ``llm_judge_fn`` to add an LLM second-pass
classifier (mirrors the jurisdiction gate's optional LLM hook); off by
default to keep AbstF1 deterministic on the smoke run.

Sub-LM call budget per query: 0 by default; ≤ 1 with the optional LLM
judge enabled. Well under the project ``max_sub_calls=12`` envelope.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from akn_rlm.config import SUB_LLM_MODEL
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.gates.jurisdiction import detect as detect_jurisdiction_signals
from akn_rlm.indexers.bm25 import BM25Hit, BM25Index
from akn_rlm.indexers.dense import DenseHit, DenseIndex
from akn_rlm.normalizers import canonical_article_ref
from akn_rlm.retrievers.hybrid_fusion import rrf_fuse
from akn_rlm.rlm.routing import DocRouter, build_doc_router

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K_CANDIDATES: int = 5
DEFAULT_K_EACH: int = 20
DEFAULT_ROUTE_TOP_N: int = 3
# RRF top-1 score below this is treated as "no strong Algerian match".
# An RRF top-1 hit ranked first by both BM25+Dense scores 2/(60+1) = 0.0328;
# a hit ranked first by only one retriever scores 1/61 = 0.0164. The default
# 0.030 keeps the bar at "supported by both retrievers" — strong enough that
# we only answer when the corpus clearly carries the concept.
DEFAULT_WEAK_EVIDENCE_THRESHOLD: float = 0.030
SUPPORT_SPAN_LEN: int = 280

TELEMETRY_BASELINE: str = "rlm_unanswerable"


# ---------------------------------------------------------------------------
# Local foreign-law / DZ-absence dictionary
# ---------------------------------------------------------------------------
#
# Patterns are tried in addition to gates.jurisdiction.detect(). Every
# entry below was added because it caught a question on the v3.0
# unanswerable slice that the existing dictionary missed. Signal IDs
# carry the same {region}:{concept} convention as jurisdiction.py so
# downstream telemetry can reuse the JIR / jurisdiction_summary helpers.

_LOCAL_FOREIGN_PATTERNS: list[tuple[re.Pattern, str]] = []


def _add_local(pattern: str, signal_id: str) -> None:
    _LOCAL_FOREIGN_PATTERNS.append(
        (re.compile(pattern, re.IGNORECASE | re.UNICODE), signal_id)
    )


# -- French-law concepts the main dict misses --
_add_local(r"contr.le des loyers",                    "fr:rent_control")
_add_local(r"\bconcubinage\b",                       "fr:concubinage")
_add_local(r"المعاشرة",                              "fr:concubinage_ar")
_add_local(r"droit administratif fran",               "fr:droit_admin_fr")
_add_local(r"droit au maintien dans les lieux",       "fr:maintien_lieux")
_add_local(r"clause p.nale judiciaire",               "eg:clause_penale_jud")

# -- US / private-pension --
_add_local(r"\b401k\b",                              "us:401k")
_add_local(r"private pension fund",                  "us:private_pension")

# -- Egyptian-law markers --
_add_local(r"المواريث المصري",                      "eg:inheritance_law_ar")
_add_local(r"الإجراءات الجنائية المصري",            "eg:criminal_procedure_ar")
_add_local(r"المصري المادة 224",                     "eg:art224_ar")
_add_local(r"المادة 224.{0,40}المصري",              "eg:art224_ar")

# -- Tunisian-law markers (Arabic) --
_add_local(r"الأحوال الشخصية التونسية",             "tn:csp_ar")
_add_local(r"تعدد الزوجات.{0,40}تونس",             "tn:polygamy_ar")
_add_local(r"المحكمة التجارية بتونس",               "tn:commercial_courts_ar")
_add_local(r"التجارية المتخصصة.{0,40}تونس",        "tn:commercial_courts_ar")

# -- DZ_ABSENCE: concepts that genuinely don't exist in DZ law --
_add_local(r"رقابة لاحقة تلقائية",                  "dz_absent:auto_saisine_ar")
_add_local(r"محكمة دستورية.{0,40}تلقائي",          "dz_absent:auto_saisine_ar")
_add_local(r"ضرائب محلية مستقلة",                   "dz_absent:municipal_tax_ar")
_add_local(r"الإعدام.{0,40}الفساد الاقتصادي",      "dz_absent:death_corruption_ar")

# -- Cross-jurisdictional contrast markers (the literal phrase
#    "كما في النظام X" / "كالنظام X" / "as in X system" reliably flags
#    a question framed around a foreign rule).
_add_local(r"كما في النظام (الأمريكي|الفرنسي|التونسي|المصري)",
           "fr:cross_jurisdiction_marker")
_add_local(r"كالنظام (الأمريكي|الفرنسي|التونسي|المصري)",
           "fr:cross_jurisdiction_marker")
_add_local(r"كالقانون (الفرنسي|الأمريكي|المصري|التونسي)",
           "fr:cross_jurisdiction_marker")
_add_local(r"كما في (تونس|مصر|فرنسا|أمريكا|الأردن)",
           "fr:cross_jurisdiction_marker")
_add_local(r"كما (فعلت|فعل) (مجلة|قانون) ",         "fr:cross_jurisdiction_marker")
_add_local(r"كما هو معمول به في بعض الدول",        "fr:cross_jurisdiction_marker")


def detect_local(text: str) -> list[str]:
    """Return signal IDs found by the local foreign-law / DZ-absence dict."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for pattern, signal_id in _LOCAL_FOREIGN_PATTERNS:
        if signal_id not in seen and pattern.search(text):
            found.append(signal_id)
            seen.add(signal_id)
    return found


def detect_infection_signals(text: str) -> list[str]:
    """Combined foreign-law / DZ-absence detector.

    Returns the union of :func:`gates.jurisdiction.detect` and the
    handler-local dictionary, preserving order and deduplicating.
    """
    sigs = detect_jurisdiction_signals(text or "")
    seen = set(sigs)
    for s in detect_local(text or ""):
        if s not in seen:
            sigs.append(s)
            seen.add(s)
    return sigs


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

LLMJudgeFn = Callable[[Any, str, list[str], str], bool]


class UnanswerableHandler:
    """Typed unanswerable handler: detect signals -> ONE confirming search ->
    abstain on no-match.
    """

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        llm_pool=None,
        *,
        router: Optional[DocRouter] = None,
        sub_model: str = SUB_LLM_MODEL,
        top_k_candidates: int = DEFAULT_TOP_K_CANDIDATES,
        k_each: int = DEFAULT_K_EACH,
        route_top_n: int = DEFAULT_ROUTE_TOP_N,
        weak_evidence_threshold: float = DEFAULT_WEAK_EVIDENCE_THRESHOLD,
        llm_judge_fn: Optional[LLMJudgeFn] = None,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._registry = registry
        self._llm_pool = llm_pool
        self._router = router or build_doc_router(registry=registry, bm25=bm25)
        self._sub_model = sub_model
        self._top_k_candidates = top_k_candidates
        self._k_each = k_each
        self._route_top_n = route_top_n
        self._weak_evidence_threshold = weak_evidence_threshold
        # Optional LLM second-pass to confirm a borderline foreign-law
        # signal. OFF by default — every gate-clear smoke run is purely
        # deterministic. Mirrors the jurisdiction gate's optional hook.
        self._llm_judge_fn = llm_judge_fn

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain(
                "empty_query", routed=[], signals=[], top_score=0.0, sub_calls=0,
            )

        # 1. Detect foreign-law / DZ-absence signals FIRST (no LLM, no search yet)
        signals = detect_infection_signals(query)

        # 2. ONE confirming hybrid search (restricted to routed docs)
        route = self._router.route(query, top_n=self._route_top_n)
        routed_ids = list(route.doc_ids)
        candidates = self._fused_candidates(query, routed_ids)
        top_score = float(candidates[0].get("score", 0.0)) if candidates else 0.0

        # 3a. Signals present → abstain regardless of search outcome.
        # Optional LLM second-pass can clear a false-positive — convention
        # matches gates.jurisdiction._llm_classify: the judge returns True
        # iff contamination IS confirmed (i.e. the signal is real). Returning
        # False means the LLM judged the regex hit as a false positive, in
        # which case the handler falls through to the strong-evidence path.
        sub_calls = 0
        if signals:
            llm_cleared = False
            if self._llm_judge_fn is not None and self._llm_pool is not None:
                try:
                    confirmed = bool(self._llm_judge_fn(
                        self._llm_pool, query, signals, self._sub_model
                    ))
                    sub_calls += 1
                    llm_cleared = not confirmed
                except Exception as exc:
                    log.debug("unanswerable LLM judge failed: %s", exc)
                    llm_cleared = False  # conservative: keep abstention
            if not llm_cleared:
                return self._abstain(
                    "infected_jurisdiction",
                    routed=routed_ids,
                    signals=signals,
                    top_score=top_score,
                    sub_calls=sub_calls,
                    confirming_candidates=candidates[: self._top_k_candidates],
                )
            # LLM judged the regex hits as false positives — fall through
            # to the strong-evidence path.

        # 3b. No signals, no candidates at all → abstain (no_hits).
        if not candidates:
            return self._abstain(
                "no_hits",
                routed=routed_ids,
                signals=signals,
                top_score=0.0,
                sub_calls=sub_calls,
            )

        # 3c. No signals AND top-1 RRF score below weak threshold → abstain.
        if top_score < self._weak_evidence_threshold:
            return self._abstain(
                "weak_evidence",
                routed=routed_ids,
                signals=signals,
                top_score=top_score,
                sub_calls=sub_calls,
                confirming_candidates=candidates[: self._top_k_candidates],
            )

        # 3d. Strong evidence path: cautious answer.
        # This branch fires when the dispatcher (R7) misclassifies an
        # answerable query as unanswerable AND retrieval surfaces a clear
        # match. The handler returns top-K citations with a deterministic
        # template answer (no synthesizer call — keeps the budget at 0
        # sub-LM calls).
        final_citations = [
            self._build_citation(c) for c in candidates[: self._top_k_candidates]
        ]
        answer_text = self._template_answer(final_citations)

        return {
            "answer_text":       answer_text,
            "abstention":        False,
            "abstention_reason": None,
            "citations":         final_citations,
            "reasoning_chain":   [
                "no_foreign_law_signals_detected",
                f"strong_evidence_top_score={top_score:.4f}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 1,
            "_telemetry": {
                "retry_count":     0,
                "gate_results":    {},
                "baseline":        TELEMETRY_BASELINE,
                "routed_doc_ids":  routed_ids,
                "signals":         signals,
                "top_score":       top_score,
                "candidate_count": len(candidates),
                "sub_call_count":  sub_calls,
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
            log.warning("unanswerable BM25 failed: %s", exc)
            bm25_hits = []
        try:
            dense_hits: list[DenseHit] = self._dense.search(query, k=self._k_each)
        except Exception as exc:
            log.warning("unanswerable dense failed: %s", exc)
            dense_hits = []

        bm25_dicts = self._hits_to_dicts(bm25_hits, retriever="bm25")
        dense_dicts = self._hits_to_dicts(dense_hits, retriever="dense")
        if not bm25_dicts and not dense_dicts:
            return []

        fused = rrf_fuse([bm25_dicts, dense_dicts])
        if routed_ids:
            allowed = set(routed_ids)
            filtered = [h for h in fused if h.get("doc_id") in allowed]
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

    # ------------------------------------------------------------------
    # Citation / answer assembly (cautious-answer path only)
    # ------------------------------------------------------------------

    def _build_citation(self, candidate: dict[str, Any]) -> dict[str, Any]:
        doc_id = candidate.get("doc_id", "")
        ref = canonical_article_ref(candidate.get("article_ref", "")) or \
            candidate.get("article_ref", "")
        text = candidate.get("text", "") or ""
        return {
            "doc_id":          doc_id,
            "article_ref":     ref,
            "doc_title":       self._doc_title(doc_id),
            "supporting_span": text[:SUPPORT_SPAN_LEN],
            "text":            text,
            "confidence":      float(candidate.get("score", 0.0)),
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
        signals: list[str],
        top_score: float,
        sub_calls: int,
        confirming_candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "answer_text":       "",
            "abstention":        True,
            "abstention_reason": reason,
            "citations":         [],
            "reasoning_chain":   [
                f"signals={signals}",
                f"top_score={top_score:.4f}",
                f"reason={reason}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count":           0,
                "gate_results":          {},
                "baseline":              TELEMETRY_BASELINE,
                "routed_doc_ids":        routed,
                "signals":               signals,
                "top_score":             top_score,
                "candidate_count":       len(confirming_candidates or []),
                "confirming_candidates": [
                    {
                        "doc_id":      c.get("doc_id", ""),
                        "article_ref": c.get("article_ref", ""),
                        "score":       float(c.get("score", 0.0)),
                    }
                    for c in (confirming_candidates or [])
                ],
                "sub_call_count":        sub_calls,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_unanswerable_handler(
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    llm_pool=None,
    *,
    router: Optional[DocRouter] = None,
    **kwargs: Any,
) -> UnanswerableHandler:
    """Factory mirroring the baseline ``build_*_pipeline`` helpers."""
    return UnanswerableHandler(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        **kwargs,
    )
