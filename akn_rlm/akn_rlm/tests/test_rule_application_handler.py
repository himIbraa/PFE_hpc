"""Tests for the Phase-2 rule_application handler.

The handler must mirror the baseline ``.run(query) -> dict`` contract so
``akn_rlm.eval.runner._answer_to_result`` consumes it unchanged. BM25,
Dense, the verifier, and the summariser are fully mocked — the suite
never touches a real index or LLM and runs in milliseconds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akn_rlm.eval.runner import _answer_to_result
from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.indexers.dense import DenseHit
from akn_rlm.rlm.handlers.rule_application import (
    DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH,
    DEFAULT_ROUTE_TOP_N,
    DEFAULT_TOP_K_CANDIDATES,
    DEFAULT_VERIFY_THRESHOLD,
    SUPPORT_SPAN_LEN,
    TELEMETRY_BASELINE,
    RuleApplicationHandler,
    build_rule_application_handler,
)
from akn_rlm.rlm.routing import RouteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm25(doc_id: str, ref: str, text: str = "نص المادة", score: float = 5.0) -> BM25Hit:
    return BM25Hit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _dense(doc_id: str, ref: str, text: str = "نص المادة", score: float = 0.5) -> DenseHit:
    return DenseHit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _stub_router(doc_ids: list[str]) -> MagicMock:
    router = MagicMock()
    router.route.return_value = RouteResult(
        doc_ids=list(doc_ids),
        scores={d: 1.0 for d in doc_ids},
        sources={d: ["alias"] for d in doc_ids},
        confidence=1.0 if doc_ids else 0.0,
    )
    return router


def _stub_verifier(verdict_for: dict[tuple[str, str], dict] | None = None,
                   default: dict | None = None):
    verdict_for = verdict_for or {}
    default = default or {
        "relevant": True,
        "supporting_span": None,
        "contradicting_span": None,
        "confidence": 0.9,
    }

    def _fn(_pool, _q, article, _model):
        key = (article.get("doc_id", ""), article.get("article_ref", ""))
        return dict(verdict_for.get(key, default))

    return MagicMock(side_effect=_fn)


def _stub_summarizer(summary: str | None = "ملخّص"):
    def _fn(_pool, _q, _articles, _model):
        return {"summary": summary, "key_articles": [], "caveats": None}

    return MagicMock(side_effect=_fn)


def _make_handler(
    *,
    bm25_hits: list[BM25Hit] | None = None,
    dense_hits: list[DenseHit] | None = None,
    routed_ids: list[str] | None = None,
    verifier_verdicts: dict[tuple[str, str], dict] | None = None,
    summary: str | None = "ملخّص",
    doc_title: str = "قانون الأسرة",
    **kwargs,
):
    bm25 = MagicMock()
    bm25.search.return_value = list(bm25_hits or [])
    dense = MagicMock()
    dense.search.return_value = list(dense_hits or [])

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)

    router = _stub_router(routed_ids if routed_ids is not None else ["84-11_1984-06-09"])
    verifier = _stub_verifier(verifier_verdicts)
    summarizer = _stub_summarizer(summary)

    handler = RuleApplicationHandler(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=MagicMock(),
        router=router,
        verifier_fn=verifier,
        summarizer_fn=summarizer,
        **kwargs,
    )
    return handler, dict(
        bm25=bm25, dense=dense, registry=registry, router=router,
        verifier=verifier, summarizer=summarizer,
    )


# ---------------------------------------------------------------------------
# Defaults & contract
# ---------------------------------------------------------------------------


def test_defaults_match_handoff_design():
    """HANDOFF §3 says rule_application uses top-K=8."""
    assert DEFAULT_TOP_K_CANDIDATES == 8
    # F5: reverted F4's 8→4 back to 8 — F4 evidence showed the tighten
    # was net-zero on RA itself but regressed layman by −0.045 (which
    # delegates to RA). Precision lift instead comes from R9.5 supervisor.
    assert DEFAULT_FINAL_TOP_K == 8
    assert DEFAULT_K_EACH == 30
    # F4: reverted R9.1's 0.5→0.3 back to 0.5 — F3 evidence shows the
    # looser threshold added noise (RA −0.004, layman −0.013).
    assert DEFAULT_VERIFY_THRESHOLD == 0.5
    assert DEFAULT_ROUTE_TOP_N == 3


def test_f4_default_verify_threshold_locked_at_0_5():
    """F4 reverted R9.1: threshold back to 0.5 after evidence of noise on
    RA + layman in the F3 244-q run."""
    from akn_rlm.rlm.handlers import rule_application as ra_mod
    assert ra_mod.DEFAULT_VERIFY_THRESHOLD == 0.5


def test_f5_default_final_top_k_locked_at_8():
    """F5 reverted F4's 8→4 — that change was net-zero on RA but hurt
    the layman handler (which delegates to RA) by −0.045."""
    from akn_rlm.rlm.handlers import rule_application as ra_mod
    assert ra_mod.DEFAULT_FINAL_TOP_K == 8


def test_factory_builds_handler():
    bm25 = MagicMock(); dense = MagicMock(); registry = MagicMock()
    h = build_rule_application_handler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MagicMock(),
        router=_stub_router(["d"]),
    )
    assert isinstance(h, RuleApplicationHandler)


def test_telemetry_baseline_tag_is_rlm_rule_application():
    handler, _ = _make_handler(bm25_hits=[_bm25("84-11_1984-06-09", "5")])
    out = handler.run("ما هي شروط الزواج؟")
    assert out["_telemetry"]["baseline"] == TELEMETRY_BASELINE
    assert TELEMETRY_BASELINE == "rlm_rule_application"


def test_run_returns_required_keys():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص المادة", score=12.0)],
    )
    out = handler.run("ما هي شروط الزواج؟")
    for key in ("answer_text", "abstention", "abstention_reason", "citations",
                "reasoning_chain", "trajectory", "tokens_used",
                "depth_max_reached", "_telemetry"):
        assert key in out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_router_called_with_route_top_n():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5")],
        route_top_n=2,
    )
    handler.run("سؤال")
    mocks["router"].route.assert_called_once()
    _args, kwargs = mocks["router"].route.call_args
    assert kwargs.get("top_n") == 2


def test_routed_doc_ids_recorded_in_telemetry():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5")],
        routed_ids=["84-11_1984-06-09", "75-58_1975-09-26"],
    )
    out = handler.run("سؤال")
    assert out["_telemetry"]["routed_doc_ids"] == [
        "84-11_1984-06-09", "75-58_1975-09-26",
    ]


def test_retrieval_filtered_by_routed_doc_ids():
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "5", "match", score=10.0),
            _bm25("66-156_1966-06-08", "9", "wrong-doc", score=20.0),
        ],
        routed_ids=["84-11_1984-06-09"],
    )
    out = handler.run("سؤال")
    citations = out["citations"]
    assert all(c["doc_id"] == "84-11_1984-06-09" for c in citations)


def test_full_pool_fallback_when_routed_filter_wipes_everything():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("66-156_1966-06-08", "9", "only-doc", score=5.0)],
        routed_ids=["84-11_1984-06-09"],  # routed doc is NOT in hits
    )
    out = handler.run("سؤال")
    # Filter would drop everything → fall back to unrestricted fused list.
    assert out["abstention"] is False
    assert any(c["doc_id"] == "66-156_1966-06-08" for c in out["citations"])


# ---------------------------------------------------------------------------
# Retrieval — BM25 + Dense fused
# ---------------------------------------------------------------------------


def test_both_retrievers_called_at_k_each():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5")],
        dense_hits=[_dense("84-11_1984-06-09", "5")],
        k_each=42,
    )
    handler.run("سؤال")
    mocks["bm25"].search.assert_called_once()
    mocks["dense"].search.assert_called_once()
    assert mocks["bm25"].search.call_args.kwargs.get("k") == 42
    assert mocks["dense"].search.call_args.kwargs.get("k") == 42


def test_top_k_candidates_truncates_pool_before_verify():
    """top_k_candidates=2 should mean only top-2 fused candidates are verified."""
    bm25_hits = [
        _bm25("84-11_1984-06-09", str(i), text=f"text-{i}", score=20.0 - i)
        for i in range(1, 6)
    ]
    handler, mocks = _make_handler(
        bm25_hits=bm25_hits,
        top_k_candidates=2,
    )
    handler.run("سؤال")
    # Verifier called exactly top_k_candidates times
    assert mocks["verifier"].call_count == 2


def test_dedup_across_repeated_articles():
    """The same (doc_id, ref) showing up twice should collapse to one citation."""
    bm25_hits = [
        _bm25("84-11_1984-06-09", "5", text="A", score=10.0),
        _bm25("84-11_1984-06-09", "5", text="B", score=8.0),  # duplicate (doc, ref)
    ]
    handler, _ = _make_handler(bm25_hits=bm25_hits)
    out = handler.run("سؤال")
    keys = [(c["doc_id"], c["article_ref"]) for c in out["citations"]]
    assert keys.count(("84-11_1984-06-09", "5")) == 1


def test_canonicalisation_collapses_arabic_bis():
    """`9 مكرر` and `9_bis` should collapse to a single canonical citation."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "9 مكرر", text="A", score=10.0)],
        dense_hits=[_dense("84-11_1984-06-09", "9_bis", text="A", score=0.6)],
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"9_bis"}


def test_canonicalisation_arabic_ordinal_first():
    """`الأولى` should canonicalise to `1`."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "الأولى", text="A", score=10.0)],
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"1"}


# ---------------------------------------------------------------------------
# Mandatory verifier
# ---------------------------------------------------------------------------


def test_verifier_called_for_every_top_k_candidate():
    """HANDOFF §3 — mandatory verify_article filter on every top-K candidate."""
    bm25_hits = [
        _bm25("84-11_1984-06-09", str(i), text=f"text-{i}", score=20.0 - i)
        for i in range(1, 11)  # 10 candidates
    ]
    handler, mocks = _make_handler(
        bm25_hits=bm25_hits,
        top_k_candidates=8,
    )
    handler.run("سؤال")
    # Verifier must be called for every top-K candidate (HANDOFF: mandatory).
    assert mocks["verifier"].call_count == 8


def test_verifier_rejected_relevance_drops_citation():
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "5", text="kept", score=10.0),
            _bm25("84-11_1984-06-09", "8", text="dropped", score=8.0),
        ],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "confidence": 0.9},
            ("84-11_1984-06-09", "8"): {"relevant": False, "confidence": 0.95},
        },
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"5"}


def test_verifier_low_confidence_drops_citation():
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "5", text="A", score=10.0),
            _bm25("84-11_1984-06-09", "8", text="B", score=8.0),
        ],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "confidence": 0.9},
            ("84-11_1984-06-09", "8"): {"relevant": True, "confidence": 0.4},
        },
        verify_threshold=0.5,
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"5"}


def test_verifier_exception_skips_candidate_without_crashing():
    def raising_verifier(*_a, **_k):
        raise RuntimeError("verifier blew up")
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5")],
    )
    handler._verifier_fn = raising_verifier
    out = handler.run("سؤال")
    # The lone candidate's verifier raised → no verified articles → abstain.
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"


def test_dedup_keeps_highest_confidence_verdict():
    """If a candidate appears twice (e.g. same article from BM25 + Dense after
    dedup, or duplicate verdicts), keep the highest-confidence one."""
    handler, mocks = _make_handler(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "5", text="A", score=10.0),
            _bm25("84-11_1984-06-09", "5", text="A2", score=8.0),
        ],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "confidence": 0.7},
        },
    )
    out = handler.run("سؤال")
    cit = next(c for c in out["citations"] if c["article_ref"] == "5")
    assert cit["confidence"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def test_summarizer_summary_used_as_answer_text():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5")],
        summary="هذا هو الجواب",
    )
    out = handler.run("سؤال")
    assert out["answer_text"] == "هذا هو الجواب"


def test_null_summary_falls_back_to_template():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص محدد", score=10.0)],
        summary=None,
    )
    out = handler.run("سؤال")
    assert "وفقًا لـ" in out["answer_text"]
    assert "المادة 5" in out["answer_text"]


def test_summarizer_exception_falls_back_to_template():
    def raising_summarizer(*_a, **_k):
        raise RuntimeError("summariser blew up")
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص محدد", score=10.0)],
    )
    handler._summarizer_fn = raising_summarizer
    out = handler.run("سؤال")
    assert "وفقًا لـ" in out["answer_text"]


def test_summarizer_called_once_with_final_citations_and_query():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "text-5", score=10.0)],
    )
    handler.run("سؤال محدد")
    mocks["summarizer"].assert_called_once()
    args = mocks["summarizer"].call_args.args
    # Signature: (llm_pool, query, articles, model)
    assert args[1] == "سؤال محدد"
    assert isinstance(args[2], list) and len(args[2]) == 1
    assert args[2][0]["article_ref"] == "5"


# ---------------------------------------------------------------------------
# Citation shape
# ---------------------------------------------------------------------------


def test_citation_shape_carries_doc_title_span_and_confidence():
    text = "نص قانوني" * 50
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", text, score=12.0)],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {
                "relevant": True,
                "confidence": 0.85,
                "supporting_span": text[10:50],
            },
        },
        doc_title="قانون الأسرة",
    )
    out = handler.run("سؤال")
    cit = out["citations"][0]
    assert cit["doc_id"] == "84-11_1984-06-09"
    assert cit["article_ref"] == "5"
    assert cit["doc_title"] == "قانون الأسرة"
    assert cit["confidence"] == pytest.approx(0.85)
    assert cit["verifier_relevant"] is True
    # Verifier's supporting_span is a substring of text → used directly.
    assert cit["supporting_span"] == text[10:50]
    assert len(cit["supporting_span"]) <= SUPPORT_SPAN_LEN
    assert cit["text"] == text


def test_supporting_span_falls_back_when_quote_not_substring():
    """When the verifier's supporting_span isn't actually in the text,
    fall back to text[:280] — the span-existence gate downstream
    requires the span to be a substring of the article text."""
    text = "abc" * 200
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", text, score=12.0)],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {
                "relevant": True,
                "confidence": 0.9,
                "supporting_span": "FABRICATED — not in text",
            },
        },
    )
    out = handler.run("سؤال")
    cit = out["citations"][0]
    assert cit["supporting_span"] == text[:SUPPORT_SPAN_LEN]


def test_supporting_span_truncates_at_280_chars():
    text = "x" * 1000
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", text, score=12.0)],
    )
    out = handler.run("سؤال")
    assert len(out["citations"][0]["supporting_span"]) == SUPPORT_SPAN_LEN


def test_template_falls_back_to_doc_id_when_title_empty():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص", score=10.0)],
        summary=None,
        doc_title="",
    )
    out = handler.run("سؤال")
    assert "84-11_1984-06-09" in out["answer_text"]


# ---------------------------------------------------------------------------
# Aggregation / final_top_k
# ---------------------------------------------------------------------------


def test_final_top_k_truncates_after_verification():
    bm25_hits = [
        _bm25("84-11_1984-06-09", str(i), text=f"text-{i}", score=20.0 - i)
        for i in range(1, 11)
    ]
    handler, _ = _make_handler(
        bm25_hits=bm25_hits,
        top_k_candidates=10,
        final_top_k=3,
    )
    out = handler.run("سؤال")
    assert len(out["citations"]) == 3


def test_citations_ranked_by_confidence_desc():
    bm25_hits = [
        _bm25("84-11_1984-06-09", "5", text="A", score=10.0),
        _bm25("84-11_1984-06-09", "8", text="B", score=8.0),
        _bm25("84-11_1984-06-09", "3", text="C", score=12.0),
    ]
    handler, _ = _make_handler(
        bm25_hits=bm25_hits,
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "confidence": 0.7},
            ("84-11_1984-06-09", "8"): {"relevant": True, "confidence": 0.95},
            ("84-11_1984-06-09", "3"): {"relevant": True, "confidence": 0.6},
        },
    )
    out = handler.run("سؤال")
    refs = [c["article_ref"] for c in out["citations"]]
    assert refs == ["8", "5", "3"]


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------


def test_empty_query_abstains_without_calling_anything():
    handler, mocks = _make_handler()
    out = handler.run("")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    mocks["bm25"].search.assert_not_called()
    mocks["dense"].search.assert_not_called()
    mocks["verifier"].assert_not_called()
    mocks["summarizer"].assert_not_called()


def test_whitespace_query_abstains():
    handler, _ = _make_handler()
    out = handler.run("   \n\t  ")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"


def test_no_hits_abstains():
    handler, mocks = _make_handler(bm25_hits=[], dense_hits=[])
    out = handler.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"
    mocks["verifier"].assert_not_called()
    mocks["summarizer"].assert_not_called()


def test_no_verified_articles_abstains():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5")],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": False, "confidence": 0.95},
        },
    )
    out = handler.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"
    mocks["summarizer"].assert_not_called()


# ---------------------------------------------------------------------------
# Retriever-failure tolerance
# ---------------------------------------------------------------------------


def test_bm25_exception_degrades_gracefully():
    bm25 = MagicMock()
    bm25.search.side_effect = RuntimeError("bm25 oops")
    dense = MagicMock()
    dense.search.return_value = [_dense("84-11_1984-06-09", "5", "نص", 0.6)]
    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="x")
    h = RuleApplicationHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MagicMock(),
        router=_stub_router(["84-11_1984-06-09"]),
        verifier_fn=_stub_verifier(),
        summarizer_fn=_stub_summarizer(),
    )
    out = h.run("سؤال")
    # Dense alone should still produce a citation.
    assert out["abstention"] is False
    assert out["citations"][0]["doc_id"] == "84-11_1984-06-09"


def test_dense_exception_degrades_gracefully():
    bm25 = MagicMock()
    bm25.search.return_value = [_bm25("84-11_1984-06-09", "5", "نص", 5.0)]
    dense = MagicMock()
    dense.search.side_effect = RuntimeError("dense oops")
    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="x")
    h = RuleApplicationHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MagicMock(),
        router=_stub_router(["84-11_1984-06-09"]),
        verifier_fn=_stub_verifier(),
        summarizer_fn=_stub_summarizer(),
    )
    out = h.run("سؤال")
    assert out["abstention"] is False


# ---------------------------------------------------------------------------
# Sub-LM call budget
# ---------------------------------------------------------------------------


def test_sub_call_budget_at_most_top_k_plus_one():
    """HANDOFF §3 says rule_application sub-LM budget ≤ verifier(top-K=8) +
    1 summariser = 9."""
    bm25_hits = [
        _bm25("84-11_1984-06-09", str(i), f"text-{i}", score=20.0 - i)
        for i in range(1, 11)
    ]
    handler, _ = _make_handler(
        bm25_hits=bm25_hits,
        top_k_candidates=8,
    )
    out = handler.run("سؤال")
    assert out["_telemetry"]["sub_call_count"] <= 9


def test_no_summariser_call_on_abstention():
    handler, mocks = _make_handler(bm25_hits=[], dense_hits=[])
    handler.run("سؤال")
    mocks["summarizer"].assert_not_called()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_telemetry_records_top_score_and_verified_count():
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "5", "نص", score=10.0),
            _bm25("84-11_1984-06-09", "8", "نص", score=8.0),
        ],
    )
    out = handler.run("سؤال")
    tel = out["_telemetry"]
    assert tel["top_score"] > 0
    assert tel["candidate_count"] == 2
    assert tel["verified_count"] == 2


# ---------------------------------------------------------------------------
# End-to-end — _answer_to_result compatibility
# ---------------------------------------------------------------------------


def test_answer_to_result_compatibility():
    """Handler output must flow through _answer_to_result without errors."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص", score=10.0)],
    )
    out = handler.run("ما هي شروط الزواج؟")
    out["_latency_s"] = 0.123
    question = {
        "id": "fam_ra_q01",
        "query": "ما هي شروط الزواج؟",
        "query_type": "rule_application",
        "legal_category": "family_law",
        "difficulty": "medium",
        "language": "ar",
        "split": "test",
        "gold_doc_ids": ["84-11_1984-06-09"],
        "gold_article_ids": ["84-11_1984-06-09#art_5"],
        "gold_citations": [{"doc_id": "84-11_1984-06-09", "article_ref": "5"}],
        "gold_abstain": False,
        "gold_answer": "...",
        "gold_reasoning_chain": [],
    }
    result = _answer_to_result(question, out)
    assert result["pred_doc_ids"] == ["84-11_1984-06-09"]
    assert result["pred_article_ids"] == ["84-11_1984-06-09#art_5"]
    assert result["predicted_abstain"] is False


def test_answer_to_result_compatibility_on_abstention():
    handler, _ = _make_handler(bm25_hits=[], dense_hits=[])
    out = handler.run("سؤال")
    out["_latency_s"] = 0.05
    question = {
        "id": "q",
        "query": "سؤال",
        "query_type": "rule_application",
        "gold_doc_ids": ["x"],
        "gold_article_ids": ["x#art_1"],
        "gold_citations": [{"doc_id": "x", "article_ref": "1"}],
        "gold_abstain": False,
    }
    result = _answer_to_result(question, out)
    assert result["predicted_abstain"] is True
    assert result["pred_doc_ids"] == []
