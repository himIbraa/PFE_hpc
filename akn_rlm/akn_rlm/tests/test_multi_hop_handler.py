"""Tests for the Phase-2 multi_hop handler.

The handler must mirror the baseline ``.run(query) -> dict`` contract so
``akn_rlm.eval.runner._answer_to_result`` consumes it unchanged. The
internal LLM-call wrappers (decomposer / verifier / summariser) are
fully mocked so the suite never touches a real LLM and runs in
milliseconds — same pattern as ``test_hybrid_rerank_baseline.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.indexers.dense import DenseHit
from akn_rlm.rlm.handlers.multi_hop import (
    DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH,
    DEFAULT_MAX_SUB_QS,
    DEFAULT_TOP_K_PER_SUBQ,
    DEFAULT_VERIFY_THRESHOLD,
    DEFAULT_VERIFY_TOP_N,
    SUPPORT_SPAN_LEN,
    TELEMETRY_BASELINE,
    MultiHopHandler,
    build_multi_hop_handler,
)
from akn_rlm.rlm.routing import RouteResult


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def _bm25(doc_id: str, ref: str, text: str, score: float = 5.0) -> BM25Hit:
    return BM25Hit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _dense(doc_id: str, ref: str, text: str, score: float = 0.5) -> DenseHit:
    return DenseHit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _stub_router(doc_ids: list[str]) -> MagicMock:
    """Build a DocRouter mock that always returns the given doc_ids."""
    router = MagicMock()
    router.route.return_value = RouteResult(
        doc_ids=list(doc_ids),
        scores={d: 1.0 for d in doc_ids},
        sources={d: ["alias"] for d in doc_ids},
        confidence=1.0 if doc_ids else 0.0,
    )
    return router


def _stub_decomposer(sub_questions: list[dict]):
    def _fn(_pool, _query, _model):
        return {"sub_questions": list(sub_questions), "max_depth_needed": 1}
    return MagicMock(side_effect=_fn)


def _stub_verifier(verdict_for: dict[tuple[str, str], dict] | None = None,
                   default: dict | None = None):
    """Verifier that returns a per-(doc_id, ref) verdict, falling back to default."""
    verdict_for = verdict_for or {}
    default = default or {
        "relevant": True,
        "supporting_span": None,
        "contradicting_span": None,
        "confidence": 0.9,
    }

    def _fn(_pool, _sub_q, article, _model):
        key = (article.get("doc_id", ""), article.get("article_ref", ""))
        return dict(verdict_for.get(key, default))

    return MagicMock(side_effect=_fn)


def _stub_summarizer(summary: str | None = "ملخّص المحامي"):
    def _fn(_pool, _q, _articles, _model):
        return {"summary": summary, "key_articles": [], "caveats": None}
    return MagicMock(side_effect=_fn)


def _make_handler(
    *,
    bm25_hits: list[BM25Hit] | None = None,
    dense_hits: list[DenseHit] | None = None,
    routed_ids: list[str] | None = None,
    sub_questions: list[dict] | None = None,
    verifier_verdicts: dict[tuple[str, str], dict] | None = None,
    summary: str | None = "ملخّص المحامي",
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

    decomposer = _stub_decomposer(
        sub_questions
        if sub_questions is not None
        else [{"id": "sq1", "text": "first sub-q", "type": "rule_application"}]
    )
    verifier = _stub_verifier(verifier_verdicts)
    summarizer = _stub_summarizer(summary)

    handler = MultiHopHandler(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=MagicMock(),
        router=router,
        decomposer_fn=decomposer,
        verifier_fn=verifier,
        summarizer_fn=summarizer,
        **kwargs,
    )
    return handler, dict(
        bm25=bm25, dense=dense, registry=registry, router=router,
        decomposer=decomposer, verifier=verifier, summarizer=summarizer,
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_match_handoff_design():
    # R9.3 budget kept (was 5 / 5 / 3 / 3 in F2).
    # F5: kept R9.3's final_top_k=10 — F4's tightening to 5 didn't help.
    assert DEFAULT_FINAL_TOP_K == 10
    assert DEFAULT_TOP_K_PER_SUBQ == 8
    assert DEFAULT_VERIFY_TOP_N == 4
    assert DEFAULT_MAX_SUB_QS == 5
    assert DEFAULT_K_EACH == 30
    # F4: reverted R9.1's 0.5→0.3 back to 0.5 (F3 evidence of noise).
    assert DEFAULT_VERIFY_THRESHOLD == 0.5


def test_f4_default_verify_threshold_locked_at_0_5():
    from akn_rlm.rlm.handlers import multi_hop as mh_mod
    assert mh_mod.DEFAULT_VERIFY_THRESHOLD == 0.5


def test_f5_default_final_top_k_locked_at_10():
    """F5 reverted F4's 10→5 — F4 evidence showed MH didn't respond
    to tightening (0.121→0.118)."""
    from akn_rlm.rlm.handlers import multi_hop as mh_mod
    assert mh_mod.DEFAULT_FINAL_TOP_K == 10


def test_r9_3_budget_expansion_locked():
    """R9.3: lock the multi_hop budget expansion. The 4 retuned defaults +
    the new per-handler max_sub_calls=25 envelope must all stay put.
    Worst-case budget is 1 decomp + max_sub_qs * verify_top_n + 1 summary
    = 1 + 5*4 + 1 = 22 calls, comfortably under the 25-call cap.
    """
    from akn_rlm.rlm.handlers import multi_hop as mh_mod
    assert mh_mod.DEFAULT_MAX_SUB_QS == 5
    assert mh_mod.DEFAULT_VERIFY_TOP_N == 4
    # F5: kept R9.3's 10 (F4 tightening to 5 didn't lift MH).
    assert mh_mod.DEFAULT_FINAL_TOP_K == 10
    assert mh_mod.DEFAULT_TOP_K_PER_SUBQ == 8
    assert mh_mod.DEFAULT_MAX_SUB_CALLS == 25
    worst_case = 1 + mh_mod.DEFAULT_MAX_SUB_QS * mh_mod.DEFAULT_VERIFY_TOP_N + 1
    assert worst_case <= mh_mod.DEFAULT_MAX_SUB_CALLS


def test_r9_3_max_sub_calls_surfaced_in_telemetry():
    """The handler ctor must accept max_sub_calls and surface it in
    _telemetry on both the success and abstain paths."""
    from unittest.mock import MagicMock
    from akn_rlm.rlm.handlers.multi_hop import (
        DEFAULT_MAX_SUB_CALLS,
        MultiHopHandler,
    )

    # Empty-query abstain path
    h = MultiHopHandler(
        bm25=MagicMock(), dense=MagicMock(), registry=MagicMock(),
        llm_pool=MagicMock(), router=_stub_router([]),
    )
    out = h.run("")
    assert out["_telemetry"]["max_sub_calls"] == DEFAULT_MAX_SUB_CALLS

    # Override path
    h2 = MultiHopHandler(
        bm25=MagicMock(), dense=MagicMock(), registry=MagicMock(),
        llm_pool=MagicMock(), router=_stub_router([]),
        max_sub_calls=99,
    )
    out2 = h2.run("")
    assert out2["_telemetry"]["max_sub_calls"] == 99


def test_factory_builds_handler():
    bm25 = MagicMock(); dense = MagicMock(); registry = MagicMock()
    h = build_multi_hop_handler(bm25, dense, registry, llm_pool=MagicMock(),
                                router=_stub_router(["d"]))
    assert isinstance(h, MultiHopHandler)


# ---------------------------------------------------------------------------
# Contract — answer dict shape
# ---------------------------------------------------------------------------


def test_run_returns_required_keys():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص المادة")],
        dense_hits=[_dense("84-11_1984-06-09", "5", "نص المادة")],
    )
    out = handler.run("ما هي شروط الزواج؟")
    required = {
        "answer_text", "abstention", "abstention_reason", "citations",
        "reasoning_chain", "trajectory", "tokens_used", "_telemetry",
    }
    assert required.issubset(out.keys())


def test_telemetry_baseline_tag_is_rlm_multi_hop():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
    )
    out = handler.run("سؤال")
    assert out["_telemetry"]["baseline"] == TELEMETRY_BASELINE


def test_telemetry_records_routed_ids_and_sub_questions():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
        routed_ids=["84-11_1984-06-09", "75-58_1975-09-26"],
        sub_questions=[
            {"id": "sq1", "text": "first?", "type": "rule_application"},
            {"id": "sq2", "text": "second?", "type": "definition"},
        ],
    )
    out = handler.run("سؤال متعدد المراحل")
    tel = out["_telemetry"]
    assert tel["routed_doc_ids"] == ["84-11_1984-06-09", "75-58_1975-09-26"]
    assert [t["text"] for t in tel["sub_questions"]] == ["first?", "second?"]
    # Decomposer + 1 verifier (only one candidate per sub-q matches the
    # routed set above) + summariser
    assert tel["sub_call_count"] >= 1


def test_citation_shape_carries_doc_title_span_and_confidence():
    text = "نص قانوني" * 50
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", text, score=12.0)],
        dense_hits=[_dense("84-11_1984-06-09", "5", text, score=0.8)],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {
                "relevant": True,
                "supporting_span": text[:120],
                "contradicting_span": None,
                "confidence": 0.91,
            }
        },
    )
    out = handler.run("ما هي المادة 5؟")
    assert out["abstention"] is False
    assert len(out["citations"]) == 1
    c = out["citations"][0]
    assert c["doc_id"] == "84-11_1984-06-09"
    assert c["article_ref"] == "5"
    # Verifier's supporting_span (substring of text) must be carried over.
    assert c["supporting_span"] == text[:120]
    assert c["confidence"] == pytest.approx(0.91)
    assert c["verifier_relevant"] is True


def test_supporting_span_falls_back_when_quote_not_in_text():
    """If verifier hallucinates a quote not in the article, fall back to text[:280]."""
    text = "نص قانوني" * 50
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", text)],
        dense_hits=[],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {
                "relevant": True,
                "supporting_span": "هذا النص لا وجود له في المادة",
                "contradicting_span": None,
                "confidence": 0.7,
            }
        },
    )
    out = handler.run("سؤال")
    assert out["citations"][0]["supporting_span"] == text[:SUPPORT_SPAN_LEN]


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def test_decomposer_called_once_with_sub_model():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")], dense_hits=[],
    )
    handler.run("سؤال متشعب")
    mocks["decomposer"].assert_called_once()
    args, _ = mocks["decomposer"].call_args
    # call_decomposer(llm_pool, query, model)
    assert args[1] == "سؤال متشعب"
    assert isinstance(args[2], str)


def test_decomposer_failure_falls_back_to_single_sub_q():
    """Decomposer raising must NOT crash the handler."""
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")], dense_hits=[],
    )
    mocks["decomposer"].side_effect = RuntimeError("LLM down")
    out = handler.run("سؤال")
    assert out["abstention"] is False
    assert len(out["citations"]) == 1
    # Single fallback sub-question = original query
    assert out["_telemetry"]["sub_questions"][0]["text"] == "سؤال"


def test_max_sub_qs_cap():
    sub_qs = [
        {"id": f"sq{i}", "text": f"sub {i}?", "type": "rule_application"}
        for i in range(1, 8)
    ]
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
        sub_questions=sub_qs,
        max_sub_qs=3,
    )
    out = handler.run("متعدد")
    assert len(out["_telemetry"]["sub_questions"]) == 3


def test_foreign_law_sub_questions_are_dropped():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
        sub_questions=[
            {"id": "sq1", "text": "OK", "type": "rule_application"},
            {"id": "sq2", "text": "FR", "type": "foreign_law"},
        ],
    )
    out = handler.run("سؤال")
    sq_traces = out["_telemetry"]["sub_questions"]
    assert all(t["text"] != "FR" for t in sq_traces)


def test_only_foreign_law_sub_qs_triggers_specific_abstention():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
        sub_questions=[
            {"id": "sq1", "text": "FR1", "type": "foreign_law"},
            {"id": "sq2", "text": "FR2", "type": "foreign_law"},
        ],
    )
    out = handler.run("forbid foreign-law multi-hop")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "decomposer_only_foreign_law"


# ---------------------------------------------------------------------------
# Doc-routing + retrieval
# ---------------------------------------------------------------------------


def test_router_called_with_route_top_n():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
        route_top_n=2,
    )
    handler.run("q")
    mocks["router"].route.assert_called_once_with("q", top_n=2)


def test_retrieval_filtered_by_routed_doc_ids():
    """Only hits whose doc_id is in the routed set should survive."""
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "5", "ok", score=10.0),
            _bm25("DIFFERENT_DOC",     "9", "drop", score=10.0),
        ],
        dense_hits=[
            _dense("DIFFERENT_DOC", "12", "drop dense"),
        ],
        routed_ids=["84-11_1984-06-09"],
    )
    out = handler.run("سؤال")
    keys = {(c["doc_id"], c["article_ref"]) for c in out["citations"]}
    assert keys == {("84-11_1984-06-09", "5")}


def test_routing_filter_falls_back_to_full_pool_when_filter_wipes_everything():
    """If every hit is from a non-routed doc, we still answer (fallback)."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("OTHER_DOC", "5", "text")],
        dense_hits=[_dense("OTHER_DOC", "5", "text")],
        routed_ids=["84-11_1984-06-09"],  # routed says X, but only OTHER_DOC has hits
    )
    out = handler.run("سؤال")
    assert out["abstention"] is False
    assert any(c["doc_id"] == "OTHER_DOC" for c in out["citations"])


def test_retrievers_called_at_k_each_for_each_sub_question():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
        sub_questions=[
            {"id": "sq1", "text": "first?", "type": "rule_application"},
            {"id": "sq2", "text": "second?", "type": "rule_application"},
        ],
        k_each=42,
    )
    handler.run("متعدد")
    assert mocks["bm25"].search.call_count == 2
    assert mocks["dense"].search.call_count == 2
    for call in mocks["bm25"].search.call_args_list:
        assert call.kwargs.get("k") == 42 or call.args[1] == 42 or 42 in call.args


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_verifier_called_only_for_top_n_per_sub_question():
    """The handler must cap verifier calls per sub-q at verify_top_n."""
    bm25_hits = [
        _bm25("84-11_1984-06-09", str(i), f"text {i}", score=20 - i)
        for i in range(1, 8)
    ]
    handler, mocks = _make_handler(
        bm25_hits=bm25_hits,
        dense_hits=[],
        verify_top_n=2,
    )
    handler.run("سؤال")
    # 1 sub-q × 2 top candidates = 2 verifier calls
    assert mocks["verifier"].call_count == 2


def test_verifier_rejection_drops_citation():
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "5", "kept"),
            _bm25("84-11_1984-06-09", "9", "drop"),
        ],
        dense_hits=[],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "supporting_span": None,
                                          "contradicting_span": None, "confidence": 0.9},
            ("84-11_1984-06-09", "9"): {"relevant": False, "supporting_span": None,
                                          "contradicting_span": None, "confidence": 0.9},
        },
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"5"}


def test_low_confidence_verdict_drops_citation():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "low conf")],
        dense_hits=[],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "supporting_span": None,
                                          "contradicting_span": None, "confidence": 0.3},
        },
        verify_threshold=0.5,
    )
    out = handler.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"


def test_verifier_exception_skips_candidate_doesnt_crash():
    bm25_hits = [
        _bm25("84-11_1984-06-09", "5", "ok"),
        _bm25("84-11_1984-06-09", "9", "ok"),
    ]
    handler, mocks = _make_handler(
        bm25_hits=bm25_hits, dense_hits=[],
    )
    side_effects = [
        RuntimeError("LLM transient"),
        {"relevant": True, "supporting_span": None,
         "contradicting_span": None, "confidence": 0.9},
    ]
    mocks["verifier"].side_effect = side_effects
    out = handler.run("سؤال")
    assert out["abstention"] is False
    assert {c["article_ref"] for c in out["citations"]} == {"9"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_dedupe_across_sub_questions_keeps_highest_confidence():
    """Same article surfaced by two sub-qs → one citation with the best confidence."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
        sub_questions=[
            {"id": "sq1", "text": "first?", "type": "rule_application"},
            {"id": "sq2", "text": "second?", "type": "rule_application"},
        ],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "supporting_span": None,
                                          "contradicting_span": None, "confidence": 0.6},
        },
    )
    # Override: the verifier returns a different conf depending on which sub-q
    # is being verified — first call 0.6, second call 0.95.
    mocks_verifier = handler._verifier_fn
    confs = iter([0.6, 0.95])

    def _staged(_pool, _sq, article, _model):
        return {"relevant": True, "supporting_span": None,
                "contradicting_span": None, "confidence": next(confs)}
    mocks_verifier.side_effect = _staged

    out = handler.run("سؤال")
    assert len(out["citations"]) == 1
    assert out["citations"][0]["confidence"] == pytest.approx(0.95)


def test_canonicalises_article_ref_in_citation():
    """Arabic ordinals / bis variants must be normalised in the citation."""
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25("75-58_1975-09-26", "الأولى", "art 1 text"),
            _bm25("84-11_1984-06-09", "9 مكرر", "art 9 bis"),
        ],
        dense_hits=[
            _dense("84-11_1984-06-09", "9_bis", "art 9 bis dense"),
        ],
        routed_ids=["75-58_1975-09-26", "84-11_1984-06-09"],
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"1", "9_bis"}


def test_truncates_to_final_top_k():
    bm25_hits = [
        _bm25("84-11_1984-06-09", str(i), f"t{i}", score=20 - i)
        for i in range(1, 8)
    ]
    handler, _ = _make_handler(
        bm25_hits=bm25_hits,
        dense_hits=[],
        verify_top_n=8,
        top_k_per_subq=8,
        final_top_k=2,
    )
    out = handler.run("سؤال")
    assert len(out["citations"]) == 2


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def test_summary_used_as_answer_text():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
        summary="هذا الحكم القانوني المركّب",
    )
    out = handler.run("سؤال")
    assert out["answer_text"] == "هذا الحكم القانوني المركّب"


def test_summarizer_null_falls_back_to_template():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص المادة")],
        dense_hits=[],
        summary=None,
    )
    out = handler.run("سؤال")
    # Template format from baselines: "وفقًا لـ <title>، المادة <ref>: <text>"
    assert "قانون الأسرة" in out["answer_text"]
    assert "المادة 5" in out["answer_text"]


def test_summarizer_failure_falls_back_to_template():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص المادة")],
        dense_hits=[],
    )
    mocks["summarizer"].side_effect = RuntimeError("LLM down")
    out = handler.run("سؤال")
    assert "المادة 5" in out["answer_text"]


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------


def test_empty_query_abstains_without_calling_llm():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")], dense_hits=[],
    )
    out = handler.run("   ")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    mocks["decomposer"].assert_not_called()
    mocks["verifier"].assert_not_called()
    mocks["summarizer"].assert_not_called()


def test_no_retrieval_hits_abstains_with_no_hits_reason():
    handler, mocks = _make_handler(
        bm25_hits=[], dense_hits=[],
    )
    out = handler.run("سؤال غامض")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"
    # Summariser must not be called when there's nothing to summarise.
    mocks["summarizer"].assert_not_called()


def test_all_candidates_rejected_abstains_with_specific_reason():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "kept")],
        dense_hits=[],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": False, "supporting_span": None,
                                          "contradicting_span": None, "confidence": 0.9},
        },
    )
    out = handler.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"


# ---------------------------------------------------------------------------
# End-to-end compatibility with the eval runner
# ---------------------------------------------------------------------------


def test_answer_to_result_consumes_output_without_branching():
    from akn_rlm.eval.runner import _answer_to_result

    handler, _ = _make_handler(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص المادة")],
        dense_hits=[_dense("84-11_1984-06-09", "5", "نص المادة")],
    )
    answer = handler.run("ما هي المادة 5؟")
    answer["_latency_s"] = 0.01

    question = {
        "id": "fam_test_q01",
        "query": "ما هي المادة 5؟",
        "query_type": "multi_hop",
        "legal_category": "family_law",
        "difficulty": "medium",
        "language": "ar",
        "split": "test",
        "gold_doc_ids": ["84-11_1984-06-09"],
        "gold_article_ids": ["84-11_1984-06-09#art_5"],
        "gold_citations": [{"doc_id": "84-11_1984-06-09", "article_ref": "5"}],
        "gold_abstain": False,
        "gold_answer": "",
        "gold_reasoning_chain": [],
    }
    result = _answer_to_result(question, answer)
    assert result["pred_doc_ids"] == ["84-11_1984-06-09"]
    assert result["pred_article_ids"] == ["84-11_1984-06-09#art_5"]
    assert result["predicted_abstain"] is False
    assert result["hcr"] == 0.0
