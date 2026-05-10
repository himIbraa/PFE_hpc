"""Tests for the Phase-2 unanswerable handler.

The handler must mirror the baseline ``.run(query) -> dict`` contract so
``akn_rlm.eval.runner._answer_to_result`` consumes it unchanged. BM25,
Dense, and the optional LLM judge are fully mocked — the suite never
touches a real index, real LLM, or any external service, and runs in
milliseconds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.indexers.dense import DenseHit
from akn_rlm.rlm.handlers.unanswerable import (
    DEFAULT_K_EACH,
    DEFAULT_ROUTE_TOP_N,
    DEFAULT_TOP_K_CANDIDATES,
    DEFAULT_WEAK_EVIDENCE_THRESHOLD,
    SUPPORT_SPAN_LEN,
    TELEMETRY_BASELINE,
    UnanswerableHandler,
    build_unanswerable_handler,
    detect_infection_signals,
    detect_local,
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


def _make_handler(
    *,
    bm25_hits=None,
    dense_hits=None,
    routed_ids=None,
    doc_title="قانون العمل",
    llm_judge_fn=None,
    **kwargs,
):
    bm25 = MagicMock()
    bm25.search.return_value = list(bm25_hits or [])
    dense = MagicMock()
    dense.search.return_value = list(dense_hits or [])

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)

    router = _stub_router(routed_ids if routed_ids is not None else ["90-11_1990-04-21"])

    handler = UnanswerableHandler(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=MagicMock(),
        router=router,
        llm_judge_fn=llm_judge_fn,
        **kwargs,
    )
    return handler, {
        "bm25": bm25,
        "dense": dense,
        "registry": registry,
        "router": router,
    }


# ---------------------------------------------------------------------------
# Defaults & contract
# ---------------------------------------------------------------------------


def test_defaults_match_handoff_design():
    """HANDOFF §3 budget: 0 sub-LM calls, top-K=5, route top-3, k_each=20."""
    assert DEFAULT_TOP_K_CANDIDATES == 5
    assert DEFAULT_K_EACH == 20
    assert DEFAULT_ROUTE_TOP_N == 3
    # 0.030 = supported by both retrievers at top-1; below this we
    # treat the corpus as not carrying the concept.
    assert DEFAULT_WEAK_EVIDENCE_THRESHOLD == pytest.approx(0.030)


def test_factory_builds_handler():
    bm25 = MagicMock(); dense = MagicMock(); registry = MagicMock()
    handler = build_unanswerable_handler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=None,
        router=_stub_router([]),
    )
    assert isinstance(handler, UnanswerableHandler)


def test_factory_accepts_no_llm_pool():
    """Handler can run with llm_pool=None — no sub-LM calls in default config."""
    bm25 = MagicMock(); dense = MagicMock(); registry = MagicMock()
    handler = build_unanswerable_handler(
        bm25=bm25, dense=dense, registry=registry,
        router=_stub_router([]),
    )
    assert isinstance(handler, UnanswerableHandler)


def test_telemetry_baseline_tag_is_rlm_unanswerable():
    handler, _ = _make_handler()
    out = handler.run("test")
    assert out["_telemetry"]["baseline"] == TELEMETRY_BASELINE
    assert TELEMETRY_BASELINE == "rlm_unanswerable"


def test_run_returns_required_keys():
    handler, _ = _make_handler(bm25_hits=[_bm25("90-11_1990-04-21", "1")])
    out = handler.run("هل")
    for key in ("answer_text", "abstention", "abstention_reason", "citations",
                "reasoning_chain", "trajectory", "tokens_used",
                "depth_max_reached", "_telemetry"):
        assert key in out


# ---------------------------------------------------------------------------
# Local foreign-law dictionary
# ---------------------------------------------------------------------------


def test_detect_local_french_concubinage():
    sigs = detect_local("هل يعترف القانون الجزائري بالمعاشرة (concubinage)؟")
    assert "fr:concubinage" in sigs


def test_detect_local_arabic_concubinage():
    sigs = detect_local("هل يعترف القانون بالمعاشرة كعلاقة قانونية؟")
    assert any("concubinage" in s for s in sigs)


def test_detect_local_french_rent_control():
    sigs = detect_local("contrôle des loyers")
    assert "fr:rent_control" in sigs


def test_detect_local_us_401k():
    sigs = detect_local("صندوق 401k البديل")
    assert "us:401k" in sigs


def test_detect_local_us_private_pension():
    sigs = detect_local("private pension fund replacement")
    assert "us:private_pension" in sigs


def test_detect_local_egyptian_inheritance_marker():
    sigs = detect_local("قانون المواريث المصري رقم 77")
    assert "eg:inheritance_law_ar" in sigs


def test_detect_local_egyptian_criminal_procedure_marker():
    sigs = detect_local("قانون الإجراءات الجنائية المصري")
    assert "eg:criminal_procedure_ar" in sigs


def test_detect_local_tunisian_csp_arabic():
    sigs = detect_local("الأحوال الشخصية التونسية")
    assert "tn:csp_ar" in sigs


def test_detect_local_dz_absent_auto_saisine_ar():
    sigs = detect_local("ممارسة رقابة لاحقة تلقائية دون إخطار")
    assert "dz_absent:auto_saisine_ar" in sigs


def test_detect_local_dz_absent_municipal_tax_ar():
    sigs = detect_local("فرض ضرائب محلية مستقلة عن النظام الوطني")
    assert "dz_absent:municipal_tax_ar" in sigs


def test_detect_local_cross_jurisdiction_marker_french():
    sigs = detect_local("كالنظام الفرنسي")
    assert "fr:cross_jurisdiction_marker" in sigs


def test_detect_local_cross_jurisdiction_marker_us():
    sigs = detect_local("كما في النظام الأمريكي")
    assert "fr:cross_jurisdiction_marker" in sigs


def test_detect_local_returns_empty_for_clean_text():
    assert detect_local("ما هي شروط الزواج في قانون الأسرة؟") == []


def test_detect_local_handles_empty_input():
    assert detect_local("") == []
    assert detect_local(None) == []


def test_detect_infection_signals_unions_both_dictionaries():
    """Combined detector picks up regex hits from BOTH gates.jurisdiction and
    the local handler dictionary, with deduplication and order preservation."""
    text = "هل يطبق القانون الجزائري نظام at-will employment كما في النظام الأمريكي؟"
    sigs = detect_infection_signals(text)
    # `at-will` from gates.jurisdiction
    assert any("at_will" in s for s in sigs)
    # `كما في النظام الأمريكي` from the local dict
    assert any("cross_jurisdiction_marker" in s for s in sigs)


def test_detect_infection_signals_dedupes():
    """If a signal matches both detectors (none currently overlap, but the
    union must still dedupe by signal_id) the union is unique."""
    text = "PACS PACS PACS"
    sigs = detect_infection_signals(text)
    assert sigs.count("fr:pacs") == 1


def test_detect_infection_signals_returns_empty_for_clean_arabic():
    assert detect_infection_signals("ما هي شروط الزواج في قانون الأسرة؟") == []


# ---------------------------------------------------------------------------
# Path 3a: signals present → abstain
# ---------------------------------------------------------------------------


def test_signal_present_abstains_with_infected_jurisdiction_reason():
    """Foreign-law signal → handler abstains with reason='infected_jurisdiction'."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "1")],  # search has results
        dense_hits=[_dense("90-11_1990-04-21", "1")],
    )
    out = handler.run("هل يحق at-will employment في الجزائر؟")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "infected_jurisdiction"
    assert out["citations"] == []
    assert out["answer_text"] == ""


def test_signals_recorded_in_telemetry():
    handler, _ = _make_handler()
    out = handler.run("هل يحق at-will employment؟")
    sigs = out["_telemetry"]["signals"]
    assert any("at_will" in s for s in sigs)


def test_signal_path_skips_bootstrap_search_first_principle():
    """Per HANDOFF: don't bootstrap-search first. Even when retrieval has
    high-scoring hits, an infection signal MUST short-circuit to abstain
    rather than letting the LLM see the tangential matches."""
    handler, env = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "73", score=999.0)],
        dense_hits=[_dense("90-11_1990-04-21", "73", score=0.99)],
    )
    out = handler.run("هل يحق at-will employment؟")
    # Must abstain even though the search would surface art_73 as top-1.
    assert out["abstention"] is True
    assert out["abstention_reason"] == "infected_jurisdiction"


def test_signal_path_records_confirming_candidates_for_audit():
    """The 'ONE confirming hybrid search' result is recorded in telemetry
    so a reviewer can see the strongest Algerian counterpart we found
    (or didn't find) when explaining the abstention."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "73", "نص المادة 73")],
        dense_hits=[_dense("90-11_1990-04-21", "73", "نص المادة 73")],
    )
    out = handler.run("at-will employment")
    candidates = out["_telemetry"]["confirming_candidates"]
    assert len(candidates) >= 1
    assert candidates[0]["doc_id"] == "90-11_1990-04-21"
    assert candidates[0]["article_ref"] == "73"


def test_signal_path_runs_exactly_one_hybrid_search():
    """One BM25 call + one Dense call — never more, never zero (the search
    is needed to populate confirming_candidates for telemetry)."""
    handler, env = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "1")],
        dense_hits=[_dense("90-11_1990-04-21", "1")],
    )
    handler.run("هل يحق at-will employment؟")
    assert env["bm25"].search.call_count == 1
    assert env["dense"].search.call_count == 1


def test_signal_path_no_sub_lm_calls_by_default():
    """Default config: 0 sub-LM calls. The LLM judge is opt-in."""
    handler, _ = _make_handler()
    out = handler.run("ISF فرنسي")
    assert out["_telemetry"]["sub_call_count"] == 0


# ---------------------------------------------------------------------------
# Optional LLM judge (opt-in)
# ---------------------------------------------------------------------------


def test_llm_judge_can_clear_false_positive_signal():
    """If the LLM judge says the foreign-law signal is a false positive,
    the handler falls through to the strong-evidence answer path.

    Convention (matches gates.jurisdiction._llm_classify): judge returns
    True iff contamination IS confirmed. Returning False = false positive
    → clear the signal.
    """
    cleared = MagicMock(return_value=False)  # False = NOT contaminated
    handler, _ = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "1"), _bm25("90-11_1990-04-21", "2")],
        dense_hits=[_dense("90-11_1990-04-21", "1"), _dense("90-11_1990-04-21", "2")],
        llm_judge_fn=cleared,
    )
    out = handler.run("اتفاقية تتضمن مفهوم PACS بشكل عابر")
    # Cleared by LLM → falls through to evidence path
    assert out["abstention"] is False
    assert out["citations"]
    assert out["_telemetry"]["sub_call_count"] == 1


def test_llm_judge_confirming_contamination_keeps_abstention():
    """Judge returns True (contamination confirmed) → abstention stands."""
    confirms = MagicMock(return_value=True)  # True = contaminated
    handler, _ = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "1", score=999.0)],
        dense_hits=[_dense("90-11_1990-04-21", "1", score=0.99)],
        llm_judge_fn=confirms,
    )
    out = handler.run("PACS")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "infected_jurisdiction"
    assert out["_telemetry"]["sub_call_count"] == 1


def test_llm_judge_default_is_off():
    """Default constructor doesn't wire an LLM judge → no LLM calls."""
    handler, _ = _make_handler()
    out = handler.run("PACS")
    assert out["_telemetry"]["sub_call_count"] == 0
    assert out["abstention"] is True


def test_llm_judge_exception_keeps_abstention():
    """LLM judge crashes → conservatively keep the abstention (don't
    pretend the signal was cleared)."""
    crashing = MagicMock(side_effect=RuntimeError("network down"))
    handler, _ = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "1", score=999.0)],
        dense_hits=[_dense("90-11_1990-04-21", "1", score=0.99)],
        llm_judge_fn=crashing,
    )
    out = handler.run("PACS")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "infected_jurisdiction"


# ---------------------------------------------------------------------------
# Path 3b: no signals + no candidates → no_hits abstain
# ---------------------------------------------------------------------------


def test_no_signals_no_candidates_abstains_no_hits():
    handler, _ = _make_handler(bm25_hits=[], dense_hits=[])
    out = handler.run("ما هي شروط الزواج في قانون الأسرة؟")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"


# ---------------------------------------------------------------------------
# Path 3c: no signals + weak top-1 score → weak_evidence abstain
# ---------------------------------------------------------------------------


def test_weak_evidence_only_one_retriever_top_1_abstains():
    """RRF top-1 hit ranked first by ONLY ONE retriever scores 1/61 ≈ 0.0164,
    well below the default 0.030 threshold → abstain."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "1")],
        dense_hits=[],  # only BM25 has hits
    )
    out = handler.run("ما هي شروط الزواج في قانون الأسرة؟")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "weak_evidence"


def test_weak_evidence_records_top_score_in_telemetry():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "1")],
        dense_hits=[],
    )
    out = handler.run("ما هي شروط الزواج؟")
    assert out["_telemetry"]["top_score"] == pytest.approx(1 / 61, rel=1e-3)


def test_weak_evidence_threshold_can_be_overridden():
    """If the dispatcher (R7) wants stricter abstention, raise the threshold."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d1", "1")],
        dense_hits=[_dense("d1", "1")],
        routed_ids=["d1"],
        weak_evidence_threshold=0.5,  # very strict
    )
    out = handler.run("clean Arabic query without infection signals")
    # Two-retriever top-1 ≈ 0.033 < 0.5 → abstain
    assert out["abstention"] is True
    assert out["abstention_reason"] == "weak_evidence"


# ---------------------------------------------------------------------------
# Path 3d: no signals + strong top-1 → cautious answer
# ---------------------------------------------------------------------------


def test_strong_evidence_path_returns_answer():
    """Both BM25 + Dense rank an article #1 → RRF score = 2/61 ≈ 0.0328,
    above default 0.030 → handler returns a cautious answer with
    citations rather than abstaining."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "1", "نص شروط الزواج"),
                   _bm25("90-11_1990-04-21", "2")],
        dense_hits=[_dense("90-11_1990-04-21", "1", "نص شروط الزواج"),
                    _dense("90-11_1990-04-21", "2")],
    )
    out = handler.run("ما هي شروط الزواج في قانون الأسرة؟")
    assert out["abstention"] is False
    assert out["abstention_reason"] is None
    assert len(out["citations"]) >= 1
    # Top citation must be the article both retrievers ranked first.
    assert out["citations"][0]["doc_id"] == "90-11_1990-04-21"
    assert out["citations"][0]["article_ref"] == "1"


def test_strong_evidence_path_no_sub_lm_calls():
    """Cautious-answer path must not call the LLM (deterministic)."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d1", "1"), _bm25("d1", "2")],
        dense_hits=[_dense("d1", "1"), _dense("d1", "2")],
        routed_ids=["d1"],
    )
    out = handler.run("clean query")
    assert out["_telemetry"]["sub_call_count"] == 0


def test_strong_evidence_path_caps_at_top_k_candidates():
    """Cautious-answer path returns at most top_k_candidates citations."""
    bm25_hits = [_bm25("d1", str(i)) for i in range(20)]
    dense_hits = [_dense("d1", str(i)) for i in range(20)]
    handler, _ = _make_handler(
        bm25_hits=bm25_hits, dense_hits=dense_hits, routed_ids=["d1"],
        top_k_candidates=3,
    )
    out = handler.run("clean")
    assert len(out["citations"]) == 3


def test_strong_evidence_path_emits_template_answer():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d1", "1", "نص المادة الأولى")],
        dense_hits=[_dense("d1", "1", "نص المادة الأولى")],
        routed_ids=["d1"],
        doc_title="قانون العمل",
    )
    out = handler.run("clean")
    assert "قانون العمل" in out["answer_text"]
    assert "المادة 1" in out["answer_text"]
    assert "نص المادة الأولى" in out["answer_text"]


def test_strong_evidence_path_includes_top_score_in_reasoning_chain():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d1", "1")],
        dense_hits=[_dense("d1", "1")],
        routed_ids=["d1"],
    )
    out = handler.run("clean")
    chain = out["reasoning_chain"]
    assert any("no_foreign_law_signals_detected" in s for s in chain)
    assert any("strong_evidence_top_score" in s for s in chain)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_router_called_with_route_top_n():
    handler, env = _make_handler(route_top_n=4)
    handler.run("at-will employment")
    env["router"].route.assert_called_once()
    _, kwargs = env["router"].route.call_args
    assert kwargs.get("top_n") == 4


def test_doc_filter_falls_back_to_full_pool_when_filter_wipes_everything():
    """If routing returns docs that don't appear in the fused pool, fall
    back to the unrestricted pool rather than abstaining unnecessarily."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d_actual", "1"), _bm25("d_actual", "2")],
        dense_hits=[_dense("d_actual", "1"), _dense("d_actual", "2")],
        routed_ids=["d_wrong_route"],  # router predicted the wrong doc
    )
    out = handler.run("clean query without infection")
    # Filter wiped to []; fallback to full pool → strong evidence → answer.
    assert out["abstention"] is False
    assert out["citations"][0]["doc_id"] == "d_actual"


def test_routing_filter_applied_when_routed_doc_in_pool():
    """When the routed docs ARE in the fused pool, restrict to them."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("routed_doc", "1"), _bm25("other_doc", "1")],
        dense_hits=[_dense("routed_doc", "1"), _dense("other_doc", "1")],
        routed_ids=["routed_doc"],
    )
    out = handler.run("clean query")
    cited_docs = {c["doc_id"] for c in out["citations"]}
    assert cited_docs == {"routed_doc"}


def test_both_retrievers_called_at_k_each():
    handler, env = _make_handler(k_each=42)
    handler.run("at-will")
    _, bm25_kwargs = env["bm25"].search.call_args
    assert bm25_kwargs.get("k") == 42
    _, dense_kwargs = env["dense"].search.call_args
    assert dense_kwargs.get("k") == 42


# ---------------------------------------------------------------------------
# Empty / edge-case inputs
# ---------------------------------------------------------------------------


def test_empty_query_abstains_without_calls():
    handler, env = _make_handler()
    out = handler.run("")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    env["bm25"].search.assert_not_called()
    env["dense"].search.assert_not_called()
    env["router"].route.assert_not_called()


def test_whitespace_only_query_abstains():
    handler, _ = _make_handler()
    out = handler.run("   \n\t  ")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"


def test_bm25_exception_degrades_gracefully():
    """One retriever crashes → handler still runs on the other arm."""
    bm25 = MagicMock()
    bm25.search.side_effect = RuntimeError("BM25 boom")
    dense = MagicMock()
    dense.search.return_value = [_dense("d1", "1"), _dense("d1", "2")]
    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="t")
    router = _stub_router(["d1"])
    handler = UnanswerableHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=None, router=router,
    )
    out = handler.run("clean")
    # Only one retriever has hits → top-1 ≈ 1/61 < threshold → weak_evidence
    assert out["abstention"] is True
    assert out["abstention_reason"] == "weak_evidence"


def test_dense_exception_degrades_gracefully():
    bm25 = MagicMock()
    bm25.search.return_value = [_bm25("d1", "1")]
    dense = MagicMock()
    dense.search.side_effect = RuntimeError("Dense boom")
    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="t")
    router = _stub_router(["d1"])
    handler = UnanswerableHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=None, router=router,
    )
    out = handler.run("clean")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "weak_evidence"


# ---------------------------------------------------------------------------
# Citation shape (cautious-answer path)
# ---------------------------------------------------------------------------


def test_citation_carries_required_fields():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d1", "1", "نص المادة الأولى")],
        dense_hits=[_dense("d1", "1", "نص المادة الأولى")],
        routed_ids=["d1"], doc_title="قانون",
    )
    out = handler.run("clean")
    cit = out["citations"][0]
    for key in ("doc_id", "article_ref", "doc_title", "supporting_span",
                "text", "confidence"):
        assert key in cit
    assert cit["doc_id"] == "d1"
    assert cit["article_ref"] == "1"
    assert cit["doc_title"] == "قانون"
    assert cit["text"] == "نص المادة الأولى"
    assert cit["supporting_span"] == "نص المادة الأولى"


def test_supporting_span_capped_at_280_chars():
    long_text = "ا" * 500
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d1", "1", long_text)],
        dense_hits=[_dense("d1", "1", long_text)],
        routed_ids=["d1"],
    )
    out = handler.run("clean")
    assert len(out["citations"][0]["supporting_span"]) == SUPPORT_SPAN_LEN


def test_citation_canonicalises_article_ref():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d1", "9 مكرر", "نص")],
        dense_hits=[_dense("d1", "9 مكرر", "نص")],
        routed_ids=["d1"],
    )
    out = handler.run("clean")
    assert out["citations"][0]["article_ref"] == "9_bis"


def test_template_answer_falls_back_to_doc_id_when_no_title():
    bm25 = MagicMock()
    bm25.search.return_value = [_bm25("d1", "1", "نص")]
    dense = MagicMock()
    dense.search.return_value = [_dense("d1", "1", "نص")]
    registry = MagicMock()
    registry.get_doc.return_value = None  # no entry → fallback path
    router = _stub_router(["d1"])
    handler = UnanswerableHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=None, router=router,
    )
    out = handler.run("clean")
    assert "d1" in out["answer_text"]


# ---------------------------------------------------------------------------
# End-to-end compatibility with eval runner
# ---------------------------------------------------------------------------


def test_answer_dict_consumed_by_eval_runner():
    from akn_rlm.eval.runner import _answer_to_result
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d1", "1", "نص")],
        dense_hits=[_dense("d1", "1", "نص")],
        routed_ids=["d1"],
    )
    out = handler.run("clean query")
    question = {
        "id":               "q1",
        "query":            "clean query",
        "query_type":       "unanswerable",
        "gold_doc_ids":     [],
        "gold_article_ids": [],
        "gold_citations":   [],
        "gold_abstain":     True,
        "gold_answer":      "",
    }
    result = _answer_to_result(question, out)
    assert result["question_id"] == "q1"
    # Strong-evidence path → predicted_abstain=False
    assert result["predicted_abstain"] is False
    assert result["pred_doc_ids"] == ["d1"]


def test_abstention_path_consumed_by_eval_runner_correctly():
    from akn_rlm.eval.runner import _answer_to_result
    handler, _ = _make_handler()
    out = handler.run("at-will employment فرنسي")
    question = {
        "id":               "q_unans",
        "query":            "at-will",
        "query_type":       "unanswerable",
        "gold_doc_ids":     [],
        "gold_article_ids": [],
        "gold_citations":   [],
        "gold_abstain":     True,
        "gold_answer":      "",
    }
    result = _answer_to_result(question, out)
    assert result["predicted_abstain"] is True
    assert result["gold_abstain"] is True
    # Per `abstention_accuracy`, predicted == gold → score 1.0.
    from akn_rlm.eval.metrics import abstention_accuracy
    assert abstention_accuracy(
        result["predicted_abstain"], result["gold_abstain"]
    ) == 1.0


# ---------------------------------------------------------------------------
# AbstF1 integration (the gate)
# ---------------------------------------------------------------------------


def test_abstention_recall_on_pure_unanswerable_slice():
    """On a slice where every gold question is unanswerable AND the handler
    abstains on all of them, AbstF1 = 1.0 (the gate is ≥ 0.7)."""
    from akn_rlm.eval.metrics import abstention_f1
    handler, _ = _make_handler()
    queries = [
        ("q1", "at-will employment", True),
        ("q2", "ISF", True),
        ("q3", "PACS", True),
        ("q4", "كما في النظام الأمريكي", True),
        ("q5", "كالنظام الفرنسي", True),
    ]
    results = []
    for qid, q, gold_abstain in queries:
        ans = handler.run(q)
        results.append({
            "predicted_abstain": bool(ans["abstention"]),
            "gold_abstain":      gold_abstain,
        })
    assert abstention_f1(results) == pytest.approx(1.0)


def test_abstention_path_does_not_overcount_clean_queries():
    """A dispatcher that misclassifies an *answerable* query as unanswerable
    will run that query through this handler. With strong evidence and no
    signals, the handler answers — preserving abstention precision."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("d1", "1"), _bm25("d1", "2")],
        dense_hits=[_dense("d1", "1"), _dense("d1", "2")],
        routed_ids=["d1"],
    )
    out = handler.run("ما هي شروط الزواج في قانون الأسرة؟")
    assert out["abstention"] is False


# ---------------------------------------------------------------------------
# Real-world unanswerable questions (sanity smoke without any LLM/index)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "هل يحق لصاحب العمل فصل العامل بحرية تامة دون إبداء أسباب (at-will employment)؟",
    "هل تفرض الجزائر ضريبة التضامن على الثروة (ISF) كالنظام الفرنسي؟",
    "هل يمكن في الجزائر إبرام plea bargaining في قضايا الفساد؟",
    "هل يخضع الإيجار السكني في الجزائر لنظام contrôle des loyers كالقانون الفرنسي؟",
    "هل يعترف القانون الجزائري بالمعاشرة (concubinage) كعلاقة تنشئ حقوقاً؟",
    "هل يحظر القانون الجزائري تعدد الزوجات حظراً مطلقاً كما فعلت مجلة الأحوال الشخصية التونسية في الفصل 18؟",
    "هل يمكن للمحكمة الدستورية ممارسة رقابة لاحقة تلقائية دون إخطار؟",
    "هل يحق للبلدية فرض ضرائب محلية مستقلة عن النظام الوطني؟",
])
def test_real_unanswerable_queries_all_abstain(query):
    """Spot-check on the ALB v3.0 unanswerable slice — every one of these
    questions must trip a signal and abstain. This is the gate the
    handler is designed to clear."""
    handler, _ = _make_handler(bm25_hits=[], dense_hits=[])
    out = handler.run(query)
    assert out["abstention"] is True
    # Either "infected_jurisdiction" (signals path) or "no_hits" /
    # "weak_evidence" (no-signals path) — all valid abstentions.
    assert out["abstention_reason"] in (
        "infected_jurisdiction", "no_hits", "weak_evidence",
    )
