"""Tests for the Phase-2 long_context handler.

The handler must mirror the baseline ``.run(query) -> dict`` contract so
``akn_rlm.eval.runner._answer_to_result`` consumes it unchanged. BM25,
Dense, and the summariser are fully mocked — the suite never touches a
real index or LLM and runs in milliseconds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akn_rlm.eval.runner import _answer_to_result
from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.indexers.dense import DenseHit
from akn_rlm.rlm.handlers.long_context import (
    DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH,
    DEFAULT_ROUTE_TOP_N,
    SUPPORT_SPAN_LEN,
    TELEMETRY_BASELINE,
    LongContextHandler,
    build_long_context_handler,
)
from akn_rlm.rlm.routing import RouteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm25(doc_id: str, ref: str, text: str = "نص", score: float = 5.0) -> BM25Hit:
    return BM25Hit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _dense(doc_id: str, ref: str, text: str = "نص", score: float = 0.5) -> DenseHit:
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


def _stub_summarizer(summary: str | None = "ملخّص شامل"):
    def _fn(_pool, _q, _articles, _model):
        return {"summary": summary, "key_articles": [], "caveats": None}
    return MagicMock(side_effect=_fn)


def _make_handler(
    *,
    bm25_hits: list[BM25Hit] | None = None,
    dense_hits: list[DenseHit] | None = None,
    routed_ids: list[str] | None = None,
    summary: str | None = "ملخّص شامل",
    doc_title: str = "القانون المدني",
    **kwargs,
):
    bm25 = MagicMock()
    bm25.search.return_value = list(bm25_hits or [])
    dense = MagicMock()
    dense.search.return_value = list(dense_hits or [])

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)

    router = _stub_router(routed_ids if routed_ids is not None else ["75-58_1975-09-26"])
    summarizer = _stub_summarizer(summary)

    handler = LongContextHandler(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=MagicMock(),
        router=router,
        summarizer_fn=summarizer,
        **kwargs,
    )
    return handler, dict(
        bm25=bm25, dense=dense, registry=registry, router=router,
        summarizer=summarizer,
    )


# ---------------------------------------------------------------------------
# Defaults & contract
# ---------------------------------------------------------------------------


def test_defaults_match_handoff_design():
    """HANDOFF §3 says long_context uses broad k=20.
    R9.4 tightened final_top_k 10 → 6 to lift Cite F1.
    """
    assert DEFAULT_K_EACH == 20
    assert DEFAULT_FINAL_TOP_K == 6
    assert DEFAULT_ROUTE_TOP_N == 3


def test_r9_4_default_final_top_k_locked_at_6():
    """Lock the R9.4 retune so a future drift back to 10 fails loudly."""
    from akn_rlm.rlm.handlers import long_context as lc_mod
    assert lc_mod.DEFAULT_FINAL_TOP_K == 6


def test_factory_builds_handler():
    bm25 = MagicMock(); dense = MagicMock(); registry = MagicMock()
    h = build_long_context_handler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MagicMock(),
        router=_stub_router(["d"]),
    )
    assert isinstance(h, LongContextHandler)


def test_telemetry_baseline_tag_is_rlm_long_context():
    handler, _ = _make_handler(bm25_hits=[_bm25("75-58_1975-09-26", "5")])
    out = handler.run("اشرح جميع أحكام عقد البيع")
    assert out["_telemetry"]["baseline"] == TELEMETRY_BASELINE
    assert TELEMETRY_BASELINE == "rlm_long_context"


def test_run_returns_required_keys():
    handler, _ = _make_handler(bm25_hits=[_bm25("75-58_1975-09-26", "5")])
    out = handler.run("سؤال طويل")
    for key in ("answer_text", "abstention", "abstention_reason", "citations",
                "reasoning_chain", "trajectory", "tokens_used",
                "depth_max_reached", "_telemetry"):
        assert key in out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_router_called_with_route_top_n():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5")],
        route_top_n=2,
    )
    handler.run("سؤال")
    mocks["router"].route.assert_called_once()
    assert mocks["router"].route.call_args.kwargs.get("top_n") == 2


def test_routed_doc_ids_recorded_in_telemetry():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5")],
        routed_ids=["75-58_1975-09-26", "84-11_1984-06-09"],
    )
    out = handler.run("سؤال")
    assert out["_telemetry"]["routed_doc_ids"] == [
        "75-58_1975-09-26", "84-11_1984-06-09",
    ]


# ---------------------------------------------------------------------------
# Retrieval — broad hybrid k=20
# ---------------------------------------------------------------------------


def test_both_retrievers_called_at_k_each():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5")],
        dense_hits=[_dense("75-58_1975-09-26", "5")],
        k_each=20,
    )
    handler.run("سؤال")
    mocks["bm25"].search.assert_called_once()
    mocks["dense"].search.assert_called_once()
    assert mocks["bm25"].search.call_args.kwargs.get("k") == 20
    assert mocks["dense"].search.call_args.kwargs.get("k") == 20


def test_k_each_passthrough():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5")],
        k_each=42,
    )
    handler.run("سؤال")
    assert mocks["bm25"].search.call_args.kwargs.get("k") == 42
    assert mocks["dense"].search.call_args.kwargs.get("k") == 42


def test_filter_by_routed_doc_ids():
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25("75-58_1975-09-26", "5", "kept", score=10.0),
            _bm25("66-156_1966-06-08", "9", "dropped", score=20.0),
        ],
        routed_ids=["75-58_1975-09-26"],
    )
    out = handler.run("سؤال")
    assert all(c["doc_id"] == "75-58_1975-09-26" for c in out["citations"])


def test_full_pool_fallback_when_filter_wipes():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("66-156_1966-06-08", "9", "only-doc", score=5.0)],
        routed_ids=["75-58_1975-09-26"],
    )
    out = handler.run("سؤال")
    assert out["abstention"] is False
    assert any(c["doc_id"] == "66-156_1966-06-08" for c in out["citations"])


# ---------------------------------------------------------------------------
# Top-K + dedup
# ---------------------------------------------------------------------------


def test_final_top_k_truncates_pool():
    bm25_hits = [
        _bm25("75-58_1975-09-26", str(i), text=f"t{i}", score=20.0 - i)
        for i in range(1, 16)
    ]
    handler, _ = _make_handler(bm25_hits=bm25_hits, final_top_k=5)
    out = handler.run("سؤال")
    assert len(out["citations"]) == 5


def test_dedup_collapses_duplicate_articles():
    """Same (doc_id, ref) showing up twice → single citation."""
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25("75-58_1975-09-26", "5", "A", 10.0),
            _bm25("75-58_1975-09-26", "5", "B", 8.0),
        ],
    )
    out = handler.run("سؤال")
    keys = [(c["doc_id"], c["article_ref"]) for c in out["citations"]]
    assert keys.count(("75-58_1975-09-26", "5")) == 1


def test_canonicalisation_collapses_arabic_bis():
    """`9 مكرر` (BM25) and `9_bis` (Dense) collapse to a single citation."""
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "9 مكرر", "A", 10.0)],
        dense_hits=[_dense("75-58_1975-09-26", "9_bis", "A", 0.6)],
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"9_bis"}


def test_canonicalisation_arabic_ordinal_first():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "الأولى", "A", 10.0)],
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"1"}


# ---------------------------------------------------------------------------
# Real summariser sub-LM call (the discriminator)
# ---------------------------------------------------------------------------


def test_summariser_called_exactly_once_per_query():
    """HANDOFF §3 says long_context budget = 1 summariser call."""
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", str(i), f"t{i}", score=20.0 - i)
                   for i in range(1, 6)],
    )
    handler.run("سؤال")
    assert mocks["summarizer"].call_count == 1


def test_summariser_receives_query_and_top_k_articles():
    bm25_hits = [
        _bm25("75-58_1975-09-26", "1", "t1", 12.0),
        _bm25("75-58_1975-09-26", "2", "t2", 10.0),
        _bm25("75-58_1975-09-26", "3", "t3", 8.0),
    ]
    handler, mocks = _make_handler(bm25_hits=bm25_hits)
    handler.run("سؤال محدد")
    args = mocks["summarizer"].call_args.args
    # signature: (llm_pool, query, articles, model)
    assert args[1] == "سؤال محدد"
    refs = [a["article_ref"] for a in args[2]]
    assert refs == ["1", "2", "3"]


def test_summary_used_as_answer_text():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5")],
        summary="هذا ملخّص متعدد المواد",
    )
    out = handler.run("سؤال")
    assert out["answer_text"] == "هذا ملخّص متعدد المواد"


def test_null_summary_falls_back_to_template():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5", "نص محدد", 10.0)],
        summary=None,
    )
    out = handler.run("سؤال")
    assert "وفقًا لـ" in out["answer_text"]
    assert "المادة 5" in out["answer_text"]


def test_summariser_exception_falls_back_to_template():
    def raising(*_a, **_k):
        raise RuntimeError("boom")
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5", "نص", 10.0)],
    )
    handler._summarizer_fn = raising
    out = handler.run("سؤال")
    assert "وفقًا لـ" in out["answer_text"]


def test_summariser_passes_sub_model_arg():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5")],
        sub_model="custom-sub",
    )
    handler.run("سؤال")
    args = mocks["summarizer"].call_args.args
    assert args[3] == "custom-sub"


# ---------------------------------------------------------------------------
# Citation shape
# ---------------------------------------------------------------------------


def test_citation_shape_carries_required_keys():
    text = "نص قانوني" * 50
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5", text, score=12.0)],
        doc_title="القانون المدني",
    )
    out = handler.run("سؤال")
    cit = out["citations"][0]
    for key in ("doc_id", "article_ref", "doc_title", "supporting_span",
                "text", "confidence"):
        assert key in cit
    assert cit["doc_title"] == "القانون المدني"
    assert cit["text"] == text
    assert len(cit["supporting_span"]) <= SUPPORT_SPAN_LEN


def test_supporting_span_truncates_at_280():
    text = "x" * 1000
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5", text, 10.0)],
    )
    out = handler.run("سؤال")
    assert len(out["citations"][0]["supporting_span"]) == SUPPORT_SPAN_LEN


def test_template_falls_back_to_doc_id_when_title_empty():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5", "نص", 10.0)],
        summary=None,
        doc_title="",
    )
    out = handler.run("سؤال")
    assert "75-58_1975-09-26" in out["answer_text"]


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


def test_empty_query_abstains_without_calling_anything():
    handler, mocks = _make_handler()
    out = handler.run("")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    mocks["bm25"].search.assert_not_called()
    mocks["dense"].search.assert_not_called()
    mocks["summarizer"].assert_not_called()


def test_no_hits_abstains():
    handler, mocks = _make_handler(bm25_hits=[], dense_hits=[])
    out = handler.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"
    mocks["summarizer"].assert_not_called()


# ---------------------------------------------------------------------------
# Retriever-failure tolerance
# ---------------------------------------------------------------------------


def test_bm25_exception_degrades_gracefully():
    bm25 = MagicMock()
    bm25.search.side_effect = RuntimeError("bm25 boom")
    dense = MagicMock()
    dense.search.return_value = [_dense("75-58_1975-09-26", "5", "نص", 0.6)]
    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="x")
    h = LongContextHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MagicMock(),
        router=_stub_router(["75-58_1975-09-26"]),
        summarizer_fn=_stub_summarizer(),
    )
    out = h.run("سؤال")
    assert out["abstention"] is False


def test_dense_exception_degrades_gracefully():
    bm25 = MagicMock()
    bm25.search.return_value = [_bm25("75-58_1975-09-26", "5", "نص", 5.0)]
    dense = MagicMock()
    dense.search.side_effect = RuntimeError("dense boom")
    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="x")
    h = LongContextHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MagicMock(),
        router=_stub_router(["75-58_1975-09-26"]),
        summarizer_fn=_stub_summarizer(),
    )
    out = h.run("سؤال")
    assert out["abstention"] is False


# ---------------------------------------------------------------------------
# Sub-LM call accounting
# ---------------------------------------------------------------------------


def test_sub_call_count_is_exactly_one_on_happy_path():
    handler, _ = _make_handler(bm25_hits=[_bm25("75-58_1975-09-26", "5")])
    out = handler.run("سؤال")
    assert out["_telemetry"]["sub_call_count"] == 1


def test_sub_call_count_is_zero_on_abstention():
    handler, _ = _make_handler(bm25_hits=[])
    out = handler.run("سؤال")
    assert out["_telemetry"]["sub_call_count"] == 0


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_answer_to_result_compatibility():
    handler, _ = _make_handler(
        bm25_hits=[_bm25("75-58_1975-09-26", "5", "نص", 10.0)],
    )
    out = handler.run("اشرح جميع أحكام عقد البيع")
    out["_latency_s"] = 0.123
    question = {
        "id": "civ_lc_q01",
        "query": "اشرح جميع أحكام عقد البيع",
        "query_type": "long_context",
        "legal_category": "civil_law",
        "difficulty": "hard",
        "language": "ar",
        "split": "test",
        "gold_doc_ids": ["75-58_1975-09-26"],
        "gold_article_ids": ["75-58_1975-09-26#art_5"],
        "gold_citations": [{"doc_id": "75-58_1975-09-26", "article_ref": "5"}],
        "gold_abstain": False,
        "gold_answer": "...",
        "gold_reasoning_chain": [],
    }
    result = _answer_to_result(question, out)
    assert result["pred_doc_ids"] == ["75-58_1975-09-26"]
    assert result["pred_article_ids"] == ["75-58_1975-09-26#art_5"]
    assert result["predicted_abstain"] is False


def test_answer_to_result_compatibility_on_abstention():
    handler, _ = _make_handler(bm25_hits=[], dense_hits=[])
    out = handler.run("سؤال")
    out["_latency_s"] = 0.05
    question = {
        "id": "q",
        "query": "سؤال",
        "query_type": "long_context",
        "gold_doc_ids": ["x"],
        "gold_article_ids": ["x#art_1"],
        "gold_citations": [{"doc_id": "x", "article_ref": "1"}],
        "gold_abstain": False,
    }
    result = _answer_to_result(question, out)
    assert result["predicted_abstain"] is True
    assert result["pred_doc_ids"] == []
