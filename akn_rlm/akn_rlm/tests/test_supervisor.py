"""Tests for the Phase-2 / R9.5 per-citation supervisor.

The supervisor is the strong-model re-ranker (default ``gpt-oss-120b``)
that fires when the verifier (small Qwen) lands in the uncertainty
band ``[0.30, 0.70]``. The unit tests below cover:

  * basic logic (drop below threshold, re-rank survivors),
  * cache (hit returns cached, key changes when input changes),
  * fail-open semantics (LLM exception, JSON parse failure, missing
    score, out-of-range score),
  * smart-trigger (``should_supervise``) under all three guard
    conditions,
  * dispatcher / handler integration via ``supervisor_fn`` injection.

Every test mocks the LLM pool — the suite never touches a real LLM and
runs in milliseconds.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from akn_rlm.rlm import supervisor as sup_mod
from akn_rlm.rlm.supervisor import (
    DEFAULT_MIN_CITATIONS,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    DEFAULT_TRIGGER_HIGH,
    DEFAULT_TRIGGER_LOW,
    clear_cache,
    should_supervise,
    supervise_citations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cit(doc_id: str, ref: str, *, conf: float = 0.5, text: str = "نص المادة") -> dict:
    return {
        "doc_id":            doc_id,
        "article_ref":       ref,
        "doc_title":         doc_id,
        "supporting_span":   text[:280],
        "text":              text,
        "confidence":        float(conf),
        "verifier_relevant": True,
    }


def _llm_pool_returning(payload) -> MagicMock:
    """Build an LLM pool whose ``call`` returns the given JSON string
    (or raises if ``payload`` is an Exception)."""
    pool = MagicMock()
    if isinstance(payload, Exception):
        pool.call.side_effect = payload
    else:
        pool.call.return_value = payload
    return pool


@pytest.fixture(autouse=True)
def _reset_supervisor_cache():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# 1. Basic logic
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    """No citations → no supervisor call, return [] unchanged."""
    pool = MagicMock()
    out = supervise_citations(pool, "Q", [])
    assert out == []
    pool.call.assert_not_called()


def test_drops_citations_below_threshold():
    """Survivors are exactly those with supervisor score >= threshold."""
    cits = [
        _cit("d1", "1", conf=0.4),
        _cit("d2", "2", conf=0.5),
        _cit("d3", "3", conf=0.6),
    ]
    payload = json.dumps({"scores": {"0": 0.85, "1": 0.10, "2": 0.55}})
    pool = _llm_pool_returning(payload)

    out = supervise_citations(pool, "Q", cits, threshold=0.3)

    refs = [c["article_ref"] for c in out]
    assert "1" in refs and "3" in refs and "2" not in refs


def test_reranks_survivors_by_supervisor_score_desc():
    """Output order is supervisor score desc, NOT input order."""
    cits = [
        _cit("d1", "1", conf=0.5),
        _cit("d2", "2", conf=0.5),
    ]
    # Input order: 0,1. Supervisor scores: 0=0.40, 1=0.95 → output 1, 0.
    payload = json.dumps({"scores": {"0": 0.40, "1": 0.95}})
    pool = _llm_pool_returning(payload)

    out = supervise_citations(pool, "Q", cits, threshold=0.3)
    assert [c["article_ref"] for c in out] == ["2", "1"]
    assert out[0]["supervisor_score"] == pytest.approx(0.95)
    assert out[1]["supervisor_score"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# 2. Cache
# ---------------------------------------------------------------------------


def test_cache_hit_skips_llm_call():
    cits = [_cit("d1", "1"), _cit("d2", "2")]
    payload = json.dumps({"scores": {"0": 0.8, "1": 0.6}})
    pool = _llm_pool_returning(payload)

    first = supervise_citations(pool, "Q", cits)
    assert pool.call.call_count == 1
    second = supervise_citations(pool, "Q", cits)
    # Same call count → cache served the second invocation.
    assert pool.call.call_count == 1
    assert [c["article_ref"] for c in first] == [c["article_ref"] for c in second]


def test_cache_key_changes_when_query_or_citations_change():
    cits1 = [_cit("d1", "1"), _cit("d2", "2")]
    cits2 = [_cit("d1", "1"), _cit("d3", "3")]
    payload = json.dumps({"scores": {"0": 0.7, "1": 0.7}})
    pool = _llm_pool_returning(payload)

    supervise_citations(pool, "Q", cits1)
    supervise_citations(pool, "Q-other", cits1)  # different query
    supervise_citations(pool, "Q", cits2)        # different citations
    assert pool.call.call_count == 3


# ---------------------------------------------------------------------------
# 3. Fail-open semantics
# ---------------------------------------------------------------------------


def test_fail_open_on_llm_exception():
    cits = [_cit("d1", "1"), _cit("d2", "2")]
    pool = _llm_pool_returning(RuntimeError("network down"))

    out = supervise_citations(pool, "Q", cits)

    # Citations returned unchanged (no supervisor_score, no re-order).
    assert out == cits


def test_fail_open_on_json_parse_failure():
    cits = [_cit("d1", "1"), _cit("d2", "2")]
    pool = _llm_pool_returning("not valid json {")

    out = supervise_citations(pool, "Q", cits)
    assert out == cits


def test_fail_open_on_missing_score_for_a_citation():
    """If the LLM doesn't score every citation, return input unchanged."""
    cits = [_cit("d1", "1"), _cit("d2", "2"), _cit("d3", "3")]
    payload = json.dumps({"scores": {"0": 0.8, "1": 0.5}})  # idx 2 missing
    pool = _llm_pool_returning(payload)

    out = supervise_citations(pool, "Q", cits)
    assert out == cits


def test_fail_open_on_out_of_range_score():
    cits = [_cit("d1", "1"), _cit("d2", "2")]
    payload = json.dumps({"scores": {"0": 1.7, "1": 0.5}})  # 1.7 > 1.0
    pool = _llm_pool_returning(payload)

    out = supervise_citations(pool, "Q", cits)
    assert out == cits


# ---------------------------------------------------------------------------
# 4. Smart-trigger (should_supervise)
# ---------------------------------------------------------------------------


def test_should_supervise_returns_false_below_min_citations():
    """F4: default min_citations bumped 2 → 3."""
    cits = [_cit("d1", "1", conf=0.5), _cit("d2", "2", conf=0.5)]  # only 2
    assert should_supervise(cits) is False
    assert should_supervise([]) is False


def test_should_supervise_default_fires_on_count_alone_F4():
    """F4 trigger: count-based, no confidence-band check by default.
    F3 telemetry showed Qwen3 confidences are bimodal so the band
    almost never matched."""
    # 3 citations, confidence anywhere → should fire by default.
    cits_low  = [_cit("d1","1",conf=0.10), _cit("d2","2",conf=0.05), _cit("d3","3",conf=0.02)]
    cits_high = [_cit("d1","1",conf=0.95), _cit("d2","2",conf=0.80), _cit("d3","3",conf=0.90)]
    cits_mid  = [_cit("d1","1",conf=0.45), _cit("d2","2",conf=0.40), _cit("d3","3",conf=0.50)]
    assert should_supervise(cits_low) is True
    assert should_supervise(cits_high) is True
    assert should_supervise(cits_mid) is True


def test_should_supervise_uncertainty_band_opt_in():
    """The original R9.5 confidence-band behavior is preserved as opt-in."""
    cits_low_conf = [_cit("d1","1",conf=0.10), _cit("d2","2",conf=0.05), _cit("d3","3",conf=0.02)]
    cits_mid_conf = [_cit("d1","1",conf=0.45), _cit("d2","2",conf=0.40), _cit("d3","3",conf=0.50)]
    band = (0.30, 0.70)
    assert should_supervise(cits_low_conf, uncertainty_band=band) is False
    assert should_supervise(cits_mid_conf, uncertainty_band=band) is True


# ---------------------------------------------------------------------------
# 5. Handler integration via supervisor_fn injection (rule_application)
# ---------------------------------------------------------------------------


def test_handler_integration_supervisor_fires_when_enough_citations_F4():
    """F4: supervisor fires whenever the verified citation set has at least
    ``DEFAULT_MIN_CITATIONS`` (3) candidates — confidence-band check is
    no longer the default.
    """
    from unittest.mock import MagicMock as MM
    from akn_rlm.indexers.bm25 import BM25Hit
    from akn_rlm.indexers.dense import DenseHit
    from akn_rlm.rlm.handlers.rule_application import RuleApplicationHandler
    from akn_rlm.rlm.routing import RouteResult

    bm25 = MM()
    # 3 BM25 hits → 3 verified citations → supervisor fires.
    bm25.search.return_value = [
        BM25Hit("d1#art_1", "d1", "1", score=5.0, text="نص 1"),
        BM25Hit("d2#art_2", "d2", "2", score=4.0, text="نص 2"),
        BM25Hit("d3#art_3", "d3", "3", score=3.0, text="نص 3"),
    ]
    dense = MM(); dense.search.return_value = []
    registry = MM(); registry.get_doc.return_value = MM(doc_title="قانون")
    router = MM(); router.route.return_value = RouteResult(
        doc_ids=["d1", "d2", "d3"], scores={"d1": 1.0}, sources={}, confidence=1.0,
    )

    # Verifier accepts everything with high confidence — under F3's
    # band-based trigger the supervisor would have skipped; under F4's
    # count-based trigger it must fire.
    def _verifier(_pool, _q, art, _model):
        return {"relevant": True, "supporting_span": None, "confidence": 0.92}

    sup = MM(side_effect=lambda pool, q, cits: [
        dict(c, supervisor_score=0.9 - 0.1 * i)
        for i, c in enumerate(reversed(cits))
    ])

    handler = RuleApplicationHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MM(),
        router=router,
        verifier_fn=_verifier,
        summarizer_fn=lambda *a, **k: {"summary": "ملخّص"},
        supervisor_fn=sup,
    )

    out = handler.run("سؤال")
    assert out["abstention"] is False
    assert out["_telemetry"]["supervisor_used"] is True
    sup.assert_called_once()
    assert all("supervisor_score" in c for c in out["citations"])


def test_handler_integration_supervisor_skips_when_too_few_citations_F4():
    """F4: supervisor must NOT fire when fewer than DEFAULT_MIN_CITATIONS
    (3) survive the verifier — there's nothing meaningful to re-rank."""
    from unittest.mock import MagicMock as MM
    from akn_rlm.indexers.bm25 import BM25Hit
    from akn_rlm.rlm.handlers.rule_application import RuleApplicationHandler
    from akn_rlm.rlm.routing import RouteResult

    bm25 = MM()
    # Only 2 candidates → 2 citations → supervisor must skip.
    bm25.search.return_value = [
        BM25Hit("d1#art_1", "d1", "1", score=5.0, text="نص 1"),
        BM25Hit("d2#art_2", "d2", "2", score=4.0, text="نص 2"),
    ]
    dense = MM(); dense.search.return_value = []
    registry = MM(); registry.get_doc.return_value = MM(doc_title="قانون")
    router = MM(); router.route.return_value = RouteResult(
        doc_ids=["d1", "d2"], scores={"d1": 1.0}, sources={}, confidence=1.0,
    )

    def _verifier(_pool, _q, art, _model):
        return {"relevant": True, "supporting_span": None, "confidence": 0.92}

    sup = MM()  # if it fires, test fails

    handler = RuleApplicationHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MM(),
        router=router, verifier_fn=_verifier,
        summarizer_fn=lambda *a, **k: {"summary": "ملخّص"},
        supervisor_fn=sup,
    )
    out = handler.run("سؤال")
    assert out["_telemetry"]["supervisor_used"] is False
    sup.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Defaults
# ---------------------------------------------------------------------------


def test_defaults_match_handoff_design():
    assert DEFAULT_MODEL == "gpt-oss-120b"
    # F7: reverted F6's 0.5 back to F4/F5's 0.3 — F6 evidence showed
    # threshold 0.5 was too aggressive (RA −0.013, MH −0.017, CD −0.028).
    # gpt-oss-120b at thr=0.5 drops foundational/scope articles that
    # ARE the gold answer for many legal queries (same R3/R4 lesson
    # learned for the verifier).
    assert DEFAULT_THRESHOLD == 0.3
    # Band constants preserved as opt-in; default trigger is count-only.
    assert DEFAULT_TRIGGER_LOW == 0.30
    assert DEFAULT_TRIGGER_HIGH == 0.70
    # F4: bumped 2 → 3 (only fire when there's enough to re-rank).
    assert DEFAULT_MIN_CITATIONS == 3


# ---------------------------------------------------------------------------
# 7. R9.6 plan supervisor
# ---------------------------------------------------------------------------

def test_plan_supervisor_returns_sub_questions_with_target_docs():
    """Happy path: planner returns valid JSON; we get sub_questions
    with filtered target_docs."""
    from akn_rlm.rlm.supervisor import supervise_plan

    payload = json.dumps({
        "sub_questions": [
            {"id": "sq1", "text": "ما الشرط الأول؟", "target_docs": ["d1", "d2"]},
            {"id": "sq2", "text": "ما الشرط الثاني؟", "target_docs": ["d3"]},
            # invented doc id should be dropped:
            {"id": "sq3", "text": "ما الشرط الثالث؟", "target_docs": ["fake_doc"]},
        ]
    })
    pool = _llm_pool_returning(payload)
    out = supervise_plan(
        pool, "سؤال طويل عن الشروط", routed_doc_ids=["d1", "d2", "d3"],
    )
    assert "sub_questions" in out
    sub_qs = out["sub_questions"]
    assert len(sub_qs) == 3
    assert sub_qs[0]["target_docs"] == ["d1", "d2"]
    assert sub_qs[1]["target_docs"] == ["d3"]
    # fake_doc filtered out → empty
    assert sub_qs[2]["target_docs"] == []


def test_plan_supervisor_skips_short_query():
    """Below min_content_tokens we don't even hit the LLM."""
    from akn_rlm.rlm.supervisor import supervise_plan

    pool = MagicMock()
    out = supervise_plan(pool, "Q", routed_doc_ids=["d1"])
    assert out == {}
    pool.call.assert_not_called()


def test_plan_supervisor_fails_open_on_llm_exception():
    from akn_rlm.rlm.supervisor import supervise_plan

    pool = _llm_pool_returning(RuntimeError("timeout"))
    out = supervise_plan(
        pool, "ما هي الشروط الأولى ثم الثانية ثم الثالثة",
        routed_doc_ids=["d1", "d2"],
    )
    assert out == {}


def test_plan_supervisor_fails_open_on_parse_failure():
    from akn_rlm.rlm.supervisor import supervise_plan

    pool = _llm_pool_returning("not json {")
    out = supervise_plan(
        pool, "ما هي الشروط الأولى ثم الثانية ثم الثالثة",
        routed_doc_ids=["d1", "d2"],
    )
    assert out == {}


def test_multi_hop_uses_plan_supervisor_when_provided():
    """When plan_supervisor_fn is injected and query is long enough,
    multi_hop uses its sub_questions and per-sub-q target_docs."""
    from unittest.mock import MagicMock as MM
    from akn_rlm.indexers.bm25 import BM25Hit
    from akn_rlm.indexers.dense import DenseHit
    from akn_rlm.rlm.handlers.multi_hop import MultiHopHandler
    from akn_rlm.rlm.routing import RouteResult

    bm25 = MM()
    bm25.search.return_value = [
        BM25Hit("d1#art_1", "d1", "1", score=5.0, text="نص 1"),
        BM25Hit("d2#art_2", "d2", "2", score=4.0, text="نص 2"),
    ]
    dense = MM()
    dense.search.return_value = [DenseHit("d1#art_1", "d1", "1", 0.5, "نص 1")]
    registry = MM(); registry.get_doc.return_value = MM(doc_title="قانون")
    router = MM(); router.route.return_value = RouteResult(
        doc_ids=["d1", "d2", "d3"], scores={"d1": 1.0, "d2": 1.0, "d3": 1.0},
        sources={}, confidence=1.0,
    )

    plan_supervisor = MM(return_value={
        "sub_questions": [
            {"id": "sq1", "text": "أ", "target_docs": ["d1"]},
            {"id": "sq2", "text": "ب", "target_docs": ["d2"]},
        ]
    })

    # Decomposer must NOT be called because plan supervisor succeeded.
    decomposer = MM()

    def _verifier(_pool, _q, art, _model):
        return {"relevant": True, "supporting_span": None, "confidence": 0.9}

    handler = MultiHopHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MM(),
        router=router,
        decomposer_fn=decomposer,
        verifier_fn=_verifier,
        summarizer_fn=lambda *a, **k: {"summary": "ملخّص"},
        plan_supervisor_fn=plan_supervisor,
    )
    out = handler.run("سؤال طويل عن أ و ب")
    assert out["_telemetry"]["plan_supervisor_used"] is True
    plan_supervisor.assert_called_once()
    decomposer.assert_not_called()
    # sub_questions trace carries the per-sub-q target_docs
    sub_qs_trace = out["_telemetry"]["sub_questions"]
    assert any(t.get("target_docs") == ["d1"] for t in sub_qs_trace)


def test_multi_hop_falls_back_to_decomposer_on_plan_failure():
    """When plan supervisor returns {} (parse failure / short query),
    multi_hop must call the existing Qwen decomposer."""
    from unittest.mock import MagicMock as MM
    from akn_rlm.rlm.handlers.multi_hop import MultiHopHandler
    from akn_rlm.rlm.routing import RouteResult

    bm25 = MM(); bm25.search.return_value = []
    dense = MM(); dense.search.return_value = []
    registry = MM()
    router = MM(); router.route.return_value = RouteResult(
        doc_ids=[], scores={}, sources={}, confidence=0.0,
    )

    plan_supervisor = MM(return_value={})  # fail path
    decomposer = MM(return_value={
        "sub_questions": [
            {"id": "sq1", "text": "أ", "type": "rule_application"},
        ]
    })

    handler = MultiHopHandler(
        bm25=bm25, dense=dense, registry=registry, llm_pool=MM(),
        router=router,
        decomposer_fn=decomposer,
        plan_supervisor_fn=plan_supervisor,
    )
    out = handler.run("سؤال طويل ثلاث كلمات")
    assert out["_telemetry"]["plan_supervisor_used"] is False
    plan_supervisor.assert_called_once()
    decomposer.assert_called_once()
