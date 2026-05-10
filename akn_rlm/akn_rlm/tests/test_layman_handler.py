"""Tests for the Phase-2 layman handler.

The handler's contribution over rule_application is the mandatory Darja →
MSA rewrite step. The inner rule_application handler is mocked so these
tests focus on rewrite logic + delegation + telemetry — they don't
re-test the rule_application pipeline (covered in
``test_rule_application_handler.py``).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akn_rlm.rlm.handlers.layman import (
    DEFAULT_REWRITE_MAX_TOKENS,
    DEFAULT_REWRITE_MODEL,
    DEFAULT_VERIFY_THRESHOLD,
    TELEMETRY_BASELINE,
    LaymanHandler,
    build_layman_handler,
    call_darja_rewriter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_rule_handler(answer: dict | None = None) -> MagicMock:
    """A mock rule_application handler whose `.run(query)` returns `answer`."""
    base = answer or {
        "answer_text": "ملخّص",
        "abstention": False,
        "abstention_reason": None,
        "citations": [
            {
                "doc_id": "84-11_1984-06-09",
                "article_ref": "5",
                "doc_title": "قانون الأسرة",
                "supporting_span": "نص",
                "text": "نص المادة 5",
                "confidence": 0.9,
                "verifier_relevant": True,
            },
        ],
        "reasoning_chain": ["routed_doc_ids=['84-11_1984-06-09']"],
        "trajectory": [],
        "tokens_used": 0,
        "depth_max_reached": 1,
        "_telemetry": {
            "retry_count":     0,
            "gate_results":    {},
            "baseline":        "rlm_rule_application",
            "routed_doc_ids":  ["84-11_1984-06-09"],
            "top_score":       0.5,
            "candidate_count": 5,
            "verified_count":  1,
            "sub_call_count":  6,
        },
    }
    rule = MagicMock()
    rule.run = MagicMock(return_value=base)
    return rule


def _stub_rewriter(rewrites: dict[str, str] | str | None = None):
    """Factory for an injectable rewriter mock.

    - If ``rewrites`` is a dict: return per-input rewrite (case-sensitive
      lookup; default = empty so handler falls back to original).
    - If ``rewrites`` is a string: always return that string.
    - If None: return empty (handler falls back to original).
    """
    def _fn(_pool, query, _model):
        if isinstance(rewrites, dict):
            return rewrites.get(query, "")
        if isinstance(rewrites, str):
            return rewrites
        return ""
    return MagicMock(side_effect=_fn)


def _make_handler(
    *,
    rewrites=None,
    rule_answer: dict | None = None,
):
    rewriter = _stub_rewriter(rewrites)
    rule_handler = _stub_rule_handler(rule_answer)
    handler = LaymanHandler(
        bm25=MagicMock(),
        dense=MagicMock(),
        registry=MagicMock(),
        llm_pool=MagicMock(),
        router=MagicMock(),
        rewriter_fn=rewriter,
        rule_handler=rule_handler,
    )
    return handler, dict(rewriter=rewriter, rule=rule_handler)


# ---------------------------------------------------------------------------
# Defaults & contract
# ---------------------------------------------------------------------------


def test_defaults_match_handoff_design():
    assert DEFAULT_REWRITE_MODEL  # non-empty string
    assert DEFAULT_REWRITE_MAX_TOKENS >= 64


def test_f4_default_verify_threshold_locked_at_0_5():
    """Layman delegates to rule_application; F4 reverted R9.1 (0.3 back
    to 0.5) — locked here so a future drift in either module fails loudly.
    """
    from akn_rlm.rlm.handlers import layman as lm_mod
    from akn_rlm.rlm.handlers import rule_application as ra_mod
    assert lm_mod.DEFAULT_VERIFY_THRESHOLD == 0.5
    assert ra_mod.DEFAULT_VERIFY_THRESHOLD == 0.5
    assert lm_mod.DEFAULT_VERIFY_THRESHOLD == ra_mod.DEFAULT_VERIFY_THRESHOLD


def test_factory_builds_handler():
    h = build_layman_handler(
        bm25=MagicMock(), dense=MagicMock(), registry=MagicMock(),
        llm_pool=MagicMock(), router=MagicMock(),
        rule_handler=_stub_rule_handler(),
    )
    assert isinstance(h, LaymanHandler)


def test_telemetry_baseline_tag_is_rlm_layman():
    handler, _ = _make_handler(rewrites="السؤال بالفصحى")
    out = handler.run("سؤال بالدارجة")
    assert out["_telemetry"]["baseline"] == TELEMETRY_BASELINE
    assert TELEMETRY_BASELINE == "rlm_layman"


def test_run_returns_required_keys():
    handler, _ = _make_handler(rewrites="MSA")
    out = handler.run("Darja")
    for key in ("answer_text", "abstention", "abstention_reason", "citations",
                "reasoning_chain", "trajectory", "tokens_used", "_telemetry"):
        assert key in out


# ---------------------------------------------------------------------------
# Rewrite path — happy path
# ---------------------------------------------------------------------------


def test_rewriter_called_once_with_original_query():
    handler, mocks = _make_handler(rewrites="MSA")
    handler.run("هل نقدر نطلق مرتي؟")
    mocks["rewriter"].assert_called_once()
    args = mocks["rewriter"].call_args.args
    # Signature: (llm_pool, query, model)
    assert args[1] == "هل نقدر نطلق مرتي؟"


def test_rewritten_query_passed_to_inner_rule_handler():
    rewritten = "هل يمكنني تطليق زوجتي؟"
    handler, mocks = _make_handler(rewrites=rewritten)
    handler.run("هل نقدر نطلق مرتي؟")
    mocks["rule"].run.assert_called_once_with(rewritten)


def test_telemetry_records_rewrite_input_and_output():
    rewritten = "هل يمكنني تطليق زوجتي؟"
    handler, _ = _make_handler(rewrites=rewritten)
    out = handler.run("هل نقدر نطلق مرتي؟")
    tel = out["_telemetry"]
    assert tel["rewrite_input"] == "هل نقدر نطلق مرتي؟"
    assert tel["rewrite_output"] == rewritten
    assert tel["rewrite_used"] is True


def test_reasoning_chain_records_rewrite():
    rewritten = "هل يمكنني تطليق زوجتي؟"
    handler, _ = _make_handler(rewrites=rewritten)
    out = handler.run("هل نقدر نطلق مرتي؟")
    chain = out["reasoning_chain"]
    assert any("darja_rewrite_used=True" in step for step in chain)
    assert any(rewritten in step for step in chain)


def test_telemetry_carries_inner_baseline_for_audit():
    handler, _ = _make_handler(rewrites="MSA")
    out = handler.run("Darja")
    assert out["_telemetry"]["inner_baseline"] == "rlm_rule_application"


# ---------------------------------------------------------------------------
# Rewrite path — fallback to original
# ---------------------------------------------------------------------------


def test_empty_rewrite_falls_back_to_original():
    handler, mocks = _make_handler(rewrites="")
    handler.run("هل نقدر نطلق مرتي؟")
    # Inner handler called with original query.
    mocks["rule"].run.assert_called_once_with("هل نقدر نطلق مرتي؟")


def test_empty_rewrite_records_rewrite_used_false():
    handler, _ = _make_handler(rewrites="")
    out = handler.run("هل نقدر نطلق مرتي؟")
    assert out["_telemetry"]["rewrite_used"] is False


def test_identical_rewrite_falls_back_to_original():
    same = "السؤال نفسه"
    handler, mocks = _make_handler(rewrites=same)
    handler.run(same)
    mocks["rule"].run.assert_called_once_with(same)
    out = mocks["rule"].run.return_value
    # Telemetry should still mark rewrite_used=False because rewrite ==
    # original.


def test_identical_rewrite_telemetry_marks_unused():
    same = "السؤال نفسه"
    handler, _ = _make_handler(rewrites=same)
    out = handler.run(same)
    assert out["_telemetry"]["rewrite_used"] is False


def test_rewriter_exception_falls_back_to_original():
    def raising(*_a, **_k):
        raise RuntimeError("rewriter blew up")
    handler, mocks = _make_handler()
    handler._rewriter_fn = raising
    handler.run("هل نقدر نطلق مرتي؟")
    mocks["rule"].run.assert_called_once_with("هل نقدر نطلق مرتي؟")


def test_whitespace_rewrite_falls_back_to_original():
    handler, mocks = _make_handler(rewrites="   \n\t   ")
    handler.run("الأصلي")
    mocks["rule"].run.assert_called_once_with("الأصلي")


# ---------------------------------------------------------------------------
# Sub-LM call accounting
# ---------------------------------------------------------------------------


def test_sub_call_count_includes_rewriter_plus_inner():
    """sub_call_count = 1 (rewriter) + inner sub_call_count."""
    rule_answer = {
        "answer_text": "ans",
        "abstention": False,
        "abstention_reason": None,
        "citations": [],
        "reasoning_chain": [],
        "trajectory": [],
        "tokens_used": 0,
        "depth_max_reached": 1,
        "_telemetry": {
            "baseline":       "rlm_rule_application",
            "sub_call_count": 7,
        },
    }
    handler, _ = _make_handler(rewrites="MSA", rule_answer=rule_answer)
    out = handler.run("Darja")
    assert out["_telemetry"]["sub_call_count"] == 1 + 7


def test_sub_call_count_when_rewriter_runs_but_falls_back():
    """Even when the rewrite is rejected (empty / identical), the rewriter
    LLM call still happened and should be counted."""
    rule_answer = {
        "answer_text": "ans", "abstention": False, "abstention_reason": None,
        "citations": [], "reasoning_chain": [], "trajectory": [],
        "tokens_used": 0, "depth_max_reached": 1,
        "_telemetry": {"baseline": "rlm_rule_application", "sub_call_count": 5},
    }
    handler, _ = _make_handler(rewrites="", rule_answer=rule_answer)
    out = handler.run("الأصلي")
    assert out["_telemetry"]["sub_call_count"] == 1 + 5


def test_rewriter_exception_does_not_count_a_sub_call():
    """Rewriter exception → sub_call_count = inner only."""
    rule_answer = {
        "answer_text": "ans", "abstention": False, "abstention_reason": None,
        "citations": [], "reasoning_chain": [], "trajectory": [],
        "tokens_used": 0, "depth_max_reached": 1,
        "_telemetry": {"baseline": "rlm_rule_application", "sub_call_count": 3},
    }
    handler, _ = _make_handler(rule_answer=rule_answer)
    handler._rewriter_fn = MagicMock(side_effect=RuntimeError("boom"))
    out = handler.run("Darja")
    assert out["_telemetry"]["sub_call_count"] == 3


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


def test_empty_query_abstains_without_calling_anything():
    handler, mocks = _make_handler()
    out = handler.run("")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    mocks["rewriter"].assert_not_called()
    mocks["rule"].run.assert_not_called()


def test_whitespace_query_abstains():
    handler, _ = _make_handler()
    out = handler.run("   \n\t  ")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"


def test_inner_abstention_propagates():
    """If the inner rule_application handler abstains, the outer handler
    also abstains."""
    rule_answer = {
        "answer_text": "",
        "abstention": True,
        "abstention_reason": "no_verified_articles",
        "citations": [],
        "reasoning_chain": [],
        "trajectory": [],
        "tokens_used": 0,
        "depth_max_reached": 0,
        "_telemetry": {"baseline": "rlm_rule_application", "sub_call_count": 4},
    }
    handler, _ = _make_handler(rewrites="MSA", rule_answer=rule_answer)
    out = handler.run("Darja")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"


def test_inner_pipeline_exception_abstains():
    handler, mocks = _make_handler(rewrites="MSA")
    mocks["rule"].run.side_effect = RuntimeError("rule blew up")
    out = handler.run("Darja")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "inner_pipeline_error"


# ---------------------------------------------------------------------------
# Citations / answer pass-through
# ---------------------------------------------------------------------------


def test_citations_pass_through_from_inner():
    cit = {
        "doc_id": "x", "article_ref": "1", "doc_title": "t",
        "supporting_span": "s", "text": "txt", "confidence": 0.7,
    }
    rule_answer = {
        "answer_text": "A",
        "abstention": False, "abstention_reason": None,
        "citations": [cit],
        "reasoning_chain": [], "trajectory": [],
        "tokens_used": 0, "depth_max_reached": 1,
        "_telemetry": {"baseline": "rlm_rule_application", "sub_call_count": 0},
    }
    handler, _ = _make_handler(rewrites="MSA", rule_answer=rule_answer)
    out = handler.run("Darja")
    assert out["citations"] == [cit]
    assert out["answer_text"] == "A"


# ---------------------------------------------------------------------------
# Direct test of the default rewriter helper
# ---------------------------------------------------------------------------


def test_default_rewriter_strips_response():
    pool = MagicMock()
    pool.call.return_value = "  هذا الرد  \n"
    out = call_darja_rewriter(pool, "سؤال", model="x")
    assert out == "هذا الرد"


def test_default_rewriter_strips_label_prefix():
    pool = MagicMock()
    pool.call.return_value = "السؤال (فصحى): هذا الرد"
    out = call_darja_rewriter(pool, "سؤال", model="x")
    assert out == "هذا الرد"


def test_default_rewriter_strips_quote_wrapping():
    pool = MagicMock()
    pool.call.return_value = '"هذا الرد"'
    out = call_darja_rewriter(pool, "سؤال", model="x")
    assert out == "هذا الرد"


def test_default_rewriter_empty_query_returns_empty():
    pool = MagicMock()
    out = call_darja_rewriter(pool, "", model="x")
    assert out == ""
    pool.call.assert_not_called()


def test_default_rewriter_llm_exception_returns_empty():
    pool = MagicMock()
    pool.call.side_effect = RuntimeError("boom")
    out = call_darja_rewriter(pool, "سؤال", model="x")
    assert out == ""


def test_default_rewriter_empty_response_returns_empty():
    pool = MagicMock()
    pool.call.return_value = ""
    out = call_darja_rewriter(pool, "سؤال", model="x")
    assert out == ""


def test_default_rewriter_uses_model_arg():
    pool = MagicMock()
    pool.call.return_value = "rewritten"
    call_darja_rewriter(pool, "q", model="custom-model")
    assert pool.call.call_args.kwargs.get("model") == "custom-model"
