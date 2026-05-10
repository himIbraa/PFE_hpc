"""Tests for the Phase-2 / R9.7 telemetry persistence.

Three concerns:
  1. ``LLMPool.call`` records per-call model into a per-handler-run
     buffer when ``start_recording`` is active, and is a no-op when
     it's not.
  2. The dispatcher snapshots calls around ``handler.run(query)`` and
     surfaces ``calls_by_model`` in the per-question telemetry.
  3. ``runner._answer_to_result`` persists ``sub_call_count`` and
     ``calls_by_model`` into the predictions.jsonl row alongside
     ``tokens_used``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.eval.runner import _answer_to_result
from akn_rlm.llm.client import LLMClient, LLMPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubClient(LLMClient):
    """Trivial client that returns a fixed string. Avoids hitting any
    real LLM."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply

    def chat(self, messages, model, max_tokens=1024, temperature=0.0):
        return self.reply


def _pool() -> LLMPool:
    return LLMPool({"qwen": _StubClient(), "gpt": _StubClient(), "google": _StubClient()})


# ---------------------------------------------------------------------------
# 1. LLMPool recording
# ---------------------------------------------------------------------------


def test_llm_pool_no_recording_by_default():
    """When start_recording is never called, .call() does NOT mutate
    any internal counter."""
    pool = _pool()
    pool.call("p", model="qwen3-30b")
    pool.call("p", model="qwen3-30b")
    # snapshot returns empty — recording is inactive.
    assert pool.snapshot_calls() == {}


def test_llm_pool_records_per_model_when_active():
    pool = _pool()
    pool.start_recording()
    pool.call("p1", model="qwen3-30b-thinking")
    pool.call("p2", model="qwen3-30b-thinking")
    pool.call("p3", model="gpt-oss-120b")
    pool.call("p4", model="google/gemma-4-31B")
    counts = pool.stop_recording()
    assert counts["qwen3-30b-thinking"] == 2
    assert counts["gpt-oss-120b"] == 1
    assert counts["google/gemma-4-31B"] == 1
    # Stopping clears the internal buffer.
    assert pool.snapshot_calls() == {}


def test_llm_pool_recording_resets_on_restart():
    pool = _pool()
    pool.start_recording()
    pool.call("p", model="qwen3-30b-thinking")
    pool.start_recording()  # restart should zero the buffer
    pool.call("p", model="gpt-oss-120b")
    counts = pool.stop_recording()
    assert counts == {"gpt-oss-120b": 1}


# ---------------------------------------------------------------------------
# 2. Dispatcher snapshots calls around handler.run
# ---------------------------------------------------------------------------


def test_dispatcher_surfaces_calls_by_model_in_telemetry():
    """The dispatcher must call start_recording / stop_recording around
    handler.run and persist the result in answer["_telemetry"]["calls_by_model"]."""
    from akn_rlm.rlm.dispatcher import RLMDispatcher

    fake_pool = MagicMock()
    # Mimic the LLMPool API the dispatcher uses for recording.
    fake_pool.start_recording = MagicMock()
    fake_pool.stop_recording = MagicMock(
        return_value={"qwen3-30b-thinking": 3, "gpt-oss-120b": 1}
    )

    handler = MagicMock()
    handler.run.return_value = {
        "answer_text":       "ok",
        "abstention":        False,
        "abstention_reason": None,
        "citations":         [],
        "_telemetry":        {"sub_call_count": 4, "baseline": "rlm_rule_application"},
    }

    dispatcher = RLMDispatcher(
        bm25=MagicMock(), dense=MagicMock(), registry=MagicMock(),
        llm_pool=fake_pool, router=MagicMock(),
        handler_overrides={"rule_application": handler},
    )

    out = dispatcher.run("سؤال", query_type="rule_application")
    fake_pool.start_recording.assert_called_once()
    fake_pool.stop_recording.assert_called_once()

    tel = out["_telemetry"]
    assert tel["calls_by_model"] == {"qwen3-30b-thinking": 3, "gpt-oss-120b": 1}
    assert tel["dispatched_handler"] == "rule_application"


# ---------------------------------------------------------------------------
# 3. runner._answer_to_result persists fields into the predictions row
# ---------------------------------------------------------------------------


def test_answer_to_result_persists_sub_call_count_and_calls_by_model():
    question = {
        "id":               "q01",
        "query":             "Q",
        "query_type":        "rule_application",
        "gold_doc_ids":      [],
        "gold_article_ids":  [],
        "gold_citations":    [],
        "gold_abstain":      False,
    }
    answer = {
        "answer_text":       "ok",
        "abstention":        False,
        "citations":         [],
        "_telemetry": {
            "sub_call_count":   7,
            "calls_by_model":   {"qwen3-30b-thinking": 5, "gpt-oss-120b": 2},
            "supervisor_used":  True,
            "dispatched_handler": "multi_hop",
        },
    }

    row = _answer_to_result(question, answer)

    assert row["sub_call_count"] == 7
    assert row["calls_by_model"] == {"qwen3-30b-thinking": 5, "gpt-oss-120b": 2}
    assert row["supervisor_used"] is True
    assert row["dispatched_handler"] == "multi_hop"


def test_answer_to_result_handles_missing_telemetry_gracefully():
    """Baselines that don't emit the new R9.7 telemetry must still
    produce a well-formed predictions row (defaults apply)."""
    question = {
        "id":               "q01",
        "query":             "Q",
        "query_type":        "rule_application",
        "gold_doc_ids":      [],
        "gold_article_ids":  [],
        "gold_citations":    [],
        "gold_abstain":      False,
    }
    answer = {
        "answer_text":  "ok",
        "abstention":   False,
        "citations":    [],
    }

    row = _answer_to_result(question, answer)

    assert row["sub_call_count"] == 0
    assert row["calls_by_model"] == {}
    assert row["supervisor_used"] is False
    assert row["dispatched_handler"] is None
