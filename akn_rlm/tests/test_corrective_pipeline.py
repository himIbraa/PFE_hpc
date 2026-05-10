"""Smoke tests for corrective.py and pipeline.py."""
from __future__ import annotations

import json
import pytest

from akn_rlm.rlm.legal_env import LegalEnv
from akn_rlm.rlm.recursion_budget import RecursionBudget
from akn_rlm.rlm.root_controller import RootController
from akn_rlm.rlm.corrective import run_with_correction
from akn_rlm.rlm.pipeline import build_pipeline


# ---------------------------------------------------------------------------
# Minimal stubs (copied from test_pipeline_smoke.py pattern)
# ---------------------------------------------------------------------------

class _StubHit:
    def __init__(self, doc_id, article_ref, score=1.0, text="stub"):
        self.chunk_id    = f"{doc_id}#art_{article_ref}"
        self.doc_id      = doc_id
        self.article_ref = article_ref
        self.score       = score
        self.text        = text


class _StubIndex:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, k=10):
        return self._hits[:k]


class _StubDocEntry:
    def __init__(self, doc_id, article_numbers):
        self.doc_id    = doc_id
        self.doc_title = f"Doc {doc_id}"
        self.doc_date  = "1984-06-09"
        from akn_rlm.normalizers import ref_to_eid
        self.article_eids = {ref_to_eid(str(n)) for n in article_numbers}


class _StubRegistry:
    def __init__(self, docs):
        self._docs = docs

    def has_article(self, doc_id, article_ref):
        from akn_rlm.normalizers import ref_to_eid
        entry = self._docs.get(doc_id)
        if entry is None:
            return False
        return ref_to_eid(str(article_ref)) in entry.article_eids

    def get_doc(self, doc_id):
        return self._docs.get(doc_id)

    def resolve_alias(self, alias):
        return None


class _SequencedLLM:
    def __init__(self, responses):
        self._responses = responses
        self._idx = 0

    def chat(self, messages, model, max_tokens=4096, temperature=0.0):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    def complete(self, prompt, model, max_tokens=1024, system=None, temperature=0.0):
        return self.chat([], model=model, max_tokens=max_tokens)


class _StubLLMPool:
    def __init__(self, llm):
        self._llm = llm

    def _route(self, model):
        return self._llm

    def call(self, prompt, model, max_tokens=1024, **kwargs):
        return self._llm.complete(prompt, model=model, max_tokens=max_tokens)


def _make_answer(answer_text="OK", abstention=False, abstention_reason=None,
                 citations=None):
    obj = {
        "answer_text": answer_text,
        "abstention": abstention,
        "abstention_reason": abstention_reason,
        "citations": citations or [],
        "reasoning_chain": ["1.", "2.", "3.", "4."],
    }
    return f"```answer\n{json.dumps(obj, ensure_ascii=False)}\n```"


def _make_env(responses):
    docs = {
        "84-11_1984-06-09": _StubDocEntry("84-11_1984-06-09", range(1, 200)),
    }
    registry = _StubRegistry(docs)
    hits = [_StubHit("84-11_1984-06-09", "54")]
    indices = {"bm25": _StubIndex(hits)}
    llm = _SequencedLLM(responses)
    pool = _StubLLMPool(llm)
    budget = RecursionBudget(max_depth=1, max_sub_calls=5, max_total_tokens=200_000)
    return LegalEnv(registry=registry, kg=None, indices=indices,
                    llm_pool=pool, budget=budget)


# ---------------------------------------------------------------------------
# Tests: corrective loop
# ---------------------------------------------------------------------------

class TestCorrectiveLoop:
    def test_clean_answer_accepted_immediately(self):
        valid_citation = {"doc_id": "84-11_1984-06-09", "article_ref": "54",
                          "supporting_span": "text", "confidence": 0.9}
        resp = _make_answer(citations=[valid_citation])
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        result = run_with_correction(ctrl, "query", query_type="rule_application")
        assert result["abstention"] is False
        assert len(result["citations"]) == 1

    def test_fictitious_citation_removed(self):
        bad_citation = {"doc_id": "NONEXISTENT", "article_ref": "1",
                        "supporting_span": "...", "confidence": 0.9}
        resp = _make_answer(citations=[bad_citation])
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        result = run_with_correction(ctrl, "query", query_type="rule_application")
        assert result["citations"] == []

    def test_abstention_passes_through(self):
        resp = _make_answer(abstention=True, abstention_reason="foreign_jurisdiction")
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        result = run_with_correction(ctrl, "ISF query", query_type="unanswerable")
        assert result["abstention"] is True

    def test_result_has_required_keys(self):
        resp = _make_answer()
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        result = run_with_correction(ctrl, "query")
        required = {"answer_text", "abstention", "citations",
                    "reasoning_chain", "trajectory", "tokens_used", "depth_max_reached"}
        assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# Tests: pipeline graph
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_pipeline_returns_answer(self):
        valid_citation = {"doc_id": "84-11_1984-06-09", "article_ref": "54",
                          "supporting_span": "text", "confidence": 0.9}
        resp = _make_answer(citations=[valid_citation])
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        answer = pipeline.run("ما هي شروط الخلع؟")
        assert isinstance(answer, dict)
        assert "answer_text" in answer

    def test_pipeline_handles_abstention(self):
        resp = _make_answer(abstention=True, abstention_reason="foreign_jurisdiction")
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        answer = pipeline.run("هل يخضع لضريبة ISF؟")
        assert answer.get("abstention") is True

    def test_pipeline_drops_bad_citation(self):
        bad = {"doc_id": "FAKE", "article_ref": "99"}
        resp = _make_answer(citations=[bad])
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        answer = pipeline.run("query")
        assert answer.get("citations") == []
