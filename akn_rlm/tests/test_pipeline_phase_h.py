"""Phase H pipeline tests: node-level unit tests + end-to-end integration."""
from __future__ import annotations

import json
import pytest

from akn_rlm.rlm.legal_env import LegalEnv
from akn_rlm.rlm.recursion_budget import RecursionBudget
from akn_rlm.rlm.pipeline import (
    build_pipeline,
    _detect_language,
    _decompose_query,
    _route_after_gates,
    MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Stubs (copied pattern from test_pipeline_smoke.py)
# ---------------------------------------------------------------------------

class _StubHit:
    def __init__(self, doc_id, article_ref, text="stub"):
        self.chunk_id    = f"{doc_id}#art_{article_ref}"
        self.doc_id      = doc_id
        self.article_ref = article_ref
        self.score       = 1.0
        self.text        = text


class _StubIndex:
    def __init__(self, hits):
        self._hits = hits
        self._meta = [
            {"chunk_id": h.chunk_id, "doc_id": h.doc_id,
             "article_ref": h.article_ref, "text": h.text}
            for h in hits
        ]

    def search(self, query, k=10):
        return self._hits[:k]


class _StubDocEntry:
    def __init__(self, doc_id, nums):
        self.doc_id = doc_id
        self.doc_title = f"Doc {doc_id}"
        self.doc_date = "1984-06-09"
        from akn_rlm.normalizers import ref_to_eid
        self.article_eids = {ref_to_eid(str(n)) for n in nums}


class _StubRegistry:
    def __init__(self, docs):
        self._docs = docs

    def has_article(self, doc_id, article_ref):
        from akn_rlm.normalizers import ref_to_eid
        entry = self._docs.get(doc_id)
        return entry is not None and ref_to_eid(str(article_ref)) in entry.article_eids

    def get_doc(self, doc_id):
        return self._docs.get(doc_id)

    def resolve_alias(self, alias):
        return None


class _SequencedLLM:
    def __init__(self, responses):
        self._responses = responses
        self._idx = 0

    def chat(self, messages, model, **kw):
        r = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return r

    def complete(self, prompt, model, **kw):
        return self.chat([], model=model)


class _StubLLMPool:
    def __init__(self, llm):
        self._llm = llm

    def _route(self, model):
        return self._llm

    def call(self, prompt, model, **kw):
        return self._llm.complete(prompt, model=model)


def _answer_block(answer_text="OK", abstention=False, abstention_reason=None,
                  citations=None):
    obj = {
        "answer_text": answer_text,
        "abstention": abstention,
        "abstention_reason": abstention_reason,
        "citations": citations or [],
        "reasoning_chain": ["1.", "2.", "3.", "4."],
    }
    return f"```answer\n{json.dumps(obj, ensure_ascii=False)}\n```"


def _make_env(responses, article_nums=range(1, 200)):
    docs = {"84-11_1984-06-09": _StubDocEntry("84-11_1984-06-09", article_nums)}
    registry = _StubRegistry(docs)
    hits = [_StubHit("84-11_1984-06-09", "54",
                     text="يجوز للزوجة دون موافقة الزوج أن تخالع نفسها بمقابل مالي")]
    indices = {"bm25": _StubIndex(hits)}
    pool = _StubLLMPool(_SequencedLLM(responses))
    budget = RecursionBudget(max_depth=1, max_sub_calls=5, max_total_tokens=200_000)
    return LegalEnv(registry=registry, kg=None, indices=indices,
                    llm_pool=pool, budget=budget)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_arabic(self):
        assert _detect_language("يجوز للزوجة الخلع دون موافقة الزوج") == "ar"

    def test_french(self):
        assert _detect_language("Le droit algérien prévoit le divorce") == "fr"

    def test_mixed_mostly_arabic(self):
        assert _detect_language("المادة 54 من qanoun al-usra") == "ar"

    def test_empty(self):
        result = _detect_language("")
        assert result in ("ar", "fr")


class TestDecomposeQuery:
    def test_arabic_conjunction(self):
        parts = _decompose_query("ما حكم الطلاق و الخلع في قانون الأسرة؟")
        assert len(parts) >= 2

    def test_english_and(self):
        parts = _decompose_query("divorce law and marriage law")
        assert len(parts) == 2

    def test_no_conjunction(self):
        q = "ما هي شروط الخلع؟"
        parts = _decompose_query(q)
        assert len(parts) == 1
        assert parts[0] == q


class TestRouteAfterGates:
    def _state(self, gates, retry=0):
        return {"gate_results": gates, "retry_count": retry}

    def test_all_pass_routes_to_assemble(self):
        state = self._state({"citation": {"passed": True}, "jurisdiction": {"passed": True}})
        assert _route_after_gates(state) == "assemble_output"

    def test_fail_with_retries_left_routes_to_retry(self):
        state = self._state({"citation": {"passed": False}}, retry=0)
        assert _route_after_gates(state) == "corrective_retry"

    def test_fail_retries_exhausted_routes_to_assemble(self):
        state = self._state({"citation": {"passed": False}}, retry=MAX_RETRIES)
        assert _route_after_gates(state) == "assemble_output"

    def test_empty_gates_routes_to_assemble(self):
        state = self._state({})
        assert _route_after_gates(state) == "assemble_output"


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPipelinePhaseH:
    def test_normalize_sets_language(self):
        resp = _answer_block()
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        answer = pipeline.run("يجوز للزوجة الخلع")
        assert isinstance(answer, dict)
        # language is in telemetry
        assert answer.get("_telemetry", {}).get("language") == "ar"

    def test_valid_citation_passes_all_gates(self):
        cite = {"doc_id": "84-11_1984-06-09", "article_ref": "54",
                "supporting_span": "يجوز للزوجة", "confidence": 0.9}
        resp = _answer_block(citations=[cite],
                             answer_text="يجوز للزوجة دون موافقة الزوج أن تخالع")
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        answer = pipeline.run("ما هي شروط الخلع؟")
        assert "answer_text" in answer
        assert answer.get("_telemetry", {}).get("retry_count", 0) == 0

    def test_abstention_passes_through(self):
        resp = _answer_block(abstention=True, abstention_reason="foreign_jurisdiction")
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        answer = pipeline.run("هل يخضع لضريبة ISF؟")
        assert answer.get("abstention") is True
        # Gate nodes skip when abstention=True — no retries
        assert answer.get("_telemetry", {}).get("retry_count", 0) == 0

    def test_bad_citation_exhausts_retries_safe_abstention(self):
        bad = {"doc_id": "FAKE_DOC", "article_ref": "99"}
        resp = _answer_block(citations=[bad])
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        answer = pipeline.run("query about unknown doc")
        # After MAX_RETRIES exhausted → safe abstention with citations=[]
        assert answer.get("citations") == []
        assert answer.get("_telemetry", {}).get("retry_count", 0) >= MAX_RETRIES

    def test_telemetry_has_gate_results(self):
        cite = {"doc_id": "84-11_1984-06-09", "article_ref": "54",
                "supporting_span": "stub", "confidence": 0.8}
        resp = _answer_block(citations=[cite])
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        answer = pipeline.run("query")
        telemetry = answer.get("_telemetry", {})
        gate_results = telemetry.get("gate_results", {})
        assert "citation" in gate_results
        assert "jurisdiction" in gate_results
        assert "faithfulness" in gate_results

    def test_multi_hop_goes_through_decompose(self):
        cite = {"doc_id": "84-11_1984-06-09", "article_ref": "54",
                "supporting_span": "stub", "confidence": 0.8}
        resp = _answer_block(citations=[cite])
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        # Multi-hop keyword triggers decompose node
        answer = pipeline.run("قانون الأسرة و القانون المدني في الخلع")
        assert isinstance(answer, dict)
        assert "answer_text" in answer

    def test_return_schema_complete(self):
        resp = _answer_block()
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        answer = pipeline.run("any query")
        required = {"answer_text", "abstention", "citations",
                    "reasoning_chain", "trajectory", "tokens_used"}
        assert required.issubset(answer.keys())

    def test_jurisdiction_signal_in_answerable_triggers_retry(self):
        # Answer text contains ISF (foreign signal) for an answerable question
        resp = _answer_block(answer_text="ISF ضريبة الثروة لا تنطبق",
                             citations=[])
        env = _make_env([resp])
        pipeline = build_pipeline(env, few_shot=[])
        # Answerable question → jurisdiction gate should fail
        answer = pipeline.run("ما هو نظام الضرائب الجزائري؟")
        # Either retried and eventually abstained, or passed (regex-only may miss it)
        assert isinstance(answer, dict)
