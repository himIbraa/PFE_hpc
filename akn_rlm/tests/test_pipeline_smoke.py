"""Smoke tests for the full RLM pipeline (Phase D).

Uses a stub LLM that returns pre-canned responses — no real API calls.
Uses the stub registry from test_legal_env.py — no corpus loading.

Asserts for each question:
  (a) controller returns without exceptions,
  (b) output has the required schema keys,
  (c) abstention flag is set correctly,
  (d) all citations pass verify_in_corpus.
"""
from __future__ import annotations

import json
import pytest

from akn_rlm.rlm.legal_env import LegalEnv
from akn_rlm.rlm.recursion_budget import RecursionBudget
from akn_rlm.rlm.root_controller import RootController


# ---------------------------------------------------------------------------
# Minimal stubs (reused from test_legal_env.py pattern)
# ---------------------------------------------------------------------------

class _StubHit:
    def __init__(self, doc_id, article_ref, score=1.0, text="stub text"):
        self.chunk_id    = f"{doc_id}#art_{article_ref}"
        self.doc_id      = doc_id
        self.article_ref = article_ref
        self.score       = score
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
    def __init__(self, doc_id, article_numbers):
        self.doc_id    = doc_id
        self.doc_title = f"Doc {doc_id}"
        self.doc_date  = "1984-06-09"
        from akn_rlm.normalizers import ref_to_eid
        self.article_eids = {ref_to_eid(str(n)) for n in article_numbers}


class _StubRegistry:
    def __init__(self, docs: dict):
        self._docs = docs

    def has_article(self, doc_id, article_ref):
        from akn_rlm.normalizers import ref_to_eid
        entry = self._docs.get(doc_id)
        if entry is None:
            return False
        eid = ref_to_eid(str(article_ref))
        return eid in entry.article_eids

    def get_doc(self, doc_id):
        return self._docs.get(doc_id)

    def resolve_alias(self, alias):
        return None


class _SequencedLLM:
    """Returns pre-canned responses in sequence, cycling if needed."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def chat(self, messages, model, max_tokens=4096, temperature=0.0) -> str:
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp

    def complete(self, prompt, model, max_tokens=1024, system=None, temperature=0.0) -> str:
        return self.chat([], model=model, max_tokens=max_tokens)


class _StubLLMPool:
    def __init__(self, llm: _SequencedLLM) -> None:
        self._llm = llm

    def _route(self, model: str):
        return self._llm

    def call(self, prompt, model, max_tokens=1024, **kwargs) -> str:
        return self._llm.complete(prompt, model=model, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_env(llm_responses: list[str], extra_articles: dict | None = None) -> LegalEnv:
    docs = {
        "84-11_1984-06-09": _StubDocEntry("84-11_1984-06-09", range(1, 200)),
        "75-58_1975-09-26": _StubDocEntry("75-58_1975-09-26", range(1, 700)),
        "75-59_1975-09-26": _StubDocEntry("75-59_1975-09-26", range(1, 800)),
    }
    if extra_articles:
        for doc_id, nums in extra_articles.items():
            docs[doc_id] = _StubDocEntry(doc_id, nums)

    registry = _StubRegistry(docs)
    hits = [
        _StubHit("84-11_1984-06-09", "54", text="يجوز للزوجة دون موافقة الزوج أن تخالع نفسها بمقابل مالي"),
        _StubHit("75-58_1975-09-26", "81", text="يعيب الرضا الغلط والتدليس"),
    ]
    indices = {"bm25": _StubIndex(hits)}
    llm = _SequencedLLM(llm_responses)
    pool = _StubLLMPool(llm)
    budget = RecursionBudget(max_depth=1, max_sub_calls=5, max_total_tokens=200_000)
    return LegalEnv(registry=registry, kg=None, indices=indices,
                    llm_pool=pool, budget=budget)


_REQUIRED_KEYS = {
    "answer_text", "abstention", "abstention_reason",
    "citations", "reasoning_chain", "trajectory",
    "tokens_used", "depth_max_reached",
}


def _make_answer_block(
    answer_text="Answer.",
    abstention=False,
    abstention_reason=None,
    citations=None,
    reasoning_chain=None,
) -> str:
    obj = {
        "answer_text": answer_text,
        "abstention": abstention,
        "abstention_reason": abstention_reason,
        "citations": citations or [],
        "reasoning_chain": reasoning_chain or ["1.", "2.", "3.", "4."],
    }
    return f"```answer\n{json.dumps(obj, ensure_ascii=False)}\n```"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestRuleApplication:
    """Q1: Family Code khul' — expects citation to art.54."""

    def test_citation_present_and_valid(self):
        citation = {"doc_id": "84-11_1984-06-09", "article_ref": "54",
                    "supporting_span": "يجوز للزوجة", "confidence": 0.95}
        resp = (
            "```python\nresults = env.search_bm25('الخلع', k=3)\n"
            "print(results)\n```\n\n"
            + _make_answer_block(
                answer_text="Art.54 applies.",
                citations=[citation],
            )
        )
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        out = ctrl.run("ما هي شروط الخلع؟", query_type="rule_application")

        assert _REQUIRED_KEYS.issubset(out.keys())
        assert out["abstention"] is False
        assert len(out["citations"]) == 1
        # All citations must pass verify_in_corpus
        for c in out["citations"]:
            assert env.verify_in_corpus(c), f"Citation failed: {c}"

    def test_schema_complete(self):
        resp = _make_answer_block(answer_text="OK", citations=[])
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        out = ctrl.run("query", query_type="rule_application")
        assert isinstance(out["answer_text"], str)
        assert isinstance(out["abstention"], bool)
        assert isinstance(out["citations"], list)
        assert isinstance(out["reasoning_chain"], list)
        assert isinstance(out["trajectory"], list)
        assert isinstance(out["tokens_used"], int)
        assert isinstance(out["depth_max_reached"], int)


class TestAbstentionForeignJurisdiction:
    """Q2: ISF wealth-tax trap — must abstain with foreign_jurisdiction."""

    def test_abstention_on_isf_query(self):
        resp = _make_answer_block(
            answer_text="ISF does not exist in Algerian law.",
            abstention=True,
            abstention_reason="foreign_jurisdiction",
            citations=[],
        )
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        out = ctrl.run(
            "هل يخضع المواطن الجزائري لضريبة الثروة ISF؟",
            query_type="unanswerable",
        )
        assert out["abstention"] is True
        assert out["citations"] == []

    def test_classifier_detects_isf(self):
        from akn_rlm.rlm.root_controller import classify_query_type
        qt = classify_query_type("ضريبة ISF في القانون الجزائري")
        assert qt == "unanswerable"


class TestAbstentionOutOfCorpus:
    """Q3: Anti-Corruption Law 06-01 (collision) — must abstain."""

    def test_fictitious_citation_dropped(self):
        # LLM hallucinates a citation to a doc not in registry
        bad_citation = {"doc_id": "NONEXISTENT_99", "article_ref": "1",
                        "supporting_span": "...", "confidence": 0.9}
        resp = _make_answer_block(
            answer_text="Article 1 of fictitious law applies.",
            citations=[bad_citation],
        )
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        out = ctrl.run("ما هو حكم الفساد؟", query_type="rule_application")
        # The bad citation must be removed by the gate
        assert all(
            env.verify_in_corpus(c) for c in out["citations"]
        ), "Gate failed to remove fictitious citation"

    def test_all_citations_removed_triggers_no_crash(self):
        bad = {"doc_id": "FAKE_DOC", "article_ref": "99"}
        resp = _make_answer_block(citations=[bad])
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        out = ctrl.run("query", query_type="rule_application")
        assert isinstance(out, dict)
        assert out["citations"] == []


class TestMultiHop:
    """Q4: Civil + Commercial Code combined — expects at least one citation."""

    def test_multi_hop_returns_citations(self):
        citations = [
            {"doc_id": "75-58_1975-09-26", "article_ref": "81",
             "supporting_span": "يعيب الرضا", "confidence": 0.88},
        ]
        resp = _make_answer_block(
            answer_text="Both codes apply.",
            citations=citations,
        )
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        out = ctrl.run(
            "هل يمكن إبطال العقد التجاري بسبب التدليس؟",
            query_type="multi_hop",
        )
        assert len(out["citations"]) >= 1
        for c in out["citations"]:
            assert env.verify_in_corpus(c)

    def test_timeout_produces_valid_schema(self):
        """If LLM loops without answer, controller returns a valid schema."""
        # Response with no answer block — triggers timeout/no_answer path
        resp = "```python\nprint('thinking...')\n```"
        env = _make_env([resp])
        ctrl = RootController(
            env=env,
            few_shot=[],
            timeout_default=0.001,  # force immediate timeout
            max_rounds=1,
        )
        out = ctrl.run("complex query", query_type="rule_application")
        assert _REQUIRED_KEYS.issubset(out.keys())
        assert out["abstention"] is True


class TestTemporalQuery:
    """Q5: temporal question — answer returned, no crash."""

    def test_temporal_query_runs(self):
        resp = _make_answer_block(
            answer_text="Before the 2005 amendment...",
            citations=[{"doc_id": "84-11_1984-06-09", "article_ref": "54",
                        "supporting_span": "original text", "confidence": 0.7}],
        )
        env = _make_env([resp])
        ctrl = RootController(env=env, few_shot=[])
        out = ctrl.run(
            "ما كان نص المادة 54 من قانون الأسرة قبل تعديل 2005؟",
            query_type="temporal",
        )
        assert _REQUIRED_KEYS.issubset(out.keys())
        assert not isinstance(out["answer_text"], type(None))

    def test_classifier_detects_temporal(self):
        from akn_rlm.rlm.root_controller import classify_query_type
        qt = classify_query_type("ما كان نص المادة قبل التعديل؟")
        assert qt == "temporal"
