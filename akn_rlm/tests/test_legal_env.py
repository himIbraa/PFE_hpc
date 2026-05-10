"""Tests for LegalEnv — Phase C.

Uses a minimal stub setup so no actual model loading is needed.
"""
from __future__ import annotations

import pytest
from akn_rlm.rlm.legal_env import LegalEnv
from akn_rlm.rlm.recursion_budget import RecursionBudget, RecursionBudgetExceeded
from akn_rlm.gates.jurisdiction import detect as detect_infection


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _StubHit:
    def __init__(self, doc_id, article_ref, score=1.0):
        self.chunk_id    = f"{doc_id}#art_{article_ref}"
        self.doc_id      = doc_id
        self.article_ref = article_ref
        self.score       = score
        self.text        = f"text of {doc_id} art {article_ref}"


class _StubIndex:
    def __init__(self, hits):
        self._hits = hits
        # expose _meta for get_article()
        self._meta = [
            {"chunk_id": h.chunk_id, "doc_id": h.doc_id,
             "article_ref": h.article_ref, "text": h.text}
            for h in hits
        ]

    def search(self, query, k=10):
        return self._hits[:k]


class _StubRegistry:
    def __init__(self, docs: dict):
        self._docs = docs       # doc_id -> _StubDocEntry

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
        return self._docs.get(alias.lower(), None) and alias


class _StubDocEntry:
    def __init__(self, doc_id, article_numbers):
        self.doc_id    = doc_id
        self.doc_title = f"Doc {doc_id}"
        self.doc_date  = "1984-06-09"
        from akn_rlm.normalizers import ref_to_eid
        self.article_eids = {ref_to_eid(str(n)) for n in article_numbers}


class _StubLLMPool:
    def call(self, prompt, model, max_tokens, **kwargs) -> str:
        return f"[stub reply for purpose]"


def _make_env(
    articles: dict | None = None,
    bm25_hits=None,
    budget: RecursionBudget | None = None,
) -> LegalEnv:
    articles = articles or {
        "84-11_1984-06-09": list(range(1, 101)),  # articles 1-100
    }
    registry_docs = {
        doc_id: _StubDocEntry(doc_id, nums)
        for doc_id, nums in articles.items()
    }
    registry = _StubRegistry(registry_docs)
    bm25_hits = bm25_hits or [
        _StubHit("84-11_1984-06-09", "54"),
        _StubHit("84-11_1984-06-09", "55"),
    ]
    indices = {"bm25": _StubIndex(bm25_hits)}
    llm_pool = _StubLLMPool()
    budget = budget or RecursionBudget(max_depth=1, max_sub_calls=5)
    return LegalEnv(
        registry=registry,
        kg=None,
        indices=indices,
        llm_pool=llm_pool,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSearchPrimitives:
    def test_search_bm25_returns_list_of_dicts(self):
        env = _make_env()
        results = env.search_bm25("الخلع", k=5)
        assert isinstance(results, list)
        for r in results:
            assert "doc_id" in r
            assert "article_ref" in r
            assert "score" in r
            assert "text" in r
            assert "retriever" in r
            assert r["retriever"] == "bm25"

    def test_search_bm25_respects_k(self):
        hits = [_StubHit("84-11_1984-06-09", str(i)) for i in range(20)]
        env = _make_env(bm25_hits=hits)
        results = env.search_bm25("test", k=5)
        assert len(results) <= 5

    def test_search_dense_returns_empty_when_not_loaded(self):
        env = _make_env()   # dense not in indices
        results = env.search_dense("test", k=5)
        assert results == []

    def test_search_hybrid_returns_list(self):
        env = _make_env()
        results = env.search_hybrid("test", k=5)
        assert isinstance(results, list)


class TestHasArticle:
    def test_existing_article(self):
        env = _make_env()
        assert env.has_article("84-11_1984-06-09", "54") is True

    def test_existing_article_art_prefix(self):
        env = _make_env()
        assert env.has_article("84-11_1984-06-09", "art_54") is True

    def test_nonexistent_article_number(self):
        env = _make_env()
        assert env.has_article("84-11_1984-06-09", "999999") is False

    def test_nonexistent_doc(self):
        env = _make_env()
        assert env.has_article("00-00_0000-00-00", "1") is False


class TestVerifyInCorpus:
    def test_valid_citation(self):
        env = _make_env()
        assert env.verify_in_corpus({"doc_id": "84-11_1984-06-09", "article_ref": "54"}) is True

    def test_invalid_article(self):
        env = _make_env()
        assert env.verify_in_corpus({"doc_id": "84-11_1984-06-09", "article_ref": "999"}) is False

    def test_missing_doc_id(self):
        env = _make_env()
        assert env.verify_in_corpus({"article_ref": "54"}) is False

    def test_missing_article_ref(self):
        env = _make_env()
        assert env.verify_in_corpus({"doc_id": "84-11_1984-06-09"}) is False


class TestDetectInfectionSignals:
    def test_at_will_detected(self):
        env = _make_env()
        sigs = env.detect_infection_signals("The at-will employment doctrine")
        assert "us:at_will" in sigs

    def test_clean_arabic_text(self):
        env = _make_env()
        sigs = env.detect_infection_signals("يجوز للزوجة تخالع نفسها بمقابل مالي")
        assert sigs == []

    def test_isf_detected(self):
        env = _make_env()
        sigs = env.detect_infection_signals("L'ISF est applicable en France")
        assert "fr:isf" in sigs

    def test_wasiya_wajiba_detected(self):
        env = _make_env()
        sigs = env.detect_infection_signals("تطبيق نظام الوصية الواجبة في القانون المصري")
        assert "eg:wasiya_wajiba" in sigs

    def test_multiple_signals(self):
        env = _make_env()
        sigs = env.detect_infection_signals("at-will and punitive damages in US law")
        assert "us:at_will" in sigs
        assert "us:punitive_damages" in sigs

    def test_returns_list_of_strings(self):
        env = _make_env()
        result = env.detect_infection_signals("stare decisis doctrine")
        assert isinstance(result, list)
        for s in result:
            assert isinstance(s, str)


class TestLLMCall:
    def test_basic_call_returns_string(self):
        env = _make_env()
        result = env.llm_call("What is Article 54?", purpose="verify", max_tokens=100)
        assert isinstance(result, str)

    def test_budget_exceeded_raises(self):
        budget = RecursionBudget(max_depth=1, max_sub_calls=1)
        env = _make_env(budget=budget)
        env.llm_call("first call", max_tokens=50)  # uses up the 1 sub-call
        with pytest.raises(RecursionBudgetExceeded):
            env.llm_call("second call", max_tokens=50)

    def test_depth_exceeded_raises(self):
        budget = RecursionBudget(max_depth=0, max_sub_calls=5)
        env = _make_env(budget=budget)
        with pytest.raises(RecursionBudgetExceeded):
            env.llm_call("any prompt", max_tokens=50)


class TestListDocuments:
    def test_returns_list_of_dicts(self):
        env = _make_env()
        docs = env.list_documents()
        assert isinstance(docs, list)
        assert len(docs) >= 1
        for d in docs:
            assert "doc_id" in d
            assert "article_count" in d

    def test_known_doc_present(self):
        env = _make_env()
        doc_ids = [d["doc_id"] for d in env.list_documents()]
        assert "84-11_1984-06-09" in doc_ids
