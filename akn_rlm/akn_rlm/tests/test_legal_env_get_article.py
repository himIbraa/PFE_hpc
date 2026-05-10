"""Tests for LegalEnv.get_article() returning full (untruncated) text."""
import pytest
from unittest.mock import MagicMock


def _make_env_with_bm25(article_text: str):
    from akn_rlm.rlm.legal_env import LegalEnv
    from akn_rlm.rlm.recursion_budget import RecursionBudget

    doc_id = "84-11_1984-06-09"
    article_ref = "40"
    eid = "art_40"
    chunk_id = f"{doc_id}#{eid}"

    # Mock registry
    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="قانون الأسرة", article_eids={eid})
    registry.has_article.return_value = True
    registry.resolve_alias.return_value = None

    # Mock BM25 index with the given article text
    bm25 = MagicMock()
    bm25._meta = [
        {"chunk_id": chunk_id, "doc_id": doc_id,
         "article_ref": article_ref, "text": article_text}
    ]

    env = LegalEnv(
        registry=registry,
        kg=None,
        indices={"bm25": bm25},
        llm_pool=MagicMock(),
        budget=RecursionBudget(),
    )
    return env, doc_id, article_ref


def test_get_article_returns_full_text():
    long_text = "أ" * 2000  # 2000-char Arabic text — previously truncated at 600
    env, doc_id, article_ref = _make_env_with_bm25(long_text)

    result = env.get_article(doc_id, article_ref)

    assert result is not None
    assert result["text"] == long_text, (
        f"Expected full 2000-char text, got {len(result['text'])} chars"
    )


def test_get_article_returns_correct_keys():
    env, doc_id, article_ref = _make_env_with_bm25("نص المادة")
    result = env.get_article(doc_id, article_ref)
    assert result is not None
    assert set(result.keys()) >= {"doc_id", "article_ref", "eid", "text", "doc_title"}


def test_get_article_none_for_missing():
    from akn_rlm.rlm.legal_env import LegalEnv
    from akn_rlm.rlm.recursion_budget import RecursionBudget

    registry = MagicMock()
    registry.get_doc.return_value = None
    registry.resolve_alias.return_value = None

    env = LegalEnv(
        registry=registry, kg=None, indices={},
        llm_pool=MagicMock(), budget=RecursionBudget(),
    )
    assert env.get_article("nonexistent_doc", "1") is None


def test_get_article_fallback_empty_text_when_not_in_bm25():
    from akn_rlm.rlm.legal_env import LegalEnv
    from akn_rlm.rlm.recursion_budget import RecursionBudget

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="Doc", article_eids={"art_99"})
    registry.has_article.return_value = True
    registry.resolve_alias.return_value = None

    bm25 = MagicMock()
    bm25._meta = []  # chunk not in BM25

    env = LegalEnv(
        registry=registry, kg=None, indices={"bm25": bm25},
        llm_pool=MagicMock(), budget=RecursionBudget(),
    )
    result = env.get_article("some_doc", "99")
    assert result is not None
    assert "text" in result  # may be empty string, but key must exist


def test_bm25_meta_stores_full_text():
    """BM25Index.build() must NOT truncate article text to 600 chars."""
    from akn_rlm.indexers.bm25 import BM25Index

    long_text = "ب" * 1500  # longer than old 600-char limit

    chunk = MagicMock()
    chunk.chunk_id = "doc1#art_1"
    chunk.doc_id = "doc1"
    chunk.article_ref = "1"
    chunk.text_norm = long_text
    chunk.text = long_text

    idx = BM25Index.build([chunk])
    stored_text = idx._meta[0]["text"]
    assert len(stored_text) == 1500, (
        f"Expected 1500 chars (no truncation), got {len(stored_text)}"
    )
