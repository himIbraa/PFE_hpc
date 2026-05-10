"""Tests for degraded-retrieval telemetry in LegalEnv.search_hybrid()."""
import pytest
from unittest.mock import MagicMock, patch


def _make_env(active_indices: list[str]):
    from akn_rlm.rlm.legal_env import LegalEnv
    from akn_rlm.rlm.recursion_budget import RecursionBudget

    indices = {}
    for name in active_indices:
        mock_idx = MagicMock()
        mock_idx.search.return_value = [
            MagicMock(chunk_id=f"{name}_1", doc_id="doc1", article_ref="1",
                      score=1.0, text="text")
        ]
        indices[name] = mock_idx

    env = LegalEnv(
        registry=MagicMock(),
        kg=None,
        indices=indices,
        llm_pool=MagicMock(),
        budget=RecursionBudget(),
    )
    env._degraded_retrievers = []
    return env


def test_no_degradation_when_all_present():
    env = _make_env(["bm25", "dense", "splade", "colbert"])
    with patch("akn_rlm.retrievers.hybrid_fusion.rrf_fuse", return_value=[]), \
         patch.object(env, "rerank", return_value=[]):
        env.search_hybrid("test query", k=5)
    assert env._degraded_retrievers == []


def test_missing_arms_recorded():
    env = _make_env(["bm25"])  # only BM25, dense/splade/colbert missing
    with patch("akn_rlm.retrievers.hybrid_fusion.rrf_fuse", return_value=[]), \
         patch.object(env, "rerank", return_value=[]):
        env.search_hybrid("test query", k=5)
    assert "dense" in env._degraded_retrievers
    assert "splade" in env._degraded_retrievers
    assert "colbert" in env._degraded_retrievers
    assert "bm25" not in env._degraded_retrievers


def test_degraded_retrievers_reset_on_begin_query():
    env = _make_env(["bm25"])
    env._degraded_retrievers = ["dense", "splade"]
    env.begin_query(clear_cache=True)
    assert env._degraded_retrievers == []


def test_failed_retriever_also_degraded():
    env = _make_env(["bm25"])
    # Make BM25 raise an exception during hybrid search
    env.indices["bm25"].search.side_effect = RuntimeError("index corrupt")
    with patch("akn_rlm.retrievers.hybrid_fusion.rrf_fuse", return_value=[]):
        result = env.search_hybrid("test query", k=5)
    assert "bm25" in env._degraded_retrievers
    assert result == []


def test_bm25_only_returns_results():
    env = _make_env(["bm25"])
    from akn_rlm.indexers.bm25 import BM25Hit
    hit = BM25Hit(chunk_id="doc1#art_1", doc_id="doc1",
                  article_ref="1", score=1.0, text="some text")
    env.indices["bm25"].search.return_value = [hit]

    fused_hits = [{"chunk_id": "doc1#art_1", "doc_id": "doc1",
                   "article_ref": "1", "score": 1.0, "text": "some text",
                   "retriever": "bm25"}]

    with patch("akn_rlm.retrievers.hybrid_fusion.rrf_fuse", return_value=fused_hits), \
         patch.object(env, "rerank", return_value=fused_hits[:5]):
        results = env.search_hybrid("test query", k=5)

    assert len(results) > 0
