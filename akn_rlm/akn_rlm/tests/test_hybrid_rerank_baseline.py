"""Tests for the hybrid + cross-encoder rerank baseline pipeline.

The pipeline must mirror the RLM ``run(query) -> dict`` contract so
``akn_rlm.eval.runner._answer_to_result`` can consume it unchanged. The
candidate pool fed to the reranker is exactly the RRF-fused
``(doc_id, canonical article_ref)`` pool from
:mod:`akn_rlm.baselines.hybrid_pipeline`, so the comparison versus B3 is
purely a measurement of the cross-encoder's contribution at the top of
the ranking — not a different retrieval recipe.

The reranker is injected so these tests don't load
``sentence-transformers`` at all.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.baselines.hybrid_rerank_pipeline import (
    DEFAULT_K_EACH,
    DEFAULT_RERANK_POOL_SIZE,
    DEFAULT_TOP_K,
    SUPPORT_SPAN_LEN,
    HybridRerankBaselinePipeline,
    build_hybrid_rerank_pipeline,
)
from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.indexers.dense import DenseHit


def _bm25(doc_id: str, ref: str, text: str, score: float = 1.0) -> BM25Hit:
    return BM25Hit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _dense(doc_id: str, ref: str, text: str, score: float = 0.5) -> DenseHit:
    return DenseHit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _identity_reranker(query: str, candidates: list[dict], k: int) -> list[dict]:
    """Simple stand-in that adds rerank_score=score and truncates to k."""
    out = []
    for c in candidates[:k]:
        entry = dict(c)
        entry["rerank_score"] = float(c.get("score", 0.0))
        out.append(entry)
    return out


def _make_pipeline(
    bm25_hits: list[BM25Hit],
    dense_hits: list[DenseHit],
    *,
    reranker=None,
    doc_title: str = "قانون الأسرة",
    top_k: int = DEFAULT_TOP_K,
    k_each: int = DEFAULT_K_EACH,
    rerank_pool_size: int = DEFAULT_RERANK_POOL_SIZE,
):
    bm25 = MagicMock()
    bm25.search.return_value = bm25_hits

    dense = MagicMock()
    dense.search.return_value = dense_hits

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)

    if reranker is None:
        reranker = MagicMock(side_effect=_identity_reranker)

    pipeline = build_hybrid_rerank_pipeline(
        bm25=bm25, dense=dense, registry=registry,
        reranker=reranker,
        top_k=top_k, k_each=k_each, rerank_pool_size=rerank_pool_size,
    )
    return pipeline, bm25, dense, registry, reranker


# ---------------------------------------------------------------------------
# Contract — answer dict shape
# ---------------------------------------------------------------------------

def test_run_returns_required_keys():
    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", "نص المادة")],
        [_dense("84-11_1984-06-09", "5", "نص المادة")],
    )
    out = pipeline.run("ما هي شروط الزواج؟")
    required = {
        "answer_text", "abstention", "abstention_reason", "citations",
        "reasoning_chain", "trajectory", "tokens_used", "_telemetry",
    }
    assert required.issubset(out.keys())


def test_telemetry_baseline_is_hybrid_rerank():
    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", "نص")], [],
    )
    out = pipeline.run("سؤال")
    assert out["_telemetry"]["baseline"] == "hybrid_rerank"


def test_citation_confidence_is_rerank_score_when_present():
    """When the reranker tags candidates with rerank_score, that value is
    surfaced as `confidence` (not the upstream RRF score)."""
    text = "نص قانوني" * 50

    def reranker(q, cands, k):
        # Promote the same single candidate with a fixed cross-encoder score.
        out = [dict(cands[0])]
        out[0]["rerank_score"] = 7.77
        return out

    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", text, score=12.3)],
        [_dense("84-11_1984-06-09", "5", text, score=0.83)],
        reranker=reranker,
    )
    out = pipeline.run("ما هي المادة 5؟")
    assert len(out["citations"]) == 1
    c = out["citations"][0]
    assert c["doc_id"] == "84-11_1984-06-09"
    assert c["article_ref"] == "5"
    assert c["confidence"] == 7.77
    assert c["supporting_span"] == text[:SUPPORT_SPAN_LEN]
    assert c["text"] == text


def test_citation_confidence_falls_back_to_rrf_score_when_reranker_returns_no_score():
    """If the reranker is degraded and just passes candidates back without
    annotating ``rerank_score``, citations still carry the RRF score."""

    def passthrough(q, cands, k):
        return [dict(c) for c in cands[:k]]  # no rerank_score added

    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", "نص", score=12.3)],
        [],
        reranker=passthrough,
    )
    out = pipeline.run("سؤال")
    assert out["citations"][0]["confidence"] > 0.0


# ---------------------------------------------------------------------------
# Retrieval / reranking behaviour
# ---------------------------------------------------------------------------

def test_default_top_k_k_each_and_pool_size():
    assert DEFAULT_TOP_K == 5
    assert DEFAULT_K_EACH == 20
    assert DEFAULT_RERANK_POOL_SIZE == 50


def test_calls_each_retriever_with_k_each():
    pipeline, bm25, dense, _, _ = _make_pipeline([], [])
    pipeline.run("q")
    bm25.search.assert_called_once_with("q", k=DEFAULT_K_EACH)
    dense.search.assert_called_once_with("q", k=DEFAULT_K_EACH)


def test_passes_through_custom_k_each_to_both_retrievers():
    pipeline, bm25, dense, _, _ = _make_pipeline([], [], k_each=33)
    pipeline.run("q")
    bm25.search.assert_called_once_with("q", k=33)
    dense.search.assert_called_once_with("q", k=33)


def test_reranker_receives_fused_candidates_and_query():
    """The reranker must be called with the fused-pool dicts and the query."""
    bm25_hits = [_bm25("doc_a", "1", "t1", score=10.0)]
    dense_hits = [_dense("doc_b", "2", "t2", score=0.9)]

    captured: dict = {}

    def capture(q, cands, k):
        captured["query"] = q
        captured["cands"] = cands
        captured["k"] = k
        return _identity_reranker(q, cands, k)

    pipeline, *_ = _make_pipeline(bm25_hits, dense_hits, reranker=capture)
    pipeline.run("ما هذا؟")

    assert captured["query"] == "ما هذا؟"
    assert captured["k"] == DEFAULT_TOP_K
    assert {c["doc_id"] for c in captured["cands"]} == {"doc_a", "doc_b"}
    # Pool entries are the post-fusion dicts, so they carry rrf scores.
    assert all("score" in c and "doc_id" in c and "article_ref" in c for c in captured["cands"])


def test_reranker_pool_capped_at_rerank_pool_size():
    """Many fused candidates → reranker only sees ``rerank_pool_size`` of them."""
    bm25_hits = [_bm25("doc_a", str(i), f"t{i}", score=20.0 - i) for i in range(20)]
    dense_hits = [_dense("doc_b", str(i), f"u{i}", score=1.0 - 0.01 * i) for i in range(20)]

    captured: dict = {}

    def capture(q, cands, k):
        captured["count"] = len(cands)
        return _identity_reranker(q, cands, k)

    pipeline, *_ = _make_pipeline(
        bm25_hits, dense_hits, reranker=capture,
        k_each=20, rerank_pool_size=8,
    )
    pipeline.run("q")
    assert captured["count"] == 8


def test_truncates_to_top_k_after_rerank():
    """Reranker may return more than top_k — pipeline still cuts to top_k."""
    def overshoot(q, cands, k):
        # Return all (>k) with rerank_score; pipeline must truncate.
        out = []
        for i, c in enumerate(cands):
            entry = dict(c)
            entry["rerank_score"] = float(len(cands) - i)
            out.append(entry)
        return out

    bm25_hits = [_bm25("doc_a", str(i), f"t{i}", score=10 - i) for i in range(5)]
    dense_hits = [_dense("doc_b", str(i), f"u{i}", score=1.0 - 0.1 * i) for i in range(5)]
    pipeline, *_ = _make_pipeline(bm25_hits, dense_hits, reranker=overshoot, top_k=3)
    out = pipeline.run("q")
    assert len(out["citations"]) == 3


def test_dedupes_repeated_doc_article_pairs():
    """Reranker returns multiple chunks of the same article → collapse to one citation."""

    def keep_all(q, cands, k):
        return [dict(c) | {"rerank_score": 1.0} for c in cands]

    pipeline, *_ = _make_pipeline(
        [
            _bm25("84-11_1984-06-09", "5", "para 1"),
            _bm25("84-11_1984-06-09", "5", "para 2"),
            _bm25("84-11_1984-06-09", "7", "art 7 text"),
        ],
        [
            _dense("84-11_1984-06-09", "5", "dense para"),
        ],
        reranker=keep_all,
    )
    out = pipeline.run("سؤال")
    keys = {(c["doc_id"], c["article_ref"]) for c in out["citations"]}
    assert keys == {
        ("84-11_1984-06-09", "5"),
        ("84-11_1984-06-09", "7"),
    }


def test_canonicalises_article_ref_in_citation():
    """Arabic ordinals / bis variants must be normalised in the citation."""
    pipeline, *_ = _make_pipeline(
        [_bm25("75-58_1975-09-26", "الأولى", "art 1 text")],
        [_dense("84-11_1984-06-09", "9 مكرر", "art 9 bis")],
    )
    out = pipeline.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"1", "9_bis"}


def test_fusion_key_is_canonical_doc_id_plus_article_ref():
    """A BM25 hit on `9 مكرر` and a Dense hit on `9_bis` must collapse to one
    fused entry before the reranker is even called — same contract as B3."""

    captured: dict = {}

    def capture(q, cands, k):
        captured["count"] = len(cands)
        return _identity_reranker(q, cands, k)

    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "9 مكرر", "Arabic surface text")],
        [_dense("84-11_1984-06-09", "9_bis", "snake_case surface text")],
        reranker=capture,
    )
    out = pipeline.run("سؤال")
    assert captured["count"] == 1  # fusion already collapsed before rerank
    assert len(out["citations"]) == 1
    c = out["citations"][0]
    assert c["doc_id"] == "84-11_1984-06-09"
    assert c["article_ref"] == "9_bis"


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------

def test_empty_query_abstains():
    pipeline, *_ = _make_pipeline([], [])
    out = pipeline.run("   ")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    assert out["citations"] == []
    assert out["answer_text"] == ""


def test_no_hits_abstains_when_both_retrievers_empty():
    pipeline, *_ = _make_pipeline([], [])
    out = pipeline.run("سؤال غامض")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"


def test_no_hits_abstains_when_reranker_returns_empty():
    """Reranker returning [] (e.g. all candidates filtered) → no_hits."""

    def empty(q, cands, k):
        return []

    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", "نص")],
        [],
        reranker=empty,
    )
    out = pipeline.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"


def test_one_retriever_empty_other_full_still_returns_citations():
    """Single-arm hits still flow through the reranker and produce citations."""
    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", "نص")],
        [],
    )
    out = pipeline.run("سؤال")
    assert out["abstention"] is False
    assert len(out["citations"]) == 1
    assert out["citations"][0]["doc_id"] == "84-11_1984-06-09"


# ---------------------------------------------------------------------------
# Template answer
# ---------------------------------------------------------------------------

def test_template_answer_uses_doc_title_and_ref():
    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", "نص المادة")],
        [_dense("84-11_1984-06-09", "5", "نص المادة")],
        doc_title="قانون الأسرة",
    )
    out = pipeline.run("سؤال")
    assert "قانون الأسرة" in out["answer_text"]
    assert "المادة 5" in out["answer_text"]
    assert "نص المادة" in out["answer_text"]


def test_template_answer_falls_back_to_doc_id_when_no_title():
    bm25 = MagicMock()
    bm25.search.return_value = [_bm25("xx_yy", "1", "نص")]
    dense = MagicMock()
    dense.search.return_value = []
    registry = MagicMock()
    registry.get_doc.return_value = None  # unknown doc

    pipeline = HybridRerankBaselinePipeline(
        bm25=bm25, dense=dense, registry=registry,
        reranker=_identity_reranker,
    )
    out = pipeline.run("سؤال")
    assert "xx_yy" in out["answer_text"]


# ---------------------------------------------------------------------------
# Compatibility with the eval runner's _answer_to_result
# ---------------------------------------------------------------------------

def test_answer_to_result_consumes_output_without_branching():
    from akn_rlm.eval.runner import _answer_to_result

    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", "نص المادة")],
        [_dense("84-11_1984-06-09", "5", "نص المادة")],
    )
    answer = pipeline.run("ما هي المادة 5؟")
    answer["_latency_s"] = 0.01

    question = {
        "id": "fam_test_q01",
        "query": "ما هي المادة 5؟",
        "query_type": "exact_article",
        "legal_category": "family_law",
        "difficulty": "easy",
        "language": "ar",
        "split": "test",
        "gold_doc_ids": ["84-11_1984-06-09"],
        "gold_article_ids": ["84-11_1984-06-09#art_5"],
        "gold_citations": [{"doc_id": "84-11_1984-06-09", "article_ref": "5"}],
        "gold_abstain": False,
        "gold_answer": "",
        "gold_reasoning_chain": [],
    }
    result = _answer_to_result(question, answer)

    assert result["pred_doc_ids"] == ["84-11_1984-06-09"]
    assert result["pred_article_ids"] == ["84-11_1984-06-09#art_5"]
    assert result["predicted_abstain"] is False
    # Deterministic baseline → no rejected citations → HCR=0
    assert result["hcr"] == 0.0
