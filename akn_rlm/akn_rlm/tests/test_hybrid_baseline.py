"""Tests for the hybrid (RRF over BM25 + Dense) baseline pipeline.

The pipeline must mirror the RLM .run(query) -> dict contract so that
``akn_rlm.eval.runner._answer_to_result`` can consume its output unchanged.
The fusion key is intentionally ``(doc_id, canonical article_ref)`` —
matching ``LegalEnv.search_hybrid`` and the rerank baseline that will
operate on the same fused candidate pool.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.baselines.hybrid_pipeline import (
    DEFAULT_K_EACH,
    DEFAULT_TOP_K,
    SUPPORT_SPAN_LEN,
    HybridBaselinePipeline,
    build_hybrid_pipeline,
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


def _make_pipeline(
    bm25_hits: list[BM25Hit],
    dense_hits: list[DenseHit],
    *,
    doc_title: str = "قانون الأسرة",
    top_k: int = DEFAULT_TOP_K,
    k_each: int = DEFAULT_K_EACH,
):
    bm25 = MagicMock()
    bm25.search.return_value = bm25_hits

    dense = MagicMock()
    dense.search.return_value = dense_hits

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)

    pipeline = build_hybrid_pipeline(
        bm25=bm25, dense=dense, registry=registry,
        top_k=top_k, k_each=k_each,
    )
    return pipeline, bm25, dense, registry


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


def test_telemetry_baseline_is_hybrid():
    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", "نص")], [],
    )
    out = pipeline.run("سؤال")
    assert out["_telemetry"]["baseline"] == "hybrid"


def test_citation_shape_carries_supporting_span_and_confidence():
    text = "نص قانوني" * 50
    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", text, score=12.3)],
        [_dense("84-11_1984-06-09", "5", text, score=0.83)],
    )
    out = pipeline.run("ما هي المادة 5؟")
    assert len(out["citations"]) == 1
    c = out["citations"][0]
    assert c["doc_id"] == "84-11_1984-06-09"
    assert c["article_ref"] == "5"
    # confidence is the RRF score (sum of 1/(k+rank)), not a raw retriever score
    assert c["confidence"] > 0.0
    assert c["supporting_span"] == text[:SUPPORT_SPAN_LEN]
    assert c["text"] == text


# ---------------------------------------------------------------------------
# Retrieval behaviour
# ---------------------------------------------------------------------------

def test_default_top_k_is_5_and_k_each_is_20():
    assert DEFAULT_TOP_K == 5
    assert DEFAULT_K_EACH == 20


def test_calls_each_retriever_with_k_each():
    """Both retrievers must be queried at the deeper k_each, not top_k."""
    pipeline, bm25, dense, _ = _make_pipeline([], [])
    pipeline.run("q")
    bm25.search.assert_called_once_with("q", k=DEFAULT_K_EACH)
    dense.search.assert_called_once_with("q", k=DEFAULT_K_EACH)


def test_passes_through_custom_k_each_to_both_retrievers():
    pipeline, bm25, dense, _ = _make_pipeline([], [], k_each=33)
    pipeline.run("q")
    bm25.search.assert_called_once_with("q", k=33)
    dense.search.assert_called_once_with("q", k=33)


def test_truncates_to_top_k_after_fusion():
    """Eight distinct (doc_id, ref) hits with top_k=3 → only 3 citations."""
    bm25_hits = [_bm25("doc_a", str(i), f"t{i}", score=10 - i) for i in range(5)]
    dense_hits = [_dense("doc_b", str(i), f"u{i}", score=1.0 - 0.1 * i) for i in range(5)]
    pipeline, *_ = _make_pipeline(bm25_hits, dense_hits, top_k=3)
    out = pipeline.run("q")
    assert len(out["citations"]) == 3


def test_dedupes_repeated_doc_article_pairs():
    """Different chunks of the same article must collapse to one citation."""
    pipeline, *_ = _make_pipeline(
        [
            _bm25("84-11_1984-06-09", "5", "para 1"),
            _bm25("84-11_1984-06-09", "5", "para 2"),
            _bm25("84-11_1984-06-09", "7", "art 7 text"),
        ],
        [
            _dense("84-11_1984-06-09", "5", "dense para"),
        ],
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
    """A BM25 hit on `9 مكرر` and a Dense hit on `9_bis` must collapse.

    This documents the fusion-key choice for the rerank baseline (B4): we
    fuse on (doc_id, canonical article_ref), NOT chunk_id, so different
    surface forms of the same article from different retrievers merge into
    a single fused entry whose RRF score is the sum of both reciprocal
    ranks.
    """
    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "9 مكرر", "Arabic surface text")],
        [_dense("84-11_1984-06-09", "9_bis", "snake_case surface text")],
    )
    out = pipeline.run("سؤال")
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
    assert out["answer_text"] == ""


def test_one_retriever_empty_other_full_still_returns_citations():
    """If only one arm has hits, the pipeline still answers from that arm."""
    pipeline, *_ = _make_pipeline(
        [_bm25("84-11_1984-06-09", "5", "نص")],
        [],  # dense empty (e.g. encoder failure or zero recall)
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
    pipeline = HybridBaselinePipeline(bm25=bm25, dense=dense, registry=registry)
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
