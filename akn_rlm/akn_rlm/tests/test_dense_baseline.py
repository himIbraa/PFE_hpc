"""Tests for the dense (FAISS + e5-small) baseline pipeline.

The pipeline must mirror the RLM .run(query) -> dict contract so that
``akn_rlm.eval.runner._answer_to_result`` can consume its output unchanged.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.baselines.dense_pipeline import (
    DEFAULT_TOP_K,
    SUPPORT_SPAN_LEN,
    DenseBaselinePipeline,
    build_dense_pipeline,
)
from akn_rlm.indexers.dense import DenseHit


def _make_pipeline(hits: list[DenseHit], doc_title: str = "قانون الأسرة"):
    dense = MagicMock()
    dense.search.return_value = hits

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)

    return build_dense_pipeline(dense=dense, registry=registry), dense, registry


def _hit(doc_id: str, ref: str, text: str, score: float = 0.5) -> DenseHit:
    return DenseHit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


# ---------------------------------------------------------------------------
# Contract — answer dict shape
# ---------------------------------------------------------------------------

def test_run_returns_required_keys():
    pipeline, _, _ = _make_pipeline([_hit("84-11_1984-06-09", "5", "نص المادة")])
    out = pipeline.run("ما هي شروط الزواج؟")
    required = {
        "answer_text", "abstention", "abstention_reason", "citations",
        "reasoning_chain", "trajectory", "tokens_used", "_telemetry",
    }
    assert required.issubset(out.keys())


def test_telemetry_baseline_is_dense():
    pipeline, _, _ = _make_pipeline([_hit("84-11_1984-06-09", "5", "نص")])
    out = pipeline.run("سؤال")
    assert out["_telemetry"]["baseline"] == "dense"


def test_citation_shape_carries_supporting_span_and_confidence():
    text = "نص قانوني" * 50
    pipeline, _, _ = _make_pipeline([_hit("84-11_1984-06-09", "5", text, score=0.83)])
    out = pipeline.run("ما هي المادة 5؟")
    assert len(out["citations"]) == 1
    c = out["citations"][0]
    assert c["doc_id"] == "84-11_1984-06-09"
    assert c["article_ref"] == "5"
    assert c["confidence"] == 0.83
    assert c["supporting_span"] == text[:SUPPORT_SPAN_LEN]
    assert c["text"] == text  # full text retained alongside the span


# ---------------------------------------------------------------------------
# Retrieval behaviour
# ---------------------------------------------------------------------------

def test_default_top_k_is_5():
    assert DEFAULT_TOP_K == 5
    dense = MagicMock()
    dense.search.return_value = []
    registry = MagicMock()
    DenseBaselinePipeline(dense=dense, registry=registry).run("q")
    dense.search.assert_called_once_with("q", k=5)


def test_passes_through_top_k_to_dense():
    dense = MagicMock()
    dense.search.return_value = []
    registry = MagicMock()
    DenseBaselinePipeline(dense=dense, registry=registry, top_k=12).run("q")
    dense.search.assert_called_once_with("q", k=12)


def test_dedupes_repeated_doc_article_pairs():
    """Different chunks of the same article must collapse to one citation."""
    pipeline, _, _ = _make_pipeline([
        _hit("84-11_1984-06-09", "5", "para 1"),
        _hit("84-11_1984-06-09", "5", "para 2"),
        _hit("84-11_1984-06-09", "7", "art 7 text"),
    ])
    out = pipeline.run("سؤال")
    assert len(out["citations"]) == 2
    keys = {(c["doc_id"], c["article_ref"]) for c in out["citations"]}
    assert keys == {
        ("84-11_1984-06-09", "5"),
        ("84-11_1984-06-09", "7"),
    }


def test_canonicalises_article_ref_in_citation():
    """Arabic ordinals / bis variants must be normalised in the citation."""
    pipeline, _, _ = _make_pipeline([
        _hit("75-58_1975-09-26", "الأولى", "art 1 text"),
        _hit("84-11_1984-06-09", "9 مكرر", "art 9 bis"),
    ])
    out = pipeline.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"1", "9_bis"}


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------

def test_empty_query_abstains():
    pipeline, _, _ = _make_pipeline([])
    out = pipeline.run("   ")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    assert out["citations"] == []
    assert out["answer_text"] == ""


def test_no_hits_abstains():
    pipeline, _, _ = _make_pipeline([])
    out = pipeline.run("سؤال غامض")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"
    assert out["answer_text"] == ""


# ---------------------------------------------------------------------------
# Template answer
# ---------------------------------------------------------------------------

def test_template_answer_uses_doc_title_and_ref():
    pipeline, _, _ = _make_pipeline(
        [_hit("84-11_1984-06-09", "5", "نص المادة")],
        doc_title="قانون الأسرة",
    )
    out = pipeline.run("سؤال")
    assert "قانون الأسرة" in out["answer_text"]
    assert "المادة 5" in out["answer_text"]
    assert "نص المادة" in out["answer_text"]


def test_template_answer_falls_back_to_doc_id_when_no_title():
    dense = MagicMock()
    dense.search.return_value = [_hit("xx_yy", "1", "نص")]
    registry = MagicMock()
    registry.get_doc.return_value = None  # unknown doc
    pipeline = DenseBaselinePipeline(dense=dense, registry=registry)
    out = pipeline.run("سؤال")
    assert "xx_yy" in out["answer_text"]


# ---------------------------------------------------------------------------
# Compatibility with the eval runner's _answer_to_result
# ---------------------------------------------------------------------------

def test_answer_to_result_consumes_output_without_branching():
    from akn_rlm.eval.runner import _answer_to_result

    pipeline, _, _ = _make_pipeline([_hit("84-11_1984-06-09", "5", "نص المادة")])
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
