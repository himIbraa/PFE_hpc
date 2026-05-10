"""Tests for the KG-augmented hybrid baseline pipeline (B6).

The pipeline mirrors the ``run(query) -> dict`` contract used by B1-B5 and
the RLM pipeline, so ``akn_rlm.eval.runner._answer_to_result`` consumes its
output without branching. Both the SPARQL function and the BM25/Dense
retrievers are mocked — these tests do not load the on-disk TTL or any
real index.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from akn_rlm.baselines.kg_hybrid_pipeline import (
    DEFAULT_EXPANSION_TERMS,
    DEFAULT_K_EACH,
    DEFAULT_KG_BOOST,
    DEFAULT_TOP_K,
    KGHybridBaselinePipeline,
    SUPPORT_SPAN_LEN,
    build_kg_hybrid_pipeline,
)
from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.indexers.dense import DenseHit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _row(uri: str, text: str = "") -> dict[str, Any]:
    """Mimic the dict shape returned by ``graphrag.sparql_query``."""
    return {"article": uri, "text": text}


def _make_pipeline(
    bm25_hits: list[BM25Hit] | None = None,
    dense_hits: list[DenseHit] | None = None,
    sparql_rows_per_token: dict[str, list[dict[str, Any]]] | None = None,
    *,
    doc_title: str = "قانون الأسرة",
    top_k: int = DEFAULT_TOP_K,
    k_each: int = DEFAULT_K_EACH,
    expansion_terms_max: int = DEFAULT_EXPANSION_TERMS,
    kg_boost: float = DEFAULT_KG_BOOST,
    alias_map: dict[str, str] | None = None,
):
    """Build a KGHybridBaselinePipeline with all I/O mocked.

    ``sparql_rows_per_token`` maps the SAFE token literal (single-quote-
    escaped, backslashes stripped) to the rows the SPARQL call should
    yield — same convention as ``test_kg_baseline.py``.
    """
    bm25_hits = bm25_hits or []
    dense_hits = dense_hits or []
    sparql_rows_per_token = sparql_rows_per_token or {}

    bm25 = MagicMock()
    bm25.search.return_value = bm25_hits
    dense = MagicMock()
    dense.search.return_value = dense_hits

    sparql_calls: list[tuple[Any, str]] = []

    def fake_sparql(kg, query: str) -> list[dict[str, Any]]:
        sparql_calls.append((kg, query))
        import re
        m = re.search(r'CONTAINS\(\?text, "([^"]*)"\)', query)
        if not m:
            return []
        token = m.group(1)
        return list(sparql_rows_per_token.get(token, []))

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)
    alias_map = alias_map or {}
    registry.resolve_alias.side_effect = lambda key: alias_map.get(key)

    kg = MagicMock(name="rdflib.Graph")

    pipeline = build_kg_hybrid_pipeline(
        kg=kg,
        bm25=bm25,
        dense=dense,
        registry=registry,
        top_k=top_k,
        k_each=k_each,
        expansion_terms_max=expansion_terms_max,
        kg_boost=kg_boost,
        sparql_fn=fake_sparql,
    )
    return pipeline, bm25, dense, registry, sparql_calls


# ---------------------------------------------------------------------------
# Contract — answer dict shape
# ---------------------------------------------------------------------------

def test_run_returns_required_keys():
    pipeline, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[_dense("84-11_1984-06-09", "5", "نص")],
    )
    out = pipeline.run("ما هي شروط زواج")
    required = {
        "answer_text", "abstention", "abstention_reason", "citations",
        "reasoning_chain", "trajectory", "tokens_used", "_telemetry",
    }
    assert required.issubset(out.keys())


def test_telemetry_baseline_is_kg_hybrid():
    pipeline, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
    )
    out = pipeline.run("سؤال")
    assert out["_telemetry"]["baseline"] == "kg_hybrid"


def test_default_top_k_and_k_each():
    assert DEFAULT_TOP_K == 5
    assert DEFAULT_K_EACH == 20


# ---------------------------------------------------------------------------
# Citation shape
# ---------------------------------------------------------------------------

def test_citation_carries_supporting_span_text_and_confidence():
    text = "نص قانوني" * 50
    pipeline, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", text, score=12.3)],
        dense_hits=[_dense("84-11_1984-06-09", "5", text, score=0.83)],
    )
    out = pipeline.run("ما هي المادة")
    assert len(out["citations"]) == 1
    c = out["citations"][0]
    assert c["doc_id"] == "84-11_1984-06-09"
    assert c["article_ref"] == "5"
    assert c["text"] == text
    assert c["supporting_span"] == text[:SUPPORT_SPAN_LEN]
    assert c["confidence"] > 0.0
    # No KG hit → kg_boosted flag is false on the citation.
    assert c["kg_boosted"] is False


# ---------------------------------------------------------------------------
# Retrieval behaviour
# ---------------------------------------------------------------------------

def test_calls_each_retriever_with_k_each():
    pipeline, bm25, dense, *_ = _make_pipeline()
    pipeline.run("بعض السؤال")
    assert bm25.search.call_args.kwargs.get("k") == DEFAULT_K_EACH
    assert dense.search.call_args.kwargs.get("k") == DEFAULT_K_EACH


def test_passes_through_custom_k_each_to_both_retrievers():
    pipeline, bm25, dense, *_ = _make_pipeline(k_each=33)
    pipeline.run("سؤال")
    assert bm25.search.call_args.kwargs.get("k") == 33
    assert dense.search.call_args.kwargs.get("k") == 33


def test_truncates_to_top_k_after_fusion():
    bm25_hits = [_bm25("doc_a", str(i), f"t{i}", score=10 - i) for i in range(5)]
    dense_hits = [_dense("doc_b", str(i), f"u{i}", score=1.0 - 0.1 * i) for i in range(5)]
    pipeline, *_ = _make_pipeline(
        bm25_hits=bm25_hits, dense_hits=dense_hits, top_k=3,
    )
    out = pipeline.run("سؤال")
    assert len(out["citations"]) == 3


def test_dedupes_repeated_doc_article_pairs():
    pipeline, *_ = _make_pipeline(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "5", "para 1"),
            _bm25("84-11_1984-06-09", "5", "para 2"),
            _bm25("84-11_1984-06-09", "7", "art 7 text"),
        ],
        dense_hits=[_dense("84-11_1984-06-09", "5", "dense para")],
    )
    out = pipeline.run("سؤال")
    keys = {(c["doc_id"], c["article_ref"]) for c in out["citations"]}
    assert keys == {
        ("84-11_1984-06-09", "5"),
        ("84-11_1984-06-09", "7"),
    }


def test_canonicalises_article_ref_in_citation():
    pipeline, *_ = _make_pipeline(
        bm25_hits=[_bm25("75-58_1975-09-26", "الأولى", "art 1 text")],
        dense_hits=[_dense("84-11_1984-06-09", "9 مكرر", "art 9 bis")],
    )
    out = pipeline.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"1", "9_bis"}


def test_fusion_collapses_surface_variants_before_kg_boost():
    """BM25 ``9 مكرر`` and Dense ``9_bis`` collapse on canonical key.

    This is the same fusion-key contract as B3/B4. The KG boost step
    operates on the post-fusion canonical key, so a KG hit on ``9_bis``
    can boost both surface variants in a single shot.
    """
    pipeline, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "9 مكرر", "Arabic surface")],
        dense_hits=[_dense("84-11_1984-06-09", "9_bis", "snake_case surface")],
    )
    out = pipeline.run("سؤال")
    assert len(out["citations"]) == 1
    c = out["citations"][0]
    assert c["doc_id"] == "84-11_1984-06-09"
    assert c["article_ref"] == "9_bis"


# ---------------------------------------------------------------------------
# Query expansion — the KG-derived surface labels are appended to the query
# ---------------------------------------------------------------------------

def test_kg_surface_labels_are_appended_to_retrieval_query():
    """When KG returns spans, distinctive content tokens from those spans
    that are NOT in the original query must appear in the rewritten query
    that BM25 and Dense receive.
    """
    art_uri = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"
    # SPARQL hit for "زواج" returns a span whose distinctive non-query
    # content token is "ميراث" (inheritance) — that's the expansion we
    # expect to see on the BM25/Dense calls.
    pipeline, bm25, dense, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
        sparql_rows_per_token={
            "زواج": [
                _row(art_uri, "ميراث ميراث ميراث وقواعد"),
            ],
        },
    )
    pipeline.run("ما هي شروط زواج")
    bm25_query = bm25.search.call_args.args[0]
    dense_query = dense.search.call_args.args[0]
    assert "ميراث" in bm25_query
    assert "ميراث" in dense_query
    # Original query must remain a prefix (we append, not replace).
    assert bm25_query.startswith("ما هي شروط زواج")


def test_no_expansion_when_kg_returns_empty():
    """If the KG yields nothing, BM25/Dense get the original query unchanged."""
    pipeline, bm25, dense, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        sparql_rows_per_token={},  # KG miss for every token
    )
    pipeline.run("ما هي شروط زواج")
    assert bm25.search.call_args.args[0] == "ما هي شروط زواج"
    assert dense.search.call_args.args[0] == "ما هي شروط زواج"


def test_expansion_skips_terms_already_in_the_query():
    """A KG-matched span containing only words already in the query
    contributes no expansion terms.
    """
    art_uri = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"
    pipeline, bm25, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        sparql_rows_per_token={
            # Span repeats "زواج" — the only content token — which is
            # already in the query, so nothing should be appended.
            "زواج": [_row(art_uri, "زواج زواج زواج")],
        },
    )
    pipeline.run("ما هي شروط زواج")
    assert bm25.search.call_args.args[0] == "ما هي شروط زواج"


def test_expansion_terms_max_caps_appended_count():
    """expansion_terms_max=2 → at most 2 new tokens appended."""
    art_uri = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"
    pipeline, bm25, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        sparql_rows_per_token={
            "زواج": [
                _row(art_uri, "ميراث طلاق نفقة حضانة عقود رهن دين"),
            ],
        },
        expansion_terms_max=2,
    )
    pipeline.run("زواج")
    rewritten = bm25.search.call_args.args[0]
    # original token "زواج" + at most 2 expansion terms → at most 3 tokens.
    assert len(rewritten.split()) <= 3


# ---------------------------------------------------------------------------
# KG bias — KG-surfaced (doc_id, ref) get a small score boost
# ---------------------------------------------------------------------------

def test_kg_surfaced_candidate_receives_boost_flag():
    """A fused candidate whose key matches a KG-surfaced URI gets
    ``kg_boosted=True`` in the output citation.
    """
    art_uri = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"
    pipeline, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        sparql_rows_per_token={"زواج": [_row(art_uri, "نص قانوني")]},
        # Use the registry alias map so the URI's "84-11" resolves to the
        # canonical doc_id used by the BM25 hit.
        alias_map={"84-11": "84-11_1984-06-09"},
    )
    out = pipeline.run("زواج")
    boosted = [c for c in out["citations"] if c.get("kg_boosted")]
    assert len(boosted) == 1
    assert boosted[0]["doc_id"] == "84-11_1984-06-09"
    assert boosted[0]["article_ref"] == "5"


def test_kg_boost_floats_kg_surfaced_article_above_unboosted():
    """A KG-surfaced article with a tiny RRF score floats above an
    unboosted article whose RRF score is slightly higher.

    Construct two candidates with very close RRF scores: only one
    retriever ranks each. The KG-surfaced article gets the small boost
    and ends up first.
    """
    art_uri_b = "https://legal.dz/resource/code/1984-06-09/84-11#art_7"
    pipeline, *_ = _make_pipeline(
        # Article 5 ranks #1 in BM25 only (RRF ~ 1/61). Article 7 ranks
        # #2 in BM25 only (RRF ~ 1/62). Without boost, 5 wins. With KG
        # boost on 7, 7 should win.
        bm25_hits=[
            _bm25("84-11_1984-06-09", "5", "نص 5", score=2.0),
            _bm25("84-11_1984-06-09", "7", "نص 7", score=1.0),
        ],
        dense_hits=[],
        sparql_rows_per_token={
            "زواج": [_row(art_uri_b, "نص قانوني")],  # KG → article 7 only
        },
        alias_map={"84-11": "84-11_1984-06-09"},
        kg_boost=0.5,                               # exaggerate for clarity
    )
    out = pipeline.run("زواج")
    refs = [c["article_ref"] for c in out["citations"]]
    assert refs[0] == "7", refs


def test_kg_boost_zero_disables_bias():
    """``kg_boost=0`` means no boost is applied — fused order is preserved."""
    art_uri = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"
    pipeline, *_ = _make_pipeline(
        bm25_hits=[
            _bm25("84-11_1984-06-09", "7", "نص 7", score=2.0),
            _bm25("84-11_1984-06-09", "5", "نص 5", score=1.0),
        ],
        sparql_rows_per_token={"زواج": [_row(art_uri, "نص")]},
        alias_map={"84-11": "84-11_1984-06-09"},
        kg_boost=0.0,
    )
    out = pipeline.run("زواج")
    refs = [c["article_ref"] for c in out["citations"]]
    # KG would have boosted 5, but kg_boost=0 disables that — 7 stays on top.
    assert refs[0] == "7", refs
    # And no citation should be flagged as boosted.
    assert all(not c.get("kg_boosted") for c in out["citations"])


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------

def test_empty_query_abstains():
    pipeline, *_ = _make_pipeline()
    out = pipeline.run("   ")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    assert out["citations"] == []
    assert out["answer_text"] == ""


def test_no_hits_when_both_retrievers_empty_regardless_of_kg():
    """KG is allowed to be empty too — both retrievers empty → no_hits."""
    pipeline, *_ = _make_pipeline()
    out = pipeline.run("سؤال غامض")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"


def test_one_retriever_empty_other_full_still_returns_citations():
    pipeline, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص")],
        dense_hits=[],
    )
    out = pipeline.run("سؤال")
    assert out["abstention"] is False
    assert len(out["citations"]) == 1
    assert out["citations"][0]["doc_id"] == "84-11_1984-06-09"


def test_kg_hit_with_zero_retrieval_still_abstains():
    """KG matched but neither BM25 nor Dense has hits → no_hits.

    The pipeline is fundamentally a hybrid retriever — KG is a *bias*
    signal, not a retrieval source. With no candidates from BM25/Dense,
    there is nothing to bias and the pipeline must abstain.
    """
    art_uri = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"
    pipeline, *_ = _make_pipeline(
        bm25_hits=[],
        dense_hits=[],
        sparql_rows_per_token={"زواج": [_row(art_uri, "نص")]},
        alias_map={"84-11": "84-11_1984-06-09"},
    )
    out = pipeline.run("زواج")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"


# ---------------------------------------------------------------------------
# SPARQL plumbing
# ---------------------------------------------------------------------------

def test_sparql_called_once_per_distinct_token():
    pipeline, _bm25, _dense, _registry, calls = _make_pipeline(
        sparql_rows_per_token={"زواج": [], "شروط": [], "عقد": []},
    )
    pipeline.run("ما هي شروط زواج عقد")
    # 3 distinct content tokens → 3 SPARQL calls.
    assert len(calls) == 3


def test_sparql_failure_does_not_propagate():
    """One failing SPARQL call must not poison the whole query."""
    art_uri = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"

    def flaky_sparql(kg, query: str) -> list[dict[str, Any]]:
        if "زواج" in query:
            raise RuntimeError("simulated KG hiccup")
        return [_row(art_uri, "نص")]

    bm25 = MagicMock()
    bm25.search.return_value = [_bm25("84-11_1984-06-09", "5", "نص")]
    dense = MagicMock()
    dense.search.return_value = []
    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="قانون الأسرة")
    registry.resolve_alias.side_effect = lambda key: None

    pipeline = KGHybridBaselinePipeline(
        kg=MagicMock(),
        bm25=bm25,
        dense=dense,
        registry=registry,
        sparql_fn=flaky_sparql,
    )
    out = pipeline.run("شروط زواج")
    # "زواج" failed but "شروط" succeeded — pipeline still answers.
    assert out["abstention"] is False
    assert any(c["article_ref"] == "5" for c in out["citations"])


# ---------------------------------------------------------------------------
# Template answer
# ---------------------------------------------------------------------------

def test_template_answer_uses_doc_title_and_ref():
    pipeline, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص المادة")],
        dense_hits=[_dense("84-11_1984-06-09", "5", "نص المادة")],
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
    registry.get_doc.return_value = None
    registry.resolve_alias.side_effect = lambda key: None
    pipeline = KGHybridBaselinePipeline(
        kg=MagicMock(),
        bm25=bm25,
        dense=dense,
        registry=registry,
        sparql_fn=lambda kg, q: [],
    )
    out = pipeline.run("سؤال")
    assert "xx_yy" in out["answer_text"]


# ---------------------------------------------------------------------------
# Compatibility with the eval runner's _answer_to_result
# ---------------------------------------------------------------------------

def test_answer_to_result_consumes_output_without_branching():
    from akn_rlm.eval.runner import _answer_to_result

    pipeline, *_ = _make_pipeline(
        bm25_hits=[_bm25("84-11_1984-06-09", "5", "نص المادة")],
        dense_hits=[_dense("84-11_1984-06-09", "5", "نص المادة")],
    )
    answer = pipeline.run("ما هي المادة 5؟")
    answer["_latency_s"] = 0.01

    question = {
        "id": "fam_test_q01",
        "query": "ما هي المادة 5؟",
        "query_type": "rule_application",
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
    assert result["hcr"] == 0.0
