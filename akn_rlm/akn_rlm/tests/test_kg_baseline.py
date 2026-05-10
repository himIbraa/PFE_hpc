"""Tests for the KG-only baseline pipeline (B5).

The pipeline mirrors the ``run(query) -> dict`` contract used by B1-B4 and
the RLM pipeline so ``akn_rlm.eval.runner._answer_to_result`` consumes its
output without branching. SPARQL is injected via ``sparql_fn`` so these
tests do not load the on-disk TTL.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from akn_rlm.baselines.kg_pipeline import (
    DEFAULT_TOP_K,
    KGBaselinePipeline,
    SUPPORT_SPAN_LEN,
    build_kg_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(uri: str, text: str = "") -> dict[str, Any]:
    """Mimic the dict shape returned by ``graphrag.sparql_query``."""
    return {"article": uri, "text": text}


def _make_pipeline(
    sparql_rows_per_token: dict[str, list[dict[str, Any]]] | None = None,
    *,
    doc_title: str = "قانون الأسرة",
    top_k: int = DEFAULT_TOP_K,
    fallback: Any = None,
    alias_map: dict[str, str] | None = None,
):
    """Build a KGBaselinePipeline whose SPARQL is fully mocked.

    ``sparql_rows_per_token`` maps the SAFE token literal (single-quote-
    escaped, backslashes stripped — same transform the pipeline applies)
    to the rows the SPARQL call should yield.
    """
    sparql_rows_per_token = sparql_rows_per_token or {}

    captured_calls: list[tuple[Any, str]] = []

    def fake_sparql(kg, query: str) -> list[dict[str, Any]]:
        captured_calls.append((kg, query))
        # Find the FILTER literal token inside the query and look it up.
        # Each token is referenced three times (paragraph / provision /
        # condition) but the lookup is the same string; just pick the first
        # match.
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

    pipeline = build_kg_pipeline(
        kg=kg,
        registry=registry,
        top_k=top_k,
        fallback=fallback,
        sparql_fn=fake_sparql,
    )
    return pipeline, registry, captured_calls


# ---------------------------------------------------------------------------
# Contract — answer dict shape
# ---------------------------------------------------------------------------

def test_run_returns_required_keys():
    pipeline, *_ = _make_pipeline({
        "زواج": [
            _row(
                "https://legal.dz/resource/code/1984-06-09/84-11#art_5",
                "نص المادة 5",
            ),
        ],
    })
    out = pipeline.run("ما هي شروط زواج")
    required = {
        "answer_text", "abstention", "abstention_reason", "citations",
        "reasoning_chain", "trajectory", "tokens_used", "_telemetry",
    }
    assert required.issubset(out.keys())


def test_telemetry_baseline_is_kg_on_hit():
    pipeline, *_ = _make_pipeline({
        "زواج": [
            _row(
                "https://legal.dz/resource/code/1984-06-09/84-11#art_5",
                "نص",
            ),
        ],
    })
    out = pipeline.run("ما هو زواج")
    assert out["_telemetry"]["baseline"] == "kg"


def test_default_top_k_is_5():
    assert DEFAULT_TOP_K == 5


# ---------------------------------------------------------------------------
# Citation shape
# ---------------------------------------------------------------------------

def test_citation_carries_supporting_span_text_and_confidence():
    long_text = "نص قانوني" * 80
    pipeline, *_ = _make_pipeline({
        "زواج": [
            _row(
                "https://legal.dz/resource/code/1984-06-09/84-11#art_5",
                long_text,
            ),
        ],
    })
    out = pipeline.run("ما هي شروط زواج")
    assert len(out["citations"]) == 1
    c = out["citations"][0]
    assert c["doc_id"]
    assert c["article_ref"] == "5"
    assert c["text"] == long_text
    assert c["supporting_span"] == long_text[:SUPPORT_SPAN_LEN]
    # Confidence is the cover score (number of matching tokens) — at least 1.
    assert c["confidence"] >= 1.0


# ---------------------------------------------------------------------------
# KG-hit path — token coverage scoring
# ---------------------------------------------------------------------------

def test_articles_ranked_by_distinct_token_coverage():
    """An article hit by 2 tokens must outrank one hit by only 1 token."""
    art_a = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"
    art_b = "https://legal.dz/resource/code/1984-06-09/84-11#art_7"
    pipeline, *_ = _make_pipeline({
        # token1 → both articles; token2 → only art_a → art_a wins.
        "زواج":  [_row(art_a, "نص 5"), _row(art_b, "نص 7")],
        "شروط":   [_row(art_a, "شروط زواج")],
    })
    out = pipeline.run("ما هي شروط زواج")
    refs = [c["article_ref"] for c in out["citations"]]
    assert refs[0] == "5"            # winner first
    assert "7" in refs               # both surfaced
    confidences = [c["confidence"] for c in out["citations"]]
    assert confidences[0] >= confidences[1]


def test_subnode_uri_resolves_to_parent_article():
    """A paragraph / right / obligation URI must collapse to its parent.

    The fragment ``art_10_para_0`` (paragraph node) and ``art_10`` (article
    node) must produce a single citation pointing at article 10.
    """
    para_uri = "https://legal.dz/resource/constitution/1963-12-08/1963#art_10_para_0"
    art_uri  = "https://legal.dz/resource/constitution/1963-12-08/1963#art_10"
    pipeline, *_ = _make_pipeline({
        "عمل":  [_row(para_uri, "نص الفقرة")],
        "حق":     [_row(art_uri, "نص المادة")],
    })
    out = pipeline.run("حق عمل")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"10"}


def test_dedupes_repeated_doc_article_pairs():
    """Different sub-node hits on the same article must collapse to one citation."""
    para_uri = (
        "https://legal.dz/resource/code/1984-06-09/84-11#art_5_para_0"
    )
    obl_uri = (
        "https://legal.dz/resource/code/1984-06-09/84-11#art_5_obligation_0_abc"
    )
    art_uri = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"
    pipeline, *_ = _make_pipeline({
        "زواج": [_row(para_uri, "para text"), _row(obl_uri, "obl text"),
                  _row(art_uri, "art text")],
    })
    out = pipeline.run("زواج")
    keys = {(c["doc_id"], c["article_ref"]) for c in out["citations"]}
    assert keys == {(c["doc_id"], "5") for c in out["citations"]}
    assert len(out["citations"]) == 1


def test_canonicalises_bis_article_ref():
    """An ``art_9_bis`` URI keeps canonical ``"9_bis"`` form."""
    pipeline, *_ = _make_pipeline({
        "ميراث": [
            _row(
                "https://legal.dz/resource/code/1984-06-09/84-11#art_9_bis",
                "مكرر",
            ),
        ],
    })
    out = pipeline.run("ميراث")
    refs = {c["article_ref"] for c in out["citations"]}
    assert "9_bis" in refs


def test_uses_registry_alias_to_canonicalise_doc_id():
    """``num`` from URI is fed to ``registry.resolve_alias`` for canonical doc_id."""
    pipeline, registry, _calls = _make_pipeline(
        sparql_rows_per_token={
            "زواج": [
                _row(
                    "https://legal.dz/resource/code/1984-06-09/84-11#art_5",
                    "نص",
                ),
            ],
        },
        alias_map={"84-11": "84-11_1984-06-09"},
    )
    out = pipeline.run("زواج")
    assert out["citations"][0]["doc_id"] == "84-11_1984-06-09"


def test_top_k_truncates_after_scoring():
    """``top_k=2`` keeps only the two highest-scoring articles."""
    rows = [
        _row(
            f"https://legal.dz/resource/code/1984-06-09/84-11#art_{i}",
            f"نص {i}",
        )
        for i in range(5)
    ]
    pipeline, *_ = _make_pipeline(
        sparql_rows_per_token={"زواج": rows},
        top_k=2,
    )
    out = pipeline.run("زواج")
    assert len(out["citations"]) == 2


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------

def test_empty_query_abstains():
    pipeline, *_ = _make_pipeline({})
    out = pipeline.run("   ")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    assert out["citations"] == []
    assert out["answer_text"] == ""


def test_no_kg_hits_abstains_when_no_fallback():
    pipeline, *_ = _make_pipeline(sparql_rows_per_token={})
    out = pipeline.run("سؤال غامض جداً")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_kg_hits"
    assert out["citations"] == []


def test_query_with_only_stopwords_abstains():
    """If every token is a stopword, the pipeline must abstain (no_tokens)."""
    pipeline, *_ = _make_pipeline({})
    # All these words are in the stopword list (from kg_pipeline._STOPWORDS).
    out = pipeline.run("ما هو ماذا كيف")
    assert out["abstention"] is True
    assert out["abstention_reason"] in {"no_tokens", "no_kg_hits"}


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------

def test_kg_miss_with_fallback_returns_fallback_answer_with_kg_fallback_telemetry():
    fallback_answer = {
        "answer_text":       "fallback text",
        "abstention":        False,
        "abstention_reason": None,
        "citations":         [{
            "doc_id":          "84-11_1984-06-09",
            "article_ref":     "5",
            "doc_title":       "قانون الأسرة",
            "supporting_span": "نص",
            "text":            "نص",
            "confidence":      0.42,
        }],
        "reasoning_chain":   [],
        "trajectory":        [],
        "tokens_used":       0,
        "depth_max_reached": 0,
        "_telemetry": {
            "retry_count":  0,
            "gate_results": {},
            "baseline":     "hybrid",
        },
    }
    fallback = MagicMock()
    fallback.run.return_value = fallback_answer

    pipeline, *_ = _make_pipeline(sparql_rows_per_token={}, fallback=fallback)
    out = pipeline.run("سؤال يفشل في الـKG")
    fallback.run.assert_called_once_with("سؤال يفشل في الـKG")
    assert out["abstention"] is False
    assert out["answer_text"] == "fallback text"
    # Telemetry rewritten so the metrics report knows this came from fallback.
    assert out["_telemetry"]["baseline"] == "kg+fallback"


def test_kg_miss_with_failing_fallback_abstains():
    fallback = MagicMock()
    fallback.run.side_effect = RuntimeError("boom")
    pipeline, *_ = _make_pipeline(sparql_rows_per_token={}, fallback=fallback)
    out = pipeline.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_kg_hits"


# ---------------------------------------------------------------------------
# SPARQL plumbing
# ---------------------------------------------------------------------------

def test_sparql_called_once_per_distinct_token():
    """The pipeline issues one SPARQL per distinct token, capped at MAX_TOKENS."""
    pipeline, _registry, calls = _make_pipeline(
        sparql_rows_per_token={
            "زواج":  [_row(
                "https://legal.dz/resource/code/1984-06-09/84-11#art_5",
                "نص",
            )],
            "شروط":   [],
            "عقد":  [],
        },
    )
    pipeline.run("ما هي شروط زواج عقد")
    # 3 distinct content tokens (after stopword drop): "شروط", "زواج"→"زواج", "عقد"→"عقد"
    # Each yields exactly one SPARQL call.
    assert len(calls) == 3


def test_sparql_failure_does_not_propagate_and_falls_back_to_other_tokens():
    """One failing SPARQL call must not poison the whole query."""
    art_uri = "https://legal.dz/resource/code/1984-06-09/84-11#art_5"

    def flaky_sparql(kg, query: str) -> list[dict[str, Any]]:
        if "زواج" in query:
            raise RuntimeError("simulated KG hiccup")
        return [_row(art_uri, "نص")]

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title="قانون الأسرة")
    registry.resolve_alias.side_effect = lambda key: None

    pipeline = KGBaselinePipeline(
        kg=MagicMock(),
        registry=registry,
        sparql_fn=flaky_sparql,
    )
    out = pipeline.run("شروط زواج")
    # "زواج" failed but "شروط" (or its variant) succeeded → answer present.
    assert out["abstention"] is False
    assert any(c["article_ref"] == "5" for c in out["citations"])


# ---------------------------------------------------------------------------
# Template answer
# ---------------------------------------------------------------------------

def test_template_answer_uses_doc_title_and_ref():
    pipeline, *_ = _make_pipeline(
        sparql_rows_per_token={
            "زواج": [_row(
                "https://legal.dz/resource/code/1984-06-09/84-11#art_5",
                "نص المادة 5",
            )],
        },
        doc_title="قانون الأسرة",
    )
    out = pipeline.run("زواج")
    assert "قانون الأسرة" in out["answer_text"]
    assert "المادة 5" in out["answer_text"]
    assert "نص المادة 5" in out["answer_text"]


def test_template_falls_back_to_doc_id_when_no_title():
    registry = MagicMock()
    registry.get_doc.return_value = None
    registry.resolve_alias.side_effect = lambda key: None

    def fake_sparql(kg, query: str) -> list[dict[str, Any]]:
        if 'CONTAINS(?text, "بصمة")' in query:
            return [_row(
                "https://legal.dz/resource/code/1984-06-09/84-11#art_1",
                "نص",
            )]
        return []

    pipeline = KGBaselinePipeline(
        kg=MagicMock(),
        registry=registry,
        sparql_fn=fake_sparql,
    )
    out = pipeline.run("بصمة")
    # No registry title and no alias → doc_id falls through to the URI's <num>.
    assert "84-11" in out["answer_text"]


# ---------------------------------------------------------------------------
# Compatibility with the eval runner's _answer_to_result
# ---------------------------------------------------------------------------

def test_answer_to_result_consumes_output_without_branching():
    from akn_rlm.eval.runner import _answer_to_result

    pipeline, *_ = _make_pipeline(
        sparql_rows_per_token={
            "زواج": [_row(
                "https://legal.dz/resource/code/1984-06-09/84-11#art_5",
                "نص المادة 5",
            )],
        },
        alias_map={"84-11": "84-11_1984-06-09"},
    )
    answer = pipeline.run("ما هي شروط زواج")
    answer["_latency_s"] = 0.01

    question = {
        "id": "fam_test_q01",
        "query": "ما هي شروط زواج",
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
    # Deterministic baseline → no rejected citations → HCR=0
    assert result["hcr"] == 0.0
