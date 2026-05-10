"""Tests for the Phase-2 exact_article handler.

The handler must mirror the baseline ``.run(query) -> dict`` contract so
``akn_rlm.eval.runner._answer_to_result`` consumes it unchanged. BM25,
the registry, the verifier, and the summariser are fully mocked — the
suite never touches a real index, real LLM, or any external service,
and runs in milliseconds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akn_rlm.eval.runner import _answer_to_result
from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.rlm.handlers.exact_article import (
    DEFAULT_BM25_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_MAX_EXPLICIT_REFS,
    DEFAULT_ROUTE_TOP_N,
    DEFAULT_TOP_K_CANDIDATES,
    DEFAULT_VERIFY_THRESHOLD,
    SUPPORT_SPAN_LEN,
    TELEMETRY_BASELINE,
    ExactArticleHandler,
    _extract_explicit_article_refs,
    _find_article_in_bm25_meta,
    build_exact_article_handler,
)
from akn_rlm.rlm.routing import RouteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm25_hit(doc_id: str, ref: str, text: str = "نص المادة", score: float = 5.0) -> BM25Hit:
    return BM25Hit(
        chunk_id=f"{doc_id}#art_{ref}",
        doc_id=doc_id,
        article_ref=ref,
        score=score,
        text=text,
    )


def _stub_router(doc_ids: list[str]) -> MagicMock:
    router = MagicMock()
    router.route.return_value = RouteResult(
        doc_ids=list(doc_ids),
        scores={d: 1.0 for d in doc_ids},
        sources={d: ["alias"] for d in doc_ids},
        confidence=1.0 if doc_ids else 0.0,
    )
    return router


def _stub_verifier(verdict_for: dict[tuple[str, str], dict] | None = None,
                   default: dict | None = None):
    verdict_for = verdict_for or {}
    default = default or {
        "relevant": True,
        "supporting_span": None,
        "contradicting_span": None,
        "confidence": 0.9,
    }

    def _fn(_pool, _q, article, _model):
        key = (article.get("doc_id", ""), article.get("article_ref", ""))
        return dict(verdict_for.get(key, default))

    return MagicMock(side_effect=_fn)


def _stub_summarizer(summary: str | None = "ملخّص"):
    def _fn(_pool, _q, _articles, _model):
        return {"summary": summary, "key_articles": [], "caveats": None}

    return MagicMock(side_effect=_fn)


def _make_handler(
    *,
    bm25_hits: list[BM25Hit] | None = None,
    bm25_meta: list[dict] | None = None,
    routed_ids: list[str] | None = None,
    verifier_verdicts: dict[tuple[str, str], dict] | None = None,
    summary: str | None = "ملخّص",
    doc_title: str = "قانون الأسرة",
    has_article_returns: bool | dict | None = None,
    **kwargs,
):
    bm25 = MagicMock()
    bm25.search.return_value = list(bm25_hits or [])
    # The handler reads ``_meta`` directly for direct lookup.
    if bm25_meta is None:
        bm25._meta = []
    else:
        bm25._meta = list(bm25_meta)

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)
    if has_article_returns is None:
        registry.has_article.return_value = True
    elif isinstance(has_article_returns, dict):
        def _has(doc_id, ref):
            return bool(has_article_returns.get((doc_id, ref), False))
        registry.has_article.side_effect = _has
    else:
        registry.has_article.return_value = bool(has_article_returns)

    router = _stub_router(routed_ids if routed_ids is not None else ["84-11_1984-06-09"])
    verifier = _stub_verifier(verifier_verdicts)
    summarizer = _stub_summarizer(summary)

    handler = ExactArticleHandler(
        bm25=bm25,
        registry=registry,
        llm_pool=MagicMock(),
        router=router,
        verifier_fn=verifier,
        summarizer_fn=summarizer,
        **kwargs,
    )
    return handler, dict(
        bm25=bm25, registry=registry, router=router,
        verifier=verifier, summarizer=summarizer,
    )


# ---------------------------------------------------------------------------
# Defaults & contract
# ---------------------------------------------------------------------------


def test_defaults_match_handoff_design():
    assert DEFAULT_TOP_K_CANDIDATES == 5
    # F7: reverted F6's 2 back to F4's 3 (F6's 2 was net neutral on EA
    # but the simultaneous threshold change made attribution noisy).
    assert DEFAULT_FINAL_TOP_K == 3
    assert DEFAULT_BM25_K == 30
    # F4: reverted R9.1's 0.5→0.3 back to 0.5 (F3 evidence of noise).
    assert DEFAULT_VERIFY_THRESHOLD == 0.5
    assert DEFAULT_ROUTE_TOP_N == 3
    assert DEFAULT_MAX_EXPLICIT_REFS == 6


def test_f4_default_verify_threshold_locked_at_0_5():
    from akn_rlm.rlm.handlers import exact_article as ea_mod
    assert ea_mod.DEFAULT_VERIFY_THRESHOLD == 0.5


def test_f7_default_final_top_k_locked_at_3():
    from akn_rlm.rlm.handlers import exact_article as ea_mod
    assert ea_mod.DEFAULT_FINAL_TOP_K == 3


def test_factory_builds_handler():
    bm25 = MagicMock(); bm25._meta = []
    registry = MagicMock()
    h = build_exact_article_handler(
        bm25=bm25, registry=registry, llm_pool=MagicMock(),
        router=_stub_router(["d"]),
    )
    assert isinstance(h, ExactArticleHandler)


def test_telemetry_baseline_tag_is_rlm_exact_article():
    handler, _ = _make_handler(bm25_hits=[_bm25_hit("84-11_1984-06-09", "5")])
    out = handler.run("ما هي شروط الزواج؟")
    assert out["_telemetry"]["baseline"] == TELEMETRY_BASELINE
    assert TELEMETRY_BASELINE == "rlm_exact_article"


def test_run_returns_required_keys():
    handler, _ = _make_handler(bm25_hits=[_bm25_hit("84-11_1984-06-09", "5")])
    out = handler.run("سؤال")
    for key in ("answer_text", "abstention", "abstention_reason", "citations",
                "reasoning_chain", "trajectory", "tokens_used",
                "depth_max_reached", "_telemetry"):
        assert key in out


# ---------------------------------------------------------------------------
# Article-number extraction
# ---------------------------------------------------------------------------


def test_extract_arabic_singular():
    refs = _extract_explicit_article_refs("بيّن المادة 7 من قانون الأسرة")
    assert "7" in refs


def test_extract_arabic_singular_arabic_digits():
    refs = _extract_explicit_article_refs("المادة ٤")
    # canonical_article_ref normalises Arabic digits → "4"
    assert "4" in refs


def test_extract_arabic_singular_bis():
    refs = _extract_explicit_article_refs("المادة 9 مكرر")
    assert "9_bis" in refs


def test_extract_arabic_singular_first_ordinal():
    refs = _extract_explicit_article_refs("المادة الأولى")
    assert "1" in refs


def test_extract_arabic_dual():
    refs = _extract_explicit_article_refs("المادتان 4 و 5 من قانون الأسرة")
    assert "4" in refs
    assert "5" in refs


def test_extract_arabic_plural_list():
    refs = _extract_explicit_article_refs("المواد 1 و 2 و 3 من قانون الأسرة")
    assert {"1", "2", "3"}.issubset(set(refs))


def test_extract_french_article():
    refs = _extract_explicit_article_refs("voir l'article 7 du Code")
    assert "7" in refs


def test_extract_french_art_dot():
    refs = _extract_explicit_article_refs("art. 12 of the Civil Code")
    assert "12" in refs


def test_extract_french_range():
    refs = _extract_explicit_article_refs("articles 4-5 of Family Code")
    assert {"4", "5"}.issubset(set(refs))


def test_extract_returns_canonical_form_only():
    """`9 مكرر` and `9_bis` should both canonicalise to a single ref."""
    refs = _extract_explicit_article_refs("see articles 9 bis and المادة 9 مكرر")
    # Both instances should canonicalise to "9_bis" — dedup keeps one.
    assert refs.count("9_bis") == 1


def test_extract_empty_query_returns_empty():
    assert _extract_explicit_article_refs("") == []
    assert _extract_explicit_article_refs(None) == []  # type: ignore


def test_extract_no_article_reference_returns_empty():
    refs = _extract_explicit_article_refs("ما هي شروط الزواج في القانون؟")
    assert refs == []


def test_extract_dedupes_repeats():
    refs = _extract_explicit_article_refs("المادة 7 ... المادة 7 ... article 7")
    assert refs.count("7") == 1


# ---------------------------------------------------------------------------
# Direct-lookup helper (BM25 meta scan)
# ---------------------------------------------------------------------------


def test_direct_lookup_finds_article_by_chunk_id():
    bm25 = MagicMock()
    bm25._meta = [
        {"chunk_id": "84-11_1984-06-09#art_7", "doc_id": "84-11_1984-06-09",
         "article_ref": "7", "text": "نص المادة 7"},
    ]
    hit = _find_article_in_bm25_meta(bm25, "84-11_1984-06-09", "7")
    assert hit is not None
    assert hit["text"] == "نص المادة 7"


def test_direct_lookup_returns_none_when_missing():
    bm25 = MagicMock(); bm25._meta = []
    hit = _find_article_in_bm25_meta(bm25, "84-11_1984-06-09", "7")
    assert hit is None


def test_direct_lookup_canonicalises_ref():
    """`9 مكرر` and `9_bis` must point at the same chunk."""
    bm25 = MagicMock()
    bm25._meta = [
        {"chunk_id": "84-11_1984-06-09#art_9_bis", "doc_id": "84-11_1984-06-09",
         "article_ref": "9 مكرر", "text": "نص"},
    ]
    hit = _find_article_in_bm25_meta(bm25, "84-11_1984-06-09", "9 مكرر")
    assert hit is not None
    assert hit["article_ref"] == "9_bis"


def test_direct_lookup_handles_doc_scan_fallback():
    """When chunk_id form doesn't exact-match (e.g. eid-case mismatch),
    fall back to a per-doc scan keyed on canonical article_ref."""
    bm25 = MagicMock()
    bm25._meta = [
        # Idiosyncratic chunk_id form
        {"chunk_id": "84-11_1984-06-09#ART_7", "doc_id": "84-11_1984-06-09",
         "article_ref": "7", "text": "fallback path"},
    ]
    hit = _find_article_in_bm25_meta(bm25, "84-11_1984-06-09", "7")
    assert hit is not None
    assert hit["text"] == "fallback path"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_router_called_with_route_top_n():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5")],
        route_top_n=2,
    )
    handler.run("سؤال")
    mocks["router"].route.assert_called_once()
    _args, kwargs = mocks["router"].route.call_args
    assert kwargs.get("top_n") == 2


def test_routed_doc_ids_recorded_in_telemetry():
    handler, _ = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5")],
        routed_ids=["84-11_1984-06-09", "75-58_1975-09-26"],
    )
    out = handler.run("سؤال")
    assert out["_telemetry"]["routed_doc_ids"] == [
        "84-11_1984-06-09", "75-58_1975-09-26",
    ]


# ---------------------------------------------------------------------------
# Direct-lookup path (explicit article numbers)
# ---------------------------------------------------------------------------


def test_explicit_number_triggers_direct_lookup():
    handler, mocks = _make_handler(
        bm25_meta=[
            {"chunk_id": "84-11_1984-06-09#art_7", "doc_id": "84-11_1984-06-09",
             "article_ref": "7", "text": "نص المادة 7"},
        ],
        bm25_hits=[],  # ensure BM25 search wouldn't find anything
        routed_ids=["84-11_1984-06-09"],
        has_article_returns=True,
    )
    out = handler.run("ما تقول المادة 7؟")
    assert out["abstention"] is False
    assert out["_telemetry"]["path"] == "direct_lookup"
    assert out["citations"][0]["article_ref"] == "7"
    # Direct path didn't need BM25 search at all.
    mocks["bm25"].search.assert_not_called()


def test_direct_lookup_skips_when_registry_has_no_article():
    handler, mocks = _make_handler(
        bm25_meta=[
            {"chunk_id": "84-11_1984-06-09#art_999", "doc_id": "84-11_1984-06-09",
             "article_ref": "999", "text": "..."},
        ],
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "8", "fallback BM25")],
        routed_ids=["84-11_1984-06-09"],
        has_article_returns={("84-11_1984-06-09", "999"): False},
    )
    out = handler.run("article 999 doesn't exist; this should fall through to BM25")
    # Fell through to BM25 path because registry says art_999 doesn't exist.
    assert out["_telemetry"]["path"] == "bm25"


def test_direct_lookup_falls_through_to_bm25_on_no_match():
    """Explicit number is named but the BM25 meta has no chunk for it →
    fall through to BM25 search."""
    handler, mocks = _make_handler(
        bm25_meta=[],
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "8", "BM25 fallback hit")],
        routed_ids=["84-11_1984-06-09"],
    )
    out = handler.run("article 7 of family code")
    assert out["_telemetry"]["path"] == "bm25"
    mocks["bm25"].search.assert_called_once()


def test_direct_lookup_runs_verifier_per_explicit_hit():
    handler, mocks = _make_handler(
        bm25_meta=[
            {"chunk_id": "84-11_1984-06-09#art_4", "doc_id": "84-11_1984-06-09",
             "article_ref": "4", "text": "t4"},
            {"chunk_id": "84-11_1984-06-09#art_5", "doc_id": "84-11_1984-06-09",
             "article_ref": "5", "text": "t5"},
        ],
        routed_ids=["84-11_1984-06-09"],
        has_article_returns=True,
    )
    out = handler.run("المادتان 4 و 5")
    assert out["_telemetry"]["path"] == "direct_lookup"
    assert mocks["verifier"].call_count == 2  # one per explicit ref
    refs = sorted(c["article_ref"] for c in out["citations"])
    assert refs == ["4", "5"]


def test_direct_lookup_caps_at_max_explicit_refs():
    """Pathological queries with many numbers shouldn't blow the LLM budget."""
    meta = [
        {"chunk_id": f"84-11_1984-06-09#art_{i}", "doc_id": "84-11_1984-06-09",
         "article_ref": str(i), "text": f"t{i}"}
        for i in range(1, 11)
    ]
    handler, mocks = _make_handler(
        bm25_meta=meta,
        routed_ids=["84-11_1984-06-09"],
        has_article_returns=True,
        max_explicit_refs=2,
    )
    out = handler.run("articles 1 2 3 4 5 6 7 8 9 10")
    # Verifier should have been called at most max_explicit_refs times.
    assert mocks["verifier"].call_count <= 2


def test_direct_lookup_explicit_refs_in_telemetry():
    handler, _ = _make_handler(
        bm25_meta=[
            {"chunk_id": "84-11_1984-06-09#art_7", "doc_id": "84-11_1984-06-09",
             "article_ref": "7", "text": "t7"},
        ],
        routed_ids=["84-11_1984-06-09"],
        has_article_returns=True,
    )
    out = handler.run("المادة 7")
    assert out["_telemetry"]["explicit_refs"] == ["7"]


# ---------------------------------------------------------------------------
# BM25 fallback path
# ---------------------------------------------------------------------------


def test_bm25_path_when_no_explicit_number():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5", "نص", score=12.0)],
        routed_ids=["84-11_1984-06-09"],
    )
    out = handler.run("ما هي الأهلية في قانون الأسرة؟")
    assert out["_telemetry"]["path"] == "bm25"
    assert out["citations"][0]["article_ref"] == "5"
    mocks["bm25"].search.assert_called_once()


def test_bm25_search_uses_bm25_k_arg():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5")],
        bm25_k=42,
    )
    handler.run("سؤال")
    assert mocks["bm25"].search.call_args.kwargs.get("k") == 42


def test_bm25_path_filters_by_routed_doc_ids():
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25_hit("84-11_1984-06-09", "5", "kept", score=10.0),
            _bm25_hit("66-156_1966-06-08", "9", "wrong-doc", score=20.0),
        ],
        routed_ids=["84-11_1984-06-09"],
    )
    out = handler.run("ما هي الأهلية؟")
    assert all(c["doc_id"] == "84-11_1984-06-09" for c in out["citations"])


def test_bm25_path_full_pool_fallback_when_filter_wipes():
    handler, _ = _make_handler(
        bm25_hits=[_bm25_hit("66-156_1966-06-08", "9", "only-doc", score=5.0)],
        routed_ids=["84-11_1984-06-09"],  # routed doc not in hits
    )
    out = handler.run("ما الأهلية؟")
    assert out["abstention"] is False
    assert any(c["doc_id"] == "66-156_1966-06-08" for c in out["citations"])


def test_bm25_path_truncates_to_top_k_candidates():
    bm25_hits = [
        _bm25_hit("84-11_1984-06-09", str(i), text=f"t{i}", score=20.0 - i)
        for i in range(1, 11)
    ]
    handler, mocks = _make_handler(
        bm25_hits=bm25_hits,
        top_k_candidates=2,
    )
    handler.run("سؤال")
    # Mandatory verifier on top-K only, so at most top_k_candidates calls.
    assert mocks["verifier"].call_count == 2


# ---------------------------------------------------------------------------
# Mandatory verifier
# ---------------------------------------------------------------------------


def test_verifier_drops_irrelevant_candidates():
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25_hit("84-11_1984-06-09", "5", "kept", 10.0),
            _bm25_hit("84-11_1984-06-09", "8", "dropped", 8.0),
        ],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "confidence": 0.9},
            ("84-11_1984-06-09", "8"): {"relevant": False, "confidence": 0.95},
        },
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"5"}


def test_verifier_drops_low_confidence():
    handler, _ = _make_handler(
        bm25_hits=[
            _bm25_hit("84-11_1984-06-09", "5", "A", 10.0),
            _bm25_hit("84-11_1984-06-09", "8", "B", 8.0),
        ],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "confidence": 0.9},
            ("84-11_1984-06-09", "8"): {"relevant": True, "confidence": 0.4},
        },
        verify_threshold=0.5,
    )
    out = handler.run("سؤال")
    refs = {c["article_ref"] for c in out["citations"]}
    assert refs == {"5"}


def test_verifier_exception_skips_candidate():
    def raising(*_a, **_k):
        raise RuntimeError("boom")
    handler, _ = _make_handler(bm25_hits=[_bm25_hit("84-11_1984-06-09", "5")])
    handler._verifier_fn = raising
    out = handler.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def test_summary_used_as_answer_text():
    handler, _ = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5")],
        summary="هذا هو الجواب",
    )
    out = handler.run("سؤال")
    assert out["answer_text"] == "هذا هو الجواب"


def test_null_summary_falls_back_to_template():
    handler, _ = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5", "نص محدد", score=10.0)],
        summary=None,
    )
    out = handler.run("سؤال")
    assert "وفقًا لـ" in out["answer_text"]
    assert "المادة 5" in out["answer_text"]


def test_summarizer_exception_falls_back_to_template():
    def raising(*_a, **_k):
        raise RuntimeError("boom")
    handler, _ = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5", "نص محدد")],
    )
    handler._summarizer_fn = raising
    out = handler.run("سؤال")
    assert "وفقًا لـ" in out["answer_text"]


# ---------------------------------------------------------------------------
# Citation shape
# ---------------------------------------------------------------------------


def test_citation_carries_doc_title_span_and_confidence():
    text = "نص قانوني" * 50
    handler, _ = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5", text, score=12.0)],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {
                "relevant": True,
                "confidence": 0.85,
                "supporting_span": text[10:50],
            },
        },
        doc_title="قانون الأسرة",
    )
    out = handler.run("سؤال")
    cit = out["citations"][0]
    assert cit["doc_title"] == "قانون الأسرة"
    assert cit["confidence"] == pytest.approx(0.85)
    assert cit["supporting_span"] == text[10:50]


def test_supporting_span_falls_back_when_quote_not_substring():
    text = "abc" * 200
    handler, _ = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5", text)],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {
                "relevant": True,
                "confidence": 0.9,
                "supporting_span": "FABRICATED — not in text",
            },
        },
    )
    out = handler.run("سؤال")
    assert out["citations"][0]["supporting_span"] == text[:SUPPORT_SPAN_LEN]


def test_supporting_span_truncates_at_280():
    text = "x" * 1000
    handler, _ = _make_handler(bm25_hits=[_bm25_hit("84-11_1984-06-09", "5", text)])
    out = handler.run("سؤال")
    assert len(out["citations"][0]["supporting_span"]) == SUPPORT_SPAN_LEN


def test_template_falls_back_to_doc_id_when_title_empty():
    handler, _ = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5", "نص")],
        summary=None,
        doc_title="",
    )
    out = handler.run("سؤال")
    assert "84-11_1984-06-09" in out["answer_text"]


# ---------------------------------------------------------------------------
# Aggregation / final_top_k
# ---------------------------------------------------------------------------


def test_final_top_k_truncates_after_verification():
    bm25_hits = [
        _bm25_hit("84-11_1984-06-09", str(i), text=f"t{i}", score=20.0 - i)
        for i in range(1, 11)
    ]
    handler, _ = _make_handler(
        bm25_hits=bm25_hits,
        top_k_candidates=10,
        final_top_k=3,
    )
    out = handler.run("سؤال")
    assert len(out["citations"]) == 3


def test_citations_ranked_by_confidence_desc():
    bm25_hits = [
        _bm25_hit("84-11_1984-06-09", "5", text="A", score=10.0),
        _bm25_hit("84-11_1984-06-09", "8", text="B", score=8.0),
    ]
    handler, _ = _make_handler(
        bm25_hits=bm25_hits,
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": True, "confidence": 0.6},
            ("84-11_1984-06-09", "8"): {"relevant": True, "confidence": 0.95},
        },
    )
    out = handler.run("سؤال")
    refs = [c["article_ref"] for c in out["citations"]]
    assert refs == ["8", "5"]


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------


def test_empty_query_abstains_without_calling_anything():
    handler, mocks = _make_handler()
    out = handler.run("")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    mocks["bm25"].search.assert_not_called()
    mocks["verifier"].assert_not_called()


def test_no_hits_abstains():
    handler, _ = _make_handler(bm25_hits=[])
    out = handler.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"


def test_no_verified_articles_abstains():
    handler, mocks = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5")],
        verifier_verdicts={
            ("84-11_1984-06-09", "5"): {"relevant": False, "confidence": 0.95},
        },
    )
    out = handler.run("سؤال")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"
    mocks["summarizer"].assert_not_called()


# ---------------------------------------------------------------------------
# Sub-LM call budget
# ---------------------------------------------------------------------------


def test_sub_call_budget_under_max_sub_calls():
    """top_k_candidates=5 + 1 summariser = 6 calls, well within max_sub_calls=12."""
    bm25_hits = [
        _bm25_hit("84-11_1984-06-09", str(i), f"t{i}", score=20.0 - i)
        for i in range(1, 8)
    ]
    handler, _ = _make_handler(bm25_hits=bm25_hits, top_k_candidates=5)
    out = handler.run("سؤال")
    assert out["_telemetry"]["sub_call_count"] <= 6


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_answer_to_result_compatibility():
    handler, _ = _make_handler(
        bm25_hits=[_bm25_hit("84-11_1984-06-09", "5", "نص", score=10.0)],
    )
    out = handler.run("ما هي شروط الزواج؟")
    out["_latency_s"] = 0.123
    question = {
        "id": "fam_ea_q01",
        "query": "ما هي شروط الزواج؟",
        "query_type": "exact_article",
        "legal_category": "family_law",
        "difficulty": "medium",
        "language": "ar",
        "split": "test",
        "gold_doc_ids": ["84-11_1984-06-09"],
        "gold_article_ids": ["84-11_1984-06-09#art_5"],
        "gold_citations": [{"doc_id": "84-11_1984-06-09", "article_ref": "5"}],
        "gold_abstain": False,
        "gold_answer": "...",
        "gold_reasoning_chain": [],
    }
    result = _answer_to_result(question, out)
    assert result["pred_doc_ids"] == ["84-11_1984-06-09"]
    assert result["pred_article_ids"] == ["84-11_1984-06-09#art_5"]
    assert result["predicted_abstain"] is False


def test_answer_to_result_compatibility_on_abstention():
    handler, _ = _make_handler(bm25_hits=[])
    out = handler.run("سؤال")
    out["_latency_s"] = 0.05
    question = {
        "id": "q",
        "query": "سؤال",
        "query_type": "exact_article",
        "gold_doc_ids": ["x"],
        "gold_article_ids": ["x#art_1"],
        "gold_citations": [{"doc_id": "x", "article_ref": "1"}],
        "gold_abstain": False,
    }
    result = _answer_to_result(question, out)
    assert result["predicted_abstain"] is True
    assert result["pred_doc_ids"] == []
