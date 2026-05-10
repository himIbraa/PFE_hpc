"""Tests for the Phase-2 conceptual_definitional handler.

The handler must mirror the baseline ``.run(query) -> dict`` contract so
``akn_rlm.eval.runner._answer_to_result`` consumes it unchanged.
SPARQL, paraphraser, ADU extractor, verifier, and summariser are fully
mocked — the suite never touches a real KG, real LLM, or rdflib, and
runs in milliseconds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from akn_rlm.indexers.bm25 import BM25Hit
from akn_rlm.indexers.dense import DenseHit
from akn_rlm.rlm.handlers.conceptual_definitional import (
    DEFAULT_ADU_EXTRACT_TOP_N,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH,
    DEFAULT_KG_LIMIT,
    DEFAULT_MIN_KG_HITS,
    DEFAULT_PARAPHRASE_COUNT,
    DEFAULT_ROUTE_TOP_N,
    DEFAULT_TOP_K_CANDIDATES,
    DEFAULT_VERIFY_THRESHOLD,
    DEFAULT_VERIFY_TOP_N,
    SUPPORT_SPAN_LEN,
    TELEMETRY_BASELINE,
    ConceptualDefinitionalHandler,
    _content_tokens,
    _extract_concept_phrases,
    _generate_paraphrases,
    _kg_phrase_lookup,
    _uri_to_doc_ref,
    build_conceptual_definitional_handler,
)
from akn_rlm.rlm.routing import RouteResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm25(doc_id: str, ref: str, text: str, score: float = 5.0) -> BM25Hit:
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


def _stub_router(doc_ids: list[str]) -> MagicMock:
    router = MagicMock()
    router.route.return_value = RouteResult(
        doc_ids=list(doc_ids),
        scores={d: 1.0 for d in doc_ids},
        sources={d: ["alias"] for d in doc_ids},
        confidence=1.0 if doc_ids else 0.0,
    )
    return router


def _stub_sparql(phrase_to_rows: dict[str, list[dict]]) -> MagicMock:
    """SPARQL stub: returns rows[phrase] for any query whose CONTAINS literal
    matches one of the keys in ``phrase_to_rows``.
    """
    rows_by_phrase = dict(phrase_to_rows)

    def _fn(query: str) -> list[dict]:
        for phrase, rows in rows_by_phrase.items():
            if f'"{phrase}"' in query:
                return list(rows)
        return []

    return MagicMock(side_effect=_fn)


def _stub_summarizer(summary: str | None = "ملخص"):
    def _fn(_pool, _q, _articles, _model):
        return {"summary": summary, "key_articles": [], "caveats": None}
    return MagicMock(side_effect=_fn)


def _stub_verifier(default=None):
    default = default or {
        "relevant": True, "supporting_span": None, "confidence": 0.9,
    }
    def _fn(_pool, _q, _article, _model):
        return dict(default)
    return MagicMock(side_effect=_fn)


def _stub_paraphrase(paraphrases: list[str]):
    def _fn(_pool, _q, _n, _model):
        return list(paraphrases)
    return MagicMock(side_effect=_fn)


def _stub_adu(claim_for=None, default=None):
    """Mock ADU extractor: returns canned Toulmin dict per article text."""
    claim_for = claim_for or {}
    default = default or {
        "claim": "كل فعل يسبب ضرراً يلزم مرتكبه بالتعويض.",
        "ground": "خطأ + ضرر + علاقة سببية.",
        "warrant": "مبدأ المسؤولية التقصيرية.",
        "backing": "",
        "rebuttal": "",
    }

    def _fn(article_text, _pool, *, model=None):
        for key, val in claim_for.items():
            if key in (article_text or ""):
                return dict(val)
        return dict(default)

    return MagicMock(side_effect=_fn)


def _make_handler(
    *,
    bm25_hits=None,
    dense_hits=None,
    routed_ids=None,
    phrase_to_rows=None,
    paraphrases=None,
    paraphrase_dense_hits=None,
    summary="ملخص",
    doc_title="قانون مدني",
    adu_default=None,
    adu_claim_for=None,
    **kwargs,
):
    # When the test sets up ``phrase_to_rows`` (KG concept-search hits)
    # but doesn't also pass ``bm25_hits``, auto-derive a single matching
    # BM25 hit per URI so the fused retrieval has candidates to feed.
    # KG is now a *secondary* (boost-only) signal — the candidate pool
    # is built from BM25/Dense, with KG hits adding a small score
    # boost. Tests that explicitly want the BM25 channel empty can
    # pass ``bm25_hits=[]``.
    if bm25_hits is None and phrase_to_rows and routed_ids:
        from akn_rlm.rlm.handlers.conceptual_definitional import _ART_SUFFIX_RE
        bm25_hits = []
        # Use the first routed_id as the canonical doc_id for each KG hit
        # — this matches the test convention that ``registry.resolve_alias``
        # is configured to map the URI's bare ``num`` to that canonical id.
        canon_doc_id = routed_ids[0] if routed_ids else None
        if canon_doc_id:
            seen: set[tuple[str, str]] = set()
            for rows in phrase_to_rows.values():
                for row in rows:
                    uri = row.get("article") or ""
                    if "#art_" in uri:
                        ref = uri.rsplit("#art_", 1)[1]
                        # Strip sub-node suffix to mirror what the real
                        # BM25 chunks carry (chunker stores canonical refs).
                        canon_ref = _ART_SUFFIX_RE.sub("", "art_" + ref)
                        if canon_ref.startswith("art_"):
                            canon_ref = canon_ref[len("art_"):]
                        key = (canon_doc_id, canon_ref)
                        if key in seen:
                            continue
                        seen.add(key)
                        bm25_hits.append(
                            _bm25(canon_doc_id, canon_ref, row.get("text") or "")
                        )

    bm25 = MagicMock()
    bm25.search.return_value = list(bm25_hits or [])

    dense = MagicMock()
    if paraphrase_dense_hits:
        # First call (original query) returns dense_hits, subsequent return paraphrase hits
        # round-robin.
        seq = [list(dense_hits or [])]
        seq.extend(list(h) for h in paraphrase_dense_hits)
        dense.search.side_effect = seq
    else:
        dense.search.return_value = list(dense_hits or [])

    registry = MagicMock()
    registry.get_doc.return_value = MagicMock(doc_title=doc_title)
    registry.resolve_alias.side_effect = lambda x: None

    router = _stub_router(routed_ids if routed_ids is not None else [])
    sparql_fn = _stub_sparql(phrase_to_rows or {})
    paraphrase_fn = _stub_paraphrase(paraphrases or [])
    summarizer_fn = _stub_summarizer(summary)
    verifier_fn = _stub_verifier()
    adu_fn = _stub_adu(claim_for=adu_claim_for, default=adu_default)

    handler = ConceptualDefinitionalHandler(
        kg=object(),
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=MagicMock(),
        router=router,
        sparql_fn=sparql_fn,
        paraphrase_fn=paraphrase_fn,
        summarizer_fn=summarizer_fn,
        verifier_fn=verifier_fn,
        adu_extract_fn=adu_fn,
        **kwargs,
    )
    return handler, {
        "bm25": bm25, "dense": dense, "registry": registry, "router": router,
        "sparql_fn": sparql_fn, "paraphrase_fn": paraphrase_fn,
        "summarizer_fn": summarizer_fn, "verifier_fn": verifier_fn,
        "adu_extract_fn": adu_fn,
    }


# ---------------------------------------------------------------------------
# Defaults / contract
# ---------------------------------------------------------------------------


def test_defaults_match_handoff_design():
    assert DEFAULT_TOP_K_CANDIDATES == 5
    # Verifier OFF by default — same R3 finding (KG+ADU is the
    # discriminator, not a generic relevance verifier).
    assert DEFAULT_VERIFY_TOP_N == 0
    # F4: reverted R9.2's 2 back to 5. F3 showed CD regressed −0.010
    # at top_k=2 (gold often 2-3 articles, top-2 dropped one).
    assert DEFAULT_FINAL_TOP_K == 5
    assert DEFAULT_K_EACH == 30
    assert DEFAULT_VERIFY_THRESHOLD == 0.4
    assert DEFAULT_ROUTE_TOP_N == 3
    assert DEFAULT_PARAPHRASE_COUNT == 3
    assert DEFAULT_ADU_EXTRACT_TOP_N == 2
    assert DEFAULT_MIN_KG_HITS == 1
    assert DEFAULT_KG_LIMIT == 50
    assert SUPPORT_SPAN_LEN == 280
    assert TELEMETRY_BASELINE == "rlm_conceptual_definitional"


def test_f4_default_final_top_k_locked_at_5():
    """F4 reverted R9.2's CD-side 2→5. TF half of R9.2 stays at 2."""
    from akn_rlm.rlm.handlers import conceptual_definitional as cd_mod
    assert cd_mod.DEFAULT_FINAL_TOP_K == 5


def test_factory_builds_handler_instance():
    bm25 = MagicMock(); dense = MagicMock(); registry = MagicMock()
    registry.get_doc.return_value = None
    registry.resolve_alias.return_value = None
    h = build_conceptual_definitional_handler(
        kg=None, bm25=bm25, dense=dense, registry=registry,
        llm_pool=MagicMock(), router=_stub_router([]),
        sparql_fn=lambda _q: [], paraphrase_fn=lambda *a, **k: [],
    )
    assert isinstance(h, ConceptualDefinitionalHandler)


def test_run_returns_required_answer_keys():
    h, _ = _make_handler(
        bm25_hits=[_bm25("75-8_1975-09-26", "124", "كل خطأ يسبب ضررا للغير")],
        dense_hits=[_dense("75-8_1975-09-26", "124", "كل خطأ يسبب ضررا للغير")],
        routed_ids=["75-8_1975-09-26"],
        phrase_to_rows={
            "المسؤولية المدنية": [
                {"article": "https://legal.dz/resource/law/1975-09-26/75-8#art_124",
                 "text": "كل خطأ يسبب ضررا للغير."},
            ],
        },
    )
    out = h.run("ما هي شروط قيام المسؤولية المدنية التقصيرية في القانون المدني الجزائري؟")
    for key in (
        "answer_text", "abstention", "abstention_reason", "citations",
        "reasoning_chain", "trajectory", "tokens_used", "depth_max_reached",
        "_telemetry",
    ):
        assert key in out
    assert out["abstention"] is False
    assert out["_telemetry"]["baseline"] == TELEMETRY_BASELINE


def test_telemetry_carries_phrases_routing_and_kg_hits():
    h, _ = _make_handler(
        routed_ids=["75-8_1975-09-26"],
        phrase_to_rows={
            "المسؤولية المدنية": [
                {"article": "https://legal.dz/resource/law/1975-09-26/75-8#art_124",
                 "text": "خطأ"}],
        },
    )
    out = h.run("ما شروط المسؤولية المدنية التقصيرية؟")
    tel = out["_telemetry"]
    assert tel["routed_doc_ids"] == ["75-8_1975-09-26"]
    assert "المسؤولية المدنية" in tel["concept_phrases"]
    assert tel["kg_hits"] >= 1
    assert tel["kg_used"] is True


# ---------------------------------------------------------------------------
# Concept phrase extraction
# ---------------------------------------------------------------------------


def test_content_tokens_drop_stopwords_and_question_words():
    toks = _content_tokens("ما هو مفهوم الدفع بعدم الدستورية في الجزائر؟")
    # "ما", "هو", "في" are stopwords; "الدفع" → "دفع" (stripped); "الجزائر" stopword
    assert "مفهوم" in toks
    assert "ما" not in toks
    assert "هو" not in toks


def test_content_tokens_keep_arabic_definite_article():
    """Surface form is preserved — CONTAINS is a literal substring match
    and the embedded ``ال`` of the second token would break a stripped
    bigram like "قانون مدني" against the KG text "القانون المدني".
    """
    toks = _content_tokens("القانون المدني")
    assert "القانون" in toks
    assert "المدني" in toks


def test_extract_concept_phrases_prefers_bigrams():
    phrases = _extract_concept_phrases(
        "ما الفرق بين الاتفاقية الجماعية واتفاقية المؤسسة؟"
    )
    # Bigram form should appear before single tokens
    assert any("اتفاقية" in p and " " in p for p in phrases)


def test_extract_concept_phrases_includes_trigrams_for_long_concepts():
    phrases = _extract_concept_phrases(
        "ما مفهوم الدفع بعدم الدستورية وكيف يمارسه المتقاضون؟"
    )
    # Should contain at least one trigram
    assert any(p.count(" ") == 2 for p in phrases)


def test_extract_concept_phrases_empty_query_returns_empty():
    assert _extract_concept_phrases("") == []
    assert _extract_concept_phrases("؟ ؟") == []


def test_extract_concept_phrases_dedupes_overlapping():
    phrases = _extract_concept_phrases("الاتفاقية الجماعية الاتفاقية الجماعية")
    # The bigram appears once even though the source has it twice
    assert phrases.count("الاتفاقية الجماعية") == 1


# ---------------------------------------------------------------------------
# KG entity lookup
# ---------------------------------------------------------------------------


def test_kg_phrase_lookup_returns_uri_scores_and_spans():
    sparql = _stub_sparql({
        "الاتفاقية الجماعية": [
            {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
             "text": "تحدد الاتفاقية الجماعية"},
            {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_117",
             "text": "تبرم الاتفاقية الجماعية"},
        ],
        "اتفاقية المؤسسة": [
            # art_115 hit by both phrases → score=2
            {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
             "text": "اتفاقية المؤسسة"},
        ],
    })
    scores, spans = _kg_phrase_lookup(
        sparql,
        ["الاتفاقية الجماعية", "اتفاقية المؤسسة"],
    )
    assert scores["https://legal.dz/resource/law/1990-04-21/90-11#art_115"] == 2.0
    assert scores["https://legal.dz/resource/law/1990-04-21/90-11#art_117"] == 1.0
    # span captured from FIRST matching phrase
    assert "تحدد" in spans["https://legal.dz/resource/law/1990-04-21/90-11#art_115"]


def test_kg_phrase_lookup_no_sparql_fn_returns_empty():
    scores, spans = _kg_phrase_lookup(None, ["انـيَ"])
    assert scores == {} and spans == {}


def test_kg_phrase_lookup_skips_failing_phrase():
    """One failing SPARQL call must not poison the rest of the lookup."""
    raises = MagicMock()

    def _fn(query: str) -> list[dict]:
        if "BAD" in query:
            raise RuntimeError("kg blew up")
        return [{"article": "https://legal.dz/resource/law/2020-12-30/2020#art_188",
                 "text": "حق الدفع"}]

    raises.side_effect = _fn
    scores, _ = _kg_phrase_lookup(raises, ["BAD", "الدفع"])
    assert scores == {
        "https://legal.dz/resource/law/2020-12-30/2020#art_188": 1.0,
    }


# ---------------------------------------------------------------------------
# URI resolution
# ---------------------------------------------------------------------------


def test_uri_to_doc_ref_law():
    registry = MagicMock()
    registry.resolve_alias.return_value = None
    doc_id, ref = _uri_to_doc_ref(
        "https://legal.dz/resource/law/1990-04-21/90-11#art_115", registry,
    )
    assert doc_id == "90-11"  # registry alias didn't resolve, fallback to num
    assert ref == "115"


def test_uri_to_doc_ref_uses_registry_alias():
    registry = MagicMock()
    registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    doc_id, ref = _uri_to_doc_ref(
        "https://legal.dz/resource/law/1990-04-21/90-11#art_115", registry,
    )
    assert doc_id == "90-11_1990-04-21"
    assert ref == "115"


def test_uri_to_doc_ref_strips_subnode_suffix():
    registry = MagicMock()
    registry.resolve_alias.return_value = None
    doc_id, ref = _uri_to_doc_ref(
        "https://legal.dz/resource/law/1990-04-21/90-11#art_115_para_1", registry,
    )
    assert doc_id == "90-11"
    assert ref == "115"


def test_uri_to_doc_ref_canonicalises_bis():
    """art_9_bis surface form survives the canonicaliser unchanged."""
    registry = MagicMock()
    registry.resolve_alias.return_value = None
    _, ref = _uri_to_doc_ref(
        "https://legal.dz/resource/law/1990-04-21/90-11#art_9_bis", registry,
    )
    assert ref == "9_bis"


def test_uri_to_doc_ref_malformed_returns_none():
    registry = MagicMock()
    registry.resolve_alias.return_value = None
    assert _uri_to_doc_ref("not a uri", registry) == (None, None)
    assert _uri_to_doc_ref("", registry) == (None, None)


# ---------------------------------------------------------------------------
# Default routing / pipeline
# ---------------------------------------------------------------------------


def test_default_does_not_call_verifier():
    """With DEFAULT_VERIFY_TOP_N=0 the verifier path is bypassed entirely."""
    h, deps = _make_handler(
        bm25_hits=[_bm25("90-11_1990-04-21", "115", "نص")],
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                "text": "نص",
            }],
        },
    )
    h.run("ما الفرق بين الاتفاقية الجماعية واتفاقية المؤسسة؟")
    deps["verifier_fn"].assert_not_called()


def test_router_called_with_route_top_n():
    h, deps = _make_handler(routed_ids=["90-11_1990-04-21"], route_top_n=4)
    h.run("ما الفرق بين الاتفاقية الجماعية والمؤسسة؟")
    deps["router"].route.assert_called_once()
    assert deps["router"].route.call_args.kwargs.get("top_n") == 4


def test_kg_hits_filtered_by_routed_doc_ids():
    """KG candidates outside the routed set are filtered out."""
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [
                {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                 "text": "تحدد"},
                {"article": "https://legal.dz/resource/law/2025-01-01/OTHER#art_1",
                 "text": "غير مرتبط"},
            ],
        },
    )
    # Make registry alias resolve 90-11 → 90-11_1990-04-21 so it joins routed set
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    cit_doc_ids = {c["doc_id"] for c in out["citations"]}
    assert cit_doc_ids == {"90-11_1990-04-21"}


def test_fused_full_pool_fallback_when_routed_filter_empty():
    """If the routed-doc filter would wipe every fused candidate, the
    handler falls back to the unrestricted fused pool — better to
    return something than to abstain when the doc-router was wrong.
    """
    h, _ = _make_handler(
        # Router predicts the wrong doc; our only retrieval hit is in
        # 90-11_1990-04-21, NOT in UNRELATED_DOC.
        routed_ids=["UNRELATED_DOC"],
        bm25_hits=[_bm25("90-11_1990-04-21", "115", "تحدد")],
        dense_hits=[_dense("90-11_1990-04-21", "115", "تحدد")],
        phrase_to_rows={},
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    assert out["abstention"] is False
    assert any(c["doc_id"] == "90-11_1990-04-21" for c in out["citations"])


# ---------------------------------------------------------------------------
# Paraphrase fallback path
# ---------------------------------------------------------------------------


def test_paraphrase_fallback_triggered_when_kg_empty():
    """KG returns nothing → paraphraser is called and dense.search runs over
    the original query AND each paraphrase."""
    paraphrase_dense_hits = [
        [_dense("84-11_1984-06-09", "47", "أسباب انحلال عقد الزواج")],
        [_dense("84-11_1984-06-09", "48", "الطلاق بإرادة الزوج")],
        [_dense("84-11_1984-06-09", "47", "duplicate")],
    ]
    h, deps = _make_handler(
        routed_ids=["84-11_1984-06-09"],
        phrase_to_rows={},  # no KG hits
        paraphrases=["q1", "q2", "q3"],
        bm25_hits=[],
        dense_hits=[_dense("84-11_1984-06-09", "47", "أسباب انحلال عقد الزواج")],
        paraphrase_dense_hits=paraphrase_dense_hits,
    )
    out = h.run("ما هي أسباب انحلال عقد الزواج؟")
    deps["paraphrase_fn"].assert_called_once()
    # 1 original-query call + 3 paraphrase calls = 4 dense.search calls
    assert deps["dense"].search.call_count == 4
    assert out["abstention"] is False
    assert out["_telemetry"]["paraphrases"] == ["q1", "q2", "q3"]
    assert out["_telemetry"]["kg_used"] is False


def test_paraphrase_called_unconditionally_to_widen_recall():
    """Paraphrase generation runs once per query (regardless of whether
    the KG concept-search returned hits) so dense retrieval over the
    paraphrases can widen recall on definitional queries where the
    article body uses different surface phrasing.
    """
    h, deps = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                "text": "نص"}],
        },
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    h.run("ما الاتفاقية الجماعية؟")
    deps["paraphrase_fn"].assert_called_once()


def test_paraphrase_count_zero_disables_paraphrase_call():
    """Setting ``paraphrase_count=0`` skips the paraphraser entirely
    (escape hatch for ablations / cost-sensitive runs).
    """
    h, deps = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                "text": "نص"}],
        },
        paraphrase_count=0,
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    h.run("ما الاتفاقية الجماعية؟")
    deps["paraphrase_fn"].assert_not_called()


def test_paraphraser_failure_does_not_crash():
    """Paraphraser raising must not crash the handler — it falls through to
    fused retrieval over the original query alone."""
    raising = MagicMock(side_effect=RuntimeError("LLM unavailable"))
    h, _ = _make_handler(
        routed_ids=["84-11_1984-06-09"],
        phrase_to_rows={},
        bm25_hits=[_bm25("84-11_1984-06-09", "47", "أسباب الانحلال")],
        dense_hits=[_dense("84-11_1984-06-09", "47", "أسباب الانحلال")],
    )
    h._paraphrase_fn = raising
    out = h.run("ما أسباب الانحلال؟")
    raising.assert_called_once()
    assert out["abstention"] is False


# ---------------------------------------------------------------------------
# ADU extraction
# ---------------------------------------------------------------------------


def test_adu_extractor_called_for_top_n_candidates():
    h, deps = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [
                {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                 "text": "نص1"},
                {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_117",
                 "text": "نص2"},
                {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_126",
                 "text": "نص3"},
            ],
        },
        adu_extract_top_n=2,
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    h.run("ما الاتفاقية الجماعية؟")
    # ADU only ran on the top-2 candidates
    assert deps["adu_extract_fn"].call_count == 2


def test_adu_claim_and_ground_become_supporting_span():
    h, _ = _make_handler(
        routed_ids=["75-8_1975-09-26"],
        phrase_to_rows={
            "المسؤولية المدنية": [{
                "article": "https://legal.dz/resource/law/1975-09-26/75-8#art_124",
                "text": "كل عمل يرتكب خطأ يسبب ضرراً للغير يلزم مرتكبه بالتعويض",
            }],
        },
        adu_default={
            "claim": "كل فعل يسبب ضرراً يلزم مرتكبه بالتعويض.",
            "ground": "خطأ + ضرر + علاقة سببية.",
            "warrant": "مبدأ المسؤولية التقصيرية.",
            "backing": "",
            "rebuttal": "",
        },
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "75-8_1975-09-26" if x == "75-8" else None
    )
    out = h.run("شروط المسؤولية المدنية التقصيرية")
    cit = out["citations"][0]
    assert "كل فعل يسبب ضرراً" in cit["supporting_span"]
    assert "خطأ + ضرر" in cit["supporting_span"]
    assert cit["adu"]["claim"] == "كل فعل يسبب ضرراً يلزم مرتكبه بالتعويض."
    assert cit["adu"]["ground"] == "خطأ + ضرر + علاقة سببية."


def test_adu_failure_falls_back_to_text_prefix():
    """If ADU extraction fails for a candidate, the citation falls back to
    text[:280] for the supporting_span."""
    raising = MagicMock(side_effect=RuntimeError("ADU LLM down"))
    h, _ = _make_handler(
        routed_ids=["75-8_1975-09-26"],
        phrase_to_rows={
            "المسؤولية المدنية": [{
                "article": "https://legal.dz/resource/law/1975-09-26/75-8#art_124",
                "text": "النص الكامل للمادة 124."}],
        },
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "75-8_1975-09-26" if x == "75-8" else None
    )
    h._adu_extract_fn = raising
    out = h.run("شروط المسؤولية المدنية")
    cit = out["citations"][0]
    assert cit["supporting_span"] == "النص الكامل للمادة 124."
    assert cit["adu"]["claim"] == ""


def test_supporting_span_caps_at_280():
    long_claim = "ا" * 600
    h, _ = _make_handler(
        routed_ids=["75-8_1975-09-26"],
        phrase_to_rows={
            "المسؤولية المدنية": [{
                "article": "https://legal.dz/resource/law/1975-09-26/75-8#art_124",
                "text": "نص"}],
        },
        adu_default={"claim": long_claim, "ground": "", "warrant": "",
                     "backing": "", "rebuttal": ""},
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "75-8_1975-09-26" if x == "75-8" else None
    )
    out = h.run("شروط المسؤولية المدنية")
    assert len(out["citations"][0]["supporting_span"]) == SUPPORT_SPAN_LEN


# ---------------------------------------------------------------------------
# Citation contract
# ---------------------------------------------------------------------------


def test_citation_carries_required_keys():
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                "text": "نص"}],
        },
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    cit = out["citations"][0]
    for key in (
        "doc_id", "article_ref", "doc_title", "supporting_span", "text",
        "confidence", "kg_hit", "adu", "verifier_relevant",
    ):
        assert key in cit
    assert cit["kg_hit"] is True
    assert cit["doc_title"] == "قانون مدني"


def test_citation_canonicalises_article_ref():
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_9_bis",
                "text": "نص"}],
        },
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    assert out["citations"][0]["article_ref"] == "9_bis"


def test_dedup_across_uris_keeps_highest_confidence():
    """Two URIs collapse to the same (doc_id, ref) — dedup keeps best confidence."""
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [
                # Same article via two URI forms (only one would actually exist
                # in the live KG, but the dedup contract should hold either way).
                {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                 "text": "نص1"},
                {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115_para_1",
                 "text": "نص2"},
            ],
        },
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    keys = [(c["doc_id"], c["article_ref"]) for c in out["citations"]]
    assert keys == [("90-11_1990-04-21", "115")]


def test_top_k_truncation():
    # Build 8 KG hits, top_k_candidates=3 → at most 3 citations
    rows = [
        {"article": f"https://legal.dz/resource/law/1990-04-21/90-11#art_{i}",
         "text": f"text-{i}"}
        for i in range(1, 9)
    ]
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={"الاتفاقية الجماعية": rows},
        top_k_candidates=3,
        adu_extract_top_n=3,
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    assert len(out["citations"]) <= 3


def test_final_top_k_truncation():
    rows = [
        {"article": f"https://legal.dz/resource/law/1990-04-21/90-11#art_{i}",
         "text": f"t-{i}"}
        for i in range(1, 9)
    ]
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={"الاتفاقية الجماعية": rows},
        final_top_k=2,
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    assert len(out["citations"]) <= 2


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def test_summary_used_as_answer_text():
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                "text": "نص"}],
        },
        summary="الاتفاقية الجماعية على مستوى القطاع. اتفاقية المؤسسة على مستوى المؤسسة.",
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    assert "على مستوى القطاع" in out["answer_text"]


def test_null_summary_falls_back_to_template():
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                "text": "نص المادة"}],
        },
        summary=None,
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    assert out["answer_text"].startswith("وفقًا لـ")


def test_summarizer_exception_falls_back_to_template():
    raising = MagicMock(side_effect=RuntimeError("summary failed"))
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                "text": "نص"}],
        },
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    h._summarizer_fn = raising
    out = h.run("ما الاتفاقية الجماعية؟")
    raising.assert_called_once()
    assert out["abstention"] is False
    assert out["answer_text"].startswith("وفقًا لـ")


# ---------------------------------------------------------------------------
# Verifier opt-in path
# ---------------------------------------------------------------------------


def test_verifier_opt_in_drops_irrelevant_candidates():
    """When verify_top_n>0, the verifier runs and dropping irrelevant
    candidates moves their slot."""
    def _fn(_pool, _q, article, _model):
        # Reject art_117 only.
        if article.get("article_ref") == "117":
            return {"relevant": False, "supporting_span": None,
                    "contradicting_span": None, "confidence": 0.0}
        return {"relevant": True, "supporting_span": None,
                "contradicting_span": None, "confidence": 0.95}
    verifier = MagicMock(side_effect=_fn)
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [
                {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                 "text": "ن1"},
                {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_117",
                 "text": "ن2"},
                {"article": "https://legal.dz/resource/law/1990-04-21/90-11#art_126",
                 "text": "ن3"},
            ],
        },
        verify_top_n=3,
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    h._verifier_fn = verifier
    out = h.run("ما الاتفاقية الجماعية؟")
    refs = {c["article_ref"] for c in out["citations"]}
    assert "117" not in refs
    assert verifier.call_count >= 2


def test_verifier_low_confidence_drops_candidate():
    def _fn(_pool, _q, _article, _model):
        return {"relevant": True, "supporting_span": None,
                "contradicting_span": None, "confidence": 0.1}
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                "text": "ن"}],
        },
        verify_top_n=2,
        verify_threshold=0.5,
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    h._verifier_fn = MagicMock(side_effect=_fn)
    out = h.run("ما الاتفاقية الجماعية؟")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_verified_articles"


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------


def test_empty_query_abstains_without_calls():
    h, deps = _make_handler()
    out = h.run("")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "empty_query"
    deps["bm25"].search.assert_not_called()
    deps["dense"].search.assert_not_called()
    deps["sparql_fn"].assert_not_called()
    deps["paraphrase_fn"].assert_not_called()


def test_no_hits_abstains():
    """KG empty AND retrieval empty AND no paraphrases yield → abstain."""
    h, _ = _make_handler(
        routed_ids=[],
        phrase_to_rows={},
        bm25_hits=[],
        dense_hits=[],
        paraphrases=[],
    )
    out = h.run("ما هي شروط شيء غير موجود؟")
    assert out["abstention"] is True
    assert out["abstention_reason"] == "no_hits"


# ---------------------------------------------------------------------------
# End-to-end shape
# ---------------------------------------------------------------------------


def test_answer_to_result_consumes_handler_output():
    """Smoke: the handler output dict is consumed by _answer_to_result without
    raising — proves baseline-shape compatibility."""
    from akn_rlm.eval.runner import _answer_to_result
    h, _ = _make_handler(
        routed_ids=["90-11_1990-04-21"],
        phrase_to_rows={
            "الاتفاقية الجماعية": [{
                "article": "https://legal.dz/resource/law/1990-04-21/90-11#art_115",
                "text": "نص"}],
        },
    )
    h._registry.resolve_alias.side_effect = (
        lambda x: "90-11_1990-04-21" if x == "90-11" else None
    )
    out = h.run("ما الاتفاقية الجماعية؟")
    out["_latency_s"] = 0.05
    record = {
        "id": "lab_cd_q01",
        "query_type": "conceptual_definitional",
        "query": "ما الاتفاقية الجماعية؟",
        "gold_doc_ids": ["90-11_1990-04-21"],
        "gold_article_ids": ["90-11_1990-04-21#art_115"],
        "answerable": True,
        "difficulty": "medium",
    }
    result = _answer_to_result(record, out)
    assert result["query_type"] == "conceptual_definitional"
    # _answer_to_result emits ranked pred_doc_ids / pred_article_ids /
    # predicted_citations — not the raw "citations" key.
    assert result.get("predicted_citations")
    assert "90-11_1990-04-21" in (result.get("pred_doc_ids") or [])


# ---------------------------------------------------------------------------
# Paraphrase generator helper
# ---------------------------------------------------------------------------


def test_generate_paraphrases_parses_json_object():
    pool = MagicMock()
    pool.call.return_value = (
        '{"paraphrases": ["كيف يُعرَّف الدفع بعدم الدستورية؟", '
        '"اشرح مفهوم DIC.", "ما تعريف الدفع؟"]}'
    )
    out = _generate_paraphrases(pool, "ما مفهوم الدفع؟", n=3, sub_model="m")
    assert len(out) == 3
    assert out[0].startswith("كيف يُعرَّف")


def test_generate_paraphrases_parses_with_surrounding_prose():
    pool = MagicMock()
    pool.call.return_value = (
        'Sure, here you go:\n{"paraphrases": ["q1", "q2"]}\nThanks.'
    )
    out = _generate_paraphrases(pool, "ما مفهوم الدفع؟", n=2, sub_model="m")
    assert out == ["q1", "q2"]


def test_generate_paraphrases_caps_at_n():
    pool = MagicMock()
    pool.call.return_value = '{"paraphrases": ["a", "b", "c", "d", "e"]}'
    out = _generate_paraphrases(pool, "q", n=3, sub_model="m")
    assert out == ["a", "b", "c"]


def test_generate_paraphrases_drops_identical_to_query():
    pool = MagicMock()
    pool.call.return_value = '{"paraphrases": ["q", "x", "q", "y"]}'
    out = _generate_paraphrases(pool, "q", n=4, sub_model="m")
    assert "q" not in out
    assert out == ["x", "y"]


def test_generate_paraphrases_empty_query_returns_empty():
    pool = MagicMock()
    out = _generate_paraphrases(pool, "", n=3, sub_model="m")
    assert out == []
    pool.call.assert_not_called()


def test_generate_paraphrases_zero_n_returns_empty():
    pool = MagicMock()
    out = _generate_paraphrases(pool, "q", n=0, sub_model="m")
    assert out == []
    pool.call.assert_not_called()


def test_generate_paraphrases_invalid_json_returns_empty():
    pool = MagicMock()
    pool.call.return_value = "not json at all"
    out = _generate_paraphrases(pool, "q", n=3, sub_model="m")
    assert out == []


def test_generate_paraphrases_llm_exception_returns_empty():
    pool = MagicMock()
    pool.call.side_effect = RuntimeError("LLM down")
    out = _generate_paraphrases(pool, "q", n=3, sub_model="m")
    assert out == []


def test_generate_paraphrases_accepts_positional_args():
    # Locks in the ParaphraseFn protocol: (llm_pool, query, n, sub_model) all
    # positional. The dispatcher / handler call site passes positional, so a
    # keyword-only `n`/`sub_model` would surface as
    # `_generate_paraphrases() takes 2 positional arguments but 4 were given`
    # (this happened in the F2 RLM dispatcher run on every CD question).
    pool = MagicMock()
    pool.call.return_value = '{"paraphrases": ["a", "b"]}'
    out = _generate_paraphrases(pool, "q", 3, "m")  # all positional
    assert out == ["a", "b"]
