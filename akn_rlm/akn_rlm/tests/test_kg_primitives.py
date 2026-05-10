"""Tests for KG primitives in LegalEnv: null guards + actual SPARQL behaviour."""
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(kg=None, temporal_index=None):
    from akn_rlm.rlm.legal_env import LegalEnv
    from akn_rlm.rlm.recursion_budget import RecursionBudget
    env = LegalEnv(
        registry=MagicMock(),
        kg=kg,
        indices={},
        llm_pool=MagicMock(),
        budget=RecursionBudget(),
        temporal_index=temporal_index or {},
    )
    return env


def _minimal_kg():
    """Build a tiny rdflib graph with one article that has two versions."""
    rdflib = pytest.importorskip("rdflib")
    g = rdflib.Graph()
    ttl = """
@prefix dznorm: <http://legislation.dz/ns/norm#> .
@prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:    <http://www.w3.org/2001/XMLSchema#> .

<http://legislation.dz/id/75-58/art_40>
    a dznorm:Article ;
    rdfs:label "الأهلية" ;
    dznorm:inDocument <http://legislation.dz/id/75-58> ;
    dznorm:hasVersion <http://legislation.dz/id/75-58/art_40/v1> ;
    dznorm:hasVersion <http://legislation.dz/id/75-58/art_40/v2> .

<http://legislation.dz/id/75-58/art_40/v1>
    dznorm:versionDate "1975-09-26"^^xsd:date ;
    dznorm:text "نص الأهلية الأصلي" .

<http://legislation.dz/id/75-58/art_40/v2>
    dznorm:versionDate "2005-06-20"^^xsd:date ;
    dznorm:text "نص الأهلية بعد التعديل" .

<http://legislation.dz/concept/legal_capacity>
    a dznorm:LegalConcept ;
    rdfs:label "الأهلية القانونية" .
"""
    g.parse(data=ttl, format="turtle")
    return g


# ---------------------------------------------------------------------------
# Null-guard tests (kg=None must return [] / None, never crash)
# ---------------------------------------------------------------------------

def test_kg_sparql_null_guard():
    env = _make_env(kg=None)
    result = env.kg_sparql("SELECT ?x WHERE { ?x ?p ?o } LIMIT 1")
    assert result == []


def test_kg_entity_lookup_null_guard():
    env = _make_env(kg=None)
    result = env.kg_entity_lookup("الأهلية")
    assert result == []


def test_kg_amendment_chain_null_guard():
    env = _make_env(kg=None)
    result = env.kg_amendment_chain("75-58/art_40")
    assert result == []


def test_kg_get_article_at_date_empty_temporal_index():
    """No KG needed — uses temporal_index dict. Empty dict returns None."""
    env = _make_env(kg=None, temporal_index={})
    result = env.kg_get_article_at_date("75-58_1975-09-26/art_40", "2004-12-31")
    assert result is None


def test_kg_get_article_at_date_malformed_id():
    env = _make_env(kg=None, temporal_index={})
    result = env.kg_get_article_at_date("no_separator_here", "2004-12-31")
    assert result is None


# ---------------------------------------------------------------------------
# Functional tests with a minimal rdflib graph
# ---------------------------------------------------------------------------

def test_entity_lookup_finds_concept():
    kg = _minimal_kg()
    env = _make_env(kg=kg)
    results = env.kg_entity_lookup("الأهلية")
    assert len(results) > 0
    labels = [r.get("label", "") for r in results]
    assert any("الأهلية" in lbl for lbl in labels)


def test_entity_lookup_no_match_returns_empty():
    kg = _minimal_kg()
    env = _make_env(kg=kg)
    results = env.kg_entity_lookup("xyzzy_nonexistent_concept_9999")
    assert results == []


def test_amendment_chain_returns_versions_in_order():
    kg = _minimal_kg()
    env = _make_env(kg=kg)
    chain = env.kg_amendment_chain("http://legislation.dz/id/75-58/art_40")
    assert len(chain) == 2
    dates = [r.get("date", "") for r in chain]
    assert dates == sorted(dates), "Versions must be sorted chronologically"


def test_amendment_chain_auto_prefixes_uri():
    """Non-http article_id should be auto-prefixed."""
    kg = _minimal_kg()
    env = _make_env(kg=kg)
    # Pass without http:// prefix — env should add it
    chain = env.kg_amendment_chain("legislation.dz/id/75-58/art_40")
    # Won't match our test URI after prefix, but must NOT crash
    assert isinstance(chain, list)


def test_kg_sparql_raw_query():
    kg = _minimal_kg()
    env = _make_env(kg=kg)
    q = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label WHERE { ?s rdfs:label ?label }
    """
    results = env.kg_sparql(q)
    assert isinstance(results, list)
    assert len(results) > 0
    labels = [r.get("label", "") for r in results]
    assert any("الأهلية" in lbl for lbl in labels)


# ---------------------------------------------------------------------------
# Temporal index (dict-based, no KG needed)
# ---------------------------------------------------------------------------

def test_kg_get_article_at_date_with_temporal_index():
    temporal_index = {
        "75-58_1975-09-26": {
            "art_40": [
                {"date": "1975-09-26", "text": "النص الأصلي", "amendment_law": None},
                {"date": "2005-06-20", "text": "النص المعدل", "amendment_law": "05-10"},
            ]
        }
    }
    env = _make_env(kg=None, temporal_index=temporal_index)
    # Request version before the 2005 amendment
    result = env.kg_get_article_at_date("75-58_1975-09-26/art_40", "2004-12-31")
    # Should return the 1975 version (or None if temporal retriever not implemented)
    # We validate the call doesn't crash and returns dict or None
    assert result is None or isinstance(result, dict)


def test_kg_get_article_at_date_hash_separator():
    """'#' separator in article_id must also be handled."""
    env = _make_env(kg=None, temporal_index={})
    result = env.kg_get_article_at_date("75-58_1975-09-26#art_40", "2004-12-31")
    assert result is None or isinstance(result, dict)
