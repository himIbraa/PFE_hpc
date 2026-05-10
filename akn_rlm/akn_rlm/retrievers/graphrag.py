"""SPARQL-based KG retriever for structured legal queries."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# SPARQL templates for common legal query patterns
_SPARQL_ARTICLE_BY_DOC = """
PREFIX dznorm: <http://legislation.dz/ns/norm#>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?article ?label ?text
WHERE {{
    ?article a dznorm:Article ;
             dznorm:inDocument <{doc_uri}> ;
             rdfs:label ?label .
    OPTIONAL {{ ?article dznorm:text ?text }}
}}
LIMIT {limit}
"""

_SPARQL_AMENDMENTS = """
PREFIX dznorm: <http://legislation.dz/ns/norm#>

SELECT ?version ?date ?text
WHERE {{
    <{article_uri}> dznorm:hasVersion ?version .
    ?version dznorm:versionDate ?date ;
             dznorm:text ?text .
}}
ORDER BY ?date
"""

_SPARQL_ENTITY_LOOKUP = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dznorm: <http://legislation.dz/ns/norm#>

SELECT ?uri ?type ?label
WHERE {{
    ?uri rdfs:label ?label .
    OPTIONAL {{ ?uri a ?type }}
    FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{surface}")))
}}
LIMIT 10
"""


def sparql_query(kg, query_str: str) -> list[dict[str, Any]]:
    """Execute a raw SPARQL SELECT against the in-memory rdflib graph."""
    try:
        results = kg.query(query_str)
        rows = []
        for row in results:
            rows.append({str(var): str(val) if val is not None else None
                         for var, val in zip(results.vars, row)})
        return rows
    except Exception as exc:
        log.error("SPARQL query failed: %s\nQuery: %s", exc, query_str[:200])
        return []


def entity_lookup(kg, surface: str) -> list[dict[str, Any]]:
    """Find KG entities matching a surface form (partial label match)."""
    q = _SPARQL_ENTITY_LOOKUP.format(surface=surface.replace('"', "'"))
    return sparql_query(kg, q)


def amendment_chain(kg, article_uri: str) -> list[dict[str, Any]]:
    """Return all versions of an article ordered by date."""
    q = _SPARQL_AMENDMENTS.format(article_uri=article_uri)
    return sparql_query(kg, q)


def articles_by_doc(kg, doc_uri: str, limit: int = 50) -> list[dict[str, Any]]:
    """List articles in a document."""
    q = _SPARQL_ARTICLE_BY_DOC.format(doc_uri=doc_uri, limit=limit)
    return sparql_query(kg, q)
