"""
RDF Knowledge Graph loader — Phase A.

Loads algerian_legal_kg.ttl into rdflib.Graph, builds a temporal index of
amendment chains, and exposes get_article_at_date().

Temporal index structure (temporal_index.json):
{
  "<canonical_doc_id>#<eid>": {
    "doc_id": "...",
    "eid": "...",
    "versions": [
      {"date": "YYYY-MM-DD", "text": "...", "status": "original|amended|repealed"},
      ...
    ]
  }
}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# RDF namespaces used in the Algerian Legal KG
_DZDOC_NS  = "https://legal.dz/ontology/document#"
_DZNORM_NS = "https://legal.dz/ontology/norm#"


def load_kg(kg_path: Optional[Path] = None) -> Any:
    """Load the merged TTL into an rdflib.Graph and return it."""
    try:
        import rdflib
    except ImportError as exc:
        raise ImportError("rdflib is required: pip install rdflib") from exc

    from akn_rlm.config import get_kg_path
    path = kg_path or get_kg_path()

    g = rdflib.Graph()
    logger.info("Loading RDF KG from %s ...", path.name)
    g.parse(str(path), format="turtle")
    logger.info("RDF graph loaded: %d triples", len(g))

    g.bind("dzdoc",  rdflib.Namespace(_DZDOC_NS))
    g.bind("dznorm", rdflib.Namespace(_DZNORM_NS))
    return g


def build_temporal_index(g: Any) -> Dict[str, Any]:
    """Build temporal index from the RDF graph.

    Queries for amendment events of the form:
      ?article dznorm:hasVersion ?version .
      ?version dznorm:versionDate ?date ;
               dznorm:versionText ?text ;
               dznorm:versionStatus ?status .

    Returns a dict keyed by "<canonical_doc_id>#<article_eid>" → version list.
    """
    try:
        from rdflib import URIRef, Literal
        from rdflib.namespace import RDF
    except ImportError:
        return {}

    DZNORM = g.namespace_manager.store.namespace("dznorm")
    if DZNORM is None:
        # Fallback to hardcoded namespace
        import rdflib
        DZNORM = rdflib.Namespace(_DZNORM_NS)

    temporal: Dict[str, Any] = {}

    QUERY = """
    PREFIX dznorm: <https://legal.dz/ontology/norm#>
    PREFIX dzdoc:  <https://legal.dz/ontology/document#>

    SELECT ?article_uri ?version_date ?version_text ?version_status
    WHERE {
        ?article_uri dznorm:hasVersion ?version .
        ?version     dznorm:versionDate ?version_date .
        OPTIONAL { ?version dznorm:versionText   ?version_text . }
        OPTIONAL { ?version dznorm:versionStatus ?version_status . }
    }
    ORDER BY ?article_uri ?version_date
    """

    try:
        results = g.query(QUERY)
    except Exception as exc:
        logger.warning("Temporal SPARQL query failed: %s", exc)
        return {}

    for row in results:
        article_uri    = str(row.article_uri)
        version_date   = str(row.version_date) if row.version_date else ""
        version_text   = str(row.version_text) if row.version_text else ""
        version_status = str(row.version_status) if row.version_status else "original"

        # Extract doc_id and eid from URI:
        # URI pattern: https://legal.dz/ontology/document#<doc_id>/<eid>
        # or: /akn/dz/act/<type>/<date>/<num>/<eid>
        article_key = _uri_to_key(article_uri)
        if not article_key:
            continue

        if article_key not in temporal:
            temporal[article_key] = {
                "uri":      article_uri,
                "versions": [],
            }

        temporal[article_key]["versions"].append({
            "date":   version_date,
            "text":   version_text,
            "status": version_status,
        })

    # Sort versions by date
    for key in temporal:
        temporal[key]["versions"].sort(key=lambda v: v["date"])

    logger.info("Temporal index built: %d article chains", len(temporal))
    return temporal


def _uri_to_key(uri: str) -> str:
    """Convert an RDF article URI to a temporal index key."""
    # Pattern 1: https://legal.dz/ontology/document#<doc_id>/<eid>
    if _DZDOC_NS in uri:
        fragment = uri.replace(_DZDOC_NS, "")
        parts = fragment.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return fragment

    # Pattern 2: /akn/dz/act/<type>/<date>/<num>/main
    if "/akn/dz/" in uri:
        parts = uri.rstrip("/").split("/")
        if len(parts) >= 6:
            num  = parts[-2] if parts[-1] == "main" else parts[-1]
            date = parts[-3] if parts[-1] == "main" else parts[-2]
            # canonical_id = num_date
            canonical = f"{num}_{date}"
            return canonical

    return ""


def get_article_at_date(
    temporal_index: Dict[str, Any],
    article_key: str,
    date: str,
) -> Optional[str]:
    """Return the article text that was in force on the given date.

    article_key: "<canonical_doc_id>#<eid>" or "<canonical_doc_id>/<eid>"
    date:        "YYYY-MM-DD"

    Returns the text of the latest version whose date <= query date,
    or None if no version exists on or before that date.
    """
    key = article_key.replace("#", "/")
    entry = temporal_index.get(key) or temporal_index.get(article_key)
    if not entry:
        return None

    versions = entry.get("versions", [])
    applicable = [v for v in versions if v.get("date", "") <= date]
    if not applicable:
        return None

    return applicable[-1].get("text")


def save_temporal_index(temporal_index: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(temporal_index, fh, ensure_ascii=False, indent=2)
    logger.info("Temporal index saved to %s (%d entries)", path, len(temporal_index))


def load_temporal_index(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
