"""Hard gate: every citation must pass has_article() before output."""
from __future__ import annotations

import logging

from akn_rlm.gates.base import GateResult
from akn_rlm.normalizers import canonical_article_ref

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-citation check
# ---------------------------------------------------------------------------

def _normalize_citation(citation) -> dict:
    """Coerce a citation to dict form with canonical article_ref.

    LLMs sometimes return strings like "84-11_1984-06-09#art_54" or
    "84-11_1984-06-09:54" instead of {"doc_id": ..., "article_ref": ...}.
    article_ref is canonicalised so downstream art_key building is consistent.
    """
    if isinstance(citation, dict):
        c = dict(citation)
        ref = str(c.get("article_ref", "")).strip()
        c["article_ref"] = canonical_article_ref(ref)
        return c
    s = str(citation).strip()
    for sep in ("#", ":", "/"):
        if sep in s:
            parts = s.split(sep, 1)
            return {
                "doc_id": parts[0].strip(),
                "article_ref": canonical_article_ref(parts[1].strip()),
            }
    return {"doc_id": s, "article_ref": ""}


def check(registry, citation) -> GateResult:
    """Return GateResult for a single citation."""
    citation = _normalize_citation(citation)
    doc_id = citation.get("doc_id", "")
    article_ref = citation.get("article_ref", "")

    if not doc_id or not article_ref:
        return GateResult(passed=False, score=0.0, details=[{
            "citation": citation,
            "reason": "missing_doc_id_or_article_ref",
        }])

    ok = registry.has_article(doc_id, article_ref)
    if not ok:
        log.warning("Citation not in corpus: doc_id=%s article_ref=%s", doc_id, article_ref)
        return GateResult(passed=False, score=0.0, details=[{
            "citation": citation,
            "reason": "not_in_corpus",
        }])

    return GateResult(passed=True, score=1.0)


# ---------------------------------------------------------------------------
# Batch gate
# ---------------------------------------------------------------------------

def run_gate(registry, citations: list) -> GateResult:
    """Run citation existence gate on all citations; return aggregate result."""
    if not citations:
        return GateResult(passed=True, score=1.0)

    failures: list[dict] = []
    for c in citations:
        result = check(registry, c)
        if not result.passed:
            failures.extend(result.details)

    passed = len(failures) == 0
    score = 1.0 - len(failures) / len(citations)
    return GateResult(passed=passed, score=score, details=failures)


# ---------------------------------------------------------------------------
# Backwards-compatible filter helper (used by RootController)
# ---------------------------------------------------------------------------

def filter_citations(
    registry,
    citations: list,
) -> tuple[list[dict], list[dict]]:
    """Split citations into (valid, rejected).  Backwards-compatible API."""
    valid: list[dict] = []
    rejected: list[dict] = []
    for c in citations:
        normed = _normalize_citation(c)
        if check(registry, normed).passed:
            valid.append(normed)
        else:
            rejected.append(normed)
    if rejected:
        log.warning("Dropped %d citation(s) that failed existence gate", len(rejected))
    return valid, rejected
