"""Hard gate: every citation's `supporting_span` must occur inside the
actual article text fetched from the registry/legal_env.

Why this gate exists
====================
Without it, the LLM can write any string into `supporting_span` even if the
text never appears in the cited article.  Example caught from smoke_05/fam_ra_q03:

    article 5 of Family Code is about engagement (الخطبة وعد بالزواج)
    LLM emitted citation:
       {"doc_id": "84-11_1984-06-09", "article_ref": "5",
        "supporting_span": "يسمح بالزواج بأكثر من زوجة واحدة"}      ← FABRICATED

The citation_existence gate accepts the citation because article 5 exists,
but the span is invented.  This new gate catches that.

Matching policy
===============
Both the candidate span and the article text are passed through
`normalize_arabic` first (alef/ya unification, diacritic stripping,
Arabic-Indic digit folding, whitespace collapse).  Then a normalized substring
check is done.  This tolerates minor reformatting by the LLM (extra spaces,
dropped diacritics) while still rejecting outright fabrications.

A short minimum length (default 12 normalized characters) prevents trivial
trigram matches like "في" or "من" from passing.
"""
from __future__ import annotations

import logging
from typing import Iterable

from akn_rlm.gates.base import GateResult
from akn_rlm.normalizers import normalize_arabic

log = logging.getLogger(__name__)


_MIN_SPAN_LEN_NORM = 12       # post-normalization characters
_CLAUSE_SPLIT_RE = None        # initialised lazily (regex import)
_MIN_CLAUSE_LEN_NORM = 12     # clauses shorter than this are ignored
_OVERLAP_PASS_THRESHOLD = 0.5  # fraction of meaningful clauses that must match


def _norm(text: str) -> str:
    """Normalize text for span comparison: arabic fold + lowercase + collapse spaces."""
    if not text:
        return ""
    return normalize_arabic(str(text)).lower()


def _split_clauses(text: str) -> list[str]:
    """Split a span into clauses on Arabic/Latin punctuation."""
    import re
    global _CLAUSE_SPLIT_RE
    if _CLAUSE_SPLIT_RE is None:
        # Splits on . ، , ; ? ! and Arabic question/comma marks
        _CLAUSE_SPLIT_RE = re.compile(r"[\.\،,؛;\?؟!]+")
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]


def check(
    legal_env,
    citation: dict,
    *,
    min_span_len: int = _MIN_SPAN_LEN_NORM,
    overlap_threshold: float = _OVERLAP_PASS_THRESHOLD,
) -> GateResult:
    """Verify supporting_span is grounded in the cited article text.

    Acceptance policy (in order):
      1. The full normalized span is a substring of normalized article text
         (exact quote — best case).
      2. OR at least ``overlap_threshold`` of the span's clauses (split on
         . ، , ; ? !) appear as substrings of the article. This tolerates
         pronoun/gender drift and trivial reordering when the LLM stitches
         multiple fragments together.

    Rejected when:
      - no supporting_span supplied -> pass (gate is no-op)
      - normalized span < min_span_len chars -> reason='span_too_short'
      - article not retrievable        -> reason='article_not_retrievable_for_span_check'
      - neither (1) nor (2) holds      -> reason='span_not_in_article'
    """
    doc_id = citation.get("doc_id", "")
    article_ref = str(citation.get("article_ref", ""))
    span = str(citation.get("supporting_span", "") or "").strip()

    if not span:
        return GateResult(passed=True, score=1.0)

    article = legal_env.get_article(doc_id, article_ref)
    if not article:
        return GateResult(passed=False, score=0.0, details=[{
            "citation": citation,
            "reason": "article_not_retrievable_for_span_check",
        }])

    article_text = article.get("text", "") if isinstance(article, dict) else ""

    norm_span = _norm(span)
    norm_text = _norm(article_text)

    if len(norm_span) < min_span_len:
        return GateResult(passed=False, score=0.0, details=[{
            "citation": citation,
            "reason": "span_too_short",
            "span_len_norm": len(norm_span),
            "min_required": min_span_len,
        }])

    # Path 1 — exact substring (the strict, best case)
    if norm_span in norm_text:
        return GateResult(passed=True, score=1.0)

    # Path 2 — clause-level overlap (tolerates minor pronoun/gender drift)
    clauses = _split_clauses(norm_span)
    meaningful = [c for c in clauses if len(c) >= _MIN_CLAUSE_LEN_NORM]
    if meaningful:
        matched = sum(1 for c in meaningful if c in norm_text)
        ratio = matched / len(meaningful)
        if ratio >= overlap_threshold:
            return GateResult(passed=True, score=ratio)

    log.warning(
        "Fabricated span: doc=%s art=%s span=%r…",
        doc_id, article_ref, span[:60],
    )
    return GateResult(passed=False, score=0.0, details=[{
        "citation": citation,
        "reason": "span_not_in_article",
        "clause_match_ratio": (matched / len(meaningful)) if meaningful else 0.0,
    }])


def filter_citations(
    legal_env,
    citations: Iterable[dict],
    *,
    min_span_len: int = _MIN_SPAN_LEN_NORM,
) -> tuple[list[dict], list[dict]]:
    """Split citations into (valid, rejected) by span existence.

    Symmetric to citation_existence.filter_citations() — caller plugs into
    the same retry/correction flow.
    """
    valid: list[dict] = []
    rejected: list[dict] = []
    for c in citations:
        result = check(legal_env, c, min_span_len=min_span_len)
        if result.passed:
            valid.append(c)
        else:
            rejected.append({**c, "_rejection_reason": result.details[0].get("reason", "")})
    if rejected:
        log.warning("Span gate dropped %d/%d citation(s)",
                    len(rejected), len(rejected) + len(valid))
    return valid, rejected


def run_gate(
    legal_env,
    citations: list,
    *,
    min_span_len: int = _MIN_SPAN_LEN_NORM,
) -> GateResult:
    """Aggregate gate result across all citations."""
    if not citations:
        return GateResult(passed=True, score=1.0)
    failures: list[dict] = []
    for c in citations:
        result = check(legal_env, c, min_span_len=min_span_len)
        if not result.passed:
            failures.extend(result.details)
    passed = len(failures) == 0
    score = 1.0 - len(failures) / len(citations)
    return GateResult(passed=passed, score=score, details=failures)
