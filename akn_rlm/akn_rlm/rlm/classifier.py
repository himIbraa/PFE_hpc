"""Intent classifier for AlgerianLegalBench query types.

Maps a raw query string to an IntentResult covering all 8 benchmark types:
  rule_application | exact_article | multi_hop | unanswerable |
  layman | long_context | conceptual_definitional | temporal_factual

Classification strategy:
  1. Answerability pre-check — if foreign-law signals detected in the query
     itself, label answerability_hint = "probably_unanswerable".
  2. Regex routing — deterministic keyword patterns covering 7 of the 8 types.
     Default (no match) → rule_application.
  3. Long-context heuristic — queries mentioning 3+ legal entities or spanning
     multiple procedural stages are promoted to long_context.

IntentResult is a frozen dataclass so it is safe to store in pipeline state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from akn_rlm.normalizers import normalize_query

# ---------------------------------------------------------------------------
# Compiled patterns (ordered — first match wins; checked in order below)
# ---------------------------------------------------------------------------

_TEMPORAL_FACTUAL_KW = re.compile(
    r"(?:منذ متى|تاريخ صدور|سنة.*إصدار|"
    r"متى.*صدر|متى.*نشر|متى.*أُصدر|متى.*صار نافذ|"
    r"when.*promulgated|when.*published|date.*enacted|year.*issued)",
    re.IGNORECASE | re.UNICODE,
)

_MULTI_HOP_KW = re.compile(
    r"(?:و(?:القانون|الق[اأ]نون)|both.*and|"
    r"العلاقة بين|interaction between|combined|"
    r"متعدد القوانين|multi.?law|"
    r"بين.*وبين|كلا القانونين)",
    re.IGNORECASE | re.UNICODE,
)

_EXACT_ARTICLE_KW = re.compile(
    r"(?:(?:الم[اآ]دة|art(?:icle)?\.?)\s*\d+|"
    r"نص الم[اآ]دة\s*\d+|"
    r"\bالم[اآ]دة\s+(?:الأولى|الثانية|الثالثة|\d+)\b)",
    re.IGNORECASE | re.UNICODE,
)

_CONCEPTUAL_KW = re.compile(
    r"(?:ما هو تعريف|عرّف|ما المقصود|ماذا يُقصد|"
    r"define|definition of|what is meant by|"
    r"ما مفهوم|مفهوم|ما معنى)",
    re.IGNORECASE | re.UNICODE,
)

_LAYMAN_KW = re.compile(
    r"(?:بكلمات بسيطة|بشكل مبسط|للمواطن العادي|"
    r"in simple terms|simply|للعموم|باختصار غير قانوني|"
    r"اشرح ببساطة|بلغة بسيطة)",
    re.IGNORECASE | re.UNICODE,
)

# Long-context: procedural / multi-stage questions
_LONG_CONTEXT_KW = re.compile(
    r"(?:خطوات|إجراءات.*تفصيلية|كيفية.*كاملة|"
    r"step.?by.?step|full procedure|detailed process|"
    r"شرح.*مراحل|مراحل.*تفصيلية|الإجراءات الكاملة)",
    re.IGNORECASE | re.UNICODE,
)

# Heuristic: how many distinct legal entities appear (law numbers, article refs)
_LEGAL_ENTITY_RE = re.compile(
    r"(?:\d{2,4}-\d{1,3}|\bالمادة\s+\d+|art(?:icle)?\s*\d+)",
    re.IGNORECASE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# IntentResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntentResult:
    """Output of the intent classifier."""
    query_type: str           # one of the 8 benchmark query types
    language: str             # "ar" | "fr"
    answerability_hint: str   # "probably_answerable" | "probably_unanswerable"
    normalized_query: str     # Arabic/French normalized form of the query
    confidence: float         # 1.0 for regex matches; 0.8 for heuristic; 0.6 default


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

def classify(query: str) -> IntentResult:
    """Classify a raw query string into an IntentResult."""
    normalized, lang = normalize_query(query)

    # Answerability pre-check via jurisdiction infection signals
    from akn_rlm.gates.jurisdiction import is_infected
    answerability = (
        "probably_unanswerable" if is_infected(query)
        else "probably_answerable"
    )

    query_type, confidence = _classify_type(query)

    return IntentResult(
        query_type=query_type,
        language=lang,
        answerability_hint=answerability,
        normalized_query=normalized,
        confidence=confidence,
    )


def _classify_type(query: str) -> tuple[str, float]:
    """Return (query_type, confidence) from regex rules."""
    # Temporal factual (must precede generic temporal patterns)
    if _TEMPORAL_FACTUAL_KW.search(query):
        return "temporal_factual", 1.0

    # Multi-hop (cross-law interactions)
    if _MULTI_HOP_KW.search(query):
        return "multi_hop", 1.0

    # Exact article lookup
    if _EXACT_ARTICLE_KW.search(query):
        return "exact_article", 1.0

    # Conceptual / definitional
    if _CONCEPTUAL_KW.search(query):
        return "conceptual_definitional", 1.0

    # Layman / plain-language
    if _LAYMAN_KW.search(query):
        return "layman", 1.0

    # Long-context (procedural keyword OR 3+ distinct legal entity references)
    if _LONG_CONTEXT_KW.search(query):
        return "long_context", 1.0
    if len(_LEGAL_ENTITY_RE.findall(query)) >= 3:
        return "long_context", 0.8

    # Default
    return "rule_application", 0.6
