"""Tests for the IntentResult classifier module."""
import pytest
from akn_rlm.rlm.classifier import classify, IntentResult


def _qt(query: str) -> str:
    return classify(query).query_type


# ---------------------------------------------------------------------------
# Query type routing — all 8 benchmark types
# ---------------------------------------------------------------------------

def test_temporal_factual():
    assert _qt("منذ متى يسري قانون الأسرة؟") == "temporal_factual"
    assert _qt("ما تاريخ صدور القانون المدني الجزائري؟") == "temporal_factual"
    assert _qt("متى صدر قانون الاستثمار؟") == "temporal_factual"


def test_multi_hop():
    assert _qt("ما العلاقة بين قانون الأسرة والقانون المدني؟") == "multi_hop"
    assert _qt("كيف يتفاعل القانون التجاري والقانون المدني معاً؟") == "multi_hop"


def test_exact_article():
    assert _qt("ما نص المادة 40 من قانون الأسرة؟") == "exact_article"
    assert _qt("اذكر المادة الأولى من القانون المدني") == "exact_article"


def test_conceptual_definitional():
    assert _qt("ما هو تعريف الوصية في القانون الجزائري؟") == "conceptual_definitional"
    assert _qt("ما المقصود بالمسؤولية التقصيرية؟") == "conceptual_definitional"


def test_layman():
    assert _qt("اشرح لي بكلمات بسيطة قانون الإرث الجزائري") == "layman"


def test_long_context_keyword():
    assert _qt("اشرح الإجراءات الكاملة للطلاق في الجزائر خطوة بخطوة") == "long_context"


def test_rule_application_default():
    # Generic question with no special markers → rule_application
    result = classify("هل يحق للمستأجر طلب تخفيض الإيجار؟")
    assert result.query_type == "rule_application"


# ---------------------------------------------------------------------------
# IntentResult schema
# ---------------------------------------------------------------------------

def test_intent_result_is_frozen():
    r = classify("ما هو القانون المدني؟")
    with pytest.raises((AttributeError, TypeError)):
        r.query_type = "something"  # type: ignore[misc]


def test_intent_result_fields_populated():
    r = classify("ما هو تعريف الوصية في القانون الجزائري؟")
    assert isinstance(r, IntentResult)
    assert r.query_type == "conceptual_definitional"
    assert r.language in ("ar", "fr")
    assert r.answerability_hint in ("probably_answerable", "probably_unanswerable")
    assert isinstance(r.normalized_query, str) and r.normalized_query
    assert 0.0 <= r.confidence <= 1.0


def test_answerability_hint_probably_answerable():
    r = classify("هل يجوز الزواج بأكثر من واحدة في الجزائر؟")
    assert r.answerability_hint == "probably_answerable"


def test_answerability_hint_probably_unanswerable_foreign():
    # ISF is a French tax concept — should trigger "probably_unanswerable"
    r = classify("What is the ISF tax rate in France?")
    assert r.answerability_hint == "probably_unanswerable"


def test_language_arabic():
    r = classify("ما هو القانون المدني الجزائري؟")
    assert r.language == "ar"


def test_language_french():
    r = classify("Quelle est la procédure d'adoption en Algérie?")
    assert r.language == "fr"


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------

def test_regex_match_high_confidence():
    r = classify("ما نص المادة 3 من قانون العقوبات؟")
    assert r.confidence == 1.0  # exact_article regex match


def test_default_lower_confidence():
    r = classify("هل يحق للموظف أخذ إجازة سنوية؟")
    assert r.confidence <= 0.8
