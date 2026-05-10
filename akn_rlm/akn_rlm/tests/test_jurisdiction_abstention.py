"""Tests for jurisdiction gate behaviour on abstentions and answers."""
import pytest
from akn_rlm.gates.jurisdiction import run_gate, detect, is_infected


# ---------------------------------------------------------------------------
# Basic canary detection
# ---------------------------------------------------------------------------

def test_french_law_detected():
    signals = detect("Le ISF est un impôt français.")
    assert "fr:isf" in signals


def test_us_law_detected():
    signals = detect("This is a class action lawsuit under US law.")
    assert "us:class_action" in signals or "us:us_law" in signals


def test_algerian_text_clean():
    signals = detect("المادة 40 من قانون الأسرة الجزائري تنص على الطلاق")
    assert signals == []


def test_is_infected_true():
    assert is_infected("The at-will employment doctrine applies here.")


def test_is_infected_false():
    assert not is_infected("قانون الأسرة الجزائري رقم 84-11 ينظم الطلاق")


# ---------------------------------------------------------------------------
# Gate: answers with foreign signals FAIL (contamination)
# ---------------------------------------------------------------------------

def test_answer_with_foreign_signal_fails():
    result = run_gate(
        "Under the ISF tax rules, the rate is 1.5%.",
        query_type="rule_application",
    )
    assert not result.passed
    assert result.score == 0.0


def test_clean_answer_passes():
    result = run_gate(
        "المادة 40 من قانون الأسرة تنص على حق الزوجة في طلب الخلع",
        query_type="rule_application",
    )
    assert result.passed


# ---------------------------------------------------------------------------
# Gate: abstentions with foreign signals PASS (correct abstention)
# ---------------------------------------------------------------------------

def test_abstention_with_foreign_signal_passes():
    # Query about ISF (French tax) → correct to abstain
    result = run_gate(
        "لا يوجد في التشريع الجزائري ما يعادل ضريبة ISF الفرنسية",
        query_type="unanswerable",
    )
    assert result.passed


def test_abstention_without_signal_passes():
    # Clean abstention (out-of-corpus Algerian law) → also passes
    result = run_gate(
        "لا توجد في مجموعة القوانين المتاحة نصوص تتعلق بهذا الموضوع",
        query_type="unanswerable",
    )
    assert result.passed


# ---------------------------------------------------------------------------
# Gate details for debugging
# ---------------------------------------------------------------------------

def test_failed_gate_includes_signals():
    result = run_gate(
        "punitive damages are available in US law",
        query_type="rule_application",
    )
    assert not result.passed
    assert result.details
    signals = [s for d in result.details for s in d.get("signals", [])]
    assert len(signals) > 0
