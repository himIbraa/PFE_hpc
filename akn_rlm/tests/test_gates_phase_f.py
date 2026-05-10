"""Phase F gate tests: GateResult, citation_existence, jurisdiction, faithfulness."""
from __future__ import annotations

import pytest

from akn_rlm.gates.base import GateResult
from akn_rlm.gates import citation_existence, faithfulness_nli
from akn_rlm.gates import jurisdiction as jurisdiction_gate


# ---------------------------------------------------------------------------
# Stub registry
# ---------------------------------------------------------------------------

class _StubDocEntry:
    def __init__(self, doc_id, article_numbers):
        self.doc_id = doc_id
        self.doc_title = f"Doc {doc_id}"
        self.doc_date = "1984-06-09"
        from akn_rlm.normalizers import ref_to_eid
        self.article_eids = {ref_to_eid(str(n)) for n in article_numbers}


class _StubRegistry:
    def __init__(self, docs):
        self._docs = docs

    def has_article(self, doc_id, article_ref):
        from akn_rlm.normalizers import ref_to_eid
        entry = self._docs.get(doc_id)
        if entry is None:
            return False
        return ref_to_eid(str(article_ref)) in entry.article_eids

    def get_doc(self, doc_id):
        return self._docs.get(doc_id)

    def resolve_alias(self, alias):
        return None


def _make_registry():
    docs = {
        "84-11_1984-06-09": _StubDocEntry("84-11_1984-06-09", range(1, 200)),
        "75-58_1975-09-26": _StubDocEntry("75-58_1975-09-26", range(1, 700)),
    }
    return _StubRegistry(docs)


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------

class TestGateResult:
    def test_passed_default(self):
        r = GateResult(passed=True)
        assert r.score == 1.0
        assert r.details == []

    def test_failure_summary_empty(self):
        r = GateResult(passed=True)
        assert r.failure_summary() == ""

    def test_failure_summary_with_reason(self):
        r = GateResult(passed=False, score=0.0, details=[
            {"reason": "not_in_corpus"},
            {"reason": "missing_doc_id_or_article_ref"},
        ])
        summary = r.failure_summary()
        assert "not_in_corpus" in summary
        assert "missing_doc_id_or_article_ref" in summary


# ---------------------------------------------------------------------------
# Citation existence gate (Phase F)
# ---------------------------------------------------------------------------

class TestCitationExistenceGate:
    def setup_method(self):
        self.registry = _make_registry()

    def test_valid_citation_passes(self):
        c = {"doc_id": "84-11_1984-06-09", "article_ref": "54"}
        result = citation_existence.check(self.registry, c)
        assert result.passed is True
        assert result.score == 1.0

    def test_nonexistent_doc_fails(self):
        c = {"doc_id": "NONEXISTENT", "article_ref": "1"}
        result = citation_existence.check(self.registry, c)
        assert result.passed is False
        assert any("not_in_corpus" in d.get("reason", "") for d in result.details)

    def test_nonexistent_article_fails(self):
        c = {"doc_id": "84-11_1984-06-09", "article_ref": "9999"}
        result = citation_existence.check(self.registry, c)
        assert result.passed is False

    def test_missing_doc_id_fails(self):
        c = {"article_ref": "54"}
        result = citation_existence.check(self.registry, c)
        assert result.passed is False
        assert any("missing" in d.get("reason", "") for d in result.details)

    def test_collision_doc_hard_fails(self):
        c = {"doc_id": "06-01_2006-02-20", "article_ref": "1"}
        result = citation_existence.check(self.registry, c)
        assert result.passed is False
        assert any("collision" in d.get("reason", "") for d in result.details)

    def test_run_gate_all_valid(self):
        cites = [
            {"doc_id": "84-11_1984-06-09", "article_ref": "54"},
            {"doc_id": "75-58_1975-09-26", "article_ref": "81"},
        ]
        result = citation_existence.run_gate(self.registry, cites)
        assert result.passed is True
        assert result.score == 1.0

    def test_run_gate_mixed(self):
        cites = [
            {"doc_id": "84-11_1984-06-09", "article_ref": "54"},
            {"doc_id": "FAKE", "article_ref": "1"},
        ]
        result = citation_existence.run_gate(self.registry, cites)
        assert result.passed is False
        assert result.score == pytest.approx(0.5)

    def test_run_gate_empty(self):
        result = citation_existence.run_gate(self.registry, [])
        assert result.passed is True

    def test_filter_citations_backwards_compat(self):
        cites = [
            {"doc_id": "84-11_1984-06-09", "article_ref": "54"},
            {"doc_id": "FAKE", "article_ref": "99"},
        ]
        valid, rejected = citation_existence.filter_citations(self.registry, cites)
        assert len(valid) == 1
        assert valid[0]["article_ref"] == "54"
        assert len(rejected) == 1


# ---------------------------------------------------------------------------
# Jurisdiction gate (Phase F)
# ---------------------------------------------------------------------------

class TestJurisdictionGate:
    def test_clean_text_passes(self):
        result = jurisdiction_gate.run_gate(
            "يجوز للزوجة الخلع دون موافقة الزوج", "rule_application"
        )
        assert result.passed is True

    def test_isf_in_answerable_fails(self):
        result = jurisdiction_gate.run_gate(
            "يخضع المواطن لضريبة الثروة ISF", "rule_application"
        )
        assert result.passed is False
        signals = result.details[0].get("signals", [])
        assert any("isf" in s for s in signals)

    def test_isf_in_unanswerable_passes(self):
        # Correct abstention: model mentions ISF to explain why it doesn't exist
        result = jurisdiction_gate.run_gate(
            "ISF لا يوجد في القانون الجزائري", "unanswerable"
        )
        assert result.passed is True
        assert any("correct_abstention" in d.get("note", "") for d in result.details)

    def test_at_will_in_answerable_fails(self):
        result = jurisdiction_gate.run_gate(
            "Employment is at-will under Algerian law", "rule_application"
        )
        assert result.passed is False

    def test_arabic_only_passes(self):
        result = jurisdiction_gate.run_gate(
            "وفقاً للمادة 54 من قانون الأسرة يحق للزوجة المخالعة", "rule_application"
        )
        assert result.passed is True

    def test_detect_function_still_works(self):
        signals = jurisdiction_gate.detect("ISF wealth tax in Algeria")
        assert "fr:isf" in signals

    def test_is_infected_boolean(self):
        assert jurisdiction_gate.is_infected("ISF tax") is True
        assert jurisdiction_gate.is_infected("القانون الجزائري") is False


# ---------------------------------------------------------------------------
# Faithfulness NLI gate (Phase F)
# ---------------------------------------------------------------------------

class TestFaithfulnessGate:
    def test_no_citations_skipped(self):
        result = faithfulness_nli.run_gate("Any answer", [])
        assert result.passed is True
        assert any("no_citations" in d.get("note", "") for d in result.details)

    def test_no_article_text_skipped(self):
        cites = [{"doc_id": "d1", "article_ref": "1"}]  # no text field
        result = faithfulness_nli.run_gate("Some answer", cites)
        assert result.passed is True

    def test_split_claims_arabic(self):
        text = "يجوز للزوجة الخلع. يحكم القاضي بالمقابل المالي."
        claims = faithfulness_nli._split_claims(text)
        assert len(claims) >= 1

    def test_split_claims_minimum_length_filter(self):
        text = "OK. Short. " + "x" * 20 + "."
        claims = faithfulness_nli._split_claims(text)
        assert all(len(c) >= faithfulness_nli._MIN_CLAIM_LEN for c in claims)

    def test_entailment_score_returns_float(self):
        score = faithfulness_nli.entailment_score(
            "يجوز للزوجة دون موافقة الزوج أن تخالع نفسها",
            "يحق للزوجة الخلع",
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_run_gate_high_entailment_passes(self):
        # Same text as premise and hypothesis → should be near-perfect entailment
        text = "يجوز للزوجة دون موافقة الزوج أن تخالع نفسها بمقابل مالي"
        cites = [{"text": text}]
        result = faithfulness_nli.run_gate(text, cites)
        assert result.passed is True

    def test_gate_result_is_gate_result_type(self):
        cites = [{"text": "some legal text"}]
        result = faithfulness_nli.run_gate("answer text", cites)
        assert isinstance(result, GateResult)

    def test_check_faithfulness_backwards_compat(self):
        passed, score = faithfulness_nli.check_faithfulness(
            "يجوز الخلع", [{"supporting_span": "يجوز للزوجة الخلع"}], threshold=0.3
        )
        assert isinstance(passed, bool)
        assert isinstance(score, float)
