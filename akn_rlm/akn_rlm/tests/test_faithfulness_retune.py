"""R8: faithfulness-gate retune.

Locks in three behaviours introduced by R8:
  1. ``SUPPORT_THRESHOLD`` is 0.55 (was 0.80 — too strict for legal Arabic).
  2. The faithfulness gate is **record-only**: a faithfulness failure does
     NOT trigger a corrective retry. Only ``citation_existence`` and
     ``jurisdiction`` failures cause the pipeline to loop back.
  3. Per-citation NLI: each claim is supported iff its highest-scoring
     citation entails it — a strong unrelated citation cannot rescue a
     claim that has no real support.

The NLI model is mocked everywhere — these tests do not load
``sentence-transformers``.
"""
from __future__ import annotations

from unittest.mock import patch

from akn_rlm.gates import faithfulness_nli
from akn_rlm.gates.base import GateResult
from akn_rlm.rlm.pipeline import (
    MAX_RETRIES,
    _RETRYABLE_GATES,
    _route_after_gates,
)


# ---------------------------------------------------------------------------
# Constant defaults
# ---------------------------------------------------------------------------

class TestRetuneConstants:
    def test_support_threshold_lowered_to_055(self):
        assert faithfulness_nli.SUPPORT_THRESHOLD == 0.55

    def test_claim_threshold_unchanged(self):
        # Claim threshold is the per-claim NLI cutoff, separate from the
        # coverage fraction. R8 only lowered the coverage fraction.
        assert faithfulness_nli.CLAIM_THRESHOLD == 0.5

    def test_llm_fallback_min_unchanged(self):
        assert faithfulness_nli.LLM_FALLBACK_MIN == 0.3

    def test_retryable_gates_excludes_faithfulness(self):
        assert "faithfulness" not in _RETRYABLE_GATES
        assert "citation" in _RETRYABLE_GATES
        assert "jurisdiction" in _RETRYABLE_GATES


# ---------------------------------------------------------------------------
# _route_after_gates: faithfulness is record-only
# ---------------------------------------------------------------------------

class TestRouteAfterGatesR8:
    def _state(self, gates, retry=0):
        return {"gate_results": gates, "retry_count": retry}

    def test_only_faithfulness_failed_routes_to_assemble(self):
        # R8 contract: faithfulness alone never causes a retry.
        state = self._state(
            {
                "citation": {"passed": True},
                "jurisdiction": {"passed": True},
                "faithfulness": {"passed": False, "score": 0.4},
            },
            retry=0,
        )
        assert _route_after_gates(state) == "assemble_output"

    def test_faithfulness_fail_with_retries_left_does_not_retry(self):
        # Even with retry budget remaining, faithfulness-only fail goes
        # straight to assemble.
        state = self._state(
            {
                "citation": {"passed": True},
                "jurisdiction": {"passed": True},
                "faithfulness": {"passed": False, "score": 0.0},
            },
            retry=0,
        )
        assert _route_after_gates(state) == "assemble_output"

    def test_citation_fail_still_retries(self):
        state = self._state({"citation": {"passed": False}}, retry=0)
        assert _route_after_gates(state) == "corrective_retry"

    def test_jurisdiction_fail_still_retries(self):
        state = self._state({"jurisdiction": {"passed": False}}, retry=0)
        assert _route_after_gates(state) == "corrective_retry"

    def test_citation_fail_plus_faithfulness_fail_retries(self):
        state = self._state(
            {
                "citation": {"passed": False},
                "faithfulness": {"passed": False},
            },
            retry=0,
        )
        assert _route_after_gates(state) == "corrective_retry"

    def test_retry_exhausted_with_only_faithfulness_failed_assembles(self):
        state = self._state(
            {
                "citation": {"passed": True},
                "jurisdiction": {"passed": True},
                "faithfulness": {"passed": False},
            },
            retry=MAX_RETRIES,
        )
        assert _route_after_gates(state) == "assemble_output"

    def test_retry_exhausted_with_citation_failed_assembles(self):
        state = self._state(
            {"citation": {"passed": False}}, retry=MAX_RETRIES
        )
        assert _route_after_gates(state) == "assemble_output"

    def test_all_pass_assembles(self):
        state = self._state(
            {
                "citation": {"passed": True},
                "jurisdiction": {"passed": True},
                "faithfulness": {"passed": True},
            },
            retry=0,
        )
        assert _route_after_gates(state) == "assemble_output"

    def test_empty_gates_assembles(self):
        # No gate has run yet → nothing retryable failed → assemble.
        assert _route_after_gates(self._state({})) == "assemble_output"


# ---------------------------------------------------------------------------
# Per-citation NLI semantics
# ---------------------------------------------------------------------------

# Canned NLI scoring. Map (premise, hypothesis_substring) → score so we can
# control which citation appears strongest for which claim without loading
# the real model.

def _canned_entailment(score_table):
    """Return a fake ``entailment_score`` that looks up scores by (premise_substr, claim_substr)."""

    def _fn(premise, claim):
        for (p_sub, c_sub), s in score_table.items():
            if p_sub in premise and c_sub in claim:
                return s
        return 0.0

    return _fn


class TestPerCitationNLI:
    """Each claim must be entailed by ITS OWN best-matching citation, not pooled."""

    def _patch_model(self):
        # Pretend NLI loaded successfully so run_gate doesn't short-circuit.
        return patch.object(faithfulness_nli, "_get_model", return_value=object())

    def test_claim_supported_by_one_of_many_citations_passes(self):
        # Claim about "divorce" is strongly entailed by citation A; citation
        # B is unrelated. Per-citation NLI: best score (A) ≥ 0.5 → supported.
        scores = {
            ("article-A-divorce", "claim-divorce"): 0.92,
            ("article-B-marriage", "claim-divorce"): 0.10,
        }
        cites = [
            {"doc_id": "d1", "article_ref": "1", "text": "article-A-divorce body"},
            {"doc_id": "d2", "article_ref": "2", "text": "article-B-marriage body"},
        ]
        answer = "claim-divorce statement that needs at least fifteen chars."
        with self._patch_model(), patch.object(
            faithfulness_nli, "entailment_score",
            side_effect=_canned_entailment(scores),
        ):
            result = faithfulness_nli.run_gate(answer, cites)
        assert isinstance(result, GateResult)
        assert result.passed is True
        assert result.score == 1.0

    def test_claim_with_no_supporting_citation_fails_per_citation(self):
        # No citation entails the claim — best score is below CLAIM_THRESHOLD.
        # Even pooling wouldn't save this; verify per-citation also rejects.
        scores = {
            ("article-A-divorce", "claim-tax"): 0.10,
            ("article-B-marriage", "claim-tax"): 0.15,
        }
        cites = [
            {"doc_id": "d1", "article_ref": "1", "text": "article-A-divorce body"},
            {"doc_id": "d2", "article_ref": "2", "text": "article-B-marriage body"},
        ]
        answer = "claim-tax statement padded to satisfy length minimum."
        with self._patch_model(), patch.object(
            faithfulness_nli, "entailment_score",
            side_effect=_canned_entailment(scores),
        ):
            result = faithfulness_nli.run_gate(answer, cites)
        assert result.passed is False  # 0/1 supported = 0.0 < 0.55
        # Detail records the best-matching citation and its (low) score.
        details = result.details
        assert details and "best_cit" in details[0]
        assert details[0]["best_score"] < faithfulness_nli.CLAIM_THRESHOLD

    def test_low_coverage_below_055_fails(self):
        # 1 of 3 claims supported (33% < 55%) → gate fails.
        scores = {
            ("art-A", "claim-one"):   0.95,   # supported
            ("art-A", "claim-two"):   0.10,
            ("art-A", "claim-three"): 0.10,
            ("art-B", "claim-one"):   0.10,
            ("art-B", "claim-two"):   0.10,
            ("art-B", "claim-three"): 0.10,
        }
        cites = [
            {"doc_id": "d1", "article_ref": "1", "text": "art-A body"},
            {"doc_id": "d2", "article_ref": "2", "text": "art-B body"},
        ]
        # Three sentences, one period each, all >= 15 chars.
        answer = (
            "claim-one statement padded long. "
            "claim-two statement padded long. "
            "claim-three statement padded long."
        )
        with self._patch_model(), patch.object(
            faithfulness_nli, "entailment_score",
            side_effect=_canned_entailment(scores),
        ):
            result = faithfulness_nli.run_gate(answer, cites)
        assert result.passed is False
        # Roughly 1/3 supported.
        assert 0.30 <= result.score <= 0.40

    def test_coverage_at_055_passes(self):
        # 2 of 3 claims supported (66% ≥ 55%) — passes under the new
        # threshold but would have FAILED at the old 0.80.
        scores = {
            ("art-A", "claim-one"):   0.95,   # supported
            ("art-A", "claim-two"):   0.95,   # supported
            ("art-A", "claim-three"): 0.10,
            ("art-B", "claim-one"):   0.10,
            ("art-B", "claim-two"):   0.10,
            ("art-B", "claim-three"): 0.10,
        }
        cites = [
            {"doc_id": "d1", "article_ref": "1", "text": "art-A body"},
            {"doc_id": "d2", "article_ref": "2", "text": "art-B body"},
        ]
        answer = (
            "claim-one statement padded long. "
            "claim-two statement padded long. "
            "claim-three statement padded long."
        )
        with self._patch_model(), patch.object(
            faithfulness_nli, "entailment_score",
            side_effect=_canned_entailment(scores),
        ):
            result = faithfulness_nli.run_gate(answer, cites)
        assert result.passed is True
        assert result.score >= faithfulness_nli.SUPPORT_THRESHOLD
        # Sanity: under the OLD 0.80 threshold this same score would fail.
        assert result.score < 0.80

    def test_old_080_threshold_still_supported_via_kwarg(self):
        # Allow callers to override the default if a stricter audit is wanted.
        scores = {
            ("art-A", "claim-one"): 0.95,
            ("art-A", "claim-two"): 0.95,
            ("art-A", "claim-three"): 0.10,
        }
        cites = [{"doc_id": "d1", "article_ref": "1", "text": "art-A body"}]
        answer = (
            "claim-one statement padded long. "
            "claim-two statement padded long. "
            "claim-three statement padded long."
        )
        with self._patch_model(), patch.object(
            faithfulness_nli, "entailment_score",
            side_effect=_canned_entailment(scores),
        ):
            result = faithfulness_nli.run_gate(answer, cites, support_threshold=0.80)
        assert result.passed is False  # 2/3 = 0.67 < 0.80


# ---------------------------------------------------------------------------
# Best-match attribution: claim → its OWN best-scoring citation
# ---------------------------------------------------------------------------

class TestBestMatchAttribution:
    def _patch_model(self):
        return patch.object(faithfulness_nli, "_get_model", return_value=object())

    def test_unsupported_claim_records_best_cit_key(self):
        # Two citations, both below threshold. The unsupported detail must
        # name the citation that scored highest, not just the first one.
        scores = {
            ("art-A", "claim"): 0.20,
            ("art-B", "claim"): 0.45,   # closer but still < CLAIM_THRESHOLD=0.5
        }
        cites = [
            {"doc_id": "d1", "article_ref": "1", "text": "art-A body"},
            {"doc_id": "d2", "article_ref": "2", "text": "art-B body"},
        ]
        answer = "claim statement that is at least fifteen characters long."
        with self._patch_model(), patch.object(
            faithfulness_nli, "entailment_score",
            side_effect=_canned_entailment(scores),
        ):
            result = faithfulness_nli.run_gate(answer, cites)
        assert result.passed is False
        assert result.details[0]["best_cit"] == "d2#2"
        assert result.details[0]["best_score"] == 0.45
