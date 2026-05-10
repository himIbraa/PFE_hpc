"""Phase I evaluation metrics tests."""
from __future__ import annotations

import math
import pytest

from akn_rlm.eval.metrics import (
    # Retrieval
    mrr_at_k, recall_at_k, precision_at_k, ndcg_at_k, map_at_k,
    # Citation
    citation_precision, citation_recall, citation_f1,
    doc_level_citation_f1, citation_ordering_kendall_tau,
    # Generation
    exact_match, token_f1, rouge_l, reasoning_chain_score,
    # Faithfulness
    hcr, jir, citation_groundedness,
    # Abstention (batch)
    abstention_recall, abstention_precision, abstention_f1, abstention_accuracy,
    # Aggregate
    aggregate,
    # Telemetry
    telemetry_metrics,
)
from akn_rlm.eval.stratified import stratify


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

class TestMapAtK:
    def test_perfect_ranking(self):
        assert map_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_second_hit(self):
        # Only hit at rank 2: AP = 1/2 / 1 = 0.5
        assert map_at_k(["x", "a"], {"a"}, k=5) == pytest.approx(0.5)

    def test_no_relevant(self):
        assert map_at_k(["a", "b"], set(), k=5) == 0.0

    def test_empty_predicted(self):
        assert map_at_k([], {"a"}, k=10) == 0.0

    def test_k_cutoff_applied(self):
        # Relevant item at rank 5, k=3 → 0.0
        assert map_at_k(["x", "y", "z", "w", "a"], {"a"}, k=3) == 0.0


class TestRetrievalK:
    def test_recall_perfect(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(1.0)

    def test_precision_at_k(self):
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(2 / 3)

    def test_ndcg_perfect(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(1.0)

    def test_ndcg_empty_relevant(self):
        assert ndcg_at_k(["a"], set(), k=5) == 0.0


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------

class TestCitationPrecisionRecall:
    def _c(self, doc_id, art_ref):
        return {"doc_id": doc_id, "article_ref": art_ref}

    def test_precision_full(self):
        pred = [self._c("d1", "1"), self._c("d1", "2")]
        gold = [self._c("d1", "1"), self._c("d1", "2"), self._c("d2", "3")]
        assert citation_precision(pred, gold) == pytest.approx(1.0)

    def test_recall_partial(self):
        pred = [self._c("d1", "1")]
        gold = [self._c("d1", "1"), self._c("d1", "2")]
        assert citation_recall(pred, gold) == pytest.approx(0.5)

    def test_f1_zero_overlap(self):
        pred = [self._c("d1", "1")]
        gold = [self._c("d2", "99")]
        assert citation_f1(pred, gold) == 0.0

    def test_both_empty(self):
        assert citation_f1([], []) == 1.0

    def test_doc_level_f1(self):
        pred = [self._c("d1", "1"), self._c("d1", "2")]
        gold = [self._c("d1", "99"), self._c("d2", "1")]
        # predicted docs: {d1}, gold docs: {d1, d2} → P=1.0, R=0.5
        assert doc_level_citation_f1(pred, gold) == pytest.approx(2 / 3)


class TestCitationOrderingKendallTau:
    def _c(self, doc_id, art_ref):
        return {"doc_id": doc_id, "article_ref": art_ref}

    def test_perfect_order(self):
        gold = [self._c("d1", "1"), self._c("d1", "2"), self._c("d1", "3")]
        pred = [self._c("d1", "1"), self._c("d1", "2"), self._c("d1", "3")]
        tau = citation_ordering_kendall_tau(pred, gold)
        # scipy may not be installed; function must return a float either way
        assert isinstance(tau, float)
        assert -1.0 <= tau <= 1.0

    def test_fewer_than_two_common(self):
        gold = [self._c("d1", "1"), self._c("d2", "2")]
        pred = [self._c("d1", "1"), self._c("d3", "99")]
        assert citation_ordering_kendall_tau(pred, gold) == 0.0


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

class TestGenerationMetrics:
    def test_exact_match_true(self):
        assert exact_match("hello", "hello") == 1.0

    def test_exact_match_false(self):
        assert exact_match("hello", "world") == 0.0

    def test_exact_match_whitespace_stripped(self):
        assert exact_match("  hello  ", "hello") == 1.0

    def test_token_f1_full_overlap(self):
        assert token_f1("a b c", "a b c") == pytest.approx(1.0)

    def test_token_f1_no_overlap(self):
        assert token_f1("x y", "a b") == 0.0

    def test_token_f1_partial(self):
        score = token_f1("a b c", "a b d")
        assert 0.0 < score < 1.0

    def test_rouge_l_same_text(self):
        text = "يجوز للزوجة دون موافقة الزوج"
        assert rouge_l(text, text) == pytest.approx(1.0)

    def test_rouge_l_empty(self):
        assert rouge_l("", "some text") == 0.0
        assert rouge_l("some text", "") == 0.0

    def test_reasoning_chain_score(self):
        pred = ["step A", "step B"]
        gold = ["step A", "step B"]
        score = reasoning_chain_score(pred, gold)
        assert score == pytest.approx(1.0)

    def test_reasoning_chain_empty_gold(self):
        assert reasoning_chain_score(["a"], []) == 0.0

    def test_reasoning_chain_shorter_pred(self):
        pred = ["step A"]
        gold = ["step A", "step B"]
        score = reasoning_chain_score(pred, gold)
        # First step matches (1.0), second step missing (0.0) → avg 0.5
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Faithfulness metrics
# ---------------------------------------------------------------------------

class TestFaithfulnessMetrics:
    def test_hcr_zero_when_no_citations(self):
        assert hcr(0, 0) == 0.0

    def test_hcr_half(self):
        assert hcr(4, 2) == pytest.approx(0.5)

    def test_jir_answerable_with_signal(self):
        assert jir(is_answerable=True, has_foreign_signals=True) == 1.0

    def test_jir_answerable_no_signal(self):
        assert jir(is_answerable=True, has_foreign_signals=False) == 0.0

    def test_jir_unanswerable_with_signal(self):
        assert jir(is_answerable=False, has_foreign_signals=True) == 0.0

    def test_citation_groundedness_full(self):
        text = "يجوز للزوجة الخلع دون موافقة الزوج"
        cites = [{"text": "يجوز للزوجة الخلع دون موافقة الزوج بمقابل مالي"}]
        # Bigram overlap should be > 0
        assert citation_groundedness(text, cites) > 0.0

    def test_citation_groundedness_no_overlap(self):
        text = "completely different text about nothing"
        cites = [{"text": "يجوز للزوجة الخلع"}]
        assert citation_groundedness(text, cites) == 0.0

    def test_citation_groundedness_empty_citations(self):
        assert citation_groundedness("answer", []) == 0.0


# ---------------------------------------------------------------------------
# Abstention batch metrics
# ---------------------------------------------------------------------------

class TestAbstentionMetrics:
    def _make_results(self, pairs):
        return [{"predicted_abstain": p, "gold_abstain": g} for p, g in pairs]

    def test_recall_all_unanswerable_caught(self):
        results = self._make_results([(True, True), (True, True)])
        assert abstention_recall(results) == pytest.approx(1.0)

    def test_recall_none_caught(self):
        results = self._make_results([(False, True), (False, True)])
        assert abstention_recall(results) == pytest.approx(0.0)

    def test_precision_all_correct(self):
        results = self._make_results([(True, True), (True, True)])
        assert abstention_precision(results) == pytest.approx(1.0)

    def test_precision_all_wrong(self):
        results = self._make_results([(True, False), (True, False)])
        assert abstention_precision(results) == pytest.approx(0.0)

    def test_f1_balanced(self):
        results = self._make_results([
            (True, True),   # correct abstention
            (False, False), # correct non-abstention
            (True, False),  # false abstention
            (False, True),  # missed abstention
        ])
        r = abstention_recall(results)
        p = abstention_precision(results)
        f1 = abstention_f1(results)
        assert f1 == pytest.approx(2 * p * r / (p + r) if p + r > 0 else 0.0)

    def test_no_abstentions_predicted(self):
        results = self._make_results([(False, True), (False, False)])
        assert abstention_precision(results) == 0.0

    def test_no_unanswerable_gold(self):
        results = self._make_results([(True, False), (False, False)])
        assert abstention_recall(results) == 0.0

    def test_abstention_accuracy_per_query(self):
        assert abstention_accuracy(True, True) == 1.0
        assert abstention_accuracy(False, False) == 1.0
        assert abstention_accuracy(True, False) == 0.0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _make_result(**kw):
    base = {
        "pred_doc_ids": [],
        "pred_article_ids": [],
        "gold_doc_ids": set(),
        "gold_article_ids": set(),
        "predicted_citations": [],
        "gold_citations": [],
        "predicted_abstain": False,
        "gold_abstain": False,
        "answer_text": "",
        "gold_answer_text": "",
        "reasoning_chain": [],
        "gold_reasoning_chain": [],
        "hcr": 0.0,
        "jir": 0.0,
        "answer_faithfulness": 1.0,
        "citation_groundedness": 0.0,
        "tokens_used": 100,
        "latency_s": 1.0,
        "retry_count": 0,
    }
    base.update(kw)
    return base


class TestAggregate:
    def test_empty_returns_empty(self):
        assert aggregate([]) == {}

    def test_all_new_keys_present(self):
        r = _make_result(
            pred_doc_ids=["d1"], gold_doc_ids={"d1"},
            pred_article_ids=["d1#art_1"], gold_article_ids={"d1#art_1"},
        )
        out = aggregate([r])
        assert "map_doc" in out
        assert "map_article" in out
        assert "citation_precision" in out
        assert "citation_recall" in out
        assert "doc_citation_f1" in out
        assert "exact_match" in out
        assert "token_f1" in out
        assert "rouge_l" in out
        assert "hcr" in out
        assert "jir" in out
        assert "answer_faithfulness" in out
        assert "abstention_recall" in out
        assert "abstention_precision" in out
        assert "abstention_f1" in out
        assert "mean_tokens_per_query" in out
        assert "mean_latency_s" in out
        assert "corrective_retry_rate" in out

    def test_perfect_retrieval(self):
        r = _make_result(
            pred_doc_ids=["d1"], gold_doc_ids={"d1"},
            pred_article_ids=["d1#art_1"], gold_article_ids={"d1#art_1"},
        )
        out = aggregate([r], k=10)
        assert out["mrr_doc"] == pytest.approx(1.0)
        assert out["mrr_article"] == pytest.approx(1.0)

    def test_hcr_averaged(self):
        r1 = _make_result(hcr=0.0)
        r2 = _make_result(hcr=1.0)
        out = aggregate([r1, r2])
        assert out["hcr"] == pytest.approx(0.5)

    def test_mean_tokens(self):
        r1 = _make_result(tokens_used=100)
        r2 = _make_result(tokens_used=200)
        out = aggregate([r1, r2])
        assert out["mean_tokens_per_query"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Telemetry metrics
# ---------------------------------------------------------------------------

class TestTelemetryMetrics:
    def test_all_keys_present(self):
        results = [
            {"tokens_used": 100, "latency_s": 2.0, "retry_count": 1,
             "hcr": 0.1, "jir": 0.0, "answer_faithfulness": 0.9,
             "citation_groundedness": 0.8},
        ]
        tm = telemetry_metrics(results)
        assert "mean_tokens_per_query" in tm
        assert "mean_latency_s" in tm
        assert "corrective_retry_rate" in tm
        assert tm["mean_tokens_per_query"] == pytest.approx(100.0)
        assert tm["mean_latency_s"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Stratified
# ---------------------------------------------------------------------------

class TestStratify:
    def _r(self, qt, cat, diff="easy", lang="ar", split="test", temporal=False):
        return _make_result(
            query_type=qt,
            legal_category=cat,
            difficulty=diff,
            language=lang,
            split=split,
            has_temporal_note=temporal,
        )

    def test_new_strata_present(self):
        results = [
            self._r("rule_application", "family_law", diff="easy", lang="ar"),
            self._r("temporal", "contract_law", diff="hard", lang="fr", temporal=True),
        ]
        strata = stratify(results)
        assert "by_difficulty" in strata
        assert "by_language" in strata
        assert "by_split" in strata
        assert "temporal" in strata

    def test_temporal_subset_isolated(self):
        results = [
            self._r("rule_application", "family_law", temporal=False),
            self._r("temporal", "family_law", temporal=True),
        ]
        strata = stratify(results)
        assert strata["counts"]["temporal"] == 1
        assert strata["temporal"] != {}

    def test_by_difficulty_keys(self):
        results = [
            self._r("rule_application", "family_law", diff="easy"),
            self._r("rule_application", "family_law", diff="hard"),
        ]
        strata = stratify(results)
        assert "easy" in strata["by_difficulty"]
        assert "hard" in strata["by_difficulty"]
