"""Unit tests for eval/metrics.py, eval/stratified.py, eval/report.py."""
from __future__ import annotations

import pytest
from akn_rlm.eval.metrics import (
    mrr_at_k, recall_at_k, precision_at_k, ndcg_at_k,
    citation_f1, abstention_accuracy, aggregate,
)
from akn_rlm.eval.stratified import stratify
from akn_rlm.eval.report import format_report


def _make_result(pred_docs=None, gold_docs=None,
                 pred_arts=None, gold_arts=None,
                 pred_cites=None, gold_cites=None,
                 pred_abstain=False, gold_abstain=False,
                 query_type="rule_application", legal_category="family_law"):
    return {
        "pred_doc_ids":       pred_docs or [],
        "pred_article_ids":   pred_arts or [],
        "gold_doc_ids":       set(gold_docs or []),
        "gold_article_ids":   set(gold_arts or []),
        "predicted_citations": pred_cites or [],
        "gold_citations":     gold_cites or [],
        "predicted_abstain":  pred_abstain,
        "gold_abstain":       gold_abstain,
        "query_type":         query_type,
        "legal_category":     legal_category,
    }


class TestMRR:
    def test_hit_at_rank_1(self):
        assert mrr_at_k(["a", "b"], {"a"}) == 1.0

    def test_hit_at_rank_2(self):
        assert mrr_at_k(["x", "a"], {"a"}) == pytest.approx(0.5)

    def test_no_hit(self):
        assert mrr_at_k(["x", "y"], {"a"}) == 0.0

    def test_k_truncation(self):
        assert mrr_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


class TestRecall:
    def test_full_recall(self):
        assert recall_at_k(["a", "b"], {"a", "b"}) == pytest.approx(1.0)

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], {"a", "b"}) == pytest.approx(0.5)

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set()) == 0.0


class TestPrecision:
    def test_full_precision(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_half_precision(self):
        assert precision_at_k(["a", "x"], {"a"}, k=2) == pytest.approx(0.5)

    def test_k_zero(self):
        assert precision_at_k([], set(), k=0) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}) == pytest.approx(1.0)

    def test_no_hit(self):
        assert ndcg_at_k(["x", "y"], {"a", "b"}) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a"], set()) == 0.0


class TestCitationF1:
    def test_perfect_match(self):
        c = [{"doc_id": "d1", "article_ref": "1"}]
        assert citation_f1(c, c) == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = [{"doc_id": "d1", "article_ref": "1"}]
        gold = [{"doc_id": "d2", "article_ref": "2"}]
        assert citation_f1(pred, gold) == 0.0

    def test_both_empty(self):
        assert citation_f1([], []) == pytest.approx(1.0)

    def test_partial(self):
        pred = [{"doc_id": "d1", "article_ref": "1"},
                {"doc_id": "d2", "article_ref": "2"}]
        gold = [{"doc_id": "d1", "article_ref": "1"},
                {"doc_id": "d3", "article_ref": "3"}]
        f1 = citation_f1(pred, gold)
        assert 0.0 < f1 < 1.0


class TestAbstentionAccuracy:
    def test_both_abstain(self):
        assert abstention_accuracy(True, True) == 1.0

    def test_both_answer(self):
        assert abstention_accuracy(False, False) == 1.0

    def test_mismatch(self):
        assert abstention_accuracy(True, False) == 0.0


class TestAggregate:
    def test_aggregate_single_result(self):
        r = _make_result(
            pred_docs=["d1"], gold_docs=["d1"],
            pred_arts=["d1#art_1"], gold_arts=["d1#art_1"],
            pred_cites=[{"doc_id": "d1", "article_ref": "1"}],
            gold_cites=[{"doc_id": "d1", "article_ref": "1"}],
        )
        agg = aggregate([r])
        assert agg["mrr_doc"] == pytest.approx(1.0)
        assert agg["mrr_article"] == pytest.approx(1.0)
        assert agg["citation_f1"] == pytest.approx(1.0)
        assert agg["abstention_acc"] == pytest.approx(1.0)

    def test_aggregate_empty(self):
        assert aggregate([]) == {}

    def test_aggregate_macro_average(self):
        r1 = _make_result(pred_docs=["a"], gold_docs=["a"])   # MRR=1
        r2 = _make_result(pred_docs=["x"], gold_docs=["a"])   # MRR=0
        agg = aggregate([r1, r2])
        assert agg["mrr_doc"] == pytest.approx(0.5)


class TestStratify:
    def test_overall_present(self):
        r = _make_result()
        s = stratify([r])
        assert "overall" in s
        assert "by_query_type" in s
        assert "by_category" in s

    def test_by_query_type_keys(self):
        r1 = _make_result(query_type="rule_application")
        r2 = _make_result(query_type="temporal")
        s = stratify([r1, r2])
        assert "rule_application" in s["by_query_type"]
        assert "temporal" in s["by_query_type"]

    def test_counts(self):
        r1 = _make_result(legal_category="family_law")
        r2 = _make_result(legal_category="contract_law")
        s = stratify([r1, r2])
        assert s["counts"]["total"] == 2
        assert s["counts"]["by_category"]["family_law"] == 1


class TestReport:
    def test_format_report_runs(self):
        r = _make_result(pred_docs=["d1"], gold_docs=["d1"])
        s = stratify([r])
        text = format_report(s)
        assert "MRR" in text
        assert "OVERALL" in text

    def test_delta_shown_for_mrr_doc(self):
        r = _make_result(pred_docs=["d1"], gold_docs=["d1"])
        s = stratify([r])
        text = format_report(s)
        assert "baseline" in text
