"""Produce a human-readable evaluation report from stratified metrics — Phase I.

Usage:
    from akn_rlm.eval.report import print_report, save_report
    strata = stratify(results)
    print_report(strata)
    save_report(strata, "eval_results/run_001")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Key metrics shown in the main panel (with baseline Δ where applicable)
_MAIN_METRICS = {
    "mrr_doc":             "MRR@K  (doc)",
    "mrr_article":         "MRR@K  (art)",
    "map_doc":             "MAP@K  (doc)",
    "map_article":         "MAP@K  (art)",
    "recall_doc":          "R@K    (doc)",
    "recall_article":      "R@K    (art)",
    "precision_doc":       "P@K    (doc)",
    "precision_article":   "P@K    (art)",
    "ndcg_doc":            "NDCG@K (doc)",
    "ndcg_article":        "NDCG@K (art)",
    "citation_f1":         "Cite F1",
    "citation_precision":  "Cite P",
    "citation_recall":     "Cite R",
    "doc_citation_f1":     "Doc Cite F1",
    "exact_match":         "Exact Match",
    "token_f1":            "Token F1",
    "rouge_l":             "ROUGE-L",
    "hcr":                 "HCR ↓",
    "jir":                 "JIR ↓",
    "answer_faithfulness": "Faithfulness",
    "citation_groundedness": "Groundedness",
    "abstention_acc":      "Abst Acc",
    "abstention_recall":   "Abst Recall",
    "abstention_precision":"Abst Prec",
    "abstention_f1":       "Abst F1",
    "mean_tokens_per_query": "Mean Tokens",
    "mean_latency_s":      "Mean Latency (s)",
    "corrective_retry_rate": "Retry Rate",
}

_COMPACT_METRICS = [
    "mrr_doc", "mrr_article", "citation_f1", "hcr", "jir",
    "abstention_recall", "abstention_f1",
]

# Published baselines (doc MRR, article MRR) for Δ annotation
_BASELINE = {
    "mrr_doc":     0.65,
    "mrr_article": 0.18,
    "citation_f1": 0.21,
    "hcr":         0.67,
    "jir":         0.10,
}

# Target thresholds
_TARGET = {
    "mrr_doc":     0.75,
    "mrr_article": 0.40,
    "citation_f1": 0.40,
    "hcr":         0.15,
    "jir":         0.10,
    "abstention_f1": 0.75,
}


def _delta_str(key: str, val: float) -> str:
    baseline = _BASELINE.get(key)
    target = _TARGET.get(key)
    parts: list[str] = []
    if baseline is not None and val == val:  # not nan
        diff = val - baseline
        parts.append(f"Δ {diff:+.3f} vs baseline")
    if target is not None and val == val:
        met = "✓" if val >= target else "✗"
        parts.append(f"{met} target={target:.2f}")
    return f"  ({', '.join(parts)})" if parts else ""


def _fmt_block(metrics: dict[str, float], metric_keys=None) -> str:
    keys = metric_keys or list(_MAIN_METRICS)
    lines = []
    for key in keys:
        label = _MAIN_METRICS.get(key, key)
        val = metrics.get(key, float("nan"))
        delta = _delta_str(key, val)
        if val == val:  # not nan
            lines.append(f"  {label:<24s} {val:.4f}{delta}")
    return "\n".join(lines) if lines else "  (no data)"


def _fmt_compact(metrics: dict[str, float]) -> str:
    parts = []
    for key in _COMPACT_METRICS:
        val = metrics.get(key, float("nan"))
        if val == val:
            label = _MAIN_METRICS.get(key, key)
            parts.append(f"{label}={val:.3f}")
    return "  " + "  ".join(parts)


def format_report(strata: dict[str, Any], title: str = "AKN-RLM Evaluation") -> str:
    counts = strata.get("counts", {})
    lines = [
        "=" * 72,
        f"  {title}",
        "=" * 72,
        f"\nTotal questions : {counts.get('total', '?')}",
        f"Temporal subset : {counts.get('temporal', 0)}",
        "",
        "── OVERALL ──────────────────────────────────────────────────────────",
        _fmt_block(strata.get("overall", {})),
    ]

    # By query type
    by_qt = strata.get("by_query_type", {})
    if by_qt:
        lines.append("\n── BY QUERY TYPE ─────────────────────────────────────────────────────")
        for qt, m in sorted(by_qt.items()):
            n = counts.get("by_query_type", {}).get(qt, "?")
            lines.append(f"\n  [{qt}]  (n={n})")
            lines.append(_fmt_compact(m))

    # By difficulty
    by_diff = strata.get("by_difficulty", {})
    if by_diff:
        lines.append("\n── BY DIFFICULTY ─────────────────────────────────────────────────────")
        for diff, m in sorted(by_diff.items()):
            n = counts.get("by_difficulty", {}).get(diff, "?")
            lines.append(f"\n  [{diff}]  (n={n})")
            lines.append(_fmt_compact(m))

    # By language
    by_lang = strata.get("by_language", {})
    if by_lang:
        lines.append("\n── BY LANGUAGE ───────────────────────────────────────────────────────")
        for lang, m in sorted(by_lang.items()):
            n = counts.get("by_language", {}).get(lang, "?")
            lines.append(f"\n  [{lang}]  (n={n})")
            lines.append(_fmt_compact(m))

    # Temporal subset
    temporal = strata.get("temporal", {})
    if temporal:
        lines.append("\n── TEMPORAL SUBSET ───────────────────────────────────────────────────")
        lines.append(_fmt_compact(temporal))

    # By legal category (compact — one line each)
    by_cat = strata.get("by_category", {})
    if by_cat:
        lines.append("\n── BY LEGAL CATEGORY (compact) ───────────────────────────────────────")
        header = f"  {'Category':<28s}  {'MRR(art)':>8s}  {'R(art)':>7s}  {'CiteF1':>7s}  {'HCR':>6s}"
        lines.append(header)
        lines.append("  " + "-" * 65)
        for cat, m in sorted(by_cat.items()):
            n = counts.get("by_category", {}).get(cat, "?")
            row = (
                f"  {cat:<28s}  "
                f"{m.get('mrr_article', 0.0):>8.4f}  "
                f"{m.get('recall_article', 0.0):>7.4f}  "
                f"{m.get('citation_f1', 0.0):>7.4f}  "
                f"{m.get('hcr', 0.0):>6.4f}  (n={n})"
            )
            lines.append(row)

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


def print_report(strata: dict[str, Any], title: str = "AKN-RLM Evaluation") -> None:
    print(format_report(strata, title))


def save_report(
    strata: dict[str, Any],
    output_path: str | Path,
    title: str = "AKN-RLM Evaluation",
) -> Path:
    """Save JSON (full metrics) + TXT (human report) to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    with out.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(strata, fh, ensure_ascii=False, indent=2, default=_default)

    txt_path = out.with_suffix(".txt")
    txt_path.write_text(format_report(strata, title), encoding="utf-8")
    return out
