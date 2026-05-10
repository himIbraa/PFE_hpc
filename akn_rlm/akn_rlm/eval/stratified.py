"""Stratified aggregation of eval results — Phase I.

Strata computed:
  overall, by_query_type, by_category, by_difficulty, by_language, by_split,
  temporal (questions with a temporal_note field)

Usage:
    from akn_rlm.eval.stratified import stratify
    strata = stratify(per_query_results)
    strata["by_query_type"]["rule_application"]  →  metrics dict
    strata["by_difficulty"]["hard"]              →  metrics dict
    strata["temporal"]                           →  metrics dict (temporal subset)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from akn_rlm.eval.metrics import aggregate


def stratify(
    results: list[dict[str, Any]],
    k: int = 10,
) -> dict[str, Any]:
    """Group results and compute per-stratum aggregated metrics.

    Each result dict must have the fields expected by metrics.aggregate()
    plus any of:
      query_type     : str
      legal_category : str
      difficulty     : str  (easy | medium | hard)
      language       : str  (ar | fr)
      split          : str  (test | dev)
      has_temporal_note : bool
    """
    by_query_type:  dict[str, list] = defaultdict(list)
    by_category:    dict[str, list] = defaultdict(list)
    by_difficulty:  dict[str, list] = defaultdict(list)
    by_language:    dict[str, list] = defaultdict(list)
    by_split:       dict[str, list] = defaultdict(list)
    temporal: list = []

    for r in results:
        by_query_type[r.get("query_type", "unknown")].append(r)
        by_category[r.get("legal_category", "unknown")].append(r)
        by_difficulty[r.get("difficulty", "unknown")].append(r)
        by_language[r.get("language", "ar")].append(r)
        by_split[r.get("split", "test")].append(r)
        if r.get("has_temporal_note"):
            temporal.append(r)

    strata: dict[str, Any] = {
        "overall": aggregate(results, k=k),
        "by_query_type": {
            qt: aggregate(s, k=k) for qt, s in by_query_type.items()
        },
        "by_category": {
            cat: aggregate(s, k=k) for cat, s in by_category.items()
        },
        "by_difficulty": {
            diff: aggregate(s, k=k) for diff, s in by_difficulty.items()
        },
        "by_language": {
            lang: aggregate(s, k=k) for lang, s in by_language.items()
        },
        "by_split": {
            spl: aggregate(s, k=k) for spl, s in by_split.items()
        },
        "temporal": aggregate(temporal, k=k) if temporal else {},
        "counts": {
            "total": len(results),
            "temporal": len(temporal),
            "by_query_type":  {qt: len(s) for qt, s in by_query_type.items()},
            "by_category":    {cat: len(s) for cat, s in by_category.items()},
            "by_difficulty":  {diff: len(s) for diff, s in by_difficulty.items()},
            "by_language":    {lang: len(s) for lang, s in by_language.items()},
            "by_split":       {spl: len(s) for spl, s in by_split.items()},
        },
    }
    return strata
