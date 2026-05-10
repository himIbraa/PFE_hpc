"""Reciprocal Rank Fusion (RRF) for hybrid retrieval."""
from __future__ import annotations

from typing import Sequence

from akn_rlm.config import RRF_K


def rrf_fuse(
    result_lists: Sequence[list[dict]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[dict]:
    """Fuse multiple ranked lists via Reciprocal Rank Fusion.

    Each dict in a result list must have "doc_id" and "article_ref" keys.
    Returns a merged list sorted by descending RRF score, with all original
    fields from the highest-scoring source hit carried over.

    Args:
        result_lists: One list per retriever, each sorted best-first.
        k:            RRF constant (default 60).
        weights:      Per-list weight multipliers (default: all 1.0).
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must have the same length as result_lists")

    scores: dict[tuple[str, str], float] = {}
    best_hit: dict[tuple[str, str], dict] = {}

    for rl, w in zip(result_lists, weights):
        for rank, hit in enumerate(rl, start=1):
            key = (hit.get("doc_id", ""), hit.get("article_ref", ""))
            rrf_score = w / (k + rank)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in best_hit:
                best_hit[key] = dict(hit)

    merged = []
    for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(best_hit[key])
        entry["rrf_score"] = score
        entry["score"]     = score   # unified field name
        merged.append(entry)
    return merged


def weighted_fuse(
    result_lists: Sequence[list[dict]],
    weights: Sequence[float],
    top_k: int = 20,
) -> list[dict]:
    """Weighted score fusion (requires scores to be comparable / normalized)."""
    if len(weights) != len(result_lists):
        raise ValueError("weights must match result_lists length")

    scores: dict[tuple[str, str], float] = {}
    best_hit: dict[tuple[str, str], dict] = {}

    for rl, w in zip(result_lists, weights):
        for hit in rl:
            key = (hit.get("doc_id", ""), hit.get("article_ref", ""))
            scores[key] = scores.get(key, 0.0) + w * float(hit.get("score", 0.0))
            if key not in best_hit:
                best_hit[key] = dict(hit)

    merged = []
    for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        entry = dict(best_hit[key])
        entry["score"] = score
        merged.append(entry)
    return merged
