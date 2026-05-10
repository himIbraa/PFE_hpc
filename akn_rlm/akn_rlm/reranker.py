"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3."""
from __future__ import annotations

import logging
from typing import Sequence

from akn_rlm.config import RERANKER_MODEL, RERANK_TOP_K

log = logging.getLogger(__name__)

_MODEL = None
_MODEL_FAILED = False


def _get_model(model_name: str = RERANKER_MODEL):
    global _MODEL, _MODEL_FAILED
    if _MODEL_FAILED:
        return None
    if _MODEL is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            log.info("Loading reranker: %s", model_name)
            _MODEL = CrossEncoder(model_name, max_length=512)
        except Exception as exc:
            _MODEL_FAILED = True
            log.warning("Could not load reranker %s: %s", model_name, exc)
            return None
    return _MODEL


def rerank(
    query: str,
    candidates: list[dict],
    k: int = RERANK_TOP_K,
    model_name: str = RERANKER_MODEL,
) -> list[dict]:
    """Rerank candidates using cross-encoder scores.

    Each candidate dict must have a "text" key.
    Returns top-k candidates with added "rerank_score" field, sorted descending.
    """
    if not candidates:
        return []

    model = _get_model(model_name)
    if model is None:
        return candidates[:k]
    pairs = [[query, c.get("text", "")] for c in candidates]
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:
        log.error("Reranker failed: %s — returning original order", exc)
        return candidates[:k]

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: float(x[0]),
        reverse=True,
    )
    result = []
    for score, candidate in scored[:k]:
        entry = dict(candidate)
        entry["rerank_score"] = float(score)
        result.append(entry)
    return result
