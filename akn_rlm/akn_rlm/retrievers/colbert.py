"""ColBERT retriever — thin wrapper over ColBERT index."""
from __future__ import annotations

import logging
from pathlib import Path

from akn_rlm.config import COLBERT_INDEX_DIR
from akn_rlm.indexers.colbert import ColBERTIndex_load

log = logging.getLogger(__name__)

_INDEX = None


def _get_index(directory: Path = COLBERT_INDEX_DIR):
    global _INDEX
    if _INDEX is None:
        _INDEX = ColBERTIndex_load(directory)
    return _INDEX


def search(query: str, k: int = 20, directory: Path = COLBERT_INDEX_DIR) -> list[dict]:
    idx = _get_index(directory)
    hits = idx.search(query, k=k)
    return [
        {
            "chunk_id":   h.chunk_id,
            "doc_id":     h.doc_id,
            "article_ref":h.article_ref,
            "score":      h.score,
            "text":       h.text,
            "retriever":  "colbert",
        }
        for h in hits
    ]


def reload() -> None:
    global _INDEX
    _INDEX = None
