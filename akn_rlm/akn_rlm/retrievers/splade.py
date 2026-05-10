"""SPLADE sparse retriever — thin wrapper over SPLADEIndex."""
from __future__ import annotations

import logging
from pathlib import Path

from akn_rlm.config import SPLADE_INDEX_PATH
from akn_rlm.indexers.splade import SPLADEIndex

log = logging.getLogger(__name__)

_INDEX: SPLADEIndex | None = None


def _get_index(path: Path = SPLADE_INDEX_PATH) -> SPLADEIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = SPLADEIndex.load(path)
    return _INDEX


def search(query: str, k: int = 20, path: Path = SPLADE_INDEX_PATH) -> list[dict]:
    idx = _get_index(path)
    hits = idx.search(query, k=k)
    return [
        {
            "chunk_id":   h.chunk_id,
            "doc_id":     h.doc_id,
            "article_ref":h.article_ref,
            "score":      h.score,
            "text":       h.text,
            "retriever":  "splade",
        }
        for h in hits
    ]


def reload() -> None:
    global _INDEX
    _INDEX = None
