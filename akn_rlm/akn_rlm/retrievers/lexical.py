"""Lexical (BM25) retriever — thin wrapper over BM25Index."""
from __future__ import annotations

import logging
from pathlib import Path

from akn_rlm.config import BM25_INDEX_PATH
from akn_rlm.indexers.bm25 import BM25Index

log = logging.getLogger(__name__)

_INDEX: BM25Index | None = None


def _get_index(path: Path = BM25_INDEX_PATH) -> BM25Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = BM25Index.load(path)
    return _INDEX


def search(query: str, k: int = 20, path: Path = BM25_INDEX_PATH) -> list[dict]:
    """Return up to k BM25 hits as plain dicts."""
    idx = _get_index(path)
    hits = idx.search(query, k=k)
    return [
        {
            "chunk_id":   h.chunk_id,
            "doc_id":     h.doc_id,
            "article_ref":h.article_ref,
            "score":      h.score,
            "text":       h.text,
            "retriever":  "bm25",
        }
        for h in hits
    ]


def reload(path: Path = BM25_INDEX_PATH) -> None:
    global _INDEX
    _INDEX = BM25Index.load(path)
