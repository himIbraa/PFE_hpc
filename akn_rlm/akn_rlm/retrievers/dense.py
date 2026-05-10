"""Dense retriever — thin wrapper over DenseIndex."""
from __future__ import annotations

import logging
from pathlib import Path

from akn_rlm.config import DENSE_FAISS_PATH, DENSE_META_PATH
from akn_rlm.indexers.dense import DenseIndex

log = logging.getLogger(__name__)

_INDEX: DenseIndex | None = None


def _get_index(
    faiss_path: Path = DENSE_FAISS_PATH,
    meta_path: Path = DENSE_META_PATH,
) -> DenseIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = DenseIndex.load(faiss_path, meta_path)
    return _INDEX


def search(
    query: str,
    k: int = 20,
    faiss_path: Path = DENSE_FAISS_PATH,
    meta_path: Path = DENSE_META_PATH,
) -> list[dict]:
    idx = _get_index(faiss_path, meta_path)
    hits = idx.search(query, k=k)
    return [
        {
            "chunk_id":   h.chunk_id,
            "doc_id":     h.doc_id,
            "article_ref":h.article_ref,
            "score":      h.score,
            "text":       h.text,
            "retriever":  "dense",
        }
        for h in hits
    ]


def reload() -> None:
    global _INDEX
    _INDEX = None
