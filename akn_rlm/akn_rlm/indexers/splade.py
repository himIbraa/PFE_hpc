"""Sparse retriever using BAAI/bge-m3 learned-sparse head.

Replaces the English-only naver/splade models.  bge-m3's sparse head
works across 100+ languages (including Arabic) because it shares the
same XLM-RoBERTa tokeniser as the dense head.

Storage format: scipy CSR matrix (n_chunks, vocab_size) + meta pickle.
Same file names as before so build_indices.py needs no path changes.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from akn_rlm.config import SPLADE_INDEX_PATH, SPLADE_MODEL
from akn_rlm.indexers._bge_m3_loader import get_bge_m3
from akn_rlm.normalizers import normalize_arabic

log = logging.getLogger(__name__)


@dataclass
class SPLADEHit:
    chunk_id: str
    doc_id: str
    article_ref: str
    score: float
    text: str


def _vocab_size(model) -> int:
    """Return the tokenizer vocabulary size of the loaded model."""
    try:
        return len(model.tokenizer)
    except Exception:
        return 250002  # XLM-RoBERTa default


def _weights_to_csr(weights_list: list, vocab_size: int):
    """Convert list of {token_id: weight} dicts to scipy CSR matrix."""
    import scipy.sparse as sp  # type: ignore
    rows, cols, data = [], [], []
    for row_idx, lw in enumerate(weights_list):
        for token_id, weight in lw.items():
            rows.append(row_idx)
            cols.append(int(token_id))
            data.append(float(weight))
    if not rows:
        return sp.csr_matrix((len(weights_list), vocab_size), dtype=np.float32)
    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(weights_list), vocab_size),
        dtype=np.float32,
    )


class SPLADEIndex:
    def __init__(self, doc_matrix, meta: list, vocab_size: int) -> None:
        self._matrix    = doc_matrix   # scipy CSR (n_chunks, vocab_size)
        self._meta      = meta
        self._vocab_size = vocab_size
        self._model     = None         # lazy-loaded at search time

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, chunks: Sequence, model_name: str = SPLADE_MODEL,
              batch_size: int = 32) -> "SPLADEIndex":
        model = get_bge_m3(model_name)
        vsz   = _vocab_size(model)
        texts = [normalize_arabic(c.text_norm or c.text) for c in chunks]
        log.info("bge-m3 sparse encoding %d chunks (vocab_size=%d)...", len(texts), vsz)

        all_weights: list = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            out   = model.encode(
                batch,
                batch_size=batch_size,
                max_length=512,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            all_weights.extend(out["lexical_weights"])
            if (start // batch_size + 1) % 20 == 0:
                log.info("  sparse: %d / %d", start + len(batch), len(texts))

        matrix = _weights_to_csr(all_weights, vsz)
        meta   = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
             "article_ref": c.article_ref, "text": (c.text_norm or c.text)[:600]}
            for c in chunks
        ]
        log.info("Sparse matrix shape: %s  nnz=%d", matrix.shape, matrix.nnz)
        return cls(matrix, meta, vsz)

    def save(self, path: Path = SPLADE_INDEX_PATH) -> None:
        import scipy.sparse as sp  # type: ignore
        path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(str(path.with_suffix(".npz")), self._matrix)
        with open(path.with_suffix(".meta.pkl"), "wb") as fh:
            pickle.dump({"meta": self._meta, "vocab_size": self._vocab_size}, fh)
        log.info("Sparse index saved → %s  (%d chunks)", path, len(self._meta))

    @classmethod
    def load(cls, path: Path = SPLADE_INDEX_PATH) -> "SPLADEIndex":
        import scipy.sparse as sp  # type: ignore
        matrix = sp.load_npz(str(path.with_suffix(".npz")))
        with open(path.with_suffix(".meta.pkl"), "rb") as fh:
            d = pickle.load(fh)
        log.info("Sparse index loaded from %s  (%d chunks)", path, len(d["meta"]))
        return cls(matrix, d["meta"], d.get("vocab_size", matrix.shape[1]))

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 20) -> List[SPLADEHit]:
        if self._model is None:
            self._model = get_bge_m3(SPLADE_MODEL)
        q_norm = normalize_arabic(query)
        out = self._model.encode(
            [q_norm],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        q_vec      = _weights_to_csr(out["lexical_weights"], self._matrix.shape[1])
        scores_raw = (q_vec @ self._matrix.T).toarray()[0]
        top_idx    = np.argpartition(scores_raw, -min(k, len(scores_raw)))[-k:]
        top_idx    = top_idx[np.argsort(scores_raw[top_idx])[::-1]]
        return [
            SPLADEHit(
                chunk_id=self._meta[i]["chunk_id"],
                doc_id=self._meta[i]["doc_id"],
                article_ref=self._meta[i]["article_ref"],
                score=float(scores_raw[i]),
                text=self._meta[i]["text"],
            )
            for i in top_idx
        ]
