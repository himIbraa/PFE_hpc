"""Late-interaction retriever using BAAI/bge-m3 multi-vector head.

Two modes:
  - Native:   colbert-ai installed → PLAID index, true MaxSim on GPU
  - Fallback: colbert-ai absent    → store bge-m3 token embeddings,
              batched MaxSim scoring (GPU via torch if available, else numpy)

Replaces the AraBERT bi-encoder fallback whose mean-pooled embeddings
collapsed to a single similarity value for all documents.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from akn_rlm.config import COLBERT_INDEX_DIR, COLBERT_MODEL
from akn_rlm.indexers._bge_m3_loader import get_bge_m3
from akn_rlm.normalizers import normalize_arabic

log = logging.getLogger(__name__)

_HAS_COLBERT = False
try:
    import colbert  # type: ignore  # noqa: F401
    _HAS_COLBERT = True
except ImportError:
    pass


@dataclass
class ColBERTHit:
    chunk_id: str
    doc_id: str
    article_ref: str
    score: float
    text: str


# ---------------------------------------------------------------------------
# MaxSim scoring (batched, GPU-first)
# ---------------------------------------------------------------------------

def _maxsim_scores(q_vecs: np.ndarray, doc_vecs_list: list) -> np.ndarray:
    """Compute MaxSim(q, d) for every document.

    q_vecs      : (q_len, dim) float32, L2-normalised
    doc_vecs_list: list of N arrays (d_len_i, dim) float16

    Returns (N,) float32 scores.
    """
    try:
        import torch  # type: ignore
        return _maxsim_torch(q_vecs, doc_vecs_list, torch)
    except ImportError:
        return _maxsim_numpy(q_vecs, doc_vecs_list)


def _maxsim_torch(q_vecs, doc_vecs_list, torch, batch_size: int = 256) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.tensor(q_vecs, device=device, dtype=torch.float32)   # (q_len, dim)
    n = len(doc_vecs_list)
    scores = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        batch = doc_vecs_list[start: start + batch_size]
        max_d = max(d.shape[0] for d in batch)
        b     = len(batch)

        padded = np.zeros((b, max_d, q_vecs.shape[1]), dtype=np.float32)
        mask   = np.full((b, max_d), -1e4, dtype=np.float32)
        for j, d in enumerate(batch):
            padded[j, : d.shape[0]] = d.astype(np.float32)
            mask  [j, : d.shape[0]] = 0.0

        d_t = torch.tensor(padded, device=device, dtype=torch.float32)  # (b, max_d, dim)
        m_t = torch.tensor(mask,   device=device, dtype=torch.float32)  # (b, max_d)

        # sim: (q_len, b, max_d)
        sim = torch.einsum("qd,bld->qbl", q, d_t) + m_t.unsqueeze(0)
        s   = sim.max(dim=2).values.sum(dim=0).cpu().numpy()             # (b,)
        scores[start: start + b] = s

    return scores


def _maxsim_numpy(q_vecs, doc_vecs_list) -> np.ndarray:
    scores = np.zeros(len(doc_vecs_list), dtype=np.float32)
    for i, d_vecs in enumerate(doc_vecs_list):
        sim = q_vecs @ d_vecs.astype(np.float32).T          # (q_len, d_len)
        scores[i] = float(sim.max(axis=1).sum())
    return scores


def _l2_norm(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-9)
    return (vecs / norms).astype(np.float16)


# ---------------------------------------------------------------------------
# Fallback bi-encoder index (bge-m3 multi-vector)
# ---------------------------------------------------------------------------

class _FallbackColBERTIndex:
    _VECS_FILE = "colbert_vecs.npy"
    _META_FILE = "meta.pkl"

    def __init__(self, vecs: np.ndarray, meta: list, model_name: str) -> None:
        # vecs: numpy object array of N fp16 arrays (d_len_i, dim)
        self._vecs       = vecs
        self._meta       = meta
        self._model_name = model_name
        self._model      = None

    @classmethod
    def build(cls, chunks: Sequence, model_name: str = COLBERT_MODEL,
              batch_size: int = 16) -> "_FallbackColBERTIndex":
        model  = get_bge_m3(model_name)
        texts  = [normalize_arabic(c.text_norm or c.text) for c in chunks]
        log.info("bge-m3 multi-vector encoding %d chunks...", len(texts))

        all_vecs: list = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start: start + batch_size]
            out   = model.encode(
                batch,
                batch_size=batch_size,
                max_length=512,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )
            for v in out["colbert_vecs"]:
                all_vecs.append(_l2_norm(v))   # fp16, L2-norm per token
            if (start // batch_size + 1) % 20 == 0:
                log.info("  colbert: %d / %d", start + len(batch), len(texts))

        vecs_arr = np.empty(len(all_vecs), dtype=object)
        for i, v in enumerate(all_vecs):
            vecs_arr[i] = v

        meta = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
             "article_ref": c.article_ref, "text": (c.text_norm or c.text)[:600]}
            for c in chunks
        ]
        log.info("ColBERT multi-vector index built: %d docs", len(meta))
        return cls(vecs_arr, meta, model_name)

    def save(self, directory: Path = COLBERT_INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(str(directory / self._VECS_FILE), self._vecs, allow_pickle=True)
        with open(directory / self._META_FILE, "wb") as fh:
            pickle.dump({"meta": self._meta, "model_name": self._model_name,
                         "mode": "bge_m3_multivec"}, fh)
        log.info("ColBERT index saved → %s  (%d docs)", directory, len(self._meta))

    @classmethod
    def load(cls, directory: Path = COLBERT_INDEX_DIR) -> "_FallbackColBERTIndex":
        vecs = np.load(str(directory / cls._VECS_FILE), allow_pickle=True)
        with open(directory / cls._META_FILE, "rb") as fh:
            d = pickle.load(fh)
        log.info("ColBERT index loaded from %s  (%d docs)", directory, len(d["meta"]))
        return cls(vecs, d["meta"], d.get("model_name", COLBERT_MODEL))

    def search(self, query: str, k: int = 20) -> List[ColBERTHit]:
        if self._model is None:
            self._model = get_bge_m3(self._model_name)
        q_norm = normalize_arabic(query)
        out    = self._model.encode(
            [q_norm],
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        q_vecs = _l2_norm(out["colbert_vecs"][0]).astype(np.float32)  # (q_len, dim)

        scores  = _maxsim_scores(q_vecs, list(self._vecs))
        top_idx = np.argpartition(scores, -min(k, len(scores)))[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [
            ColBERTHit(
                chunk_id=self._meta[i]["chunk_id"],
                doc_id=self._meta[i]["doc_id"],
                article_ref=self._meta[i]["article_ref"],
                score=float(scores[i]),
                text=self._meta[i]["text"],
            )
            for i in top_idx
        ]


# ---------------------------------------------------------------------------
# Native colbert-ai PLAID index (used when colbert-ai is installed)
# ---------------------------------------------------------------------------

class _NativeColBERTIndex:
    _COLLECTION_FILE = "collection.pkl"
    _META_FILE       = "chunk_meta.pkl"
    _EXPERIMENT      = "akn_legal"

    def __init__(self, index_root: Path, meta: list) -> None:
        self._index_root = index_root
        self._meta       = meta
        self._searcher   = None

    @classmethod
    def build(cls, chunks: Sequence, model_name: str = COLBERT_MODEL,
              nbits: int = 2) -> "_NativeColBERTIndex":
        from colbert import Indexer  # type: ignore
        from colbert.infra import Run, RunConfig, ColBERTConfig  # type: ignore

        COLBERT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        texts = [normalize_arabic(c.text_norm or c.text) for c in chunks]
        meta  = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
             "article_ref": c.article_ref, "text": t[:600]}
            for c, t in zip(chunks, texts)
        ]
        log.info("ColBERT PLAID: indexing %d passages...", len(texts))
        with Run().context(
            RunConfig(nranks=1, experiment=cls._EXPERIMENT,
                      root=str(COLBERT_INDEX_DIR.parent))
        ):
            config  = ColBERTConfig(nbits=nbits, doc_maxlen=512, query_maxlen=64)
            indexer = Indexer(checkpoint=model_name, config=config)
            indexer.index(name=cls._EXPERIMENT, collection=texts, overwrite=True)

        with open(COLBERT_INDEX_DIR / cls._META_FILE, "wb") as fh:
            pickle.dump({"meta": meta, "model_name": model_name, "mode": "native"}, fh)
        return cls(COLBERT_INDEX_DIR, meta)

    def save(self, directory: Path = COLBERT_INDEX_DIR) -> None:
        log.info("ColBERT PLAID index at %s  (%d passages)", self._index_root, len(self._meta))

    @classmethod
    def load(cls, directory: Path = COLBERT_INDEX_DIR) -> "_NativeColBERTIndex":
        with open(directory / cls._META_FILE, "rb") as fh:
            d = pickle.load(fh)
        return cls(directory, d["meta"])

    def _get_searcher(self):
        from colbert import Searcher   # type: ignore
        from colbert.infra import Run, RunConfig  # type: ignore
        if self._searcher is None:
            with Run().context(
                RunConfig(nranks=1, experiment=self._EXPERIMENT,
                          root=str(self._index_root.parent))
            ):
                self._searcher = Searcher(index=self._EXPERIMENT)
        return self._searcher

    def search(self, query: str, k: int = 20) -> List[ColBERTHit]:
        searcher = self._get_searcher()
        results  = searcher.search(normalize_arabic(query), k=k)
        return [
            ColBERTHit(
                chunk_id=self._meta[pid]["chunk_id"],
                doc_id=self._meta[pid]["doc_id"],
                article_ref=self._meta[pid]["article_ref"],
                score=float(score),
                text=self._meta[pid]["text"],
            )
            for pid, _, score in zip(*results)
        ]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def ColBERTIndex_build(chunks: Sequence, **kwargs):
    if _HAS_COLBERT:
        return _NativeColBERTIndex.build(chunks, **kwargs)
    log.info("colbert-ai not installed; using bge-m3 multi-vector fallback")
    return _FallbackColBERTIndex.build(chunks, **kwargs)


def ColBERTIndex_load(directory: Path = COLBERT_INDEX_DIR):
    native_meta = directory / _NativeColBERTIndex._META_FILE
    if native_meta.exists():
        return _NativeColBERTIndex.load(directory)
    return _FallbackColBERTIndex.load(directory)
