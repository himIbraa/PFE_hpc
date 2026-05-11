"""Dense retriever: model-agnostic FAISS IndexFlatIP.

Supports both:
- intfloat/multilingual-e5-small  (legacy F5 baseline, 384-dim, requires prefixes)
- BAAI/bge-m3                      (HPC ceiling-breaker, 1024-dim, no prefix)

The active model is selected by the EMBED_MODEL config / env var. The same
DenseIndex serializes either; corpus_hash.txt records which model produced
the on-disk index so loaders can sanity-check.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from akn_rlm.config import (
    DENSE_FAISS_PATH,
    DENSE_META_PATH,
    EMBED_MODEL,
)
from akn_rlm.normalizers import normalize_arabic

log = logging.getLogger(__name__)


def _is_e5(model_name: str) -> bool:
    n = model_name.lower()
    return n.startswith("intfloat/") and "e5" in n


def _is_bge_m3(model_name: str) -> bool:
    return "bge-m3" in model_name.lower()


def _is_gte_qwen_instruct(model_name: str) -> bool:
    # Alibaba-NLP/gte-Qwen2-7B-instruct or any "gte-*-instruct" variant.
    n = model_name.lower()
    return "gte" in n and "qwen" in n and "instruct" in n


# Default instruction prefix used by gte-Qwen2-7B-instruct and gte-multilingual-
# reranker family per the model cards. The instruction can be tuned per task;
# for a legal-IR corpus the default works well enough as a first cut.
GTE_QWEN_INSTRUCT_QUERY = (
    "Instruct: Given an Arabic legal question, retrieve the article(s) that "
    "answer it.\nQuery: "
)


def _doc_format(text: str, model_name: str) -> str:
    if _is_e5(model_name):
        return f"passage: {text}"
    # gte-Qwen2-7B-instruct and BGE-m3: documents take the bare text.
    return text


def _query_format(text: str, model_name: str) -> str:
    if _is_e5(model_name):
        return f"query: {text}"
    if _is_gte_qwen_instruct(model_name):
        return GTE_QWEN_INSTRUCT_QUERY + text
    return text


@dataclass
class DenseHit:
    chunk_id: str
    doc_id: str
    article_ref: str
    score: float
    text: str


def _needs_fp16(model_name: str) -> bool:
    """Whether to load the model in fp16 to fit on a typical GPU slice.

    Triggered by:
      - gte-Qwen2-7B-instruct (7B params, 28 GB at fp32 → won't fit on
        a 22 GB MIG slice; fp16 is 14 GB and works)
      - any name containing "7b" / "8b" / "13b" (size-based heuristic)
      - explicit AKN_FORCE_FP16=1 env flag
    """
    if os.getenv("AKN_FORCE_FP16", "").strip().lower() in {"1", "true", "yes"}:
        return True
    n = model_name.lower()
    if _is_gte_qwen_instruct(model_name):
        return True
    return any(tag in n for tag in ("-7b", "-8b", "-13b", "-14b", "-30b"))


def _load_model(model_name: str = EMBED_MODEL):
    """Return a loaded SentenceTransformer (or compatible) for the requested model.

    Device: CUDA if available else CPU.
    fp16: forced for large encoders (see :func:`_needs_fp16`); default
    precision otherwise.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore
    try:
        import torch  # type: ignore
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    kwargs: dict = {"device": device, "trust_remote_code": True}
    if device == "cuda" and _needs_fp16(model_name):
        # sentence-transformers passes model_kwargs through to
        # transformers from_pretrained; torch_dtype is the canonical key
        # (transformers >=5.x deprecates it in favour of `dtype` but
        # still accepts it with a FutureWarning).
        kwargs["model_kwargs"] = {"torch_dtype": "float16"}
        log.info("Loading dense encoder: %s on %s (fp16)", model_name, device)
    else:
        log.info("Loading dense encoder: %s on %s", model_name, device)
    model = SentenceTransformer(model_name, **kwargs)
    return model


class DenseIndex:
    """FAISS IndexFlatIP over normalised dense embeddings.

    Output dim depends on the configured EMBED_MODEL:
    - multilingual-e5-small -> 384
    - BAAI/bge-m3            -> 1024
    """

    def __init__(self, index, meta_df, *, model_name: str = EMBED_MODEL) -> None:
        self._index   = index
        self._meta_df = meta_df
        self._model   = None         # lazy-loaded at search time
        self._model_name = model_name

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        chunks: Sequence,
        model_name: str = EMBED_MODEL,
        batch_size: int = 16,
    ) -> "DenseIndex":
        import faiss    # type: ignore
        import pandas as pd  # type: ignore

        model = _load_model(model_name)
        raw_texts = [normalize_arabic(c.text_norm or c.text) for c in chunks]
        prefixed  = [_doc_format(t, model_name) for t in raw_texts]
        log.info(
            "Dense encoding %d chunks with %s (batch=%d)...",
            len(prefixed), model_name, batch_size,
        )
        embs = model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = embs.shape[1]
        log.info("Embedding dim=%d, building FAISS IndexFlatIP...", dim)
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        meta_df = pd.DataFrame(
            {
                "chunk_id":   [c.chunk_id    for c in chunks],
                "doc_id":     [c.doc_id      for c in chunks],
                "article_ref":[c.article_ref  for c in chunks],
                "text_norm":  raw_texts,
            }
        )
        log.info("Dense index built: %d vectors", index.ntotal)
        return cls(index, meta_df, model_name=model_name)

    def save(
        self,
        faiss_path: Path = DENSE_FAISS_PATH,
        meta_path: Path = DENSE_META_PATH,
    ) -> None:
        import faiss  # type: ignore
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(faiss_path))
        self._meta_df.to_parquet(meta_path, index=False)
        # Record which model built this index so loaders / build scripts can
        # detect a model swap (BGE-m3 has 1024-dim, e5-small 384-dim — the
        # FAISS file alone wouldn't make the source obvious).
        # NOTE: `with_suffix` replaces the existing suffix, so the file lives
        # next to dense.faiss as dense.faiss.model.txt (and not dense.model.txt).
        meta_sidecar = faiss_path.parent / (faiss_path.name + ".model.txt")
        meta_sidecar.write_text(self._model_name, encoding="utf-8")
        log.info("Dense index saved -> %s + %s (model=%s)", faiss_path, meta_path, self._model_name)

    @classmethod
    def load(
        cls,
        faiss_path: Path = DENSE_FAISS_PATH,
        meta_path: Path = DENSE_META_PATH,
    ) -> "DenseIndex":
        import faiss  # type: ignore
        import pandas as pd  # type: ignore
        index = faiss.read_index(str(faiss_path))
        meta_df = pd.read_parquet(meta_path)
        # Pick up the recorded model name; fall back to current EMBED_MODEL.
        # NOTE: `with_suffix` replaces the existing suffix, so the file lives
        # next to dense.faiss as dense.faiss.model.txt (and not dense.model.txt).
        meta_sidecar = faiss_path.parent / (faiss_path.name + ".model.txt")
        if meta_sidecar.exists():
            stored = meta_sidecar.read_text(encoding="utf-8").strip()
        else:
            stored = EMBED_MODEL
        log.info(
            "Dense index loaded from %s  (%d vectors, model=%s)", faiss_path, index.ntotal, stored,
        )
        return cls(index, meta_df, model_name=stored)

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 20) -> List[DenseHit]:
        if self._model is None:
            self._model = _load_model(self._model_name)
        q_text = _query_format(normalize_arabic(query), self._model_name)
        q_emb = self._model.encode(
            [q_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, indices = self._index.search(q_emb, k)
        scores = scores[0]
        indices = indices[0]
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            row = self._meta_df.iloc[idx]
            results.append(
                DenseHit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    article_ref=row["article_ref"],
                    score=float(score),
                    text=row["text_norm"],
                )
            )
        return results
