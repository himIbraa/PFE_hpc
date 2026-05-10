"""Dense retriever: BAAI/bge-m3 + FAISS IndexFlatIP."""
from __future__ import annotations

import logging
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

# multilingual-e5-small requires instruction prefixes (per official docs):
#   queries   → "query: <text>"
#   passages  → "passage: <text>"
_QUERY_PREFIX = "query: "
_DOC_PREFIX   = "passage: "


@dataclass
class DenseHit:
    chunk_id: str
    doc_id: str
    article_ref: str
    score: float
    text: str


def _load_model(model_name: str = EMBED_MODEL):
    from sentence_transformers import SentenceTransformer  # type: ignore
    log.info("Loading dense encoder: %s", model_name)
    # Force CPU to avoid Windows pagefile OOM when mmap-ing safetensors to VRAM
    model = SentenceTransformer(
        model_name,
        device="cpu",
        trust_remote_code=True,
    )
    return model


class DenseIndex:
    def __init__(self, index, meta_df) -> None:
        self._index   = index    # faiss.IndexFlatIP
        self._meta_df = meta_df  # pandas DataFrame: chunk_id, doc_id, article_ref, text_norm
        self._model   = None     # lazy-loaded at search time

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
        # Apply "passage: " prefix required by multilingual-e5-small
        raw_texts = [normalize_arabic(c.text_norm or c.text) for c in chunks]
        prefixed  = [_DOC_PREFIX + t for t in raw_texts]
        log.info("Dense encoding %d chunks (batch=%d)...", len(prefixed), batch_size)
        embs = model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-norm -> inner product == cosine
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = embs.shape[1]
        log.info("Embedding dim=%d, building FAISS IndexFlatIP...", dim)
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        # Store full article text (no truncation) for get_article() retrieval
        meta_df = pd.DataFrame(
            {
                "chunk_id":   [c.chunk_id    for c in chunks],
                "doc_id":     [c.doc_id      for c in chunks],
                "article_ref":[c.article_ref  for c in chunks],
                "text_norm":  raw_texts,   # full text, no [:600] limit
            }
        )
        log.info("Dense index built: %d vectors", index.ntotal)
        return cls(index, meta_df)

    def save(
        self,
        faiss_path: Path = DENSE_FAISS_PATH,
        meta_path: Path = DENSE_META_PATH,
    ) -> None:
        import faiss  # type: ignore
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(faiss_path))
        self._meta_df.to_parquet(meta_path, index=False)
        log.info("Dense index saved -> %s + %s", faiss_path, meta_path)

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
        log.info(
            "Dense index loaded from %s  (%d vectors)", faiss_path, index.ntotal
        )
        return cls(index, meta_df)

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 20) -> List[DenseHit]:
        if self._model is None:
            self._model = _load_model()
        # Apply "query: " prefix required by multilingual-e5-small
        q_text = _QUERY_PREFIX + normalize_arabic(query)
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
