"""BM25 retriever using rank_bm25.BM25Okapi."""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from rank_bm25 import BM25Okapi

from akn_rlm.config import BM25_INDEX_PATH
from akn_rlm.normalizers import normalize_arabic

log = logging.getLogger(__name__)

_PUNCT_RE = re.compile(
    r"[،؛؟!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]+"
)

# Legal IDs like 06-01, 2006-02-20, 75-58_1975-09-26 must be single tokens.
# We protect them before _PUNCT_RE by replacing separators with U+E001 (PUA),
# which is NOT in the _PUNCT_RE range and survives normalize_arabic().
_LEGAL_ID_RE = re.compile(r'\b\d{2,4}(?:[-_]\d{2,4})+\b')
_LEGAL_SEP = ""

# Arabic prefixes stripped longest-first; only when remainder >= 3 chars
_AR_PREFIXES = ["وال", "بال", "فال", "كال", "لل", "ال", "ب", "ل", "و", "ف", "ك"]


def _arabic_stem(token: str) -> str:
    """Strip common Arabic definite-article and preposition prefixes.

    Turns "للخلع" -> "خلع", "الزوجة" -> "زوجة", etc.
    Minimum remainder is 3 characters to avoid over-stripping.
    """
    for prefix in _AR_PREFIXES:
        if token.startswith(prefix) and len(token) - len(prefix) >= 3:
            return token[len(prefix):]
    return token


def _tokenize(text: str) -> List[str]:
    text = normalize_arabic(text)
    # Protect legal IDs as single tokens before punctuation stripping.
    text = _LEGAL_ID_RE.sub(
        lambda m: m.group().replace("-", _LEGAL_SEP).replace("_", _LEGAL_SEP),
        text,
    )
    text = _PUNCT_RE.sub(" ", text)
    return [_arabic_stem(t) for t in text.split() if t]


@dataclass
class BM25Hit:
    chunk_id: str
    doc_id: str
    article_ref: str
    score: float
    text: str


class BM25Index:
    def __init__(self, bm25: BM25Okapi, meta: list) -> None:
        self._bm25 = bm25
        self._meta = meta  # list[dict]: chunk_id, doc_id, article_ref, text

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, chunks: Sequence) -> "BM25Index":
        log.info("Building BM25 index over %d chunks", len(chunks))
        tokenized = [_tokenize(c.text_norm or c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        meta = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "article_ref": c.article_ref,
                "text": c.text_norm or c.text,  # full text; no truncation
            }
            for c in chunks
        ]
        return cls(bm25, meta)

    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"bm25": self._bm25, "meta": self._meta}, fh)
        log.info("BM25 index saved -> %s  (%d chunks)", path, len(self._meta))

    @classmethod
    def load(cls, path: Path = BM25_INDEX_PATH) -> "BM25Index":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        log.info("BM25 index loaded from %s  (%d chunks)", path, len(d["meta"]))
        return cls(d["bm25"], d["meta"])

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 20) -> List[BM25Hit]:
        toks = _tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            BM25Hit(
                chunk_id=self._meta[i]["chunk_id"],
                doc_id=self._meta[i]["doc_id"],
                article_ref=self._meta[i]["article_ref"],
                score=float(scores[i]),
                text=self._meta[i]["text"],
            )
            for i in top_idx
        ]
