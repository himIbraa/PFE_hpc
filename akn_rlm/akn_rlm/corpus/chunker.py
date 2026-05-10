"""
Chunker — Phase A.

Produces two granularities of text chunks from Article objects:

  article_chunks()   — one Chunk per article (full article text)
  paragraph_chunks() — one Chunk per paragraph within each article

Both granularities carry the same metadata so they can be stored in the
same FAISS/SPLADE/BM25 index and retrieved at the right level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from akn_rlm.normalizers import canonical_article_ref, normalize_arabic


@dataclass
class Chunk:
    """A text unit ready for indexing."""
    chunk_id:     str            # unique: "{doc_id}#{eid}#{para_idx}" or "{doc_id}#{eid}"
    doc_id:       str            # canonical FRBR doc_id
    eid:          str            # article eId
    article_ref:  str            # human-readable article number
    para_idx:     int            # -1 for article-level chunks
    text:         str            # raw Arabic text
    text_norm:    str            # normalize_arabic(text)
    granularity:  str            # "article" | "paragraph"
    ancestors:    Dict[str, str] = field(default_factory=dict)
    doc_title:    str = ""
    doc_date:     str = ""
    doc_type:     str = ""
    frbr_uri:     str = ""

    def __repr__(self) -> str:
        return f"Chunk({self.chunk_id!r} [{len(self.text)} chars])"


def article_chunks(articles: list) -> List[Chunk]:
    """One Chunk per article (full concatenated text). Skips empty/repealed articles."""
    chunks: List[Chunk] = []
    for art in articles:
        if not art.text_ar.strip():
            continue   # repealed / API-failed articles: registered in registry but not indexed
        chunk_id = f"{art.doc_id}#{art.eid}"
        chunks.append(Chunk(
            chunk_id    = chunk_id,
            doc_id      = art.doc_id,
            eid         = art.eid,
            article_ref = canonical_article_ref(art.article_ref),
            para_idx    = -1,
            text        = art.text_ar,
            text_norm   = art.text_normalized,
            granularity = "article",
            ancestors   = dict(art.ancestors),
            doc_title   = art.doc_title,
            doc_date    = art.doc_date,
            doc_type    = art.doc_type,
            frbr_uri    = art.frbr_uri,
        ))
    return chunks


def paragraph_chunks(articles: list) -> List[Chunk]:
    """One Chunk per paragraph within each article.

    Falls back to the full article text as a single chunk when no
    individual paragraphs are available.
    """
    chunks: List[Chunk] = []
    for art in articles:
        paras = art.paragraphs if art.paragraphs else [art.text_ar]
        for idx, para_text in enumerate(paras):
            if not para_text.strip():
                continue
            chunk_id = f"{art.doc_id}#{art.eid}#p{idx}"
            chunks.append(Chunk(
                chunk_id    = chunk_id,
                doc_id      = art.doc_id,
                eid         = art.eid,
                article_ref = canonical_article_ref(art.article_ref),
                para_idx    = idx,
                text        = para_text,
                text_norm   = normalize_arabic(para_text),
                granularity = "paragraph",
                ancestors   = dict(art.ancestors),
                doc_title   = art.doc_title,
                doc_date    = art.doc_date,
                doc_type    = art.doc_type,
                frbr_uri    = art.frbr_uri,
            ))
    return chunks
