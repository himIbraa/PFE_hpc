"""
Build all four retrieval indices for AKN-RLM.

Usage:
    conda run -n pfe_env python scripts/build_indices.py
    conda run -n pfe_env python scripts/build_indices.py --quick-check
    conda run -n pfe_env python scripts/build_indices.py --index dense --force
"""
from __future__ import annotations

import hashlib
import io
import logging
import sys
import time
from pathlib import Path

import click

# UTF-8 console on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make project importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akn_rlm.config import (
    BM25_INDEX_PATH,
    COLBERT_INDEX_DIR,
    DENSE_FAISS_PATH,
    DENSE_META_PATH,
    ENABLE_COLBERT,
    ENABLE_DENSE,
    ENABLE_SPLADE,
    INDICES_DIR,
    SPLADE_INDEX_PATH,
)
from akn_rlm.corpus.akn_parser import parse_all
from akn_rlm.corpus.chunker import article_chunks
from akn_rlm.indexers.bm25 import BM25Index
from akn_rlm.indexers.colbert import ColBERTIndex_build, ColBERTIndex_load
from akn_rlm.indexers.dense import DenseIndex
from akn_rlm.indexers.splade import SPLADEIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_indices")

_HASH_FILE = INDICES_DIR / "corpus_hash.txt"

# ---------------------------------------------------------------------------
# Content hash (based on XML filenames + sizes -- fast, no full read)
# ---------------------------------------------------------------------------

def _corpus_hash() -> str:
    from akn_rlm.config import get_akn_dir
    akn_dir = get_akn_dir()
    files = sorted(akn_dir.glob("*.xml"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode("utf-8"))
        h.update(str(f.stat().st_size).encode("utf-8"))
    return h.hexdigest()[:20]


def _load_saved_hash() -> str:
    if _HASH_FILE.exists():
        return _HASH_FILE.read_text(encoding="utf-8").strip()
    return ""


def _save_hash(h: str) -> None:
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    _HASH_FILE.write_text(h, encoding="utf-8")


# ---------------------------------------------------------------------------
# Index existence checks
# ---------------------------------------------------------------------------

def _bm25_exists()    -> bool: return BM25_INDEX_PATH.exists()
def _splade_exists()  -> bool: return SPLADE_INDEX_PATH.with_suffix(".npz").exists()
def _dense_exists()   -> bool: return DENSE_FAISS_PATH.exists() and DENSE_META_PATH.exists()
def _colbert_exists() -> bool: return COLBERT_INDEX_DIR.exists() and any(COLBERT_INDEX_DIR.iterdir() if COLBERT_INDEX_DIR.exists() else [])


# ---------------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------------

def build_bm25(chunks, force: bool) -> None:
    if _bm25_exists() and not force:
        log.info("BM25 index already exists -- skipping (use --force to rebuild)")
        return
    t0 = time.time()
    idx = BM25Index.build(chunks)
    idx.save(BM25_INDEX_PATH)
    log.info("BM25 done in %.1fs", time.time() - t0)


def build_splade(chunks, force: bool) -> None:
    if _splade_exists() and not force:
        log.info("SPLADE index already exists -- skipping (use --force to rebuild)")
        return
    t0 = time.time()
    idx = SPLADEIndex.build(chunks)
    idx.save(SPLADE_INDEX_PATH)
    log.info("SPLADE done in %.1fs", time.time() - t0)


def build_dense(chunks, force: bool) -> None:
    if _dense_exists() and not force:
        log.info("Dense index already exists -- skipping (use --force to rebuild)")
        return
    t0 = time.time()
    idx = DenseIndex.build(chunks)
    idx.save(DENSE_FAISS_PATH, DENSE_META_PATH)
    log.info("Dense done in %.1fs", time.time() - t0)


def build_colbert(chunks, force: bool) -> None:
    if _colbert_exists() and not force:
        log.info("ColBERT index already exists -- skipping (use --force to rebuild)")
        return
    t0 = time.time()
    idx = ColBERTIndex_build(chunks)
    idx.save(COLBERT_INDEX_DIR)
    log.info("ColBERT done in %.1fs", time.time() - t0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

# Semantic query (dense / SPLADE / ColBERT): "conditions of khul'"
_SANITY_QUERY_SEMANTIC = "شروط الخلع"
# BM25-friendly query: uses tokens that literally appear in Art.54 after prefix-stripping
# Art.54 text contains: "للخلع"->"خلع", "بمقابل"->"مقابل", "الزوجة"->"زوجة"
_SANITY_QUERY_BM25     = "الخلع بمقابل مالي الزوجة"
_FAMILY_CODE_ID        = "84-11_1984-06-09"
_TARGET_ART            = "54"


def _article_ref_match(hit_article_ref: str) -> bool:
    """True when the hit is approximately Art.54 of Family Code."""
    ref = hit_article_ref.strip()
    return ref in ("54", "art_54", "art54")


def _quick_check() -> bool:
    log.info("=" * 60)
    log.info("QUICK CHECK — target: %s art.%s", _FAMILY_CODE_ID, _TARGET_ART)
    log.info("  BM25 query   : %s", _SANITY_QUERY_BM25)
    log.info("  Semantic query: %s", _SANITY_QUERY_SEMANTIC)
    log.info("=" * 60)
    all_pass = True

    def _check_hits(name: str, hits, query: str) -> None:
        nonlocal all_pass
        top10 = hits[:10]
        target_hits = [
            (i + 1, h) for i, h in enumerate(top10)
            if h.doc_id == _FAMILY_CODE_ID and _article_ref_match(h.article_ref)
        ]
        if target_hits:
            rank, h = target_hits[0]
            log.info("  PASS  %-10s  Art.54 at rank %d  score=%.4f", name, rank, h.score)
        else:
            all_pass = False
            top5 = [(h.doc_id, h.article_ref, round(h.score, 4)) for h in top10[:5]]
            log.warning("  FAIL  %-10s  Art.54 NOT in top-10  top5=%s", name, top5)

    # BM25 — uses lexical query matching Art.54 vocabulary
    if _bm25_exists():
        bm25 = BM25Index.load(BM25_INDEX_PATH)
        _check_hits("BM25", bm25.search(_SANITY_QUERY_BM25, k=10), _SANITY_QUERY_BM25)
    else:
        log.warning("  SKIP  BM25   (not built)")

    # SPLADE (bge-m3 sparse head — works for Arabic)
    if not ENABLE_SPLADE:
        log.info("  SKIP  SPLADE (disabled — ENABLE_SPLADE=False)")
    elif _splade_exists():
        splade = SPLADEIndex.load(SPLADE_INDEX_PATH)
        _check_hits("SPLADE", splade.search(_SANITY_QUERY_SEMANTIC, k=10), _SANITY_QUERY_SEMANTIC)
    else:
        log.warning("  SKIP  SPLADE (not built)")

    # Dense
    if not ENABLE_DENSE:
        log.info("  SKIP  Dense  (disabled — ENABLE_DENSE=False)")
    elif _dense_exists():
        dense = DenseIndex.load(DENSE_FAISS_PATH, DENSE_META_PATH)
        _check_hits("Dense", dense.search(_SANITY_QUERY_SEMANTIC, k=10), _SANITY_QUERY_SEMANTIC)
    else:
        log.warning("  SKIP  Dense  (not built)")

    # ColBERT
    if _colbert_exists():
        colbert = ColBERTIndex_load(COLBERT_INDEX_DIR)
        _check_hits("ColBERT", colbert.search(_SANITY_QUERY_SEMANTIC, k=10), _SANITY_QUERY_SEMANTIC)
    else:
        log.warning("  SKIP  ColBERT (not built)")

    status = "PASS" if all_pass else "FAIL"
    log.info("=" * 60)
    log.info("Quick-check: %s", status)
    log.info("=" * 60)
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--index",
    type=click.Choice(["bm25", "splade", "dense", "colbert", "all"]),
    default="all",
    show_default=True,
    help="Which index to build.",
)
@click.option("--force",       is_flag=True, help="Rebuild even if index already exists.")
@click.option("--quick-check", is_flag=True, help="Run sanity check after building.")
@click.option("--check-only",  is_flag=True, help="Only run sanity check, skip building.")
def main(index: str, force: bool, quick_check: bool, check_only: bool) -> None:
    if check_only:
        ok = _quick_check()
        sys.exit(0 if ok else 1)

    current_hash = _corpus_hash()
    saved_hash   = _load_saved_hash()
    if current_hash == saved_hash and not force:
        # Only skip if every requested index already exists
        missing = []
        if index in ("bm25",    "all") and not _bm25_exists():                                   missing.append("bm25")
        if index in ("splade",  "all") and ENABLE_SPLADE  and not _splade_exists():          missing.append("splade")
        if index in ("dense",   "all") and ENABLE_DENSE   and not _dense_exists():           missing.append("dense")
        if index in ("colbert", "all") and ENABLE_COLBERT and not _colbert_exists():         missing.append("colbert")

        if not missing:
            log.info("Corpus hash unchanged (%s) and all indices present — nothing to do.", current_hash)
            if quick_check:
                ok = _quick_check()
                sys.exit(0 if ok else 1)
            return
        log.info("Corpus hash unchanged but missing indices: %s — building them.", missing)

    log.info("Parsing corpus...")
    t0 = time.time()
    articles = parse_all()
    chunks   = article_chunks(articles)
    log.info("Corpus: %d articles, %d indexable chunks (%.1fs)", len(articles), len(chunks), time.time() - t0)

    if index in ("bm25",    "all"): build_bm25(chunks, force)
    if index in ("splade",  "all") and ENABLE_SPLADE:  build_splade(chunks,  force)
    if index in ("dense",   "all") and ENABLE_DENSE:   build_dense(chunks,   force)
    if index in ("colbert", "all") and ENABLE_COLBERT: build_colbert(chunks, force)

    _save_hash(current_hash)
    log.info("All requested indices built.  Hash saved: %s", current_hash)

    if quick_check:
        ok = _quick_check()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
