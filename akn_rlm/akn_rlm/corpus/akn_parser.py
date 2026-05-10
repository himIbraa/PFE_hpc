"""
AKN Parser — Phase A.

Parses all Akoma Ntoso XML files with lxml.  For each <article> element
produces one Article object containing:
  - canonical FRBR-derived doc_id
  - article_ref  (human-readable number from <num>)
  - text_ar      (raw Arabic concatenation of all <p> in the article)
  - text_normalized (normalize_arabic(text_ar))
  - frbr_uri     (<FRBRthis value="..."> of the Work FRBR)
  - ancestors    (book/part/chapter/section/subsection nums found above)
  - eid          (eId attribute of the <article> element)
  - filename_stem, doc_title, doc_date, doc_type
  - paragraphs   (list of individual <p> texts, for paragraph-level chunking)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lxml import etree

from akn_rlm.normalizers import normalize_arabic

logger = logging.getLogger(__name__)

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
_NS    = {"akn": AKN_NS}
_T     = lambda local: f"{{{AKN_NS}}}{local}"   # noqa: E731

SKIP_MARKER = "[API CONVERSION FAILED"

CONTAINER_TAGS = frozenset(["book", "part", "title", "chapter", "section", "subsection"])


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Article:
    """One article extracted from an AKN XML file."""
    doc_id:           str           # canonical FRBR id, e.g. '75-58_1975-09-26'
    article_ref:      str           # normalised number, e.g. '4' or '4 مكرر'
    text_ar:          str           # raw Arabic full-article text
    text_normalized:  str           # normalize_arabic(text_ar)
    frbr_uri:         str           # FRBRWork/FRBRthis @value
    ancestors:        Dict[str, str]  # {book:"1", chapter:"2", ...}
    eid:              str           # <article eId="art_4">
    filename_stem:    str           # original XML filename stem
    doc_title:        str
    doc_date:         str
    doc_type:         str           # law | order | decree | ...
    paragraphs:       List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Article({self.doc_id}/{self.eid} [{len(self.text_ar)} chars])"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local(element: etree._Element) -> str:
    return etree.QName(element.tag).localname if element.tag else ""


def _collect_text(element: etree._Element) -> str:
    """Recursively collect all inner text, skipping SKIP_MARKER nodes."""
    parts: List[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_collect_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _parse_frbr(root: etree._Element) -> Tuple[str, str, str, str, str]:
    """Return (canonical_doc_id, frbr_uri, doc_date, doc_type, doc_title)."""
    frbr_number = ""
    frbr_date   = ""
    frbr_uri    = ""
    doc_type    = "law"
    doc_title   = ""

    work = root.find(f".//{_T('FRBRWork')}")
    if work is not None:
        this_el = work.find(_T("FRBRthis"))
        if this_el is not None:
            frbr_uri = this_el.get("value", "")
            # /akn/dz/act/<type>/<date>/<number>/main
            parts = frbr_uri.strip("/").split("/")
            if len(parts) >= 6:
                doc_type   = parts[3]    # law | order | decree | ...
                frbr_date  = parts[4]    # 1975-09-26
                frbr_number = parts[5]   # 75-58

        num_el = work.find(_T("FRBRnumber"))
        if num_el is not None and not frbr_number:
            frbr_number = num_el.get("value", "")

        date_el = work.find(_T("FRBRdate"))
        if date_el is not None and not frbr_date:
            frbr_date = date_el.get("date", "")

    title_el = root.find(f".//{_T('docTitle')}")
    if title_el is not None:
        doc_title = _collect_text(title_el).strip()

    # canonical_id: number_date  e.g. "75-58_1975-09-26"
    # Guard against double-date: some FRBR paths embed the date in the number slot
    # (e.g. /akn/dz/act/law/2001-08-19/01-14_2001-08-19/main).
    if frbr_number and frbr_date:
        if frbr_number.endswith(f"_{frbr_date}") or frbr_number == frbr_date:
            canonical_id = frbr_number
        else:
            canonical_id = f"{frbr_number}_{frbr_date}"
    elif frbr_number:
        canonical_id = frbr_number
    else:
        canonical_id = ""

    # Reject obviously-invalid canonical IDs (FRBR placeholder literals like
    # "[NUMBER]" or non-numeric stubs like "main").  Use filename stem instead.
    _INVALID_NUMBER = re.compile(r"^\[.*\]|^main$", re.IGNORECASE)
    if canonical_id and _INVALID_NUMBER.match(canonical_id.split("_")[0]):
        canonical_id = ""   # will be overwritten from filename_stem downstream

    return canonical_id, frbr_uri, frbr_date, doc_type, doc_title


def _extract_ancestors(article_el: etree._Element) -> Dict[str, str]:
    ancestors: Dict[str, str] = {}
    for anc in article_el.iterancestors():
        tag = _local(anc)
        if tag in CONTAINER_TAGS:
            num_el = anc.find(_T("num"))
            if num_el is not None:
                ancestors[tag] = _collect_text(num_el).strip()
            else:
                ancestors[tag] = anc.get("eId", "")
    return ancestors


def _extract_paragraphs(article_el: etree._Element) -> List[str]:
    """Return list of non-empty <p> texts inside the article."""
    paragraphs: List[str] = []
    content = article_el.find(_T("content"))
    search_root = content if content is not None else article_el

    for p_el in search_root.findall(f".//{_T('p')}"):
        text = _collect_text(p_el).strip()
        if text and SKIP_MARKER not in text:
            paragraphs.append(text)

    if not paragraphs:
        raw = _collect_text(search_root).strip()
        if raw and SKIP_MARKER not in raw:
            paragraphs = [raw]

    return paragraphs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_akn_file(xml_path: Path) -> List[Article]:
    """Parse a single AKN XML file → list of Article objects."""
    try:
        tree = etree.parse(str(xml_path))
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", xml_path.name, exc)
        return []

    root = tree.getroot()
    body = root.find(f".//{_T('body')}")
    if body is None:
        logger.debug("No <body> in %s — skipping", xml_path.name)
        return []

    canonical_id, frbr_uri, doc_date, doc_type, doc_title = _parse_frbr(root)
    filename_stem = xml_path.stem

    if not canonical_id:
        canonical_id = filename_stem
        logger.info("FRBR missing in %s — using filename stem as doc_id", xml_path.name)

    articles: List[Article] = []
    from akn_rlm.normalizers import normalize_article_ref

    for art_el in body.findall(f".//{_T('article')}"):
        eid = art_el.get("eId", "")

        num_el = art_el.find(_T("num"))
        raw_ref = _collect_text(num_el).strip() if num_el is not None else eid

        paragraphs = _extract_paragraphs(art_el)
        # Always emit the Article so the eId is registered in the registry
        # and has_article() returns True for repealed/empty articles.
        # The chunker will skip articles where text_ar is empty.
        text_ar         = " ".join(paragraphs)
        text_normalized = normalize_arabic(text_ar)
        ancestors       = _extract_ancestors(art_el)
        article_ref     = normalize_article_ref(raw_ref) if raw_ref else eid

        articles.append(Article(
            doc_id          = canonical_id,
            article_ref     = article_ref,
            text_ar         = text_ar,
            text_normalized = text_normalized,
            frbr_uri        = frbr_uri,
            ancestors       = ancestors,
            eid             = eid,
            filename_stem   = filename_stem,
            doc_title       = doc_title,
            doc_date        = doc_date,
            doc_type        = doc_type,
            paragraphs      = paragraphs,
        ))

    return articles


def parse_all(akn_dir: Optional[Path] = None) -> List[Article]:
    """Parse all *.xml files in akn_dir and return a flat Article list."""
    from akn_rlm.config import get_akn_dir
    directory = akn_dir or get_akn_dir()

    xml_files = sorted(directory.glob("*.xml"))
    logger.info("Parsing %d AKN files from %s", len(xml_files), directory)

    all_articles: List[Article] = []
    for xml_path in xml_files:
        file_articles = parse_akn_file(xml_path)
        all_articles.extend(file_articles)
        logger.debug("  %s → %d articles", xml_path.name, len(file_articles))

    logger.info("Total articles parsed: %d", len(all_articles))
    return all_articles
