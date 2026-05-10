"""
ArticleRegistry — canonical ID mapping and article lookup.

Design decisions:
  - Canonical doc_id is derived from FRBR metadata inside the XML, not the filename.
  - Historical benchmark aliases remain accepted even after on-disk repairs
    (for example 12-2003 → 03-12 and 06-15 → 06-154).
  - Known filename/type mismatches can be documented here, but repaired corpus
    files should be preferred over runtime abstention rules.

Public API:
    registry.has_article(doc_id, article_ref) -> bool
    registry.resolve_alias(alias)             -> Optional[str]  (canonical doc_id)
    registry.get_doc(doc_id)                  -> Optional[DocEntry]
    registry.is_missing(doc_id)               -> bool
    registry.is_collision_filename(filename)  -> bool
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from akn_rlm.normalizers import canonical_article_ref, normalize_arabic, ref_to_eid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known data-quality constants (never derived at runtime — documented facts)
# ---------------------------------------------------------------------------

# Filenames whose content does NOT match what the filename implies.
# Kept for compatibility with verification/reporting helpers.
FILENAME_COLLISIONS: Dict[str, tuple] = {}

# Canonical doc IDs present in older metadata but still unavailable in the live AKN
# corpus.  The current repaired dataset has no known hard gaps at runtime.
MISSING_FROM_CORPUS: Set[str] = set()

# Known type / metadata mismatches (filename_stem or canonical_id → description).
# These do NOT block use; they affect amendment event indexing.
METADATA_MISMATCHES: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Alias map  — english_alias / abbreviation → canonical_doc_id
#
# Structure: alias_key (lower, stripped) → canonical_doc_id
# Must cover all 23 benchmark categories and all known abbreviations.
# ---------------------------------------------------------------------------

# This mapping is populated at module load time and augmented from parsed FRBR.
# Format: lowercase alias string → canonical_doc_id (FRBR-derived)
_STATIC_ALIASES: Dict[str, Optional[str]] = {
    # ── Civil Code ───────────────────────────────────────────────────────────
    "civil code":              "75-58_1975-09-26",
    "civil_code":              "75-58_1975-09-26",
    "cciv":                    "75-58_1975-09-26",
    "q.m":                     "75-58_1975-09-26",   # ق.م
    "قانون مدني":              "75-58_1975-09-26",
    "القانون المدني":          "75-58_1975-09-26",
    "75-58":                   "75-58_1975-09-26",
    "75-8_1975-09-26":         "75-58_1975-09-26",   # filename-based id from benchmark
    "75-8":                    "75-58_1975-09-26",
    "أمر 75-58":               "75-58_1975-09-26",
    "أمر 75-8":                "75-58_1975-09-26",

    # ── Family Code ──────────────────────────────────────────────────────────
    "family code":             "84-11_1984-06-09",
    "family_code":             "84-11_1984-06-09",
    "family law":              "84-11_1984-06-09",
    "cfam":                    "84-11_1984-06-09",
    "قانون الأسرة":            "84-11_1984-06-09",
    "84-11":                   "84-11_1984-06-09",
    "84-11_1984-06-09":        "84-11_1984-06-09",

    # ── Commercial Code ──────────────────────────────────────────────────────
    "commercial code":         "75-59_1975-09-26",
    "commercial_code":         "75-59_1975-09-26",
    "ccom":                    "75-59_1975-09-26",
    "القانون التجاري":         "75-59_1975-09-26",
    "75-59":                   "75-59_1975-09-26",
    "75-59_1975-09-26":        "75-59_1975-09-26",   # FRBR id used in benchmark
    "1975_1975-09-26":         "75-59_1975-09-26",   # filename-based

    # ── Civil Procedure ──────────────────────────────────────────────────────
    "civil procedure":         "08-09_2008-02-25",
    "civil_procedure":         "08-09_2008-02-25",
    "code of civil procedure": "08-09_2008-02-25",
    "cpca":                    "08-09_2008-02-25",
    "cpc":                     "08-09_2008-02-25",
    "الإجراءات المدنية":       "08-09_2008-02-25",
    "08-09":                   "08-09_2008-02-25",
    "08-09_2008-02-25":        "08-09_2008-02-25",

    # ── Criminal Procedure ───────────────────────────────────────────────────
    "criminal procedure":      "25-14_2025-08-03",
    "criminal_procedure":      "25-14_2025-08-03",
    "cpp":                     "25-14_2025-08-03",
    "الإجراءات الجزائية":      "25-14_2025-08-03",
    "25-14":                   "25-14_2025-08-03",
    "25-14_2025-08-03":        "25-14_2025-08-03",

    # ── Penal Code ───────────────────────────────────────────────────────────
    "penal code":              "66-156_1966-06-08",
    "penal_code":              "66-156_1966-06-08",
    "criminal code":           "66-156_1966-06-08",
    "cpen":                    "66-156_1966-06-08",
    "قانون العقوبات":          "66-156_1966-06-08",
    "66-156":                  "66-156_1966-06-08",
    "66-156_1966-06-08":       "66-156_1966-06-08",   # FRBR id used in benchmark
    "6-5_1966-06-08":          "66-156_1966-06-08",   # filename-based
    "6-5":                     "66-156_1966-06-08",

    # ── Old Criminal Procedure ────────────────────────────────────────────────
    "66-155":                  "66-155_1966-06-08",
    "66-155_1966-06-08":       "66-155_1966-06-08",
    "قانون الإجراءات الجزائية القديم": "66-155_1966-06-08",

    # ── Investment Law ───────────────────────────────────────────────────────
    "investment law":          "22-18_2022-07-24",
    "investment_law":          "22-18_2022-07-24",
    "cinv":                    "22-18_2022-07-24",
    "قانون الاستثمار":         "22-18_2022-07-24",
    "22-18":                   "22-18_2022-07-24",
    "22-18_2022-07-24":        "22-18_2022-07-24",

    # ── Anti-Money Laundering 2005 ───────────────────────────────────────────
    "anti-money laundering":   "05-01_2005-02-06",
    "anti_money_laundering":   "05-01_2005-02-06",
    "aml":                     "05-01_2005-02-06",
    "تبييض الأموال":           "05-01_2005-02-06",
    "05-01":                   "05-01_2005-02-06",
    "05-01_2005-02-06":        "05-01_2005-02-06",

    # ── Anti-Corruption ───────────────────────────────────────────────────────
    "anti-corruption":         "06-01_2006-02-20",
    "anti_corruption":         "06-01_2006-02-20",
    "acor":                    "06-01_2006-02-20",
    "مكافحة الفساد":           "06-01_2006-02-20",
    "الوقاية من الفساد":       "06-01_2006-02-20",
    "06-01_2006-02-20":        "06-01_2006-02-20",
    "06-01":                   "06-01_2006-02-20",

    # ── Constitution 2020 ────────────────────────────────────────────────────
    "constitution":              "2020_2020-12-30",
    "const":                     "2020_2020-12-30",
    "الدستور":                   "2020_2020-12-30",
    "2020_2020-12-30":           "2020_2020-12-30",
    "دستور 2020":                "2020_2020-12-30",
    # Benchmark and canonical registry use "constitution_2020-12-30" as doc_id
    "constitution_2020-12-30":   "2020_2020-12-30",
    "constitution_2020":         "2020_2020-12-30",

    # ── Labour Law ───────────────────────────────────────────────────────────
    "labour law":              "90-11_1990-04-21",
    "labor law":               "90-11_1990-04-21",
    "labour_law":              "90-11_1990-04-21",
    "labor_law":               "90-11_1990-04-21",
    "قانون العمل":             "90-11_1990-04-21",
    "90-11":                   "90-11_1990-04-21",
    "90-11_1990-04-21":        "90-11_1990-04-21",
    "labor_law_90-11":         "90-11_1990-04-21",   # filename alias

    # ── Health Law ───────────────────────────────────────────────────────────
    "health law":              "18-11_2018-07-02",
    "health_law":              "18-11_2018-07-02",
    "قانون الصحة":             "18-11_2018-07-02",
    "18-11":                   "18-11_2018-07-02",
    "18-11_2018-07-02":        "18-11_2018-07-02",

    # ── Consumer Protection ──────────────────────────────────────────────────
    "consumer protection":     "09-03_2009-02-25",
    "consumer_protection":     "09-03_2009-02-25",
    "consumer law":            "09-03_2009-02-25",
    "consumer_law":            "09-03_2009-02-25",
    "09-03":                   "09-03_2009-02-25",
    "09-03_2009-02-25":        "09-03_2009-02-25",

    # ── E-Commerce ───────────────────────────────────────────────────────────
    "e-commerce":              "18-05_2018-05-10",
    "e_commerce":              "18-05_2018-05-10",
    "ecommerce":               "18-05_2018-05-10",
    "18-05":                   "18-05_2018-05-10",
    "18-05_2018-05-10":        "18-05_2018-05-10",

    # ── Nationality / Citizenship ─────────────────────────────────────────────
    "nationality code":        "70-86_1970-12-15",
    "nationality_code":        "70-86_1970-12-15",
    "citizenship law":         "70-86_1970-12-15",
    "قانون الجنسية":           "70-86_1970-12-15",
    "70-86":                   "70-86_1970-12-15",
    "70-86_1970-12-15":        "70-86_1970-12-15",

    # ── Military Justice ─────────────────────────────────────────────────────
    "military justice":        "71-28_1971-04-22",
    "military_justice":        "71-28_1971-04-22",
    "القضاء العسكري":          "71-28_1971-04-22",
    "71-28":                   "71-28_1971-04-22",
    "13-12_1971-04-22":        "71-28_1971-04-22",   # filename alias
    "13-12":                   "71-28_1971-04-22",
    "71-28_1971-04-22":        "71-28_1971-04-22",

    # ── Prison Organisation ──────────────────────────────────────────────────
    "prison law":              "05-04_2005-02-06",
    "prison_law":              "05-04_2005-02-06",
    "prison organisation":     "05-04_2005-02-06",
    "تنظيم السجون":            "05-04_2005-02-06",
    "05-04":                   "05-04_2005-02-06",
    "05-2004_2005-02-06":      "05-04_2005-02-06",   # filename alias
    "05-04_2005-02-06":        "05-04_2005-02-06",

    # ── Public Procurement ────────────────────────────────────────────────────
    "public procurement":      "15-247_2015-09-16",
    "public_procurement":      "15-247_2015-09-16",
    "الصفقات العمومية":        "15-247_2015-09-16",
    "15-247":                  "15-247_2015-09-16",
    "15-247_2015-09-16":       "15-247_2015-09-16",

    # ── Municipality Law ─────────────────────────────────────────────────────
    "municipality law":        "11-10_2011-06-22",
    "municipality_law":        "11-10_2011-06-22",
    "local administration":    "11-10_2011-06-22",
    "قانون البلدية":           "11-10_2011-06-22",
    "11-10":                   "11-10_2011-06-22",
    "11-10_2011-06-22":        "11-10_2011-06-22",

    # ── Narcotics Law ────────────────────────────────────────────────────────
    "narcotics law":           "04-18_2004-12-25",
    "narcotics_law":           "04-18_2004-12-25",
    "المخدرات":                "04-18_2004-12-25",
    "04-18":                   "04-18_2004-12-25",
    "04-18_2004-12-25":        "04-18_2004-12-25",

    # ── Intellectual Property / Patent ────────────────────────────────────────
    "patent law":              "03-07_2003-07-19",
    "patent_law":              "03-07_2003-07-19",
    "ip law":                  "03-05_2003-07-19",
    "ip_law":                  "03-05_2003-07-19",
    "copyright law":           "03-05_2003-07-19",
    "copyright_law":           "03-05_2003-07-19",
    "حقوق المؤلف":             "03-05_2003-07-19",
    "03-05":                   "03-05_2003-07-19",
    "03-05_2003-07-19":        "03-05_2003-07-19",
    "براءات الاختراع":         "03-07_2003-07-19",
    "03-07":                   "03-07_2003-07-19",
    "03-07_2003-07-19":        "03-07_2003-07-19",
    "ordonnance-03-07-ar":     "03-07_2003-07-19",   # filename alias

    # ── Organic Law on Judicial Organisation ─────────────────────────────────
    "judicial organisation":   "05-11_2005-07-17",
    "05-11":                   "05-11_2005-07-17",
    "05-11_2005-07-17":        "05-11_2005-07-17",
    "11-05_2005-07-17":        "05-11_2005-07-17",   # filename alias
    "2005_2005-07-17":         "05-11_2005-07-17",   # legacy alias from doc_id_aliases.py

    # ── AML Regulation 2012 ──────────────────────────────────────────────────
    "03-12":                   "03-12_2012-11-28",
    "03-12_2012-11-28":        "03-12_2012-11-28",
    "12-2003_2012-11-28":      "03-12_2012-11-28",   # filename alias

    # ── Organic Law on Information ────────────────────────────────────────────
    "12-05":                   "12-05_2012-01-12",
    "12-05_2012-01-12":        "12-05_2012-01-12",
    "12-2015_2012-01-12":      "12-05_2012-01-12",   # filename alias

    # ── Wilaya Law ────────────────────────────────────────────────────────────
    "07-12_2012-02-21":        "07-12_2012-02-21",

    # ── Road Traffic ─────────────────────────────────────────────────────────
    "traffic law":             "01-14_2001-08-19",
    "traffic_law":             "01-14_2001-08-19",
    "01-14_2001-08-19":        "01-14_2001-08-19",
    "01-14":                   "01-14_2001-08-19",

    # ── Road Traffic Amendment ────────────────────────────────────────────────
    "05-17_2017-02-16":        "05-17_2017-02-16",

    # ── AML Executive Decree 2006 ─────────────────────────────────────────────
    "06-05_2006-01-09":        "06-05_2006-01-09",
    "06-05":                   "06-05_2006-01-09",
    "06-154":                  "06-154_2006-05-11",
    "06-154_2006-05-11":       "06-154_2006-05-11",
    "06-15":                   "06-154_2006-05-11",
    "06-15_2006-05-11":        "06-154_2006-05-11",

    # ── E-Commerce / Specific Laws ────────────────────────────────────────────
    "22-21_2022-07-20":        "22-21_2022-07-20",
    "15-20_2015-12-30":        "15-20_2015-12-30",
    "22-09_2022-05-05":        "22-09_2022-05-05",

    # ── Constitutional amendments ─────────────────────────────────────────────
    "02-03_2002-04-10":        "02-03_2002-04-10",
    "08-19_2008-11-15":        "08-19_2008-11-15",
    "16-01_2016-03-06":        "16-01_2016-03-06",
    "1963_1963-12-08":         "1963_1963-12-08",
    "1976_1976-11-19":         "1976_1976-11-19",
    "1988_1988-11-03":         "1988_1988-11-03",
    "constitution_1988":       "1988_1988-11-03",
    "constitution_1988-11-03": "1988_1988-11-03",
    "1989_1989-02-23":         "1989_1989-02-23",
    "constitution_1989-02-23": "1989_1989-02-23",
    "1996_1996-11-28":         "1996_1996-11-28",
    "79-06_1979-07-01":        "79-06_1979-01-01",
    "79-06_1979-01-01":        "79-06_1979-01-01",
    "80-01_1980-01-12":        "80-01_1980-01-01",
    "80-01_1980-01-01":        "80-01_1980-01-01",
    "94-03_1994-04-11":        "94-03_1994-04-11",
    "96-21_1996-07-09":        "96-21_1996-07-09",
    "97-02_1997-01-11":        "97-02_1997-01-11",

    # ── Benchmark category shortcodes ─────────────────────────────────────────
    "civil_law":               "75-58_1975-09-26",
    "family_law":              "84-11_1984-06-09",
    "commercial_law":          "75-59_1975-09-26",
    "civil_procedure":         "08-09_2008-02-25",
    "criminal_procedure":      "25-14_2025-08-03",
    "criminal_law":            "66-156_1966-06-08",
    "constitutional_law":      "2020_2020-12-30",
    "labor_law":               "90-11_1990-04-21",
    "consumer_law":            "09-03_2009-02-25",
    "e_commerce":              "18-05_2018-05-10",
    "anti_corruption_law":     "06-01_2006-02-20",
    "environmental_law":       "03-10_2003-07-19",
    "housing_law":             "11-04_2011-02-17",
    "ip_benchmark_law":        "03-05_2003-07-19",

    # ── Social Security law (now in corpus as of new dataset) ────────────────
    "social security":         "83-11_1983-07-02",
    "social_security":         "83-11_1983-07-02",
    "التأمينات الاجتماعية":    "83-11_1983-07-02",
    "83-11":                   "83-11_1983-07-02",
    "83-11_1983-07-02":        "83-11_1983-07-02",

    # ── Newly restored benchmark laws ─────────────────────────────────────────
    "11-04":                   "11-04_2011-02-17",
    "11-04_2011-02-17":        "11-04_2011-02-17",
    "housing law":             "11-04_2011-02-17",
    "03-10_2003-07-19":        "03-10_2003-07-19",
    "03-10":                   "03-10_2003-07-19",
    "environment law":         "03-10_2003-07-19",
    "03-05_2003-07-19":        "03-05_2003-07-19",
}


# ---------------------------------------------------------------------------
# DocEntry
# ---------------------------------------------------------------------------

@dataclass
class DocEntry:
    canonical_id:  str
    doc_title:     str
    doc_date:      str
    doc_type:      str
    filename_stem: str
    article_eids:  Set[str]             = field(default_factory=set)
    article_refs:  Dict[str, str]       = field(default_factory=dict)  # ref → eid


# ---------------------------------------------------------------------------
# ArticleRegistry
# ---------------------------------------------------------------------------

class ArticleRegistry:
    """
    Registry of all canonical documents and their articles.

    Built from the list of Article objects returned by parse_all().
    """

    def __init__(self) -> None:
        self._docs:    Dict[str, DocEntry]          = {}    # canonical_id → DocEntry
        self._aliases: Dict[str, Optional[str]]     = {}    # lower(alias) → canonical_id
        self._collision_filenames: Set[str]         = set()
        self._built = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, articles: list) -> None:
        """Populate registry from a list of Article objects."""
        from akn_rlm.corpus.akn_parser import Article as _Article

        # Track which filename produced which canonical id
        filename_to_canonical: Dict[str, str] = {}
        # Track which filenames have already had their one-time warning emitted
        _warned_collision: Set[str] = set()
        _warned_mismatch:  Set[str] = set()

        for art in articles:
            canonical_id  = art.doc_id
            filename_stem = art.filename_stem

            # ── Collision detection (log ONCE per filename) ──────────────────
            if filename_stem in FILENAME_COLLISIONS and filename_stem not in _warned_collision:
                implied_id, frbr_id = FILENAME_COLLISIONS[filename_stem]
                if canonical_id == frbr_id:
                    logger.info(
                        "Filename collision (known): %s.xml implies %s but FRBR says %s — "
                        "registered under %s; benchmark questions citing %s must abstain.",
                        filename_stem, implied_id, frbr_id, frbr_id, implied_id,
                    )
                    self._collision_filenames.add(filename_stem)
                    _warned_collision.add(filename_stem)
                    # Still add articles under the real canonical_id (frbr_id)

            # ── Type / metadata mismatch logging (log ONCE per filename) ─────
            if filename_stem in METADATA_MISMATCHES and filename_stem not in _warned_mismatch:
                logger.info(
                    "Metadata mismatch (known): %s.xml — %s",
                    filename_stem, METADATA_MISMATCHES[filename_stem],
                )
                _warned_mismatch.add(filename_stem)

            # ── Register doc entry ──────────────────────────────────────────
            if canonical_id not in self._docs:
                self._docs[canonical_id] = DocEntry(
                    canonical_id  = canonical_id,
                    doc_title     = art.doc_title,
                    doc_date      = art.doc_date,
                    doc_type      = art.doc_type,
                    filename_stem = filename_stem,
                )
                filename_to_canonical[filename_stem] = canonical_id

            entry = self._docs[canonical_id]

            # ── Register article ─────────────────────────────────────────────
            if art.eid:
                entry.article_eids.add(art.eid.lower())
            # Canonical form is the primary key.  Also keep the raw normalized
            # Arabic form as a fallback so legacy lookups still resolve.
            canon_ref = canonical_article_ref(art.article_ref)
            if canon_ref and art.eid:
                entry.article_refs[canon_ref] = art.eid.lower()
            normalized_ref = normalize_arabic(art.article_ref)
            if normalized_ref and art.eid:
                entry.article_refs[normalized_ref] = art.eid.lower()
                # Also store bare numeric variant
                bare = re.sub(r"[^\d]", "", normalized_ref)
                if bare:
                    entry.article_refs[bare] = art.eid.lower()

        # ── Load static aliases ──────────────────────────────────────────────
        for alias, doc_id in _STATIC_ALIASES.items():
            self._aliases[alias.lower().strip()] = doc_id

        # ── Auto-add canonical ids as aliases to themselves ──────────────────
        for canonical_id in self._docs:
            self._aliases[canonical_id.lower()] = canonical_id

        # ── Log missing-from-corpus at INFO (known gaps, not runtime errors) ──
        for missing_id in MISSING_FROM_CORPUS:
            logger.info("Missing from corpus (known): %s — benchmark questions citing this will abstain.", missing_id)

        self._built = True
        logger.info(
            "ArticleRegistry built: %d canonical documents, %d alias keys",
            len(self._docs), len(self._aliases),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_alias(self, alias: str) -> Optional[str]:
        """Return canonical doc_id for any alias, or None if not in corpus.

        Returns None for:
          - any entry in MISSING_FROM_CORPUS
          - truly unknown aliases
        """
        if not alias:
            return None

        key = alias.lower().strip()

        # Direct static alias lookup
        if key in self._aliases:
            return self._aliases[key]

        # Try normalizing Arabic and looking up again
        normalized = normalize_arabic(alias).lower().strip()
        if normalized in self._aliases:
            return self._aliases[normalized]

        # Try looking up as a canonical id directly
        if alias in self._docs:
            return alias

        return None

    def get_doc(self, doc_id: str) -> Optional[DocEntry]:
        """Return DocEntry for a canonical doc_id (or alias)."""
        canonical = self.resolve_alias(doc_id) or doc_id
        return self._docs.get(canonical)

    def has_article(self, doc_id: str, article_ref: str) -> bool:
        """Return True iff the article exists in the corpus.

        Accepts article_ref in any of these forms:
          - bare number: '4', '٤'
          - with prefix: 'المادة 4', 'Art. 4'
          - eId form: 'art_4', 'art_4_bis'
          - bis variant: '4 مكرر', '4 bis'
          - Arabic ordinal: 'الأولى', 'الاولي'
          - amended suffix: '33 (معدلة)'

        Never returns True for a fabricated article number.
        """
        # Resolve doc_id through alias map
        canonical = self.resolve_alias(doc_id) or doc_id
        entry = self._docs.get(canonical)
        if entry is None:
            return False

        # 1. Canonical form (the primary lookup path now)
        canon = canonical_article_ref(article_ref)
        if canon:
            if canon in entry.article_refs:
                return True
            # Many laws use eid form `art_<canon>` directly
            eid_canon = f"art_{canon}".lower()
            if eid_canon in entry.article_eids:
                return True

        # 2. eId-form pass-through (e.g. 'art_4')
        eid = ref_to_eid(article_ref).lower()
        if eid in entry.article_eids:
            return True

        # 3. Raw Arabic-normalized form (legacy fallback)
        norm_ref = normalize_arabic(article_ref).lower().strip()
        if norm_ref in entry.article_refs:
            return True

        return False

    def is_missing(self, doc_id: str) -> bool:
        """Return True if doc_id is in MISSING_FROM_CORPUS."""
        key = doc_id.lower().strip()
        for missing in MISSING_FROM_CORPUS:
            if key == missing.lower():
                return True
        # Anti-corruption: alias returns None even though doc_id looks valid
        alias_result = self.resolve_alias(doc_id)
        if alias_result is None and doc_id not in self._docs:
            # Could be genuinely unknown OR missing
            norm = normalize_arabic(doc_id).lower().strip()
            for missing in MISSING_FROM_CORPUS:
                if norm == normalize_arabic(missing).lower().strip():
                    return True
        return False

    def is_collision_filename(self, filename_stem: str) -> bool:
        """Return True if this filename is a known content collision."""
        return filename_stem in self._collision_filenames

    @property
    def canonical_ids(self) -> Set[str]:
        return set(self._docs.keys())

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def article_count(self) -> int:
        return sum(len(e.article_eids) for e in self._docs.values())

    def articles_for_doc(self, doc_id: str) -> Set[str]:
        """Return set of eIds for a canonical doc_id."""
        entry = self.get_doc(doc_id)
        return entry.article_eids if entry else set()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Serialise registry to a JSON-compatible dict (for indices/article_registry.json)."""
        out: dict = {}
        for cid, entry in self._docs.items():
            out[cid] = {
                "title":        entry.doc_title,
                "date":         entry.doc_date,
                "type":         entry.doc_type,
                "filename":     entry.filename_stem,
                "article_eids": sorted(entry.article_eids),
                "article_refs": entry.article_refs,
            }
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            import json
            json.dump(self.to_json(), fh, ensure_ascii=False, indent=2)
        logger.info("ArticleRegistry saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ArticleRegistry":
        """Load a pre-built registry from JSON (without re-parsing XML)."""
        with open(path, encoding="utf-8") as fh:
            import json
            data = json.load(fh)
        reg = cls()
        for cid, info in data.items():
            entry = DocEntry(
                canonical_id  = cid,
                doc_title     = info.get("title", ""),
                doc_date      = info.get("date", ""),
                doc_type      = info.get("type", ""),
                filename_stem = info.get("filename", ""),
                article_eids  = set(info.get("article_eids", [])),
                article_refs  = info.get("article_refs", {}),
            )
            reg._docs[cid] = entry
        # Re-load static aliases
        for alias, doc_id in _STATIC_ALIASES.items():
            reg._aliases[alias.lower().strip()] = doc_id
        for cid in reg._docs:
            reg._aliases[cid.lower()] = cid
        reg._built = True
        return reg


# ---------------------------------------------------------------------------
# Module-level singleton builder
# ---------------------------------------------------------------------------

_REGISTRY: Optional[ArticleRegistry] = None


def get_registry(
    articles: Optional[list] = None,
    json_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> ArticleRegistry:
    """Return the module-level registry, building it lazily if needed."""
    global _REGISTRY
    if _REGISTRY is not None and not force_rebuild:
        return _REGISTRY

    if json_path is not None and json_path.exists() and not force_rebuild:
        _REGISTRY = ArticleRegistry.load(json_path)
        return _REGISTRY

    if articles is None:
        from akn_rlm.corpus.akn_parser import parse_all
        articles = parse_all()

    _REGISTRY = ArticleRegistry()
    _REGISTRY.build(articles)
    return _REGISTRY
