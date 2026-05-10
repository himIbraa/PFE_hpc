"""
Test suite for ArticleRegistry — Phase A.

Tests:
  (a) Lookup by canonical FRBR id
  (b) Lookup by Arabic abbreviation
  (c) Restored document aliases resolve to live canonical ids
  (d) has_article() returns False for fabricated article numbers
  (e) Legacy filename aliases still resolve after corpus repairs

Requires the actual AKN XML files at the configured path.
Tests are skipped when the data files are not available.
"""
from __future__ import annotations

import logging
import pytest
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _data_available() -> bool:
    try:
        from akn_rlm.config import get_akn_dir
        akn_dir = get_akn_dir()
        return any(akn_dir.glob("*.xml"))
    except (FileNotFoundError, ImportError):
        return False


DATA_AVAILABLE = _data_available()
needs_data = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason="AKN XML files not available",
)


@pytest.fixture(scope="module")
def registry():
    """Build the ArticleRegistry once for the whole module."""
    from akn_rlm.corpus.akn_parser import parse_all
    from akn_rlm.corpus.article_registry import ArticleRegistry
    articles = parse_all()
    reg = ArticleRegistry()
    reg.build(articles)
    return reg


# ── (a) Lookup by canonical FRBR id ──────────────────────────────────────────

@needs_data
class TestLookupByCanonicalId:
    def test_civil_code_frbr_id(self, registry):
        entry = registry.get_doc("75-58_1975-09-26")
        assert entry is not None, "Civil Code should be in registry"
        assert entry.canonical_id == "75-58_1975-09-26"

    def test_family_code_frbr_id(self, registry):
        entry = registry.get_doc("84-11_1984-06-09")
        assert entry is not None

    def test_commercial_code_frbr_id(self, registry):
        entry = registry.get_doc("75-59_1975-09-26")
        assert entry is not None

    def test_criminal_procedure_frbr_id(self, registry):
        entry = registry.get_doc("25-14_2025-08-03")
        assert entry is not None

    def test_penal_code_frbr_id(self, registry):
        # Penal Code FRBR is 66-156 even though filename is 6-5
        entry = registry.get_doc("66-156_1966-06-08")
        assert entry is not None, "Penal Code should be indexed under FRBR id 66-156"

    def test_filename_alias_resolves_to_frbr_canonical(self, registry):
        # Benchmark uses 75-8_1975-09-26 (filename) — must resolve to FRBR canonical
        canonical = registry.resolve_alias("75-8_1975-09-26")
        assert canonical == "75-58_1975-09-26"

    def test_all_canonical_ids_present(self, registry):
        assert registry.doc_count >= 44, (
            f"Expected ≥44 canonical documents, got {registry.doc_count}"
        )

    def test_total_articles_in_range(self, registry):
        # article_count() returns unique eIds across the deduplicated registry.
        # It should stay close to the raw parsed article count, while allowing
        # a small reduction for known collisions / duplicate material.
        from akn_rlm.corpus.akn_parser import parse_all
        raw_article_count = len(parse_all())
        count = registry.article_count()
        assert 0 < count <= raw_article_count, (
            f"Registry article_count should be in (0, {raw_article_count}], got {count}"
        )
        assert count >= int(raw_article_count * 0.95), (
            f"Registry dropped too many articles: raw={raw_article_count}, registry={count}"
        )


# ── (b) Lookup by Arabic abbreviation ────────────────────────────────────────

@needs_data
class TestArabicAbbreviationLookup:
    def test_cciv_alias(self, registry):
        canonical = registry.resolve_alias("Cciv")
        assert canonical == "75-58_1975-09-26"

    def test_cfam_alias(self, registry):
        canonical = registry.resolve_alias("Cfam")
        assert canonical == "84-11_1984-06-09"

    def test_ccom_alias(self, registry):
        canonical = registry.resolve_alias("Ccom")
        assert canonical == "75-59_1975-09-26"

    def test_cpca_alias(self, registry):
        canonical = registry.resolve_alias("CPCA")
        assert canonical == "08-09_2008-02-25"

    def test_cpp_alias(self, registry):
        canonical = registry.resolve_alias("CPP")
        assert canonical == "25-14_2025-08-03"

    def test_cinv_alias(self, registry):
        canonical = registry.resolve_alias("Cinv")
        assert canonical == "22-18_2022-07-24"

    def test_const_alias(self, registry):
        canonical = registry.resolve_alias("Const")
        assert canonical == "2020_2020-12-30"

    def test_arabic_civil_code(self, registry):
        canonical = registry.resolve_alias("القانون المدني")
        assert canonical == "75-58_1975-09-26"

    def test_arabic_family_code(self, registry):
        canonical = registry.resolve_alias("قانون الأسرة")
        assert canonical == "84-11_1984-06-09"

    def test_arabic_aml(self, registry):
        canonical = registry.resolve_alias("تبييض الأموال")
        assert canonical == "05-01_2005-02-06"

    def test_case_insensitive(self, registry):
        assert registry.resolve_alias("civil code") == "75-58_1975-09-26"
        assert registry.resolve_alias("CIVIL CODE") == "75-58_1975-09-26"


# ── (c) Restored benchmark documents ──────────────────────────────────────────

@needs_data
class TestRestoredDocuments:
    def test_06_01_alias_resolves(self, registry):
        assert registry.resolve_alias("06-01_2006-02-20") == "06-01_2006-02-20"

    def test_anti_corruption_alias_resolves(self, registry):
        assert registry.resolve_alias("anti_corruption") == "06-01_2006-02-20"
        assert registry.resolve_alias("acor") == "06-01_2006-02-20"

    def test_06_01_registered_as_separate_doc(self, registry):
        assert "06-01_2006-02-20" in registry.canonical_ids

    def test_06_01_articles_present(self, registry):
        assert registry.has_article("06-01_2006-02-20", "25") is True
        assert registry.has_article("06-01_2006-02-20", "29") is True

    def test_06_15_legacy_alias_redirects_to_06_154(self, registry):
        assert registry.resolve_alias("06-15_2006-05-11") == "06-154_2006-05-11"

    def test_is_collision_filename(self, registry):
        assert not registry.is_collision_filename("06-01_2006-02-20")


# ── (d) has_article() returns False for fabricated article numbers ────────────

@needs_data
class TestHasArticle:
    def test_fabricated_article_returns_false(self, registry):
        # These article numbers do not exist in any document
        assert registry.has_article("75-58_1975-09-26", "art_999999") is False
        assert registry.has_article("84-11_1984-06-09", "art_88888") is False
        assert registry.has_article("25-14_2025-08-03", "art_77777_bis") is False

    def test_fabricated_bare_number_returns_false(self, registry):
        assert registry.has_article("75-58_1975-09-26", "999999") is False
        assert registry.has_article("84-11_1984-06-09", "99999") is False

    def test_real_article_returns_true(self, registry):
        # Article 1 exists in Civil Code
        assert registry.has_article("75-58_1975-09-26", "art_1") is True

    def test_arabic_ref_resolves(self, registry):
        # 'المادة 1' should resolve to art_1 in Civil Code
        assert registry.has_article("75-58_1975-09-26", "المادة 1") is True

    def test_wrong_doc_fabricated_returns_false(self, registry):
        # Article may exist but in the wrong document
        # Art 1 of Civil Code should not exist in Commercial Code under a
        # known-different doc_id
        # (we can't be 100% sure art_1 doesn't exist in Commercial Code,
        # so we test with a clearly nonexistent number instead)
        assert registry.has_article("08-09_2008-02-25", "art_999998") is False

    def test_missing_doc_returns_false(self, registry):
        assert registry.has_article("06-154_2006-05-11", "art_1") is True
        assert registry.has_article("99-99_9999-99-99", "art_1") is False

    def test_restored_doc_id_returns_true(self, registry):
        assert registry.has_article("06-01_2006-02-20", "art_25") is True


# ── (e) FRBR-vs-filename mismatch is non-fatal ───────────────────────────────

@needs_data
class TestMetadataMismatches:
    def test_15_247_loaded_with_repaired_type(self, registry):
        entry = registry.get_doc("15-247_2015-09-16")
        assert entry is not None
        assert entry.doc_type == "presidential-decree"

    def test_03_12_loaded_with_legacy_alias(self, registry):
        entry_frbr = registry.get_doc("03-12_2012-11-28")
        assert entry_frbr is not None
        assert registry.resolve_alias("12-2003_2012-11-28") == "03-12_2012-11-28"

    def test_ordonnance_03_07_loaded(self, registry):
        """ordonnance-03-07-ar.xml uses presidential-decree root but should load."""
        entry = registry.get_doc("03-07_2003-07-19")
        assert entry is not None

    def test_no_active_mismatch_warning_logged(self, caplog):
        from akn_rlm.corpus.akn_parser import parse_all
        from akn_rlm.corpus.article_registry import ArticleRegistry
        with caplog.at_level(logging.INFO, logger="akn_rlm.corpus.article_registry"):
            reg = ArticleRegistry()
            articles = parse_all()
            reg.build(articles)
        mismatch_msgs = [r.message for r in caplog.records if "MISMATCH" in r.message]
        assert mismatch_msgs == []

    def test_filename_based_ids_resolve_via_alias(self, registry):
        """Filename-based IDs that differ from FRBR must resolve through alias map."""
        # 12-2003_2012-11-28 (filename) → 03-12_2012-11-28 (FRBR canonical)
        assert registry.resolve_alias("12-2003_2012-11-28") == "03-12_2012-11-28"
        # 13-12_1971-04-22 (filename) → 71-28_1971-04-22 (FRBR canonical)
        assert registry.resolve_alias("13-12_1971-04-22") == "71-28_1971-04-22"
        # 11-05_2005-07-17 (filename) → 05-11_2005-07-17 (FRBR canonical)
        assert registry.resolve_alias("11-05_2005-07-17") == "05-11_2005-07-17"

    def test_restored_missing_docs_are_now_live(self, registry):
        assert registry.get_doc("03-05_2003-07-19") is not None
        assert registry.get_doc("03-10_2003-07-19") is not None
        assert registry.get_doc("11-04_2011-02-17") is not None
        assert registry.get_doc("66-155_1966-06-08") is not None
