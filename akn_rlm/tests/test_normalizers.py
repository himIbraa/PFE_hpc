"""Tests for akn_rlm.normalizers — Phase B."""
from __future__ import annotations

import pytest
from akn_rlm.normalizers import normalize_arabic, normalize_french, ref_to_eid


class TestArabicNormalizer:
    def test_alef_variants(self):
        # Each of the four Alef variants must collapse to plain Alef (U+0627)
        assert normalize_arabic("أإآٱ") == "اااا"

    def test_ya_variant(self):
        assert normalize_arabic("ى") == "ي"

    def test_diacritics_stripped(self):
        # shadda + fatha + kasra stripped
        assert normalize_arabic("كَتَبَ") == "كتب"
        assert normalize_arabic("مُحَمَّد") == "محمد"

    def test_tatweel_stripped(self):
        assert normalize_arabic("طـلـاق") == "طلاق"

    def test_nfc_normalization(self):
        # NFC should not change fully-composed Arabic text
        import unicodedata
        s = unicodedata.normalize("NFD", "الطلاق")
        # After NFD then normalize_arabic (which applies NFC) we should get canonical form
        assert normalize_arabic(s) == "الطلاق"

    def test_talaq_forms(self):
        # The spec requirement: diacritic variant == base form
        base    = normalize_arabic("الطلاق")
        with_shadda = normalize_arabic("الطّلاق")   # shadda on ط
        no_art  = normalize_arabic("طلاق")
        assert with_shadda == base,   "shadda form must equal base form"
        assert no_art == "طلاق",      "form without article normalizes to itself"

    def test_arabic_indic_digits(self):
        assert normalize_arabic("٠١٢٣٤٥٦٧٨٩") == "0123456789"

    def test_persian_digits(self):
        assert normalize_arabic("۰۱۲۳۴۵۶۷۸۹") == "0123456789"

    def test_ta_marbuta_off_by_default(self):
        assert normalize_arabic("مدرسة") == "مدرسة"

    def test_ta_marbuta_on(self):
        assert normalize_arabic("مدرسة", ta_marbuta=True) == "مدرسه"

    def test_whitespace_collapsed(self):
        assert normalize_arabic("  طلاق   ")  == "طلاق"
        assert normalize_arabic("كلمة  أخرى") == "كلمة اخري"

    def test_empty(self):
        assert normalize_arabic("") == ""
        assert normalize_arabic(None) == ""   # type: ignore


class TestFrenchNormalizer:
    def test_accents_stripped(self):
        assert normalize_french("éàüç") == "eauc"

    def test_lowercase(self):
        assert normalize_french("Code Civil") == "code civil"

    def test_whitespace(self):
        assert normalize_french("  loi   pénale  ") == "loi penale"


class TestRefToEid:
    def test_already_eid(self):
        assert ref_to_eid("art_4") == "art_4"
        assert ref_to_eid("ART_4_BIS") == "art_4_bis"
        assert ref_to_eid("art_4_bis_1") == "art_4_bis_1"

    def test_plain_number(self):
        assert ref_to_eid("4")  == "art_4"
        assert ref_to_eid("54") == "art_54"

    def test_arabic_ref(self):
        assert ref_to_eid("المادة 4") == "art_4"

    def test_bis(self):
        assert ref_to_eid("4 bis")   == "art_4_bis"
        assert ref_to_eid("4 bis 1") == "art_4_bis_1"

    def test_mukrrar(self):
        assert ref_to_eid("4 مكرر")    == "art_4_bis"
        assert ref_to_eid("4 مكرر 2")  == "art_4_bis_2"
