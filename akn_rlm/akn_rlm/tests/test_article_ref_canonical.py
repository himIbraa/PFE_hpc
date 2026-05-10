"""Comprehensive tests for canonical_article_ref() and ref_to_eid().

Covers every form observed in the AKN corpus and AlgerianLegalBench v3.0 gold:
  - Plain numerics (Western / Arabic-Indic)
  - Article-word prefixes (Arabic + Latin)
  - Arabic ordinals "الأولى" with hamza/ya variants
  - bis variants (Arabic مكرر / Latin bis) bare and numbered
  - Parenthesised number variants: "9 مكرر(1)", "9 مكرر (1)"
  - Feminine "مكررة"
  - Amended-suffix "( معدلة)"
  - eid-form pass-through "art_X"
  - Edge cases: empty, None-like, whitespace-only
"""
from akn_rlm.normalizers import canonical_article_ref, ref_to_eid


# ---------------------------------------------------------------------------
# canonical_article_ref
# ---------------------------------------------------------------------------

class TestCanonicalArticleRef:
    """Each test maps INPUT -> expected canonical form."""

    # Plain numerics
    def test_plain_numeric(self):
        assert canonical_article_ref("1") == "1"
        assert canonical_article_ref("54") == "54"
        assert canonical_article_ref("123") == "123"

    def test_arabic_indic_digits(self):
        assert canonical_article_ref("٤") == "4"
        assert canonical_article_ref("١٢٣") == "123"

    def test_persian_digits(self):
        assert canonical_article_ref("۴") == "4"

    def test_whitespace_padding(self):
        assert canonical_article_ref("  9  ") == "9"
        assert canonical_article_ref("\t54\n") == "54"

    # Article-word prefixes
    def test_arabic_prefix(self):
        assert canonical_article_ref("المادة 4") == "4"
        assert canonical_article_ref("مادة 4") == "4"
        assert canonical_article_ref("المادة 9 مكرر") == "9_bis"

    def test_latin_prefix(self):
        assert canonical_article_ref("Art. 4") == "4"
        assert canonical_article_ref("article 4") == "4"
        assert canonical_article_ref("art 4") == "4"
        assert canonical_article_ref("ART 9") == "9"

    # Arabic ordinals — both hamza-form and post-normalize_arabic form
    def test_ordinal_first_hamza(self):
        assert canonical_article_ref("الأولى") == "1"
        assert canonical_article_ref("الأول") == "1"

    def test_ordinal_first_normalized(self):
        # After normalize_arabic: alef-hamza → alef, ya-with-dots → ya
        assert canonical_article_ref("الاولي") == "1"
        assert canonical_article_ref("الاولى") == "1"
        assert canonical_article_ref("الاول") == "1"

    def test_ordinal_first_no_alef(self):
        assert canonical_article_ref("أولى") == "1"
        assert canonical_article_ref("اولي") == "1"

    def test_ordinal_second(self):
        assert canonical_article_ref("الثانية") == "2"
        assert canonical_article_ref("الثاني") == "2"

    def test_ordinal_third(self):
        assert canonical_article_ref("الثالثة") == "3"
        assert canonical_article_ref("الثالث") == "3"

    def test_ordinal_fourth_to_tenth(self):
        cases = [
            ("الرابعة", "4"), ("الرابع", "4"),
            ("الخامسة", "5"), ("الخامس", "5"),
            ("السادسة", "6"), ("السادس", "6"),
            ("السابعة", "7"), ("السابع", "7"),
            ("الثامنة", "8"), ("الثامن", "8"),
            ("التاسعة", "9"), ("التاسع", "9"),
            ("العاشرة", "10"), ("العاشر", "10"),
        ]
        for inp, exp in cases:
            assert canonical_article_ref(inp) == exp, f"{inp!r} → expected {exp!r}"

    # bis variants (Arabic مكرر)
    def test_mukrrar_bare(self):
        assert canonical_article_ref("9 مكرر") == "9_bis"
        assert canonical_article_ref("4 مكرر") == "4_bis"
        assert canonical_article_ref("123 مكرر") == "123_bis"

    def test_mukrrar_with_number(self):
        assert canonical_article_ref("9 مكرر 1") == "9_bis_1"
        assert canonical_article_ref("10 مكرر 4") == "10_bis_4"
        assert canonical_article_ref("16 مكرر 6") == "16_bis_6"

    def test_mukrrar_parenthesised(self):
        assert canonical_article_ref("3 مكرر(1)") == "3_bis_1"
        assert canonical_article_ref("3 مكرر (1)") == "3_bis_1"
        assert canonical_article_ref("3 مكرر ( 1 )") == "3_bis_1"

    def test_mukrrara_feminine(self):
        # "مكررة" (feminine) treated same as "مكرر"
        assert canonical_article_ref("53 مكررة") == "53_bis"

    # bis variants (Latin)
    def test_latin_bis_bare(self):
        assert canonical_article_ref("9_bis") == "9_bis"
        assert canonical_article_ref("9 bis") == "9_bis"
        assert canonical_article_ref("9bis") == "9_bis"
        assert canonical_article_ref("13_bis") == "13_bis"

    def test_latin_bis_with_number(self):
        assert canonical_article_ref("9_bis_1") == "9_bis_1"
        assert canonical_article_ref("9 bis 1") == "9_bis_1"

    def test_latin_bis_case_insensitive(self):
        assert canonical_article_ref("9 BIS") == "9_bis"
        assert canonical_article_ref("9_BIS_1") == "9_bis_1"

    # eid-form pass-through
    def test_eid_form_input(self):
        assert canonical_article_ref("art_4") == "4"
        assert canonical_article_ref("art_9_bis") == "9_bis"
        assert canonical_article_ref("art_9_bis_1") == "9_bis_1"
        assert canonical_article_ref("ART_4") == "4"

    # Amended-suffix
    def test_amended_suffix(self):
        assert canonical_article_ref("33 (معدلة)") == "33"
        assert canonical_article_ref("33 ( معدلة)") == "33"
        assert canonical_article_ref("33 ( معدلة )") == "33"
        assert canonical_article_ref("33 (modified)") == "33"
        assert canonical_article_ref("33 (amended)") == "33"

    # Edge cases
    def test_empty(self):
        assert canonical_article_ref("") == ""
        assert canonical_article_ref(None) == ""
        assert canonical_article_ref("   ") == ""

    def test_only_word_prefix(self):
        # Just "المادة" alone → empty
        assert canonical_article_ref("المادة") == ""

    def test_unknown_text_passes_through(self):
        # Non-numeric, non-bis, non-ordinal text stays as-is (lowercased/underscored)
        # This is the safety fall-through; downstream code should handle empty
        # or unmatched canonical forms.
        out = canonical_article_ref("xyz")
        assert out == "xyz"


# ---------------------------------------------------------------------------
# ref_to_eid (must produce 'art_<canonical>')
# ---------------------------------------------------------------------------

class TestRefToEid:

    def test_plain_numeric(self):
        assert ref_to_eid("4") == "art_4"
        assert ref_to_eid("123") == "art_123"

    def test_with_prefix(self):
        assert ref_to_eid("Art. 4") == "art_4"
        assert ref_to_eid("المادة 4") == "art_4"

    def test_arabic_ordinal_first(self):
        # The actual bug from the corpus — must now resolve to art_1
        assert ref_to_eid("الأولى") == "art_1"
        assert ref_to_eid("الاولي") == "art_1"
        assert ref_to_eid("الاولى") == "art_1"

    def test_mukrrar(self):
        assert ref_to_eid("4 مكرر") == "art_4_bis"
        assert ref_to_eid("4 مكرر 1") == "art_4_bis_1"
        assert ref_to_eid("9 مكرر(1)") == "art_9_bis_1"

    def test_eid_passthrough(self):
        assert ref_to_eid("art_4_bis") == "art_4_bis"
        assert ref_to_eid("art_9_bis_1") == "art_9_bis_1"

    def test_amended(self):
        assert ref_to_eid("33 (معدلة)") == "art_33"

    def test_empty(self):
        assert ref_to_eid("") == ""
        assert ref_to_eid("   ") == ""


# ---------------------------------------------------------------------------
# Idempotence — applying canonical_article_ref to its own output is a no-op
# ---------------------------------------------------------------------------

class TestIdempotence:

    def test_canonical_is_idempotent(self):
        cases = [
            "1", "54", "9_bis", "9_bis_1", "art_9_bis",
            "الأولى", "الاولي", "9 مكرر", "33 (معدلة)",
            "13_bis", "53 مكررة",
        ]
        for inp in cases:
            once = canonical_article_ref(inp)
            twice = canonical_article_ref(once)
            assert once == twice, f"{inp!r}: {once!r} != {twice!r}"


# ---------------------------------------------------------------------------
# Cross-form equivalence — different inputs that mean the same article must
# produce the same canonical form. This is what makes art_key matching work.
# ---------------------------------------------------------------------------

class TestCrossFormEquivalence:

    def test_article_one_all_forms_equal(self):
        forms = ["1", "art_1", "المادة 1", "الأولى", "الاولي",
                 "الاولى", "أولى", "اولي", "Art. 1"]
        canonicals = {canonical_article_ref(f) for f in forms}
        assert canonicals == {"1"}, f"Expected all forms to canonicalise to '1', got {canonicals}"

    def test_article_nine_bis_all_forms_equal(self):
        forms = ["9 مكرر", "9_bis", "9 bis", "art_9_bis", "9bis", "9 BIS"]
        canonicals = {canonical_article_ref(f) for f in forms}
        assert canonicals == {"9_bis"}, f"got {canonicals}"

    def test_article_nine_bis_one_all_forms_equal(self):
        forms = ["9 مكرر 1", "9 مكرر(1)", "9 مكرر (1)",
                 "9_bis_1", "9 bis 1", "art_9_bis_1"]
        canonicals = {canonical_article_ref(f) for f in forms}
        assert canonicals == {"9_bis_1"}, f"got {canonicals}"

    def test_real_corpus_smoke(self):
        # Real refs found in the BM25 index that previously failed
        problem_cases = [
            ("الاولي", "1"),
            ("9 مكرر", "9_bis"),
            ("9 مكرر 1", "9_bis_1"),
            ("10 مكرر 4", "10_bis_4"),
            ("33 ( معدلة)", "33"),
            ("3 مكرر(1)", "3_bis_1"),
            ("53 مكررة", "53_bis"),
            ("16 مكرر 6", "16_bis_6"),
        ]
        for inp, exp in problem_cases:
            assert canonical_article_ref(inp) == exp, \
                f"corpus form {inp!r} → expected {exp!r}, got {canonical_article_ref(inp)!r}"

    def test_real_benchmark_smoke(self):
        # Real refs found in benchmark gold that must match corpus refs
        benchmark_cases = [
            ("8 مكرر", "8_bis"),
            ("1 مكرر", "1_bis"),
            ("45 مكرر", "45_bis"),
            ("9 مكرر", "9_bis"),
            ("8 مكرر 1", "8_bis_1"),
            ("13_bis", "13_bis"),
            ("4_bis", "4_bis"),
            ("17 مكرر", "17_bis"),
            ("350 مكرر", "350_bis"),
        ]
        for inp, exp in benchmark_cases:
            assert canonical_article_ref(inp) == exp, \
                f"benchmark form {inp!r} → expected {exp!r}, got {canonical_article_ref(inp)!r}"
