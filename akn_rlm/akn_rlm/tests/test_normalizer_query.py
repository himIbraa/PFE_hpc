"""Tests for normalize_query() and legal-ID preservation in BM25 tokenizer."""
import pytest
from akn_rlm.normalizers import normalize_query, normalize_arabic, _detect_language
from akn_rlm.indexers.bm25 import _tokenize


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def test_detect_arabic():
    assert _detect_language("ما هو القانون المدني؟") == "ar"


def test_detect_french():
    assert _detect_language("Quelle est la loi applicable?") == "fr"


def test_detect_mixed_arabic_dominant():
    # >15% Arabic chars → Arabic
    assert _detect_language("القانون 06-01 code") == "ar"


# ---------------------------------------------------------------------------
# normalize_query output and language tag
# ---------------------------------------------------------------------------

def test_normalize_query_arabic_lang_tag():
    text = "ما هو قانون الأسرة الجزائري؟"
    norm, lang = normalize_query(text)
    assert lang == "ar"
    assert isinstance(norm, str)
    assert len(norm) > 0


def test_normalize_query_french_lang_tag():
    text = "Quelle est la procédure civile?"
    norm, lang = normalize_query(text)
    assert lang == "fr"
    assert isinstance(norm, str)


def test_normalize_query_strips_diacritics():
    # تَاريخ  → تاريخ
    text = "مَتَى صَدَرَ القانون؟"
    norm, _ = normalize_query(text)
    # No tashkeel in output
    import re
    assert not re.search(r'[ً-ٟ]', norm)


def test_normalize_query_alef_unification():
    text = "أحكام الإرث في القانون"
    norm, _ = normalize_query(text)
    # All alef variants → ا (U+0627)
    assert "أ" not in norm
    assert "إ" not in norm


# ---------------------------------------------------------------------------
# BM25 tokenizer legal-ID preservation
# ---------------------------------------------------------------------------

def test_tokenize_preserves_law_number():
    tokens = _tokenize("يتضمن القانون 06-01 مكافحة الفساد")
    # 06-01 must appear as a single token (with _LEGAL_SEP instead of -)
    compound = [t for t in tokens if "06" in t and "01" in t]
    assert len(compound) == 1, f"Expected 06-01 as single token, got: {tokens}"


def test_tokenize_preserves_date():
    tokens = _tokenize("صدر بتاريخ 2006-02-20")
    compound = [t for t in tokens if "2006" in t]
    assert len(compound) == 1, f"Date split unexpectedly: {tokens}"


def test_tokenize_preserves_canonical_id():
    # 75-58_1975-09-26 should be a single token
    tokens = _tokenize("الأمر 75-58_1975-09-26 القانون المدني")
    compound = [t for t in tokens if "75" in t and "58" in t]
    assert len(compound) >= 1, f"Canonical ID split: {tokens}"


def test_tokenize_still_splits_words():
    tokens = _tokenize("القانون المدني الجزائري")
    assert len(tokens) >= 2


def test_tokenize_consistency_index_vs_query():
    """The same legal ID must produce the same token at index time and query time."""
    # Index-time token from a corpus chunk
    index_tokens = _tokenize("تطبيقاً للمادة 4 من القانون 84-11_1984-06-09")
    # Query-time token
    query_tokens = _tokenize("ما هو حكم المادة 4 من القانون 84-11_1984-06-09؟")
    # Both should contain the same compound token for 84-11_1984-06-09
    def _find_compound(tokens):
        return [t for t in tokens if "84" in t and "11" in t and "1984" in t]
    assert _find_compound(index_tokens), f"Not found in index tokens: {index_tokens}"
    assert _find_compound(query_tokens), f"Not found in query tokens: {query_tokens}"
    # They must be identical
    assert _find_compound(index_tokens) == _find_compound(query_tokens)
