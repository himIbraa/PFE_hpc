"""
Arabic and French text normalizers.
All Arabic text MUST pass through normalize_arabic() before any retrieval or comparison.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Arabic normalizer
# ---------------------------------------------------------------------------

_ALEF_RE       = re.compile(r"[أإآٱ]")   # أإآٱ
_YA_RE         = re.compile(r"ى")                         # ى
_DIACRITICS_RE = re.compile(r"[ً-ٰٟـ]")   # tashkeel + tatweel
_SPACE_RE      = re.compile(r"\s+")

# Arabic-Indic digits -> ASCII digits
_AR_INDIC_TABLE = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789",
)

# Persian digits (used occasionally in Algerian legal texts)
_FA_DIGITS_TABLE = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹",
    "0123456789",
)


def normalize_arabic(text: str, *, ta_marbuta: bool = False) -> str:
    """Normalize Arabic orthography for indexing and comparison.

    Applies (in order):
      1. NFC Unicode normalization
      2. Alef variants (U+0623 U+0625 U+0622 U+0671) -> U+0627
      3. Ya variant (U+0649) -> U+064A
      4. Strip tashkeel and tatweel
      5. Arabic-Indic digits -> ASCII digits
      6. Persian digits -> ASCII digits
      7. Optionally: ta marbuta (U+0629) -> ha (U+0647)  [ta_marbuta=True]
      8. Collapse whitespace
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _ALEF_RE.sub("ا", text)   # -> ا
    text = _YA_RE.sub("ي", text)     # -> ي
    # Digit translation BEFORE diacritics: Arabic-Indic U+0660-U+0669 fall
    # inside the diacritics range U+064B-U+0670 and would be stripped otherwise.
    text = text.translate(_AR_INDIC_TABLE)
    text = text.translate(_FA_DIGITS_TABLE)
    text = _DIACRITICS_RE.sub("", text)
    if ta_marbuta:
        text = text.replace("ة", "ه")  # ة -> ه
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# French normalizer (for French-minority texts)
# ---------------------------------------------------------------------------

_FR_SPACE_RE = re.compile(r"\s+")


def normalize_french(text: str) -> str:
    """Normalize French text: NFD -> ASCII fold, lowercase, collapse spaces."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = _FR_SPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Article reference normalizer
# ---------------------------------------------------------------------------
#
# An article reference may appear in many forms across XML, benchmark gold,
# and LLM citations:
#
#   numeric:        "1", "54", "123"
#   prefixed:       "المادة 4", "مادة 4", "Art. 4", "article 4", "art 4"
#   eid-form:       "art_1", "art_9_bis", "art_9_bis_1"
#   arabic ordinal: "الأولى" / "الاولى" / "الاولي" (hamza/ya variants)
#                   "الثانية", "الثالثة" ... "العاشرة"
#   bis (Arabic):   "9 مكرر", "9 مكرر 1", "9 مكرر(1)", "53 مكررة"
#   bis (Latin):    "9_bis", "9 bis", "9_bis_1", "13_bis"
#   modified:       "33 (معدلة)", "33 ( معدلة)"   <- means "amended"; same article
#
# canonical_article_ref() collapses ALL of these to a single deterministic
# token so both index-write and gold-conversion sides agree on equality.
#
# Canonical form is eid-style WITHOUT the "art_" prefix:
#   "1", "9_bis", "9_bis_1", "33"   (lowercase, underscores, ASCII)
#
# ref_to_eid() simply prepends "art_" to the canonical form.
# ---------------------------------------------------------------------------

_ARTICLE_PREFIX_RE = re.compile(
    r"^(?:المادة|مادة|Art\.|article|art)\s*",
    re.IGNORECASE | re.UNICODE,
)

# Arabic ordinals — keys are post-normalize_arabic forms (alef variants → ا,
# ya → ي, no diacritics).  Both masculine and feminine forms are included.
# We also list the originals (with hamza) so the lookup works whether or not
# normalize_arabic has been applied yet.
_ARABIC_ORDINALS: dict[str, str] = {
    # Originals (with hamza/ya variants) — kept so callers that pass the raw
    # form still resolve, even though normalize_arabic would normally fold them.
    "الأولى": "1", "الأول": "1",
    "الاولى": "1", "الاولي": "1", "الاول": "1",
    "أولى": "1", "اولي": "1", "اولى": "1",
    "الثانية": "2", "الثاني": "2",
    "الثالثة": "3", "الثالث": "3",
    "الرابعة": "4", "الرابع": "4",
    "الخامسة": "5", "الخامس": "5",
    "السادسة": "6", "السادس": "6",
    "السابعة": "7", "السابع": "7",
    "الثامنة": "8", "الثامن": "8",
    "التاسعة": "9", "التاسع": "9",
    "العاشرة": "10", "العاشر": "10",
}

# "(معدلة)" or "( معدلة)" suffix means "amended" — drop it; same article.
_AMENDED_SUFFIX_RE = re.compile(r"\s*\(\s*(?:معدلة|معدله|مُعدَّلة|modified|amended)\s*\)\s*$", re.IGNORECASE)

# Canonical bis patterns — collect mukrrar/bis/ter into "_bis" or "_bis_N".
# Order matters: longest patterns first.
_MUKRRAR_NUM_RE      = re.compile(r"\s*مكرر(?:ة)?\s*[\(\s]*(\d+)\s*\)?", re.UNICODE)
_MUKRRAR_BARE_RE     = re.compile(r"\s*مكرر(?:ة)?\b", re.UNICODE)
_LATIN_BIS_NUM_RE    = re.compile(r"[\s_]*bis[\s_]*(\d+)", re.IGNORECASE)
_LATIN_BIS_BARE_RE   = re.compile(r"[\s_]*bis\b", re.IGNORECASE)


def canonical_article_ref(ref: str) -> str:
    """Collapse any article-ref form to a deterministic ASCII canonical token.

    Returns lowercase, underscore-separated form WITHOUT the 'art_' prefix.
    Plain numerals stay numeric; bis variants become '{N}_bis' or '{N}_bis_{K}'.
    Returns '' for empty input.

    Examples:
        '1'                -> '1'
        'الأولى'            -> '1'
        'الاولي'            -> '1'   (post-normalize_arabic form)
        'art_1'            -> '1'
        'Art. 4'           -> '4'
        '9 مكرر'            -> '9_bis'
        '9 مكرر 1'          -> '9_bis_1'
        '9 مكرر(1)'         -> '9_bis_1'
        '53 مكررة'          -> '53_bis'
        '9_bis'            -> '9_bis'
        '9 bis'            -> '9_bis'
        '13_bis'           -> '13_bis'
        '33 (معدلة)'        -> '33'
        '3 مكرر(1)'         -> '3_bis_1'
        ''                 -> ''
        None               -> ''
    """
    if not ref:
        return ""
    s = str(ref).strip()
    if not s:
        return ""

    # Strip "art_" prefix (case-insensitive) ────────────────────────────────
    if s.lower().startswith("art_"):
        s = s[4:]

    # Apply Arabic normalization (alef/ya/diacritic folding, indic digits → ASCII)
    s = normalize_arabic(s)

    # Strip article-prefix words ("المادة", "مادة", "Art.", "article", "art")
    s = _ARTICLE_PREFIX_RE.sub("", s).strip()

    # Drop "(معدلة)" / "( معدلة)" amended-suffix
    s = _AMENDED_SUFFIX_RE.sub("", s).strip()

    # Arabic ordinal lookup (after normalize_arabic so both forms resolve)
    if s in _ARABIC_ORDINALS:
        return _ARABIC_ORDINALS[s]

    # Convert mukrrar variants (with optional number, parenthesised or not)
    # First the numbered form: "9 مكرر 1", "9 مكرر(1)", "9 مكرر (1)"
    s = _MUKRRAR_NUM_RE.sub(r"_bis_\1", s)
    # Then the bare form: "9 مكرر", "9 مكررة"
    s = _MUKRRAR_BARE_RE.sub("_bis", s)

    # Convert Latin "bis" variants
    s = _LATIN_BIS_NUM_RE.sub(r"_bis_\1", s)
    s = _LATIN_BIS_BARE_RE.sub("_bis", s)

    # Collapse separators (spaces, multiple underscores) to single underscore
    s = re.sub(r"[\s_]+", "_", s).strip("_")

    return s.lower()


def normalize_article_ref(ref: str) -> str:
    """Backward-compatible alias of canonical_article_ref().

    Returns canonical form (no 'art_' prefix). Existing callers that compared
    bare numerics like '4' continue to work; callers that depended on Arabic
    forms now get the canonical form which is the desired behaviour.
    """
    return canonical_article_ref(ref)


# ---------------------------------------------------------------------------
# Language detection + unified query normalizer
# ---------------------------------------------------------------------------

_AR_RANGE_RE = re.compile(r'[؀-ۿ]')


def _detect_language(text: str) -> str:
    """Return 'ar' if >15 % of characters are Arabic-script, else 'fr'."""
    if not text:
        return "ar"
    arabic_chars = len(_AR_RANGE_RE.findall(text))
    return "ar" if arabic_chars / max(len(text), 1) > 0.15 else "fr"


def normalize_query(query: str) -> tuple[str, str]:
    """Normalize a query string for retrieval.

    Detects language, applies Arabic or French normalization, and returns
    ``(normalized_query, language)`` where language is ``"ar"`` or ``"fr"``.
    The normalized form is safe to pass directly to BM25 / vector searches.
    """
    lang = _detect_language(query)
    if lang == "ar":
        return normalize_arabic(query), lang
    return normalize_french(query), lang


def ref_to_eid(ref: str) -> str:
    """Convert a (possibly Arabic) article reference to AKN eId form.

    Built on top of canonical_article_ref(); simply prepends 'art_'.

    Examples:
        '4'             -> 'art_4'
        'Art. 4'        -> 'art_4'
        'الأولى'         -> 'art_1'
        'الاولي'         -> 'art_1'
        '4 مكرر'         -> 'art_4_bis'
        '4 مكرر 1'       -> 'art_4_bis_1'
        '4 مكرر(1)'      -> 'art_4_bis_1'
        '53 مكررة'       -> 'art_53_bis'
        '9_bis'         -> 'art_9_bis'
        'art_4_bis'     -> 'art_4_bis'
        '33 (معدلة)'    -> 'art_33'
    """
    canonical = canonical_article_ref(ref)
    if not canonical:
        return ""
    return f"art_{canonical}"
