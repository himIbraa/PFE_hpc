"""Tests for the span-existence gate.

The gate checks that the citation's supporting_span actually appears inside
the article text returned by legal_env.get_article().  Fabricated spans
(reason='span_not_in_article') must be rejected; legitimate spans, even
with minor reformatting, must pass.
"""
from akn_rlm.gates import span_existence


# ---------------------------------------------------------------------------
# Fake legal_env that returns canned article text
# ---------------------------------------------------------------------------

class _FakeEnv:
    def __init__(self, articles: dict[tuple[str, str], str]):
        self._articles = articles

    def get_article(self, doc_id, article_ref):
        text = self._articles.get((doc_id, article_ref))
        if text is None:
            return None
        return {"doc_id": doc_id, "article_ref": article_ref, "text": text}


# Real Family Code article 5 text (no diacritics, normalized form)
_ART_5_FAMILY = (
    "الخطبة وعد بالزواج. يجوز للطرفين العدول عن الخطبة. اذا ترتب عن العدول "
    "عن الخطبة ضرر مادي او معنوي لاحد الطرفين جاز الحكم له بالتعويض."
)


def _env_with_art_5():
    return _FakeEnv({("84-11_1984-06-09", "5"): _ART_5_FAMILY})


# ---------------------------------------------------------------------------
# Per-citation check
# ---------------------------------------------------------------------------

class TestSpanExistenceCheck:

    def test_legitimate_span_passes(self):
        env = _env_with_art_5()
        cite = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "5",
            "supporting_span": "الخطبة وعد بالزواج. يجوز للطرفين العدول عن الخطبة",
        }
        result = span_existence.check(env, cite)
        assert result.passed is True
        assert result.score == 1.0

    def test_fabricated_span_rejected(self):
        # The exact bug from smoke_05/fam_ra_q03: art 5 cited with polygamy text
        env = _env_with_art_5()
        fabricated = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "5",
            "supporting_span": "يسمح بالزواج باكثر من زوجة واحدة في حدود الشريعة الاسلامية",
        }
        result = span_existence.check(env, fabricated)
        assert result.passed is False
        assert result.details[0]["reason"] == "span_not_in_article"

    def test_diacritic_tolerance(self):
        # Span with extra diacritics should still match after normalization
        env = _env_with_art_5()
        cite = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "5",
            "supporting_span": "الخِطبَةُ وَعدٌ بالزواج",
        }
        result = span_existence.check(env, cite)
        assert result.passed is True

    def test_alef_hamza_tolerance(self):
        # LLM might emit hamza variants; normalizer folds them
        env = _env_with_art_5()
        cite = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "5",
            "supporting_span": "إذا ترتب عن العدول عن الخطبة ضرر مادي أو معنوي",
        }
        result = span_existence.check(env, cite)
        assert result.passed is True

    def test_span_too_short_rejected(self):
        env = _env_with_art_5()
        cite = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "5",
            "supporting_span": "الخطبة",   # < 12 normalized chars
        }
        result = span_existence.check(env, cite)
        assert result.passed is False
        assert result.details[0]["reason"] == "span_too_short"

    def test_no_span_passes(self):
        # No supporting_span supplied -> gate is no-op
        env = _env_with_art_5()
        cite = {"doc_id": "84-11_1984-06-09", "article_ref": "5"}
        result = span_existence.check(env, cite)
        assert result.passed is True

    def test_empty_span_passes(self):
        env = _env_with_art_5()
        cite = {"doc_id": "84-11_1984-06-09", "article_ref": "5", "supporting_span": "  "}
        result = span_existence.check(env, cite)
        assert result.passed is True

    def test_article_not_found_rejected(self):
        env = _FakeEnv({})  # nothing
        cite = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "999",
            "supporting_span": "any reasonably long phrase here",
        }
        result = span_existence.check(env, cite)
        assert result.passed is False
        assert result.details[0]["reason"] == "article_not_retrievable_for_span_check"


# ---------------------------------------------------------------------------
# Batch filter
# ---------------------------------------------------------------------------

class TestSpanFilter:

    def test_filter_separates_valid_and_rejected(self):
        env = _env_with_art_5()
        legitimate = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "5",
            "supporting_span": "الخطبة وعد بالزواج. يجوز للطرفين العدول",
        }
        fabricated = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "5",
            "supporting_span": "الزواج باكثر من زوجة في حدود الشريعة",
        }
        valid, rejected = span_existence.filter_citations(env, [legitimate, fabricated])
        assert len(valid) == 1
        assert len(rejected) == 1
        assert rejected[0]["_rejection_reason"] == "span_not_in_article"

    def test_run_gate_aggregates(self):
        env = _env_with_art_5()
        legit = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "5",
            "supporting_span": "الخطبة وعد بالزواج. يجوز للطرفين",
        }
        fab = {
            "doc_id": "84-11_1984-06-09",
            "article_ref": "5",
            "supporting_span": "الزواج باكثر من زوجة في حدود الشريعة الاسلامية",
        }
        result = span_existence.run_gate(env, [legit, fab])
        assert result.passed is False
        assert result.score == 0.5     # 1/2 failed

    def test_empty_citations_pass(self):
        env = _env_with_art_5()
        result = span_existence.run_gate(env, [])
        assert result.passed is True
        assert result.score == 1.0
