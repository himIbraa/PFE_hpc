"""Regression: sub-LM prompt templates must be safely format()-able.

Each template is loaded by ``call_decomposer`` / ``call_verifier`` /
``call_summarizer`` and rendered with ``str.format(**fields)``. JSON
examples inside the templates contain literal ``{`` and ``}`` braces
that MUST be escaped as ``{{`` and ``}}``; otherwise ``.format()``
raises ``KeyError`` on the JSON example before the LLM is even
called. This test fails the build if a future edit re-introduces an
unescaped brace inside any template.
"""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "rlm" / "prompts"


def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def test_sub_decomposer_template_renders_without_keyerror():
    tmpl = _read("sub_decomposer.txt")
    rendered = tmpl.format(question="ما هي شروط الزواج؟")
    # The literal { and } from the JSON example must survive.
    assert '{\n  "sub_questions"' in rendered or '{ "sub_questions"' in rendered
    assert "ما هي شروط الزواج؟" in rendered


def test_sub_verifier_template_renders_without_keyerror():
    tmpl = _read("sub_verifier.txt")
    rendered = tmpl.format(
        sub_question="هل يجب الإذن للقاضي؟",
        doc_id="84-11_1984-06-09",
        article_ref="5",
        article_text="نص المادة الخامسة",
    )
    assert '"relevant"' in rendered
    assert "84-11_1984-06-09" in rendered
    assert "نص المادة الخامسة" in rendered


def test_sub_summarizer_template_renders_without_keyerror():
    tmpl = _read("sub_summarizer.txt")
    rendered = tmpl.format(
        question="ما هي شروط الزواج؟",
        articles_block="[84-11 art.5]\nنص",
    )
    assert '"summary"' in rendered
    assert "ما هي شروط الزواج؟" in rendered
    assert "[84-11 art.5]" in rendered
