"""Sub-LM workers: verifier, summarizer, decomposer.

Every worker uses parse_strict_json() with one automatic retry
on JSON parse failure.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def parse_strict_json(
    text: str,
    default: dict | None = None,
    retry_fn=None,
) -> dict:
    """Extract and parse the first JSON object from text.

    On failure, calls retry_fn(reminder) once (if provided) and tries again.
    Returns default (or {}) on total failure, with a logged warning.
    """
    def _try_parse(s: str) -> dict | None:
        start = s.find("{")
        end = s.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(s[start:end])
        except json.JSONDecodeError:
            return None

    result = _try_parse(text)
    if result is not None:
        return result

    if retry_fn is not None:
        reminder = (
            "Your previous output was not valid JSON. "
            "Return ONLY the JSON object — no prose, no markdown fences, no explanation."
        )
        try:
            text2 = retry_fn(reminder)
            result = _try_parse(text2)
            if result is not None:
                return result
        except Exception as exc:
            log.warning("Retry in parse_strict_json failed: %s", exc)

    log.warning("parse_strict_json: could not parse JSON from output, returning default")
    return default if default is not None else {}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def call_verifier(
    llm_pool,
    sub_question: str,
    article: dict,
    model: str,
) -> dict:
    """Return relevance judgment for (sub_question, article).

    Output schema:
      {"relevant": bool, "supporting_span": str|None,
       "contradicting_span": str|None, "confidence": float}
    """
    template = _load_prompt("sub_verifier.txt")
    prompt = template.format(
        sub_question=sub_question,
        doc_id=article.get("doc_id", ""),
        article_ref=article.get("article_ref", ""),
        article_text=article.get("text", ""),
    )
    default = {"relevant": False, "supporting_span": None,
               "contradicting_span": None, "confidence": 0.0}

    def _call(p: str) -> str:
        return llm_pool.call(p, model=model, max_tokens=512, temperature=0.0)

    raw = _call(prompt)
    return parse_strict_json(raw, default=default, retry_fn=lambda reminder: _call(reminder))


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

def call_summarizer(
    llm_pool,
    question: str,
    articles: list[dict],
    model: str,
) -> dict:
    """Synthesise relevant articles into a concise answer.

    Output schema:
      {"summary": str|None, "key_articles": list[str], "caveats": str|None}
    """
    template = _load_prompt("sub_summarizer.txt")
    articles_block = "\n\n".join(
        f"[{a.get('doc_id', '')} art.{a.get('article_ref', '')}]\n{a.get('text', '')}"
        for a in articles
    )
    prompt = template.format(question=question, articles_block=articles_block)
    default = {"summary": None, "key_articles": [], "caveats": "summarizer failed"}

    def _call(p: str) -> str:
        return llm_pool.call(p, model=model, max_tokens=768, temperature=0.0)

    raw = _call(prompt)
    return parse_strict_json(raw, default=default, retry_fn=lambda r: _call(r))


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

def call_decomposer(
    llm_pool,
    question: str,
    model: str,
) -> dict:
    """Break a complex question into atomic sub-questions.

    Output schema:
      {"sub_questions": list[dict], "dependency_order": list[str],
       "max_depth_needed": int}
    """
    template = _load_prompt("sub_decomposer.txt")
    prompt = template.format(question=question)
    default = {
        "sub_questions": [{"id": "sq1", "text": question, "type": "rule_application",
                           "target_codes": [], "requires_temporal": False}],
        "dependency_order": ["sq1"],
        "max_depth_needed": 1,
    }

    def _call(p: str) -> str:
        return llm_pool.call(p, model=model, max_tokens=768, temperature=0.0)

    raw = _call(prompt)
    return parse_strict_json(raw, default=default, retry_fn=lambda r: _call(r))
