"""Argumentation-unit extractor: full Toulmin structure from legal article text.

Returns a dict with keys:
  claim    — the central normative assertion of the article
  ground   — the factual conditions / applicability criteria
  warrant  — the legal principle connecting ground to claim
  backing  — the broader legal/doctrinal source that supports the warrant
  rebuttal — the exception or defeating condition (may be empty string)
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

_TOULMIN_PROMPT = """\
You are an expert in legal argumentation analysis (Toulmin model).

Extract the full Toulmin argumentation structure from the following Algerian legal article.

Definitions:
- claim:    The central normative rule or obligation stated by the article.
- ground:   The factual conditions or applicability criteria (who/when/what triggers the rule).
- warrant:  The legal principle or ratio legis connecting the ground to the claim.
- backing:  The broader doctrinal or constitutional source that legitimises the warrant (may be empty).
- rebuttal: The exception, limitation, or defeating condition (may be empty if none stated).

Return ONLY valid JSON with exactly these five keys and string values.
Do NOT add any text before or after the JSON object.

Article text:
{article_text}

JSON output:"""

_FALLBACK_KEYS = ("claim", "ground", "warrant", "backing", "rebuttal")
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_json(raw: str) -> dict | None:
    """Extract and parse the first JSON object from raw LLM output."""
    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group())
            if all(k in obj for k in _TOULMIN_KEYS):
                return obj
        except json.JSONDecodeError:
            pass
    return None


_TOULMIN_KEYS = _FALLBACK_KEYS


def extract(article_text: str, llm_pool, *, model: str, max_tokens: int = 512) -> dict:
    """Extract full Toulmin structure from article_text using llm_pool.

    Falls back to empty-string values on any failure.
    """
    empty = {k: "" for k in _FALLBACK_KEYS}
    if not article_text or not article_text.strip():
        return empty

    prompt = _TOULMIN_PROMPT.format(article_text=article_text[:2000])
    try:
        raw = llm_pool.call(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as exc:
        log.warning("ADU extractor LLM call failed: %s", exc)
        return empty

    parsed = _parse_json(raw)
    if parsed is None:
        log.debug("ADU extractor: could not parse JSON from: %r", raw[:200])
        return empty

    # Normalise: ensure all five keys exist, fill missing with ""
    return {k: str(parsed.get(k, "")) for k in _FALLBACK_KEYS}
