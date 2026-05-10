"""Per-citation supervisor — Phase 2 / R9.5.

Re-ranks candidate citations using a stronger LLM (default
``gpt-oss-120b``) to score per-article entailment of the query. The
supervisor is the discriminator the verifier (running on the smaller
sub-LM) is too noisy to be: it sees ALL candidate articles in a single
prompt, so it can directly compare them to each other and to the
question, rather than judging each article in isolation.

Output: a re-ranked subset of the input citations with a new
``supervisor_score`` field on each kept citation. Citations whose
supervisor score is ``< threshold`` are dropped.

Cache: keyed on ``(sha256(query), tuple(sorted(doc_id, article_ref)))``
so identical (query, citation-set) pairs return the cached supervisor
output without re-prompting. The cache is in-memory and process-local;
the F3 dispatcher run is single-process so cross-question cache reuse
is a pure win.

Fail-open semantics: any LLM exception, JSON parse failure, missing
score, or out-of-range score returns the input citations unchanged. A
transient supervisor failure can never make a handler worse than its
pre-supervisor state.

Smart trigger: see :func:`should_supervise`. The supervisor only fires
when the verifier's confidence is in the uncertainty band
(``[0.30, 0.70]`` by default) AND there are at least 2 candidate
citations. Already-high-confidence cases skip — re-supervising them
would burn budget without changing the outcome.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

from akn_rlm.rlm.sub_worker import parse_strict_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: gpt-oss-120b is the strongest model in the AI Grid pool and the
#: cheapest place to spend the supervisor budget.
DEFAULT_MODEL: str = "gpt-oss-120b"

#: Drop a citation if its supervisor score is strictly less than this.
#: F6 tried 0.5 (more aggressive filter) and regressed RA/MH/CD by
#: total −0.058 on those strata (gpt-oss-120b at thr=0.5 dropped
#: foundational/scope articles that ARE the gold answer for many
#: queries — same R3/R4 lesson as the verifier). F7 reverted to 0.3.
DEFAULT_THRESHOLD: float = 0.3

#: Lower bound of the verifier-confidence uncertainty band — opt-in
#: only (set ``uncertainty_band=`` in :func:`should_supervise`). The
#: original R9.5 design used [0.30, 0.70], but F3 telemetry showed
#: Qwen3 confidences are bimodal so the band almost never matches —
#: the supervisor fired 0 times in 244 q. F4 default trigger is
#: count-only (see :func:`should_supervise`).
DEFAULT_TRIGGER_LOW: float = 0.30

#: Upper bound of the verifier-confidence uncertainty band. Opt-in only.
DEFAULT_TRIGGER_HIGH: float = 0.70

#: Minimum number of candidate citations needed to fire the supervisor.
#: Bumped 2 → 3 in F4: with the tightened per-handler ``final_top_k``
#: (RA 4, EA 3, MH 5, CD 5, LC 6) we only want the supervisor when
#: there are at least 3 candidates to discriminate among — below that
#: the supervisor's marginal lift is too small to pay the latency.
DEFAULT_MIN_CITATIONS: int = 3

#: Max characters of article text passed to the supervisor per
#: candidate. Keeps the prompt size bounded across 5-10 candidates.
_MAX_TEXT_CHARS: int = 1500

# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict[str, Any]]] = {}


def clear_cache() -> None:
    """Reset the supervisor cache. Useful in tests; harmless in
    production (the cache is process-local)."""
    _CACHE.clear()


def _cache_key(
    query: str, citations: list[dict[str, Any]],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    qh = hashlib.sha256((query or "").encode("utf-8")).hexdigest()
    keys = tuple(
        sorted(
            (str(c.get("doc_id", "")), str(c.get("article_ref", "")))
            for c in citations
        )
    )
    return (qh, keys)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

_SUPERVISOR_PROMPT = """\
You are a legal-research supervisor reviewing candidate articles cited \
in response to a question about Algerian law. For EACH candidate, score \
on a 0.0-1.0 scale how strongly the article TEXT entails or directly \
supports an answer to the QUESTION. 1.0 = the article directly answers \
the question; 0.0 = the article is irrelevant; 0.5 = the article is \
topically related but does not on its own support an answer.

Return ONLY a JSON object exactly of the form:
{{"scores": {{"0": 0.85, "1": 0.20, "2": 0.55}}}}

Use the candidate's bracketed index as the JSON key (string). Score \
EVERY candidate. Do not include any prose, no markdown fences, no \
explanation.

QUESTION:
{question}

CANDIDATES:
{candidate_block}

Return only the JSON object.
"""


SupervisorFn = Callable[[Any, str, list[dict[str, Any]]], list[dict[str, Any]]]


def supervise_citations(
    llm_pool,
    query: str,
    citations: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Re-rank ``citations`` by an LLM-supervised entailment score.

    Returns the input list unchanged on any LLM/JSON failure (fail-open).
    """
    if not citations:
        return citations
    if not isinstance(query, str) or not query.strip():
        return citations

    key = _cache_key(query, citations)
    cached = _CACHE.get(key)
    if cached is not None:
        # Return defensive copies so callers can mutate without polluting
        # cached entries.
        return [dict(c) for c in cached]

    block_lines: list[str] = []
    for i, c in enumerate(citations):
        doc_id = c.get("doc_id", "")
        ref = c.get("article_ref", "")
        text = (c.get("text") or c.get("supporting_span") or "")[:_MAX_TEXT_CHARS]
        block_lines.append(f"[{i}] [{doc_id} art.{ref}]\n{text}")
    candidate_block = "\n\n".join(block_lines)

    prompt = _SUPERVISOR_PROMPT.format(
        question=query.strip(),
        candidate_block=candidate_block,
    )

    try:
        raw = llm_pool.call(prompt, model=model, max_tokens=512, temperature=0.0)
    except Exception as exc:
        log.warning(
            "supervisor LLM call failed (%s) — returning citations unchanged",
            exc,
        )
        return citations

    parsed = parse_strict_json(raw, default={})
    scores_raw = parsed.get("scores") if isinstance(parsed, dict) else None
    if not isinstance(scores_raw, dict) or not scores_raw:
        log.warning(
            "supervisor returned no scores dict — returning citations unchanged"
        )
        return citations

    parsed_scores: list[float] = []
    for i in range(len(citations)):
        s_raw = scores_raw.get(str(i))
        if s_raw is None:
            s_raw = scores_raw.get(i)
        try:
            s = float(s_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            log.warning(
                "supervisor missing/unparseable score for citation %d — "
                "returning citations unchanged",
                i,
            )
            return citations
        if not 0.0 <= s <= 1.0:
            log.warning(
                "supervisor score %s for citation %d out of [0,1] range — "
                "returning citations unchanged",
                s, i,
            )
            return citations
        parsed_scores.append(s)

    survivors: list[tuple[float, dict[str, Any]]] = []
    for s, c in zip(parsed_scores, citations):
        if s >= threshold:
            new_c = dict(c)
            new_c["supervisor_score"] = float(s)
            survivors.append((s, new_c))
    survivors.sort(key=lambda t: t[0], reverse=True)
    out = [c for _, c in survivors]

    _CACHE[key] = [dict(c) for c in out]
    return out


# ---------------------------------------------------------------------------
# Smart trigger
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plan supervisor (R9.6) — multi_hop sub-question planner
# ---------------------------------------------------------------------------

#: Minimum number of content tokens (whitespace-separated, length >= 2)
#: required before the plan supervisor will fire. Below this, the
#: question is too short to benefit from decomposition.
DEFAULT_PLAN_MIN_CONTENT_TOKENS: int = 3


_PLAN_SUPERVISOR_PROMPT = """\
You are a legal-research planner for the Algerian legal corpus. Given a \
multi-hop question, decompose it into 2-5 atomic sub-questions that, when \
answered together, fully answer the original question. For each \
sub-question, ALSO predict which of the provided ``ROUTED DOC IDs`` are \
most likely to contain the answer (a subset, possibly empty if no doc \
fits).

Return ONLY a JSON object exactly of the form:
{{"sub_questions": [{{"id": "sq1", "text": "...", "target_docs": ["doc_a", "doc_b"]}}, ...]}}

Rules:
- 2-5 sub-questions max.
- ``id`` strings must be unique (e.g. ``sq1``, ``sq2``).
- ``text`` is the atomic sub-question in Modern Standard Arabic.
- ``target_docs`` is a list (possibly empty) of doc_ids drawn ONLY \
from the ROUTED DOC IDs list. Do not invent doc_ids.
- Do not include any prose, no markdown fences, no explanation.

QUESTION:
{question}

ROUTED DOC IDs:
{routed_block}

Return only the JSON object.
"""


PlanSupervisorFn = Callable[..., dict[str, Any]]


def _count_content_tokens(query: str) -> int:
    if not query:
        return 0
    return sum(1 for tok in query.split() if len(tok) >= 2)


def supervise_plan(
    llm_pool,
    query: str,
    *,
    routed_doc_ids: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    min_content_tokens: int = DEFAULT_PLAN_MIN_CONTENT_TOKENS,
) -> dict[str, Any]:
    """Have ``gpt-oss-120b`` write a multi-hop sub-question plan.

    Returns a dict with shape ``{"sub_questions": [{"id", "text",
    "target_docs"}, ...]}`` on success. Returns ``{}`` when:

      - the query has fewer than ``min_content_tokens`` content tokens,
      - the LLM raises,
      - the JSON does not parse,
      - the parsed JSON does not match the expected shape.

    The caller (multi_hop handler) treats ``{}`` as "fall back to the
    existing Qwen-based decomposer". This keeps the plan supervisor
    fail-open: a plan-supervisor failure can never make multi_hop
    worse than its pre-R9.6 state.
    """
    if not isinstance(query, str) or not query.strip():
        return {}
    if _count_content_tokens(query) < min_content_tokens:
        return {}

    routed_doc_ids = list(routed_doc_ids or [])
    routed_block = "\n".join(f"- {d}" for d in routed_doc_ids) if routed_doc_ids else "(none)"
    prompt = _PLAN_SUPERVISOR_PROMPT.format(
        question=query.strip(),
        routed_block=routed_block,
    )

    try:
        raw = llm_pool.call(prompt, model=model, max_tokens=768, temperature=0.0)
    except Exception as exc:
        log.warning("plan supervisor LLM call failed (%s) — falling back", exc)
        return {}

    parsed = parse_strict_json(raw, default={})
    if not isinstance(parsed, dict):
        return {}
    sub_qs_raw = parsed.get("sub_questions")
    if not isinstance(sub_qs_raw, list) or not sub_qs_raw:
        return {}

    routed_set = set(routed_doc_ids)
    out_sub_qs: list[dict[str, Any]] = []
    for i, sq in enumerate(sub_qs_raw):
        if not isinstance(sq, dict):
            continue
        text = (sq.get("text") or "").strip()
        if not text:
            continue
        sq_id = sq.get("id") or f"sq{i + 1}"
        targets_raw = sq.get("target_docs") or []
        if not isinstance(targets_raw, list):
            targets_raw = []
        # Filter target_docs to those actually routed (drop hallucinated IDs).
        targets = [t for t in targets_raw if isinstance(t, str) and (not routed_set or t in routed_set)]
        out_sub_qs.append({
            "id":          str(sq_id),
            "text":        text,
            "target_docs": targets,
            "type":        sq.get("type") or "rule_application",
        })
        if len(out_sub_qs) >= 5:
            break

    if not out_sub_qs:
        return {}
    return {"sub_questions": out_sub_qs}


# ---------------------------------------------------------------------------
# Smart trigger (R9.5)
# ---------------------------------------------------------------------------

def should_supervise(
    citations: list[dict[str, Any]],
    *,
    min_citations: int = DEFAULT_MIN_CITATIONS,
    uncertainty_band: tuple[float, float] | None = None,
) -> bool:
    """Decide whether to invoke :func:`supervise_citations` on the given
    citations.

    Default trigger (F4): fires whenever ``len(citations) >=
    min_citations``. The original R9.5 confidence-band trigger
    ([0.30, 0.70]) is preserved as an opt-in via ``uncertainty_band=
    (low, high)`` — F3 telemetry showed Qwen3 verifier confidences
    are bimodal (rejected < 0.3 or accepted ≥ 0.7+) so the band
    almost never matched and the supervisor fired 0 times in 244 q.
    F4 lets the supervisor actually run on every multi-citation set
    so it can earn the latency it costs.

    Citations are assumed to be sorted by ``confidence`` desc.
    """
    if not citations or len(citations) < min_citations:
        return False
    if uncertainty_band is not None:
        low, high = uncertainty_band
        top_conf = float(citations[0].get("confidence", 0.0) or 0.0)
        return low <= top_conf <= high
    return True
