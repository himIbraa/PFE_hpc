"""Corrective loop helpers — used internally by pipeline.py.

NOTE: ``run_with_correction()`` below is a DEPRECATED standalone entry point
retained only for backward compatibility.  The live code path is the
LangGraph corrective_retry node in ``rlm/pipeline.py``.  Do not call
``run_with_correction()`` directly; use ``build_pipeline(env).run(query)``
instead.

Corrective loop (Phase G): run all three faithfulness gates after each
RootController attempt; build targeted retry hints per failure mode; emit a
safe abstention after MAX_RETRIES exhausted.

Failure mode → retry hint mapping:
  citation_existence  "Citations X were not found in the corpus.
                       Re-retrieve and only cite articles returned by
                       env.get_article calls."
  jurisdiction        "Your answer referenced foreign-law concepts: {hits}.
                       The question is about Algerian law only.
                       Either find the Algerian rule or abstain."
  faithfulness        "The following claims are not supported by the cited
                       articles: {claims}.
                       Re-read the articles or remove unsupported claims."
"""
from __future__ import annotations

import logging
from typing import Any

from akn_rlm.gates import citation_existence, faithfulness_nli
from akn_rlm.gates import jurisdiction as jurisdiction_gate
from akn_rlm.rlm.legal_env import LegalEnv
from akn_rlm.rlm.root_controller import RootController

log = logging.getLogger(__name__)

MAX_RETRIES = 3

_SAFE_ABSTENTION_BASE: dict[str, Any] = {
    "answer_text": "",
    "abstention": True,
    "abstention_reason": "system_could_not_verify",
    "citations": [],
    "reasoning_chain": [],
    "trajectory": [],
    "tokens_used": 0,
    "depth_max_reached": 0,
}


# ---------------------------------------------------------------------------
# Retry hint builders
# ---------------------------------------------------------------------------

def _hint_citation(rejected: list[dict]) -> str:
    rejected_str = "; ".join(
        f"{c.get('doc_id', '?')} art.{c.get('article_ref', '?')}"
        for c in rejected
    )
    return (
        f"Citations {rejected_str} were not found in the corpus. "
        "Re-retrieve and only cite articles returned by `env.get_article` calls."
    )


def _hint_jurisdiction(signals: list[str]) -> str:
    return (
        f"Your answer referenced foreign-law concepts: {', '.join(signals)}. "
        "The question is about Algerian law only. "
        "Either find the Algerian rule or abstain."
    )


def _hint_faithfulness(unsupported_claims: list[str]) -> str:
    claims_str = "; ".join(f'"{c}"' for c in unsupported_claims[:3])
    return (
        f"The following claims are not supported by the cited articles: {claims_str}. "
        "Re-read the articles or remove unsupported claims."
    )


def _build_retry_hint(failures: list[tuple[str, Any]]) -> str:
    """Combine multiple per-gate hints into one correction message."""
    parts: list[str] = []
    for gate_name, details in failures:
        if gate_name == "citation":
            parts.append(_hint_citation(details))
        elif gate_name == "jurisdiction":
            parts.append(_hint_jurisdiction(details))
        elif gate_name == "faithfulness":
            parts.append(_hint_faithfulness(details))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Article text enrichment
# ---------------------------------------------------------------------------

def _enrich_citations(env: LegalEnv, citations: list[dict]) -> list[dict]:
    """Fetch full article text for citations that only have supporting_span."""
    enriched = []
    for c in citations:
        c2 = dict(c)
        if not c2.get("text"):
            try:
                art = env.get_article(c["doc_id"], str(c.get("article_ref", "")))
                c2["text"] = (art or {}).get("text", "")
            except Exception:
                pass
        enriched.append(c2)
    return enriched


# ---------------------------------------------------------------------------
# Main corrective loop
# ---------------------------------------------------------------------------

def run_with_correction(
    controller: RootController,
    query: str,
    query_type: str | None = None,
) -> dict[str, Any]:
    """Run the RLM pipeline with up to MAX_RETRIES corrective retries.

    Each retry appends a structured correction hint to the few-shot context
    so the LLM can self-correct.  If all retries are exhausted, emits a safe
    abstention with abstention_reason='system_could_not_verify'.
    """
    env = controller.env
    llm_pool = env.llm_pool
    qt = query_type or "rule_application"

    correction_notes: list[dict] = []
    result: dict[str, Any] = {}
    attempt = 0

    while attempt <= MAX_RETRIES:
        # Build controller (augmented few-shot on retries)
        if attempt == 0:
            ctrl = controller
        else:
            augmented = list(controller.few_shot or []) + correction_notes
            ctrl = RootController(
                env=env,
                system_prompt=controller.system_prompt,
                few_shot=augmented,
                root_model=controller.root_model,
                timeout_default=controller.timeout_default,
                timeout_multihop=controller.timeout_multihop,
                max_rounds=controller.max_rounds,
            )

        result = ctrl.run(query, query_type=qt)

        # ── Gate 1: Citation existence ────────────────────────────────────
        # _raw_citations is already normalized dicts (root_controller stashes valid+rejected)
        raw_citations = result.get("_raw_citations", result.get("citations", []))
        valid_citations = result.get("citations", [])
        # compare by (doc_id, article_ref) since object identity differs after normalize
        valid_keys = {(c.get("doc_id", ""), str(c.get("article_ref", "")))
                      for c in valid_citations}
        rejected = [c for c in raw_citations
                    if (c.get("doc_id", ""), str(c.get("article_ref", ""))) not in valid_keys]

        failures: list[tuple[str, Any]] = []
        if rejected:
            failures.append(("citation", rejected))

        # ── Gate 2: Jurisdiction (skip for abstentions) ───────────────────
        if not result.get("abstention"):
            jur_result = jurisdiction_gate.run_gate(
                result.get("answer_text", ""),
                qt,
                llm_pool=None,   # regex-only in corrective loop (LLM is expensive per retry)
            )
            if not jur_result.passed:
                signals = [
                    sig
                    for d in jur_result.details
                    for sig in d.get("signals", [])
                ]
                failures.append(("jurisdiction", signals))

        # ── Gate 3: Faithfulness (skip for abstentions, skip if no cites) ─
        if not result.get("abstention") and valid_citations:
            enriched = _enrich_citations(env, valid_citations)
            faith_result = faithfulness_nli.run_gate(
                result.get("answer_text", ""),
                enriched,
                llm_pool=None,   # NLI-only; LLM fallback is expensive per-retry
            )
            if not faith_result.passed:
                unsupported = [
                    d.get("claim", "")
                    for d in faith_result.details
                    if d.get("claim")
                ]
                failures.append(("faithfulness", unsupported))

        # ── All gates passed ──────────────────────────────────────────────
        if not failures:
            log.info("Corrective loop: all gates passed on attempt %d", attempt)
            break

        if attempt >= MAX_RETRIES:
            log.warning(
                "Corrective loop: exhausted %d retries; emitting safe abstention",
                MAX_RETRIES,
            )
            safe = dict(_SAFE_ABSTENTION_BASE)
            safe["trajectory"] = result.get("trajectory", [])
            safe["tokens_used"] = result.get("tokens_used", 0)
            result = safe
            break

        # Build targeted retry hint
        hint = _build_retry_hint(failures)
        correction_notes.append({
            "query_type": qt,
            "query": query,
            "assistant_turn": f"[Previous attempt rejected — please correct]\n\n{hint}",
        })
        log.info(
            "Corrective loop: attempt %d failed gates=%s; retrying",
            attempt, [f[0] for f in failures],
        )
        attempt += 1

    result.pop("_raw_citations", None)
    return result
