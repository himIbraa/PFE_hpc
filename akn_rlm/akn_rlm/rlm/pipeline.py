"""LangGraph pipeline — Phase H.

Graph topology (→ = always, conditional edges for decompose + retry):
  START
  └─> normalize
  └─> classify_intent
  ├─> decompose          (multi_hop only)
  └─> root_rlm_run
  └─> citation_gate
  └─> jurisdiction_gate
  └─> faithfulness_gate
  ├─> assemble_output    (all gates pass OR retries exhausted)
  └─> corrective_retry   (any gate fails AND retry_count < MAX_RETRIES)
      └─> root_rlm_run   (loop back)
  └─> END

env and few_shot are captured by node closures so they are never placed in the
JSON-serialisable state dict (LangGraph strips unknown TypedDict keys).
"""
from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from akn_rlm.gates import citation_existence, faithfulness_nli
from akn_rlm.gates import jurisdiction as jurisdiction_gate
from akn_rlm.rlm.classifier import classify, IntentResult
from akn_rlm.rlm.corrective import (
    _enrich_citations,
    _hint_citation,
    _hint_faithfulness,
    _hint_jurisdiction,
)
from akn_rlm.rlm.legal_env import LegalEnv
from akn_rlm.rlm.root_controller import RootController

log = logging.getLogger(__name__)

MAX_RETRIES = 3

_SAFE_ABSTENTION: dict[str, Any] = {
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
# State schema — only JSON-serialisable fields
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    query: str
    normalized_query: str      # language-normalized form of the query
    language: str              # "ar" | "fr"
    query_type: str
    answerability_hint: str    # "probably_answerable" | "probably_unanswerable"
    intent_confidence: float
    decomposer_result: dict    # full decomposer output (sub_questions, dependency_order, …)
    sub_questions: list[str]
    rlm_output: dict           # raw RootController.run() output
    gate_results: dict         # {gate_name: {"passed": bool, "score": float, ...}}
    retry_count: int
    final_output: dict         # assembled clean answer for the caller
    telemetry: dict
    correction_notes: list[dict]
    error: str | None


# ---------------------------------------------------------------------------
# Utilities (module-level, no env dependency)
# ---------------------------------------------------------------------------

def _gate_to_dict(result) -> dict:
    return {"passed": result.passed, "score": result.score, "details": result.details}


# ---------------------------------------------------------------------------
# Gate routing (module-level — referenced by add_conditional_edges)
# ---------------------------------------------------------------------------

# R8: faithfulness is RECORD-ONLY — its score is logged into telemetry but a
# faithfulness failure never triggers a corrective retry. Only citation_existence
# and jurisdiction failures (which indicate fabrication / contamination, not
# weak support) cause the loop to retry.
_RETRYABLE_GATES: tuple[str, ...] = ("citation", "jurisdiction")


def _route_after_gates(state: PipelineState) -> str:
    gates = state.get("gate_results", {})
    retryable_failed = any(
        not gates.get(name, {}).get("passed", True) for name in _RETRYABLE_GATES
    )
    if not retryable_failed:
        return "assemble_output"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "assemble_output"
    return "corrective_retry"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_pipeline(env: LegalEnv, few_shot: list[dict] | None = None) -> Any:
    """Build and compile the Phase H LangGraph pipeline.

    Returns a compiled graph with a `.run(query: str) -> dict` helper that
    executes the full node sequence and returns the final answer dict.
    """
    _few_shot = few_shot or []

    # ── Node implementations (closures capturing env + _few_shot) ─────────

    def node_normalize(state: PipelineState) -> PipelineState:
        from akn_rlm.normalizers import normalize_query
        query = state.get("query", "")
        normalized, lang = normalize_query(query)
        return {
            **state,
            "normalized_query": normalized,
            "language": lang,
            "gate_results": {},
            "correction_notes": [],
            "retry_count": state.get("retry_count", 0),
        }

    def node_classify_intent(state: PipelineState) -> PipelineState:
        intent: IntentResult = classify(state.get("query", ""))
        return {
            **state,
            "query_type": intent.query_type,
            "answerability_hint": intent.answerability_hint,
            "intent_confidence": intent.confidence,
            "language": intent.language,
        }

    def node_decompose(state: PipelineState) -> PipelineState:
        """Call sub-LM decomposer and store full result in pipeline state."""
        from akn_rlm.rlm.sub_worker import call_decomposer
        from akn_rlm.config import SUB_LLM_MODEL
        query = state.get("query", "")
        try:
            result = call_decomposer(env.llm_pool, query, SUB_LLM_MODEL)
            sub_qs = [sq.get("text", "") for sq in result.get("sub_questions", [])]
        except Exception as exc:
            log.warning("Decomposer failed: %s — falling back to original query", exc)
            result = {}
            sub_qs = [query]
        return {**state, "decomposer_result": result, "sub_questions": sub_qs}

    def node_root_rlm_run(state: PipelineState) -> PipelineState:
        query = state.get("query", "")
        qt = state.get("query_type", "rule_application")
        env.budget.max_depth = 2 if qt in {"multi_hop", "long_context"} else 1
        env.budget.reset()  # fresh counters for each attempt (retries included)

        answerability_hint = state.get("answerability_hint", "probably_answerable")
        augmented = list(_few_shot) + list(state.get("correction_notes", []))

        # For multi_hop / long_context: prepend decomposed sub-questions as context
        sub_questions = state.get("sub_questions", [])
        if sub_questions and qt in {"multi_hop", "long_context"}:
            sq_hint = {
                "query_type": qt,
                "query": query,
                "assistant_turn": (
                    "# Decomposed sub-questions (answer each then synthesise):\n"
                    + "\n".join(f"- {sq}" for sq in sub_questions)
                ),
            }
            augmented = [sq_hint] + augmented

        # Inject answerability hint so the controller can abstain quickly if warranted
        if answerability_hint == "probably_unanswerable":
            augmented = augmented + [{
                "query_type": qt,
                "query": query,
                "assistant_turn": (
                    "# Classifier hint: this query may concern foreign law or an "
                    "absent concept. You MUST still search before abstaining (Rule 4), "
                    "but if searches confirm absence, abstain immediately."
                ),
            }]

        ctrl = RootController(env=env, few_shot=augmented)
        try:
            raw = ctrl.run(query, query_type=qt)
        except Exception as exc:
            log.error("root_rlm_run failed: %s", exc, exc_info=True)
            raw = {
                **_SAFE_ABSTENTION,
                "abstention_reason": "pipeline_error",
                "trajectory": [],
                "tokens_used": 0,
                "depth_max_reached": 0,
            }
            return {**state, "rlm_output": raw, "gate_results": {}, "error": str(exc)}
        # Reset gate_results each attempt so stale verdicts don't bleed through
        return {**state, "rlm_output": raw, "gate_results": {}}

    def node_citation_gate(state: PipelineState) -> PipelineState:
        rlm = state.get("rlm_output", {})
        # _raw_citations: normalized dicts (valid + rejected) stashed by root_controller
        raw_cites = rlm.get("_raw_citations", rlm.get("citations", []))
        result = citation_existence.run_gate(env.registry, raw_cites)
        valid = rlm.get("citations", [])
        valid_keys = {(c.get("doc_id", ""), str(c.get("article_ref", "")))
                      for c in valid}
        rejected = [c for c in raw_cites
                    if (c.get("doc_id", ""), str(c.get("article_ref", ""))) not in valid_keys]
        gr = dict(state.get("gate_results", {}))
        gr["citation"] = {**_gate_to_dict(result), "rejected": rejected}
        return {**state, "gate_results": gr}

    def node_jurisdiction_gate(state: PipelineState) -> PipelineState:
        rlm = state.get("rlm_output", {})
        gr = dict(state.get("gate_results", {}))
        qt = state.get("query_type", "rule_application")
        answer_text = rlm.get("answer_text", "") or ""
        # Run gate on both answers AND abstentions.
        # For abstentions the gate passes when foreign signals are present
        # (correct abstention) and fails when signals are absent but the model
        # still chose to abstain with no valid reason.
        effective_qt = "unanswerable" if rlm.get("abstention") else qt
        result = jurisdiction_gate.run_gate(
            answer_text, effective_qt, llm_pool=env.llm_pool
        )
        gr["jurisdiction"] = _gate_to_dict(result)
        return {**state, "gate_results": gr}

    def node_faithfulness_gate(state: PipelineState) -> PipelineState:
        rlm = state.get("rlm_output", {})
        gr = dict(state.get("gate_results", {}))
        if rlm.get("abstention"):
            gr["faithfulness"] = {"passed": True, "score": 1.0, "details": []}
            return {**state, "gate_results": gr}
        enriched = _enrich_citations(env, rlm.get("citations", []))
        result = faithfulness_nli.run_gate(
            rlm.get("answer_text", ""), enriched, llm_pool=env.llm_pool
        )
        gr["faithfulness"] = _gate_to_dict(result)
        return {**state, "gate_results": gr}

    def node_corrective_retry(state: PipelineState) -> PipelineState:
        gates = state.get("gate_results", {})
        parts: list[str] = []

        cit = gates.get("citation", {})
        if not cit.get("passed", True):
            rejected = cit.get("rejected", [])
            if rejected:
                parts.append(_hint_citation(rejected))

        jur = gates.get("jurisdiction", {})
        if not jur.get("passed", True):
            signals = [
                s for d in jur.get("details", []) for s in d.get("signals", [])
            ]
            if signals:
                parts.append(_hint_jurisdiction(signals))

        faith = gates.get("faithfulness", {})
        if not faith.get("passed", True):
            unsupported = [
                d.get("claim", "")
                for d in faith.get("details", [])
                if d.get("claim")
            ]
            if unsupported:
                parts.append(_hint_faithfulness(unsupported))

        hint = "\n\n".join(parts) or "Please revise your answer to be accurate and well-cited."
        notes = list(state.get("correction_notes", []))
        notes.append({
            "query_type": state.get("query_type", "rule_application"),
            "query": state.get("query", ""),
            "assistant_turn": f"[Previous attempt rejected — please correct]\n\n{hint}",
        })
        retry = state.get("retry_count", 0)
        log.info(
            "corrective_retry: attempt %d; failed gates=%s",
            retry,
            [k for k, v in gates.items() if not v.get("passed", True)],
        )
        return {**state, "correction_notes": notes, "retry_count": retry + 1}

    def node_assemble_output(state: PipelineState) -> PipelineState:
        import time
        rlm = state.get("rlm_output", {})
        gates = state.get("gate_results", {})
        # R8: only retryable gates (citation / jurisdiction) drive safe-abstention.
        # Faithfulness is record-only — a faithfulness failure leaves the answer
        # in place and surfaces the score in telemetry for downstream review.
        retryable_failed = any(
            not gates.get(name, {}).get("passed", True) for name in _RETRYABLE_GATES
        )
        retry = state.get("retry_count", 0)

        if retryable_failed and retry >= MAX_RETRIES:
            log.warning("Pipeline: exhausted %d retries → safe abstention", MAX_RETRIES)
            final: dict[str, Any] = dict(_SAFE_ABSTENTION)
            final["trajectory"] = rlm.get("trajectory", [])
            final["tokens_used"] = rlm.get("tokens_used", 0)
        else:
            final = dict(rlm)
            final.pop("_raw_citations", None)

        # Read accurate telemetry from the budget object (via closure over env).
        final["depth_max_reached"] = env.budget.max_depth_reached
        final["sub_call_count"] = env.budget.sub_calls_used

        final["_telemetry"] = {
            "retry_count": retry,
            "gate_results": gates,
            "language": state.get("language", "ar"),
            "depth_max_reached": env.budget.max_depth_reached,
            "sub_call_count": env.budget.sub_calls_used,
            "tokens_used": final.get("tokens_used", 0),
            "degraded_retrievers": getattr(env, "_degraded_retrievers", []),
            "query_type": state.get("query_type", "rule_application"),
            "answerability_hint": state.get("answerability_hint", "probably_answerable"),
        }
        return {**state, "final_output": final, "telemetry": final["_telemetry"]}

    # ── Graph wiring ──────────────────────────────────────────────────────

    graph = StateGraph(PipelineState)
    graph.add_node("normalize", node_normalize)
    graph.add_node("classify_intent", node_classify_intent)
    graph.add_node("decompose", node_decompose)
    graph.add_node("root_rlm_run", node_root_rlm_run)
    graph.add_node("citation_gate", node_citation_gate)
    graph.add_node("jurisdiction_gate", node_jurisdiction_gate)
    graph.add_node("faithfulness_gate", node_faithfulness_gate)
    graph.add_node("corrective_retry", node_corrective_retry)
    graph.add_node("assemble_output", node_assemble_output)

    graph.set_entry_point("normalize")
    graph.add_edge("normalize", "classify_intent")
    _DECOMPOSE_TYPES = {"multi_hop", "long_context"}
    graph.add_conditional_edges(
        "classify_intent",
        lambda s: "decompose" if s.get("query_type") in _DECOMPOSE_TYPES else "root_rlm_run",
        {"decompose": "decompose", "root_rlm_run": "root_rlm_run"},
    )
    graph.add_edge("decompose", "root_rlm_run")
    graph.add_edge("root_rlm_run", "citation_gate")
    graph.add_edge("citation_gate", "jurisdiction_gate")
    graph.add_edge("jurisdiction_gate", "faithfulness_gate")
    graph.add_conditional_edges(
        "faithfulness_gate",
        _route_after_gates,
        {"assemble_output": "assemble_output", "corrective_retry": "corrective_retry"},
    )
    graph.add_edge("corrective_retry", "root_rlm_run")
    graph.add_edge("assemble_output", END)

    compiled = graph.compile()

    def run(query: str) -> dict:
        env.begin_query(clear_cache=True)
        initial: PipelineState = {"query": query, "retry_count": 0, "error": None}
        result = compiled.invoke(initial)
        return result.get("final_output", result.get("rlm_output", {}))

    compiled.run = run  # type: ignore[attr-defined]
    return compiled
