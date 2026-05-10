"""RLM dispatcher — Phase 2 / R7.

Routes a query to the right Phase-2 typed handler based on its
``query_type``. The Phase-2 handlers (R2-R6) are self-contained
baseline-shaped pipelines that intentionally bypass the freeform-Python
``RootController``; this dispatcher is the production seam that wires
them together so the benchmark runner can dispatch every question to
its native handler in a single sweep.

Pipeline:

  1. If ``query_type`` is supplied (the benchmark always tags every
     question), use it directly. Otherwise fall back to
     ``akn_rlm.rlm.classifier.classify`` to pick.

  2. Map the type to a handler key (the eight ALB v3.0 types are
     1-to-1; the legacy ``temporal`` alias coalesces to
     ``temporal_factual``; unknown types fall back to
     ``rule_application`` — the deterministic-template default).

  3. **Lazy-build** the chosen handler. KG-loading handlers
     (``temporal_factual``, ``conceptual_definitional``) only parse the
     ~26-second TTL when actually dispatched. This keeps the gate cheap
     for slices that don't touch the KG.

  4. Forward ``handler.run(query)``. The answer is shaped exactly like
     the deterministic baselines so
     :func:`akn_rlm.eval.runner._answer_to_result` consumes it
     unchanged.

  5. Patch telemetry: keep the inner handler's ``baseline`` tag (so
     ``compare_baselines.py`` can still pick out per-handler runs from
     the predictions if desired) and add ``dispatched_handler`` /
     ``dispatched_query_type`` / ``dispatch_baseline = "rlm_dispatched"``
     so the dispatched run gets its own column in the comparison
     table.

The long_context handler gets a **per-summariser timeout** wrapper
(default 60 s) before being constructed: HANDOFF §R6.4 documents one
``com_lc_q01`` query that hung Qwen3-30B-A3B-Thinking for ~5 hours on a
10-article prompt, which would let one query hostage an entire
benchmark sweep. On timeout the wrapper raises ``TimeoutError`` so the
existing ``LongContextHandler`` fallback (deterministic Arabic
template) takes over.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Optional

from akn_rlm.config import SUB_LLM_MODEL
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.indexers.bm25 import BM25Index
from akn_rlm.indexers.dense import DenseIndex
from akn_rlm.rlm.classifier import classify as _classify_intent
from akn_rlm.rlm.handlers import (
    LAYMAN_DEFAULT_REWRITE_MODEL,
    build_conceptual_definitional_handler,
    build_exact_article_handler,
    build_layman_handler,
    build_long_context_handler,
    build_multi_hop_handler,
    build_rule_application_handler,
    build_temporal_factual_handler,
    build_unanswerable_handler,
)
from akn_rlm.rlm.routing import DocRouter, build_doc_router
from akn_rlm.rlm.sub_worker import call_summarizer
from akn_rlm.rlm.supervisor import (
    DEFAULT_MODEL as SUPERVISOR_DEFAULT_MODEL,
    DEFAULT_THRESHOLD as SUPERVISOR_DEFAULT_THRESHOLD,
    PlanSupervisorFn,
    SupervisorFn,
    supervise_citations,
    supervise_plan,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Wall-clock seconds before the long_context summariser is forcibly
#: aborted and the handler falls back to the deterministic Arabic
#: template. HANDOFF §R6.4 records a 5-hour hang on ``com_lc_q01``;
#: 60 s is the budget we picked to bound the full-244-q run.
DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S: float = 60.0

#: Telemetry tag stamped onto every dispatched answer. Lets
#: ``compare_baselines.py`` pull dispatched runs out as a single column
#: regardless of which inner handler ran.
DISPATCH_BASELINE: str = "rlm_dispatched"

#: Handler key reached when ``query_type`` is missing / unknown.
DEFAULT_FALLBACK_HANDLER: str = "rule_application"

#: Map every benchmark ``query_type`` (and the legacy ``temporal``
#: alias from ``root_controller.classify_query_type``) to its handler
#: key. Keys MUST match the eight ``query_type`` strings used by the
#: ALB v3.0 records and by ``classifier.classify``.
TYPE_TO_HANDLER: dict[str, str] = {
    "rule_application":         "rule_application",
    "exact_article":            "exact_article",
    "multi_hop":                "multi_hop",
    "unanswerable":             "unanswerable",
    "layman":                   "layman",
    "long_context":             "long_context",
    "conceptual_definitional":  "conceptual_definitional",
    "temporal_factual":         "temporal_factual",
    # Legacy alias emitted by ``root_controller.classify_query_type``;
    # the benchmark itself only ever uses ``temporal_factual``.
    "temporal":                 "temporal_factual",
}

#: Handler keys that need the KG. KG load is ~26 s; we keep the
#: rest of the dispatcher cheap by deferring this until truly needed.
_KG_HANDLER_KEYS: frozenset[str] = frozenset({
    "temporal_factual",
    "conceptual_definitional",
})


ClassifierFn = Callable[[str], str]
KGLoaderFn = Callable[[], Any]


# ---------------------------------------------------------------------------
# Timeout-wrapped summariser for long_context
# ---------------------------------------------------------------------------

def _make_timeout_summarizer(
    timeout_s: float,
    inner: Callable = call_summarizer,
) -> Callable:
    """Wrap a summariser with a thread-based wall-clock timeout.

    Why threads: the underlying LLM call is a blocking native HTTP
    request that doesn't honour ``signal.alarm`` (and ``signal.alarm``
    is unix-only — this project runs on Windows). We submit the call
    to a single-worker executor and ``.result(timeout=...)``. On
    timeout we raise ``TimeoutError`` so the long_context handler's
    existing ``except Exception`` path falls back to the template.

    The thread that's stuck on the slow LLM call cannot be cancelled
    (Python doesn't support thread cancellation for arbitrary native
    calls), so we ``shutdown(wait=False)`` and let the OS reclaim it
    when the process exits. This is acceptable for a CLI benchmark
    runner.
    """

    def wrapped(pool, query, articles, model):
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lc_summarizer"
        )
        fut = executor.submit(inner, pool, query, articles, model)
        try:
            result = fut.result(timeout=timeout_s)
        except FuturesTimeoutError:
            executor.shutdown(wait=False)
            log.warning(
                "long_context summariser exceeded %.1fs — falling back "
                "to deterministic template", timeout_s,
            )
            raise TimeoutError(
                f"long_context summariser timeout after {timeout_s:.1f}s"
            )
        executor.shutdown(wait=False)
        return result

    return wrapped


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class RLMDispatcher:
    """Wires the eight Phase-2 typed handlers behind a single
    ``run(query, query_type=None)`` entrypoint."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        registry: ArticleRegistry,
        llm_pool,
        *,
        router: Optional[DocRouter] = None,
        kg: Any = None,
        kg_loader: Optional[KGLoaderFn] = None,
        sub_model: str = SUB_LLM_MODEL,
        rewrite_model: str = LAYMAN_DEFAULT_REWRITE_MODEL,
        classifier_fn: Optional[ClassifierFn] = None,
        long_context_timeout_s: float = DEFAULT_LONG_CONTEXT_SUMMARIZER_TIMEOUT_S,
        handler_overrides: Optional[dict[str, Any]] = None,
        supervisor_fn: Optional[SupervisorFn] = supervise_citations,
        supervisor_model: str = SUPERVISOR_DEFAULT_MODEL,
        supervisor_threshold: float = SUPERVISOR_DEFAULT_THRESHOLD,
        # R9.6: plan_supervisor stays *opt-in* — empirical evidence on
        # the n=10 multi_hop slice shows it regresses Cite F1 from
        # 0.167 (R9.3 alone) to 0.070 when wired through the
        # dispatcher. The seam + tests are kept so a future operator
        # can flip it on with a stronger plan-prompt or a different
        # model.
        plan_supervisor_fn: Optional[PlanSupervisorFn] = None,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._registry = registry
        self._llm_pool = llm_pool
        self._router = router or build_doc_router(registry=registry, bm25=bm25)
        self._kg = kg
        self._kg_loader = kg_loader
        self._sub_model = sub_model
        self._rewrite_model = rewrite_model
        self._classifier_fn = classifier_fn
        self._long_context_timeout_s = float(long_context_timeout_s)
        # R9.5 + R9.6: supervisor seams. ``supervisor_fn`` is the
        # per-citation re-ranker injected into the LLM-using handlers
        # (RA, MH, EA, layman, LC). ``plan_supervisor_fn`` is the
        # multi_hop sub-question planner from R9.6.
        if supervisor_fn is not None:
            def _bound_supervisor(pool, q, cits):
                return supervisor_fn(
                    pool, q, cits,
                    threshold=supervisor_threshold,
                    model=supervisor_model,
                )
            self._supervisor_fn: Optional[SupervisorFn] = _bound_supervisor
        else:
            self._supervisor_fn = None
        self._plan_supervisor_fn = plan_supervisor_fn
        # Pre-populated overrides are honoured as-is — the dispatcher
        # never replaces an injected handler. Tests use this to inject
        # mocks; production code can use it to swap in a custom
        # implementation.
        self._handlers: dict[str, Any] = dict(handler_overrides or {})

    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        query_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Dispatch ``query`` to the handler chosen by ``query_type``."""
        if not isinstance(query, str) or not query.strip():
            return self._abstain_envelope(
                reason="empty_query",
                dispatched_handler=None,
                dispatched_query_type=query_type,
            )

        resolved_type = (query_type or "").strip() or self._classify(query)
        handler_key = TYPE_TO_HANDLER.get(resolved_type, DEFAULT_FALLBACK_HANDLER)

        try:
            handler = self._get_handler(handler_key)
        except Exception as exc:
            log.error(
                "Dispatcher could not build handler %s: %s",
                handler_key, exc,
            )
            return self._abstain_envelope(
                reason="dispatch_build_error",
                dispatched_handler=handler_key,
                dispatched_query_type=resolved_type,
                error=str(exc),
            )

        # R9.7: per-handler-run call recording. The pool exposes
        # start_recording / stop_recording; we snapshot on entry and
        # exit so even an exception path leaves the pool in a clean
        # state. Mocks that don't implement the methods are tolerated.
        recording_active = False
        if hasattr(self._llm_pool, "start_recording"):
            try:
                self._llm_pool.start_recording()
                recording_active = True
            except Exception:
                recording_active = False

        try:
            answer = handler.run(query)
        except Exception as exc:
            if recording_active:
                try:
                    self._llm_pool.stop_recording()
                except Exception:
                    pass
            log.error(
                "Dispatcher handler %s raised: %s", handler_key, exc,
            )
            return self._abstain_envelope(
                reason="dispatch_pipeline_error",
                dispatched_handler=handler_key,
                dispatched_query_type=resolved_type,
                error=str(exc),
            )

        calls_by_model: dict[str, int] = {}
        if recording_active:
            try:
                calls_by_model = self._llm_pool.stop_recording()
            except Exception:
                calls_by_model = {}

        if not isinstance(answer, dict):
            log.error(
                "Dispatcher handler %s returned non-dict: %r",
                handler_key, type(answer).__name__,
            )
            return self._abstain_envelope(
                reason="dispatch_bad_answer_shape",
                dispatched_handler=handler_key,
                dispatched_query_type=resolved_type,
            )

        # Patch telemetry: keep the inner handler's tag (so per-handler
        # eval runs still classify correctly) and add the dispatcher's
        # bookkeeping.
        tel = answer.setdefault("_telemetry", {})
        if not isinstance(tel, dict):
            tel = {}
            answer["_telemetry"] = tel
        tel["dispatched_handler"] = handler_key
        tel["dispatched_query_type"] = resolved_type
        tel["dispatch_baseline"] = DISPATCH_BASELINE
        # Preserve the inner handler's "baseline" if present; otherwise
        # stamp the dispatcher tag so downstream tooling never sees a
        # blank baseline field.
        tel.setdefault("baseline", DISPATCH_BASELINE)
        # R9.7: per-handler-run model counts.
        tel["calls_by_model"] = calls_by_model
        return answer

    # ------------------------------------------------------------------
    # Type resolution
    # ------------------------------------------------------------------

    def _classify(self, query: str) -> str:
        """Resolve ``query_type`` when the caller didn't supply one."""
        if self._classifier_fn is not None:
            try:
                return self._classifier_fn(query)
            except Exception as exc:
                log.warning(
                    "Custom classifier raised %s — falling back to "
                    "akn_rlm.rlm.classifier", exc,
                )
        try:
            return _classify_intent(query).query_type
        except Exception as exc:
            log.warning(
                "Default classifier raised %s — falling back to %s",
                exc, DEFAULT_FALLBACK_HANDLER,
            )
            return DEFAULT_FALLBACK_HANDLER

    # ------------------------------------------------------------------
    # Lazy handler construction
    # ------------------------------------------------------------------

    def _get_handler(self, key: str) -> Any:
        if key not in self._handlers:
            self._handlers[key] = self._build(key)
        return self._handlers[key]

    def _build(self, key: str) -> Any:
        common: dict[str, Any] = dict(
            bm25=self._bm25,
            dense=self._dense,
            registry=self._registry,
            llm_pool=self._llm_pool,
            router=self._router,
            sub_model=self._sub_model,
        )
        if key == "rule_application":
            return build_rule_application_handler(
                **common, supervisor_fn=self._supervisor_fn,
            )
        if key == "exact_article":
            # exact_article never uses dense — see HANDOFF §R6.2.
            return build_exact_article_handler(
                bm25=self._bm25,
                registry=self._registry,
                llm_pool=self._llm_pool,
                router=self._router,
                sub_model=self._sub_model,
                supervisor_fn=self._supervisor_fn,
            )
        if key == "multi_hop":
            mh_kwargs: dict[str, Any] = dict(common)
            mh_kwargs["supervisor_fn"] = self._supervisor_fn
            if self._plan_supervisor_fn is not None:
                mh_kwargs["plan_supervisor_fn"] = self._plan_supervisor_fn
            return build_multi_hop_handler(**mh_kwargs)
        if key == "long_context":
            timeout_summarizer = _make_timeout_summarizer(
                self._long_context_timeout_s
            )
            return build_long_context_handler(
                bm25=self._bm25,
                dense=self._dense,
                registry=self._registry,
                llm_pool=self._llm_pool,
                router=self._router,
                sub_model=self._sub_model,
                summarizer_fn=timeout_summarizer,
                supervisor_fn=self._supervisor_fn,
            )
        if key == "layman":
            return build_layman_handler(
                bm25=self._bm25,
                dense=self._dense,
                registry=self._registry,
                llm_pool=self._llm_pool,
                router=self._router,
                sub_model=self._sub_model,
                rewrite_model=self._rewrite_model,
                supervisor_fn=self._supervisor_fn,
            )
        if key == "unanswerable":
            return build_unanswerable_handler(
                bm25=self._bm25,
                dense=self._dense,
                registry=self._registry,
                llm_pool=self._llm_pool,
                router=self._router,
                sub_model=self._sub_model,
            )
        if key in _KG_HANDLER_KEYS:
            kg = self._get_kg()
            if key == "temporal_factual":
                return build_temporal_factual_handler(
                    kg=kg,
                    bm25=self._bm25,
                    dense=self._dense,
                    registry=self._registry,
                    llm_pool=self._llm_pool,
                    router=self._router,
                    sub_model=self._sub_model,
                )
            if key == "conceptual_definitional":
                return build_conceptual_definitional_handler(
                    kg=kg,
                    bm25=self._bm25,
                    dense=self._dense,
                    registry=self._registry,
                    llm_pool=self._llm_pool,
                    router=self._router,
                    sub_model=self._sub_model,
                )
        # Unreachable when TYPE_TO_HANDLER is exhaustive — guard anyway
        # so a typo in a future key is loud, not silent.
        raise ValueError(f"Unknown dispatcher handler key: {key!r}")

    def _get_kg(self) -> Any:
        if self._kg is None:
            if self._kg_loader is None:
                raise RuntimeError(
                    "Handler requires the KG but neither `kg` nor "
                    "`kg_loader` was supplied to RLMDispatcher."
                )
            log.info("Lazy-loading KG via supplied loader …")
            self._kg = self._kg_loader()
        return self._kg

    # ------------------------------------------------------------------
    # Abstention envelope shared by every error path
    # ------------------------------------------------------------------

    @staticmethod
    def _abstain_envelope(
        *,
        reason: str,
        dispatched_handler: Optional[str],
        dispatched_query_type: Optional[str],
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        telemetry: dict[str, Any] = {
            "retry_count":            0,
            "gate_results":           {},
            "baseline":               DISPATCH_BASELINE,
            "dispatch_baseline":      DISPATCH_BASELINE,
            "dispatched_handler":     dispatched_handler,
            "dispatched_query_type":  dispatched_query_type,
            "sub_call_count":         0,
        }
        if error is not None:
            telemetry["error"] = error
        return {
            "answer_text":       "",
            "abstention":        True,
            "abstention_reason": reason,
            "citations":         [],
            "reasoning_chain":   [
                f"dispatched_query_type={dispatched_query_type}",
                f"dispatched_handler={dispatched_handler}",
                f"reason={reason}",
            ],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry":        telemetry,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dispatcher(
    bm25: BM25Index,
    dense: DenseIndex,
    registry: ArticleRegistry,
    llm_pool,
    *,
    router: Optional[DocRouter] = None,
    kg: Any = None,
    kg_loader: Optional[KGLoaderFn] = None,
    **kwargs: Any,
) -> RLMDispatcher:
    """Factory mirroring the per-handler ``build_*_handler`` helpers."""
    return RLMDispatcher(
        bm25=bm25,
        dense=dense,
        registry=registry,
        llm_pool=llm_pool,
        router=router,
        kg=kg,
        kg_loader=kg_loader,
        **kwargs,
    )
