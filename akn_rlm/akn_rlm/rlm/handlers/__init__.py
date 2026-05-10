"""Phase-2 typed per-query-type handlers.

Each handler is a small state machine — the LLM fills slots, doesn't
write code. Handlers expose a uniform ``.run(query) -> dict`` contract
shaped exactly like the deterministic baselines so
:func:`akn_rlm.eval.runner._answer_to_result` can consume the output
unchanged.
"""

from akn_rlm.rlm.handlers.multi_hop import (
    DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH,
    DEFAULT_MAX_SUB_QS,
    DEFAULT_ROUTE_TOP_N,
    DEFAULT_TOP_K_PER_SUBQ,
    DEFAULT_VERIFY_TOP_N,
    DEFAULT_VERIFY_THRESHOLD,
    MultiHopHandler,
    build_multi_hop_handler,
)
from akn_rlm.rlm.handlers.temporal_factual import (
    DEFAULT_TOP_K_CANDIDATES as TEMPORAL_DEFAULT_TOP_K_CANDIDATES,
    LATEST_VERSION_DATE,
    TemporalFactualHandler,
    build_temporal_factual_handler,
)
from akn_rlm.rlm.handlers.conceptual_definitional import (
    DEFAULT_ADU_EXTRACT_TOP_N as CONCEPTUAL_DEFAULT_ADU_EXTRACT_TOP_N,
    DEFAULT_PARAPHRASE_COUNT as CONCEPTUAL_DEFAULT_PARAPHRASE_COUNT,
    DEFAULT_TOP_K_CANDIDATES as CONCEPTUAL_DEFAULT_TOP_K_CANDIDATES,
    ConceptualDefinitionalHandler,
    build_conceptual_definitional_handler,
)
from akn_rlm.rlm.handlers.unanswerable import (
    DEFAULT_K_EACH as UNANSWERABLE_DEFAULT_K_EACH,
    DEFAULT_TOP_K_CANDIDATES as UNANSWERABLE_DEFAULT_TOP_K_CANDIDATES,
    DEFAULT_WEAK_EVIDENCE_THRESHOLD as UNANSWERABLE_DEFAULT_WEAK_EVIDENCE_THRESHOLD,
    UnanswerableHandler,
    build_unanswerable_handler,
    detect_infection_signals,
)
from akn_rlm.rlm.handlers.rule_application import (
    DEFAULT_FINAL_TOP_K as RULE_APP_DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH as RULE_APP_DEFAULT_K_EACH,
    DEFAULT_TOP_K_CANDIDATES as RULE_APP_DEFAULT_TOP_K_CANDIDATES,
    DEFAULT_VERIFY_THRESHOLD as RULE_APP_DEFAULT_VERIFY_THRESHOLD,
    RuleApplicationHandler,
    build_rule_application_handler,
)
from akn_rlm.rlm.handlers.exact_article import (
    DEFAULT_BM25_K as EXACT_ART_DEFAULT_BM25_K,
    DEFAULT_FINAL_TOP_K as EXACT_ART_DEFAULT_FINAL_TOP_K,
    DEFAULT_TOP_K_CANDIDATES as EXACT_ART_DEFAULT_TOP_K_CANDIDATES,
    DEFAULT_VERIFY_THRESHOLD as EXACT_ART_DEFAULT_VERIFY_THRESHOLD,
    ExactArticleHandler,
    build_exact_article_handler,
)
from akn_rlm.rlm.handlers.layman import (
    DEFAULT_REWRITE_MODEL as LAYMAN_DEFAULT_REWRITE_MODEL,
    LaymanHandler,
    build_layman_handler,
    call_darja_rewriter,
)
from akn_rlm.rlm.handlers.long_context import (
    DEFAULT_FINAL_TOP_K as LONG_CTX_DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH as LONG_CTX_DEFAULT_K_EACH,
    LongContextHandler,
    build_long_context_handler,
)

__all__ = [
    "CONCEPTUAL_DEFAULT_ADU_EXTRACT_TOP_N",
    "CONCEPTUAL_DEFAULT_PARAPHRASE_COUNT",
    "CONCEPTUAL_DEFAULT_TOP_K_CANDIDATES",
    "ConceptualDefinitionalHandler",
    "DEFAULT_FINAL_TOP_K",
    "DEFAULT_K_EACH",
    "DEFAULT_MAX_SUB_QS",
    "DEFAULT_ROUTE_TOP_N",
    "DEFAULT_TOP_K_PER_SUBQ",
    "DEFAULT_VERIFY_TOP_N",
    "DEFAULT_VERIFY_THRESHOLD",
    "EXACT_ART_DEFAULT_BM25_K",
    "EXACT_ART_DEFAULT_FINAL_TOP_K",
    "EXACT_ART_DEFAULT_TOP_K_CANDIDATES",
    "EXACT_ART_DEFAULT_VERIFY_THRESHOLD",
    "ExactArticleHandler",
    "LATEST_VERSION_DATE",
    "LAYMAN_DEFAULT_REWRITE_MODEL",
    "LONG_CTX_DEFAULT_FINAL_TOP_K",
    "LONG_CTX_DEFAULT_K_EACH",
    "LaymanHandler",
    "LongContextHandler",
    "MultiHopHandler",
    "RULE_APP_DEFAULT_FINAL_TOP_K",
    "RULE_APP_DEFAULT_K_EACH",
    "RULE_APP_DEFAULT_TOP_K_CANDIDATES",
    "RULE_APP_DEFAULT_VERIFY_THRESHOLD",
    "RuleApplicationHandler",
    "TEMPORAL_DEFAULT_TOP_K_CANDIDATES",
    "TemporalFactualHandler",
    "UNANSWERABLE_DEFAULT_K_EACH",
    "UNANSWERABLE_DEFAULT_TOP_K_CANDIDATES",
    "UNANSWERABLE_DEFAULT_WEAK_EVIDENCE_THRESHOLD",
    "UnanswerableHandler",
    "build_conceptual_definitional_handler",
    "build_exact_article_handler",
    "build_layman_handler",
    "build_long_context_handler",
    "build_multi_hop_handler",
    "build_rule_application_handler",
    "build_temporal_factual_handler",
    "build_unanswerable_handler",
    "call_darja_rewriter",
    "detect_infection_signals",
]
