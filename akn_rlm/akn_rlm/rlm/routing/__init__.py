"""Doc-level routing utilities for Phase-2 RLM handlers.

The router predicts 1-3 relevant ``doc_id`` values for each query so that
downstream retrieval can be restricted to those documents. This is the
'doc routing' discipline change called out in §3 of HANDOFF.md.
"""

from akn_rlm.rlm.routing.doc_router import (
    DEFAULT_TOP_N,
    DocRouter,
    RouteResult,
    build_doc_router,
)

__all__ = [
    "DEFAULT_TOP_N",
    "DocRouter",
    "RouteResult",
    "build_doc_router",
]
