"""Shared base types for all faithfulness gates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateResult:
    """Uniform return type from every gate.

    passed : True iff the gate accepted the answer.
    score  : 0.0 → completely failed, 1.0 → fully passed.
    details: list of per-failure dicts; each has at least a "reason" key.
    """
    passed: bool
    score: float = 1.0
    details: list[dict[str, Any]] = field(default_factory=list)

    def failure_summary(self) -> str:
        """One-line summary of all failures (for logging / retry hints)."""
        return "; ".join(d.get("reason", str(d)) for d in self.details)
