"""RecursionBudget — depth, token, and sub-call enforcer for the RLM loop."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field


class RecursionBudgetExceeded(RuntimeError):
    """Raised when any RLM budget limit is breached."""


@dataclass
class RecursionBudget:
    """Immutable limits + mutable counters for one query lifecycle.

    Create one per query; pass it into LegalEnv so every sub-call
    charges the same budget object.

    Usage:
        budget = RecursionBudget()
        with budget.descend():
            # depth-1 sub-call
            budget.charge(prompt_text, max_tokens=1024)
    """

    max_depth:       int = 1        # raised to 2 only for multi_hop query type
    max_total_tokens:int = 200_000
    max_sub_calls:   int = 12

    # mutable counters — use field(default=...) so dataclass doesn't complain
    _depth:             int = field(default=0, init=False, repr=False)
    _tokens:            int = field(default=0, init=False, repr=False)
    _sub_calls:         int = field(default=0, init=False, repr=False)
    _max_depth_reached: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    @contextmanager
    def descend(self):
        """Context manager: increment depth for the duration of a sub-call.

        Raises RecursionBudgetExceeded("depth") immediately if at limit.
        sub_calls counter increments on entry (not on exit) so partial
        failures still count against the budget.
        """
        if self._depth >= self.max_depth:
            raise RecursionBudgetExceeded(
                f"Recursion depth limit reached ({self._depth}/{self.max_depth})"
            )
        self._sub_calls += 1
        if self._sub_calls > self.max_sub_calls:
            raise RecursionBudgetExceeded(
                f"Sub-call limit reached ({self._sub_calls}/{self.max_sub_calls})"
            )
        self._depth += 1
        if self._depth > self._max_depth_reached:
            self._max_depth_reached = self._depth
        try:
            yield self
        finally:
            self._depth -= 1

    def charge(self, prompt: str, max_tokens: int) -> None:
        """Deduct estimated tokens from the total budget.

        Estimation: len(prompt) // 4 (rough char-to-token ratio) + max_tokens.
        Raises RecursionBudgetExceeded("tokens") if over budget.
        """
        estimated = len(prompt) // 4 + max_tokens
        self._tokens += estimated
        if self._tokens > self.max_total_tokens:
            raise RecursionBudgetExceeded(
                f"Token budget exceeded ({self._tokens}/{self.max_total_tokens})"
            )

    # ------------------------------------------------------------------
    @property
    def depth(self) -> int:
        return self._depth

    @property
    def tokens_used(self) -> int:
        return self._tokens

    @property
    def sub_calls_used(self) -> int:
        return self._sub_calls

    @property
    def max_depth_reached(self) -> int:
        return self._max_depth_reached

    def reset(self) -> None:
        """Reset mutable counters before a new top-level query."""
        self._depth = 0
        self._tokens = 0
        self._sub_calls = 0
        self._max_depth_reached = 0

    def snapshot(self) -> dict:
        return {
            "depth": self._depth,
            "tokens": self._tokens,
            "sub_calls": self._sub_calls,
            "max_depth_reached": self._max_depth_reached,
            "max_depth": self.max_depth,
            "max_total_tokens": self.max_total_tokens,
            "max_sub_calls": self.max_sub_calls,
        }
