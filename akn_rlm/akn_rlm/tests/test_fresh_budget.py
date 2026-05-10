"""Tests for RecursionBudget reset semantics."""
import pytest
from akn_rlm.rlm.recursion_budget import RecursionBudget, RecursionBudgetExceeded


def test_fresh_budget_counters_start_at_zero():
    b = RecursionBudget()
    assert b.tokens_used == 0
    assert b.sub_calls_used == 0
    assert b.max_depth_reached == 0
    assert b.depth == 0


def test_reset_clears_all_counters():
    b = RecursionBudget()
    b.charge("some prompt text" * 100, max_tokens=512)
    with b.descend():
        pass
    assert b.tokens_used > 0
    assert b.sub_calls_used > 0

    b.reset()
    assert b.tokens_used == 0
    assert b.sub_calls_used == 0
    assert b.max_depth_reached == 0
    assert b.depth == 0


def test_reset_does_not_change_limits():
    b = RecursionBudget(max_depth=2, max_sub_calls=5, max_total_tokens=50_000)
    b.reset()
    assert b.max_depth == 2
    assert b.max_sub_calls == 5
    assert b.max_total_tokens == 50_000


def test_max_depth_raised_after_descend():
    b = RecursionBudget(max_depth=2)
    with b.descend():
        pass
    assert b.max_depth_reached == 1
    b.reset()
    assert b.max_depth_reached == 0


def test_sub_calls_counted():
    b = RecursionBudget(max_sub_calls=10)
    for _ in range(3):
        with b.descend():
            pass
    assert b.sub_calls_used == 3


def test_depth_limit_enforced():
    b = RecursionBudget(max_depth=1)
    with b.descend():
        with pytest.raises(RecursionBudgetExceeded, match="depth"):
            with b.descend():
                pass


def test_sub_call_limit_enforced():
    b = RecursionBudget(max_sub_calls=2)
    with b.descend():
        pass
    with b.descend():
        pass
    with pytest.raises(RecursionBudgetExceeded, match="Sub-call"):
        with b.descend():
            pass


def test_token_limit_enforced():
    b = RecursionBudget(max_total_tokens=100)
    with pytest.raises(RecursionBudgetExceeded, match="Token"):
        b.charge("x" * 404, max_tokens=1)  # 404//4=101 + 1 = 102 → over 100


def test_reset_between_queries_simulated():
    """Simulates two sequential benchmark queries sharing the same budget."""
    b = RecursionBudget(max_sub_calls=12, max_total_tokens=200_000)

    # Query 1
    b.reset()
    b.charge("prompt text" * 50, max_tokens=1024)
    with b.descend():
        pass
    tokens_after_q1 = b.tokens_used
    sub_after_q1 = b.sub_calls_used

    # Query 2 — must start fresh
    b.reset()
    assert b.tokens_used == 0, "Tokens must reset between queries"
    assert b.sub_calls_used == 0, "Sub-calls must reset between queries"
