"""Tests for RecursionBudget — Phase C."""
from __future__ import annotations

import pytest
from akn_rlm.rlm.recursion_budget import RecursionBudget, RecursionBudgetExceeded


class TestDepthEnforcement:
    def test_single_descent_ok(self):
        b = RecursionBudget(max_depth=1)
        with b.descend():
            assert b.depth == 1
        assert b.depth == 0

    def test_depth_restored_on_exit(self):
        b = RecursionBudget(max_depth=1)
        with b.descend():
            pass
        assert b.depth == 0

    def test_exceeds_depth_raises(self):
        b = RecursionBudget(max_depth=1)
        with pytest.raises(RecursionBudgetExceeded, match="depth"):
            with b.descend():
                with b.descend():   # depth-2: must raise
                    pass

    def test_multihop_depth_2_ok(self):
        b = RecursionBudget(max_depth=2)
        with b.descend():
            with b.descend():
                assert b.depth == 2

    def test_depth_3_raises_even_with_max_2(self):
        b = RecursionBudget(max_depth=2)
        with pytest.raises(RecursionBudgetExceeded):
            with b.descend():
                with b.descend():
                    with b.descend():
                        pass

    def test_depth_restored_after_exception(self):
        b = RecursionBudget(max_depth=2)
        try:
            with b.descend():
                with b.descend():
                    with b.descend():
                        pass
        except RecursionBudgetExceeded:
            pass
        # depth should be back to 0 after all context managers unwind
        assert b.depth == 0


class TestSubCallLimit:
    def test_sub_calls_counted(self):
        b = RecursionBudget(max_depth=1, max_sub_calls=3)
        for _ in range(3):
            with b.descend():
                pass
        assert b.sub_calls_used == 3

    def test_exceeds_sub_calls_raises(self):
        b = RecursionBudget(max_depth=1, max_sub_calls=2)
        with b.descend():
            pass
        with b.descend():
            pass
        with pytest.raises(RecursionBudgetExceeded, match="sub_calls|[Ss]ub"):
            with b.descend():
                pass


class TestTokenBudget:
    def test_charge_accumulates(self):
        b = RecursionBudget(max_total_tokens=10_000)
        b.charge("hello", 100)
        assert b.tokens_used > 0

    def test_charge_over_limit_raises(self):
        b = RecursionBudget(max_total_tokens=100)
        with pytest.raises(RecursionBudgetExceeded, match="[Tt]oken"):
            # prompt of 400 chars ≈ 100 tokens + max_tokens=200 > 100
            b.charge("x" * 400, 200)

    def test_charge_within_limit_ok(self):
        b = RecursionBudget(max_total_tokens=200_000)
        b.charge("short prompt", 512)
        assert b.tokens_used < 200_000


class TestSnapshot:
    def test_snapshot_fields(self):
        b = RecursionBudget(max_depth=1, max_total_tokens=50_000, max_sub_calls=5)
        snap = b.snapshot()
        assert snap["max_depth"] == 1
        assert snap["max_total_tokens"] == 50_000
        assert snap["max_sub_calls"] == 5
        assert snap["depth"] == 0
        assert snap["sub_calls"] == 0
        assert snap["tokens"] == 0


class TestReset:
    def test_reset_clears_counters(self):
        b = RecursionBudget(max_depth=2, max_total_tokens=50_000, max_sub_calls=5)
        b.charge("hello", 100)
        with b.descend():
            pass
        assert b.tokens_used > 0
        assert b.sub_calls_used == 1
        assert b.max_depth_reached == 1

        b.reset()

        assert b.depth == 0
        assert b.tokens_used == 0
        assert b.sub_calls_used == 0
        assert b.max_depth_reached == 0
