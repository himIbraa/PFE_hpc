"""Tests for decomposer trigger and IntentResult in pipeline state."""
import pytest
from unittest.mock import MagicMock, patch


def _make_env():
    env = MagicMock()
    from akn_rlm.rlm.recursion_budget import RecursionBudget
    env.budget = RecursionBudget()
    env.budget.reset = MagicMock(side_effect=env.budget.reset)
    env.begin_query = MagicMock()
    env.llm_pool = MagicMock()
    env._degraded_retrievers = []
    return env


# ---------------------------------------------------------------------------
# Classifier routing for decompose trigger
# ---------------------------------------------------------------------------

def test_multi_hop_triggers_decompose():
    from akn_rlm.rlm.classifier import classify
    r = classify("ما العلاقة بين قانون الأسرة والقانون المدني؟")
    assert r.query_type == "multi_hop"


def test_long_context_triggers_decompose():
    from akn_rlm.rlm.classifier import classify
    r = classify("اشرح الإجراءات الكاملة للطلاق خطوة بخطوة")
    assert r.query_type == "long_context"


def test_rule_application_does_not_trigger_decompose():
    from akn_rlm.rlm.classifier import classify
    r = classify("هل يحق للمستأجر طلب تخفيض الإيجار؟")
    assert r.query_type not in ("multi_hop", "long_context")


# ---------------------------------------------------------------------------
# Decomposer result stored in pipeline state
# ---------------------------------------------------------------------------

def test_decomposer_result_key_populated():
    """node_decompose must store the full decomposer dict, not just sub_questions."""
    from akn_rlm.rlm.pipeline import build_pipeline

    env = _make_env()
    decomposer_output = {
        "sub_questions": [
            {"id": "sq1", "text": "أولاً", "type": "rule_application",
             "target_codes": ["84-11"], "requires_temporal": False},
        ],
        "dependency_order": ["sq1"],
        "max_depth_needed": 1,
    }

    with patch("akn_rlm.rlm.sub_worker.call_decomposer", return_value=decomposer_output), \
         patch("akn_rlm.rlm.root_controller.RootController.run",
               return_value={
                   "answer_text": "test", "abstention": False,
                   "citations": [], "reasoning_chain": [],
                   "trajectory": [], "tokens_used": 0,
                   "depth_max_reached": 0, "sub_call_count": 0,
               }), \
         patch("akn_rlm.gates.citation_existence.run_gate",
               return_value=MagicMock(passed=True, score=1.0, details=[])), \
         patch("akn_rlm.gates.jurisdiction.run_gate",
               return_value=MagicMock(passed=True, score=1.0, details=[])), \
         patch("akn_rlm.gates.faithfulness_nli.run_gate",
               return_value=MagicMock(passed=True, score=1.0, details=[])):

        pipeline = build_pipeline(env)
        # Inject a multi_hop query so decompose node is triggered
        from akn_rlm.rlm.pipeline import PipelineState
        initial: PipelineState = {
            "query": "ما العلاقة بين قانون الأسرة والقانون المدني؟",
            "retry_count": 0,
            "error": None,
        }
        result = pipeline.invoke(initial)
        assert "decomposer_result" in result
        assert "sub_questions" in result.get("decomposer_result", {})


# ---------------------------------------------------------------------------
# max_depth raised for multi_hop and long_context
# ---------------------------------------------------------------------------

def test_budget_max_depth_set_to_2_for_multi_hop():
    """node_root_rlm_run must set max_depth=2 for multi_hop queries."""
    from akn_rlm.rlm.pipeline import build_pipeline
    env = _make_env()

    captured_depth = []

    original_reset = env.budget.reset.__wrapped__ if hasattr(env.budget.reset, '__wrapped__') else None

    def _capture_depth():
        captured_depth.append(env.budget.max_depth)

    env.budget.reset = _capture_depth

    with patch("akn_rlm.rlm.root_controller.RootController.run",
               return_value={
                   "answer_text": "ok", "abstention": False,
                   "citations": [], "reasoning_chain": [],
                   "trajectory": [], "tokens_used": 0,
                   "depth_max_reached": 0, "sub_call_count": 0,
               }), \
         patch("akn_rlm.rlm.sub_worker.call_decomposer",
               return_value={"sub_questions": [], "dependency_order": [], "max_depth_needed": 2}), \
         patch("akn_rlm.gates.citation_existence.run_gate",
               return_value=MagicMock(passed=True, score=1.0, details=[])), \
         patch("akn_rlm.gates.jurisdiction.run_gate",
               return_value=MagicMock(passed=True, score=1.0, details=[])), \
         patch("akn_rlm.gates.faithfulness_nli.run_gate",
               return_value=MagicMock(passed=True, score=1.0, details=[])):

        pipeline = build_pipeline(env)
        from akn_rlm.rlm.pipeline import PipelineState
        initial: PipelineState = {
            "query": "ما العلاقة بين قانون الأسرة والقانون المدني؟",
            "retry_count": 0,
            "error": None,
        }
        pipeline.invoke(initial)

    assert any(d == 2 for d in captured_depth), (
        f"Expected max_depth=2 for multi_hop, captured: {captured_depth}"
    )
