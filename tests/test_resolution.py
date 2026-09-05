"""
Tests for the terminal resolution nodes:
    1. HumanReviewInterrupt — EVAL_MODE interrupt bypass
    2. CategorizeException — strict taxonomy + timing float split

Usage:
    python -m pytest tests/test_resolution.py -v
    python -m tests.test_resolution
"""
from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.nodes.resolution import (
    _evaluate_timing_float,
    categorize_exception,
    human_review_interrupt,
)
from app.db.enums import ExceptionCategory, ExceptionSeverity


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════


def _make_cluster(
    cluster_id: str = "cluster_test_001",
    primary_type: str = "order",
    primary_id: str = "order_001",
    candidates: list[dict] | None = None,
    has_amount_collision: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal CandidateCluster dict for testing."""
    if candidates is None:
        candidates = [
            {
                "entity_type": "settlement",
                "entity_id": "setl_001",
                "score": 0.72,
                "amount_paise": 100000,
                "timestamp": "2026-08-20T10:00:00+05:30",
            },
        ]
    result = {
        "cluster_id": cluster_id,
        "primary_entity_type": primary_type,
        "primary_entity_id": primary_id,
        "candidate_matches": candidates,
        "window_start": "2026-08-18T00:00:00+05:30",
        "window_end": "2026-08-22T00:00:00+05:30",
        "aggregate_delta_paise": 0,
        "has_amount_collision": has_amount_collision,
    }
    result.update(extra)
    return result


def _make_base_state(**overrides: Any) -> dict[str, Any]:
    """Build a minimal ClusterState dict for testing."""
    state: dict[str, Any] = {
        "cluster": _make_cluster(),
        "reconciliation_run_id": str(uuid.uuid4()),
        "eval_mode": False,
        "exhausted_models": [],
        "model_chain": ["model-a", "model-b"],
        "messages": [],
        "current_model_index": 0,
        "iteration_count": 0,
        "retry_count": 0,
        "last_error_delta": None,
        "verification_feedback": None,
        "cited_evidence": [],
        "decision": None,
        "verification_result": None,
        "outcome": "",
        "model_used": None,
        "processing_ms": 0,
        "exception_category": None,
        "reasoning_trace": "",
    }
    state.update(overrides)
    return state


# ═══════════════════════════════════════════════════════════
#  TEST 1: EVAL_MODE=true NEVER calls interrupt()
# ═══════════════════════════════════════════════════════════


class TestHumanReviewInterrupt:
    """HumanReviewInterrupt dual-mode tests."""

    def test_eval_mode_true_never_calls_interrupt(self):
        """EVAL_MODE=true must NEVER call langgraph interrupt().

        We patch interrupt() and assert it was never called.
        The result must tag ESCALATED_UNRESOLVED and continue.
        """
        state = _make_base_state(
            eval_mode=True,
            reasoning_trace="prior_trace",
            cited_evidence=[{"order_id": "ord_1"}],
            verification_feedback="some feedback",
        )

        with patch(
            "app.agent.nodes.resolution.interrupt"
        ) as mock_interrupt:
            result = human_review_interrupt(state)

        # interrupt() must NEVER be called in EVAL_MODE
        mock_interrupt.assert_not_called()

        # Must tag ESCALATED_UNRESOLVED
        assert result["outcome"] == "exception"
        assert result["exception_category"] == ExceptionCategory.ESCALATED_UNRESOLVED.value
        assert "EVAL_MODE=true" in result["reasoning_trace"]
        assert "ESCALATED_UNRESOLVED" in result["reasoning_trace"]

    def test_eval_mode_false_calls_interrupt_approved(self):
        """EVAL_MODE=false must call interrupt() and process APPROVED response."""
        state = _make_base_state(
            eval_mode=False,
            reasoning_trace="prior_trace",
            cluster=_make_cluster(cluster_id="clust_42"),
        )

        with patch(
            "app.agent.nodes.resolution.interrupt",
            return_value={"decision": "approved"},
        ) as mock_interrupt:
            result = human_review_interrupt(state)

        # interrupt() MUST be called
        mock_interrupt.assert_called_once()

        # Approved → outcome is verified, NOT exception
        assert result["outcome"] == "verified"
        assert "APPROVED" in result["reasoning_trace"]

    def test_eval_mode_false_calls_interrupt_rejected(self):
        """EVAL_MODE=false must call interrupt() and process REJECTED response."""
        state = _make_base_state(
            eval_mode=False,
            reasoning_trace="prior_trace",
        )

        with patch(
            "app.agent.nodes.resolution.interrupt",
            return_value={"decision": "rejected"},
        ) as mock_interrupt:
            result = human_review_interrupt(state)

        mock_interrupt.assert_called_once()
        assert result["outcome"] == "exception"
        assert result["exception_category"] == ExceptionCategory.ESCALATED_UNRESOLVED.value
        assert "REJECTED" in result["reasoning_trace"]

    def test_eval_mode_false_invalid_response_defaults_rejected(self):
        """Invalid human response defaults to REJECTED."""
        state = _make_base_state(eval_mode=False)

        with patch(
            "app.agent.nodes.resolution.interrupt",
            return_value={"decision": "maybe"},
        ):
            result = human_review_interrupt(state)

        assert result["outcome"] == "exception"
        assert result["exception_category"] == ExceptionCategory.ESCALATED_UNRESOLVED.value

    def test_eval_mode_false_string_response(self):
        """interrupt() can return a plain string too."""
        state = _make_base_state(eval_mode=False)

        with patch(
            "app.agent.nodes.resolution.interrupt",
            return_value="approved",
        ):
            result = human_review_interrupt(state)

        assert result["outcome"] == "verified"

    def test_interrupt_payload_contains_required_fields(self):
        """The payload passed to interrupt() must surface the required fields."""
        run_id = str(uuid.uuid4())
        state = _make_base_state(
            eval_mode=False,
            reconciliation_run_id=run_id,
            cluster=_make_cluster(cluster_id="clust_99"),
            cited_evidence=[{"order_id": "o1"}],
            reasoning_trace="some trace",
            verification_feedback="delta too big",
        )

        with patch(
            "app.agent.nodes.resolution.interrupt",
            return_value={"decision": "approved"},
        ) as mock_interrupt:
            human_review_interrupt(state)

        payload = mock_interrupt.call_args[0][0]
        assert payload["cluster_id"] == "clust_99"
        assert payload["thread_id"] == f"{run_id}:cluster_clust_99"
        assert payload["cited_evidence"] == [{"order_id": "o1"}]
        assert payload["reasoning_trace"] == "some trace"
        assert payload["verification_feedback"] == "delta too big"


# ═══════════════════════════════════════════════════════════
#  TEST 2: CategorizeException — strict enum + timing split
# ═══════════════════════════════════════════════════════════


class TestCategorizeException:
    """CategorizeException strict taxonomy tests."""

    def test_valid_category_passes_through(self):
        """A valid ExceptionCategory value passes strict validation."""
        state = _make_base_state(
            exception_category=ExceptionCategory.GATEWAY_FEE_MISMATCH.value,
        )

        result = categorize_exception(state)

        assert result["outcome"] == "exception"
        assert result["exception_category"] == "GATEWAY_FEE_MISMATCH"

    def test_unknown_category_remapped_to_escalated(self):
        """Unknown freeform category string → ESCALATED_UNRESOLVED."""
        state = _make_base_state(
            exception_category="SOME_WEIRD_CATEGORY",
        )

        result = categorize_exception(state)

        assert result["exception_category"] == ExceptionCategory.ESCALATED_UNRESOLVED.value

    def test_null_category_remapped_to_escalated(self):
        """Null/None category → ESCALATED_UNRESOLVED."""
        state = _make_base_state(
            exception_category=None,
            decision=None,
        )

        result = categorize_exception(state)

        assert result["exception_category"] == ExceptionCategory.ESCALATED_UNRESOLVED.value

    def test_empty_string_category_remapped_to_escalated(self):
        """Empty string category → ESCALATED_UNRESOLVED."""
        state = _make_base_state(
            exception_category="",
        )

        result = categorize_exception(state)

        assert result["exception_category"] == ExceptionCategory.ESCALATED_UNRESOLVED.value

    def test_llm_proposed_category_used_if_no_exception_category(self):
        """If exception_category is None, falls back to LLM's proposed_category."""
        state = _make_base_state(
            exception_category=None,
            decision={
                "decision": "no_match",
                "confidence": 0.3,
                "matched_entity_ids": [],
                "proposed_category": "MISSING_SETTLEMENT_RECORD",
                "reasoning": "no settlement found",
            },
        )

        result = categorize_exception(state)

        assert result["exception_category"] == "MISSING_SETTLEMENT_RECORD"

    def test_all_valid_categories_accepted(self):
        """Every valid ExceptionCategory value must pass validation."""
        for cat in ExceptionCategory:
            state = _make_base_state(exception_category=cat.value)
            result = categorize_exception(state)
            assert result["exception_category"] == cat.value, (
                f"Category {cat.value} was not accepted"
            )

    def test_category_payload_built(self):
        """Each category produces a structured payload."""
        state = _make_base_state(
            exception_category=ExceptionCategory.GATEWAY_FEE_MISMATCH.value,
            verification_result={"delta_paise": 640, "passed": False},
        )

        result = categorize_exception(state)

        assert "exception_payload" in result
        payload = result["exception_payload"]
        assert payload["verification_delta_paise"] == 640


# ═══════════════════════════════════════════════════════════
#  TEST 3: TIMING_SETTLEMENT_FLOAT T+2 split
# ═══════════════════════════════════════════════════════════


class TestTimingFloatSplit:
    """TIMING_SETTLEMENT_FLOAT T+2 business-day split tests."""

    def test_timing_lag_pending_within_t_plus_2(self):
        """≤2 business days → PENDING_FLOAT (is_overdue=False).

        Aug 18 (Tue) → Aug 20 (Thu) = 2 business days → PENDING.
        """
        cluster = _make_cluster(
            order_timestamp="2026-08-18T10:00:00+05:30",
            settlement_value_date="2026-08-20T10:00:00+05:30",
        )
        state = _make_base_state(
            cluster=cluster,
            exception_category=ExceptionCategory.TIMING_SETTLEMENT_FLOAT.value,
        )

        result = categorize_exception(state)

        assert result["exception_category"] == "TIMING_SETTLEMENT_FLOAT"
        assert result["exception_is_overdue"] is False
        payload = result["exception_payload"]
        assert payload["sub_type"] == "PENDING_FLOAT"
        assert payload["is_overdue"] is False
        assert payload["business_days_elapsed"] is not None
        assert payload["business_days_elapsed"] <= 2

    def test_timing_lag_anomalous_exceeds_t_plus_2(self):
        """>2 business days → ANOMALOUS_FLOAT_OVERDUE (is_overdue=True).

        Aug 18 (Tue) → Aug 25 (Tue) = 5 business days → OVERDUE.
        """
        cluster = _make_cluster(
            order_timestamp="2026-08-18T10:00:00+05:30",
            settlement_value_date="2026-08-25T10:00:00+05:30",
        )
        state = _make_base_state(
            cluster=cluster,
            exception_category=ExceptionCategory.TIMING_SETTLEMENT_FLOAT.value,
        )

        result = categorize_exception(state)

        assert result["exception_category"] == "TIMING_SETTLEMENT_FLOAT"
        assert result["exception_is_overdue"] is True
        payload = result["exception_payload"]
        assert payload["sub_type"] == "ANOMALOUS_FLOAT_OVERDUE"
        assert payload["is_overdue"] is True
        assert payload["business_days_elapsed"] is not None
        assert payload["business_days_elapsed"] > 2

    def test_timing_lag_exact_boundary_t_plus_2(self):
        """Exactly 2 business days → PENDING_FLOAT (boundary case).

        Aug 19 (Wed) → Aug 21 (Fri) = exactly 2 business days.
        """
        cluster = _make_cluster(
            order_timestamp="2026-08-19T10:00:00+05:30",
            settlement_value_date="2026-08-21T10:00:00+05:30",
        )
        state = _make_base_state(
            cluster=cluster,
            exception_category=ExceptionCategory.TIMING_SETTLEMENT_FLOAT.value,
        )

        result = categorize_exception(state)

        assert result["exception_is_overdue"] is False
        payload = result["exception_payload"]
        assert payload["sub_type"] == "PENDING_FLOAT"
        assert payload["business_days_elapsed"] == 2

    def test_timing_lag_just_over_boundary(self):
        """3 business days → ANOMALOUS_FLOAT_OVERDUE (just past boundary).

        Aug 19 (Wed) → Aug 24 (Mon) = 3 business days.
        """
        cluster = _make_cluster(
            order_timestamp="2026-08-19T10:00:00+05:30",
            settlement_value_date="2026-08-24T10:00:00+05:30",
        )
        state = _make_base_state(
            cluster=cluster,
            exception_category=ExceptionCategory.TIMING_SETTLEMENT_FLOAT.value,
        )

        result = categorize_exception(state)

        assert result["exception_is_overdue"] is True
        payload = result["exception_payload"]
        assert payload["sub_type"] == "ANOMALOUS_FLOAT_OVERDUE"
        assert payload["business_days_elapsed"] == 3

    def test_timing_lag_same_day(self):
        """Same day → 0 business days → PENDING_FLOAT."""
        cluster = _make_cluster(
            order_timestamp="2026-08-19T08:00:00+05:30",
            settlement_value_date="2026-08-19T18:00:00+05:30",
        )
        state = _make_base_state(
            cluster=cluster,
            exception_category=ExceptionCategory.TIMING_SETTLEMENT_FLOAT.value,
        )

        result = categorize_exception(state)

        assert result["exception_is_overdue"] is False
        payload = result["exception_payload"]
        assert payload["sub_type"] == "PENDING_FLOAT"
        assert payload["business_days_elapsed"] == 0

    def test_timing_lag_spans_weekend(self):
        """Span including weekend: Fri → next Tue.

        Aug 21 (Fri) → Aug 25 (Tue) = 2 business days (Mon+Tue) → PENDING.
        """
        cluster = _make_cluster(
            order_timestamp="2026-08-21T10:00:00+05:30",
            settlement_value_date="2026-08-25T10:00:00+05:30",
        )
        state = _make_base_state(
            cluster=cluster,
            exception_category=ExceptionCategory.TIMING_SETTLEMENT_FLOAT.value,
        )

        result = categorize_exception(state)

        assert result["exception_is_overdue"] is False
        payload = result["exception_payload"]
        assert payload["sub_type"] == "PENDING_FLOAT"

    def test_non_timing_category_has_null_is_overdue(self):
        """Non-TIMING_SETTLEMENT_FLOAT categories have is_overdue=None."""
        state = _make_base_state(
            exception_category=ExceptionCategory.GATEWAY_FEE_MISMATCH.value,
        )

        result = categorize_exception(state)

        assert result["exception_is_overdue"] is None

    def test_timing_float_severity_escalation(self):
        """Overdue float escalates severity from LOW to MEDIUM."""
        cluster = _make_cluster(
            order_timestamp="2026-08-18T10:00:00+05:30",
            settlement_value_date="2026-08-25T10:00:00+05:30",
        )
        state = _make_base_state(
            cluster=cluster,
            exception_category=ExceptionCategory.TIMING_SETTLEMENT_FLOAT.value,
        )

        result = categorize_exception(state)

        assert result["exception_severity"] == ExceptionSeverity.MEDIUM.value


# ═══════════════════════════════════════════════════════════
#  TEST 4: _evaluate_timing_float directly
# ═══════════════════════════════════════════════════════════


class TestEvaluateTimingFloat:
    """Direct tests for _evaluate_timing_float helper."""

    def test_missing_timestamps_defaults_overdue(self):
        """Missing timestamps → conservative default to ANOMALOUS_FLOAT_OVERDUE."""
        cluster = _make_cluster()
        # No order_timestamp or settlement_value_date set

        result = _evaluate_timing_float(cluster)

        assert result["is_overdue"] is True
        assert result["sub_type"] == "ANOMALOUS_FLOAT_OVERDUE"

    def test_date_objects_accepted(self):
        """Plain date objects work as timestamps."""
        cluster = _make_cluster(
            order_timestamp=date(2026, 8, 19),
            settlement_value_date=date(2026, 8, 21),
        )

        result = _evaluate_timing_float(cluster)

        assert result["is_overdue"] is False
        assert result["biz_days"] == 2

    def test_datetime_objects_accepted(self):
        """Full datetime objects work too."""
        cluster = _make_cluster(
            order_timestamp=datetime(2026, 8, 19, 10, 0, tzinfo=timezone.utc),
            settlement_value_date=datetime(2026, 8, 21, 10, 0, tzinfo=timezone.utc),
        )

        result = _evaluate_timing_float(cluster)

        assert result["is_overdue"] is False


# ═══════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════


def _run_all_tests():
    """Run all tests manually (no pytest required)."""
    passed = 0
    failed = 0
    errors: list[str] = []

    test_classes = [
        TestHumanReviewInterrupt,
        TestCategorizeException,
        TestTimingFloatSplit,
        TestEvaluateTimingFloat,
    ]

    for cls in test_classes:
        instance = cls()
        for attr in sorted(dir(instance)):
            if not attr.startswith("test_"):
                continue
            method = getattr(instance, attr)
            test_name = f"{cls.__name__}.{attr}"
            try:
                method()
                passed += 1
                print(f"  ✓ {test_name}")
            except Exception as e:
                failed += 1
                errors.append(f"  ✗ {test_name}: {e}")
                print(f"  ✗ {test_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for err in errors:
            print(err)
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = _run_all_tests()
    sys.exit(0 if success else 1)
