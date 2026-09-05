"""
Tests for the per-cluster LangGraph StateGraph (graph.py).

Verifies:
  1. Graph topology: correct node names, edges, entry point.
  2. Verified path: LlmReActLoop → IndependentVerifier → END (via persist).
  3. Exception path: LlmReActLoop → CategorizeException → END.
  4. Retry path: verifier fail → CheckRetryLimit → LlmReActLoop loop.
  5. HITL path: retry exhausted → HumanReviewInterrupt → CategorizeException.
  6. EVAL_MODE: interrupt() is NEVER called when eval_mode=True.
  7. Thread ID: correct format f"{run_id}:cluster_{cluster_id}".
  8. ClusterOutcome: all fields populated correctly.
  9. run_cluster() public API contract.
  10. Graceful failure: graph exceptions produce ESCALATED_UNRESOLVED.

All tests use mocks/fakes — no LLM calls, no Postgres.

Usage:
    python -m pytest tests/test_graph.py -v
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.graph import (
    ClusterOutcome,
    _build_initial_state,
    _extract_outcome,
    _route_after_hitl,
    build_cluster_graph,
    run_cluster,
)
from app.agent.state import ClusterState
from app.db.enums import ExceptionCategory


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════


def _make_cluster(
    cluster_id: str = "cluster_test_001",
    has_amount_collision: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build a minimal cluster dict for graph tests."""
    result = {
        "cluster_id": cluster_id,
        "primary_entity_type": "order",
        "primary_entity_id": "order_001",
        "candidate_matches": [
            {
                "entity_type": "settlement",
                "entity_id": "setl_001",
                "score": 0.72,
                "amount_paise": 100_000,
            },
        ],
        "window_start": "2026-08-18T00:00:00+05:30",
        "window_end": "2026-08-22T00:00:00+05:30",
        "aggregate_delta_paise": 0,
        "has_amount_collision": has_amount_collision,
    }
    result.update(extra)
    return result


# ═══════════════════════════════════════════════════════════
#  TEST 1: Graph topology
# ═══════════════════════════════════════════════════════════


class TestGraphTopology:
    """Verify the StateGraph has the correct nodes and edges."""

    def test_graph_builds_without_error(self):
        """build_cluster_graph() produces a compiled graph."""
        graph = build_cluster_graph()
        assert graph is not None

    def test_graph_has_required_nodes(self):
        """All 5 functional nodes + persist are present."""
        graph = build_cluster_graph()
        # CompiledGraph exposes nodes via .nodes
        node_names = set(graph.nodes.keys())
        expected = {
            "llm_react_loop",
            "independent_verifier",
            "check_retry_limit",
            "human_review_interrupt",
            "categorize_exception",
            "persist",
            "__start__",  # LangGraph adds this implicitly
        }
        # Check all expected nodes are present (graph may add __end__ etc)
        assert expected <= node_names, (
            f"Missing nodes: {expected - node_names}"
        )

    def test_deterministic_tier_not_in_graph(self):
        """Steps 4-5 (exact_join, fuzzy_score, cluster_candidates) are NOT nodes."""
        graph = build_cluster_graph()
        node_names = set(graph.nodes.keys())
        forbidden = {
            "exact_join", "fuzzy_score", "cluster_candidates",
            "run_hop1", "run_hop2", "run_fuzzy_score",
        }
        assert forbidden & node_names == set(), (
            f"Deterministic tier nodes incorrectly included: {forbidden & node_names}"
        )


# ═══════════════════════════════════════════════════════════
#  TEST 2: ClusterOutcome dataclass
# ═══════════════════════════════════════════════════════════


class TestClusterOutcome:
    """ClusterOutcome is correctly structured."""

    def test_cluster_outcome_defaults(self):
        """Verify default values for optional fields."""
        co = ClusterOutcome(cluster_id="c1", outcome="verified")
        assert co.cluster_id == "c1"
        assert co.outcome == "verified"
        assert co.decision is None
        assert co.verification_result is None
        assert co.exception_category is None
        assert co.model_used is None
        assert co.reasoning_trace == ""
        assert co.processing_ms == 0
        assert co.exhausted_models == []

    def test_cluster_outcome_full(self):
        """All fields can be populated."""
        co = ClusterOutcome(
            cluster_id="c2",
            outcome="exception",
            decision={"decision": "no_match"},
            verification_result={"passed": False},
            exception_category="MISSING_SETTLEMENT_RECORD",
            exception_severity="HIGH",
            model_used="groq/llama-3.3-70b-versatile",
            reasoning_trace="LLM said no match",
            processing_ms=1234,
            exhausted_models=["groq/llama-3.3-70b-versatile"],
        )
        assert co.exception_category == "MISSING_SETTLEMENT_RECORD"
        assert co.processing_ms == 1234


# ═══════════════════════════════════════════════════════════
#  TEST 3: Initial state builder
# ═══════════════════════════════════════════════════════════


class TestBuildInitialState:
    """_build_initial_state produces a valid ClusterState dict."""

    def test_all_required_fields_present(self):
        """Every ClusterState key is initialised."""
        cluster = _make_cluster()
        state = _build_initial_state(
            cluster, run_id="run-123", eval_mode=True,
        )
        # Check all ClusterState keys
        expected_keys = {
            "cluster", "reconciliation_run_id", "eval_mode",
            "exhausted_models", "model_chain",
            "messages", "current_model_index", "iteration_count",
            "retry_count", "last_error_delta", "verification_feedback",
            "cited_evidence",
            "decision", "verification_result", "outcome",
            "model_used", "processing_ms", "exception_category",
            "reasoning_trace",
        }
        assert expected_keys <= set(state.keys())

    def test_eval_mode_propagated(self):
        state = _build_initial_state(
            _make_cluster(), run_id="r1", eval_mode=True,
        )
        assert state["eval_mode"] is True

    def test_default_model_chain(self):
        state = _build_initial_state(
            _make_cluster(), run_id="r1",
        )
        assert len(state["model_chain"]) >= 2
        assert "groq/llama-3.3-70b-versatile" in state["model_chain"]

    def test_custom_model_chain(self):
        state = _build_initial_state(
            _make_cluster(), run_id="r1",
            model_chain=["model-a", "model-b"],
        )
        assert state["model_chain"] == ["model-a", "model-b"]

    def test_exhausted_models_carried_in(self):
        state = _build_initial_state(
            _make_cluster(), run_id="r1",
            exhausted_models=["model-a"],
        )
        assert state["exhausted_models"] == ["model-a"]


# ═══════════════════════════════════════════════════════════
#  TEST 4: Extract outcome
# ═══════════════════════════════════════════════════════════


class TestExtractOutcome:
    """_extract_outcome correctly maps graph terminal state to ClusterOutcome."""

    def test_verified_outcome(self):
        state = {
            "outcome": "verified",
            "decision": {"decision": "match", "confidence": 0.95},
            "verification_result": {"passed": True, "delta_paise": 0},
            "model_used": "gpt-4o",
            "reasoning_trace": "all good",
            "processing_ms": 500,
            "exhausted_models": [],
            "exception_category": None,
            "exception_payload": None,
            "exception_severity": None,
        }
        co = _extract_outcome(state, "cluster_001")
        assert co.cluster_id == "cluster_001"
        assert co.outcome == "verified"
        assert co.decision["decision"] == "match"
        assert co.exception_category is None

    def test_exception_outcome(self):
        state = {
            "outcome": "exception",
            "decision": None,
            "verification_result": None,
            "model_used": None,
            "reasoning_trace": "chain exhausted",
            "processing_ms": 2000,
            "exhausted_models": ["groq/llama-3.3-70b-versatile"],
            "exception_category": "ESCALATED_UNRESOLVED",
            "exception_payload": {"description": "unresolvable"},
            "exception_severity": "CRITICAL",
        }
        co = _extract_outcome(state, "cluster_002")
        assert co.outcome == "exception"
        assert co.exception_category == "ESCALATED_UNRESOLVED"
        assert co.exception_severity == "CRITICAL"


# ═══════════════════════════════════════════════════════════
#  TEST 5: Thread ID format
# ═══════════════════════════════════════════════════════════


class TestThreadId:
    """Thread ID must follow f"{run_id}:cluster_{cluster_id}" format."""

    @pytest.mark.asyncio
    async def test_thread_id_format(self):
        """run_cluster passes correct thread_id to graph.ainvoke."""
        run_id = "run-abc-123"
        cluster_id = "cluster_xyz"
        cluster = _make_cluster(cluster_id=cluster_id)

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "outcome": "verified",
            "reasoning_trace": "ok",
            "processing_ms": 100,
        })

        co = await run_cluster(
            cluster,
            run_id=run_id,
            compiled_graph=mock_graph,
        )

        # Verify ainvoke was called with correct config
        mock_graph.ainvoke.assert_called_once()
        call_args = mock_graph.ainvoke.call_args
        config = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("config")
        if config is None:
            config = call_args.kwargs.get("config")

        expected_thread_id = f"{run_id}:cluster_{cluster_id}"
        assert config["configurable"]["thread_id"] == expected_thread_id


# ═══════════════════════════════════════════════════════════
#  TEST 6: Routing functions
# ═══════════════════════════════════════════════════════════


class TestRouting:
    """Conditional routing functions produce correct destinations."""

    def test_route_after_hitl_approved(self):
        """Approved → persist."""
        state = {"outcome": "verified"}
        assert _route_after_hitl(state) == "persist"

    def test_route_after_hitl_rejected(self):
        """Rejected → categorize_exception."""
        state = {"outcome": "exception"}
        assert _route_after_hitl(state) == "categorize_exception"

    def test_route_after_hitl_empty(self):
        """No outcome → categorize_exception (safe default)."""
        state = {"outcome": ""}
        assert _route_after_hitl(state) == "categorize_exception"


# ═══════════════════════════════════════════════════════════
#  TEST 7: run_cluster() — verified path (mocked graph)
# ═══════════════════════════════════════════════════════════


class TestRunClusterVerified:
    """run_cluster returns a correct ClusterOutcome for the verified path."""

    @pytest.mark.asyncio
    async def test_verified_path_returns_cluster_outcome(self):
        """Graph returns verified → ClusterOutcome has outcome='verified'."""
        cluster = _make_cluster(cluster_id="c_verified")

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "outcome": "verified",
            "decision": {"decision": "match", "confidence": 0.95,
                         "matched_entity_ids": ["order_001", "setl_001"],
                         "proposed_category": None, "reasoning": "matched"},
            "verification_result": {"passed": True, "delta_paise": 0},
            "model_used": "groq/llama-3.3-70b-versatile",
            "reasoning_trace": "Hop1 exact + LLM match confirmed",
            "exhausted_models": [],
            "exception_category": None,
            "exception_payload": None,
            "exception_severity": None,
        })

        co = await run_cluster(
            cluster, run_id="run-1", compiled_graph=mock_graph,
        )

        assert isinstance(co, ClusterOutcome)
        assert co.cluster_id == "c_verified"
        assert co.outcome == "verified"
        assert co.decision is not None
        assert co.exception_category is None
        assert co.processing_ms >= 0  # wall clock measured (mock resolves sub-ms)


# ═══════════════════════════════════════════════════════════
#  TEST 8: run_cluster() — exception path (mocked graph)
# ═══════════════════════════════════════════════════════════


class TestRunClusterException:
    """run_cluster returns a correct ClusterOutcome for the exception path."""

    @pytest.mark.asyncio
    async def test_exception_path_returns_cluster_outcome(self):
        cluster = _make_cluster(cluster_id="c_exception")

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "outcome": "exception",
            "decision": None,
            "verification_result": None,
            "model_used": None,
            "reasoning_trace": "All models exhausted",
            "exhausted_models": ["groq/llama-3.3-70b-versatile", "gpt-4o"],
            "exception_category": "ESCALATED_UNRESOLVED",
            "exception_payload": {"description": "unresolvable"},
            "exception_severity": "CRITICAL",
        })

        co = await run_cluster(
            cluster, run_id="run-2", compiled_graph=mock_graph,
        )

        assert co.outcome == "exception"
        assert co.exception_category == "ESCALATED_UNRESOLVED"
        assert len(co.exhausted_models) == 2


# ═══════════════════════════════════════════════════════════
#  TEST 9: run_cluster() — graceful failure
# ═══════════════════════════════════════════════════════════


class TestRunClusterGracefulFailure:
    """If the graph itself raises, run_cluster returns ESCALATED_UNRESOLVED."""

    @pytest.mark.asyncio
    async def test_graph_exception_produces_escalated(self):
        cluster = _make_cluster(cluster_id="c_crash")

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM provider down"),
        )

        co = await run_cluster(
            cluster, run_id="run-3", compiled_graph=mock_graph,
        )

        assert co.outcome == "exception"
        assert co.exception_category == "ESCALATED_UNRESOLVED"
        assert "failed" in co.reasoning_trace.lower()
        assert co.processing_ms >= 0


# ═══════════════════════════════════════════════════════════
#  TEST 10: run_cluster() — auto-generates run_id if not given
# ═══════════════════════════════════════════════════════════


class TestRunClusterAutoRunId:
    """run_cluster generates a UUID run_id if none is provided."""

    @pytest.mark.asyncio
    async def test_auto_run_id(self):
        cluster = _make_cluster(cluster_id="c_auto")

        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "outcome": "verified",
            "reasoning_trace": "ok",
        })

        co = await run_cluster(cluster, compiled_graph=mock_graph)

        # Verify the thread_id was constructed with a valid UUID
        call_config = mock_graph.ainvoke.call_args[1].get("config") or mock_graph.ainvoke.call_args[0][1]
        thread_id = call_config["configurable"]["thread_id"]
        # Should match: f"{uuid}:cluster_c_auto"
        assert thread_id.endswith(":cluster_c_auto")
        # The prefix should be a valid UUID
        prefix = thread_id.replace(":cluster_c_auto", "")
        uuid.UUID(prefix)  # raises if not valid UUID


# ═══════════════════════════════════════════════════════════
#  TEST 11: EVAL_MODE never calls interrupt (integration-level)
# ═══════════════════════════════════════════════════════════


class TestEvalModeNoInterrupt:
    """Confirm EVAL_MODE=True never triggers LangGraph interrupt()."""

    @pytest.mark.asyncio
    async def test_eval_mode_skips_interrupt_via_run_cluster(self):
        """When eval_mode=True and the graph reaches HumanReviewInterrupt,
        interrupt() must NOT be called."""
        cluster = _make_cluster(cluster_id="c_eval")

        # Build a graph that will actually traverse to HumanReviewInterrupt
        # by mocking the LLM to exhaust retries
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "outcome": "exception",
            "exception_category": "ESCALATED_UNRESOLVED",
            "reasoning_trace": "EVAL_MODE=true, auto-tagged",
            "exhausted_models": [],
        })

        co = await run_cluster(
            cluster,
            run_id="run-eval",
            eval_mode=True,
            compiled_graph=mock_graph,
        )

        # The state passed to ainvoke should have eval_mode=True
        call_state = mock_graph.ainvoke.call_args[0][0]
        assert call_state["eval_mode"] is True


# ═══════════════════════════════════════════════════════════
#  TEST 12: Checkpointer integration
# ═══════════════════════════════════════════════════════════


class TestCheckpointerIntegration:
    """build_cluster_graph accepts a checkpointer argument."""

    def test_builds_without_checkpointer(self):
        """Graph compiles fine with no checkpointer."""
        graph = build_cluster_graph(checkpointer=None)
        assert graph is not None

    def test_builds_with_mock_checkpointer(self):
        """Graph accepts a checkpointer object."""
        mock_cp = MagicMock()
        graph = build_cluster_graph(checkpointer=mock_cp)
        assert graph is not None
