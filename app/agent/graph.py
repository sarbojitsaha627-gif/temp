"""
Per-cluster LangGraph StateGraph — wires Steps 7+8 only.

Topology:
    LlmReActLoop → IndependentVerifier
        → conditional[pass → END, fail → CheckRetryLimit]
    CheckRetryLimit
        → conditional[retry → LlmReActLoop, exhausted → HumanReviewInterrupt]
    HumanReviewInterrupt
        → conditional[approved → END, rejected/bypass → CategorizeException]
    CategorizeException → END

Design constraints:
  • Scoped strictly to per-cluster LLM reasoning.
  • Deterministic tier (Steps 4–5) is NOT included as graph nodes.
  • Postgres checkpointer via langgraph-checkpoint-postgres.
  • thread_id = f"{run_id}:cluster_{cluster_id}"
  • Exposes run_cluster(cluster) → ClusterOutcome as the public API.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from langgraph.graph import END, StateGraph

from app.agent.nodes.adjudication import (
    DEFAULT_MODEL_CHAIN,
    check_retry_limit,
    independent_verifier,
    llm_react_loop,
    route_after_llm,
    route_after_retry_check,
    route_after_verifier,
)
from app.agent.nodes.resolution import (
    categorize_exception,
    human_review_interrupt,
)
from app.agent.state import ClusterState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  CLUSTER OUTCOME — typed return from run_cluster()
# ═══════════════════════════════════════════════════════════


@dataclass
class ClusterOutcome:
    """Result of running a single cluster through the adjudication graph.

    Attributes
    ----------
    cluster_id : str
        The cluster identifier that was processed.
    outcome : str
        Terminal state: "verified" | "exception"
    decision : dict | None
        The LLM's structured AdjudicationDecision, if one was produced.
    verification_result : dict | None
        The IndependentVerifier's deterministic result, if verification ran.
    exception_category : str | None
        If outcome == "exception", the taxonomy category assigned.
    exception_payload : dict | None
        Category-specific structured payload for exception_staging.
    exception_severity : str | None
        Severity level of the exception (LOW/MEDIUM/HIGH/CRITICAL).
    model_used : str | None
        Which model in the chain actually resolved the cluster.
    reasoning_trace : str
        Accumulated reasoning trace for audit/persistence.
    processing_ms : int
        Wall-clock milliseconds for this cluster's full adjudication.
    exhausted_models : list[str]
        Models that were exhausted during this run (carried back to batch).
    """
    cluster_id: str
    outcome: str
    decision: Optional[dict[str, Any]] = None
    verification_result: Optional[dict[str, Any]] = None
    exception_category: Optional[str] = None
    exception_payload: Optional[dict[str, Any]] = None
    exception_severity: Optional[str] = None
    model_used: Optional[str] = None
    reasoning_trace: str = ""
    processing_ms: int = 0
    exhausted_models: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ═══════════════════════════════════════════════════════════


def _route_after_hitl(state: ClusterState) -> Literal[
    "persist", "categorize_exception"
]:
    """Route after HumanReviewInterrupt.

    - approved → persist (END path)
    - exception / rejected / EVAL_MODE bypass → categorize_exception
    """
    if state.get("outcome") == "verified":
        return "persist"
    return "categorize_exception"


def _make_persist_node():
    """Lightweight pass-through node that finalises the outcome.

    Real persistence (writing match_groups / exception_staging) is handled
    by the batch orchestrator after the graph returns — not inside the
    per-cluster graph itself.
    """
    def persist(state: ClusterState) -> dict[str, Any]:
        outcome = state.get("outcome", "")
        if not outcome:
            outcome = "verified"
        return {"outcome": outcome}
    return persist


def build_cluster_graph(
    *,
    llm_factory: Any = None,
    tool_map: dict[str, Any] | None = None,
    checkpointer: Any = None,
) -> Any:
    """Build and compile the per-cluster LangGraph StateGraph.

    Parameters
    ----------
    llm_factory : callable(model_name: str) -> ChatModel, optional
        Factory that returns a configured LLM for the given model identifier.
        If None, llm_react_loop will use its built-in default.
    tool_map : dict[str, ToolFunction], optional
        Maps tool names to their callable functions.
    checkpointer : langgraph Checkpointer, optional
        Postgres checkpointer for durable state.  If None, the graph runs
        without checkpointing (suitable for tests / eval mode).

    Returns
    -------
    CompiledGraph
        Ready for .ainvoke() with a ClusterState dict.
    """
    graph = StateGraph(ClusterState)

    # ── Bind factory/tools into the llm_react_loop node ──
    async def bound_llm_react_loop(state: ClusterState) -> dict[str, Any]:
        return await llm_react_loop(
            state, llm_factory=llm_factory, tool_map=tool_map,
        )

    # ── Add nodes ────────────────────────────────────────
    graph.add_node("llm_react_loop", bound_llm_react_loop)
    graph.add_node("independent_verifier", independent_verifier)
    graph.add_node("check_retry_limit", check_retry_limit)
    graph.add_node("human_review_interrupt", human_review_interrupt)
    graph.add_node("categorize_exception", categorize_exception)
    graph.add_node("persist", _make_persist_node())

    # ── Entry point ──────────────────────────────────────
    graph.set_entry_point("llm_react_loop")

    # ── LlmReActLoop → IndependentVerifier | CategorizeException
    graph.add_conditional_edges(
        "llm_react_loop",
        route_after_llm,
        {
            "independent_verifier": "independent_verifier",
            "categorize_exception": "categorize_exception",
        },
    )

    # ── IndependentVerifier → persist (pass) | CheckRetryLimit (fail)
    graph.add_conditional_edges(
        "independent_verifier",
        route_after_verifier,
        {
            "persist": "persist",
            "check_retry_limit": "check_retry_limit",
        },
    )

    # ── CheckRetryLimit → LlmReActLoop (retry) | HumanReviewInterrupt
    graph.add_conditional_edges(
        "check_retry_limit",
        route_after_retry_check,
        {
            "llm_react_loop": "llm_react_loop",
            "human_review_interrupt": "human_review_interrupt",
        },
    )

    # ── HumanReviewInterrupt → persist (approved) | CategorizeException
    graph.add_conditional_edges(
        "human_review_interrupt",
        _route_after_hitl,
        {
            "persist": "persist",
            "categorize_exception": "categorize_exception",
        },
    )

    # ── Terminal edges ───────────────────────────────────
    graph.add_edge("categorize_exception", END)
    graph.add_edge("persist", END)

    # ── Compile with optional checkpointer ───────────────
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


# ═══════════════════════════════════════════════════════════
#  POSTGRES CHECKPOINTER FACTORY
# ═══════════════════════════════════════════════════════════


def get_postgres_checkpointer():
    """Create a Postgres checkpointer for durable graph state.

    Uses the same DATABASE_URL as the rest of the application, but
    converts to the psycopg sync/async connection string format
    that langgraph-checkpoint-postgres expects.

    Returns None if the database is not available (e.g. in tests).
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/finance_controller",
        )
        # langgraph-checkpoint-postgres wants psycopg format
        pg_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

        return AsyncPostgresSaver.from_conn_string(pg_url)
    except Exception as e:
        logger.warning(
            "Could not create Postgres checkpointer: %s. "
            "Graph will run without checkpointing.", e,
        )
        return None


# ═══════════════════════════════════════════════════════════
#  PUBLIC API: run_cluster()
# ═══════════════════════════════════════════════════════════


def _build_initial_state(
    cluster: dict[str, Any],
    *,
    run_id: str,
    eval_mode: bool = False,
    exhausted_models: list[str] | None = None,
    model_chain: list[str] | None = None,
) -> dict[str, Any]:
    """Build the initial ClusterState dict from a cluster payload."""
    return {
        # Orchestrator inputs
        "cluster": cluster,
        "reconciliation_run_id": run_id,
        "eval_mode": eval_mode,
        "exhausted_models": exhausted_models or [],
        "model_chain": model_chain or list(DEFAULT_MODEL_CHAIN),
        # LlmReActLoop working state
        "messages": [],
        "current_model_index": 0,
        "iteration_count": 0,
        "retry_count": 0,
        "last_error_delta": None,
        "verification_feedback": None,
        "cited_evidence": [],
        # Output fields (initialised empty)
        "decision": None,
        "verification_result": None,
        "outcome": "",
        "model_used": None,
        "processing_ms": 0,
        "exception_category": None,
        "reasoning_trace": "",
    }


def _extract_outcome(final_state: dict[str, Any], cluster_id: str) -> ClusterOutcome:
    """Extract a ClusterOutcome from the graph's terminal state."""
    return ClusterOutcome(
        cluster_id=cluster_id,
        outcome=final_state.get("outcome", "exception"),
        decision=final_state.get("decision"),
        verification_result=final_state.get("verification_result"),
        exception_category=final_state.get("exception_category"),
        exception_payload=final_state.get("exception_payload"),
        exception_severity=final_state.get("exception_severity"),
        model_used=final_state.get("model_used"),
        reasoning_trace=final_state.get("reasoning_trace", ""),
        processing_ms=final_state.get("processing_ms", 0),
        exhausted_models=final_state.get("exhausted_models", []),
    )


async def run_cluster(
    cluster: dict[str, Any],
    *,
    run_id: str | None = None,
    eval_mode: bool = False,
    exhausted_models: list[str] | None = None,
    model_chain: list[str] | None = None,
    llm_factory: Any = None,
    tool_map: dict[str, Any] | None = None,
    checkpointer: Any = None,
    compiled_graph: Any = None,
) -> ClusterOutcome:
    """Run a single cluster through the per-cluster adjudication graph.

    This is the public entry point called by the batch orchestrator.

    Parameters
    ----------
    cluster : dict
        A CandidateCluster-shaped dict (from ClusterCandidates output).
        Must contain at minimum: cluster_id, primary_entity_type,
        primary_entity_id, candidate_matches, has_amount_collision.
    run_id : str, optional
        Reconciliation run UUID. Generated if not provided.
    eval_mode : bool
        If True, HumanReviewInterrupt is bypassed (no blocking interrupt).
    exhausted_models : list[str], optional
        Models already known exhausted from prior clusters in this run.
    model_chain : list[str], optional
        Ordered list of model identifiers. Defaults to DEFAULT_MODEL_CHAIN.
    llm_factory : callable, optional
        Factory for creating LLM instances.
    tool_map : dict, optional
        Tool name → callable mapping for the ReAct loop.
    checkpointer : Checkpointer, optional
        Postgres checkpointer. If None, graph runs without checkpointing.
    compiled_graph : CompiledGraph, optional
        Pre-compiled graph instance (for reuse across clusters in a batch).
        If not provided, a new graph is built and compiled.

    Returns
    -------
    ClusterOutcome
        Typed result with outcome, decision, traces, and exhausted model info.
    """
    cluster_id = cluster.get("cluster_id", f"unknown_{uuid.uuid4().hex[:8]}")
    if run_id is None:
        run_id = str(uuid.uuid4())

    # Build or reuse the compiled graph
    if compiled_graph is None:
        compiled_graph = build_cluster_graph(
            llm_factory=llm_factory,
            tool_map=tool_map,
            checkpointer=checkpointer,
        )

    # Build the initial state
    initial_state = _build_initial_state(
        cluster,
        run_id=run_id,
        eval_mode=eval_mode,
        exhausted_models=exhausted_models,
        model_chain=model_chain,
    )

    # Thread ID: f"{run_id}:cluster_{cluster_id}"
    thread_id = f"{run_id}:cluster_{cluster_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # Execute the graph
    t0 = time.perf_counter()
    try:
        final_state = await compiled_graph.ainvoke(initial_state, config=config)
    except Exception:
        logger.exception(
            "Graph execution failed for cluster %s", cluster_id,
        )
        # Return a graceful exception outcome rather than crash the batch
        return ClusterOutcome(
            cluster_id=cluster_id,
            outcome="exception",
            exception_category="ESCALATED_UNRESOLVED",
            reasoning_trace=(
                f"Graph execution failed for cluster {cluster_id}. "
                "Escalated to ESCALATED_UNRESOLVED."
            ),
            processing_ms=int((time.perf_counter() - t0) * 1000),
            exhausted_models=exhausted_models or [],
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Inject processing time into the state before extraction
    if isinstance(final_state, dict):
        final_state["processing_ms"] = elapsed_ms

    outcome = _extract_outcome(final_state, cluster_id)
    outcome.processing_ms = elapsed_ms

    logger.info(
        "Cluster %s completed: outcome=%s, model=%s, %dms",
        cluster_id, outcome.outcome, outcome.model_used, elapsed_ms,
    )

    return outcome
