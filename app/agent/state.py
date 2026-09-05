"""
LangGraph state definition for the per-cluster adjudication subgraph.

Each cluster gets its own graph invocation with:
    thread_id = f"{batch_id}:cluster_{cluster_id}"

The state is a TypedDict consumed by LangGraph's StateGraph.  All fields
are populated by the batch orchestrator before invocation, then mutated
by the three adjudication nodes (LlmReActLoop, IndependentVerifier,
CheckRetryLimit).
"""
from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agent.schemas import AdjudicationDecision, VerificationResult


class ClusterState(TypedDict):
    """Per-cluster graph state.

    Orchestrator-provided fields (set before graph invocation):
        cluster               — The CandidateCluster object from ClusterCandidates
        reconciliation_run_id — UUID string of the current batch run
        eval_mode             — If True, bypass blocking HITL interrupt
        exhausted_models      — Models already known exhausted (from ReconciliationRun)
        model_chain           — Ordered list of model identifiers to try

    LlmReActLoop working state:
        messages              — LangGraph message list (uses add_messages reducer)
        current_model_index   — Index into model_chain for the current attempt
        iteration_count       — Tool-call iterations in current model attempt (max 5)
        retry_count           — Verification retries across all models (max 2)
        last_error_delta      — Signed paise discrepancy from last verifier failure
        verification_feedback — Diagnostic string from verifier for prompt injection
        cited_evidence        — Accumulated tool-call artifacts for citation checking

    Output fields (set by nodes):
        decision              — Structured AdjudicationDecision from LLM
        verification_result   — VerificationResult from IndependentVerifier
        outcome               — Final routing: "verified" | "hitl" | "exception"
        model_used            — Which model resolved the cluster (for reporting)
        processing_ms         — Wall-clock ms for this cluster's adjudication
        exception_category    — If outcome=="exception", the category to stage
        reasoning_trace       — Accumulated reasoning for match_groups persistence
    """
    # ── Orchestrator inputs ────────────────────────────────
    cluster: dict[str, Any]
    reconciliation_run_id: str
    eval_mode: bool
    exhausted_models: list[str]
    model_chain: list[str]

    # ── LlmReActLoop working state ─────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    current_model_index: int
    iteration_count: int
    retry_count: int
    last_error_delta: Optional[int]
    verification_feedback: Optional[str]
    cited_evidence: list[dict[str, Any]]

    # ── Node outputs ───────────────────────────────────────
    decision: Optional[dict[str, Any]]
    verification_result: Optional[dict[str, Any]]
    outcome: str
    model_used: Optional[str]
    processing_ms: int
    exception_category: Optional[str]
    reasoning_trace: str
