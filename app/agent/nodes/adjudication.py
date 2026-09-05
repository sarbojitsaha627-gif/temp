"""
LLM adjudication core — per-cluster LangGraph nodes.

Three nodes:
    1. llm_react_loop      — Bounded ReAct loop with model-chain fallback
    2. independent_verifier — Pure deterministic verification (no LLM)
    3. check_retry_limit    — Retry-with-error-delta routing

Plus routing functions for LangGraph conditional edges.

Design invariants (from plan.md):
  • Max 5 tool-call iterations per model attempt.
  • Max 2 verification retries (CheckRetryLimit).
  • Max 3 API retries with exponential backoff per tool call.
  • Model chain fallback wipes message history completely.
  • LLM only proposes IDs + categories; arithmetic in deterministic code.
  • exhausted_models pre-check before any API call.
  • 429 parsing: rolling-window → retry once; daily quota → mark exhausted.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from app.agent.schemas import (
    AdjudicationDecision,
    AdjudicationVerdict,
    VerificationResult,
)
from app.agent.state import ClusterState
from app.agent.tools import verify_amount_match_logic
from app.db.enums import ExceptionCategory

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

MAX_TOOL_ITERATIONS = 5
MAX_VERIFICATION_RETRIES = 2
MAX_API_RETRIES = 3
API_BACKOFF_DELAYS = [1, 2, 4]  # seconds

DEFAULT_MODEL_CHAIN = [
    "groq/llama-3.3-70b-versatile",
    "groq/openai-gpt-oss-120b",
    "gpt-4o",
]


# ═══════════════════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER
# ═══════════════════════════════════════════════════════════

def _build_system_prompt(
    cluster: dict[str, Any],
    *,
    retry_count: int = 0,
    last_error_delta: int | None = None,
    verification_feedback: str | None = None,
) -> str:
    """Build the system prompt for one cluster adjudication attempt.

    Injects error-delta context on retry so the LLM can self-correct.
    """
    primary_type = cluster.get("primary_entity_type", "unknown")
    primary_id = cluster.get("primary_entity_id", "unknown")
    candidates = cluster.get("candidate_matches", [])
    has_collision = cluster.get("has_amount_collision", False)

    cand_lines = []
    for c in candidates:
        cand_lines.append(
            f"  - {c.get('entity_type', '?')}:{c.get('entity_id', '?')} "
            f"amount={c.get('amount_paise', 0)} paise  "
            f"score={c.get('score', 0.0):.3f}"
        )
    cand_block = "\n".join(cand_lines) if cand_lines else "  (none)"

    prompt = f"""\
You are a payment reconciliation agent. Your job is to determine which \
candidate entity matches the primary entity in this cluster.

## Cluster Context
- Primary entity: {primary_type} `{primary_id}`
- Candidates:
{cand_block}
- Amount collision: {"YES — multiple candidates share the same amount" if has_collision else "No"}
- Window: {cluster.get('window_start', '?')} → {cluster.get('window_end', '?')}

## Rules
1. Use the provided tools to query entity details (query_order, \
query_settlement, query_bank_transaction, query_refund, query_dispute, \
check_fee_schedule).
2. Use verify_amount_match to check mathematical balance. \
NEVER compute amounts yourself — always delegate to the tool.
3. After gathering evidence, return your final decision as a JSON object \
with this exact schema:
   {{
     "decision": "match" | "no_match" | "partial",
     "confidence": 0.0–1.0,
     "matched_entity_ids": ["id1", "id2", ...],
     "proposed_category": null | "<ExceptionCategory value>",
     "reasoning": "step-by-step explanation"
   }}
4. Every ID in matched_entity_ids MUST have been retrieved via a tool call \
in this session. Do not cite IDs you have not queried.
5. If you cannot resolve the match, set decision to "no_match" and propose \
an appropriate exception category."""

    if retry_count > 0 and last_error_delta is not None:
        prompt += f"""

## RETRY CONTEXT (Attempt {retry_count + 1})
Your previous answer failed verification.
- Discrepancy delta: {last_error_delta:+d} paise (expected - calculated)
- Verifier feedback: {verification_feedback or 'N/A'}

Re-examine the entities. A common cause is an omitted secondary deduction \
(18% GST on fees, partner discount, or partial refund). Use the tools \
again to find the missing component."""

    return prompt


# ═══════════════════════════════════════════════════════════
#  LLM INVOCATION HELPER
# ═══════════════════════════════════════════════════════════

class _RateLimitInfo:
    """Parsed 429 response metadata."""
    def __init__(self, is_daily: bool, retry_after: float):
        self.is_daily = is_daily
        self.retry_after = retry_after


def _parse_rate_limit(error: Exception) -> _RateLimitInfo | None:
    """Extract rate-limit info from a 429 error.

    Returns None if the error is not a rate-limit error.
    Detects daily/hard quota vs rolling-window limits.
    """
    error_str = str(error).lower()
    error_body = getattr(error, "body", None) or {}
    if isinstance(error_body, str):
        try:
            error_body = json.loads(error_body)
        except (json.JSONDecodeError, TypeError):
            error_body = {}

    # Check for 429 status
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status != 429 and "429" not in error_str and "rate" not in error_str:
        return None

    # Detect daily quota exhaustion keywords
    daily_keywords = ["daily", "rpd", "tpd", "quota", "exceeded", "limit reached"]
    is_daily = any(kw in error_str for kw in daily_keywords)
    if not is_daily and isinstance(error_body, dict):
        msg = str(error_body.get("error", {}).get("message", "")).lower()
        is_daily = any(kw in msg for kw in daily_keywords)

    # Parse retry-after
    retry_after = 60.0  # default
    headers = getattr(error, "headers", None) or {}
    if isinstance(headers, dict):
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra is not None:
            try:
                retry_after = float(ra)
            except (ValueError, TypeError):
                pass

    return _RateLimitInfo(is_daily=is_daily, retry_after=retry_after)


async def _invoke_llm_with_retry(
    llm: Any,
    messages: list[BaseMessage],
    *,
    tool_choice: str | None = None,
    tools: list | None = None,
) -> AIMessage:
    """Invoke an LLM with exponential backoff on transient failures.

    Raises on:
      - Rate-limit (429) — caller handles via _parse_rate_limit
      - Unrecoverable errors after MAX_API_RETRIES
    """
    last_error: Exception | None = None

    for attempt in range(MAX_API_RETRIES):
        try:
            kwargs: dict[str, Any] = {}
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

            result = await llm.ainvoke(messages, **kwargs)
            return result

        except Exception as e:
            last_error = e

            # Check for rate-limit — let caller handle
            rl_info = _parse_rate_limit(e)
            if rl_info is not None:
                raise

            # Transient error — backoff and retry
            if attempt < MAX_API_RETRIES - 1:
                delay = API_BACKOFF_DELAYS[attempt]
                logger.warning(
                    "LLM API attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1, MAX_API_RETRIES, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "LLM API exhausted %d retries: %s", MAX_API_RETRIES, e,
                )

    raise last_error  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════
#  TOOL EXECUTION HELPER
# ═══════════════════════════════════════════════════════════

async def _execute_tool_calls(
    ai_message: AIMessage,
    tool_map: dict[str, Any],
) -> tuple[list[ToolMessage], list[dict[str, Any]]]:
    """Execute tool calls from an AI message, return ToolMessages + artifacts.

    Uses the content_and_artifact response format: each tool returns
    (string_for_llm, raw_artifact_dict).
    """
    tool_messages: list[ToolMessage] = []
    artifacts: list[dict[str, Any]] = []

    for tc in ai_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]

        tool_fn = tool_map.get(tool_name)
        if tool_fn is None:
            tool_messages.append(ToolMessage(
                content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                tool_call_id=tool_id,
                name=tool_name,
            ))
            continue

        try:
            result = await tool_fn.ainvoke(tool_args)

            # content_and_artifact tools return (content, artifact)
            if isinstance(result, tuple) and len(result) == 2:
                content, artifact = result
                if isinstance(artifact, dict):
                    artifacts.append(artifact)
                tool_messages.append(ToolMessage(
                    content=content if isinstance(content, str) else json.dumps(content),
                    tool_call_id=tool_id,
                    name=tool_name,
                ))
            else:
                # Plain string result
                content = result if isinstance(result, str) else json.dumps(result)
                tool_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=tool_id,
                    name=tool_name,
                ))

        except Exception as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            tool_messages.append(ToolMessage(
                content=json.dumps({"error": f"Tool execution failed: {e}"}),
                tool_call_id=tool_id,
                name=tool_name,
            ))

    return tool_messages, artifacts


def _extract_decision_from_message(ai_message: AIMessage) -> AdjudicationDecision | None:
    """Try to parse an AdjudicationDecision from an AI message's text content."""
    content = ai_message.content
    if not content or not isinstance(content, str):
        return None

    # Try to find JSON in the content
    # Look for JSON block (possibly in markdown code fence)
    text = content.strip()
    if "```" in text:
        # Extract from code fence
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                text = cleaned
                break

    # Find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
        return AdjudicationDecision.model_validate(data)
    except (json.JSONDecodeError, Exception):
        return None


# ═══════════════════════════════════════════════════════════
#  NODE 1: LlmReActLoop
# ═══════════════════════════════════════════════════════════

async def llm_react_loop(
    state: ClusterState,
    *,
    llm_factory: Any = None,
    tool_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """LlmReActLoop node — bounded ReAct loop with model-chain fallback.

    Processes one cluster through the model chain:
    1. Pre-check exhausted_models before attempting any model
    2. Build prompt with cluster context (+ error delta on retry)
    3. Run tool-call loop (max 5 iterations)
    4. Extract structured AdjudicationDecision
    5. On failure: fall back to next model with clean message history

    Parameters
    ----------
    llm_factory : callable(model_name: str) -> ChatModel
        Factory function that returns a configured LLM for the given model.
        Injected by the graph builder; tests can provide a mock.
    tool_map : dict[str, ToolFunction]
        Maps tool names to their callable functions.
        Injected by the graph builder; tests can provide mocks.
    """
    start_ms = time.monotonic_ns() // 1_000_000

    cluster = state["cluster"]
    model_chain = state.get("model_chain", DEFAULT_MODEL_CHAIN)
    exhausted_models = list(state.get("exhausted_models") or [])
    current_model_index = state.get("current_model_index", 0)
    retry_count = state.get("retry_count", 0)
    last_error_delta = state.get("last_error_delta")
    verification_feedback = state.get("verification_feedback")

    # Accumulate new evidence across the loop
    new_evidence: list[dict[str, Any]] = []

    # ── Walk the model chain ─────────────────────────────────
    while current_model_index < len(model_chain):
        model_name = model_chain[current_model_index]

        # Pre-check exhausted_models — skip if already known exhausted
        if model_name in exhausted_models:
            logger.info(
                "Model %s already exhausted, skipping to next", model_name,
            )
            current_model_index += 1
            continue

        # Build fresh messages for this model (never carry history across models)
        system_prompt = _build_system_prompt(
            cluster,
            retry_count=retry_count,
            last_error_delta=last_error_delta,
            verification_feedback=verification_feedback,
        )

        # Build the initial user message with candidate summary
        user_content = (
            f"Resolve cluster '{cluster.get('cluster_id', '?')}': "
            f"primary {cluster.get('primary_entity_type', '?')}:"
            f"{cluster.get('primary_entity_id', '?')} with "
            f"{len(cluster.get('candidate_matches', []))} candidates. "
            f"Query each entity, verify amounts with the tool, "
            f"then return your AdjudicationDecision."
        )

        model_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        # Get or create the LLM instance
        try:
            llm = llm_factory(model_name) if llm_factory else None
            if llm is None:
                raise ValueError(f"No LLM factory provided for {model_name}")
        except Exception as e:
            logger.error("Failed to create LLM for %s: %s", model_name, e)
            current_model_index += 1
            continue

        # Collect tools metadata for the LLM
        tools_for_llm = list((tool_map or {}).values())

        # ── Tool-call iteration loop ─────────────────────────
        iteration_count = 0
        decision: AdjudicationDecision | None = None
        model_failed = False

        while iteration_count < MAX_TOOL_ITERATIONS:
            # First turn: force tool use. Subsequent: auto
            tc = "required" if iteration_count == 0 else "auto"

            try:
                ai_msg = await _invoke_llm_with_retry(
                    llm,
                    model_messages,
                    tool_choice=tc,
                    tools=tools_for_llm,
                )
            except Exception as e:
                rl_info = _parse_rate_limit(e)
                if rl_info is not None:
                    if rl_info.is_daily:
                        # Daily quota exhausted — mark globally, skip to next model
                        logger.warning(
                            "Model %s daily quota exhausted, marking exhausted",
                            model_name,
                        )
                        if model_name not in exhausted_models:
                            exhausted_models.append(model_name)
                        model_failed = True
                        break
                    else:
                        # Rolling-window limit — sleep and retry once on same model
                        logger.info(
                            "Model %s rolling rate limit, sleeping %.1fs",
                            model_name, rl_info.retry_after,
                        )
                        await asyncio.sleep(rl_info.retry_after)
                        try:
                            ai_msg = await _invoke_llm_with_retry(
                                llm,
                                model_messages,
                                tool_choice=tc,
                                tools=tools_for_llm,
                            )
                        except Exception:
                            # Rolling retry also failed — fall back to next model
                            model_failed = True
                            break
                else:
                    # Hard failure (500, timeout, parse) — already retried 3x
                    logger.error(
                        "Model %s hard failure after retries: %s",
                        model_name, e,
                    )
                    model_failed = True
                    break

            model_messages.append(ai_msg)

            # Check if the model wants to call tools
            if ai_msg.tool_calls:
                tool_msgs, artifacts = await _execute_tool_calls(
                    ai_msg, tool_map or {},
                )
                model_messages.extend(tool_msgs)
                new_evidence.extend(artifacts)
                iteration_count += 1
                continue

            # No tool calls — try to extract the decision
            decision = _extract_decision_from_message(ai_msg)
            if decision is not None:
                break

            # Model returned text but no parseable decision — count as iteration
            iteration_count += 1

            # Prompt the model to return structured output
            model_messages.append(HumanMessage(
                content=(
                    "Please return your final decision as a JSON object with "
                    "the AdjudicationDecision schema: {decision, confidence, "
                    "matched_entity_ids, proposed_category, reasoning}."
                )
            ))

        # ── Post-loop: check outcome ─────────────────────────
        if model_failed:
            # Clean state reset — wipe messages, try next model
            logger.info(
                "Model %s failed for cluster %s, falling back to next model",
                model_name, cluster.get("cluster_id", "?"),
            )
            current_model_index += 1
            continue

        if decision is None and iteration_count >= MAX_TOOL_ITERATIONS:
            # Iteration exhaustion on this model — try next model
            logger.warning(
                "Model %s exhausted %d iterations without decision for cluster %s",
                model_name, MAX_TOOL_ITERATIONS, cluster.get("cluster_id", "?"),
            )
            current_model_index += 1
            continue

        if decision is not None:
            # Success — record which model resolved it
            elapsed = (time.monotonic_ns() // 1_000_000) - start_ms
            existing_evidence = list(state.get("cited_evidence") or [])
            existing_evidence.extend(new_evidence)

            return {
                "decision": decision.model_dump(),
                "model_used": model_name,
                "current_model_index": current_model_index,
                "iteration_count": iteration_count,
                "exhausted_models": exhausted_models,
                "cited_evidence": existing_evidence,
                "messages": model_messages,
                "processing_ms": elapsed,
                "outcome": "",  # will be set by verifier/routing
                "reasoning_trace": (
                    f"LlmReActLoop resolved by {model_name} "
                    f"(iter={iteration_count}, retry={retry_count}): "
                    f"{decision.reasoning}"
                ),
            }

        # Shouldn't reach here, but if decision is None and not iteration-exhausted:
        current_model_index += 1

    # ── All models exhausted ─────────────────────────────────
    elapsed = (time.monotonic_ns() // 1_000_000) - start_ms
    existing_evidence = list(state.get("cited_evidence") or [])
    existing_evidence.extend(new_evidence)

    return {
        "decision": None,
        "model_used": None,
        "current_model_index": current_model_index,
        "exhausted_models": exhausted_models,
        "cited_evidence": existing_evidence,
        "messages": [],
        "processing_ms": elapsed,
        "outcome": "exception",
        "exception_category": ExceptionCategory.ESCALATED_UNRESOLVED.value,
        "reasoning_trace": (
            f"LLM model chain exhausted after {len(model_chain)} models. "
            f"Exhausted models: {exhausted_models}. "
            f"Cluster {cluster.get('cluster_id', '?')} routed to ESCALATED_UNRESOLVED."
        ),
    }


# ═══════════════════════════════════════════════════════════
#  NODE 2: IndependentVerifier
# ═══════════════════════════════════════════════════════════

def independent_verifier(state: ClusterState) -> dict[str, Any]:
    """IndependentVerifier — pure deterministic, no LLM.

    Checks:
    1. Citation validation: every matched_entity_id must appear in cited_evidence
    2. Amount re-verification: re-run verify_amount_match_logic independently
    3. Taxonomy validation: proposed_category must be a valid ExceptionCategory
    4. Amount-collision fallback gate: if has_amount_collision AND model is not
       primary → require confidence >= 0.9

    Returns state updates including verification_result.
    """
    decision_data = state.get("decision")
    if decision_data is None:
        # No decision from LLM — auto-fail
        result = VerificationResult(
            passed=False,
            delta_paise=0,
            tolerance_paise=0,
            feedback="No AdjudicationDecision provided by LLM.",
            failed_checks=["no_decision"],
        )
        return {
            "verification_result": result.model_dump(),
            "outcome": "",  # routing decides
        }

    try:
        decision = AdjudicationDecision.model_validate(decision_data)
    except Exception as e:
        result = VerificationResult(
            passed=False,
            delta_paise=0,
            tolerance_paise=0,
            feedback=f"Failed to parse AdjudicationDecision: {e}",
            failed_checks=["parse_error"],
        )
        return {
            "verification_result": result.model_dump(),
            "outcome": "",
        }

    cluster = state["cluster"]
    cited_evidence = state.get("cited_evidence") or []
    model_chain = state.get("model_chain", DEFAULT_MODEL_CHAIN)
    model_used = state.get("model_used")
    current_model_index = state.get("current_model_index", 0)

    failed_checks: list[str] = []
    feedback_parts: list[str] = []

    # ── 1. Citation validation ───────────────────────────────
    # Build the set of entity IDs that were actually retrieved via tools
    cited_ids: set[str] = set()
    for ev in cited_evidence:
        if isinstance(ev, dict):
            # Each tool artifact has entity-specific ID fields
            for key in ("order_id", "settlement_id", "bank_txn_id",
                        "refund_id", "dispute_id"):
                val = ev.get(key)
                if val and not ev.get("error"):
                    cited_ids.add(val)

    hallucinated = [
        eid for eid in decision.matched_entity_ids
        if eid not in cited_ids
    ]
    if hallucinated:
        failed_checks.append("hallucinated_citation")
        feedback_parts.append(
            f"Hallucinated citation: {hallucinated} not in cited_evidence. "
            f"Cited IDs: {sorted(cited_ids)}."
        )

    # ── 2. Amount re-verification ────────────────────────────
    # Extract amounts from cited_evidence by entity type
    orders_gross: list[int] = []
    refunds: list[int] = []
    disputes: list[int] = []
    settlements_net: list[int] = []
    settlements_fee: list[int] = []
    settlements_tax: list[int] = []
    bank_amounts: list[int] = []
    bank_charges: list[int] = []

    for ev in cited_evidence:
        if not isinstance(ev, dict) or ev.get("error"):
            continue

        entity = ev.get("entity")
        eid_in_evidence = None
        if entity == "order":
            eid_in_evidence = ev.get("order_id")
        elif entity == "settlement":
            eid_in_evidence = ev.get("settlement_id")
        elif entity == "bank_transaction":
            eid_in_evidence = ev.get("bank_txn_id")
        elif entity == "refund":
            eid_in_evidence = ev.get("refund_id")
        elif entity == "dispute":
            eid_in_evidence = ev.get("dispute_id")

        # Only include entities that are in the decision's matched IDs
        if eid_in_evidence and eid_in_evidence in decision.matched_entity_ids:
            if entity == "order":
                gross = ev.get("gross_amount_paise")
                if gross is not None:
                    orders_gross.append(gross)
            elif entity == "refund":
                amt = ev.get("amount_paise")
                if amt is not None:
                    refunds.append(amt)
            elif entity == "dispute":
                amt = ev.get("amount_paise")
                if amt is not None:
                    disputes.append(amt)
            elif entity == "settlement":
                net = ev.get("net_amount_paise")
                fee = ev.get("fee_base_paise")
                tax = ev.get("fee_tax_gst_paise")
                if net is not None:
                    settlements_net.append(net)
                if fee is not None:
                    settlements_fee.append(fee)
                if tax is not None:
                    settlements_tax.append(tax)
            elif entity == "bank_transaction":
                amt = ev.get("amount_paise")
                charges = ev.get("bank_charges_paise", 0)
                if amt is not None:
                    bank_amounts.append(amt)
                    bank_charges.append(charges)

    # Only verify amounts if we have entities to verify
    has_entities = (orders_gross or refunds or disputes or
                    settlements_net or bank_amounts)

    if decision.decision == AdjudicationVerdict.MATCH and has_entities:
        amount_result = verify_amount_match_logic(
            orders_gross_paise=orders_gross,
            refunds_paise=refunds,
            disputes_paise=disputes,
            settlements_net_paise=settlements_net,
            settlements_fee_paise=settlements_fee,
            settlements_tax_paise=settlements_tax,
            bank_txns_amount_paise=bank_amounts,
            bank_txns_charges_paise=bank_charges,
        )
        delta_paise = amount_result["delta_paise"]
        tolerance_paise = amount_result["tolerance_paise"]

        if not amount_result["is_match"]:
            failed_checks.append("amount_mismatch")
            breakdown = amount_result.get("breakdown", {})
            feedback_parts.append(
                f"Amount verification failed. "
                f"Delta = {delta_paise} paise (tolerance = {tolerance_paise}). "
                f"Breakdown: orders_gross={breakdown.get('orders_gross', 0)}, "
                f"refunds={breakdown.get('refunds', 0)}, "
                f"settlements_net={breakdown.get('settlements_net', 0)}, "
                f"settlements_fee={breakdown.get('settlements_fee', 0)}, "
                f"settlements_tax={breakdown.get('settlements_tax', 0)}, "
                f"bank_amounts={breakdown.get('bank_amounts', 0)}, "
                f"bank_charges={breakdown.get('bank_charges', 0)}."
            )
    else:
        delta_paise = 0
        tolerance_paise = len(orders_gross) * 1  # ±1 paisa per order

    # ── 3. Taxonomy validation ───────────────────────────────
    if decision.proposed_category is not None:
        valid_categories = {e.value for e in ExceptionCategory}
        if decision.proposed_category not in valid_categories:
            failed_checks.append("invalid_category")
            feedback_parts.append(
                f"Invalid proposed_category '{decision.proposed_category}'. "
                f"Valid categories: {sorted(valid_categories)}."
            )

    # ── 4. Amount-collision fallback-model confidence gate ────
    has_collision = cluster.get("has_amount_collision", False)
    primary_model = model_chain[0] if model_chain else None
    is_fallback = model_used != primary_model and model_used is not None

    if (has_collision and is_fallback
            and decision.decision == AdjudicationVerdict.MATCH):
        if decision.confidence < 0.9:
            failed_checks.append("amount_collision_confidence_gate")
            feedback_parts.append(
                "Verification passed but routed to human review: "
                "amount-collision cluster resolved by fallback model "
                f"below confidence threshold. "
                f"confidence={decision.confidence:.2f} < 0.9 required."
            )

    # ── Build result ─────────────────────────────────────────
    passed = len(failed_checks) == 0
    feedback = " | ".join(feedback_parts) if feedback_parts else "All checks passed."

    result = VerificationResult(
        passed=passed,
        delta_paise=delta_paise,
        tolerance_paise=tolerance_paise,
        feedback=feedback,
        failed_checks=failed_checks,
    )

    # Update reasoning trace
    existing_trace = state.get("reasoning_trace", "")
    verifier_trace = (
        f" → IndependentVerifier: {'PASS' if passed else 'FAIL'} "
        f"[{', '.join(failed_checks) if failed_checks else 'clean'}]. "
        f"{feedback}"
    )

    return {
        "verification_result": result.model_dump(),
        "outcome": "verified" if passed else "",
        "last_error_delta": delta_paise,
        "verification_feedback": feedback,
        "reasoning_trace": existing_trace + verifier_trace,
    }


# ═══════════════════════════════════════════════════════════
#  NODE 3: CheckRetryLimit
# ═══════════════════════════════════════════════════════════

def check_retry_limit(state: ClusterState) -> dict[str, Any]:
    """CheckRetryLimit — retry-with-error-delta routing.

    On verification failure:
      - retry_count < 2 → increment, inject error delta, route to LlmReActLoop
      - retry_count >= 2 → route to HumanReviewInterrupt (Step 8 placeholder)
    """
    retry_count = state.get("retry_count", 0)
    vr_data = state.get("verification_result")
    cluster = state.get("cluster", {})

    delta = 0
    feedback = ""
    if vr_data:
        delta = vr_data.get("delta_paise", 0)
        feedback = vr_data.get("feedback", "")

    if retry_count < MAX_VERIFICATION_RETRIES:
        new_count = retry_count + 1
        existing_trace = state.get("reasoning_trace", "")
        retry_trace = (
            f" → CheckRetryLimit: retry {new_count}/{MAX_VERIFICATION_RETRIES}. "
            f"Injecting last_error_delta={delta:+d} paise."
        )
        return {
            "retry_count": new_count,
            "last_error_delta": delta,
            "verification_feedback": feedback,
            "outcome": "retry",
            "reasoning_trace": existing_trace + retry_trace,
            # Reset iteration count for new LLM attempt
            "iteration_count": 0,
            # Wipe messages for fresh attempt with error context
            "messages": [RemoveMessage(id="__all__")],
        }
    else:
        existing_trace = state.get("reasoning_trace", "")
        hitl_trace = (
            f" → CheckRetryLimit: retries exhausted ({retry_count}/{MAX_VERIFICATION_RETRIES}). "
            f"Routing to HumanReviewInterrupt."
        )
        return {
            "outcome": "hitl",
            "reasoning_trace": existing_trace + hitl_trace,
        }


# ═══════════════════════════════════════════════════════════
#  HITL PLACEHOLDER (Step 8)
# ═══════════════════════════════════════════════════════════

def hitl_placeholder(state: ClusterState) -> dict[str, Any]:
    """Placeholder for HumanReviewInterrupt — to be implemented in Step 8.

    In EVAL_MODE: auto-tag ESCALATED_UNRESOLVED and continue.
    In normal mode: would block via LangGraph interrupt() (Step 8).
    """
    eval_mode = state.get("eval_mode", False)
    existing_trace = state.get("reasoning_trace", "")

    if eval_mode:
        return {
            "outcome": "exception",
            "exception_category": ExceptionCategory.ESCALATED_UNRESOLVED.value,
            "reasoning_trace": (
                existing_trace +
                " → HumanReviewInterrupt: EVAL_MODE=true, "
                "auto-tagged ESCALATED_UNRESOLVED."
            ),
        }
    else:
        # In production this would use LangGraph interrupt()
        # For now, treat as escalated
        return {
            "outcome": "hitl",
            "exception_category": ExceptionCategory.ESCALATED_UNRESOLVED.value,
            "reasoning_trace": (
                existing_trace +
                " → HumanReviewInterrupt: pending human review."
            ),
        }


# ═══════════════════════════════════════════════════════════
#  ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════

def route_after_llm(state: ClusterState) -> Literal[
    "independent_verifier", "categorize_exception"
]:
    """Route after LlmReActLoop.

    - decision exists → independent_verifier
    - no decision (chain exhausted / iteration exhausted) → categorize_exception
    """
    if state.get("outcome") == "exception":
        return "categorize_exception"
    if state.get("decision") is not None:
        return "independent_verifier"
    return "categorize_exception"


def route_after_verifier(state: ClusterState) -> Literal[
    "persist", "check_retry_limit"
]:
    """Route after IndependentVerifier.

    - passed → persist (PersistMatchOrException)
    - failed → check_retry_limit
    """
    vr_data = state.get("verification_result")
    if vr_data and vr_data.get("passed"):
        return "persist"
    return "check_retry_limit"


def route_after_retry_check(state: ClusterState) -> Literal[
    "llm_react_loop", "human_review_interrupt"
]:
    """Route after CheckRetryLimit.

    - retries remain → llm_react_loop
    - exhausted → human_review_interrupt
    """
    if state.get("outcome") == "retry":
        return "llm_react_loop"
    return "human_review_interrupt"
