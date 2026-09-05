"""
Terminal resolution nodes for the per-cluster LangGraph subgraph.

Two nodes:
    1. human_review_interrupt — Dual-mode HITL gate
    2. categorize_exception  — Strict 8+3 taxonomy classification

Design invariants (from plan.md):
  • EVAL_MODE=true  → skip interrupt() entirely, tag ESCALATED_UNRESOLVED,
    continue immediately.
  • EVAL_MODE=false → LangGraph interrupt() + Postgres checkpointer,
    thread_id = f"{run_id}:cluster_{cluster_id}", surfaces cited_evidence /
    reasoning_trace / verification_feedback, accepts
    {decision: approved|rejected}.  hitl_status becomes APPROVED or REJECTED.
  • CategorizeException enforces a strict ExceptionCategory enum.
    Unknown / null / freeform labels → ESCALATED_UNRESOLVED.
  • TIMING_SETTLEMENT_FLOAT uses business_days_between() from the Indian
    calendar: ≤2 business days pending → is_overdue=False (PENDING_FLOAT);
    >2 → is_overdue=True (ANOMALOUS_FLOAT_OVERDUE).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt

from app.agent.state import ClusterState
from app.db.enums import ExceptionCategory, ExceptionSeverity, HitlStatus
from app.synthetic.calendar import business_days_between

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  SEVERITY MAPPING
# ═══════════════════════════════════════════════════════════

_CATEGORY_SEVERITY: dict[ExceptionCategory, ExceptionSeverity] = {
    ExceptionCategory.TIMING_SETTLEMENT_FLOAT: ExceptionSeverity.LOW,
    ExceptionCategory.GATEWAY_FEE_MISMATCH: ExceptionSeverity.MEDIUM,
    ExceptionCategory.UNRECONCILED_BANK_FEE: ExceptionSeverity.MEDIUM,
    ExceptionCategory.SPLIT_PAYOUT_PARTIAL_DROP: ExceptionSeverity.HIGH,
    ExceptionCategory.CHARGEBACK_DEBIT_UNMATCHED: ExceptionSeverity.HIGH,
    ExceptionCategory.CURRENCY_CONVERSION_VARIANCE: ExceptionSeverity.MEDIUM,
    ExceptionCategory.SUSPICIOUS_ROUND_NUMBER_DRAIN: ExceptionSeverity.CRITICAL,
    ExceptionCategory.MISSING_SETTLEMENT_RECORD: ExceptionSeverity.HIGH,
    ExceptionCategory.UNMAPPED_BANK_DEPOSIT: ExceptionSeverity.HIGH,
    ExceptionCategory.ESCALATED_UNRESOLVED: ExceptionSeverity.CRITICAL,
    ExceptionCategory.UNACCOUNTED_LEDGER_LEAK: ExceptionSeverity.CRITICAL,
}


# ═══════════════════════════════════════════════════════════
#  NODE: HumanReviewInterrupt
# ═══════════════════════════════════════════════════════════

def human_review_interrupt(state: ClusterState) -> dict[str, Any]:
    """HumanReviewInterrupt — dual-mode HITL gate.

    EVAL_MODE=true:
        Skips interrupt() entirely.  Tags ESCALATED_UNRESOLVED and continues
        immediately so batch coverage and report always complete.

    EVAL_MODE=false:
        Uses LangGraph ``interrupt()`` with Postgres checkpointer.
        Thread is suspended at ``thread_id = f"{run_id}:cluster_{cluster_id}"``.
        Surfaces: cited_evidence, reasoning_trace, verification_feedback.
        Accepts ``{decision: "approved" | "rejected"}``.
        hitl_status becomes APPROVED or REJECTED (never a merged "resolved").
    """
    eval_mode = state.get("eval_mode", False)
    existing_trace = state.get("reasoning_trace", "")
    cluster = state.get("cluster", {})
    run_id = state.get("reconciliation_run_id", "unknown")
    cluster_id = cluster.get("cluster_id", "unknown")

    if eval_mode:
        # ── EVAL_MODE: bypass interrupt entirely ─────────────
        logger.info(
            "EVAL_MODE=true — skipping interrupt() for cluster %s, "
            "tagging ESCALATED_UNRESOLVED",
            cluster_id,
        )
        return {
            "outcome": "exception",
            "exception_category": ExceptionCategory.ESCALATED_UNRESOLVED.value,
            "reasoning_trace": (
                existing_trace
                + " → HumanReviewInterrupt: EVAL_MODE=true, "
                "auto-tagged ESCALATED_UNRESOLVED."
            ),
        }

    # ── PRODUCTION MODE: LangGraph interrupt() ───────────
    review_payload = {
        "cluster_id": cluster_id,
        "reconciliation_run_id": run_id,
        "thread_id": f"{run_id}:cluster_{cluster_id}",
        "cited_evidence": state.get("cited_evidence", []),
        "reasoning_trace": existing_trace,
        "verification_feedback": state.get("verification_feedback"),
        "decision": state.get("decision"),
        "verification_result": state.get("verification_result"),
    }

    # interrupt() suspends this thread's checkpoint in Postgres.
    # The human reviewer sees review_payload and responds with
    # {"decision": "approved"} or {"decision": "rejected"}.
    human_response = interrupt(review_payload)

    # ── Process human response ───────────────────────────
    human_decision = "rejected"  # safe default
    if isinstance(human_response, dict):
        raw = human_response.get("decision", "rejected")
        human_decision = raw if raw in ("approved", "rejected") else "rejected"
    elif isinstance(human_response, str):
        human_decision = human_response if human_response in (
            "approved", "rejected"
        ) else "rejected"

    if human_decision == "approved":
        return {
            "outcome": "verified",
            "reasoning_trace": (
                existing_trace
                + " → HumanReviewInterrupt: APPROVED by human reviewer."
            ),
        }
    else:
        return {
            "outcome": "exception",
            "exception_category": ExceptionCategory.ESCALATED_UNRESOLVED.value,
            "reasoning_trace": (
                existing_trace
                + " → HumanReviewInterrupt: REJECTED by human reviewer."
            ),
        }


# ═══════════════════════════════════════════════════════════
#  TIMING FLOAT HELPER
# ═══════════════════════════════════════════════════════════

def _evaluate_timing_float(cluster: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the TIMING_SETTLEMENT_FLOAT sub-type using T+2 rule.

    Extracts order timestamp and settlement value_date from the cluster
    or its candidate_matches.  Computes business-day elapsed:
        ≤ 2 business days → PENDING_FLOAT (is_overdue=False)
        > 2 business days → ANOMALOUS_FLOAT_OVERDUE (is_overdue=True)

    Returns a dict with: is_overdue, sub_type, biz_days, description.
    """
    # Try to find the order timestamp (primary or from candidates)
    order_ts = cluster.get("order_timestamp") or cluster.get("order_created_at")
    settlement_date = cluster.get("settlement_value_date") or cluster.get("value_date")

    # Also scan candidates for timestamps
    for cand in cluster.get("candidate_matches", []):
        if cand.get("entity_type") == "order" and not order_ts:
            order_ts = cand.get("timestamp") or cand.get("order_created_at")
        if cand.get("entity_type") == "settlement" and not settlement_date:
            settlement_date = cand.get("value_date") or cand.get("timestamp")

    # Use window_end as fallback reference date (today-like)
    reference_date = settlement_date or cluster.get("window_end")

    if not order_ts or not reference_date:
        # Cannot compute — default to overdue (conservative)
        return {
            "is_overdue": True,
            "sub_type": "ANOMALOUS_FLOAT_OVERDUE",
            "biz_days": None,
            "description": (
                "Unable to compute business-day delta (missing timestamps). "
                "Defaulting to ANOMALOUS_FLOAT_OVERDUE."
            ),
        }

    # Parse timestamps to date objects
    order_date = _to_date(order_ts)
    ref_date = _to_date(reference_date)

    if order_date is None or ref_date is None:
        return {
            "is_overdue": True,
            "sub_type": "ANOMALOUS_FLOAT_OVERDUE",
            "biz_days": None,
            "description": (
                "Failed to parse timestamps for T+2 calculation. "
                "Defaulting to ANOMALOUS_FLOAT_OVERDUE."
            ),
        }

    biz_days = business_days_between(order_date, ref_date)

    if biz_days <= 2:
        return {
            "is_overdue": False,
            "sub_type": "PENDING_FLOAT",
            "biz_days": biz_days,
            "description": (
                f"Settlement pending {biz_days} business day(s) since order. "
                f"Within T+2 window — normal operational float."
            ),
        }
    else:
        return {
            "is_overdue": True,
            "sub_type": "ANOMALOUS_FLOAT_OVERDUE",
            "biz_days": biz_days,
            "description": (
                f"Settlement pending {biz_days} business day(s) since order. "
                f"Exceeds T+2 window — flagged as anomalous overdue float."
            ),
        }


def _to_date(value: Any):
    """Best-effort parse of a date-ish value to ``datetime.date``."""
    from datetime import date as date_type

    if isinstance(value, date_type):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        # Try ISO parse (handles 2026-08-20, 2026-08-20T10:00:00+05:30, etc.)
        try:
            return datetime.fromisoformat(value).date()
        except (ValueError, TypeError):
            pass
    return None


# ═══════════════════════════════════════════════════════════
#  CATEGORY-SPECIFIC PAYLOAD BUILDERS
# ═══════════════════════════════════════════════════════════

def _build_category_payload(
    category: ExceptionCategory,
    cluster: dict[str, Any],
    *,
    reasoning_trace: str = "",
    decision_data: dict[str, Any] | None = None,
    verification_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured diagnostic payload for the given exception category.

    Each category produces a category-specific JSONB payload that is stored
    in exception_staging.payload.
    """
    base: dict[str, Any] = {
        "cluster_id": cluster.get("cluster_id"),
        "primary_entity_type": cluster.get("primary_entity_type"),
        "primary_entity_id": cluster.get("primary_entity_id"),
    }

    if category == ExceptionCategory.TIMING_SETTLEMENT_FLOAT:
        float_info = _evaluate_timing_float(cluster)
        base.update({
            "sub_type": float_info["sub_type"],
            "is_overdue": float_info["is_overdue"],
            "business_days_elapsed": float_info["biz_days"],
            "description": float_info["description"],
        })

    elif category == ExceptionCategory.GATEWAY_FEE_MISMATCH:
        base.update({
            "expected_fee_structure": "2% MDR + 18% GST per contracted schedule",
            "verification_delta_paise": (
                verification_result.get("delta_paise", 0)
                if verification_result else 0
            ),
            "description": (
                "Fee schedule variance detected. Settlement fee deviates from "
                "contracted MDR + GST structure."
            ),
        })

    elif category == ExceptionCategory.UNRECONCILED_BANK_FEE:
        base.update({
            "description": (
                "Bank charges or debits present without corresponding "
                "gateway settlement advice."
            ),
        })

    elif category == ExceptionCategory.SPLIT_PAYOUT_PARTIAL_DROP:
        base.update({
            "description": (
                "Settlement payout covers only part of multi-order bundle "
                "or a refund was dropped from the payout."
            ),
            "verification_delta_paise": (
                verification_result.get("delta_paise", 0)
                if verification_result else 0
            ),
        })

    elif category == ExceptionCategory.CHARGEBACK_DEBIT_UNMATCHED:
        base.update({
            "description": (
                "Dispute/chargeback debit found without a corresponding "
                "order record in the merchant ledger."
            ),
        })

    elif category == ExceptionCategory.CURRENCY_CONVERSION_VARIANCE:
        base.update({
            "description": (
                "FX exchange rate variance detected on cross-border "
                "transaction."
            ),
        })

    elif category == ExceptionCategory.SUSPICIOUS_ROUND_NUMBER_DRAIN:
        base.update({
            "description": (
                "Repeated round-sum drain or duplicate payout anomaly "
                "detected."
            ),
        })

    elif category == ExceptionCategory.MISSING_SETTLEMENT_RECORD:
        base.update({
            "description": (
                "Order captured in merchant ledger with zero settlement "
                "advice or bank credit. Genuinely absent, not merely late."
            ),
        })

    elif category == ExceptionCategory.UNMAPPED_BANK_DEPOSIT:
        base.update({
            "description": (
                "Bank statement credit received with zero corresponding "
                "gateway settlement advice or order."
            ),
        })

    elif category == ExceptionCategory.ESCALATED_UNRESOLVED:
        base.update({
            "description": (
                "Human review bypassed (EVAL_MODE=true) or record is "
                "unresolvable by all automated paths."
            ),
            "reasoning_trace": reasoning_trace,
        })

    elif category == ExceptionCategory.UNACCOUNTED_LEDGER_LEAK:
        base.update({
            "description": (
                "Global batch conservation equation has a non-zero "
                "residual. Delta attributed to ledger leak."
            ),
        })

    # Attach LLM decision metadata if available
    if decision_data:
        base["llm_proposed_category"] = decision_data.get("proposed_category")
        base["llm_confidence"] = decision_data.get("confidence")
        base["llm_reasoning"] = decision_data.get("reasoning")

    return base


# ═══════════════════════════════════════════════════════════
#  NODE: CategorizeException
# ═══════════════════════════════════════════════════════════

def categorize_exception(state: ClusterState) -> dict[str, Any]:
    """CategorizeException — strict 8+3 taxonomy enforcement.

    Evaluates deterministic rule predicates against the canonical taxonomy:
    1. If a valid ExceptionCategory is already set (by LLM proposed_category
       or routing), validate it against the enum. Unknown / null / freeform →
       ESCALATED_UNRESOLVED.
    2. For TIMING_SETTLEMENT_FLOAT, apply the T+2 business-day split:
       ≤2 biz days → PENDING_FLOAT (is_overdue=False)
       >2 biz days → ANOMALOUS_FLOAT_OVERDUE (is_overdue=True)
    3. Build category-specific payload for exception_staging.

    Returns state updates with exception_category, outcome, and
    exception_payload for downstream PersistMatchOrException.
    """
    cluster = state.get("cluster", {})
    existing_trace = state.get("reasoning_trace", "")
    decision_data = state.get("decision")
    verification_result = state.get("verification_result")

    # ── 1. Determine raw category ────────────────────────
    raw_category = state.get("exception_category")

    # If LLM proposed a category and nothing is set yet, use it
    if raw_category is None and decision_data and isinstance(decision_data, dict):
        raw_category = decision_data.get("proposed_category")

    # ── 2. Strict enum validation ────────────────────────
    valid_values = {e.value for e in ExceptionCategory}
    category: ExceptionCategory

    if raw_category is not None and raw_category in valid_values:
        category = ExceptionCategory(raw_category)
    else:
        # Reject/remap unknown, null, empty, freeform to ESCALATED_UNRESOLVED
        if raw_category is not None:
            logger.warning(
                "Unknown exception category '%s' rejected; "
                "remapped to ESCALATED_UNRESOLVED",
                raw_category,
            )
        category = ExceptionCategory.ESCALATED_UNRESOLVED

    # ── 3. TIMING_SETTLEMENT_FLOAT sub-type split ────────
    is_overdue: bool | None = None
    if category == ExceptionCategory.TIMING_SETTLEMENT_FLOAT:
        float_info = _evaluate_timing_float(cluster)
        is_overdue = float_info["is_overdue"]

    # ── 4. Severity ──────────────────────────────────────
    severity = _CATEGORY_SEVERITY.get(category, ExceptionSeverity.MEDIUM)

    # Overdue float escalates severity
    if category == ExceptionCategory.TIMING_SETTLEMENT_FLOAT and is_overdue:
        severity = ExceptionSeverity.MEDIUM

    # ── 5. Build category-specific payload ───────────────
    payload = _build_category_payload(
        category,
        cluster,
        reasoning_trace=existing_trace,
        decision_data=decision_data if isinstance(decision_data, dict) else None,
        verification_result=(
            verification_result if isinstance(verification_result, dict) else None
        ),
    )

    # ── 6. Build reasoning trace ─────────────────────────
    cat_trace = (
        f" → CategorizeException: category={category.value}, "
        f"severity={severity.value}"
    )
    if category == ExceptionCategory.TIMING_SETTLEMENT_FLOAT:
        sub = payload.get("sub_type", "unknown")
        biz = payload.get("business_days_elapsed")
        cat_trace += f", sub_type={sub}, biz_days={biz}, is_overdue={is_overdue}"
    cat_trace += "."

    return {
        "outcome": "exception",
        "exception_category": category.value,
        "reasoning_trace": existing_trace + cat_trace,
        # Extra fields for PersistMatchOrException to consume
        "exception_payload": payload,
        "exception_severity": severity.value,
        "exception_is_overdue": is_overdue,
    }
