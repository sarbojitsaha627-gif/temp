"""
LangChain tools for the Finance Controller LLM agent.

Every tool call returns a tuple `(string_for_llm, raw_artifact)` by using
`@tool(response_format="content_and_artifact")`. The custom ToolNode in the
LangGraph orchestrator extracts the `raw_artifact` (a dict) and automatically
appends it to the cluster's `cited_evidence` JSONB array.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.engine import async_session
from app.db.models import (
    BankTransaction,
    Dispute,
    FeeSchedule,
    Order,
    Refund,
    Settlement,
)


@tool(response_format="content_and_artifact")
async def query_order(order_id: str) -> tuple[str, dict[str, Any]]:
    """Query a merchant order by order_id to get its gross amount, status, and timestamps."""
    async with async_session() as session:
        stmt = select(Order).where(Order.order_id == order_id)
        result = await session.execute(stmt)
        order = result.scalar_one_or_none()
        if not order:
            err = {"error": f"Order {order_id} not found."}
            return json.dumps(err), err
        
        data = {
            "entity": "order",
            "order_id": order.order_id,
            "payment_id": order.payment_id,
            "gross_amount_paise": order.gross_amount_paise,
            "status": order.status.value,
            "captured_at": order.captured_at.isoformat() if order.captured_at else None,
        }
        return json.dumps(data), data


@tool(response_format="content_and_artifact")
async def query_refund(refund_id: str) -> tuple[str, dict[str, Any]]:
    """Query a refund by refund_id to get its amount."""
    async with async_session() as session:
        stmt = select(Refund).where(Refund.refund_id == refund_id)
        result = await session.execute(stmt)
        ref = result.scalar_one_or_none()
        if not ref:
            err = {"error": f"Refund {refund_id} not found."}
            return json.dumps(err), err
        data = {
            "entity": "refund",
            "refund_id": ref.refund_id,
            "order_id": ref.order_id,
            "amount_paise": ref.amount_paise,
        }
        return json.dumps(data), data


@tool(response_format="content_and_artifact")
async def query_dispute(dispute_id: str) -> tuple[str, dict[str, Any]]:
    """Query a dispute/chargeback by dispute_id to get its amount and reason."""
    async with async_session() as session:
        stmt = select(Dispute).where(Dispute.dispute_id == dispute_id)
        result = await session.execute(stmt)
        disp = result.scalar_one_or_none()
        if not disp:
            err = {"error": f"Dispute {dispute_id} not found."}
            return json.dumps(err), err
        data = {
            "entity": "dispute",
            "dispute_id": disp.dispute_id,
            "order_id": disp.order_id,
            "amount_paise": disp.amount_paise,
            "reason": disp.reason,
        }
        return json.dumps(data), data


@tool(response_format="content_and_artifact")
async def query_settlement(settlement_id: str) -> tuple[str, dict[str, Any]]:
    """Query a settlement by settlement_id to get its breakdown and line items."""
    async with async_session() as session:
        stmt = (
            select(Settlement)
            .options(selectinload(Settlement.items))
            .where(Settlement.settlement_id == settlement_id)
        )
        result = await session.execute(stmt)
        setl = result.scalar_one_or_none()
        if not setl:
            err = {"error": f"Settlement {settlement_id} not found."}
            return json.dumps(err), err
        
        items_data = [
            {
                "order_id": item.order_id,
                "gross_paise": item.gross_paise,
                "fee_paise": item.fee_paise,
                "tax_paise": item.tax_paise,
                "net_paise": item.net_paise,
            }
            for item in setl.items
        ]
        
        data = {
            "entity": "settlement",
            "settlement_id": setl.settlement_id,
            "utr": setl.utr,
            "gross_amount_paise": setl.gross_amount_paise,
            "net_amount_paise": setl.net_amount_paise,
            "fee_base_paise": setl.fee_base_paise,
            "fee_tax_gst_paise": setl.fee_tax_gst_paise,
            "value_date": setl.value_date.isoformat() if setl.value_date else None,
            "items": items_data,
        }
        return json.dumps(data), data


@tool(response_format="content_and_artifact")
async def query_bank_transaction(bank_txn_id: str) -> tuple[str, dict[str, Any]]:
    """Query a bank transaction by bank_txn_id to get its amount and narration."""
    async with async_session() as session:
        stmt = select(BankTransaction).where(BankTransaction.bank_txn_id == bank_txn_id)
        result = await session.execute(stmt)
        btxn = result.scalar_one_or_none()
        if not btxn:
            err = {"error": f"Bank transaction {bank_txn_id} not found."}
            return json.dumps(err), err
            
        data = {
            "entity": "bank_transaction",
            "bank_txn_id": btxn.bank_txn_id,
            "amount_paise": btxn.amount_paise,
            "bank_charges_paise": btxn.bank_charges_paise,
            "raw_narration": btxn.raw_narration,
            "canonical_utr": btxn.canonical_utr,
            "txn_date": btxn.txn_date.isoformat() if btxn.txn_date else None,
        }
        return json.dumps(data), data


@tool(response_format="content_and_artifact")
async def check_fee_schedule(method: str) -> tuple[str, dict[str, Any]]:
    """Check the contracted fee schedule (MDR and GST) for a payment method (e.g. 'upi', 'card')."""
    async with async_session() as session:
        stmt = select(FeeSchedule).where(FeeSchedule.method == method)
        stmt = stmt.order_by(FeeSchedule.effective_from.desc()).limit(1)
        result = await session.execute(stmt)
        fs = result.scalar_one_or_none()
        if not fs:
            err = {"error": f"No fee schedule found for method {method}."}
            return json.dumps(err), err
        
        data = {
            "entity": "fee_schedule",
            "method": fs.method,
            "mdr_basis_points": fs.mdr_basis_points,
            "gst_rate_basis_points": fs.gst_rate_basis_points,
        }
        return json.dumps(data), data


@tool(response_format="content_and_artifact")
def flag_for_human_review(reason: str) -> tuple[str, dict[str, Any]]:
    """Flag the current cluster for human review if it cannot be resolved deterministically."""
    data = {"action": "flag_for_human_review", "reason": reason}
    return json.dumps(data), data


# ═══════════════════════════════════════════════════════════
#  DETERMINISTIC VERIFICATION (Shared Logic)
# ═══════════════════════════════════════════════════════════

def verify_amount_match_logic(
    orders_gross_paise: list[int],
    refunds_paise: list[int],
    disputes_paise: list[int],
    settlements_net_paise: list[int],
    settlements_fee_paise: list[int],
    settlements_tax_paise: list[int],
    bank_txns_amount_paise: list[int],
    bank_txns_charges_paise: list[int],
) -> dict[str, Any]:
    """
    Pure deterministic 3-way match verification with GST tolerance.
    Shared by both the LLM tool and the IndependentVerifier node.
    """
    total_orders = sum(orders_gross_paise)
    total_refunds = sum(refunds_paise)
    total_disputes = sum(disputes_paise)
    
    total_settlement_net = sum(settlements_net_paise)
    total_settlement_fee = sum(settlements_fee_paise)
    total_settlement_tax = sum(settlements_tax_paise)
    
    total_bank_amount = sum(bank_txns_amount_paise)
    total_bank_charges = sum(bank_txns_charges_paise)
    
    # Hop 1: Orders vs Settlements
    hop1_left = total_orders - total_refunds - total_disputes
    hop1_right = total_settlement_net + total_settlement_fee + total_settlement_tax
    
    # Hop 2: Settlements vs Bank Transactions
    hop2_left = total_settlement_net
    hop2_right = total_bank_amount + total_bank_charges
    
    has_hop1 = bool(orders_gross_paise or refunds_paise or disputes_paise or settlements_fee_paise or settlements_tax_paise)
    has_hop2 = bool(bank_txns_amount_paise or bank_txns_charges_paise)
    
    delta_paise = 0
    
    if has_hop1 and has_hop2:
        # Full 3-way match (Hop 1 + Hop 2):
        # expected = orders - refunds - disputes
        # actual = bank_amount + bank_charges + setl_fee + setl_tax
        expected = hop1_left
        actual = hop2_right + total_settlement_fee + total_settlement_tax
        delta_paise = expected - actual
    elif has_hop1:
        # Only Hop 1
        delta_paise = hop1_left - hop1_right
    elif has_hop2:
        # Only Hop 2
        delta_paise = hop2_left - hop2_right
    
    # GST rounding tolerance: ±1 paisa per order item (plan.md §Operational Robustness §1)
    tolerance = len(orders_gross_paise) * 1
    
    is_match = abs(delta_paise) <= tolerance
    
    return {
        "is_match": is_match,
        "delta_paise": delta_paise,
        "tolerance_paise": tolerance,
        "breakdown": {
            "orders_gross": total_orders,
            "refunds": total_refunds,
            "disputes": total_disputes,
            "settlements_net": total_settlement_net,
            "settlements_fee": total_settlement_fee,
            "settlements_tax": total_settlement_tax,
            "bank_amounts": total_bank_amount,
            "bank_charges": total_bank_charges,
        },
        "feedback": (
            f"Match {'passed' if is_match else 'failed'}. "
            f"Delta = {delta_paise} paise (Tolerance = {tolerance} paise)."
        )
    }


@tool(response_format="content_and_artifact")
def verify_amount_match(
    orders_gross_paise: list[int] = None,
    refunds_paise: list[int] = None,
    disputes_paise: list[int] = None,
    settlements_net_paise: list[int] = None,
    settlements_fee_paise: list[int] = None,
    settlements_tax_paise: list[int] = None,
    bank_txns_amount_paise: list[int] = None,
    bank_txns_charges_paise: list[int] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Verify if the monetary amounts balance across Hop 1 (Orders ↔ Settlements) 
    and Hop 2 (Settlements ↔ BankTxns), applying a ±1 paisa GST tolerance per order.
    Returns the discrepancy delta and whether the match is mathematically valid.
    """
    result = verify_amount_match_logic(
        orders_gross_paise=orders_gross_paise or [],
        refunds_paise=refunds_paise or [],
        disputes_paise=disputes_paise or [],
        settlements_net_paise=settlements_net_paise or [],
        settlements_fee_paise=settlements_fee_paise or [],
        settlements_tax_paise=settlements_tax_paise or [],
        bank_txns_amount_paise=bank_txns_amount_paise or [],
        bank_txns_charges_paise=bank_txns_charges_paise or [],
    )
    return json.dumps(result), result
