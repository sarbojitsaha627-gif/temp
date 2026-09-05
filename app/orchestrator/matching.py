"""
ExactIdJoin — Two-pass deterministic matching.

Hop 1: Orders/Refunds/Disputes ↔ Settlements  (via payment_id / order_id)
Hop 2: Settlements ↔ BankTransactions            (via canonical_utr)

Design invariants (from plan.md Decision #5):
  • Hop 1 conservation: Σ orders.gross - Σ refunds - Σ disputes
        = settlement.net + settlement.fee + settlement.tax
  • Hop 2 conservation: Σ settlements.net = bank_txn.credit - bank_charges
  • Confident exact matches write directly to match_groups + match_allocations.
  • ID-match with failed conservation → demoted to ClusterCandidates
        with original_tier_hint = "exact_id_conservation_failed".
  • GST rounding tolerance: ±1 paisa per order item (§Operational Robustness §1).
  • All monetary arithmetic uses signed integer paise.  Zero floats.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.enums import EntityType, MatchTier
from app.db.models import (
    BankTransaction,
    Dispute,
    MatchAllocation,
    MatchGroup,
    Order,
    Refund,
    Settlement,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  RESULT CONTAINERS
# ═══════════════════════════════════════════════════════════


@dataclass
class ExactMatchResult:
    """A confirmed exact match ready for persistence."""
    match_group_id: uuid.UUID
    tier: MatchTier
    confidence_score: float
    verified: bool
    residual_paise: int
    reasoning_trace: str
    allocations: list[dict[str, Any]]
    hop: int  # 1 or 2


@dataclass
class DemotedRecord:
    """A record whose ID matched but conservation failed — sent to clustering."""
    entity_type: str
    entity_id: str
    amount_paise: int
    original_tier_hint: str
    reasoning_trace: str
    related_entities: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime | None = None


@dataclass
class UnmatchedRecord:
    """A record with no ID match — sent to fuzzy scoring."""
    entity_type: str
    entity_id: str
    amount_paise: int
    timestamp: datetime | None = None
    raw_narration: str | None = None
    canonical_utr: str | None = None


@dataclass
class ExactJoinOutput:
    """Complete output of the two-pass ExactIdJoin."""
    hop1_matches: list[ExactMatchResult]
    hop2_matches: list[ExactMatchResult]
    demoted: list[DemotedRecord]
    unmatched_orders: list[UnmatchedRecord]
    unmatched_settlements: list[UnmatchedRecord]
    unmatched_bank_txns: list[UnmatchedRecord]


# ═══════════════════════════════════════════════════════════
#  HOP 1: Orders/Refunds/Disputes ↔ Settlements
# ═══════════════════════════════════════════════════════════


async def _run_hop1(
    session: AsyncSession,
    run_id: uuid.UUID,
) -> tuple[list[ExactMatchResult], list[DemotedRecord], list[UnmatchedRecord], list[UnmatchedRecord]]:
    """Hop 1: Match orders to settlements via settlement_items.order_id.

    For each settlement, fetches its line items and looks up
    corresponding orders.  Verifies conservation:
        Σ(items.gross) - Σ(refunds) - Σ(disputes) = settlement.net + fee + tax

    Returns (matches, demoted, unmatched_orders, unmatched_settlements).
    """
    matches: list[ExactMatchResult] = []
    demoted: list[DemotedRecord] = []

    # Load all settlements for this run, eagerly loading items
    stmt = (
        select(Settlement)
        .options(selectinload(Settlement.items))
        .where(Settlement.reconciliation_run_id == run_id)
    )
    result = await session.execute(stmt)
    settlements = list(result.scalars().all())

    # Load all orders for this run
    stmt_orders = select(Order).where(Order.reconciliation_run_id == run_id)
    result_orders = await session.execute(stmt_orders)
    all_orders = {o.order_id: o for o in result_orders.scalars().all()}

    # Load refunds for this run
    stmt_refunds = select(Refund).where(Refund.reconciliation_run_id == run_id)
    result_refunds = await session.execute(stmt_refunds)
    all_refunds: dict[str, list[Refund]] = {}
    for r in result_refunds.scalars().all():
        all_refunds.setdefault(r.order_id, []).append(r)

    # Load disputes for this run
    stmt_disputes = select(Dispute).where(Dispute.reconciliation_run_id == run_id)
    result_disputes = await session.execute(stmt_disputes)
    all_disputes: dict[str, list[Dispute]] = {}
    for d in result_disputes.scalars().all():
        all_disputes.setdefault(d.order_id, []).append(d)

    matched_order_ids: set[str] = set()
    matched_settlement_ids: set[str] = set()

    for settlement in settlements:
        if not settlement.items:
            continue

        # Collect order IDs from settlement items
        item_order_ids = [item.order_id for item in settlement.items]
        found_orders: list[Order] = []
        missing_orders: list[str] = []

        for oid in item_order_ids:
            order = all_orders.get(oid)
            if order:
                found_orders.append(order)
            else:
                missing_orders.append(oid)

        if missing_orders:
            # Some orders referenced by settlement items don't exist
            # Demote the settlement for LLM investigation
            demoted.append(DemotedRecord(
                entity_type="settlement",
                entity_id=settlement.settlement_id,
                amount_paise=settlement.gross_amount_paise,
                original_tier_hint="exact_id_missing_order_refs",
                reasoning_trace=(
                    f"Settlement {settlement.settlement_id} references "
                    f"orders {missing_orders} not found in this batch."
                ),
                related_entities=[
                    {"entity_type": "order", "entity_id": oid}
                    for oid in item_order_ids
                ],
                timestamp=settlement.settlement_created_at,
            ))
            continue

        if not found_orders:
            continue

        # ── Conservation check ─────────────────────────────
        # Left side: Σ orders.gross - Σ refunds - Σ disputes
        total_orders_gross = sum(o.gross_amount_paise for o in found_orders)
        order_ids_in_match = [o.order_id for o in found_orders]

        total_refunds = 0
        refund_records: list[Refund] = []
        for oid in order_ids_in_match:
            for ref in all_refunds.get(oid, []):
                total_refunds += ref.amount_paise
                refund_records.append(ref)

        total_disputes_val = 0
        dispute_records: list[Dispute] = []
        for oid in order_ids_in_match:
            for disp in all_disputes.get(oid, []):
                total_disputes_val += disp.amount_paise
                dispute_records.append(disp)

        left_side = total_orders_gross - total_refunds - total_disputes_val

        # Right side: settlement.net + fee + tax
        right_side = (
            settlement.net_amount_paise
            + settlement.fee_base_paise
            + settlement.fee_tax_gst_paise
        )

        delta = left_side - right_side

        # GST rounding tolerance: ±1 paisa per order item
        tolerance = len(found_orders) * 1

        if abs(delta) <= tolerance:
            # ── EXACT MATCH — conservation passed ─────────
            mg_id = uuid.uuid4()
            allocations: list[dict[str, Any]] = []

            # Orders
            for order in found_orders:
                allocations.append({
                    "entity_type": EntityType.ORDER,
                    "entity_id": order.order_id,
                    "allocated_paise": order.gross_amount_paise,
                })

            # Refunds (negative allocation)
            for ref in refund_records:
                allocations.append({
                    "entity_type": EntityType.REFUND,
                    "entity_id": ref.refund_id,
                    "allocated_paise": -ref.amount_paise,
                })

            # Disputes (negative allocation)
            for disp in dispute_records:
                allocations.append({
                    "entity_type": EntityType.DISPUTE_DEBIT,
                    "entity_id": disp.dispute_id,
                    "allocated_paise": -disp.amount_paise,
                })

            # Settlement
            allocations.append({
                "entity_type": EntityType.SETTLEMENT,
                "entity_id": settlement.settlement_id,
                "allocated_paise": settlement.net_amount_paise,
            })

            trace = (
                f"Hop1 ExactIdJoin: {len(found_orders)} orders matched "
                f"settlement {settlement.settlement_id}. "
                f"Conservation: {left_side} = {right_side} "
                f"(delta={delta}, tolerance={tolerance})."
            )
            if abs(delta) > 0:
                trace += (
                    f" GST rounding accumulation: {delta} paise "
                    f"across {len(found_orders)} orders."
                )

            matches.append(ExactMatchResult(
                match_group_id=mg_id,
                tier=MatchTier.EXACT,
                confidence_score=1.0,
                verified=True,
                residual_paise=delta,
                reasoning_trace=trace,
                allocations=allocations,
                hop=1,
            ))

            matched_order_ids.update(order_ids_in_match)
            matched_settlement_ids.add(settlement.settlement_id)
            for ref in refund_records:
                matched_order_ids.add(ref.order_id)
        else:
            # ── DEMOTION — ID match but conservation failed ─
            demoted.append(DemotedRecord(
                entity_type="settlement",
                entity_id=settlement.settlement_id,
                amount_paise=settlement.gross_amount_paise,
                original_tier_hint="exact_id_conservation_failed",
                reasoning_trace=(
                    f"Hop1 ExactIdJoin conservation failed for "
                    f"settlement {settlement.settlement_id}: "
                    f"left={left_side}, right={right_side}, "
                    f"delta={delta} (tolerance={tolerance})."
                ),
                related_entities=[
                    {"entity_type": "order", "entity_id": o.order_id}
                    for o in found_orders
                ] + [
                    {"entity_type": "settlement", "entity_id": settlement.settlement_id}
                ],
                timestamp=settlement.settlement_created_at,
            ))

    # ── Collect unmatched orders ─────────────────────────────
    unmatched_orders = [
        UnmatchedRecord(
            entity_type="order",
            entity_id=o.order_id,
            amount_paise=o.gross_amount_paise,
            timestamp=o.order_created_at,
        )
        for o in all_orders.values()
        if o.order_id not in matched_order_ids
    ]

    # ── Collect unmatched settlements ────────────────────────
    unmatched_settlements = [
        UnmatchedRecord(
            entity_type="settlement",
            entity_id=s.settlement_id,
            amount_paise=s.gross_amount_paise,
            timestamp=s.settlement_created_at,
            canonical_utr=s.utr,
        )
        for s in settlements
        if s.settlement_id not in matched_settlement_ids
        and not any(d.entity_id == s.settlement_id for d in demoted)
    ]

    return matches, demoted, unmatched_orders, unmatched_settlements


# ═══════════════════════════════════════════════════════════
#  HOP 2: Settlements ↔ BankTransactions  (via canonical_utr)
# ═══════════════════════════════════════════════════════════


async def _run_hop2(
    session: AsyncSession,
    run_id: uuid.UUID,
    hop1_matched_settlement_ids: set[str],
) -> tuple[list[ExactMatchResult], list[DemotedRecord], list[UnmatchedRecord]]:
    """Hop 2: Match settlements to bank transactions via canonical UTR.

    Conservation: settlement.net = bank_txn.amount - bank_charges

    Only processes settlements that were successfully matched in Hop 1.

    Returns (matches, demoted, unmatched_bank_txns).
    """
    matches: list[ExactMatchResult] = []
    demoted: list[DemotedRecord] = []

    # Load settlements matched in Hop 1
    stmt_setl = (
        select(Settlement)
        .where(
            Settlement.reconciliation_run_id == run_id,
            Settlement.utr.isnot(None),
        )
    )
    result_setl = await session.execute(stmt_setl)
    settlements_by_utr: dict[str, Settlement] = {}
    for s in result_setl.scalars().all():
        if s.utr:
            settlements_by_utr[s.utr.upper()] = s

    # Load bank transactions
    stmt_btx = select(BankTransaction).where(
        BankTransaction.reconciliation_run_id == run_id
    )
    result_btx = await session.execute(stmt_btx)
    all_bank_txns = list(result_btx.scalars().all())

    matched_bank_txn_ids: set[str] = set()

    for btx in all_bank_txns:
        if not btx.canonical_utr:
            continue

        settlement = settlements_by_utr.get(btx.canonical_utr)
        if settlement is None:
            continue

        # ── Conservation check ─────────────────────────────
        # settlement.net = bank_txn.amount - bank_charges
        expected = settlement.net_amount_paise
        actual = btx.amount_paise - btx.bank_charges_paise
        delta = expected - actual

        # Tolerance: ±1 paisa (single settlement → single bank txn)
        tolerance = 1

        if abs(delta) <= tolerance:
            # ── EXACT MATCH ─────────────────────────────────
            mg_id = uuid.uuid4()
            allocations = [
                {
                    "entity_type": EntityType.SETTLEMENT,
                    "entity_id": settlement.settlement_id,
                    "allocated_paise": settlement.net_amount_paise,
                },
                {
                    "entity_type": EntityType.BANK_TRANSACTION,
                    "entity_id": btx.bank_txn_id,
                    "allocated_paise": btx.amount_paise,
                },
            ]

            trace = (
                f"Hop2 ExactIdJoin: settlement {settlement.settlement_id} "
                f"(UTR={btx.canonical_utr}) matched bank_txn {btx.bank_txn_id}. "
                f"Conservation: net={expected}, "
                f"bank_amount-charges={actual} (delta={delta})."
            )

            matches.append(ExactMatchResult(
                match_group_id=mg_id,
                tier=MatchTier.EXACT,
                confidence_score=1.0,
                verified=True,
                residual_paise=delta,
                reasoning_trace=trace,
                allocations=allocations,
                hop=2,
            ))

            matched_bank_txn_ids.add(btx.bank_txn_id)
        else:
            # ── DEMOTION ────────────────────────────────────
            demoted.append(DemotedRecord(
                entity_type="bank_transaction",
                entity_id=btx.bank_txn_id,
                amount_paise=btx.amount_paise,
                original_tier_hint="exact_id_conservation_failed",
                reasoning_trace=(
                    f"Hop2 ExactIdJoin conservation failed: "
                    f"settlement {settlement.settlement_id} net={expected}, "
                    f"bank_txn {btx.bank_txn_id} amount-charges={actual}, "
                    f"delta={delta}."
                ),
                related_entities=[
                    {"entity_type": "settlement", "entity_id": settlement.settlement_id},
                    {"entity_type": "bank_transaction", "entity_id": btx.bank_txn_id},
                ],
                timestamp=btx.txn_date,
            ))

    # ── Collect unmatched bank transactions ──────────────────
    unmatched_bank_txns = [
        UnmatchedRecord(
            entity_type="bank_transaction",
            entity_id=btx.bank_txn_id,
            amount_paise=btx.amount_paise,
            timestamp=btx.txn_date,
            raw_narration=btx.raw_narration,
            canonical_utr=btx.canonical_utr,
        )
        for btx in all_bank_txns
        if btx.bank_txn_id not in matched_bank_txn_ids
        and not any(d.entity_id == btx.bank_txn_id for d in demoted)
    ]

    return matches, demoted, unmatched_bank_txns


# ═══════════════════════════════════════════════════════════
#  PUBLIC API: run_exact_join
# ═══════════════════════════════════════════════════════════


async def run_exact_join(
    session: AsyncSession,
    run_id: uuid.UUID,
) -> ExactJoinOutput:
    """Execute the two-pass ExactIdJoin.

    Hop 1: Orders ↔ Settlements (via settlement_items.order_id)
    Hop 2: Settlements ↔ BankTransactions (via canonical_utr)

    Returns an ExactJoinOutput containing exact matches, demoted records,
    and unmatched records from each hop.
    """
    # ── Hop 1 ────────────────────────────────────────────────
    hop1_matches, hop1_demoted, unmatched_orders, unmatched_settlements = (
        await _run_hop1(session, run_id)
    )

    # Track which settlements were matched in Hop 1
    hop1_matched_setl_ids: set[str] = set()
    for m in hop1_matches:
        for alloc in m.allocations:
            if alloc["entity_type"] == EntityType.SETTLEMENT:
                hop1_matched_setl_ids.add(alloc["entity_id"])

    # ── Hop 2 ────────────────────────────────────────────────
    hop2_matches, hop2_demoted, unmatched_bank_txns = await _run_hop2(
        session, run_id, hop1_matched_setl_ids,
    )

    all_demoted = hop1_demoted + hop2_demoted

    logger.info(
        "ExactIdJoin: Hop1 %d matches, Hop2 %d matches, "
        "%d demoted, %d unmatched orders, %d unmatched settlements, "
        "%d unmatched bank_txns",
        len(hop1_matches), len(hop2_matches), len(all_demoted),
        len(unmatched_orders), len(unmatched_settlements),
        len(unmatched_bank_txns),
    )

    return ExactJoinOutput(
        hop1_matches=hop1_matches,
        hop2_matches=hop2_matches,
        demoted=all_demoted,
        unmatched_orders=unmatched_orders,
        unmatched_settlements=unmatched_settlements,
        unmatched_bank_txns=unmatched_bank_txns,
    )
