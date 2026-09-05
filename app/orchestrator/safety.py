"""
BatchSafetyChecks & ReportMatchRateThroughput.

Global conservation equation evaluated across all processed rows in the batch:
  Σ orders.gross - Σ refunds - Σ disputes
  = Σ bank_txns.credit + Σ gateway_fees + Σ taxes + Σ bank_charges + Σ residuals

Any non-zero variance is attributed to UNACCOUNTED_LEDGER_LEAK.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.enums import (
    ExceptionCategory,
    ExceptionSeverity,
    RunStatus,
)
from app.db.models import (
    BankTransaction,
    Dispute,
    ExceptionStaging,
    MatchGroup,
    Order,
    ReconciliationRun,
    Refund,
    Settlement,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  RESULT CONTAINER
# ═══════════════════════════════════════════════════════════


@dataclass
class SafetyReport:
    """Final metrics and safety results for the batch."""
    total_orders: int
    total_bank_txns: int
    total_settlements: int
    matched_groups: int
    exceptions: int
    match_rate: float
    ledger_leak_variance_paise: int
    processing_ms: int
    exhausted_models: list[str]


# ═══════════════════════════════════════════════════════════
#  BATCH SAFETY CHECKS
# ═══════════════════════════════════════════════════════════


async def run_batch_safety_checks(
    session: AsyncSession,
    run_id: uuid.UUID,
) -> int:
    """Evaluate global conservation for the entire batch.

    Returns
    -------
    int
        The net discrepancy delta (variance_paise). 0 means perfectly balanced.
    """
    # Sum all orders
    stmt_o = select(func.coalesce(func.sum(Order.gross_amount_paise), 0)).where(
        Order.reconciliation_run_id == run_id
    )
    total_orders_gross = (await session.execute(stmt_o)).scalar() or 0

    # Sum all refunds
    stmt_r = select(func.coalesce(func.sum(Refund.amount_paise), 0)).where(
        Refund.reconciliation_run_id == run_id
    )
    total_refunds = (await session.execute(stmt_r)).scalar() or 0

    # Sum all disputes
    stmt_d = select(func.coalesce(func.sum(Dispute.amount_paise), 0)).where(
        Dispute.reconciliation_run_id == run_id
    )
    total_disputes = (await session.execute(stmt_d)).scalar() or 0

    # Left side (Merchant Ledger)
    left_side = total_orders_gross - total_refunds - total_disputes

    # Sum all bank transactions (credits - charges)
    stmt_bt = select(
        func.coalesce(func.sum(BankTransaction.amount_paise), 0),
        func.coalesce(func.sum(BankTransaction.bank_charges_paise), 0),
    ).where(BankTransaction.reconciliation_run_id == run_id)
    bt_res = (await session.execute(stmt_bt)).one()
    total_bank_credits = bt_res[0] or 0
    total_bank_charges = bt_res[1] or 0

    # Sum gateway fees and taxes from settlements
    stmt_s = select(
        func.coalesce(func.sum(Settlement.fee_base_paise), 0),
        func.coalesce(func.sum(Settlement.fee_tax_gst_paise), 0),
    ).where(Settlement.reconciliation_run_id == run_id)
    s_res = (await session.execute(stmt_s)).one()
    total_gateway_fees = s_res[0] or 0
    total_gateway_taxes = s_res[1] or 0

    # Sum all match group residuals (e.g. GST rounding accumulation)
    stmt_mg = select(func.coalesce(func.sum(MatchGroup.residual_paise), 0)).where(
        MatchGroup.reconciliation_run_id == run_id
    )
    total_residuals = (await session.execute(stmt_mg)).scalar() or 0

    # Right side (Actual Realized Cash + Fees + Known Residuals)
    right_side = (
        total_bank_credits
        - total_bank_charges
        + total_gateway_fees
        + total_gateway_taxes
        + total_residuals
    )

    # Note: exception staging does NOT carry monetary balances in the core ledger.
    # The variance simply represents money that came in but isn't accounted for
    # in the expected merchant ledger, or vice-versa.
    variance_paise = left_side - right_side

    if variance_paise != 0:
        logger.error(
            "BatchSafetyChecks failed for run %s: left_side=%d, right_side=%d, "
            "variance_paise=%d. Attributing to UNACCOUNTED_LEDGER_LEAK.",
            run_id, left_side, right_side, variance_paise,
        )

        exc = ExceptionStaging(
            id=uuid.uuid4(),
            reconciliation_run_id=run_id,
            category=ExceptionCategory.UNACCOUNTED_LEDGER_LEAK,
            severity=ExceptionSeverity.CRITICAL,
            variance_paise=variance_paise,
            payload={
                "merchant_ledger_expected": left_side,
                "realized_cash_plus_fees": right_side,
                "variance": variance_paise,
                "breakdown": {
                    "orders_gross": total_orders_gross,
                    "refunds": total_refunds,
                    "disputes": total_disputes,
                    "bank_credits": total_bank_credits,
                    "bank_charges": total_bank_charges,
                    "gateway_fees": total_gateway_fees,
                    "gateway_taxes": total_gateway_taxes,
                    "match_group_residuals": total_residuals,
                }
            },
            description=(
                f"Global batch conservation equation failed with a "
                f"residual variance of {variance_paise} paise."
            )
        )
        session.add(exc)
        await session.flush()
    else:
        logger.info(
            "BatchSafetyChecks passed for run %s: perfectly balanced.", run_id,
        )

    return variance_paise


# ═══════════════════════════════════════════════════════════
#  REPORT METRICS
# ═══════════════════════════════════════════════════════════


async def finalize_run_report(
    session: AsyncSession,
    run: ReconciliationRun,
    *,
    variance_paise: int,
    batch_start_time_ns: int,
    exhausted_models: set[str],
) -> SafetyReport:
    """Compute final metrics, update the run record, and mark it COMPLETED."""
    run_id = run.id

    # Count source records
    stmt_cnt = select(
        select(func.count()).select_from(Order).where(Order.reconciliation_run_id == run_id).scalar_subquery(),
        select(func.count()).select_from(BankTransaction).where(BankTransaction.reconciliation_run_id == run_id).scalar_subquery(),
        select(func.count()).select_from(Settlement).where(Settlement.reconciliation_run_id == run_id).scalar_subquery(),
    )
    cnt_res = (await session.execute(stmt_cnt)).one()
    o_cnt, bt_cnt, s_cnt = cnt_res

    # Total matched groups
    stmt_mg = select(func.count()).select_from(MatchGroup).where(
        MatchGroup.reconciliation_run_id == run_id
    )
    mg_cnt = (await session.execute(stmt_mg)).scalar() or 0

    # Total exceptions
    stmt_exc = select(func.count()).select_from(ExceptionStaging).where(
        ExceptionStaging.reconciliation_run_id == run_id
    )
    exc_cnt = (await session.execute(stmt_exc)).scalar() or 0

    # Compute match rate (matched vs total records)
    total_records = o_cnt + bt_cnt + s_cnt
    match_rate = 0.0
    if total_records > 0:
        # Number of unique entities in match_allocations
        from app.db.models import MatchAllocation
        stmt_alloc = (
            select(func.count(func.distinct(MatchAllocation.entity_id)))
            .join(MatchGroup)
            .where(MatchGroup.reconciliation_run_id == run_id)
        )
        matched_entities = (await session.execute(stmt_alloc)).scalar() or 0
        match_rate = matched_entities / total_records

    # Finalize processing time
    elapsed_ms = int((time.monotonic_ns() - batch_start_time_ns) // 1_000_000)

    run.total_records = total_records
    run.match_rate = match_rate
    run.ledger_leak_variance_paise = variance_paise
    run.processing_ms = elapsed_ms
    run.exhausted_models = list(exhausted_models)
    run.status = RunStatus.COMPLETED
    from datetime import datetime, timezone
    run.run_completed_at = datetime.now(timezone.utc)

    session.add(run)
    await session.flush()

    report = SafetyReport(
        total_orders=o_cnt,
        total_bank_txns=bt_cnt,
        total_settlements=s_cnt,
        matched_groups=mg_cnt,
        exceptions=exc_cnt,
        match_rate=match_rate,
        ledger_leak_variance_paise=variance_paise,
        processing_ms=elapsed_ms,
        exhausted_models=list(exhausted_models),
    )

    logger.info(
        "Run %s completed in %d ms: %d records, match_rate=%.2f, "
        "variance=%d, exhausted=%s",
        run_id, elapsed_ms, total_records, match_rate, variance_paise,
        report.exhausted_models,
    )

    return report
