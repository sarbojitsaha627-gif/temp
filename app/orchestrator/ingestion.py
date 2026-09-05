"""
IngestBatch — Data ingestion, normalisation, and duplicate batch guard.

Responsibilities (from plan.md):
  1. Canonical UTR normalisation  (§Operational Robustness §2)
  2. Timestamp normalisation to UTC
  3. SHA-256 source checksum + duplicate batch guard  (§Operational Robustness §3)
  4. Create/resume a ReconciliationRun record

Design invariants:
  • All monetary values are integers (paise).  Zero floats.
  • canonical_utr is extracted by regex; raw_narration is preserved untouched.
  • Duplicate completed batches are rejected.
  • Partial/failed prior runs with the same checksum allow resume.
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.enums import RunStatus
from app.db.models import (
    BankTransaction,
    Dispute,
    Order,
    Refund,
    Settlement,
    SettlementItem,
    ReconciliationRun,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  CANONICAL UTR NORMALISATION  (plan.md §Operational Robustness §2)
# ═══════════════════════════════════════════════════════════

# Matches standard NEFT/RTGS/IMPS UTR formats from major Indian banks.
# Pattern: 4 uppercase letters + 12-18 alphanumeric chars, OR 12-22 digits.
_UTR_PATTERN = re.compile(
    r"([A-Z]{4}[A-Z0-9]{12,18}|[0-9]{12,22})", re.IGNORECASE
)


def extract_canonical_utr(raw_text: str | None) -> str | None:
    """Extract the core UTR identifier from CBS-mangled narration strings.

    Strips bank-specific prefixes/suffixes (HDFC, ICICI, SBI, Axis, etc.),
    truncation artefacts, and leading zeros.

    Returns
    -------
    str | None
        The canonical UTR in uppercase, or None if the input is empty.
        If the regex fails to match, the trimmed uppercase raw text is
        returned as a fallback so that FuzzyScore can still attempt
        narration-based matching.
    """
    if not raw_text:
        return None
    match = _UTR_PATTERN.search(raw_text.strip())
    return match.group(1).upper() if match else raw_text.strip().upper()


def _normalize_timestamp(ts: datetime | str | None) -> datetime | None:
    """Normalise a timestamp to UTC.

    Accepts datetime objects and ISO-format strings.
    Returns None if the input cannot be parsed.
    """
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None
    if ts.tzinfo is None:
        # Assume IST (UTC+5:30) if no timezone info
        from datetime import timedelta

        ist = timezone(timedelta(hours=5, minutes=30))
        ts = ts.replace(tzinfo=ist)
    return ts.astimezone(timezone.utc)


# ═══════════════════════════════════════════════════════════
#  SOURCE DATA CHECKSUM  (plan.md §Operational Robustness §3)
# ═══════════════════════════════════════════════════════════


def compute_source_checksum(data: bytes | str) -> str:
    """Compute SHA-256 hex digest of source file content."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# ═══════════════════════════════════════════════════════════
#  DUPLICATE BATCH GUARD
# ═══════════════════════════════════════════════════════════


async def check_duplicate_batch(
    session: AsyncSession,
    checksum: str,
) -> ReconciliationRun | None:
    """Check for a prior run with the same source checksum.

    Returns
    -------
    ReconciliationRun | None
        The prior run if a **completed** duplicate exists (caller should reject).
        None if no prior run exists or the prior run was partial/failed
        (caller should proceed or resume).

    Raises
    ------
    ValueError
        If a completed duplicate exists — includes the prior run_id and timestamp.
    """
    stmt = (
        select(ReconciliationRun)
        .where(ReconciliationRun.source_checksum == checksum)
        .order_by(ReconciliationRun.run_started_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    prior_run = result.scalar_one_or_none()

    if prior_run is None:
        return None

    if prior_run.status == RunStatus.COMPLETED:
        raise ValueError(
            f"Duplicate batch detected (matches run {prior_run.id} "
            f"completed at {prior_run.run_completed_at}). Skipping."
        )

    # Partial or failed run — allow resume
    logger.info(
        "Prior run %s found with status %s — resuming.",
        prior_run.id, prior_run.status.value,
    )
    return prior_run


# ═══════════════════════════════════════════════════════════
#  INGEST BATCH — Main ingestion function
# ═══════════════════════════════════════════════════════════


async def ingest_batch(
    session: AsyncSession,
    *,
    orders: Sequence[dict[str, Any]],
    settlements: Sequence[dict[str, Any]],
    bank_transactions: Sequence[dict[str, Any]],
    settlement_items: Sequence[dict[str, Any]] | None = None,
    refunds: Sequence[dict[str, Any]] | None = None,
    disputes: Sequence[dict[str, Any]] | None = None,
    source_checksum: str | None = None,
    eval_mode: bool = False,
) -> ReconciliationRun:
    """Ingest a batch of source data into the database.

    Steps:
      1. Create or resume a ReconciliationRun.
      2. Normalise timestamps to UTC.
      3. Extract canonical UTR from bank transaction narrations.
      4. Persist all source records with the run_id FK.

    Parameters
    ----------
    session : AsyncSession
        Active database session (caller manages transaction/commit).
    orders, settlements, bank_transactions : sequence of dicts
        Source data rows.  Keys must match model column names.
    settlement_items : sequence of dicts, optional
        Line-item breakdowns for settlements.
    refunds, disputes : sequence of dicts, optional
        Refund/dispute records.
    source_checksum : str, optional
        Pre-computed SHA-256 hex digest.  If None, duplicate guard is skipped.
    eval_mode : bool
        Passed through to the ReconciliationRun record.

    Returns
    -------
    ReconciliationRun
        The created or resumed run record.
    """
    # ── 1. Duplicate guard ───────────────────────────────────
    prior_run: ReconciliationRun | None = None
    if source_checksum:
        prior_run = await check_duplicate_batch(session, source_checksum)

    if prior_run is not None:
        # Resume: update status and return
        prior_run.status = RunStatus.IN_PROGRESS
        session.add(prior_run)
        await session.flush()
        return prior_run

    # ── 2. Create new ReconciliationRun ──────────────────────
    run = ReconciliationRun(
        id=uuid.uuid4(),
        source_checksum=source_checksum,
        status=RunStatus.IN_PROGRESS,
        eval_mode=eval_mode,
        total_records=(
            len(orders) + len(settlements) + len(bank_transactions)
            + len(refunds or []) + len(disputes or [])
        ),
    )
    session.add(run)
    await session.flush()

    run_id = run.id

    # ── 3. Ingest orders ─────────────────────────────────────
    for row in orders:
        row = dict(row)  # copy to avoid mutating caller's data
        row.setdefault("id", uuid.uuid4())
        row["reconciliation_run_id"] = run_id
        # Normalise timestamps
        for ts_field in ("order_created_at", "captured_at"):
            if ts_field in row:
                row[ts_field] = _normalize_timestamp(row[ts_field])
        session.add(Order(**row))

    # ── 4. Ingest settlements ────────────────────────────────
    settlement_uuid_map: dict[str, uuid.UUID] = {}  # settlement_id → UUID
    for row in settlements:
        row = dict(row)
        sid = uuid.uuid4()
        row.setdefault("id", sid)
        settlement_uuid_map[row.get("settlement_id", "")] = row["id"]
        row["reconciliation_run_id"] = run_id
        for ts_field in ("settlement_created_at", "value_date"):
            if ts_field in row:
                row[ts_field] = _normalize_timestamp(row[ts_field])
        session.add(Settlement(**row))

    # ── 5. Ingest settlement items ───────────────────────────
    for row in (settlement_items or []):
        row = dict(row)
        row.setdefault("id", uuid.uuid4())
        # Resolve settlement FK
        setl_id_str = row.pop("settlement_id_str", None) or row.get("settlement_id")
        if isinstance(setl_id_str, str) and setl_id_str in settlement_uuid_map:
            row["settlement_id"] = settlement_uuid_map[setl_id_str]
        session.add(SettlementItem(**row))

    # ── 6. Ingest bank transactions (with UTR normalisation) ─
    for row in bank_transactions:
        row = dict(row)
        row.setdefault("id", uuid.uuid4())
        row["reconciliation_run_id"] = run_id
        # Canonical UTR extraction (plan.md §Operational Robustness §2)
        raw_narration = row.get("raw_narration", "")
        if "canonical_utr" not in row or row["canonical_utr"] is None:
            row["canonical_utr"] = extract_canonical_utr(raw_narration)
        for ts_field in ("txn_date", "value_date"):
            if ts_field in row:
                row[ts_field] = _normalize_timestamp(row[ts_field])
        session.add(BankTransaction(**row))

    # ── 7. Ingest refunds ────────────────────────────────────
    for row in (refunds or []):
        row = dict(row)
        row.setdefault("id", uuid.uuid4())
        row["reconciliation_run_id"] = run_id
        for ts_field in ("refund_created_at",):
            if ts_field in row:
                row[ts_field] = _normalize_timestamp(row[ts_field])
        session.add(Refund(**row))

    # ── 8. Ingest disputes ───────────────────────────────────
    for row in (disputes or []):
        row = dict(row)
        row.setdefault("id", uuid.uuid4())
        row["reconciliation_run_id"] = run_id
        for ts_field in ("dispute_created_at",):
            if ts_field in row:
                row[ts_field] = _normalize_timestamp(row[ts_field])
        session.add(Dispute(**row))

    await session.flush()
    logger.info(
        "IngestBatch complete: run_id=%s, %d orders, %d settlements, "
        "%d bank_txns, %d refunds, %d disputes",
        run_id, len(orders), len(settlements), len(bank_transactions),
        len(refunds or []), len(disputes or []),
    )

    return run
