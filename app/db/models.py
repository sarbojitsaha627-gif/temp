"""
SQLAlchemy 2.0 ORM models for the Finance Controller.

Design invariants (from plan.md):
  • All monetary fields are BIGINT (paise). Zero floats anywhere.
  • allocated_paise is SIGNED — no CHECK(>0) — to support refunds/clawbacks.
  • UNIQUE(entity_type, entity_id) on match_allocations prevents double-spend.
  • exception_staging.category uses the strict 8+3 enum taxonomy.
  • reconciliation_runs stores source_checksum (SHA-256) for duplicate batch guard.
  • bank_transactions has both raw_narration and canonical_utr per UTR normalisation rule.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.engine import Base
from app.db.enums import (
    BankTxnDirection,
    EntityType,
    ExceptionCategory,
    ExceptionSeverity,
    HitlStatus,
    MatchTier,
    OrderStatus,
    RunStatus,
    SettlementStatus,
    TransactionType,
)


# ════════════════════════════════════════════════════════
#  SOURCE DATA TABLES
# ════════════════════════════════════════════════════════


class Order(Base):
    """
    Merchant-side order / payment capture record.
    Hop 1 left-hand entity.
    """

    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    order_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
        comment="Razorpay order_id (e.g. order_xxx)",
    )
    payment_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
        comment="Razorpay payment_id (e.g. pay_xxx)",
    )
    transaction_type: Mapped[TransactionType] = mapped_column(
        Enum(TransactionType, name="transaction_type_enum", create_constraint=True),
        nullable=False, default=TransactionType.PAYMENT,
    )
    status: Mapped[OrderStatus] = mapped_column(
        Enum(OrderStatus, name="order_status_enum", create_constraint=True),
        nullable=False, default=OrderStatus.CREATED,
    )
    customer_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    customer_phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    method: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="upi / card / netbanking / wallet",
    )

    # ── Monetary fields: ALL BigInteger paise, signed ───
    gross_amount_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="Total order amount in paise",
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")

    # ── Timestamps ──────────────────────────────────────
    order_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    captured_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # ── Bookkeeping ─────────────────────────────────────
    reconciliation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reconciliation_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )


class Refund(Base):
    """
    Refund issued against an order.
    Negative cash flow — allocated_paise will be negative in match_allocations.
    """

    __tablename__ = "refunds"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    refund_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
        comment="Razorpay refund_id (e.g. rfnd_xxx)",
    )
    order_id: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
        comment="FK-like reference to orders.order_id",
    )
    payment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ── Monetary: BigInteger paise, signed ──────────────
    amount_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False,
        comment="Refund amount in paise (positive value; negated at allocation time)",
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")

    # ── Timestamps ──────────────────────────────────────
    refund_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )

    reconciliation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reconciliation_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )


class Dispute(Base):
    """
    Chargeback / dispute debit against an order.
    Negative cash flow.
    """

    __tablename__ = "disputes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dispute_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
    )
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Monetary: BigInteger paise ──────────────────────
    amount_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False,
        comment="Dispute debit amount in paise (positive; negated at allocation)",
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")

    # ── Timestamps ──────────────────────────────────────
    dispute_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )

    reconciliation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reconciliation_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )


class Settlement(Base):
    """
    Gateway settlement advice (Razorpay payout batch).
    Hop 1 right-hand entity / Hop 2 left-hand entity.

    Conservation: gross_amount_paise = net_amount_paise + fee_base_paise + fee_tax_gst_paise
    """

    __tablename__ = "settlements"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    settlement_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
        comment="Razorpay settlement_id (e.g. setl_xxx)",
    )
    # UTR links settlement → bank transaction (Hop 2 join key)
    utr: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
        comment="Unique Transaction Reference for bank payout",
    )

    status: Mapped[SettlementStatus] = mapped_column(
        Enum(SettlementStatus, name="settlement_status_enum", create_constraint=True),
        nullable=False, default=SettlementStatus.CREATED,
    )

    # ── Monetary: ALL BigInteger paise ──────────────────
    gross_amount_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False,
        comment="Total gross settlement before fees",
    )
    fee_base_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0,
        comment="Platform / MDR fee in paise (pre-GST)",
    )
    fee_tax_gst_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0,
        comment="18% GST on fee_base in paise",
    )
    net_amount_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False,
        comment="Net payout = gross - fee_base - fee_tax_gst",
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")

    # ── Timestamps ──────────────────────────────────────
    settlement_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    value_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Bank value date for T+2 float calculation",
    )

    reconciliation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reconciliation_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    # ── Relationships ───────────────────────────────────
    items: Mapped[list[SettlementItem]] = relationship(
        back_populates="settlement", cascade="all, delete-orphan",
    )


class SettlementItem(Base):
    """
    Raw line-item breakdown from a payment gateway settlement report.

    This is SOURCE DATA — what the gateway claims each order contributed to
    a settlement — not a derived/ground-truth table.  Hop 1 ExactIdJoin
    reads these rows to VERIFY:
      (a) each referenced order_id exists in the orders table,
      (b) line items sum to the parent settlement's gross/net/fee totals,
      (c) per-item fees align with fee_schedule (catches GATEWAY_FEE_MISMATCH).

    If verification fails (missing order, sum mismatch, fee breach), the
    settlement is demoted to ClusterCandidates → LlmReActLoop.
    """

    __tablename__ = "settlement_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    settlement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("settlements.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    order_id: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
        comment="Claimed order_id from gateway report (references orders.order_id)",
    )

    # ── Monetary: BigInteger paise — the gateway's claimed breakdown ──
    gross_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False,
        comment="This line item's claimed gross amount in paise",
    )
    fee_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0,
        comment="This line item's claimed platform/MDR fee in paise",
    )
    tax_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0,
        comment="This line item's claimed GST on fee in paise",
    )
    net_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False,
        comment="This line item's claimed net payout in paise (gross - fee - tax)",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    # ── Relationships ───────────────────────────────────
    settlement: Mapped[Settlement] = relationship(
        back_populates="items",
    )


class BankTransaction(Base):
    """
    Bank statement line item.
    Hop 2 right-hand entity.

    Stores both raw_narration and canonical_utr (per UTR normalisation rule).
    ExactIdJoin Hop 2 matches on canonical_utr, not the raw string.
    """

    __tablename__ = "bank_transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    bank_txn_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
        comment="Bank-assigned transaction reference",
    )

    # ── UTR normalisation per plan.md §Operational Robustness §2 ──
    raw_narration: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Original CBS narration string (untouched)",
    )
    canonical_utr: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
        comment="Normalised UTR extracted by regex; Hop 2 join key",
    )

    direction: Mapped[BankTxnDirection] = mapped_column(
        Enum(BankTxnDirection, name="bank_txn_direction_enum", create_constraint=True),
        nullable=False,
    )

    # ── Monetary: BigInteger paise ──────────────────────
    amount_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False,
        comment="Credit or debit amount in paise",
    )
    bank_charges_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0,
        comment="Bank-side charges (deducted from credit)",
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")

    # ── Timestamps ──────────────────────────────────────
    txn_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        comment="Transaction posting date",
    )
    value_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    reconciliation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reconciliation_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )


class FeeSchedule(Base):
    """
    Contracted fee schedule for deterministic fee-breach detection.
    Used by CategorizeException → GATEWAY_FEE_MISMATCH.
    """

    __tablename__ = "fee_schedule"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    merchant_id: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
    )
    method: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="upi / card / netbanking / wallet",
    )

    # Rate as basis points (integer).  200 = 2.00%
    mdr_basis_points: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="MDR rate in basis points (e.g. 200 = 2%)",
    )
    gst_rate_basis_points: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1800,
        comment="GST rate on fee in basis points (1800 = 18%)",
    )

    effective_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )


# ════════════════════════════════════════════════════════
#  RECONCILIATION RESULT TABLES
# ════════════════════════════════════════════════════════


class ReconciliationRun(Base):
    """
    One batch reconciliation execution.
    source_checksum enables the duplicate batch guard (plan.md §Operational Robustness §3).
    """

    __tablename__ = "reconciliation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_checksum: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
        comment="SHA-256 hex digest of source file bytes",
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status_enum", create_constraint=True),
        nullable=False, default=RunStatus.PENDING,
    )
    eval_mode: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )

    # ── Aggregate metrics (populated at report phase) ───
    total_records: Mapped[int | None] = mapped_column(Integer, nullable=True)
    match_rate: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Match rate as decimal 0.0–1.0 (metric only, never used for money)",
    )
    ledger_leak_variance_paise: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True,
        comment="Global conservation residual in paise (signed); 0 = balanced",
    )
    processing_ms: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="Total wall-clock ms for batch",
    )
    exhausted_models: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="List of models that hit daily quota exhaustion",
    )

    # ── Timestamps ──────────────────────────────────────
    run_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    run_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # ── Relationships ───────────────────────────────────
    match_groups: Mapped[list[MatchGroup]] = relationship(
        back_populates="reconciliation_run", cascade="all, delete-orphan",
    )
    exceptions: Mapped[list[ExceptionStaging]] = relationship(
        back_populates="reconciliation_run", cascade="all, delete-orphan",
    )


class MatchGroup(Base):
    """
    One reconciliation match result (parent).
    Replaces the original spec's reconciliation_results entirely.

    Fields per plan.md Decision #5:
      tier, confidence_score, verified, residual_paise, reasoning_trace,
      cited_evidence, hitl_status, original_tier_hint, processing_ms.
    """

    __tablename__ = "match_groups"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    reconciliation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reconciliation_runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    tier: Mapped[MatchTier] = mapped_column(
        Enum(MatchTier, name="match_tier_enum", create_constraint=True),
        nullable=False,
    )
    confidence_score: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="0.0–1.0 match confidence (metric only, never money)",
    )
    verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        comment="True iff IndependentVerifier passed integer paise check",
    )
    residual_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0,
        comment="Signed residual after verification (0 = perfect balance)",
    )
    reasoning_trace: Mapped[str | None] = mapped_column(Text, nullable=True)
    cited_evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    hitl_status: Mapped[HitlStatus] = mapped_column(
        Enum(HitlStatus, name="hitl_status_enum", create_constraint=True),
        nullable=False, default=HitlStatus.NOT_REQUIRED,
    )
    original_tier_hint: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
        comment="Provenance tag if demoted (e.g. exact_id_conservation_failed)",
    )
    model_used: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
        comment="Model that successfully resolved the cluster",
    )
    processing_ms: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="Wall-clock ms for this match group",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    # ── Relationships ───────────────────────────────────
    reconciliation_run: Mapped[ReconciliationRun] = relationship(
        back_populates="match_groups",
    )
    allocations: Mapped[list[MatchAllocation]] = relationship(
        back_populates="match_group", cascade="all, delete-orphan",
    )


class MatchAllocation(Base):
    """
    Individual entity participation in a match group (M:N junction).

    Invariants:
      • UNIQUE(entity_type, entity_id) — zero double-spend across both hops.
      • allocated_paise is SIGNED BigInteger — no CHECK(>0) — supports
        refund/clawback negative cash flows.
    """

    __tablename__ = "match_allocations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("match_groups.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    entity_type: Mapped[EntityType] = mapped_column(
        Enum(EntityType, name="entity_type_enum", create_constraint=True),
        nullable=False,
    )
    entity_id: Mapped[str] = mapped_column(
        String(64), nullable=False,
        comment="References the domain ID in the source table (order_id, refund_id, etc.)",
    )
    allocated_paise: Mapped[int] = mapped_column(
        BigInteger, nullable=False,
        comment="Signed allocation amount in paise; negative for refunds/clawbacks",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    # ── Constraints ─────────────────────────────────────
    __table_args__ = (
        UniqueConstraint("entity_type", "entity_id", name="uq_allocation_entity"),
        Index("ix_allocation_entity_lookup", "entity_type", "entity_id"),
    )

    # ── Relationships ───────────────────────────────────
    match_group: Mapped[MatchGroup] = relationship(
        back_populates="allocations",
    )


class ExceptionStaging(Base):
    """
    Staged exception records — one per unmatched / unresolvable entity.

    category uses the strict 8+3 enum taxonomy from plan.md Section 3.
    """

    __tablename__ = "exception_staging"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    reconciliation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reconciliation_runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    category: Mapped[ExceptionCategory] = mapped_column(
        Enum(ExceptionCategory, name="exception_category_enum", create_constraint=True),
        nullable=False,
    )
    severity: Mapped[ExceptionSeverity] = mapped_column(
        Enum(ExceptionSeverity, name="exception_severity_enum", create_constraint=True),
        nullable=False, default=ExceptionSeverity.MEDIUM,
    )

    entity_type: Mapped[str | None] = mapped_column(
        String(32), nullable=True, comment="Source entity type if applicable",
    )
    entity_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="Source entity ID if applicable",
    )

    variance_paise: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True,
        comment="Signed variance / residual in paise (e.g. leak delta)",
    )
    is_overdue: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True,
        comment=(
            "For TIMING_SETTLEMENT_FLOAT only: "
            "False = PENDING_FLOAT (≤T+2), True = ANOMALOUS_FLOAT_OVERDUE (>T+2). "
            "NULL for all other categories."
        ),
    )
    payload: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="Structured diagnostic metadata (reasoning, evidence, etc.)",
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    # ── Relationships ───────────────────────────────────
    reconciliation_run: Mapped[ReconciliationRun] = relationship(
        back_populates="exceptions",
    )
