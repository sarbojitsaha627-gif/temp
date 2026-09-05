"""
Domain enumerations for the Finance Controller schema.

These Python enums map 1:1 to Postgres ENUM types created in the migration.
Every value matches the canonical taxonomy from plan.md.
"""
from __future__ import annotations

import enum


# ── Match tier (match_groups.tier) ──────────────────────
class MatchTier(str, enum.Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    LLM = "llm"
    HUMAN = "human"


# ── HITL status (match_groups.hitl_status) ──────────────
class HitlStatus(str, enum.Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EVAL_BYPASSED = "eval_bypassed"


# ── Entity types participating in allocations ───────────
class EntityType(str, enum.Enum):
    ORDER = "order"
    REFUND = "refund"
    DISPUTE_DEBIT = "dispute_debit"
    SETTLEMENT = "settlement"
    BANK_TRANSACTION = "bank_transaction"


# ── Exception category — canonical 8+3 taxonomy ────────
class ExceptionCategory(str, enum.Enum):
    # 8 core domain categories
    TIMING_SETTLEMENT_FLOAT = "TIMING_SETTLEMENT_FLOAT"
    GATEWAY_FEE_MISMATCH = "GATEWAY_FEE_MISMATCH"
    UNRECONCILED_BANK_FEE = "UNRECONCILED_BANK_FEE"
    SPLIT_PAYOUT_PARTIAL_DROP = "SPLIT_PAYOUT_PARTIAL_DROP"
    CHARGEBACK_DEBIT_UNMATCHED = "CHARGEBACK_DEBIT_UNMATCHED"
    CURRENCY_CONVERSION_VARIANCE = "CURRENCY_CONVERSION_VARIANCE"
    SUSPICIOUS_ROUND_NUMBER_DRAIN = "SUSPICIOUS_ROUND_NUMBER_DRAIN"
    MISSING_SETTLEMENT_RECORD = "MISSING_SETTLEMENT_RECORD"
    # +1 addition from taxonomy mapping (unmapped bank deposit)
    UNMAPPED_BANK_DEPOSIT = "UNMAPPED_BANK_DEPOSIT"
    # 3 operational/additive categories  (but only 2 left after moving
    # UNMAPPED_BANK_DEPOSIT into the domain set — so 9 domain + 2 operational = 11 total)
    ESCALATED_UNRESOLVED = "ESCALATED_UNRESOLVED"
    UNACCOUNTED_LEDGER_LEAK = "UNACCOUNTED_LEDGER_LEAK"


# ── Exception severity ──────────────────────────────────
class ExceptionSeverity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Order transaction type ──────────────────────────────
class TransactionType(str, enum.Enum):
    PAYMENT = "payment"
    AUTHORIZATION = "authorization"


# ── Order / Settlement status ───────────────────────────
class OrderStatus(str, enum.Enum):
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    SETTLED = "settled"
    REFUNDED = "refunded"
    FAILED = "failed"


class SettlementStatus(str, enum.Enum):
    CREATED = "created"
    PROCESSED = "processed"
    SETTLED = "settled"
    FAILED = "failed"


# ── Reconciliation run status ───────────────────────────
class RunStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ── Bank transaction direction ──────────────────────────
class BankTxnDirection(str, enum.Enum):
    CREDIT = "credit"
    DEBIT = "debit"
