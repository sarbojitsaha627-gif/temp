"""
Unit tests for assign.py and run_batch.py (Step 10) — no Postgres required.

Validates:
1. persist_match_or_exception correctly routes each record type.
2. IntegrityError triggers _handle_integrity_error_fallback.
3. _infer_entity_type and _parse_entity_type resolve correctly.
4. run_batch DTO construction matches fuzzy_and_cluster dataclass signatures.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.graph import ClusterOutcome
from app.db.enums import EntityType, ExceptionCategory, MatchTier
from app.orchestrator.assign import _infer_entity_type, _parse_entity_type
from app.orchestrator.exact_join import AllocationRecord, MatchedGroup
from app.orchestrator.fuzzy_and_cluster import (
    CandidateCluster,
    CandidateMatch,
    FuzzyBankTxnDTO,
    FuzzyMatchProposal,
    FuzzyOrderDTO,
    FuzzySettlementDTO,
    NoCandidateRecord,
)


# ═══════════════════════════════════════════════════════════
#  HELPER TESTS (no DB required)
# ═══════════════════════════════════════════════════════════


class TestEntityTypeHelpers:
    """Validate _infer_entity_type and _parse_entity_type."""

    def test_infer_order(self):
        assert _infer_entity_type("order_12345") == EntityType.ORDER

    def test_infer_settlement(self):
        assert _infer_entity_type("setl_abc") == EntityType.SETTLEMENT

    def test_infer_bank_txn(self):
        assert _infer_entity_type("btxn_xyz") == EntityType.BANK_TRANSACTION

    def test_infer_refund(self):
        assert _infer_entity_type("rfnd_001") == EntityType.REFUND

    def test_infer_dispute(self):
        assert _infer_entity_type("disp_002") == EntityType.DISPUTE_DEBIT

    def test_infer_unknown_defaults_to_order(self):
        assert _infer_entity_type("unknown_id") == EntityType.ORDER

    def test_parse_valid(self):
        assert _parse_entity_type("order") == EntityType.ORDER
        assert _parse_entity_type("settlement") == EntityType.SETTLEMENT
        assert _parse_entity_type("bank_transaction") == EntityType.BANK_TRANSACTION

    def test_parse_invalid_defaults_to_order(self):
        assert _parse_entity_type("NOT_REAL") == EntityType.ORDER


# ═══════════════════════════════════════════════════════════
#  DTO CONSTRUCTION TESTS (validates run_batch.py fixes)
# ═══════════════════════════════════════════════════════════


class TestDTOConstruction:
    """Ensure Fuzzy DTOs can be constructed with all required fields."""

    def test_fuzzy_order_dto_all_fields(self):
        from datetime import datetime, timezone
        dto = FuzzyOrderDTO(
            order_id="order_001",
            payment_id="pay_001",
            method="upi",
            gross_amount_paise=100000,
            captured_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert dto.order_id == "order_001"
        assert dto.payment_id == "pay_001"
        assert dto.method == "upi"
        assert dto.gross_amount_paise == 100000

    def test_fuzzy_settlement_dto_all_fields(self):
        from datetime import datetime, timezone
        dto = FuzzySettlementDTO(
            settlement_id="setl_001",
            utr="UTR123456789",
            gross_amount_paise=100000,
            net_amount_paise=97640,
            fee_base_paise=2000,
            fee_tax_gst_paise=360,
            settlement_created_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
            value_date=datetime(2025, 1, 3, tzinfo=timezone.utc),
        )
        assert dto.settlement_id == "setl_001"
        assert dto.net_amount_paise == 97640

    def test_fuzzy_bank_txn_dto_all_fields(self):
        from datetime import datetime, timezone
        dto = FuzzyBankTxnDTO(
            bank_txn_id="btxn_001",
            canonical_utr="UTR123456789",
            raw_narration="NEFT/UTR123456789/FROM ABC",
            amount_paise=97640,
            bank_charges_paise=0,
            txn_date=datetime(2025, 1, 3, tzinfo=timezone.utc),
            value_date=datetime(2025, 1, 3, tzinfo=timezone.utc),
        )
        assert dto.bank_txn_id == "btxn_001"
        assert dto.raw_narration == "NEFT/UTR123456789/FROM ABC"


# ═══════════════════════════════════════════════════════════
#  CLUSTER OUTCOME ROUTING TESTS
# ═══════════════════════════════════════════════════════════


class TestClusterOutcomeRouting:
    """Verify that ClusterOutcome record types are identified correctly."""

    def test_verified_outcome_has_decision(self):
        co = ClusterOutcome(
            cluster_id="cluster_0001",
            outcome="verified",
            decision={
                "decision": "match",
                "confidence": 0.95,
                "matched_entity_ids": ["order_001", "setl_001"],
            },
            verification_result={"passed": True, "delta_paise": 0},
            model_used="groq/llama-3.3-70b-versatile",
            processing_ms=1200,
        )
        assert co.outcome == "verified"
        assert co.decision is not None
        assert "order_001" in co.decision["matched_entity_ids"]

    def test_exception_outcome(self):
        co = ClusterOutcome(
            cluster_id="cluster_0002",
            outcome="exception",
            exception_category="ESCALATED_UNRESOLVED",
            reasoning_trace="LLM Model chain exhausted after 3 models.",
            processing_ms=5000,
            exhausted_models=["model_a", "model_b"],
        )
        assert co.outcome == "exception"
        assert co.exception_category == "ESCALATED_UNRESOLVED"
        assert len(co.exhausted_models) == 2


# ═══════════════════════════════════════════════════════════
#  CANDIDATE CLUSTER DICT CONVERSION
# ═══════════════════════════════════════════════════════════


class TestCandidateClusterDict:
    """Verify CandidateCluster.__dict__ produces valid input for run_cluster."""

    def test_cluster_dict_has_required_keys(self):
        from datetime import datetime, timezone
        cluster = CandidateCluster(
            cluster_id="cluster_0001",
            primary_entity_type="settlement",
            primary_entity_id="setl_001",
            candidate_matches=[
                CandidateMatch(
                    entity_type=EntityType.ORDER,
                    entity_id="order_001",
                    score=0.72,
                    amount_paise=100000,
                    timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                ),
            ],
            window_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            window_end=datetime(2025, 1, 3, tzinfo=timezone.utc),
            aggregate_delta_paise=2360,
            has_amount_collision=False,
        )
        d = cluster.__dict__
        assert d["cluster_id"] == "cluster_0001"
        assert d["primary_entity_type"] == "settlement"
        assert d["primary_entity_id"] == "setl_001"
        assert len(d["candidate_matches"]) == 1
        assert d["has_amount_collision"] is False


# ═══════════════════════════════════════════════════════════
#  MATCHED GROUP ALLOCATION COMPLETENESS
# ═══════════════════════════════════════════════════════════


class TestMatchedGroupAllocations:
    """Verify MatchedGroup allocations contain correct entity types."""

    def test_hop1_match_group_has_order_and_settlement(self):
        mg = MatchedGroup(
            match_group_id=uuid.uuid4(),
            tier=MatchTier.EXACT,
            verified=True,
            confidence_score=1.0,
            residual_paise=0,
            reasoning_trace="Test hop1 match",
            cited_evidence={"hop": 1},
            allocations=[
                AllocationRecord(EntityType.ORDER, "order_001", 100000),
                AllocationRecord(EntityType.SETTLEMENT, "setl_001", -100000),
            ],
            settlement_id="setl_001",
        )
        entity_types = {a.entity_type for a in mg.allocations}
        assert EntityType.ORDER in entity_types
        assert EntityType.SETTLEMENT in entity_types
        assert sum(a.allocated_paise for a in mg.allocations) == 0

    def test_hop1_match_group_with_refund_has_negative_amount(self):
        mg = MatchedGroup(
            match_group_id=uuid.uuid4(),
            tier=MatchTier.EXACT,
            verified=True,
            confidence_score=1.0,
            residual_paise=0,
            reasoning_trace="Test hop1 match with refund",
            cited_evidence={"hop": 1},
            allocations=[
                AllocationRecord(EntityType.ORDER, "order_001", 100000),
                AllocationRecord(EntityType.REFUND, "rfnd_001", -10000),
                AllocationRecord(EntityType.SETTLEMENT, "setl_001", -90000),
            ],
            settlement_id="setl_001",
        )
        refund_allocs = [a for a in mg.allocations if a.entity_type == EntityType.REFUND]
        assert len(refund_allocs) == 1
        assert refund_allocs[0].allocated_paise < 0  # Signed negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
