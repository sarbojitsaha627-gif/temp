"""
Tests for the batch orchestrator (assign.py and run_batch.py).

Verifies the full pipeline end-to-end, specifically the concurrency guard
and double-spend protection for the duplicate_bank_credit scenario.
"""
from __future__ import annotations

import os
# --- Override DATABASE_URL before app imports ---
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

# --- Monkeypatch Postgres types for SQLite testing ---
import sqlalchemy
import sqlalchemy.dialects.postgresql
sqlalchemy.dialects.postgresql.JSONB = sqlalchemy.JSON
sqlalchemy.dialects.postgresql.UUID = sqlalchemy.Uuid
# -----------------------------------------------------

import asyncio
import os
import random
import uuid
from typing import Any

import pytest
from sqlalchemy import select

from app.agent.graph import ClusterOutcome
from app.db.engine import Base, async_session, engine
from app.db.enums import EntityType, ExceptionCategory, MatchTier
from app.db.models import (
    BankTransaction,
    ExceptionStaging,
    FeeSchedule,
    MatchAllocation,
    MatchGroup,
    Order,
    ReconciliationRun,
    Refund,
    Settlement,
    SettlementItem,
)
from app.orchestrator.run_batch import run_batch
from app.synthetic.generate import DataCollector, _generate_all

# Mock graph response that simulates the LLM matching everything
# in its cluster.
async def mock_run_cluster(cluster, **kwargs) -> ClusterOutcome:
    await asyncio.sleep(0.01)  # tiny delay to force concurrency overlaps
    
    return ClusterOutcome(
        cluster_id=cluster["cluster_id"],
        outcome="verified",
        decision={
            "decision": "match",
            "confidence": 0.9,
            "matched_entity_ids": [m["entity_id"] for m in cluster["candidate_matches"]] + [cluster["primary_entity_id"]],
            "proposed_category": None,
            "reasoning": "Mocked LLM verified this cluster.",
        },
        verification_result={"passed": True, "delta_paise": 0},
        model_used="mock_model",
        processing_ms=10,
    )


import pytest_asyncio

@pytest_asyncio.fixture(scope="module")
async def setup_db():
    """Create all tables in the test database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def seed_data(setup_db):
    """Generate synthetic data and populate the DB."""
    # Reset seed to get deterministic data
    random.seed(42)
    import importlib
    import app.synthetic.generate as gen_mod
    importlib.reload(gen_mod)
    
    dc = DataCollector()
    gen_mod._generate_all(dc)

    run_id = uuid.uuid4()
    
    async with async_session() as session:
        # Clear tables just in case
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        
        # Create run
        run = ReconciliationRun(id=run_id)
        session.add(run)

        # Insert orders
        for o in dc.orders:
            session.add(Order(
                id=o.id, order_id=o.order_id, payment_id=o.payment_id,
                transaction_type=o.transaction_type, status=o.status,
                customer_email=o.customer_email, customer_phone=o.customer_phone,
                method=o.method, gross_amount_paise=o.gross_amount_paise,
                currency=o.currency, order_created_at=o.order_created_at,
                captured_at=o.captured_at, reconciliation_run_id=run_id,
            ))
            
        # Insert settlements
        for s in dc.settlements:
            session.add(Settlement(
                id=s.id, settlement_id=s.settlement_id, utr=s.utr,
                status=s.status, gross_amount_paise=s.gross_amount_paise,
                fee_base_paise=s.fee_base_paise, fee_tax_gst_paise=s.fee_tax_gst_paise,
                net_amount_paise=s.net_amount_paise, currency=s.currency,
                settlement_created_at=s.settlement_created_at,
                value_date=s.value_date, reconciliation_run_id=run_id,
            ))

        # Insert items
        for si in dc.settlement_items:
            session.add(SettlementItem(
                id=si.id, settlement_id=si.settlement_id, order_id=si.order_id,
                gross_paise=si.gross_paise, fee_paise=si.fee_paise,
                tax_paise=si.tax_paise, net_paise=si.net_paise,
            ))

        # Insert bank txns
        for bt in dc.bank_txns:
            session.add(BankTransaction(
                id=bt.id, bank_txn_id=bt.bank_txn_id, raw_narration=bt.raw_narration,
                canonical_utr=bt.canonical_utr, direction=bt.direction,
                amount_paise=bt.amount_paise, bank_charges_paise=bt.bank_charges_paise,
                currency=bt.currency, txn_date=bt.txn_date, value_date=bt.value_date,
                reconciliation_run_id=run_id,
            ))

        # Insert fee schedule
        for fs in dc.fee_schedules:
            session.add(FeeSchedule(
                id=fs.id, merchant_id=fs.merchant_id, method=fs.method,
                mdr_basis_points=fs.mdr_basis_points,
                gst_rate_basis_points=fs.gst_rate_basis_points,
                effective_from=fs.effective_from, effective_to=fs.effective_to,
            ))

        await session.commit()
    
    return run_id, dc


@pytest.mark.asyncio
async def test_duplicate_bank_credit_concurrency_guard(seed_data, monkeypatch):
    """
    Test that the concurrent graph execution gracefully degrades 
    duplicate entities via IntegrityError catching.
    """
    run_id, dc = seed_data
    
    # Mock the LLM run_cluster so we don't make real API calls
    import app.orchestrator.run_batch as rb_mod
    monkeypatch.setattr(rb_mod, "run_cluster", mock_run_cluster)
    
    # Run the batch orchestration
    await run_batch(run_id, eval_mode=True)
    
    # Verify the duplicate bank credit scenario
    async with async_session() as session:
        # Find the two bank transactions associated with duplicate_bank_credit
        # From ground truth we know these are btxn_synth_035 and btxn_synth_037
        # Wait, the ground truth uses exact names, we can look up by amounts 
        # or just find all match allocations and exception stagings.
        
        # Let's inspect MatchAllocations
        allocations = (await session.execute(
            select(MatchAllocation).where(
                MatchAllocation.entity_type == EntityType.BANK_TRANSACTION
            )
        )).scalars().all()
        allocated_btxn_ids = [a.entity_id for a in allocations]
        
        # There should be exactly 51 bank txns total (based on synthetic data size)
        # But for duplicate bank credits, they share the exact amount, date, and UTR is None.
        # Since they are duplicate, the LLM is spun up for both.
        # One will succeed in persisting. The other will throw IntegrityError.
        
        exceptions = (await session.execute(
            select(ExceptionStaging).where(
                ExceptionStaging.category == ExceptionCategory.ESCALATED_UNRESOLVED
            )
        )).scalars().all()
        
        # We expect at least one ESCALATED_UNRESOLVED due to the double-spend protection
        # because the LLM is mocked to always return "match" for both clusters that contain the same order.
        assert len(exceptions) >= 1, "Expected at least one double-spend fallback to trigger."
        
        # We can specifically verify that no entity_id appears twice in MatchAllocation
        # across the entire run.
        all_allocations = (await session.execute(select(MatchAllocation))).scalars().all()
        seen = set()
        for a in all_allocations:
            key = (a.entity_type, a.entity_id)
            assert key not in seen, f"Double allocation detected for {key}!"
            seen.add(key)
        
        # Run should be marked as completed
        run = await session.get(ReconciliationRun, run_id)
        assert run.status.value == "completed"
        assert run.processing_ms > 0
