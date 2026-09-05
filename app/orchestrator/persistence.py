"""
PersistMatchOrException — Atomic persistence of reconciliation results.

Writes to:
  • match_groups + match_allocations  (for successful matches)
  • exception_staging                 (for categorised exceptions)

Design invariants (from plan.md):
  • UNIQUE(entity_type, entity_id) on match_allocations — zero double-spend.
  • Catches Postgres UniqueViolation / SQLAlchemy IntegrityError on concurrent writes.
  • On collision: rolls back, queries the winning match_group_id, re-routes
    orphaned entities to CategorizeException(ESCALATED_UNRESOLVED).
  • allocated_paise is signed BigInteger (no CHECK(>0)).
  • All operations are atomic within a single transaction.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.enums import (
    EntityType,
    ExceptionCategory,
    ExceptionSeverity,
    HitlStatus,
    MatchTier,
)
from app.db.models import (
    ExceptionStaging,
    MatchAllocation,
    MatchGroup,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  RESULT CONTAINER
# ═══════════════════════════════════════════════════════════


@dataclass
class PersistResult:
    """Result of a single persist operation."""
    success: bool
    match_group_id: uuid.UUID | None = None
    exception_id: uuid.UUID | None = None
    collision_detected: bool = False
    collision_winner_id: uuid.UUID | None = None
    error_message: str | None = None


# ═══════════════════════════════════════════════════════════
#  PERSIST MATCH GROUP
# ═══════════════════════════════════════════════════════════


async def persist_match_group(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    match_group_id: uuid.UUID | None = None,
    tier: MatchTier,
    confidence_score: float = 1.0,
    verified: bool = True,
    residual_paise: int = 0,
    reasoning_trace: str = "",
    cited_evidence: dict | None = None,
    hitl_status: HitlStatus = HitlStatus.NOT_REQUIRED,
    original_tier_hint: str | None = None,
    model_used: str | None = None,
    processing_ms: int | None = None,
    allocations: list[dict[str, Any]] | None = None,
) -> PersistResult:
    """Persist a match group with its allocations atomically.

    Handles IntegrityError (UNIQUE violation on entity_type, entity_id)
    gracefully by rolling back and querying the winner.

    Parameters
    ----------
    session : AsyncSession
        Active database session.  Caller manages commit/rollback at
        the batch level.
    run_id : UUID
        The reconciliation run ID.
    allocations : list of dicts
        Each dict: {entity_type, entity_id, allocated_paise}.

    Returns
    -------
    PersistResult
    """
    mg_id = match_group_id or uuid.uuid4()

    mg = MatchGroup(
        id=mg_id,
        reconciliation_run_id=run_id,
        tier=tier,
        confidence_score=confidence_score,
        verified=verified,
        residual_paise=residual_paise,
        reasoning_trace=reasoning_trace,
        cited_evidence=cited_evidence,
        hitl_status=hitl_status,
        original_tier_hint=original_tier_hint,
        model_used=model_used,
        processing_ms=processing_ms,
    )

    alloc_objects: list[MatchAllocation] = []
    for alloc in (allocations or []):
        entity_type_val = alloc["entity_type"]
        if isinstance(entity_type_val, str):
            entity_type_val = EntityType(entity_type_val)

        alloc_objects.append(MatchAllocation(
            id=uuid.uuid4(),
            match_group_id=mg_id,
            entity_type=entity_type_val,
            entity_id=alloc["entity_id"],
            allocated_paise=alloc["allocated_paise"],
        ))

    try:
        session.add(mg)
        for ao in alloc_objects:
            session.add(ao)
        await session.flush()

        return PersistResult(
            success=True,
            match_group_id=mg_id,
        )

    except IntegrityError as e:
        # ── UNIQUE violation on (entity_type, entity_id) ─────
        await session.rollback()
        logger.warning(
            "IntegrityError persisting match group %s: %s",
            mg_id, e,
        )

        # Query the winning match_group_id for the colliding entity
        collision_winner = await _find_collision_winner(
            session, allocations or [],
        )

        return PersistResult(
            success=False,
            match_group_id=mg_id,
            collision_detected=True,
            collision_winner_id=collision_winner,
            error_message=str(e),
        )


async def _find_collision_winner(
    session: AsyncSession,
    allocations: list[dict[str, Any]],
) -> uuid.UUID | None:
    """Query match_allocations to find which match_group claimed the entity."""
    for alloc in allocations:
        entity_type_val = alloc["entity_type"]
        if isinstance(entity_type_val, EntityType):
            entity_type_val = entity_type_val.value
        elif isinstance(entity_type_val, str):
            pass  # already a string

        stmt = (
            select(MatchAllocation.match_group_id)
            .where(
                MatchAllocation.entity_type == entity_type_val,
                MatchAllocation.entity_id == alloc["entity_id"],
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        winner = result.scalar_one_or_none()
        if winner:
            return winner
    return None


# ═══════════════════════════════════════════════════════════
#  PERSIST EXCEPTION
# ═══════════════════════════════════════════════════════════


async def persist_exception(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    category: ExceptionCategory,
    severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
    entity_type: str | None = None,
    entity_id: str | None = None,
    variance_paise: int | None = None,
    is_overdue: bool | None = None,
    payload: dict[str, Any] | None = None,
    description: str | None = None,
) -> PersistResult:
    """Persist an exception to exception_staging.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    run_id : UUID
        The reconciliation run ID.
    category : ExceptionCategory
        Must be a valid 8+3 taxonomy value.

    Returns
    -------
    PersistResult
    """
    exc_id = uuid.uuid4()

    exc = ExceptionStaging(
        id=exc_id,
        reconciliation_run_id=run_id,
        category=category,
        severity=severity,
        entity_type=entity_type,
        entity_id=entity_id,
        variance_paise=variance_paise,
        is_overdue=is_overdue,
        payload=payload,
        description=description,
    )

    try:
        session.add(exc)
        await session.flush()

        return PersistResult(
            success=True,
            exception_id=exc_id,
        )
    except IntegrityError as e:
        await session.rollback()
        logger.error(
            "Failed to persist exception %s: %s", exc_id, e,
        )
        return PersistResult(
            success=False,
            error_message=str(e),
        )


# ═══════════════════════════════════════════════════════════
#  COLLISION HANDLER — Re-route orphaned entities
# ═══════════════════════════════════════════════════════════


async def handle_collision(
    session: AsyncSession,
    *,
    run_id: uuid.UUID,
    persist_result: PersistResult,
    cluster_id: str,
    primary_entity_type: str,
    primary_entity_id: str,
) -> PersistResult:
    """Handle a UNIQUE collision by escalating the orphaned primary entity.

    When a concurrent cluster has already claimed one of the entities
    in this match group, the losing cluster's primary entity is routed
    to CategorizeException(ESCALATED_UNRESOLVED).

    Returns the PersistResult of the exception persistence.
    """
    reasoning = (
        f"Entity claimed by concurrent cluster "
        f"(winner match_group={persist_result.collision_winner_id}); "
        f"re-routed to exception staging."
    )

    return await persist_exception(
        session,
        run_id=run_id,
        category=ExceptionCategory.ESCALATED_UNRESOLVED,
        severity=ExceptionSeverity.CRITICAL,
        entity_type=primary_entity_type,
        entity_id=primary_entity_id,
        payload={
            "cluster_id": cluster_id,
            "collision_winner_match_group_id": str(persist_result.collision_winner_id),
            "reasoning_trace": reasoning,
        },
        description=reasoning,
    )
