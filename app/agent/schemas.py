"""
Pydantic schemas for the LLM adjudication pipeline.

AdjudicationDecision: Structured output required from LlmReActLoop.
VerificationResult: Output of IndependentVerifier's deterministic checks.

Design invariants (from plan.md):
  • The LLM only proposes IDs, categories, and reasoning — never arithmetic.
  • proposed_category must be a valid ExceptionCategory enum value or None.
  • confidence is a float 0.0–1.0 used only for the fallback-model gate,
    never for monetary calculations.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.db.enums import ExceptionCategory


class AdjudicationVerdict(str, Enum):
    """Possible decisions from the LLM adjudication."""
    MATCH = "match"
    NO_MATCH = "no_match"
    PARTIAL = "partial"


class AdjudicationDecision(BaseModel):
    """Structured output the LLM must return after tool-calling.

    The LLM decides WHICH entities match and WHY, but never computes
    amounts — that is left to the deterministic verify_amount_match tool.
    """
    decision: AdjudicationVerdict = Field(
        description="match, no_match, or partial",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model's self-assessed confidence (0.0–1.0)",
    )
    matched_entity_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Entity IDs (order_id, settlement_id, bank_txn_id, refund_id, "
            "dispute_id) that the model believes belong to this match group. "
            "Every ID here MUST have been retrieved via a tool call."
        ),
    )
    proposed_category: Optional[str] = Field(
        default=None,
        description=(
            "If decision is no_match or partial, propose an ExceptionCategory "
            "from the 8+3 taxonomy. Must be a valid enum value or null."
        ),
    )
    reasoning: str = Field(
        description="Step-by-step explanation of the matching rationale.",
    )

    @field_validator("proposed_category")
    @classmethod
    def validate_category(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Accept valid ExceptionCategory values
        valid = {e.value for e in ExceptionCategory}
        if v not in valid:
            raise ValueError(
                f"proposed_category '{v}' is not a valid ExceptionCategory. "
                f"Valid values: {sorted(valid)}"
            )
        return v


class VerificationResult(BaseModel):
    """Output of IndependentVerifier — pure deterministic, no LLM."""
    passed: bool
    delta_paise: int = Field(
        description="Signed discrepancy: expected - calculated (paise)",
    )
    tolerance_paise: int = Field(
        description="Allowable GST rounding tolerance (±1 paisa per order item)",
    )
    feedback: str = Field(
        description="Human-readable diagnostic string for retry injection",
    )
    failed_checks: list[str] = Field(
        default_factory=list,
        description=(
            "List of check names that failed: "
            "hallucinated_citation, amount_mismatch, invalid_category, "
            "amount_collision_confidence_gate"
        ),
    )
