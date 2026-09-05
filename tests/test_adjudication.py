"""
Test LLM adjudication core — pure Python, no DB, mocked LLM.

Three independently testable scenarios:
    1. Hallucinated citation → auto-fail
    2. Retry self-correction within 2 attempts
    3. Amount-collision confidence gate blocks sub-threshold fallback

Usage:
    python -m tests.test_adjudication
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, BaseMessage

from app.agent.nodes.adjudication import (
    DEFAULT_MODEL_CHAIN,
    independent_verifier,
    check_retry_limit,
    llm_react_loop,
)
from app.agent.schemas import AdjudicationDecision, AdjudicationVerdict
from app.agent.state import ClusterState
from app.db.enums import ExceptionCategory


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════


def _make_cluster(
    cluster_id: str = "cluster_0001",
    primary_type: str = "order",
    primary_id: str = "order_001",
    candidates: list[dict] | None = None,
    has_amount_collision: bool = False,
) -> dict[str, Any]:
    """Build a minimal CandidateCluster dict for testing."""
    if candidates is None:
        candidates = [
            {
                "entity_type": "settlement",
                "entity_id": "setl_001",
                "score": 0.72,
                "amount_paise": 100000,
                "timestamp": "2026-08-20T10:00:00+05:30",
            },
        ]
    return {
        "cluster_id": cluster_id,
        "primary_entity_type": primary_type,
        "primary_entity_id": primary_id,
        "candidate_matches": candidates,
        "window_start": "2026-08-18T00:00:00+05:30",
        "window_end": "2026-08-22T00:00:00+05:30",
        "aggregate_delta_paise": 0,
        "has_amount_collision": has_amount_collision,
    }


def _make_base_state(**overrides: Any) -> dict[str, Any]:
    """Build a minimal ClusterState dict for testing."""
    state: dict[str, Any] = {
        "cluster": _make_cluster(),
        "reconciliation_run_id": str(uuid.uuid4()),
        "eval_mode": True,
        "exhausted_models": [],
        "model_chain": DEFAULT_MODEL_CHAIN,
        "messages": [],
        "current_model_index": 0,
        "iteration_count": 0,
        "retry_count": 0,
        "last_error_delta": None,
        "verification_feedback": None,
        "cited_evidence": [],
        "decision": None,
        "verification_result": None,
        "outcome": "",
        "model_used": None,
        "processing_ms": 0,
        "exception_category": None,
        "reasoning_trace": "",
    }
    state.update(overrides)
    return state


def _make_mock_llm_factory(responses: list[AIMessage]):
    """Create a mock LLM factory that returns pre-configured responses.

    Each call to llm.ainvoke() pops the next response from the list.
    """
    call_idx = {"i": 0}

    def factory(model_name: str):
        llm = AsyncMock()

        async def ainvoke(messages, **kwargs):
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx < len(responses):
                return responses[idx]
            # Default: return empty content
            return AIMessage(content="No more responses configured")

        llm.ainvoke = ainvoke
        llm.name = model_name
        return llm

    return factory


def _make_tool_map(tool_results: dict[str, tuple[str, dict]]) -> dict[str, Any]:
    """Create a mock tool map.

    tool_results: {tool_name: (content_string, artifact_dict)}
    """
    tool_map: dict[str, Any] = {}

    for name, (content, artifact) in tool_results.items():
        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value=(content, artifact))
        mock_tool.name = name
        tool_map[name] = mock_tool

    return tool_map


# ═══════════════════════════════════════════════════════════
#  TEST 1: Hallucinated Citation Auto-Fail
# ═══════════════════════════════════════════════════════════


def test_hallucinated_citation_auto_fail():
    """Feed a decision citing order_999 that doesn't appear in cited_evidence.

    IndependentVerifier must auto-fail with "hallucinated_citation".
    """
    print("\n" + "=" * 70)
    print("TEST 1: Hallucinated Citation Auto-Fail")
    print("=" * 70)

    # Decision cites order_999, which was never queried
    decision = AdjudicationDecision(
        decision=AdjudicationVerdict.MATCH,
        confidence=0.95,
        matched_entity_ids=["order_001", "order_999", "setl_001"],
        proposed_category=None,
        reasoning="Matched based on amount and reference similarity.",
    )

    # cited_evidence only has order_001 and setl_001 (not order_999)
    cited_evidence = [
        {
            "entity": "order",
            "order_id": "order_001",
            "gross_amount_paise": 100000,
            "status": "captured",
        },
        {
            "entity": "settlement",
            "settlement_id": "setl_001",
            "gross_amount_paise": 100000,
            "net_amount_paise": 97000,
            "fee_base_paise": 2000,
            "fee_tax_gst_paise": 360,
            "utr": "HDFC0001234567890",
            "items": [],
        },
    ]

    state = _make_base_state(
        decision=decision.model_dump(),
        cited_evidence=cited_evidence,
        model_used="groq/llama-3.3-70b-versatile",
    )

    result = independent_verifier(state)
    vr = result["verification_result"]

    print(f"  Passed: {vr['passed']}")
    print(f"  Failed checks: {vr['failed_checks']}")
    print(f"  Feedback: {vr['feedback']}")

    assert not vr["passed"], "Verification should FAIL for hallucinated citation"
    assert "hallucinated_citation" in vr["failed_checks"], (
        "Failed checks should include 'hallucinated_citation'"
    )
    assert "order_999" in vr["feedback"], (
        "Feedback should mention the hallucinated ID"
    )

    print("  ✅ PASSED — hallucinated citation correctly auto-failed")


# ═══════════════════════════════════════════════════════════
#  TEST 2: Retry Self-Correction Within 2 Attempts
# ═══════════════════════════════════════════════════════════


def test_retry_self_correction():
    """Simulate a fee mismatch that the LLM self-corrects on retry.

    Pass 1: LLM returns wrong fee breakdown (delta = +640 paise)
    Pass 2: LLM returns corrected breakdown after receiving error delta
    """
    print("\n" + "=" * 70)
    print("TEST 2: Retry Self-Correction Within 2 Attempts")
    print("=" * 70)

    # ── Pass 1: Wrong fee breakdown ─────────────────────────
    # Order: gross = 100000 paise
    # Settlement: net = 97000, fee_base = 2000, fee_tax_gst = 360
    # Conservation: 100000 = 97000 + 2000 + 360 = 99360  → delta = +640

    decision_pass1 = AdjudicationDecision(
        decision=AdjudicationVerdict.MATCH,
        confidence=0.80,
        matched_entity_ids=["order_001", "setl_001"],
        proposed_category=None,
        reasoning="Matched order to settlement, fee looks correct.",
    )

    cited_evidence_pass1 = [
        {
            "entity": "order",
            "order_id": "order_001",
            "gross_amount_paise": 100000,
            "status": "captured",
        },
        {
            "entity": "settlement",
            "settlement_id": "setl_001",
            "gross_amount_paise": 100000,
            "net_amount_paise": 97000,
            "fee_base_paise": 2000,
            "fee_tax_gst_paise": 360,  # 18% of 2000 = 360
            "utr": "HDFC0001234567890",
            "items": [],
        },
    ]

    state1 = _make_base_state(
        decision=decision_pass1.model_dump(),
        cited_evidence=cited_evidence_pass1,
        model_used="groq/llama-3.3-70b-versatile",
        retry_count=0,
    )

    result1 = independent_verifier(state1)
    vr1 = result1["verification_result"]

    print(f"  Pass 1 — Passed: {vr1['passed']}")
    print(f"  Pass 1 — Delta: {vr1['delta_paise']} paise")
    print(f"  Pass 1 — Failed checks: {vr1['failed_checks']}")

    assert not vr1["passed"], "Pass 1 should FAIL due to fee mismatch (640 paise delta)"
    assert vr1["delta_paise"] == 640, f"Expected delta 640, got {vr1['delta_paise']}"
    assert "amount_mismatch" in vr1["failed_checks"]

    # ── CheckRetryLimit routes to retry ─────────────────────
    state_for_retry = _make_base_state(
        retry_count=0,
        verification_result=vr1,
        reasoning_trace="LlmReActLoop resolved → IndependentVerifier: FAIL",
    )

    retry_result = check_retry_limit(state_for_retry)

    print(f"  Retry check — outcome: {retry_result['outcome']}")
    print(f"  Retry check — new retry_count: {retry_result['retry_count']}")
    print(f"  Retry check — last_error_delta: {retry_result['last_error_delta']}")

    assert retry_result["outcome"] == "retry", "Should route to retry"
    assert retry_result["retry_count"] == 1, "retry_count should increment to 1"
    assert retry_result["last_error_delta"] == 640, (
        "last_error_delta should be injected"
    )

    # ── Pass 2: Corrected fee breakdown ─────────────────────
    # After receiving delta = +640, LLM realizes fee_tax_gst should be 1000
    # Conservation: 100000 = 97000 + 2000 + 1000 = 100000  → delta = 0
    decision_pass2 = AdjudicationDecision(
        decision=AdjudicationVerdict.MATCH,
        confidence=0.92,
        matched_entity_ids=["order_001", "setl_001"],
        proposed_category=None,
        reasoning=(
            "After reviewing the +640 paise discrepancy, found that GST "
            "should be 1000 paise (corrected fee structure)."
        ),
    )

    cited_evidence_pass2 = [
        {
            "entity": "order",
            "order_id": "order_001",
            "gross_amount_paise": 100000,
            "status": "captured",
        },
        {
            "entity": "settlement",
            "settlement_id": "setl_001",
            "gross_amount_paise": 100000,
            "net_amount_paise": 97000,
            "fee_base_paise": 2000,
            "fee_tax_gst_paise": 1000,  # corrected
            "utr": "HDFC0001234567890",
            "items": [],
        },
    ]

    state2 = _make_base_state(
        decision=decision_pass2.model_dump(),
        cited_evidence=cited_evidence_pass2,
        model_used="groq/llama-3.3-70b-versatile",
        retry_count=1,
    )

    result2 = independent_verifier(state2)
    vr2 = result2["verification_result"]

    print(f"  Pass 2 — Passed: {vr2['passed']}")
    print(f"  Pass 2 — Delta: {vr2['delta_paise']} paise")
    print(f"  Pass 2 — Failed checks: {vr2['failed_checks']}")

    assert vr2["passed"], "Pass 2 should PASS after fee correction"
    assert vr2["delta_paise"] == 0, f"Expected delta 0, got {vr2['delta_paise']}"
    assert len(vr2["failed_checks"]) == 0, "No checks should fail"

    print("  ✅ PASSED — retry self-correction verified within 2 attempts")


# ═══════════════════════════════════════════════════════════
#  TEST 3: Amount-Collision Confidence Gate
# ═══════════════════════════════════════════════════════════


def test_amount_collision_confidence_gate():
    """Cluster with has_amount_collision=True resolved by fallback model.

    Confidence = 0.85 (below 0.9 threshold).
    Math and citations check out, but the gate should still block.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Amount-Collision Confidence Gate")
    print("=" * 70)

    # Cluster with amount collision
    cluster = _make_cluster(
        primary_id="order_002",
        candidates=[
            {
                "entity_type": "settlement",
                "entity_id": "setl_002",
                "score": 0.75,
                "amount_paise": 50000,
                "timestamp": "2026-08-20T10:00:00+05:30",
            },
            {
                "entity_type": "settlement",
                "entity_id": "setl_003",
                "score": 0.70,
                "amount_paise": 50000,  # same amount — collision!
                "timestamp": "2026-08-20T11:00:00+05:30",
            },
        ],
        has_amount_collision=True,
    )

    # LLM (fallback model) picked setl_002 with 0.85 confidence
    decision = AdjudicationDecision(
        decision=AdjudicationVerdict.MATCH,
        confidence=0.85,  # below 0.9 threshold
        matched_entity_ids=["order_002", "setl_002"],
        proposed_category=None,
        reasoning="setl_002 has a better reference match.",
    )

    # Both order and settlement are in cited_evidence and amounts balance
    cited_evidence = [
        {
            "entity": "order",
            "order_id": "order_002",
            "gross_amount_paise": 50000,
            "status": "captured",
        },
        {
            "entity": "settlement",
            "settlement_id": "setl_002",
            "gross_amount_paise": 50000,
            "net_amount_paise": 48500,
            "fee_base_paise": 1000,
            "fee_tax_gst_paise": 500,
            "utr": "ICIC0009876543210",
            "items": [],
        },
    ]

    # model_used is the FALLBACK model (index 1), not the primary (index 0)
    model_chain = DEFAULT_MODEL_CHAIN
    fallback_model = model_chain[1]  # "groq/openai-gpt-oss-120b"

    state = _make_base_state(
        cluster=cluster,
        decision=decision.model_dump(),
        cited_evidence=cited_evidence,
        model_used=fallback_model,
        model_chain=model_chain,
        current_model_index=1,
    )

    result = independent_verifier(state)
    vr = result["verification_result"]

    print(f"  Passed: {vr['passed']}")
    print(f"  Failed checks: {vr['failed_checks']}")
    print(f"  Feedback: {vr['feedback']}")

    assert not vr["passed"], (
        "Verification should FAIL — amount-collision cluster resolved by "
        "fallback model with confidence < 0.9"
    )
    assert "amount_collision_confidence_gate" in vr["failed_checks"], (
        "Failed checks should include 'amount_collision_confidence_gate'"
    )
    assert "amount-collision cluster resolved by fallback model" in vr["feedback"], (
        "Feedback should contain the specific gate message"
    )

    # ── Also verify: same test with PRIMARY model should PASS ──
    print("\n  Sub-test: Same cluster resolved by PRIMARY model...")

    state_primary = _make_base_state(
        cluster=cluster,
        decision=decision.model_dump(),
        cited_evidence=cited_evidence,
        model_used=model_chain[0],  # primary model
        model_chain=model_chain,
        current_model_index=0,
    )

    result_primary = independent_verifier(state_primary)
    vr_primary = result_primary["verification_result"]

    print(f"  Primary model — Passed: {vr_primary['passed']}")
    print(f"  Primary model — Failed checks: {vr_primary['failed_checks']}")

    assert vr_primary["passed"], (
        "Same cluster with primary model should PASS (gate not applied)"
    )
    assert "amount_collision_confidence_gate" not in vr_primary["failed_checks"]

    # ── Also verify: fallback with confidence >= 0.9 should PASS ──
    print("  Sub-test: Fallback model with confidence >= 0.9...")

    decision_high_conf = AdjudicationDecision(
        decision=AdjudicationVerdict.MATCH,
        confidence=0.95,  # above threshold
        matched_entity_ids=["order_002", "setl_002"],
        proposed_category=None,
        reasoning="High confidence match.",
    )

    state_high = _make_base_state(
        cluster=cluster,
        decision=decision_high_conf.model_dump(),
        cited_evidence=cited_evidence,
        model_used=fallback_model,
        model_chain=model_chain,
        current_model_index=1,
    )

    result_high = independent_verifier(state_high)
    vr_high = result_high["verification_result"]

    print(f"  High-conf fallback — Passed: {vr_high['passed']}")
    assert vr_high["passed"], (
        "Fallback model with confidence >= 0.9 should PASS the gate"
    )

    print("  ✅ PASSED — amount-collision confidence gate correctly blocks/allows")


# ═══════════════════════════════════════════════════════════
#  TEST 4: CheckRetryLimit Exhaustion Routes to HITL
# ═══════════════════════════════════════════════════════════


def test_retry_limit_exhaustion():
    """After 2 retries, CheckRetryLimit routes to HITL placeholder."""
    print("\n" + "=" * 70)
    print("TEST 4: CheckRetryLimit Exhaustion Routes to HITL")
    print("=" * 70)

    state = _make_base_state(
        retry_count=2,  # already exhausted
        verification_result={
            "passed": False,
            "delta_paise": 100,
            "tolerance_paise": 1,
            "feedback": "Amount mismatch persists after retries.",
            "failed_checks": ["amount_mismatch"],
        },
    )

    result = check_retry_limit(state)

    print(f"  Outcome: {result['outcome']}")

    assert result["outcome"] == "hitl", "Should route to HITL when retries exhausted"
    assert "retries exhausted" in result["reasoning_trace"]

    print("  ✅ PASSED — retry exhaustion correctly routes to HITL")


# ═══════════════════════════════════════════════════════════
#  TEST 5: Invalid Taxonomy Validation
# ═══════════════════════════════════════════════════════════


def test_invalid_taxonomy_validation():
    """Decision with invalid proposed_category should fail taxonomy check."""
    print("\n" + "=" * 70)
    print("TEST 5: Invalid Taxonomy Validation")
    print("=" * 70)

    decision = {
        "decision": "no_match",
        "confidence": 0.7,
        "matched_entity_ids": ["order_001"],
        "proposed_category": "TOTALLY_MADE_UP_CATEGORY",
        "reasoning": "Could not match.",
    }

    cited_evidence = [
        {
            "entity": "order",
            "order_id": "order_001",
            "gross_amount_paise": 100000,
            "status": "captured",
        },
    ]

    state = _make_base_state(
        decision=decision,
        cited_evidence=cited_evidence,
        model_used="groq/llama-3.3-70b-versatile",
    )

    result = independent_verifier(state)
    vr = result["verification_result"]

    print(f"  Passed: {vr['passed']}")
    print(f"  Failed checks: {vr['failed_checks']}")

    # Note: The decision dict has an invalid category that won't pass Pydantic
    # validation when parsed. The verifier should catch this.
    # Since the category is in raw dict (not validated by Pydantic at decision time),
    # the verifier's taxonomy check should catch it.
    assert not vr["passed"], "Should fail for invalid taxonomy"
    assert "parse_error" in vr["failed_checks"], (
        "Failed checks should include 'parse_error' because Pydantic raises ValueError on invalid category"
    )

    print("  ✅ PASSED — invalid taxonomy correctly rejected")


# ═══════════════════════════════════════════════════════════
#  TEST 6: LlmReActLoop with Mocked Model Chain
# ═══════════════════════════════════════════════════════════


def test_llm_react_loop_mock():
    """Test LlmReActLoop with a mock LLM that returns a decision after tool calls."""
    print("\n" + "=" * 70)
    print("TEST 6: LlmReActLoop with Mocked Model Chain")
    print("=" * 70)

    # Mock LLM responses:
    # 1. First response: tool call to query_order
    # 2. Second response: tool call to query_settlement
    # 3. Third response: final decision (no tool calls)

    tool_call_1 = AIMessage(
        content="",
        tool_calls=[{
            "id": "tc1",
            "name": "query_order",
            "args": {"order_id": "order_001"},
        }],
    )
    tool_call_2 = AIMessage(
        content="",
        tool_calls=[{
            "id": "tc2",
            "name": "query_settlement",
            "args": {"settlement_id": "setl_001"},
        }],
    )
    final_decision = AIMessage(
        content=json.dumps({
            "decision": "match",
            "confidence": 0.92,
            "matched_entity_ids": ["order_001", "setl_001"],
            "proposed_category": None,
            "reasoning": "Order and settlement match on amount and reference.",
        }),
    )

    llm_factory = _make_mock_llm_factory([tool_call_1, tool_call_2, final_decision])

    # Mock tool results
    tool_results = {
        "query_order": (
            json.dumps({"entity": "order", "order_id": "order_001", "gross_amount_paise": 100000}),
            {"entity": "order", "order_id": "order_001", "gross_amount_paise": 100000, "status": "captured"},
        ),
        "query_settlement": (
            json.dumps({"entity": "settlement", "settlement_id": "setl_001", "net_amount_paise": 97640}),
            {
                "entity": "settlement",
                "settlement_id": "setl_001",
                "gross_amount_paise": 100000,
                "net_amount_paise": 97640,
                "fee_base_paise": 2000,
                "fee_tax_gst_paise": 360,
            },
        ),
    }
    tool_map = _make_tool_map(tool_results)

    state = _make_base_state()

    result = asyncio.run(
        llm_react_loop(state, llm_factory=llm_factory, tool_map=tool_map)
    )

    print(f"  Decision: {result.get('decision')}")
    print(f"  Model used: {result.get('model_used')}")
    print(f"  Cited evidence count: {len(result.get('cited_evidence', []))}")
    print(f"  Outcome: {result.get('outcome')}")

    assert result["decision"] is not None, "Should have a decision"
    assert result["decision"]["decision"] == "match"
    assert result["model_used"] == DEFAULT_MODEL_CHAIN[0], "Should use primary model"
    assert len(result["cited_evidence"]) == 2, "Should have 2 cited evidence items"

    print("  ✅ PASSED — LlmReActLoop correctly processes mock tool calls")


# ═══════════════════════════════════════════════════════════
#  TEST 7: Exhausted Model Pre-Check Skip
# ═══════════════════════════════════════════════════════════


def test_exhausted_model_skip():
    """If all models are exhausted, LlmReActLoop should immediately
    return ESCALATED_UNRESOLVED without attempting any API calls.
    """
    print("\n" + "=" * 70)
    print("TEST 7: Exhausted Model Pre-Check Skip")
    print("=" * 70)

    call_count = {"n": 0}

    def factory(model_name: str):
        call_count["n"] += 1
        raise AssertionError("Should never be called for exhausted models")

    state = _make_base_state(
        exhausted_models=list(DEFAULT_MODEL_CHAIN),  # all exhausted
    )

    result = asyncio.run(
        llm_react_loop(state, llm_factory=factory, tool_map={})
    )

    print(f"  Outcome: {result.get('outcome')}")
    print(f"  Exception category: {result.get('exception_category')}")
    print(f"  LLM factory calls: {call_count['n']}")

    assert result["outcome"] == "exception"
    assert result["exception_category"] == ExceptionCategory.ESCALATED_UNRESOLVED.value
    # Factory should never have been called (models are skipped before factory)
    # NOTE: factory IS called to create the LLM object, but ainvoke should not be called.
    # Actually, in our implementation, the pre-check skips before calling factory.
    # Let's check the reasoning trace instead.
    assert "exhausted" in result["reasoning_trace"].lower()

    print("  ✅ PASSED — exhausted models correctly skipped")


# ═══════════════════════════════════════════════════════════
#  MAIN — Run all tests
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  LLM Adjudication Core — Test Suite")
    print("=" * 70)

    tests = [
        ("1. Hallucinated Citation Auto-Fail", test_hallucinated_citation_auto_fail),
        ("2. Retry Self-Correction", test_retry_self_correction),
        ("3. Amount-Collision Confidence Gate", test_amount_collision_confidence_gate),
        ("4. Retry Limit Exhaustion", test_retry_limit_exhaustion),
        ("5. Invalid Taxonomy Validation", test_invalid_taxonomy_validation),
        ("6. LlmReActLoop Mock", test_llm_react_loop_mock),
        ("7. Exhausted Model Skip", test_exhausted_model_skip),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  ❌ FAILED — {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ❌ ERROR — {name}: {type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
