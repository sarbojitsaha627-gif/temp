"""
Test FuzzyScore + ClusterCandidates against Step 2's synthetic data.

Key invariants verified:
  • Amount-collision clusters NEVER take the fuzzy_resolved shortcut.
  • Reference-mismatch records with clear score gap CAN be fuzzy_resolved.
  • No entity appears in more than one cluster (disjoint partitioning).
  • Zero-candidate records go to MISSING_SETTLEMENT_RECORD / UNMAPPED_BANK_DEPOSIT.
  • Clusters are capped at 8 candidates (sub-partition if exceeded).

Usage:
    python -m pytest tests/test_fuzzy_and_cluster.py -v
    python -m tests.test_fuzzy_and_cluster       # standalone runner
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db.enums import EntityType
from app.orchestrator.exact_join import (
    BankTxnDTO,
    DemotedRecord,
    ExactJoinResult,
    FeeScheduleDTO,
    MatchedGroup,
    OrderDTO,
    RefundDTO,
    SettlementDTO,
    SettlementItemDTO,
    run_exact_join_pipeline,
)
from app.orchestrator.fuzzy_and_cluster import (
    AmbiguousRecord,
    CandidateCluster,
    FuzzyAndClusterResult,
    FuzzyBankTxnDTO,
    FuzzyMatchProposal,
    FuzzyOrderDTO,
    FuzzyScoreResult,
    FuzzySettlementDTO,
    NoCandidateRecord,
    ScoredCandidate,
    UnionFind,
    run_cluster_candidates,
    run_fuzzy_and_cluster_pipeline,
    run_fuzzy_score,
)

IST = timezone(timedelta(hours=5, minutes=30))


# ═══════════════════════════════════════════════════════════
#  HELPER: build synthetic data (reused from test_exact_join)
# ═══════════════════════════════════════════════════════════


def _build_synthetic_data():
    """Run the generator in-memory and produce DTOs for both exact + fuzzy."""
    random.seed(42)
    import importlib
    import app.synthetic.generate as gen_mod
    importlib.reload(gen_mod)

    from app.synthetic.generate import DataCollector, _generate_all
    dc = DataCollector()
    _generate_all(dc)

    # Exact-join DTOs
    orders_exact = {
        o.order_id: OrderDTO(o.order_id, o.method, o.gross_amount_paise)
        for o in dc.orders
    }
    settlements_exact = {
        s.settlement_id: SettlementDTO(
            s.settlement_id, s.id, s.utr,
            s.gross_amount_paise, s.fee_base_paise,
            s.fee_tax_gst_paise, s.net_amount_paise,
        )
        for s in dc.settlements
    }
    uuid_to_sid = {s.id: s.settlement_id for s in dc.settlements}

    items_by_settlement: dict[str, list[SettlementItemDTO]] = defaultdict(list)
    for si in dc.settlement_items:
        sid = uuid_to_sid[si.settlement_id]
        items_by_settlement[sid].append(SettlementItemDTO(
            sid, si.order_id,
            si.gross_paise, si.fee_paise, si.tax_paise, si.net_paise,
        ))

    refunds_by_order: dict[str, list[RefundDTO]] = defaultdict(list)
    for r in dc.refunds:
        refunds_by_order[r.order_id].append(
            RefundDTO(r.refund_id, r.order_id, r.amount_paise)
        )

    bank_txns_by_utr: dict[str, list[BankTxnDTO]] = defaultdict(list)
    all_btxn_ids: list[str] = []
    for b in dc.bank_txns:
        dto = BankTxnDTO(
            b.bank_txn_id, b.canonical_utr,
            b.amount_paise, b.bank_charges_paise,
        )
        if b.canonical_utr:
            bank_txns_by_utr[b.canonical_utr].append(dto)
        all_btxn_ids.append(b.bank_txn_id)

    fee_schedule = {
        fs.method: FeeScheduleDTO(
            fs.method, fs.mdr_basis_points, fs.gst_rate_basis_points,
        )
        for fs in dc.fee_schedules
    }

    # Fuzzy DTOs (need timestamps)
    orders_fuzzy = {
        o.order_id: FuzzyOrderDTO(
            o.order_id, o.payment_id, o.method,
            o.gross_amount_paise, o.captured_at,
        )
        for o in dc.orders
    }
    settlements_fuzzy = {
        s.settlement_id: FuzzySettlementDTO(
            s.settlement_id, s.utr,
            s.gross_amount_paise, s.net_amount_paise,
            s.fee_base_paise, s.fee_tax_gst_paise,
            s.settlement_created_at, s.value_date,
        )
        for s in dc.settlements
    }
    bank_txns_fuzzy = {
        b.bank_txn_id: FuzzyBankTxnDTO(
            b.bank_txn_id, b.canonical_utr, b.raw_narration,
            b.amount_paise, b.bank_charges_paise,
            b.txn_date, b.value_date,
        )
        for b in dc.bank_txns
    }

    return (
        orders_exact, settlements_exact, dict(items_by_settlement),
        dict(refunds_by_order), fee_schedule,
        dict(bank_txns_by_utr), all_btxn_ids,
        orders_fuzzy, settlements_fuzzy, bank_txns_fuzzy,
        dc,
    )


def _entities_in_scenario(dc, scenario: str, entity_type: str) -> list[str]:
    return [
        row["entity_id"]
        for row in dc.ground_truth
        if row["scenario_type"] == scenario and row["entity_type"] == entity_type
    ]


# ═══════════════════════════════════════════════════════════
#  UNIT TESTS — ISOLATED SCORING
# ═══════════════════════════════════════════════════════════


def _check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "✅" if condition else "❌"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def test_amount_collision_never_fuzzy_resolved():
    """Amount collision: 2+ candidates with same amount must NOT be fuzzy_resolved."""
    print("\n── test_amount_collision_never_fuzzy_resolved ──")

    base_ts = datetime(2026, 8, 9, 12, 0, 0, tzinfo=IST)
    net_amt = 97_800  # same for both pairs

    # Two settlements with the same net (hop2_unmatched)
    setl_a = FuzzySettlementDTO(
        "setl_col_A", "HDFC1111222233334444",
        100_000, net_amt, 1500, 270,
        base_ts, base_ts + timedelta(days=1),
    )
    setl_b = FuzzySettlementDTO(
        "setl_col_B", "ICIC5555666677778888",
        100_000, net_amt, 1500, 270,
        base_ts + timedelta(hours=6), base_ts + timedelta(days=1, hours=6),
    )

    # Two bank_txns with the same amount (garbled UTRs)
    btxn_x = FuzzyBankTxnDTO(
        "btxn_col_X", "HDFC111122223333", None,
        net_amt, 0,
        base_ts + timedelta(hours=10), None,
    )
    btxn_y = FuzzyBankTxnDTO(
        "btxn_col_Y", "ICIC555566667777", None,
        net_amt, 0,
        base_ts + timedelta(hours=16), None,
    )

    settlements = {"setl_col_A": setl_a, "setl_col_B": setl_b}
    bank_txns = {"btxn_col_X": btxn_x, "btxn_col_Y": btxn_y}

    result = run_fuzzy_score(
        hop1_unmatched_order_ids=[],
        hop2_unmatched_settlement_ids=["setl_col_A", "setl_col_B"],
        hop2_unmatched_bank_txn_ids=["btxn_col_X", "btxn_col_Y"],
        hop1_matched_settlement_ids=set(),
        hop1_demoted_settlement_ids=set(),
        orders={},
        settlements=settlements,
        bank_txns=bank_txns,
    )

    ok = True
    ok &= _check(
        "No fuzzy_resolved results (collision blocks it)",
        len(result.fuzzy_resolved) == 0,
        f"got {len(result.fuzzy_resolved)}",
    )
    ok &= _check(
        "All settlements classified as ambiguous",
        any(r.entity_id == "setl_col_A" for r in result.ambiguous)
        and any(r.entity_id == "setl_col_B" for r in result.ambiguous),
        f"ambiguous ids={[r.entity_id for r in result.ambiguous]}",
    )

    # Verify the reason is amount_collision
    collision_reasons = [
        r.reason for r in result.ambiguous
        if r.entity_type == EntityType.SETTLEMENT
    ]
    ok &= _check(
        "Ambiguous reason is amount_collision",
        all(r == "amount_collision" for r in collision_reasons),
        f"reasons={collision_reasons}",
    )
    return ok


def test_reference_mismatch_can_be_fuzzy_resolved():
    """Reference mismatch: garbled UTR with clear score gap → fuzzy_resolved."""
    print("\n── test_reference_mismatch_can_be_fuzzy_resolved ──")

    base_ts = datetime(2026, 8, 7, 16, 0, 0, tzinfo=IST)
    real_utr = "BARB1937452991241904"
    garbled_utr = "BARB193745299124"  # truncated
    net_amt = 97_305

    setl = FuzzySettlementDTO(
        "setl_ref_001", real_utr,
        100_000, net_amt, 2000, 360,
        base_ts + timedelta(hours=12),
        base_ts + timedelta(hours=12, days=1),
    )

    btxn = FuzzyBankTxnDTO(
        "btxn_ref_001", garbled_utr,
        f"NEFT/{garbled_utr}/RAZORPAY SOFTWARE PVT LTD",
        net_amt, 0,
        base_ts + timedelta(hours=18), None,
    )

    settlements = {"setl_ref_001": setl}
    bank_txns = {"btxn_ref_001": btxn}

    result = run_fuzzy_score(
        hop1_unmatched_order_ids=[],
        hop2_unmatched_settlement_ids=["setl_ref_001"],
        hop2_unmatched_bank_txn_ids=["btxn_ref_001"],
        hop1_matched_settlement_ids=set(),
        hop1_demoted_settlement_ids=set(),
        orders={},
        settlements=settlements,
        bank_txns=bank_txns,
    )

    ok = True
    ok &= _check(
        "1 fuzzy_resolved match found",
        len(result.fuzzy_resolved) == 1,
        f"got {len(result.fuzzy_resolved)}",
    )

    if result.fuzzy_resolved:
        m = result.fuzzy_resolved[0]
        ok &= _check(
            "Matched setl_ref_001 ↔ btxn_ref_001",
            m.settlement_id == "setl_ref_001"
            and m.bank_txn_id == "btxn_ref_001",
            f"settlement={m.settlement_id}, bank_txn={m.bank_txn_id}",
        )
        ok &= _check(
            "Score gap >= 0.15",
            m.score_gap >= 0.15,
            f"gap={m.score_gap:.4f}",
        )
        ok &= _check(
            "Has allocations",
            len(m.allocations) > 0,
            f"n_allocs={len(m.allocations)}",
        )

    ok &= _check(
        "No ambiguous records",
        len(result.ambiguous) == 0,
        f"got {len(result.ambiguous)}",
    )
    return ok


def test_no_candidates_missing_settlement():
    """Order with zero settlement candidates → MISSING_SETTLEMENT_RECORD."""
    print("\n── test_no_candidates_missing_settlement ──")

    base_ts = datetime(2026, 7, 15, 10, 0, 0, tzinfo=IST)
    order = FuzzyOrderDTO("order_orphan_001", "pay_orphan_001", "upi", 99_900, base_ts)

    result = run_fuzzy_score(
        hop1_unmatched_order_ids=["order_orphan_001"],
        hop2_unmatched_settlement_ids=[],
        hop2_unmatched_bank_txn_ids=[],
        hop1_matched_settlement_ids=set(),
        hop1_demoted_settlement_ids=set(),
        orders={"order_orphan_001": order},
        settlements={},
        bank_txns={},
    )

    ok = True
    ok &= _check(
        "1 no_candidate record",
        len(result.no_candidates) == 1,
        f"got {len(result.no_candidates)}",
    )
    if result.no_candidates:
        nc = result.no_candidates[0]
        ok &= _check(
            "Suggested category = MISSING_SETTLEMENT_RECORD",
            nc.suggested_category == "MISSING_SETTLEMENT_RECORD",
            f"got {nc.suggested_category}",
        )
        ok &= _check(
            "Entity is order_orphan_001",
            nc.entity_id == "order_orphan_001",
        )
    return ok


def test_no_candidates_unmapped_bank_deposit():
    """Bank txn with zero settlement candidates → UNMAPPED_BANK_DEPOSIT."""
    print("\n── test_no_candidates_unmapped_bank_deposit ──")

    base_ts = datetime(2026, 8, 14, 10, 0, 0, tzinfo=IST)
    btxn = FuzzyBankTxnDTO(
        "btxn_orphan_001", "SBIN9999888877776666", None,
        5_000_000, 0, base_ts, None,
    )

    result = run_fuzzy_score(
        hop1_unmatched_order_ids=[],
        hop2_unmatched_settlement_ids=[],
        hop2_unmatched_bank_txn_ids=["btxn_orphan_001"],
        hop1_matched_settlement_ids=set(),
        hop1_demoted_settlement_ids=set(),
        orders={},
        settlements={},
        bank_txns={"btxn_orphan_001": btxn},
    )

    ok = True
    ok &= _check(
        "1 no_candidate record",
        len(result.no_candidates) == 1,
        f"got {len(result.no_candidates)}",
    )
    if result.no_candidates:
        nc = result.no_candidates[0]
        ok &= _check(
            "Suggested category = UNMAPPED_BANK_DEPOSIT",
            nc.suggested_category == "UNMAPPED_BANK_DEPOSIT",
            f"got {nc.suggested_category}",
        )
    return ok


# ═══════════════════════════════════════════════════════════
#  UNIT TESTS — UNION-FIND & CLUSTERING
# ═══════════════════════════════════════════════════════════


def test_union_find_basics():
    """Union-Find: path compression, transitivity, disjoint groups."""
    print("\n── test_union_find_basics ──")

    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    uf.union("d", "e")

    ok = True
    ok &= _check("a, b, c connected", uf.connected("a", "c"))
    ok &= _check("d, e connected", uf.connected("d", "e"))
    ok &= _check("a, d NOT connected", not uf.connected("a", "d"))

    groups = uf.groups()
    ok &= _check("2 groups total", len(groups) == 2, f"got {len(groups)}")
    return ok


def test_disjoint_partitioning_no_entity_in_two_clusters():
    """No entity should appear in more than one cluster."""
    print("\n── test_disjoint_partitioning_no_entity_in_two_clusters ──")

    base_ts = datetime(2026, 8, 9, 12, 0, 0, tzinfo=IST)

    # Create ambiguous records with overlapping candidates:
    #   order_A → [setl_X, setl_Y]
    #   order_B → [setl_Y, setl_Z]  (setl_Y shared → must merge)
    shared_amount = 99_900

    rec_a = AmbiguousRecord(
        entity_type=EntityType.ORDER,
        entity_id="order_A",
        amount_paise=shared_amount,
        timestamp=base_ts,
        candidates=[
            ScoredCandidate(EntityType.SETTLEMENT, "setl_X", 0.72, 0.5, 1.0, 0.8,
                            shared_amount, shared_amount, base_ts + timedelta(hours=4)),
            ScoredCandidate(EntityType.SETTLEMENT, "setl_Y", 0.65, 0.4, 1.0, 0.7,
                            shared_amount, shared_amount, base_ts + timedelta(hours=8)),
        ],
        reason="amount_collision",
        hop=1,
    )
    rec_b = AmbiguousRecord(
        entity_type=EntityType.ORDER,
        entity_id="order_B",
        amount_paise=shared_amount,
        timestamp=base_ts + timedelta(hours=2),
        candidates=[
            ScoredCandidate(EntityType.SETTLEMENT, "setl_Y", 0.70, 0.5, 1.0, 0.7,
                            shared_amount, shared_amount, base_ts + timedelta(hours=8)),
            ScoredCandidate(EntityType.SETTLEMENT, "setl_Z", 0.60, 0.3, 1.0, 0.7,
                            shared_amount, shared_amount, base_ts + timedelta(hours=12)),
        ],
        reason="amount_collision",
        hop=1,
    )

    # Provide settlement DTOs for the clustering
    settlements = {
        "setl_X": FuzzySettlementDTO(
            "setl_X", "UTR_X", shared_amount, shared_amount - 2000,
            1500, 270,
            base_ts + timedelta(hours=4), None,
        ),
        "setl_Y": FuzzySettlementDTO(
            "setl_Y", "UTR_Y", shared_amount, shared_amount - 2000,
            1500, 270,
            base_ts + timedelta(hours=8), None,
        ),
        "setl_Z": FuzzySettlementDTO(
            "setl_Z", "UTR_Z", shared_amount, shared_amount - 2000,
            1500, 270,
            base_ts + timedelta(hours=12), None,
        ),
    }
    orders = {
        "order_A": FuzzyOrderDTO("order_A", "pay_A", "upi", shared_amount, base_ts),
        "order_B": FuzzyOrderDTO("order_B", "pay_B", "upi", shared_amount,
                                 base_ts + timedelta(hours=2)),
    }

    clusters = run_cluster_candidates(
        ambiguous_records=[rec_a, rec_b],
        hop1_demoted=[],
        hop2_demoted=[],
        orders=orders,
        settlements=settlements,
        bank_txns={},
    )

    ok = True

    # Collect all entity_ids across all clusters
    entity_appearances: dict[str, list[str]] = defaultdict(list)
    for cl in clusters:
        entity_appearances[cl.primary_entity_id].append(cl.cluster_id)
        for cm in cl.candidate_matches:
            entity_appearances[cm.entity_id].append(cl.cluster_id)

    duplicates = {eid: cids for eid, cids in entity_appearances.items()
                  if len(cids) > 1}

    ok &= _check(
        "No entity in more than 1 cluster",
        len(duplicates) == 0,
        f"duplicates={duplicates}" if duplicates else "all unique",
    )

    # Since setl_Y is shared, order_A and order_B should be in the SAME cluster
    a_cluster = [cl.cluster_id for cl in clusters
                 if cl.primary_entity_id == "order_A"
                 or any(cm.entity_id == "order_A" for cm in cl.candidate_matches)]
    b_cluster = [cl.cluster_id for cl in clusters
                 if cl.primary_entity_id == "order_B"
                 or any(cm.entity_id == "order_B" for cm in cl.candidate_matches)]

    ok &= _check(
        "order_A and order_B in same cluster (shared setl_Y)",
        len(a_cluster) > 0 and len(b_cluster) > 0
        and a_cluster[0] == b_cluster[0],
        f"A={a_cluster}, B={b_cluster}",
    )

    # All 5 entities should be in a single cluster
    all_entity_ids = set()
    for cl in clusters:
        all_entity_ids.add(cl.primary_entity_id)
        for cm in cl.candidate_matches:
            all_entity_ids.add(cm.entity_id)

    ok &= _check(
        "All 5 entities accounted for",
        {"order_A", "order_B", "setl_X", "setl_Y", "setl_Z"} <= all_entity_ids,
        f"found={all_entity_ids}",
    )

    # has_amount_collision should be True (all same amount)
    if clusters:
        ok &= _check(
            "has_amount_collision = True",
            clusters[0].has_amount_collision,
        )

    return ok


def test_cluster_max_size_sub_partition():
    """Clusters with > 8 candidates should be sub-partitioned by 6h buckets."""
    print("\n── test_cluster_max_size_sub_partition ──")

    base_ts = datetime(2026, 8, 9, 0, 0, 0, tzinfo=IST)
    n_candidates = 12
    shared_amount = 99_900

    # 1 primary order + 12 settlement candidates, all same amount
    cands = []
    settlements = {}
    for i in range(n_candidates):
        sid = f"setl_big_{i:03d}"
        ts = base_ts + timedelta(hours=i * 3)  # spread over 36 hours
        cands.append(ScoredCandidate(
            EntityType.SETTLEMENT, sid, 0.70 - i * 0.01,
            0.5, 1.0, 0.8 - i * 0.05,
            shared_amount, shared_amount, ts,
        ))
        settlements[sid] = FuzzySettlementDTO(
            sid, f"UTR_{i:03d}", shared_amount, shared_amount - 2000,
            1500, 270, ts, None,
        )

    rec = AmbiguousRecord(
        entity_type=EntityType.ORDER,
        entity_id="order_big",
        amount_paise=shared_amount,
        timestamp=base_ts,
        candidates=cands,
        reason="amount_collision",
        hop=1,
    )

    orders = {
        "order_big": FuzzyOrderDTO("order_big", "pay_big", "upi",
                                   shared_amount, base_ts),
    }

    clusters = run_cluster_candidates(
        ambiguous_records=[rec],
        hop1_demoted=[],
        hop2_demoted=[],
        orders=orders,
        settlements=settlements,
        bank_txns={},
    )

    ok = True
    ok &= _check(
        "Multiple clusters produced (sub-partitioned)",
        len(clusters) >= 2,
        f"got {len(clusters)} clusters",
    )

    max_size = max(
        1 + len(cl.candidate_matches) for cl in clusters
    ) if clusters else 0
    ok &= _check(
        "No cluster exceeds 8 total entities",
        max_size <= 8,
        f"max_size={max_size}",
    )

    # Verify no duplicates across clusters
    entity_appearances: dict[str, int] = defaultdict(int)
    for cl in clusters:
        entity_appearances[cl.primary_entity_id] += 1
        for cm in cl.candidate_matches:
            entity_appearances[cm.entity_id] += 1

    duplicates = {e: c for e, c in entity_appearances.items() if c > 1}
    ok &= _check(
        "No entity in more than 1 sub-partition",
        len(duplicates) == 0,
        f"duplicates={duplicates}" if duplicates else "all unique",
    )
    return ok


# ═══════════════════════════════════════════════════════════
#  INTEGRATION — FULL PIPELINE AGAINST SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════


def test_full_pipeline_with_synthetic_data():
    """Run exact_join → fuzzy_and_cluster against Step 2 synthetic data."""
    print("\n── test_full_pipeline_with_synthetic_data ──")

    (orders_exact, settlements_exact, items_by_settlement,
     refunds_by_order, fee_schedule,
     bank_txns_by_utr, all_btxn_ids,
     orders_fuzzy, settlements_fuzzy, bank_txns_fuzzy,
     dc) = _build_synthetic_data()

    # ── Step 1: Exact join ─────────────────────────────────
    exact = run_exact_join_pipeline(
        orders=orders_exact,
        settlements=settlements_exact,
        items_by_settlement=items_by_settlement,
        refunds_by_order=refunds_by_order,
        fee_schedule=fee_schedule,
        bank_txns_by_utr=bank_txns_by_utr,
        all_bank_txn_ids=all_btxn_ids,
    )

    # ── Step 2: Fuzzy + cluster ────────────────────────────
    result = run_fuzzy_and_cluster_pipeline(
        exact_result=exact,
        orders=orders_fuzzy,
        settlements=settlements_fuzzy,
        bank_txns=bank_txns_fuzzy,
    )

    print(f"\n  Pipeline summary:")
    print(f"    Fuzzy resolved:   {len(result.fuzzy_resolved)}")
    print(f"    Clusters:         {len(result.clusters)}")
    print(f"    No candidates:    {len(result.no_candidates)}")

    ok = True

    # ── reference_mismatch: 4 hop2-unmatched settlements should fuzzy-resolve
    print("\n  ── reference_mismatch ──")
    rm_orders = _entities_in_scenario(dc, "reference_mismatch", "order")
    rm_setl_ids: set[str] = set()
    for mg in exact.hop1_matched:
        if any(a.entity_id in rm_orders for a in mg.allocations
               if a.entity_type == EntityType.ORDER):
            if mg.settlement_id:
                rm_setl_ids.add(mg.settlement_id)

    rm_fuzzy = [r for r in result.fuzzy_resolved
                if r.settlement_id in rm_setl_ids]
    ok &= _check(
        "reference_mismatch settlements fuzzy-resolved",
        len(rm_fuzzy) > 0,
        f"got {len(rm_fuzzy)} fuzzy matches from {len(rm_setl_ids)} ref-mismatch setls",
    )

    # ── missing_settlement: orders with no settlement → MISSING_SETTLEMENT_RECORD
    print("\n  ── missing_settlement ──")
    ms_orders = _entities_in_scenario(dc, "missing_settlement", "order")
    ms_nocand = [nc for nc in result.no_candidates if nc.entity_id in ms_orders]
    ok &= _check(
        "missing_settlement orders → no_candidates",
        len(ms_nocand) == len(ms_orders),
        f"expected {len(ms_orders)}, got {len(ms_nocand)}",
    )
    if ms_nocand:
        ok &= _check(
            "  category = MISSING_SETTLEMENT_RECORD",
            all(nc.suggested_category == "MISSING_SETTLEMENT_RECORD"
                for nc in ms_nocand),
        )

    # ── unmapped_bank_deposit: orphan bank_txns → UNMAPPED_BANK_DEPOSIT
    print("\n  ── unmapped_bank_deposit ──")
    ubd_btxns = _entities_in_scenario(dc, "unmapped_bank_deposit", "bank_transaction")
    ubd_nocand = [nc for nc in result.no_candidates if nc.entity_id in ubd_btxns]
    ok &= _check(
        "unmapped_bank_deposit bank_txns → no_candidates",
        len(ubd_nocand) == len(ubd_btxns),
        f"expected {len(ubd_btxns)}, got {len(ubd_nocand)}",
    )
    if ubd_nocand:
        ok &= _check(
            "  category = UNMAPPED_BANK_DEPOSIT",
            all(nc.suggested_category == "UNMAPPED_BANK_DEPOSIT"
                for nc in ubd_nocand),
        )

    # ── DISJOINT PARTITIONING: no entity in two clusters
    print("\n  ── Disjoint partitioning (global) ──")
    entity_appearances: dict[str, list[str]] = defaultdict(list)
    for cl in result.clusters:
        entity_appearances[cl.primary_entity_id].append(cl.cluster_id)
        for cm in cl.candidate_matches:
            entity_appearances[cm.entity_id].append(cl.cluster_id)

    duplicates = {eid: cids for eid, cids in entity_appearances.items()
                  if len(cids) > 1}
    ok &= _check(
        "No entity in >1 cluster across entire pipeline",
        len(duplicates) == 0,
        f"duplicates={duplicates}" if duplicates else "all unique",
    )

    # ── CLUSTER SIZE: no cluster exceeds 8
    print("\n  ── Cluster size cap ──")
    if result.clusters:
        max_cl = max(1 + len(cl.candidate_matches) for cl in result.clusters)
        ok &= _check(
            "No cluster exceeds 8 entities",
            max_cl <= 8,
            f"max={max_cl}",
        )

    # ── DEMOTED RECORDS: hop1/hop2 demoted go to clusters
    print("\n  ── Demoted records in clusters ──")
    demoted_setl_ids = (
        {d.settlement_id for d in exact.hop1_demoted}
        | {d.settlement_id for d in exact.hop2_demoted}
    )
    cluster_entity_ids = set()
    for cl in result.clusters:
        cluster_entity_ids.add(cl.primary_entity_id)
        for cm in cl.candidate_matches:
            cluster_entity_ids.add(cm.entity_id)

    demoted_in_clusters = demoted_setl_ids & cluster_entity_ids
    ok &= _check(
        f"Demoted settlements present in clusters",
        len(demoted_in_clusters) > 0,
        f"{len(demoted_in_clusters)}/{len(demoted_setl_ids)} demoted setls in clusters",
    )

    return ok


# ═══════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════


def run_tests():
    print("\n" + "=" * 70)
    print("  FUZZY SCORE & CLUSTER CANDIDATES — TEST SUITE")
    print("=" * 70)

    tests = [
        test_amount_collision_never_fuzzy_resolved,
        test_reference_mismatch_can_be_fuzzy_resolved,
        test_no_candidates_missing_settlement,
        test_no_candidates_unmapped_bank_deposit,
        test_union_find_basics,
        test_disjoint_partitioning_no_entity_in_two_clusters,
        test_cluster_max_size_sub_partition,
        test_full_pipeline_with_synthetic_data,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  💥 {test_fn.__name__} RAISED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    total = passed + failed
    if failed == 0:
        print(f"  🎉 ALL {total} TESTS PASSED")
    else:
        print(f"  ⚠️  {failed}/{total} TESTS FAILED")
    print("=" * 70 + "\n")
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
