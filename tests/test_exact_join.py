"""
Test ExactIdJoin against Step 2's synthetic data (pure Python, no DB).

Converts the in-memory DataCollector objects → DTOs, runs the pipeline,
and verifies per-scenario outcomes.

Usage:
    python -m tests.test_exact_join
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.orchestrator.exact_join import (
    BankTxnDTO,
    ExactJoinResult,
    FeeScheduleDTO,
    OrderDTO,
    RefundDTO,
    SettlementDTO,
    SettlementItemDTO,
    run_exact_join_pipeline,
)


# ═══════════════════════════════════════════════════════════
#  LOAD SYNTHETIC DATA (pure Python, no DB)
# ═══════════════════════════════════════════════════════════

def _build_synthetic_data():
    """Run the generator in-memory and convert ORM objects to DTOs."""
    # Reset seed to match generate.py module-level seed(42)
    random.seed(42)
    import importlib
    import app.synthetic.generate as gen_mod
    importlib.reload(gen_mod)  # re-triggers random.seed(42)

    from app.synthetic.generate import DataCollector, _generate_all
    dc = DataCollector()
    _generate_all(dc)

    # ── Orders ────────────────────────────────────────────
    orders = {
        o.order_id: OrderDTO(o.order_id, o.method, o.gross_amount_paise)
        for o in dc.orders
    }

    # ── Settlements ───────────────────────────────────────
    settlements = {
        s.settlement_id: SettlementDTO(
            s.settlement_id, s.id, s.utr,
            s.gross_amount_paise, s.fee_base_paise,
            s.fee_tax_gst_paise, s.net_amount_paise,
        )
        for s in dc.settlements
    }
    uuid_to_sid = {s.id: s.settlement_id for s in dc.settlements}

    # ── Settlement Items (group by settlement_id string) ──
    items_by_settlement: dict[str, list[SettlementItemDTO]] = defaultdict(list)
    for si in dc.settlement_items:
        sid = uuid_to_sid[si.settlement_id]
        items_by_settlement[sid].append(SettlementItemDTO(
            sid, si.order_id,
            si.gross_paise, si.fee_paise, si.tax_paise, si.net_paise,
        ))

    # ── Refunds ───────────────────────────────────────────
    refunds_by_order: dict[str, list[RefundDTO]] = defaultdict(list)
    for r in dc.refunds:
        refunds_by_order[r.order_id].append(
            RefundDTO(r.refund_id, r.order_id, r.amount_paise)
        )

    # ── Bank Transactions ─────────────────────────────────
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

    # ── Fee Schedule ──────────────────────────────────────
    fee_schedule = {
        fs.method: FeeScheduleDTO(
            fs.method, fs.mdr_basis_points, fs.gst_rate_basis_points,
        )
        for fs in dc.fee_schedules
    }

    return (
        orders, settlements, dict(items_by_settlement),
        dict(refunds_by_order), fee_schedule,
        dict(bank_txns_by_utr), all_btxn_ids, dc,
    )


# ═══════════════════════════════════════════════════════════
#  HELPER: identify which scenario each entity belongs to
# ═══════════════════════════════════════════════════════════

def _build_scenario_map(dc) -> dict[str, str]:
    """Map entity_id → scenario_type from ground_truth rows."""
    m: dict[str, str] = {}
    for row in dc.ground_truth:
        m[row["entity_id"]] = row["scenario_type"]
    # Also tag orders by their summary tally
    for scenario, tables in dc.summary.items():
        pass  # summary doesn't track individual IDs
    return m


def _orders_in_scenario(dc, scenario: str) -> list[str]:
    """Return order_ids that belong to a given scenario."""
    return [
        row["entity_id"]
        for row in dc.ground_truth
        if row["scenario_type"] == scenario and row["entity_type"] == "order"
    ]


# ═══════════════════════════════════════════════════════════
#  TEST VERIFICATION
# ═══════════════════════════════════════════════════════════

def _check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "✅" if condition else "❌"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def run_tests():
    print("\n" + "=" * 70)
    print("  EXACT JOIN TEST — Step 2 Synthetic Data")
    print("=" * 70)

    (orders, settlements, items_by_settlement,
     refunds_by_order, fee_schedule,
     bank_txns_by_utr, all_btxn_ids, dc) = _build_synthetic_data()

    print(f"\n  Source counts: {len(orders)} orders, "
          f"{len(settlements)} settlements, "
          f"{len(all_btxn_ids)} bank_txns, "
          f"{sum(len(v) for v in items_by_settlement.values())} items")

    # ── Run pipeline ──────────────────────────────────────
    result = run_exact_join_pipeline(
        orders=orders,
        settlements=settlements,
        items_by_settlement=items_by_settlement,
        refunds_by_order=refunds_by_order,
        fee_schedule=fee_schedule,
        bank_txns_by_utr=bank_txns_by_utr,
        all_bank_txn_ids=all_btxn_ids,
    )

    # ── Print bucket sizes ────────────────────────────────
    print(f"\n  HOP 1")
    print(f"    Matched settlements:  {len(result.hop1_matched)}")
    print(f"    Demoted settlements:  {len(result.hop1_demoted)}")
    print(f"    Unmatched orders:     {len(result.hop1_unmatched_orders)}")
    print(f"\n  HOP 2")
    print(f"    Matched bank_txns:    {len(result.hop2_matched)}")
    print(f"    Demoted:              {len(result.hop2_demoted)}")
    print(f"    Unmatched settlements:{len(result.hop2_unmatched_settlements)}")
    print(f"    Unmatched bank_txns:  {len(result.hop2_unmatched_bank_txns)}")

    # ── Build lookup sets for verification ────────────────
    h1_matched_setl_ids = {mg.settlement_id for mg in result.hop1_matched}
    h1_matched_order_ids: set[str] = set()
    for mg in result.hop1_matched:
        for a in mg.allocations:
            if a.entity_type.value == "order":
                h1_matched_order_ids.add(a.entity_id)

    h1_demoted_setl_ids = {d.settlement_id for d in result.hop1_demoted}
    h1_demoted_checks = {d.settlement_id: d.failed_check for d in result.hop1_demoted}

    h2_matched_setl_ids = {mg.settlement_id for mg in result.hop2_matched}
    h2_matched_btxn_ids = {mg.bank_txn_id for mg in result.hop2_matched}

    # ══════════════════════════════════════════════════════
    #  PER-SCENARIO VERIFICATION
    # ══════════════════════════════════════════════════════
    passed = 0
    failed = 0

    def check(label, cond, detail=""):
        nonlocal passed, failed
        if _check(label, cond, detail):
            passed += 1
        else:
            failed += 1

    # ── clean_match ───────────────────────────────────────
    print("\n  ── clean_match ──")
    cm_oids = _orders_in_scenario(dc, "clean_match")
    cm_in_h1 = [oid for oid in cm_oids if oid in h1_matched_order_ids]
    check("21 clean_match orders in hop1_matched",
          len(cm_in_h1) == 21, f"got {len(cm_in_h1)}")

    # Count clean_match settlements (setl_synth_001..017)
    cm_setls = [sid for sid in h1_matched_setl_ids
                if any(mg.settlement_id == sid and
                       any(a.entity_id in cm_oids
                           for a in mg.allocations
                           if a.entity_type.value == "order")
                       for mg in result.hop1_matched)]
    # Simpler: check all 17 settlements matched
    cm_setl_ids_expected = {f"setl_synth_{i:03d}" for i in range(1, 18)}
    cm_setl_in_h1 = cm_setl_ids_expected & h1_matched_setl_ids
    check("17 clean_match settlements in hop1_matched",
          len(cm_setl_in_h1) == 17, f"got {len(cm_setl_in_h1)}")
    cm_setl_in_h2 = cm_setl_ids_expected & h2_matched_setl_ids
    check("17 clean_match settlements in hop2_matched",
          len(cm_setl_in_h2) == 17, f"got {len(cm_setl_in_h2)}")

    # ── gst_rounding_edge_case ────────────────────────────
    print("\n  ── gst_rounding_edge_case ──")
    gst_oids = _orders_in_scenario(dc, "gst_rounding_edge_case")
    gst_in_h1 = [oid for oid in gst_oids if oid in h1_matched_order_ids]
    check("5 GST rounding orders in hop1_matched",
          len(gst_in_h1) == 5, f"got {len(gst_in_h1)}")
    # Check residuals are 0 (items generated with same _fees function)
    gst_residuals = [
        mg.residual_paise for mg in result.hop1_matched
        if any(a.entity_id in gst_oids for a in mg.allocations
               if a.entity_type.value == "order")
    ]
    check("GST rounding residuals all 0",
          all(r == 0 for r in gst_residuals),
          f"residuals={gst_residuals}")

    # ── timing_lag_pending ────────────────────────────────
    print("\n  ── timing_lag_pending ──")
    tlp_oids = _orders_in_scenario(dc, "timing_lag_pending")
    tlp_unmatched = [oid for oid in tlp_oids
                     if oid in result.hop1_unmatched_orders]
    check("4 timing_lag_pending orders unmatched",
          len(tlp_unmatched) == 4, f"got {len(tlp_unmatched)}")

    # ── timing_lag_anomalous ──────────────────────────────
    print("\n  ── timing_lag_anomalous ──")
    tla_oids = _orders_in_scenario(dc, "timing_lag_anomalous")
    tla_unmatched = [oid for oid in tla_oids
                     if oid in result.hop1_unmatched_orders]
    check("3 timing_lag_anomalous orders unmatched",
          len(tla_unmatched) == 3, f"got {len(tla_unmatched)}")

    # ── partial_refund ────────────────────────────────────
    print("\n  ── partial_refund ──")
    pr_oids = _orders_in_scenario(dc, "partial_refund")
    # Ground truth: first 2 → match_groups (exact), last 2 → exception
    pr_clean = [oid for oid in pr_oids[:2] if oid in h1_matched_order_ids]
    check("2 clean partial_refund orders in hop1_matched",
          len(pr_clean) == 2, f"got {len(pr_clean)}")

    # The broken partial refunds: their settlements should be demoted
    pr_broken_oids = pr_oids[2:]
    # Find settlements containing these orders in demoted list
    pr_demoted = [
        d for d in result.hop1_demoted
        if any(oid in d.order_ids for oid in pr_broken_oids)
    ]
    check("2 broken partial_refund settlements demoted",
          len(pr_demoted) == 2, f"got {len(pr_demoted)}")
    if pr_demoted:
        check("  demoted via conservation_failed",
              all(d.failed_check == "conservation_failed" for d in pr_demoted),
              f"checks={[d.failed_check for d in pr_demoted]}")

    # ── chargeback_unmatched ──────────────────────────────
    print("\n  ── chargeback_unmatched ──")
    # Orders should pass hop1 (disputes not checked in conservation)
    cb_order_ids = [f"order_synth_{i:03d}" for i in range(38, 41)]
    cb_in_h1 = [oid for oid in cb_order_ids if oid in h1_matched_order_ids]
    check("3 chargeback orders in hop1_matched (disputes NOT checked)",
          len(cb_in_h1) == 3, f"got {len(cb_in_h1)}")

    # ── reference_mismatch ────────────────────────────────
    print("\n  ── reference_mismatch ──")
    rm_oids = _orders_in_scenario(dc, "reference_mismatch")
    rm_in_h1 = [oid for oid in rm_oids if oid in h1_matched_order_ids]
    check("4 reference_mismatch orders pass hop1",
          len(rm_in_h1) == 4, f"got {len(rm_in_h1)}")
    # But their settlements should be unmatched in hop2 (garbled UTR)
    rm_setl_ids = [mg.settlement_id for mg in result.hop1_matched
                   if any(a.entity_id in rm_oids for a in mg.allocations
                          if a.entity_type.value == "order")]
    rm_in_h2_unmatched = [sid for sid in rm_setl_ids
                          if sid in result.hop2_unmatched_settlements]
    check("4 reference_mismatch settlements unmatched in hop2",
          len(rm_in_h2_unmatched) == 4, f"got {len(rm_in_h2_unmatched)}")

    # ── duplicate_bank_credit ─────────────────────────────
    print("\n  ── duplicate_bank_credit ──")
    dbc_oids = _orders_in_scenario(dc, "duplicate_bank_credit")
    dbc_in_h1 = [oid for oid in dbc_oids if oid in h1_matched_order_ids]
    check("2 dup_bank_credit orders in hop1_matched",
          len(dbc_in_h1) == 2, f"got {len(dbc_in_h1)}")
    # The duplicate bank_txns should be in hop2_unmatched
    dbc_gt_btxns = [
        row["entity_id"] for row in dc.ground_truth
        if row["scenario_type"] == "duplicate_bank_credit"
        and row["entity_type"] == "bank_transaction"
    ]
    dbc_btxn_unmatched = [bid for bid in dbc_gt_btxns
                          if bid in result.hop2_unmatched_bank_txns]
    check("2 duplicate bank_txns in hop2_unmatched",
          len(dbc_btxn_unmatched) == 2, f"got {len(dbc_btxn_unmatched)}")

    # ── missing_settlement ────────────────────────────────
    print("\n  ── missing_settlement ──")
    ms_oids = _orders_in_scenario(dc, "missing_settlement")
    ms_unmatched = [oid for oid in ms_oids
                    if oid in result.hop1_unmatched_orders]
    check("2 missing_settlement orders unmatched",
          len(ms_unmatched) == 2, f"got {len(ms_unmatched)}")

    # ── unmapped_bank_deposit ─────────────────────────────
    print("\n  ── unmapped_bank_deposit ──")
    ubd_btxns = [
        row["entity_id"] for row in dc.ground_truth
        if row["scenario_type"] == "unmapped_bank_deposit"
    ]
    ubd_unmatched = [bid for bid in ubd_btxns
                     if bid in result.hop2_unmatched_bank_txns]
    check("2 unmapped_bank_deposit in hop2_unmatched",
          len(ubd_unmatched) == 2, f"got {len(ubd_unmatched)}")

    # ── broken_settlement_item_missing_order ──────────────
    print("\n  ── broken_settlement_item_missing_order ──")
    bsimo_setls = [
        row["entity_id"] for row in dc.ground_truth
        if row["scenario_type"] == "broken_settlement_item_missing_order"
    ]
    bsimo_demoted = [d for d in result.hop1_demoted
                     if d.settlement_id in bsimo_setls]
    check("2 missing-order settlements demoted in hop1",
          len(bsimo_demoted) == 2,
          f"got {len(bsimo_demoted)}")
    if bsimo_demoted:
        check("  demoted via missing_order check",
              all(d.failed_check == "missing_order" for d in bsimo_demoted),
              f"checks={[d.failed_check for d in bsimo_demoted]}")
        check("  original_tier_hint=exact_id_missing_order",
              all(d.original_tier_hint == "exact_id_missing_order"
                  for d in bsimo_demoted))

    # ── broken_settlement_item_sum_mismatch ───────────────
    print("\n  ── broken_settlement_item_sum_mismatch ──")
    bsism_setls = [
        row["entity_id"] for row in dc.ground_truth
        if row["scenario_type"] == "broken_settlement_item_sum_mismatch"
    ]
    bsism_demoted = [d for d in result.hop1_demoted
                     if d.settlement_id in bsism_setls]
    check("2 sum-mismatch settlements demoted in hop1",
          len(bsism_demoted) == 2,
          f"got {len(bsism_demoted)}")
    if bsism_demoted:
        check("  demoted via sum_mismatch check",
              all(d.failed_check == "sum_mismatch" for d in bsism_demoted),
              f"checks={[d.failed_check for d in bsism_demoted]}")
        check("  original_tier_hint=exact_id_conservation_failed",
              all(d.original_tier_hint == "exact_id_conservation_failed"
                  for d in bsism_demoted))

    # ── unreconciled_bank_fee ─────────────────────────────
    print("\n  ── unreconciled_bank_fee ──")
    ubf_oids = _orders_in_scenario(dc, "unreconciled_bank_fee")
    if ubf_oids:
        # No ground truth for the ORDER (only for bank_txn), but the
        # order's settlement should still match in hop1.
        # The bank_txn has bank_charges but conservation still balances:
        # settlement.net == (amount + charges)
        ubf_btxn_id = [
            row["entity_id"] for row in dc.ground_truth
            if row["scenario_type"] == "unreconciled_bank_fee"
        ][0]
        check("bank_fee bank_txn in hop2_matched (conservation passes)",
              ubf_btxn_id in h2_matched_btxn_ids,
              f"btxn={ubf_btxn_id}")

    # ── amount_collision_cluster ──────────────────────────
    print("\n  ── amount_collision_cluster ──")
    acc_oids = _orders_in_scenario(dc, "amount_collision_cluster")
    acc_in_h1 = [oid for oid in acc_oids if oid in h1_matched_order_ids]
    check("4 collision orders in hop1_matched (items provide definitive link)",
          len(acc_in_h1) == 4, f"got {len(acc_in_h1)}")

    # ── suspicious_round_number ───────────────────────────
    print("\n  ── suspicious_round_number ──")
    srn_btxn = [
        row["entity_id"] for row in dc.ground_truth
        if row["scenario_type"] == "suspicious_round_number"
    ]
    srn_unmatched = [b for b in srn_btxn
                     if b in result.hop2_unmatched_bank_txns]
    check("1 suspicious_round_number bank_txn in hop2_unmatched",
          len(srn_unmatched) == 1, f"got {len(srn_unmatched)}")

    # ── retry_correctable ─────────────────────────────────
    print("\n  ── retry_correctable ──")
    rc_oids = _orders_in_scenario(dc, "retry_correctable")
    rc_demoted = [
        d for d in result.hop1_demoted
        if any(oid in d.order_ids for oid in rc_oids)
    ]
    check("2 retry_correctable settlements demoted (fee breach)",
          len(rc_demoted) == 2, f"got {len(rc_demoted)}")
    if rc_demoted:
        check("  demoted via fee_mismatch",
              all(d.failed_check == "fee_mismatch" for d in rc_demoted),
              f"checks={[d.failed_check for d in rc_demoted]}")

    # ══════════════════════════════════════════════════════
    #  AGGREGATE CHECKS
    # ══════════════════════════════════════════════════════
    print("\n  ── Aggregate checks ──")

    # All orders accounted for
    total_claimed = len(h1_matched_order_ids)
    total_in_demoted = len({
        oid for d in result.hop1_demoted for oid in d.order_ids
        if oid in orders  # exclude ghost orders
    })
    total_unmatched = len(result.hop1_unmatched_orders)
    check(f"All {len(orders)} orders accounted for "
          f"(matched={total_claimed} + demoted={total_in_demoted} "
          f"+ unmatched={total_unmatched})",
          total_claimed + total_in_demoted + total_unmatched == len(orders))

    # No crash — demoted records have useful traces
    check("All demoted records have non-empty reasoning_trace",
          all(d.reasoning_trace for d in result.hop1_demoted))

    # Hop1 matched groups are all verified=True
    check("All hop1_matched groups verified=True",
          all(mg.verified for mg in result.hop1_matched))

    # Hop2 matched groups are all verified=True
    check("All hop2_matched groups verified=True",
          all(mg.verified for mg in result.hop2_matched))

    # ══════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    total = passed + failed
    if failed == 0:
        print(f"  🎉 ALL {total} CHECKS PASSED")
    else:
        print(f"  ⚠️  {failed}/{total} CHECKS FAILED")
    print("=" * 70)

    # ── Print demoted details for debugging ───────────────
    if result.hop1_demoted:
        print(f"\n  Demoted settlements detail:")
        for d in result.hop1_demoted:
            print(f"    {d.settlement_id}: {d.failed_check} "
                  f"| hint={d.original_tier_hint} | Δ={d.delta_paise}p")
            print(f"      orders={d.order_ids}")
            print(f"      trace={d.reasoning_trace[:120]}...")

    print()
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
