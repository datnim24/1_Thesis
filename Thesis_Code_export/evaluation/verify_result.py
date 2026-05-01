"""Standalone constraint verifier for universal and legacy result JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.result_schema import (
    convert_legacy_result,
    load_reference_data,
    load_result,
    normalize_schedule_entry,
    reconstruct_gc_trajectory,
    reconstruct_rc_trajectory,
    reconstruct_restock_timeline,
    validate_result,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sidecar_data(result_path: Path):
    stem = result_path.stem
    prefix = stem[:-7] if stem.endswith("_result") else stem
    schedule_path = result_path.with_name(f"{prefix}_schedule.json")
    cancelled_path = result_path.with_name(f"{prefix}_cancelled.json")
    ups_path = result_path.with_name(f"{prefix}_ups.json")
    restocks_path = result_path.with_name(f"{prefix}_restocks.json")
    schedule = _load_json(schedule_path) if schedule_path.exists() else None
    cancelled = _load_json(cancelled_path) if cancelled_path.exists() else None
    ups = _load_json(ups_path) if ups_path.exists() else None
    restocks = _load_json(restocks_path) if restocks_path.exists() else None
    return schedule, cancelled, ups, restocks


def _load_any_result(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    result_path = Path(path)
    raw = _load_json(result_path)
    if {"metadata", "experiment", "kpi", "schedule", "parameters"}.issubset(raw):
        result = load_result(result_path)
        return result, validate_result(result)

    schedule, cancelled, ups, restocks = _sidecar_data(result_path)
    result = convert_legacy_result(
        raw,
        source="auto",
        schedule=schedule,
        cancelled=cancelled,
        ups=ups,
        restocks=restocks,
    )
    return result, validate_result(result)


def _pass(label: str, detail: str) -> tuple[bool, str]:
    return True, f"  PASS [{label}]: {detail}"


def _fail(label: str, detail: str) -> tuple[bool, str]:
    return False, f"  FAIL [{label}]: {detail}"


def _canonical_pair_set(params: dict[str, Any], gc_traj: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for raw_pair in params.get("feasible_gc_pairs", []):
        if isinstance(raw_pair, tuple):
            pairs.add((str(raw_pair[0]), str(raw_pair[1])))
        else:
            text = str(raw_pair)
            if "_" in text:
                line_id, sku = text.split("_", 1)
                pairs.add((line_id, sku))
    for pair_key in gc_traj:
        if pair_key == "events" or "_" not in pair_key:
            continue
        line_id, sku = pair_key.split("_", 1)
        pairs.add((line_id, sku))
    return pairs


def _infer_cancel_time(entry: dict[str, Any], ups_events: list[dict[str, Any]]) -> int | None:
    explicit = entry.get("cancel_time")
    if explicit not in (None, ""):
        return int(explicit)
    matches = [
        int(event["t"])
        for event in ups_events
        if event.get("roaster_id") == entry["roaster"]
        and entry["start"] <= int(event.get("t", -1)) < entry["end"]
    ]
    return min(matches) if matches else None


def _consume_interval(entry: dict[str, Any], dc: int, ups_events: list[dict[str, Any]]) -> tuple[int, int]:
    start = int(entry.get("pipeline_start", entry["start"]))
    end = int(entry.get("pipeline_end", start + dc))
    if entry.get("status") == "cancelled":
        cancel_time = _infer_cancel_time(entry, ups_events)
        if cancel_time is not None:
            end = min(end, cancel_time)
    return start, max(start, end)


def verify(result: dict[str, Any]) -> tuple[bool, list[str]]:
    refs = load_reference_data(result["metadata"].get("input_dir"))
    params = result["parameters"]
    schedule = [normalize_schedule_entry(entry) for entry in result["schedule"]]
    cancelled = [normalize_schedule_entry(entry) for entry in result.get("cancelled_batches", [])]
    completed = [entry for entry in schedule if entry["status"] == "completed"]
    ups_events = result.get("ups_events", [])
    deterministic = not ups_events
    sigma = int(params["sigma"])
    dc = int(params["DC"])
    shift_length = int(params["SL"])
    max_start = int(params["MS"])
    max_rc = int(params["max_rc"])
    kpi = result["kpi"]

    lines: list[str] = []
    all_passed = True

    ndg_completed = sum(1 for entry in completed if entry["sku"] == "NDG")
    busta_completed = sum(1 for entry in completed if entry["sku"] == "BUSTA")
    if deterministic:
        ok = ndg_completed == refs["job_batches"].get("J1", 3) and busta_completed == refs["job_batches"].get("J2", 1)
        passed, line = (_pass if ok else _fail)(
            "C1",
            f"NDG: {ndg_completed}/{refs['job_batches'].get('J1', 3)}, "
            f"BUSTA: {busta_completed}/{refs['job_batches'].get('J2', 1)}",
        )
    else:
        passed, line = _pass("C1", f"Reactive mode count only: NDG={ndg_completed}, BUSTA={busta_completed}")
    all_passed &= passed
    lines.append(line)

    eligibility_violations: list[str] = []
    for entry in completed:
        allowed = refs["roaster_eligible_skus"].get(entry["roaster"], [])
        if entry["sku"] not in allowed:
            eligibility_violations.append(f"{entry['batch_id']} {entry['sku']} on {entry['roaster']}")
    passed, line = (
        _pass("C3", f"All {len(completed)} batches on eligible roasters")
        if not eligibility_violations
        else _fail("C3", "; ".join(eligibility_violations[:5]))
    )
    all_passed &= passed
    lines.append(line)

    overlap_violations: list[str] = []
    setup_violations: list[str] = []
    roaster_pairs_checked = 0
    for roaster in refs["roasters"]:
        batches = sorted((entry for entry in completed if entry["roaster"] == roaster), key=lambda item: (item["start"], item["end"]))
        if batches:
            first = batches[0]
            initial_sku = refs["roaster_initial_sku"].get(roaster, "PSC")
            if first["sku"] != initial_sku and first["start"] < sigma:
                setup_violations.append(f"{roaster} first batch {first['batch_id']} starts {first['start']} < sigma={sigma}")
        for left, right in zip(batches, batches[1:]):
            roaster_pairs_checked += 1
            if left["end"] > right["start"]:
                overlap_violations.append(f"{roaster}: {left['batch_id']} [{left['start']},{left['end']}) overlaps {right['batch_id']} [{right['start']},{right['end']})")
            if left["sku"] != right["sku"]:
                gap = right["start"] - left["end"]
                if gap < sigma:
                    setup_violations.append(f"{roaster}: {left['batch_id']}->{right['batch_id']} gap={gap} < sigma={sigma}")
    passed, line = (
        _pass("C4", f"No roaster overlaps ({roaster_pairs_checked} pairs checked)")
        if not overlap_violations
        else _fail("C4", "; ".join(overlap_violations[:5]))
    )
    all_passed &= passed
    lines.append(line)

    passed, line = (
        _pass("C5", "All SKU transitions have sufficient setup gap")
        if not setup_violations
        else _fail("C5", "; ".join(setup_violations[:5]))
    )
    all_passed &= passed
    lines.append(line)

    downtime_violations = []
    for entry in completed:
        roaster_slots = refs["downtime_slots"].get(entry["roaster"], set())
        if set(range(entry["start"], entry["end"])).intersection(roaster_slots):
            downtime_violations.append(
                f"{entry['roaster']} {entry['batch_id']} [{entry['start']},{entry['end']})"
            )
    passed, line = (
        _pass("C7", "No batches overlap planned downtime")
        if not downtime_violations
        else _fail("C7", "; ".join(downtime_violations[:5]))
    )
    all_passed &= passed
    lines.append(line)

    pipeline_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in completed:
        pipeline_groups[entry["pipeline"]].append(entry)
    pipeline_violations: list[str] = []
    pipeline_pairs_checked = 0
    for pipeline, batches in pipeline_groups.items():
        batches.sort(key=lambda item: (item["pipeline_start"], item["pipeline_end"]))
        for left, right in zip(batches, batches[1:]):
            pipeline_pairs_checked += 1
            if left["pipeline_end"] > right["pipeline_start"]:
                pipeline_violations.append(
                    f"{pipeline}: {left['batch_id']} [{left['pipeline_start']},{left['pipeline_end']}) overlaps "
                    f"{right['batch_id']} [{right['pipeline_start']},{right['pipeline_end']})"
                )
    passed, line = (
        _pass("C8", f"No pipeline overlaps ({pipeline_pairs_checked} pairs checked)")
        if not pipeline_violations
        else _fail("C8", "; ".join(pipeline_violations[:5]))
    )
    all_passed &= passed
    lines.append(line)

    shift_violations = [
        f"{entry['batch_id']} start={entry['start']} end={entry['end']}"
        for entry in completed
        if entry["start"] > max_start or entry["end"] > shift_length
    ]
    passed, line = (
        _pass("C9", f"All {len(completed)} batches within shift")
        if not shift_violations
        else _fail("C9", "; ".join(shift_violations[:5]))
    )
    all_passed &= passed
    lines.append(line)

    traj = reconstruct_rc_trajectory(schedule, params)
    for line_id in ("L1", "L2"):
        stocks = traj[line_id]
        consumption_times = set(traj[f"{line_id}_events"]["consumptions"])
        stockout_count = sum(1 for slot in consumption_times if stocks[slot] < 0)
        min_stock = min(stocks) if stocks else 0
        if deterministic:
            passed, line = (
                _pass("C10", f"{line_id}: min_stock={min_stock}, stockout_events={stockout_count}")
                if stockout_count == 0
                else _fail("C10", f"{line_id}: stockout_events={stockout_count}, min_stock={min_stock}")
            )
        else:
            expected = int(kpi.get("stockout_events", {}).get(line_id, 0))
            passed, line = (
                _pass("C10", f"{line_id}: stockout_events={stockout_count}, KPI={expected}, min_stock={min_stock}")
                if stockout_count == expected
                else _fail("C10", f"{line_id}: reconstructed={stockout_count}, KPI={expected}, min_stock={min_stock}")
            )
        all_passed &= passed
        lines.append(line)

        max_stock_seen = max(stocks) if stocks else 0
        passed, line = (
            _pass("C11", f"{line_id}: max_stock={max_stock_seen} <= {max_rc}")
            if max_stock_seen <= max_rc
            else _fail("C11", f"{line_id}: max_stock={max_stock_seen} > {max_rc}")
        )
        all_passed &= passed
        lines.append(line)

    for job_id, due in refs["job_due"].items():
        job_entries = [entry for entry in completed if entry["job_id"] == job_id]
        latest_end = max((entry["end"] for entry in job_entries), default=0)
        tardiness = max(0, latest_end - due)
        reported = float(kpi.get("tardiness_min", {}).get(job_id, 0.0))
        passed, line = (
            _pass("C12", f"{job_id}: tardiness={tardiness} min")
            if abs(tardiness - reported) <= 0.5
            else _fail("C12", f"{job_id}: reconstructed={tardiness} vs KPI={reported}")
        )
        all_passed &= passed
        lines.append(line)

    net_profit = float(kpi["net_profit"])
    total_revenue = float(kpi["total_revenue"])
    total_costs = float(kpi["total_costs"])
    passed, line = (
        _pass("ACCT", f"net_profit=${net_profit:,.0f} = ${total_revenue:,.0f} - ${total_costs:,.0f}")
        if abs(net_profit - (total_revenue - total_costs)) <= 0.1
        else _fail("ACCT", f"net_profit=${net_profit:,.2f} but revenue-costs=${total_revenue - total_costs:,.2f}")
    )
    all_passed &= passed
    lines.append(line)

    expected_revenue = (
        int(kpi["psc_count"]) * float(params["sku_revenue"]["PSC"])
        + int(kpi["ndg_count"]) * float(params["sku_revenue"]["NDG"])
        + int(kpi["busta_count"]) * float(params["sku_revenue"]["BUSTA"])
    )
    passed, line = (
        _pass("REV", f"Revenue consistent: ${total_revenue:,.0f}")
        if abs(total_revenue - expected_revenue) <= 0.1
        else _fail("REV", f"reported=${total_revenue:,.2f} vs expected=${expected_revenue:,.2f}")
    )
    all_passed &= passed
    lines.append(line)

    # ── GC-silo & restock checks (skip gracefully for legacy results) ──
    restocks_raw = result.get("restocks", [])
    has_gc = bool(restocks_raw) or bool(params.get("gc_capacity")) or bool(params.get("gc_init"))

    if not has_gc:
        lines.append("  PASS [GC-*]: legacy mode, no GC data — GC/restock checks skipped")
    else:
        gc_traj = reconstruct_gc_trajectory(schedule, restocks_raw, params, cancelled)
        restock_tl = reconstruct_restock_timeline(restocks_raw)
        gc_cap_raw = params.get("gc_capacity", {"L1_PSC": 40, "L1_NDG": 10, "L1_BUSTA": 10, "L2_PSC": 40})
        gc_cap = {}
        for k, v in gc_cap_raw.items():
            if "_" in k:
                parts = k.split("_", 1)
                gc_cap[(parts[0], parts[1])] = int(v)
            elif isinstance(k, tuple):
                gc_cap[k] = int(v)
        trajectories = {k: v for k, v in gc_traj.items() if k != "events"}

        roast_time_by_sku = params.get("roast_time_by_sku", {"PSC": 15, "NDG": 17, "BUSTA": 18})
        roaster_line_map = refs.get("roaster_pipeline", {"R1": "L1", "R2": "L1", "R3": "L2", "R4": "L2", "R5": "L2"})
        feasible_pairs = _canonical_pair_set(params, gc_traj)
        started_batches = completed + cancelled

        # GC-CAP: GC inventory stays within [0, capacity]
        cap_violations: list[str] = []
        for pair_key, slots in trajectories.items():
            parts = pair_key.split("_", 1)
            pair_tuple = (parts[0], parts[1])
            cap = gc_cap.get(pair_tuple, 40)
            for slot_idx, stock in enumerate(slots):
                if stock < 0:
                    cap_violations.append(f"{pair_key} slot={slot_idx} stock={stock}<0")
                    break
                if stock > cap:
                    cap_violations.append(f"{pair_key} slot={slot_idx} stock={stock}>{cap}")
                    break
        passed, line = (
            _pass("GC-CAP", f"GC stays within [0, cap] for all {len(trajectories)} pairs")
            if not cap_violations
            else _fail("GC-CAP", "; ".join(cap_violations[:5]))
        )
        all_passed &= passed
        lines.append(line)

        # GC-DEDUCT: Every started batch consumes 1 GC from the correct silo, even if later cancelled
        deduct_violations: list[str] = []
        for entry in started_batches:
            roaster = entry["roaster"]
            sku = entry["sku"]
            home_line = roaster_line_map.get(roaster)
            if home_line is None:
                continue
            pair = (home_line, sku)
            if pair not in feasible_pairs:
                deduct_violations.append(
                    f"{entry['batch_id']} {sku} on {roaster} needs GC pair {home_line}_{sku} which is not feasible"
                )
        passed, line = (
            _pass("GC-DEDUCT", f"All {len(started_batches)} started batches draw from feasible GC pairs")
            if not deduct_violations
            else _fail("GC-DEDUCT", "; ".join(deduct_violations[:5]))
        )
        all_passed &= passed
        lines.append(line)

        final_gc_reported = kpi.get("gc_final", {})
        if final_gc_reported:
            final_reconstructed = {pair_key: slots[-1] for pair_key, slots in trajectories.items() if slots}
            final_gc_violations = []
            for pair_key, stock in sorted(final_reconstructed.items()):
                reported = int(final_gc_reported.get(pair_key, stock))
                if stock != reported:
                    final_gc_violations.append(
                        f"{pair_key}: reconstructed={stock} vs reported={reported}"
                    )
            passed, line = (
                _pass("GC-FINAL", f"Reconstructed GC matches exported final stock for {len(final_reconstructed)} pairs")
                if not final_gc_violations
                else _fail("GC-FINAL", "; ".join(final_gc_violations[:5]))
            )
            all_passed &= passed
            lines.append(line)

        # GC-RESTOCK: Each restock adds correct qty to correct GC silo
        restock_violations: list[str] = []
        for rst in restock_tl:
            pair_key = f"{rst['line_id']}_{rst['sku']}"
            if pair_key not in trajectories:
                restock_violations.append(f"restock at t={rst['start']} targets unknown pair {pair_key}")
            elif rst["qty"] <= 0:
                restock_violations.append(f"restock at t={rst['start']} has non-positive qty={rst['qty']}")
        passed, line = (
            _pass("GC-RESTOCK", f"All {len(restock_tl)} restocks target valid pairs with positive qty")
            if not restock_violations
            else _fail("GC-RESTOCK", "; ".join(restock_violations[:5]))
        )
        all_passed &= passed
        lines.append(line)

        # GC-FEASIBLE: Restocks only for feasible (line, sku) pairs
        feasible_violations: list[str] = []
        for rst in restock_tl:
            pair = (rst["line_id"], rst["sku"])
            if pair not in feasible_pairs:
                feasible_violations.append(f"restock t={rst['start']} pair=({rst['line_id']},{rst['sku']}) not feasible")
        passed, line = (
            _pass("GC-FEASIBLE", f"All {len(restock_tl)} restocks target feasible pairs")
            if not feasible_violations
            else _fail("GC-FEASIBLE", "; ".join(feasible_violations[:5]))
        )
        all_passed &= passed
        lines.append(line)

        # GC-OVERFLOW: No restock causes GC to exceed capacity
        overflow_violations: list[str] = []
        for pair_key, slots in trajectories.items():
            parts = pair_key.split("_", 1)
            pair_tuple = (parts[0], parts[1])
            cap = gc_cap.get(pair_tuple, 40)
            for slot_idx, stock in enumerate(slots):
                if stock > cap:
                    overflow_violations.append(f"{pair_key} slot={slot_idx} stock={stock}>{cap}")
                    break
        passed, line = (
            _pass("GC-OVERFLOW", f"No restock causes overflow for {len(trajectories)} pairs")
            if not overflow_violations
            else _fail("GC-OVERFLOW", "; ".join(overflow_violations[:5]))
        )
        all_passed &= passed
        lines.append(line)

        # RST-MUTEX: Only one restock active globally at a time
        mutex_violations: list[str] = []
        for i, left in enumerate(restock_tl):
            for right in restock_tl[i + 1:]:
                if left["end"] > right["start"]:
                    mutex_violations.append(
                        f"[{left['start']},{left['end']}) overlaps [{right['start']},{right['end']})"
                    )
        passed, line = (
            _pass("RST-MUTEX", f"No overlapping restocks ({len(restock_tl)} restocks)")
            if not mutex_violations
            else _fail("RST-MUTEX", "; ".join(mutex_violations[:5]))
        )
        all_passed &= passed
        lines.append(line)

        # RST-PIPE: Restock intervals don't overlap with batch consume intervals on same pipeline
        pipe_violations: list[str] = []
        for rst in restock_tl:
            rst_line = rst["line_id"]
            rst_start, rst_end = rst["start"], rst["end"]
            for entry in started_batches:
                if entry["pipeline"] != rst_line:
                    continue
                ps, pe = _consume_interval(entry, dc, ups_events)
                if ps < rst_end and rst_start < pe:
                    pipe_violations.append(
                        f"restock [{rst_start},{rst_end}) on {rst_line} overlaps "
                        f"batch {entry['batch_id']} pipeline [{ps},{pe})"
                    )
        passed, line = (
            _pass("RST-PIPE", f"No restock/pipeline overlaps on {len(restock_tl)} restocks")
            if not pipe_violations
            else _fail("RST-PIPE", "; ".join(pipe_violations[:5]))
        )
        all_passed &= passed
        lines.append(line)

        # SKU-DUR: SKU-specific roast durations are respected
        dur_violations: list[str] = []
        for entry in completed:
            expected_dur = roast_time_by_sku.get(entry["sku"])
            if expected_dur is None:
                continue
            actual_dur = entry["end"] - entry["start"]
            if actual_dur != expected_dur:
                dur_violations.append(
                    f"{entry['batch_id']} {entry['sku']}: duration={actual_dur} expected={expected_dur}"
                )
        passed, line = (
            _pass("SKU-DUR", f"All {len(completed)} batches match SKU-specific durations")
            if not dur_violations
            else _fail("SKU-DUR", "; ".join(dur_violations[:5]))
        )
        all_passed &= passed
        lines.append(line)

    per_roaster = Counter(entry["roaster"] for entry in completed)
    psc_per_output = Counter(entry["output_line"] for entry in completed if entry["sku"] == "PSC")
    lines.append("------------------------------------------------------------")
    lines.append(
        "  SUMMARY: total_batches="
        f"{len(completed)}, per_roaster="
        + ", ".join(f"{roaster}={per_roaster.get(roaster, 0)}" for roaster in refs["roasters"])
    )
    lines.append(
        "  SUMMARY: PSC output="
        + ", ".join(f"{line_id}={psc_per_output.get(line_id, 0)}" for line_id in refs["lines"])
    )
    lines.append(
        "  RESULT: " + ("ALL CONSTRAINTS PASSED [OK]" if all_passed else "CONSTRAINT VIOLATIONS FOUND [FAIL]")
    )
    return all_passed, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a universal or legacy thesis result JSON.")
    parser.add_argument("result_json", help="Path to the result JSON file.")
    parser.add_argument("--verbose", action="store_true", help="Print the full PASS/FAIL report.")
    args = parser.parse_args(argv)

    result, validation_errors = _load_any_result(args.result_json)
    ok, lines = verify(result)

    print("=" * 60)
    print("  CONSTRAINT VERIFICATION")
    print("=" * 60)
    if validation_errors:
        for error in validation_errors:
            print(f"  WARN [SCHEMA]: {error}")
        print("-" * 60)
    if args.verbose or not ok:
        for line in lines:
            print(line)
    else:
        print(lines[-1])
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
