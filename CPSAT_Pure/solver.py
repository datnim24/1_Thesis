"""Solve wrapper and result extractor for CP_SAT_v2."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ortools.sat.python import cp_model


logger = logging.getLogger("cpsat_solver_v2")

_verbose = False
_heartbeat_interval_sec = 300.0


def set_verbose(flag: bool) -> None:
    global _verbose
    _verbose = bool(flag)


class ProgressCallback(cp_model.CpSolverSolutionCallback):
    """Collect incumbent history for reporting."""

    def __init__(self, logger_ref: logging.Logger):
        super().__init__()
        self._logger = logger_ref
        self._t0 = time.time()
        self._count = 0
        self._history: list[dict[str, Any]] = []
        self._best_obj: float | None = None
        self._best_bound: float | None = None
        self._gap_pct: float | None = None
        self._last_elapsed: float | None = None

    def on_solution_callback(self) -> None:
        self._count += 1
        elapsed = time.time() - self._t0
        obj = float(self.ObjectiveValue())
        bound = float(self.BestObjectiveBound())
        gap_pct = None
        if abs(bound) > 1e-6:
            gap_pct = abs(bound - obj) / abs(bound) * 100.0

        self._best_obj = obj
        self._best_bound = bound
        self._gap_pct = gap_pct
        self._last_elapsed = elapsed

        self._logger.info(
            f"[CP-SAT] Incumbent #{self._count:3d} | "
            f"obj={obj:>12,.0f} | "
            f"bound={bound:>12,.0f} | "
            f"gap={f'{gap_pct:.3f}%' if gap_pct is not None else 'N/A':>8} | "
            f"t={elapsed:>7.2f}s"
        )
        self._history.append(
            {
                "incumbent": int(self._count),
                "obj": obj,
                "bound": bound,
                "gap_pct": round(gap_pct, 4) if gap_pct is not None else None,
                "elapsed_s": round(elapsed, 3),
            }
        )

    @property
    def incumbent_count(self) -> int:
        return self._count

    @property
    def solution_history(self) -> list[dict[str, Any]]:
        return self._history

    def snapshot(self) -> dict[str, Any]:
        return {
            "incumbent_count": self._count,
            "best_obj": self._best_obj,
            "best_bound": self._best_bound,
            "gap_pct": self._gap_pct,
            "elapsed": self._last_elapsed,
        }


def _solver_heartbeat(
    stop_event: threading.Event,
    start_time: float,
    time_limit: float | int | None,
    callback: ProgressCallback,
) -> None:
    while not stop_event.wait(_heartbeat_interval_sec):
        elapsed = time.time() - start_time
        remaining = max(0.0, float(time_limit) - elapsed) if time_limit else None
        snap = callback.snapshot()

        parts = [f"elapsed={elapsed:.0f}s"]
        if remaining is not None:
            parts.append(f"remaining={remaining:.0f}s")
        parts.append(f"incumbents={snap['incumbent_count']}")
        if snap["best_obj"] is not None:
            parts.append(f"incumbent={snap['best_obj']:,.0f}")
        if snap["best_bound"] is not None:
            parts.append(f"bound={snap['best_bound']:,.0f}")
        if snap["gap_pct"] is not None:
            parts.append(f"gap={snap['gap_pct']:.2f}%")
        logger.info("Solve progress | %s", " | ".join(parts))


def _extract_schedule(
    solver: cp_model.CpSolver,
    d: dict[str, Any],
    cp_vars: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build schedule list from interval-based variables."""

    schedule: list[dict[str, Any]] = []
    for batch_id in d["all_batches"]:
        assigned_roaster = None
        for roaster in d["sched_eligible_roasters"][batch_id]:
            if solver.Value(cp_vars["lit"][batch_id][roaster]) == 1:
                assigned_roaster = roaster
                break

        if assigned_roaster is None:
            continue

        sku = d["batch_sku"][batch_id]
        duration = int(d["roast_time_by_sku"][sku])
        start_time = int(solver.Value(cp_vars["start"][batch_id]))

        output_line = None
        if d["sku_credits_rc"][sku]:
            if (
                assigned_roaster == "R3"
                and d["allow_r3_flex"]
                and batch_id in cp_vars.get("z_l1", {})
            ):
                output_line = "L1" if solver.Value(cp_vars["z_l1"][batch_id]) == 1 else "L2"
            else:
                output_line = d["roaster_can_output"][assigned_roaster][0]

        schedule.append(
            {
                "batch_id": str(batch_id),
                "job_id": batch_id[0] if d["batch_is_mto"][batch_id] else None,
                "sku": sku,
                "roaster": assigned_roaster,
                "start": start_time,
                "end": start_time + duration,
                "pipeline": d["roaster_pipeline"][assigned_roaster],
                "pipeline_start": start_time,
                "pipeline_end": start_time + int(d["consume_time"]),
                "output_line": output_line,
                "is_mto": bool(d["batch_is_mto"][batch_id]),
                "status": "completed",
            }
        )

    schedule.sort(key=lambda entry: (entry["start"], entry["roaster"], entry["batch_id"]))
    return schedule


def _extract_restocks(
    solver: cp_model.CpSolver,
    d: dict[str, Any],
    cp_vars: dict[str, Any],
) -> list[dict[str, Any]]:
    restocks: list[dict[str, Any]] = []
    for pair, slot_list in cp_vars.get("restock_slots", {}).items():
        for slot in slot_list:
            if solver.Value(slot["present"]) == 1:
                start_time = int(solver.Value(slot["start"]))
                restocks.append(
                    {
                        "line_id": pair[0],
                        "sku": pair[1],
                        "start": start_time,
                        "end": start_time + int(d["restock_duration"]),
                        "qty": int(d["restock_qty"]),
                    }
                )
    restocks.sort(key=lambda item: (item["start"], item["line_id"], item["sku"]))
    return restocks


def _build_rc_timeline(
    schedule: list[dict[str, Any]],
    d: dict[str, Any],
) -> dict[str, list[int]]:
    shift_length = int(d["shift_length"])
    rc_level = {line_id: [0] * shift_length for line_id in d["lines"]}

    for line_id in d["lines"]:
        level = int(d["rc_init"][line_id])
        consumption_set = set(d["consumption_events"][line_id])
        completions: dict[int, int] = {}

        for entry in schedule:
            if entry["sku"] != "PSC" or entry["output_line"] != line_id:
                continue
            completion = int(entry["end"])
            if completion < shift_length:
                completions[completion] = completions.get(completion, 0) + 1

        for minute in range(shift_length):
            if minute in completions:
                level += completions[minute]
            if minute in consumption_set:
                level -= 1
            rc_level[line_id][minute] = level

    return rc_level


def _build_gc_timeline(
    schedule: list[dict[str, Any]],
    restocks: list[dict[str, Any]],
    d: dict[str, Any],
) -> dict[tuple[str, str], list[int]]:
    shift_length = int(d["shift_length"])
    gc_level = {pair: [0] * shift_length for pair in d["feasible_gc_pairs"]}

    start_events: dict[tuple[str, str], dict[int, int]] = {
        pair: {} for pair in d["feasible_gc_pairs"]
    }
    for entry in schedule:
        pair = (entry["pipeline"], entry["sku"])
        if pair not in start_events:
            continue
        start_time = int(entry["start"])
        if start_time < shift_length:
            start_events[pair][start_time] = start_events[pair].get(start_time, 0) + 1

    restock_events: dict[tuple[str, str], dict[int, int]] = {
        pair: {} for pair in d["feasible_gc_pairs"]
    }
    for rst in restocks:
        pair = (rst["line_id"], rst["sku"])
        completion = int(rst["end"])
        if completion < shift_length:
            restock_events[pair][completion] = restock_events[pair].get(completion, 0) + int(rst["qty"])

    for pair in d["feasible_gc_pairs"]:
        level = int(d["gc_init"][pair])
        for minute in range(shift_length):
            if minute in restock_events[pair]:
                level += restock_events[pair][minute]
            if minute in start_events[pair]:
                level -= start_events[pair][minute]
            gc_level[pair][minute] = level

    return gc_level


def _extract_model_rc_timeline(
    solver: cp_model.CpSolver,
    d: dict[str, Any],
    cp_vars: dict[str, Any],
) -> dict[str, list[int]]:
    return {
        line_id: [
            int(solver.Value(cp_vars["rc_level"][line_id][minute]))
            for minute in range(int(d["shift_length"]))
        ]
        for line_id in d["lines"]
    }


def _count_schedule_setup_events(
    schedule: list[dict[str, Any]],
    d: dict[str, Any],
) -> tuple[int, dict[str, dict[str, str]]]:
    by_roaster: dict[str, list[dict[str, Any]]] = {roaster: [] for roaster in d["roasters"]}
    for entry in schedule:
        by_roaster[entry["roaster"]].append(entry)
    for roaster in by_roaster:
        by_roaster[roaster].sort(key=lambda item: (item["start"], item["end"], item["batch_id"]))

    setup_events = 0
    setup_label: dict[str, dict[str, str]] = {}
    for roaster, entries in by_roaster.items():
        prev_sku = d["roaster_initial_sku"][roaster]
        for index, entry in enumerate(entries):
            if index == 0:
                needs_setup = bool(
                    entry["is_mto"] and entry["sku"] != prev_sku
                )
            else:
                needs_setup = bool(entry["sku"] != prev_sku)
            setup_label[entry["batch_id"]] = {
                "setup": f"Yes({d['setup_time']}m)" if needs_setup else "No"
            }
            if needs_setup:
                setup_events += 1
            prev_sku = entry["sku"]
    return setup_events, setup_label


def _compute_idle_overflow(
    schedule: list[dict[str, Any]],
    d: dict[str, Any],
) -> tuple[int, int]:
    """Compute safety-idle and overflow-idle costs from extracted schedule."""

    shift_length = int(d["shift_length"])
    max_rc = int(d["max_rc"])
    safety_stock = int(d["safety_stock"])

    busy_intervals = {roaster: [] for roaster in d["roasters"]}
    for entry in schedule:
        busy_intervals[entry["roaster"]].append((int(entry["start"]), int(entry["end"])))

    rc_level = _build_rc_timeline(schedule, d)

    idle_min = 0
    over_min = 0
    for roaster in d["roasters"]:
        line_id = d["roaster_line"][roaster]
        downtime = d["downtime_slots"].get(roaster, set())
        for minute in range(shift_length):
            if minute in downtime:
                continue
            busy = any(start_time <= minute < end_time for start_time, end_time in busy_intervals[roaster])
            if busy:
                continue
            if rc_level[line_id][minute] < safety_stock:
                idle_min += 1
            if roaster == "R3" and d["allow_r3_flex"]:
                if rc_level["L1"][minute] >= max_rc and rc_level["L2"][minute] >= max_rc:
                    over_min += 1
            else:
                out_line = d["roaster_can_output"][roaster][0]
                if rc_level[out_line][minute] >= max_rc:
                    over_min += 1

    return idle_min, over_min


def solve(
    d: dict[str, Any],
    model: cp_model.CpModel,
    cp_vars: dict[str, Any],
    num_workers: int = 1,
) -> dict[str, Any] | None:
    """Solve and extract results. Returns same result dict schema."""

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(d["time_limit"])
    solver.parameters.relative_gap_limit = float(d["mip_gap"])
    solver.parameters.log_search_progress = _verbose
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.cp_model_presolve = False
    solver.parameters.hint_conflict_limit = 100000
    solver.parameters.search_branching = cp_model.HINT_SEARCH

    callback = ProgressCallback(logger)

    build_meta = getattr(model, "_build_meta", None)
    if build_meta:
        logger.info("-" * 60)
        logger.info("MODEL STATISTICS (v2)")
        logger.info("  Formulation : %s", build_meta.get("formulation", "unknown"))
        logger.info("  Build time  : %.2fs", build_meta.get("build_seconds", 0.0))
        logger.info("  Variables   : %d", sum(build_meta.get("var_counts", {}).values()))
        logger.info("  Constraints : %d", sum(build_meta.get("constraint_counts", {}).values()))
        logger.info("  Notes       : %s", "; ".join(build_meta.get("notes", [])))
        logger.info("-" * 60)

    logger.info(
        "CP-SAT solve starting | time_limit=%ss | gap_target=%.2f%% | workers=%d",
        d["time_limit"],
        float(d["mip_gap"]) * 100.0,
        num_workers,
    )

    t_start = time.time()
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_solver_heartbeat,
        args=(heartbeat_stop, t_start, d.get("time_limit"), callback),
        daemon=True,
        name="cpsat-v2-heartbeat",
    )
    heartbeat_thread.start()
    try:
        status = solver.Solve(model, callback)
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)

    solve_time = time.time() - t_start
    status_name = solver.StatusName(status)
    feasible_statuses = (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if status not in feasible_statuses:
        logger.error("CP-SAT returned %s", status_name)
        return None

    obj_value = float(solver.ObjectiveValue())
    best_bound = float(solver.BestObjectiveBound())
    gap_pct = None
    if abs(best_bound) > 1e-6:
        gap_pct = abs(best_bound - obj_value) / abs(best_bound) * 100.0

    num_branches = int(solver.NumBranches())
    num_conflicts = int(solver.NumConflicts())

    logger.info(
        "CP-SAT finished | status=%s | obj=%s | bound=%s | gap=%s | incumbents=%d | branches=%d | conflicts=%d | time=%.2fs",
        status_name,
        f"{obj_value:,.0f}",
        f"{best_bound:,.0f}",
        f"{gap_pct:.4f}%" if gap_pct is not None else "N/A",
        callback.incumbent_count,
        num_branches,
        num_conflicts,
        solve_time,
    )

    schedule = _extract_schedule(solver, d, cp_vars)
    restocks = _extract_restocks(solver, d, cp_vars)

    setup_events = int(
        sum(
            solver.Value(setup_var)
            for batch_map in cp_vars.get("setup_before", {}).values()
            for setup_var in batch_map.values()
        )
    )
    idle_min = int(
        sum(
            solver.Value(cp_vars["idle"][roaster][minute])
            for roaster in d["roasters"]
            for minute in range(int(d["shift_length"]))
        )
    )
    over_min = int(
        sum(
            solver.Value(cp_vars["over"][roaster][minute])
            for roaster in d["roasters"]
            for minute in range(int(d["shift_length"]))
        )
    )

    posthoc_setup_events, setup_labels = _count_schedule_setup_events(schedule, d)
    for entry in schedule:
        entry["setup"] = setup_labels.get(entry["batch_id"], {}).get("setup", "No")

    psc_count = sum(1 for entry in schedule if entry["sku"] == "PSC")
    ndg_count = sum(1 for entry in schedule if entry["sku"] == "NDG")
    busta_count = sum(1 for entry in schedule if entry["sku"] == "BUSTA")

    revenue_psc = psc_count * float(d["sku_revenue"]["PSC"])
    revenue_ndg = ndg_count * float(d["sku_revenue"]["NDG"])
    revenue_busta = busta_count * float(d["sku_revenue"]["BUSTA"])
    total_revenue = revenue_psc + revenue_ndg + revenue_busta

    tardiness_min = {
        job_id: float(max(0, solver.Value(cp_vars["tard"][job_id])))
        for job_id in d["jobs"]
    }
    tard_cost_pure = sum(tardiness_min.values()) * float(d["cost_tardiness"])
    setup_cost = setup_events * float(d["cost_setup"])
    idle_cost = idle_min * float(d["cost_idle"])
    over_cost = over_min * float(d["cost_overflow"])

    stockout_events = {line_id: 0 for line_id in d["lines"]}
    for (line_id, t), bv in cp_vars.get("stockout", {}).items():
        if solver.Value(bv) == 1:
            stockout_events[line_id] += 1
    stockout_total = sum(stockout_events.values())
    stockout_cost = float(stockout_total) * float(d["cost_stockout"])

    mto_skipped = int(sum(solver.Value(bv) for bv in cp_vars.get("skipped", {}).values()))
    skip_cost = float(mto_skipped) * float(d["cost_skip_mto"])

    # Engine convention: kpi.tard_cost contains BOTH pure tardiness AND MTO skip penalty.
    tard_cost = tard_cost_pure + skip_cost
    total_costs = tard_cost + setup_cost + idle_cost + over_cost + stockout_cost
    exact_profit = total_revenue - total_costs

    rc_timeline = _extract_model_rc_timeline(solver, d, cp_vars)
    gc_timeline = _build_gc_timeline(schedule, restocks, d)
    rc_final = {
        line_id: int(levels[-1]) if levels else int(d["rc_init"][line_id])
        for line_id, levels in rc_timeline.items()
    }
    stockout_duration = {
        line_id: int(sum(1 for v in levels if v < 0))
        for line_id, levels in rc_timeline.items()
    }
    gc_final = {
        f"{pair[0]}_{pair[1]}": int(levels[-1]) if levels else int(d["gc_init"][pair])
        for pair, levels in gc_timeline.items()
    }

    debug_idle_min, debug_over_min = _compute_idle_overflow(schedule, d)
    if setup_events != posthoc_setup_events:
        logger.warning(
            "Exact setup count (%d) differs from schedule replay (%d).",
            setup_events,
            posthoc_setup_events,
        )
    if idle_min != debug_idle_min or over_min != debug_over_min:
        logger.warning(
            "Exact idle/overflow (%d, %d) differs from schedule replay (%d, %d).",
            idle_min,
            over_min,
            debug_idle_min,
            debug_over_min,
        )
    if abs(obj_value - exact_profit) > 1e-6:
        logger.warning(
            "Solver objective %.6f differs from exact deterministic profit %.6f.",
            obj_value,
            exact_profit,
        )

    return {
        "solver_engine": "CP-SAT",
        "solver_name": "CP_SAT_v2 (OR-Tools)",
        "status": status_name,
        "solve_time": round(float(solve_time), 3),
        "num_incumbents": int(callback.incumbent_count),
        "solution_history": list(callback.solution_history),
        "obj_value": round(obj_value, 2),
        "best_bound": round(best_bound, 2),
        "lp_bound": round(best_bound, 2),
        "gap_pct": round(gap_pct, 4) if gap_pct is not None else None,
        "node_count": int(num_branches),
        "num_branches": int(num_branches),
        "num_conflicts": int(num_conflicts),
        "net_profit": round(exact_profit, 2),
        "objective_profit": round(exact_profit, 2),
        "total_revenue": round(total_revenue, 2),
        "total_costs": round(total_costs, 2),
        "psc_count": int(psc_count),
        "ndg_count": int(ndg_count),
        "busta_count": int(busta_count),
        "total_batches": int(psc_count + ndg_count + busta_count),
        "revenue_psc": round(revenue_psc, 2),
        "revenue_ndg": round(revenue_ndg, 2),
        "revenue_busta": round(revenue_busta, 2),
        "tardiness_min": {job_id: round(value, 2) for job_id, value in tardiness_min.items()},
        "tard_cost": round(tard_cost, 2),
        "tard_cost_pure": round(tard_cost_pure, 2),
        "skip_cost": round(skip_cost, 2),
        "mto_skipped": int(mto_skipped),
        "stockout_events": dict(stockout_events),
        "stockout_duration": dict(stockout_duration),
        "stockout_cost": round(stockout_cost, 2),
        "setup_events": int(setup_events),
        "setup_cost": round(setup_cost, 2),
        "idle_min": round(float(idle_min), 2),
        "idle_cost": round(idle_cost, 2),
        "over_min": round(float(over_min), 2),
        "over_cost": round(over_cost, 2),
        "restock_count": int(len(restocks)),
        "allow_r3_flex": bool(d["allow_r3_flex"]),
        "inventory_option": "FULL_DETERMINISTIC",
        "schedule": schedule,
        "restocks": restocks,
        "sku_revenue": {sku: float(value) for sku, value in d["sku_revenue"].items()},
        "cost_tardiness": float(d["cost_tardiness"]),
        "cost_idle": float(d["cost_idle"]),
        "cost_overflow": float(d["cost_overflow"]),
        "cost_setup": float(d["cost_setup"]),
        "cost_stockout": float(d["cost_stockout"]),
        "cost_skip_mto": float(d["cost_skip_mto"]),
        "roast_time_by_sku": dict(d["roast_time_by_sku"]),
        "gc_init": {f"{pair[0]}_{pair[1]}": int(value) for pair, value in d["gc_init"].items()},
        "gc_final": gc_final,
        "gc_capacity": {f"{pair[0]}_{pair[1]}": int(value) for pair, value in d["gc_capacity"].items()},
        "rc_init": dict(d["rc_init"]),
        "rc_final": rc_final,
        "restock_duration": int(d["restock_duration"]),
        "restock_qty": int(d["restock_qty"]),
        "input_dir": d.get("input_dir"),
        "model_notes": list((getattr(model, "_build_meta", {}) or {}).get("notes", [])),
    }
