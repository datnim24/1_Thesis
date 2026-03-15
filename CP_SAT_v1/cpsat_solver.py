"""Stage 2 CP-SAT solver wrapper and result extractor."""

from __future__ import annotations

from ortools.sat.python import cp_model
import logging
import threading
import time


logger = logging.getLogger("cpsat_solver")

_verbose = False
_heartbeat_interval_sec = 300.0


def set_verbose(flag: bool):
    global _verbose
    _verbose = bool(flag)


def _solver_heartbeat(
    stop_event: threading.Event,
    start_time: float,
    time_limit: float | int | None,
    phase: str,
):
    while not stop_event.wait(_heartbeat_interval_sec):
        elapsed = time.time() - start_time
        if time_limit is None:
            logger.info("%s still running - elapsed: %.0fs", phase, elapsed)
            continue

        remaining = max(0.0, float(time_limit) - elapsed)
        logger.info(
            "%s still running - elapsed: %.0fs, remaining solver time: %.0fs",
            phase,
            elapsed,
            remaining,
        )


class ProgressCallback(cp_model.CpSolverSolutionCallback):
    """Fires on every new incumbent. Logs objective, bound, gap, and elapsed time."""

    def __init__(self, logger_ref):
        super().__init__()
        self._logger = logger_ref
        self._t0 = time.time()
        self._count = 0
        self._history: list[dict] = []

    def on_solution_callback(self):
        self._count += 1
        elapsed = time.time() - self._t0
        obj = self.ObjectiveValue()
        bound = self.BestObjectiveBound()
        gap = abs(bound - obj) / max(abs(bound), 1.0) * 100

        self._logger.info(
            f"[CP-SAT] Incumbent #{self._count:3d} | "
            f"obj={obj:>12,.0f} | "
            f"bound={bound:>12,.0f} | "
            f"gap={gap:>7.3f}% | "
            f"t={elapsed:>7.2f}s"
        )
        self._history.append(
            {
                "incumbent": int(self._count),
                "obj": float(obj),
                "bound": float(bound),
                "gap_pct": round(float(gap), 4),
                "elapsed_s": round(float(elapsed), 3),
            }
        )

    @property
    def incumbent_count(self):
        return self._count

    @property
    def solution_history(self):
        return self._history


def solve_lp_relaxation(
    d: dict,
    model: cp_model.CpModel,
    time_limit: int = 60,
) -> float | None:
    """
    Ask CP-SAT to report the LP relaxation bound.
    Sets linearization_level=2 so CP-SAT computes a tighter LP bound
    without requiring the normal full search path.
    """
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.linearization_level = 2
    solver.parameters.log_search_progress = _verbose
    solver.parameters.num_search_workers = 1

    t_start = time.time()
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_solver_heartbeat,
        args=(heartbeat_stop, t_start, time_limit, "LP relaxation"),
        daemon=True,
        name="cpsat-lp-heartbeat",
    )
    heartbeat_thread.start()
    try:
        status = solver.Solve(model)
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)

    status_name = solver.StatusName(status)
    bound = solver.BestObjectiveBound()
    logger.info(
        "LP relaxation bound: %s (status: %s)",
        f"{bound:,.0f}" if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) and bound is not None else "N/A",
        status_name,
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return float(bound) if bound is not None else None


def solve(
    d: dict,
    model: cp_model.CpModel,
    cp_vars: dict,
    num_workers: int = 1,
) -> dict | None:
    """Solve the CP-SAT model and return a JSON-safe results dict."""
    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = float(d["time_limit"])
    solver.parameters.relative_gap_limit = float(d["mip_gap"])
    solver.parameters.log_search_progress = _verbose
    solver.parameters.num_search_workers = int(num_workers)

    callback = ProgressCallback(logger)

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
        args=(heartbeat_stop, t_start, d.get("time_limit"), "CP-SAT solve"),
        daemon=True,
        name="cpsat-solve-heartbeat",
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
    obj_value = solver.ObjectiveValue() if status in feasible_statuses else None
    best_bound = solver.BestObjectiveBound() if status in feasible_statuses else None
    gap_pct = None
    if obj_value is not None and best_bound is not None and abs(best_bound) > 1e-6:
        gap_pct = abs(best_bound - obj_value) / abs(best_bound) * 100.0

    logger.info(
        "CP-SAT finished | status=%s | obj=%s | bound=%s | gap=%s | incumbents=%d | time=%.2fs",
        status_name,
        f"{obj_value:,.0f}" if obj_value is not None else "N/A",
        f"{best_bound:,.0f}" if best_bound is not None else "N/A",
        f"{gap_pct:.4f}%" if gap_pct is not None else "N/A",
        callback.incumbent_count,
        solve_time,
    )

    if status == cp_model.INFEASIBLE:
        logger.error("CP-SAT: Model is INFEASIBLE")
        return None
    if status == cp_model.UNKNOWN:
        logger.error("CP-SAT: No solution found (UNKNOWN status - time limit before first feasible?)")
        return None
    if status == cp_model.FEASIBLE:
        logger.warning("CP-SAT: Time limit hit - returning best feasible solution (not proven optimal)")

    schedule = _extract_schedule(d, cp_vars, solver)

    psc_count = sum(1 for entry in schedule if entry["sku"] == "PSC")
    ndg_count = sum(1 for entry in schedule if entry["sku"] == "NDG")
    busta_count = sum(1 for entry in schedule if entry["sku"] == "BUSTA")

    rev_psc = float(psc_count * d["sku_revenue"]["PSC"])
    rev_ndg = float(ndg_count * d["sku_revenue"]["NDG"])
    rev_busta = float(busta_count * d["sku_revenue"]["BUSTA"])
    total_rev = rev_psc + rev_ndg + rev_busta

    tard_minutes: dict[str, float] = {}
    for job_id in d["jobs"]:
        tard_minutes[job_id] = float(max(0, solver.Value(cp_vars["tard"][job_id])))
    total_tard_min = sum(tard_minutes.values())
    total_tard_cost = total_tard_min * float(d["cost_tardiness"])

    total_idle_min = sum(
        float(solver.Value(cp_vars["idle"][r][k]))
        for r in d["roasters"]
        for k in cp_vars["idle"].get(r, {})
    )
    total_idle_cost = total_idle_min * float(d["cost_idle"])

    total_over_min = sum(
        float(solver.Value(cp_vars["over"][r][k]))
        for r in d["roasters"]
        for k in cp_vars["over"].get(r, {})
    )
    total_over_cost = total_over_min * float(d["cost_overflow"])

    total_cost = total_tard_cost + total_idle_cost + total_over_cost
    net_profit = total_rev - total_cost

    return {
        "solver_engine": "CP-SAT",
        "solver_name": "CP-SAT (OR-Tools)",
        "status": status_name,
        "solve_time": round(float(solve_time), 3),
        "num_incumbents": int(callback.incumbent_count),
        "solution_history": list(callback.solution_history),
        "obj_value": round(float(obj_value), 2) if obj_value is not None else None,
        "best_bound": round(float(best_bound), 2) if best_bound is not None else None,
        "lp_bound": round(float(best_bound), 2) if best_bound is not None else None,
        "gap_pct": round(float(gap_pct), 4) if gap_pct is not None else None,
        "net_profit": round(float(net_profit), 2),
        "total_revenue": round(float(total_rev), 2),
        "total_costs": round(float(total_cost), 2),
        "psc_count": int(psc_count),
        "ndg_count": int(ndg_count),
        "busta_count": int(busta_count),
        "revenue_psc": round(float(rev_psc), 2),
        "revenue_ndg": round(float(rev_ndg), 2),
        "revenue_busta": round(float(rev_busta), 2),
        "tardiness_min": {job_id: round(float(value), 2) for job_id, value in tard_minutes.items()},
        "tard_cost": round(float(total_tard_cost), 2),
        "idle_min": round(float(total_idle_min), 2),
        "idle_cost": round(float(total_idle_cost), 2),
        "over_min": round(float(total_over_min), 2),
        "over_cost": round(float(total_over_cost), 2),
        "allow_r3_flex": bool(d["allow_r3_flex"]),
        "schedule": schedule,
        "sku_revenue": {sku: float(value) for sku, value in d["sku_revenue"].items()},
        "cost_tardiness": float(d["cost_tardiness"]),
        "cost_idle": float(d["cost_idle"]),
        "cost_overflow": float(d["cost_overflow"]),
    }


def _extract_schedule(d: dict, cp_vars: dict, solver: cp_model.CpSolver) -> list[dict]:
    schedule: list[dict] = []

    for batch_id in d["all_batches"]:
        is_mto = bool(d["batch_is_mto"][batch_id])
        if is_mto:
            active = True
        else:
            active = solver.Value(cp_vars["active"][batch_id]) == 1
        if not active:
            continue

        roaster_assigned = None
        for roaster_id in d["batch_eligible_roasters"][batch_id]:
            if solver.Value(cp_vars["assign"][batch_id][roaster_id]) == 1:
                roaster_assigned = roaster_id
                break

        if roaster_assigned is None:
            logger.warning("Active batch %s has no assigned roaster - skipping", batch_id)
            continue

        start_val = int(solver.Value(cp_vars["start"][batch_id]))
        end_val = int(start_val + d["process_time"])
        sku = d["batch_sku"][batch_id]

        if not d["sku_credits_rc"][sku]:
            output_line = None
        elif roaster_assigned == "R3" and d["allow_r3_flex"] and batch_id in cp_vars["y"]:
            output_line = "L1" if solver.Value(cp_vars["y"][batch_id]) == 1 else "L2"
        elif roaster_assigned == "R3":
            output_line = "L2"
        else:
            output_line = d["roaster_can_output"][roaster_assigned][0]

        pipeline_line = d["roaster_pipeline"][roaster_assigned]
        pipeline_window = (
            f"{pipeline_line}[{start_val}..{start_val + d['consume_time'] - 1}]"
        )

        schedule.append(
            {
                "batch_id": str(batch_id),
                "job_id": batch_id[0] if is_mto else None,
                "sku": sku,
                "roaster": roaster_assigned,
                "start": int(start_val),
                "end": int(end_val),
                "pipeline": pipeline_window,
                "output_line": output_line,
                "is_mto": bool(is_mto),
            }
        )

    schedule.sort(key=lambda entry: (entry["start"], entry["roaster"], entry["batch_id"]))

    prev_sku = {roaster_id: d["roaster_initial_sku"][roaster_id] for roaster_id in d["roasters"]}
    for entry in schedule:
        roaster_id = entry["roaster"]
        this_sku = entry["sku"]
        entry["setup"] = f"Yes({d['setup_time']}m)" if this_sku != prev_sku[roaster_id] else "No"
        prev_sku[roaster_id] = this_sku

    return schedule
