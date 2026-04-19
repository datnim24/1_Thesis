"""Reactive CP-SAT Oracle — offline re-optimisation at each UPS event.

For each UPS event that fired during a simulated shift, the oracle:
1. Reads state.trace[ups.t] — the world state right after the disruption
2. Builds a fresh CP-SAT model over the remaining horizon [ups.t, 479]
3. Fixes in-progress batches on healthy roasters as immovable intervals
4. Blocks the downed roaster for [ups.t, ups.t + ups.duration)
5. Solves offline (not real-time) — configurable time limit
6. Returns Oracle*(t0): best achievable profit for this disruption scenario
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ortools.sat.python import cp_model

from .model import build_reactive
from .snapshot import build_reactive_d, reconstruct_mto_remaining

logger = logging.getLogger("reactive_cpsat.oracle")


def _extract_schedule(
    solver: cp_model.CpSolver,
    d: dict[str, Any],
    cp_vars: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract the oracle's recommended schedule from solved model."""
    schedule: list[dict[str, Any]] = []
    t0 = d.get("t0", 0)

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
        if d["sku_credits_rc"].get(sku, False):
            if (
                assigned_roaster == "R3"
                and d["allow_r3_flex"]
                and batch_id in cp_vars.get("z_l1", {})
            ):
                output_line = "L1" if solver.Value(cp_vars["z_l1"][batch_id]) == 1 else "L2"
            else:
                output_line = d["roaster_can_output"][assigned_roaster][0]

        schedule.append({
            "batch_id": str(batch_id),
            "job_id": batch_id[0] if d["batch_is_mto"][batch_id] else None,
            "sku": sku,
            "roaster": assigned_roaster,
            "start": start_time + t0,       # absolute time
            "end": start_time + duration + t0,
            "start_rel": start_time,         # relative to t0
            "end_rel": start_time + duration,
            "pipeline": d["roaster_pipeline"][assigned_roaster],
            "output_line": output_line,
            "is_mto": bool(d["batch_is_mto"][batch_id]),
            "status": "oracle_scheduled",
        })

    # Include fixed intervals in schedule for completeness
    for fi in d.get("fixed_intervals", []):
        schedule.append({
            "batch_id": f"fixed_{fi['roaster']}_{fi['sku']}",
            "job_id": fi.get("job_id"),
            "sku": fi["sku"],
            "roaster": fi["roaster"],
            "start": fi["start"] + t0,
            "end": fi["end"] + t0,
            "start_rel": fi["start"],
            "end_rel": fi["end"],
            "pipeline": fi.get("pipeline_line", ""),
            "output_line": fi.get("output_line"),
            "is_mto": fi.get("is_mto", False),
            "status": "fixed_in_progress",
        })

    schedule.sort(key=lambda e: (e["start"], e["roaster"], e["batch_id"]))
    return schedule


def _extract_restocks(
    solver: cp_model.CpSolver,
    d: dict[str, Any],
    cp_vars: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract the oracle's restock plan."""
    restocks: list[dict[str, Any]] = []
    t0 = d.get("t0", 0)
    for pair, slot_list in cp_vars.get("restock_slots", {}).items():
        for slot in slot_list:
            if solver.Value(slot["present"]) == 1:
                start_time = int(solver.Value(slot["start"]))
                restocks.append({
                    "line_id": pair[0],
                    "sku": pair[1],
                    "start": start_time + t0,
                    "end": start_time + int(d["restock_duration"]) + t0,
                    "start_rel": start_time,
                    "end_rel": start_time + int(d["restock_duration"]),
                    "qty": int(d["restock_qty"]),
                })
    restocks.sort(key=lambda r: (r["start"], r["line_id"], r["sku"]))
    return restocks


class ReactiveCPSATOracle:
    """Offline oracle that re-solves CP-SAT at each UPS disruption event."""

    def __init__(self, base_params: dict[str, Any], time_limit_sec: int = 120):
        self.base_params = base_params
        self.time_limit_sec = time_limit_sec

    def solve_from_snapshot(
        self,
        trace_at_t0: dict,
        ups_event: Any,
        mto_remaining: dict[str, int],
    ) -> dict[str, Any]:
        """Solve the reactive model for a single UPS event.

        Returns a result dict with oracle profit, status, schedule, etc.
        """
        t0 = int(ups_event.t)
        ups_roaster = str(ups_event.roaster_id)
        ups_duration = int(ups_event.duration)

        logger.info(
            "Oracle solve START: t0=%d, roaster=%s, duration=%d, horizon_remaining=%d",
            t0, ups_roaster, ups_duration, 480 - t0,
        )

        # Build the reactive d-dict
        d = build_reactive_d(trace_at_t0, ups_event, self.base_params, mto_remaining)
        d["time_limit"] = self.time_limit_sec

        # Build the model
        model, cp_vars = build_reactive(d)

        # Handle trivial case (horizon too short)
        if cp_vars.get("trivial", False):
            logger.info("Oracle solve END (trivial): t0=%d, profit=0.0", t0)
            return {
                "ups_t": t0,
                "ups_roaster": ups_roaster,
                "ups_duration": ups_duration,
                "oracle_profit": 0.0,
                "oracle_status": "Trivial",
                "solve_time": 0.0,
                "gap_pct": None,
                "schedule": [],
                "restocks": [],
            }

        # Solve — use all available cores (offline oracle, not real-time)
        import os
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_sec
        solver.parameters.num_search_workers = min(8, os.cpu_count() or 4)
        solver.parameters.log_search_progress = False
        solver.parameters.hint_conflict_limit = 100_000
        solver.parameters.search_branching = cp_model.HINT_SEARCH

        t_start = time.perf_counter()
        status = solver.Solve(model)
        solve_time = time.perf_counter() - t_start

        feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        if not feasible:
            status_name = solver.StatusName(status)
            logger.warning(
                "Oracle solve END (infeasible): t0=%d, status=%s, time=%.1fs",
                t0, status_name, solve_time,
            )
            return {
                "ups_t": t0,
                "ups_roaster": ups_roaster,
                "ups_duration": ups_duration,
                "oracle_profit": None,
                "oracle_status": "Infeasible",
                "solve_time": round(solve_time, 2),
                "gap_pct": None,
                "schedule": [],
                "restocks": [],
            }

        obj_value = solver.ObjectiveValue()
        best_bound = solver.BestObjectiveBound()
        gap_pct = None
        if abs(obj_value) > 1e-9:
            gap_pct = round(100.0 * abs(best_bound - obj_value) / abs(obj_value), 2)

        oracle_status = "Optimal" if status == cp_model.OPTIMAL else "Feasible(TL)"
        schedule = _extract_schedule(solver, d, cp_vars)
        restocks = _extract_restocks(solver, d, cp_vars)

        logger.info(
            "Oracle solve END: t0=%d, roaster=%s, status=%s, "
            "profit=$%s, gap=%.2f%%, time=%.1fs, batches=%d",
            t0, ups_roaster, oracle_status,
            f"{obj_value:,.0f}", gap_pct or 0.0, solve_time, len(schedule),
        )

        return {
            "ups_t": t0,
            "ups_roaster": ups_roaster,
            "ups_duration": ups_duration,
            "oracle_profit": round(float(obj_value), 2),
            "oracle_status": oracle_status,
            "solve_time": round(solve_time, 2),
            "gap_pct": gap_pct,
            "schedule": schedule,
            "restocks": restocks,
        }

    def solve_all_events(self, state: Any) -> list[dict[str, Any]]:
        """Iterate over all UPS events that fired and solve from each snapshot.

        Parameters
        ----------
        state : SimulationState
            Full state returned by engine.run(). Must have .trace, .ups_events_fired,
            .completed_batches, .cancelled_batches attributes.

        Returns
        -------
        list[dict]
            One result dict per UPS event.
        """
        results: list[dict[str, Any]] = []
        events = state.ups_events_fired

        if not events:
            logger.info("No UPS events fired — nothing for oracle to solve.")
            return results

        logger.info("Oracle solving %d UPS event(s)...", len(events))

        for i, ev in enumerate(events):
            t0 = int(ev.t)

            # Validate trace availability
            if t0 >= len(state.trace):
                logger.error(
                    "UPS event at t=%d but trace only has %d entries — skipping",
                    t0, len(state.trace),
                )
                continue

            trace_at_t0 = state.trace[t0]

            # Reconstruct mto_remaining at this event's time
            mto_remaining = reconstruct_mto_remaining(
                completed_batches=state.completed_batches,
                cancelled_batches=state.cancelled_batches,
                trace_at_t0=trace_at_t0,
                base_params=self.base_params,
                ups_roaster=str(ev.roaster_id),
            )

            result = self.solve_from_snapshot(trace_at_t0, ev, mto_remaining)
            result["ups_index"] = i
            results.append(result)

        return results
