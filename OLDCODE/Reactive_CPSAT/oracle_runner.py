"""Full-shift CP-SAT oracle runner.

Solves one CP-SAT model over the full shift with perfect UPS foresight, then
replays the solver's schedule through the SimulationEngine to produce a KPI
directly comparable to the RL/heuristic methods (same engine, same costing).

The CP-SAT objective is reported alongside as the *theoretical upper bound*;
the simulated KPI is what an omniscient planner would actually realise.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("reactive_cpsat.oracle_runner")


def run_oracle_full_shift(
    base_params: dict[str, Any],
    sim_params: dict[str, Any],
    ups_events: list,
    time_limit_sec: int = 300,
) -> dict[str, Any]:
    """Solve full-shift oracle and replay its plan in the simulation engine.

    Parameters
    ----------
    base_params
        CP-SAT d-dict base (from ``Reactive_CPSAT.data.load``).
    sim_params
        SimulationEngine params (from ``env.data_bridge.get_sim_params``).
    ups_events
        Pre-generated UPS events — the same list is passed to both the
        solver (as forced-idle slots) and the simulation engine (as actual
        disruptions).
    time_limit_sec
        CP-SAT solver time limit for this single full-shift solve.

    Returns
    -------
    dict
        ``{
            "oracle_profit": float,      # CP-SAT objective (upper bound)
            "oracle_status": str,
            "best_bound": float,
            "gap_pct": float,
            "solve_time": float,
            "schedule": list,            # completed batches from the sim
            "cancelled_batches": list,
            "restocks": list,
            "ups_events": list,
            "baseline_kpi": dict,        # KPI produced by replaying the plan
            "ups_events_total": int,
        }``
    """
    from env.simulation_engine import SimulationEngine
    from .oracle import FixedScheduleStrategy, FullShiftOracleCPSAT

    oracle = FullShiftOracleCPSAT(base_params, time_limit_sec=time_limit_sec)
    solve_result = oracle.solve(ups_events)

    status = solve_result.get("oracle_status", "Unknown")
    logger.info(
        "FullShiftOracle plan ready: status=%s, CP-SAT profit=%s, batches=%d, restocks=%d",
        status,
        solve_result.get("oracle_profit"),
        len(solve_result.get("schedule", [])),
        len(solve_result.get("restocks", [])),
    )

    if solve_result.get("oracle_profit") is None:
        logger.warning(
            "FullShiftOracle produced no usable plan (%s) — simulation will replay nothing.",
            status,
        )

    strategy = FixedScheduleStrategy(
        sim_params,
        solve_result.get("schedule", []),
        solve_result.get("restocks", []),
    )
    engine = SimulationEngine(sim_params)
    kpi, state = engine.run(strategy, list(ups_events))
    baseline_kpi = kpi.to_dict()

    logger.info(
        "Oracle plan replayed: sim_net_profit=$%s, CP-SAT obj=%s (gap=%s%%)",
        f"{baseline_kpi.get('net_profit', 0):,.0f}",
        solve_result.get("oracle_profit"),
        solve_result.get("gap_pct"),
    )

    schedule = [
        {
            "batch_id": str(b.batch_id),
            "sku": b.sku,
            "roaster": b.roaster,
            "start": int(b.start),
            "end": int(b.end),
            "output_line": b.output_line,
            "is_mto": bool(b.is_mto),
            "status": "completed",
        }
        for b in state.completed_batches
    ]
    cancelled = [
        {
            "batch_id": str(b.batch_id),
            "sku": b.sku,
            "roaster": b.roaster,
            "start": int(b.start),
            "end": int(b.end),
            "output_line": b.output_line,
            "is_mto": bool(b.is_mto),
            "status": "cancelled",
        }
        for b in state.cancelled_batches
    ]
    restocks = [
        {
            "line_id": r.line_id,
            "sku": r.sku,
            "start": int(r.start),
            "end": int(r.end),
            "qty": int(r.qty),
        }
        for r in state.completed_restocks
    ]
    ups_dump = [
        {"t": int(ev.t), "roaster_id": str(ev.roaster_id), "duration": int(ev.duration)}
        for ev in state.ups_events_fired
    ]

    return {
        "oracle_profit": solve_result.get("oracle_profit"),
        "oracle_status": status,
        "best_bound": solve_result.get("best_bound"),
        "gap_pct": solve_result.get("gap_pct"),
        "solve_time": solve_result.get("solve_time"),
        "cpsat_schedule": solve_result.get("schedule", []),
        "cpsat_restocks": solve_result.get("restocks", []),
        "schedule": schedule,
        "cancelled_batches": cancelled,
        "restocks": restocks,
        "ups_events": ups_dump,
        "baseline_kpi": baseline_kpi,
        "ups_events_total": len(state.ups_events_fired),
    }
