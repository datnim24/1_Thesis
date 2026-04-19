"""Runner for the pure CP-SAT v3 solver.

Wraps data loading + UPS-as-downtime injection + model build + solve into one
call that returns a result dict compatible with ``master_eval``'s CP-SAT slot.
No simulation engine, no replay, no oracle objective / replay divergence.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable

from .data import load as load_data
from .model import build as build_model
from .solver import solve as solve_model


logger = logging.getLogger("cpsat_pure_runner")


def _merge_ups_as_downtime(d: dict[str, Any], ups_events: Iterable[Any]) -> int:
    """Add each UPS event as extra minutes in ``downtime_slots`` for its roaster.

    Matches the SimulationEngine's effective DOWN window semantics. Because the
    engine runs ``_step_roaster_timers`` in the same slot UPS fires — burning
    one tick before the next decision point — the observed DOWN span is
    ``duration - 1`` minutes starting at ``t`` (engine flips to IDLE at
    ``t + duration - 1``). We mirror that here so CP-SAT's forbidden starts
    align with what the env actually enforces.

    Events pointing at unknown roasters or entirely outside the shift are
    silently dropped. Returns the number of events applied.
    """

    shift_length = int(d["shift_length"])
    downtime = d["downtime_slots"]
    applied = 0
    for ev in ups_events:
        if ev is None:
            continue
        roaster_id = getattr(ev, "roaster_id", None) if not isinstance(ev, dict) else ev.get("roaster_id")
        t = getattr(ev, "t", None) if not isinstance(ev, dict) else ev.get("t")
        duration = getattr(ev, "duration", None) if not isinstance(ev, dict) else ev.get("duration")
        if roaster_id is None or t is None or duration is None:
            continue
        if roaster_id not in downtime:
            continue
        effective_duration = max(0, int(duration) - 1)
        start_min = max(0, int(t))
        end_min = min(shift_length, int(t) + effective_duration)
        if end_min <= start_min:
            continue
        downtime[roaster_id].update(range(start_min, end_min))
        applied += 1
    return applied


def run_pure_cpsat(
    time_limit_sec: int | float,
    ups_events: Iterable[Any] | None = None,
    input_dir: str = "Input_data",
    num_workers: int = 8,
    mip_gap: float | None = None,
) -> dict[str, Any]:
    """Load Input_data, merge UPS as planned stops, build and solve v3 CP-SAT.

    Returns the full result dict produced by ``solver.solve``. The dict's
    ``net_profit`` is env-equivalent under the deterministic (all-UPS-known)
    assumption — no replay gap, because UPS events were first-class constraints
    during the solve.
    """

    overrides: dict[str, Any] = {"time_limit": int(max(1, time_limit_sec))}
    if mip_gap is not None:
        overrides["mip_gap"] = float(mip_gap)

    d = load_data(input_dir=input_dir, overrides=overrides)

    applied = 0
    ups_list = list(ups_events) if ups_events else []
    if ups_list:
        applied = _merge_ups_as_downtime(d, ups_list)
        logger.info(
            "Merged %d UPS events as planned downtime across %d roasters.",
            applied,
            sum(1 for slots in d["downtime_slots"].values() if slots),
        )
    d["ups_events_applied"] = applied
    d["ups_events_list"] = ups_list

    t_build = time.perf_counter()
    model, cp_vars = build_model(d)
    build_time = time.perf_counter() - t_build

    result = solve_model(d, model, cp_vars, num_workers=int(num_workers))
    if result is None:
        raise RuntimeError("Pure CP-SAT returned no feasible solution within the time budget.")

    result["ups_events_applied"] = applied
    result["build_time_sec"] = round(build_time, 3)
    return result
