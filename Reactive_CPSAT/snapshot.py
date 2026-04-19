"""Translate a simulation trace snapshot + UPS event into a d-dict for CP-SAT.

The reactive d-dict mirrors CP_SAT_v2/data.py output but with:
  - Shortened horizon  (480 - t0)
  - RC / GC initial values read from the trace
  - In-progress batches on healthy roasters as fixed intervals
  - The downed roaster blocked for the UPS repair duration
  - Remaining MTO jobs rebuilt from simulation history
"""

from __future__ import annotations

import ast
import logging
import math
from copy import deepcopy
from typing import Any

logger = logging.getLogger("reactive_cpsat.snapshot")


def _parse_gc_stock(gc_flat: dict[str, int]) -> dict[tuple[str, str], int]:
    """Convert flattened "L1_PSC" keys back to (line, sku) tuples."""
    result: dict[tuple[str, str], int] = {}
    for key, value in gc_flat.items():
        parts = key.split("_", 1)
        if len(parts) == 2:
            result[(parts[0], parts[1])] = int(value)
    return result


def _parse_batch_id(raw: Any) -> tuple | None:
    """Recover a tuple batch_id from whatever the trace serialised it as."""
    if isinstance(raw, (list, tuple)):
        return tuple(raw)
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, tuple):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return None


def reconstruct_mto_remaining(
    completed_batches: list,
    cancelled_batches: list,
    trace_at_t0: dict,
    base_params: dict,
    ups_roaster: str,
) -> dict[str, int]:
    """Reconstruct mto_remaining at the moment of UPS (after UPS processing).

    At time t0 (post-UPS processing):
      mto_remaining[job] = initial - completed_by_t0 - running_on_other_roasters

    Running batches on non-UPS roasters will become fixed intervals in the
    oracle model, so they are NOT counted as remaining-to-schedule.
    """
    t0 = trace_at_t0["t"]
    remaining: dict[str, int] = {}
    for job in base_params["jobs"]:
        remaining[job] = int(base_params["job_batches"][job])

    # Subtract completed MTO batches that finished by t0
    for batch in completed_batches:
        if batch.is_mto and int(batch.end) <= t0:
            job_id = batch.batch_id[0] if isinstance(batch.batch_id, tuple) else batch.batch_id
            if job_id in remaining:
                remaining[job_id] -= 1

    # Subtract running MTO batches on non-UPS roasters (they become fixed intervals)
    for r_id, batch_info in trace_at_t0["current_batch"].items():
        if batch_info is None:
            continue
        if r_id == ups_roaster:
            continue  # UPS roaster batch was cancelled; not in progress
        if not batch_info.get("is_mto", False):
            continue
        bid = _parse_batch_id(batch_info["batch_id"])
        if bid is not None:
            job_id = bid[0]
        else:
            job_id = str(batch_info["batch_id"])
        if job_id in remaining:
            remaining[job_id] -= 1

    # Clamp to >= 0
    for job in remaining:
        remaining[job] = max(0, remaining[job])

    return remaining


def build_oracle_d(
    base_params: dict,
    ups_events: list,
) -> dict[str, Any]:
    """Build a d-dict for the FULL-SHIFT oracle CP-SAT model.

    Unlike build_reactive_d(), this mode assumes perfect foresight: every UPS
    event is known at t=0 and encoded as forced-idle intervals on its roaster.
    The solver produces the theoretical upper bound achievable given the UPS
    schedule — no online re-solve, no re-planning.
    """
    SL = int(base_params["shift_length"])

    d: dict[str, Any] = {}

    d["shift_length"] = SL
    d["setup_time"] = int(base_params["setup_time"])
    d["consume_time"] = int(base_params["consume_time"])
    d["max_rc"] = int(base_params["max_rc"])
    d["safety_stock"] = int(base_params["safety_stock"])
    d["restock_duration"] = int(base_params["restock_duration"])
    d["restock_qty"] = int(base_params["restock_qty"])
    d["allow_r3_flex"] = bool(base_params["allow_r3_flex"])
    d["cost_tardiness"] = base_params["cost_tardiness"]
    d["cost_idle"] = base_params["cost_idle"]
    d["cost_overflow"] = base_params["cost_overflow"]
    d["cost_setup"] = base_params["cost_setup"]
    d["time_limit"] = int(base_params.get("time_limit", 120))
    d["mip_gap"] = float(base_params.get("mip_gap", 0.0))

    d["lines"] = list(base_params["lines"])
    d["roasters"] = list(base_params["roasters"])
    d["roaster_line"] = dict(base_params["roaster_line"])
    d["roaster_pipeline"] = dict(base_params["roaster_pipeline"])
    d["roaster_can_output"] = {k: list(v) for k, v in base_params["roaster_can_output"].items()}
    d["roaster_eligible_skus"] = {k: list(v) for k, v in base_params["roaster_eligible_skus"].items()}
    d["skus"] = list(base_params["skus"])
    d["sku_revenue"] = dict(base_params["sku_revenue"])
    d["sku_credits_rc"] = dict(base_params["sku_credits_rc"])
    d["roast_time_by_sku"] = dict(base_params["roast_time_by_sku"])
    d["sku_eligible_roasters"] = {k: list(v) for k, v in base_params["sku_eligible_roasters"].items()}

    d["rc_init"] = dict(base_params["rc_init"])
    d["gc_init"] = dict(base_params["gc_init"])
    d["gc_capacity"] = dict(base_params["gc_capacity"])
    d["feasible_gc_pairs"] = list(base_params["feasible_gc_pairs"])
    d["roaster_initial_sku"] = dict(base_params["roaster_initial_sku"])
    d["consumption_events"] = {k: list(v) for k, v in base_params["consumption_events"].items()}

    # Planned downtime + ALL UPS events merged upfront. This is the key oracle
    # property: the solver sees every disruption before it happens.
    downtime = {r: set(slots) for r, slots in base_params["downtime_slots"].items()}
    for ev in ups_events:
        r = str(ev.roaster_id)
        t = int(ev.t)
        dur = int(ev.duration)
        end = min(t + dur, SL)
        if r not in downtime:
            downtime[r] = set()
        downtime[r].update(range(t, end))
    d["downtime_slots"] = downtime

    # Oracle mode: solve from t=0, no prior state
    d["t0"] = 0
    d["ups_roaster"] = ""    # multiple UPS — not a single event
    d["ups_duration"] = 0
    d["fixed_intervals"] = []

    d["jobs"] = list(base_params["jobs"])
    d["job_sku"] = dict(base_params["job_sku"])
    d["job_batches"] = dict(base_params["job_batches"])
    d["job_due"] = dict(base_params["job_due"])
    d["job_release"] = dict(base_params.get("job_release", {}))

    d["mto_batches"] = list(base_params["mto_batches"])
    d["psc_pool"] = list(base_params["psc_pool"])
    d["all_batches"] = list(base_params["all_batches"])
    d["psc_pool_per_roaster"] = int(base_params["psc_pool_per_roaster"])

    d["batch_sku"] = dict(base_params["batch_sku"])
    d["batch_is_mto"] = dict(base_params["batch_is_mto"])
    d["batch_eligible_roasters"] = {k: list(v) for k, v in base_params["batch_eligible_roasters"].items()}
    d["sched_eligible_roasters"] = {k: list(v) for k, v in base_params["sched_eligible_roasters"].items()}
    d["MS_by_sku"] = dict(base_params["MS_by_sku"])
    d["pair_to_roasters"] = dict(base_params.get("pair_to_roasters", {}))

    d["stockout_soft"] = True
    # Pull penalties from shift_parameters.csv so CP-SAT's objective exactly
    # mirrors the simulation engine's cost model.
    d["stockout_penalty"] = int(float(base_params.get("cost_stockout", 1500)))
    d["c_skip_mto"] = int(float(base_params.get("cost_skip_mto", 100000)))
    d["mto_soft"] = True

    d["input_dir"] = base_params.get("input_dir", "Input_data")

    logger.info(
        "Oracle d-dict built: horizon=%d, ups_events=%d, total_downtime_slots=%d, "
        "mto_batches=%d, psc_pool=%d, rc_init=%s, gc_init_sum=%d",
        SL, len(ups_events),
        sum(len(v) for v in d["downtime_slots"].values()),
        len(d["mto_batches"]), len(d["psc_pool"]),
        d["rc_init"], sum(d["gc_init"].values()),
    )

    return d


def build_reactive_d(
    trace_at_t0: dict,
    ups_event: Any,
    base_params: dict,
    mto_remaining: dict[str, int],
) -> dict[str, Any]:
    """Build a d-dict for the reactive CP-SAT model.

    Parameters
    ----------
    trace_at_t0 : dict
        ``state.trace[ups.t]`` — snapshot at the moment the UPS fired.
    ups_event : UPSEvent
        The disruption event (t, roaster_id, duration).
    base_params : dict
        Full parameter dict from ``CP_SAT_v2/data.py load()`` (or equivalent).
    mto_remaining : dict
        Per-job remaining MTO batch counts at t0 (already accounting for
        completed batches and in-progress fixed intervals).

    Returns
    -------
    dict
        A d-dict consumable by ``Reactive_CPSAT.model.build_reactive``.
    """
    t0 = int(ups_event.t)
    ups_roaster = str(ups_event.roaster_id)
    ups_duration = int(ups_event.duration)

    # ── Horizon ──────────────────────────────────────────────────────────
    total_sl = int(base_params["shift_length"])
    remaining_horizon = total_sl - t0
    if remaining_horizon <= 0:
        remaining_horizon = 1  # degenerate; model will be trivially empty

    d: dict[str, Any] = {}

    # ── Copy unchanged scalar params ─────────────────────────────────────
    d["shift_length"] = remaining_horizon
    d["setup_time"] = int(base_params["setup_time"])
    d["consume_time"] = int(base_params["consume_time"])
    d["max_rc"] = int(base_params["max_rc"])
    d["safety_stock"] = int(base_params["safety_stock"])
    d["restock_duration"] = int(base_params["restock_duration"])
    d["restock_qty"] = int(base_params["restock_qty"])
    d["allow_r3_flex"] = bool(base_params["allow_r3_flex"])
    d["cost_tardiness"] = base_params["cost_tardiness"]
    d["cost_idle"] = base_params["cost_idle"]
    d["cost_overflow"] = base_params["cost_overflow"]
    d["cost_setup"] = base_params["cost_setup"]
    d["time_limit"] = int(base_params.get("time_limit", 120))
    d["mip_gap"] = float(base_params.get("mip_gap", 0.0))

    # ── Topology (unchanged) ─────────────────────────────────────────────
    d["lines"] = list(base_params["lines"])
    d["roasters"] = list(base_params["roasters"])
    d["roaster_line"] = dict(base_params["roaster_line"])
    d["roaster_pipeline"] = dict(base_params["roaster_pipeline"])
    d["roaster_can_output"] = {k: list(v) for k, v in base_params["roaster_can_output"].items()}
    d["roaster_eligible_skus"] = {k: list(v) for k, v in base_params["roaster_eligible_skus"].items()}
    d["skus"] = list(base_params["skus"])
    d["sku_revenue"] = dict(base_params["sku_revenue"])
    d["sku_credits_rc"] = dict(base_params["sku_credits_rc"])
    d["roast_time_by_sku"] = dict(base_params["roast_time_by_sku"])
    d["sku_eligible_roasters"] = {k: list(v) for k, v in base_params["sku_eligible_roasters"].items()}

    # ── RC initial from trace ────────────────────────────────────────────
    d["rc_init"] = {line: int(trace_at_t0["rc_stock"][line]) for line in d["lines"]}

    # ── GC initial from trace (flattened "L1_PSC" → tuple keys) ──────────
    gc_parsed = _parse_gc_stock(trace_at_t0["gc_stock"])
    d["gc_init"] = {}
    d["gc_capacity"] = dict(base_params["gc_capacity"])
    d["feasible_gc_pairs"] = list(base_params["feasible_gc_pairs"])
    for pair in d["feasible_gc_pairs"]:
        d["gc_init"][pair] = gc_parsed.get(pair, 0)

    # ── Roaster initial SKU from trace ───────────────────────────────────
    d["roaster_initial_sku"] = {}
    for roaster in d["roasters"]:
        status = trace_at_t0["status"].get(roaster, "IDLE")
        if status == "SETUP":
            # Setup was aborted by UPS or is in progress — revert to last_sku
            d["roaster_initial_sku"][roaster] = trace_at_t0["last_sku"].get(
                roaster, base_params["roaster_initial_sku"].get(roaster, "PSC")
            )
        else:
            d["roaster_initial_sku"][roaster] = trace_at_t0["last_sku"].get(
                roaster, base_params["roaster_initial_sku"].get(roaster, "PSC")
            )

    # ── Consumption events — shifted to relative horizon ─────────────────
    d["consumption_events"] = {}
    for line in d["lines"]:
        original_events = base_params["consumption_events"][line]
        d["consumption_events"][line] = sorted(
            ev - t0 for ev in original_events if t0 <= ev < total_sl
        )

    # ── Planned downtime — shifted to relative horizon ───────────────────
    d["downtime_slots"] = {}
    for roaster in d["roasters"]:
        original = base_params["downtime_slots"].get(roaster, set())
        d["downtime_slots"][roaster] = {
            slot - t0 for slot in original
            if t0 <= slot < total_sl
        }

    # ── UPS blocking on downed roaster ───────────────────────────────────
    block_end = min(ups_duration, remaining_horizon)
    ups_block = set(range(0, block_end))
    existing = d["downtime_slots"].get(ups_roaster, set())
    d["downtime_slots"][ups_roaster] = existing | ups_block

    # Store UPS metadata for the model
    d["ups_roaster"] = ups_roaster
    d["ups_duration"] = ups_duration
    d["t0"] = t0

    # ── MTO jobs from mto_remaining ──────────────────────────────────────
    d["jobs"] = [job for job in base_params["jobs"] if mto_remaining.get(job, 0) > 0]
    d["job_sku"] = {job: base_params["job_sku"][job] for job in d["jobs"]}
    d["job_batches"] = {job: mto_remaining[job] for job in d["jobs"]}
    # Due times shifted to relative horizon (clamp to >= 0)
    d["job_due"] = {
        job: max(0, int(base_params["job_due"][job]) - t0)
        for job in d["jobs"]
    }
    d["job_release"] = {
        job: max(0, int(base_params.get("job_release", {}).get(job, 0)) - t0)
        for job in d["jobs"]
    }

    # ── Build batch lists ────────────────────────────────────────────────
    mto_batches: list[tuple] = []
    for job in d["jobs"]:
        for idx in range(d["job_batches"][job]):
            mto_batches.append((job, idx))

    # PSC pool — one pool per roaster, sized to fill remaining horizon
    min_roast = min(d["roast_time_by_sku"].values())
    psc_pool: list[tuple] = []
    for roaster in d["roasters"]:
        if "PSC" not in d["roaster_eligible_skus"][roaster]:
            continue
        psc_roast = int(d["roast_time_by_sku"]["PSC"])
        pool_size = max(0, remaining_horizon // psc_roast)
        for idx in range(pool_size):
            psc_pool.append((roaster, idx))

    d["mto_batches"] = mto_batches
    d["psc_pool"] = psc_pool
    d["all_batches"] = mto_batches + psc_pool
    d["psc_pool_per_roaster"] = max(0, remaining_horizon // int(d["roast_time_by_sku"].get("PSC", 15)))

    d["batch_sku"] = {}
    d["batch_is_mto"] = {}
    d["batch_eligible_roasters"] = {}
    d["sched_eligible_roasters"] = {}
    for batch_id in mto_batches:
        sku = d["job_sku"][batch_id[0]]
        d["batch_sku"][batch_id] = sku
        d["batch_is_mto"][batch_id] = True
        eligible = [r for r in d["roasters"] if sku in d["roaster_eligible_skus"][r]]
        d["batch_eligible_roasters"][batch_id] = eligible
        d["sched_eligible_roasters"][batch_id] = eligible
    for batch_id in psc_pool:
        d["batch_sku"][batch_id] = "PSC"
        d["batch_is_mto"][batch_id] = False
        d["batch_eligible_roasters"][batch_id] = [batch_id[0]]
        d["sched_eligible_roasters"][batch_id] = [batch_id[0]]

    d["MS_by_sku"] = {
        sku: remaining_horizon - int(d["roast_time_by_sku"][sku])
        for sku in d["skus"]
    }

    d["pair_to_roasters"] = dict(base_params.get("pair_to_roasters", {}))

    # ── Fixed intervals for in-progress batches on healthy roasters ──────
    fixed_intervals: list[dict[str, Any]] = []
    for r_id, batch_info in trace_at_t0["current_batch"].items():
        if batch_info is None:
            continue
        if r_id == ups_roaster:
            continue  # Cancelled by UPS
        status = trace_at_t0["status"].get(r_id, "IDLE")
        if status != "RUNNING":
            continue

        remaining_time = int(trace_at_t0["remaining"].get(r_id, 0))
        if remaining_time <= 0:
            continue

        sku = batch_info["sku"]

        # Pipeline remaining for the consume interval
        pipeline_line = d["roaster_pipeline"].get(r_id, "")
        pipe_remaining = 0
        if pipeline_line:
            pipe_mode = trace_at_t0["pipeline_mode"].get(pipeline_line, "FREE")
            if pipe_mode == "CONSUME":
                pipe_remaining = int(trace_at_t0["pipeline_busy"].get(pipeline_line, 0))

        output_line = None
        if d["sku_credits_rc"].get(sku, False):
            output_line = d["roaster_can_output"].get(r_id, [None])[0]

        bid = _parse_batch_id(batch_info["batch_id"])
        is_mto = batch_info.get("is_mto", False)
        job_id = None
        if is_mto and bid is not None:
            job_id = bid[0]

        fixed_intervals.append({
            "roaster": r_id,
            "sku": sku,
            "start": 0,
            "end": remaining_time,
            "pipeline_line": pipeline_line,
            "pipeline_remaining": pipe_remaining,
            "output_line": output_line,
            "is_mto": is_mto,
            "job_id": job_id,
        })

    d["fixed_intervals"] = fixed_intervals

    # ── Soft stockout flag ───────────────────────────────────────────────
    d["stockout_soft"] = True
    d["stockout_penalty"] = 1500  # $ per consumption event where RC < 0

    # ── Input dir (for reference data loading) ───────────────────────────
    d["input_dir"] = base_params.get("input_dir", "Input_data")

    logger.info(
        "Reactive d-dict built: t0=%d, horizon=%d, ups_roaster=%s, "
        "ups_dur=%d, mto_jobs=%d, mto_batches=%d, psc_pool=%d, "
        "fixed_intervals=%d, rc_init=%s, gc_init_sum=%d",
        t0, remaining_horizon, ups_roaster, ups_duration,
        len(d["jobs"]), len(mto_batches), len(psc_pool),
        len(fixed_intervals),
        d["rc_init"],
        sum(d["gc_init"].values()),
    )

    return d
