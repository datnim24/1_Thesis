"""CP_SAT_v2 model builder (soft-constraint, engine-consistent).

This version keeps the interval-based backbone for assignment/capacity logic,
and uses a soft cost model that mirrors the simulation engine exactly:

- PSC revenue
- MTO revenue (conditional on activation)
- tardiness cost (per minute past due)
- per-event stockout cost (RC < 0 at consume event)
- per-batch MTO skip cost (rolled into tard_cost in reporting)
- exact setup-event cost
- exact safety-idle cost
- exact overflow-idle cost

UPS is handled by pre-merging events into downtime_slots in runner.py
(perfect-information mode).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from ortools.sat.python import cp_model


logger = logging.getLogger("cpsat_model_v2")


def safe_id(batch_id: Any) -> str:
    return (
        str(batch_id)
        .replace(" ", "")
        .replace("'", "")
        .replace(",", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _forbidden_starts(downtime_slots: set[int], duration: int, max_start: int) -> set[int]:
    forbidden: set[int] = set()
    for down_slot in downtime_slots:
        for start in range(max(0, down_slot - duration + 1), min(max_start, down_slot) + 1):
            forbidden.add(start)
    return forbidden


def _as_int_coefficient(value: Any, label: str) -> int:
    numeric = float(value)
    rounded = round(numeric)
    if abs(numeric - rounded) > 1e-9:
        raise ValueError(f"CP-SAT objective coefficient '{label}' must be integral, got {value!r}")
    return int(rounded)


def _earliest_start(d: dict[str, Any], batch_id: Any) -> int:
    if not d["batch_is_mto"][batch_id]:
        return 0

    sku = d["batch_sku"][batch_id]
    release = int(d["job_release"].get(batch_id[0], 0))
    eligible = d["sched_eligible_roasters"][batch_id]
    all_need_setup = all(d["roaster_initial_sku"][roaster] != sku for roaster in eligible)
    return max(release, int(d["setup_time"])) if all_need_setup else release


def _allowed_starts(d: dict[str, Any], batch_id: Any, roaster: str) -> list[int]:
    sku = d["batch_sku"][batch_id]
    earliest = _earliest_start(d, batch_id)
    latest = int(d["MS_by_sku"][sku])
    downtime = d["downtime_slots"].get(roaster, set())
    if not downtime:
        return list(range(earliest, latest + 1))

    forbidden = _forbidden_starts(downtime, int(d["roast_time_by_sku"][sku]), latest)
    return [start for start in range(earliest, latest + 1) if start not in forbidden]


def _max_restocks(pair: tuple[str, str], d: dict[str, Any]) -> int:
    """Upper bound on restocks needed for one silo."""

    line_id, sku = pair
    init = int(d["gc_init"][pair])
    qty = int(d["restock_qty"])
    max_possible_batches = sum(
        1
        for batch_id in d["all_batches"]
        if d["batch_sku"][batch_id] == sku
        and any(
            d["roaster_pipeline"][roaster] == line_id
            for roaster in d["sched_eligible_roasters"][batch_id]
        )
    )
    needed = max(0, math.ceil((max_possible_batches - init) / max(qty, 1)))
    hard_cap = max(1, int(d["shift_length"]) // max(1, int(d["restock_duration"])))
    return max(1, min(needed + 2, hard_cap))


def _try_dispatch_hint(d: dict[str, Any]) -> dict[str, Any] | None:
    """Build a feasible deterministic hint from the env dispatch heuristic.

    Passes any merged UPS events to the hint simulator so its schedule respects
    the same planned-downtime windows the CP-SAT model constrains.
    """

    try:
        from dispatch.dispatching_heuristic import DispatchingHeuristic
        from env.data_bridge import get_sim_params
        from env.simulation_engine import SimulationEngine
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.warning("Dispatch hint unavailable: %s", exc)
        return None

    try:
        hint_params = get_sim_params()
        if (
            set(hint_params["all_batches"]) != set(d["all_batches"])
            or dict(hint_params["job_due"]) != dict(d["job_due"])
            or dict(hint_params["rc_init"]) != dict(d["rc_init"])
            or dict(hint_params["gc_init"]) != dict(d["gc_init"])
            or int(hint_params["SL"]) != int(d["shift_length"])
            or bool(hint_params["allow_r3_flex"]) != bool(d["allow_r3_flex"])
        ):
            logger.warning("Skipping dispatch hint because env params do not match CP_SAT_v2 input data.")
            return None

        engine = SimulationEngine(hint_params)
        strategy = DispatchingHeuristic(hint_params)
        hint_ups = list(d.get("ups_events_list") or [])
        if hint_ups:
            logger.info("Dispatch hint will respect %d UPS events as downtime.", len(hint_ups))
        kpi, state = engine.run(strategy, hint_ups)
    except Exception as exc:  # pragma: no cover - heuristic hint is optional
        logger.warning("Dispatch hint generation failed: %s", exc)
        return None

    schedule_by_batch = {batch.batch_id: batch for batch in state.completed_batches}
    restocks_by_pair: dict[tuple[str, str], list[Any]] = {}
    for restock in state.completed_restocks:
        restocks_by_pair.setdefault((restock.line_id, restock.sku), []).append(restock)
    for pair in restocks_by_pair:
        restocks_by_pair[pair].sort(key=lambda item: (int(item.start), int(item.end)))

    job_completion: dict[str, int] = {job_id: 0 for job_id in d["jobs"]}
    by_roaster: dict[str, list[Any]] = {roaster: [] for roaster in d["roasters"]}
    for batch in state.completed_batches:
        by_roaster[batch.roaster].append(batch)
        if batch.is_mto:
            job_completion[batch.batch_id[0]] = max(job_completion[batch.batch_id[0]], int(batch.end))
    for roaster in by_roaster:
        by_roaster[roaster].sort(key=lambda item: (int(item.start), int(item.end), str(item.batch_id)))

    tardiness = {
        job_id: max(0, job_completion[job_id] - int(d["job_due"][job_id]))
        for job_id in d["jobs"]
    }

    logger.info(
        "Applied dispatch heuristic seed candidate: %d batches, %d restocks, profit=%s",
        len(state.completed_batches),
        len(state.completed_restocks),
        kpi.net_profit(),
    )
    return {
        "schedule_by_batch": schedule_by_batch,
        "restocks_by_pair": restocks_by_pair,
        "tardiness": tardiness,
        "by_roaster": by_roaster,
    }


def build(d: dict[str, Any]) -> tuple[cp_model.CpModel, dict[str, Any]]:
    """Build deterministic CP-SAT model with exact setup/idle/overflow objective."""

    t_build = time.perf_counter()

    model = cp_model.CpModel()

    ST = int(d["setup_time"])
    DC = int(d["consume_time"])
    SL = int(d["shift_length"])
    MRC = int(d["max_rc"])
    SS = int(d["safety_stock"])
    RST_DUR = int(d["restock_duration"])
    RST_QTY = int(d["restock_qty"])

    all_batches = d["all_batches"]
    psc_pool = d["psc_pool"]
    mto_batches = d["mto_batches"]
    batch_sku = d["batch_sku"]
    sched_elig = d["sched_eligible_roasters"]
    roasters = d["roasters"]
    lines = d["lines"]
    roast_time = d["roast_time_by_sku"]
    feasible_gc_pairs = list(d["feasible_gc_pairs"])

    logger.info("Building CP_SAT_v2 (exact deterministic objective, no UPS)")
    logger.info("R3 mode: %s", "flex" if d["allow_r3_flex"] else "fixed")

    start: dict[Any, cp_model.IntVar] = {}
    end_expr: dict[Any, Any] = {}
    lit: dict[Any, dict[str, cp_model.IntVar]] = {}
    roast_interval: dict[Any, dict[str, cp_model.IntervalVar]] = {}
    consume_interval: dict[Any, dict[str, cp_model.IntervalVar]] = {}
    start_choice: dict[Any, dict[str, dict[int, cp_model.IntVar]]] = {}
    allowed_starts: dict[tuple[Any, str], list[int]] = {}

    tard: dict[str, cp_model.IntVar] = {}
    z_l1: dict[Any, cp_model.IntVar] = {}
    z_l2: dict[Any, cp_model.IntVar] = {}
    route_choice_l1: dict[Any, dict[int, cp_model.IntVar]] = {}
    route_choice_l2: dict[Any, dict[int, cp_model.IntVar]] = {}

    restock_slots: dict[tuple[str, str], list[dict[str, Any]]] = {}

    first_arc: dict[str, dict[Any, cp_model.IntVar]] = {roaster: {} for roaster in roasters}
    last_arc: dict[str, dict[Any, cp_model.IntVar]] = {roaster: {} for roaster in roasters}
    seq_arc: dict[str, dict[tuple[Any, Any], cp_model.IntVar]] = {roaster: {} for roaster in roasters}
    empty_arc: dict[str, cp_model.IntVar] = {}
    setup_before: dict[Any, dict[str, cp_model.IntVar]] = {}
    setup_start_choice: dict[Any, dict[str, dict[int, cp_model.IntVar]]] = {}

    running: dict[str, dict[int, cp_model.IntVar]] = {roaster: {} for roaster in roasters}
    setup_active: dict[str, dict[int, cp_model.IntVar]] = {roaster: {} for roaster in roasters}
    rc_level: dict[str, dict[int, cp_model.IntVar]] = {line_id: {} for line_id in lines}
    low: dict[str, dict[int, cp_model.IntVar]] = {line_id: {} for line_id in lines}
    full: dict[str, dict[int, cp_model.IntVar]] = {line_id: {} for line_id in lines}
    full_both: dict[int, cp_model.IntVar] = {}
    idle: dict[str, dict[int, cp_model.IntVar]] = {roaster: {} for roaster in roasters}
    over: dict[str, dict[int, cp_model.IntVar]] = {roaster: {} for roaster in roasters}

    running_terms: dict[str, list[list[cp_model.IntVar]]] = {
        roaster: [[] for _ in range(SL)] for roaster in roasters
    }
    setup_terms: dict[str, list[list[cp_model.IntVar]]] = {
        roaster: [[] for _ in range(SL)] for roaster in roasters
    }
    rc_completion_terms: dict[str, list[list[cp_model.IntVar]]] = {
        line_id: [[] for _ in range(SL)] for line_id in lines
    }

    var_counts = {
        "start": 0,
        "lit": 0,
        "roast_intervals": 0,
        "consume_intervals": 0,
        "start_choice": 0,
        "restock_intervals": 0,
        "z_route": 0,
        "route_choice": 0,
        "tard": 0,
        "first_last_arc": 0,
        "ordering_arc": 0,
        "setup_before": 0,
        "setup_start_choice": 0,
        "running": 0,
        "setup_active": 0,
        "rc_level": 0,
        "low": 0,
        "full": 0,
        "full_both": 0,
        "idle": 0,
        "over": 0,
        "skipped": 0,
        "stockout": 0,
    }
    constraint_counts = {
        "C1_C2": 0,
        "START_LINK": 0,
        "C4": 0,
        "SEQ": 0,
        "C5": 0,
        "C6": 0,
        "C7": 0,
        "C8_C17": 0,
        "C10_C11": 0,
        "C12": 0,
        "C13": 0,
        "C16": 0,
        "C18": 0,
        "RC_STATE": 0,
        "IDLE_OVER": 0,
        "SYM": 0,
        "RST_SYM": 0,
        "ROUTE_LINK": 0,
        "SETUP_EXACT": 0,
    }

    for batch_id in all_batches:
        sku = batch_sku[batch_id]
        duration = int(roast_time[sku])
        earliest = _earliest_start(d, batch_id)
        latest = int(d["MS_by_sku"][sku])
        start_var = model.NewIntVar(earliest, latest, f"start_{safe_id(batch_id)}")
        start[batch_id] = start_var
        end_expr[batch_id] = start_var + duration
        lit[batch_id] = {}
        roast_interval[batch_id] = {}
        consume_interval[batch_id] = {}
        start_choice[batch_id] = {}

        for roaster in sched_elig[batch_id]:
            present = model.NewBoolVar(f"lit_{safe_id(batch_id)}_{roaster}")
            lit[batch_id][roaster] = present
            roast_interval[batch_id][roaster] = model.new_optional_fixed_size_interval_var(
                start_var,
                duration,
                present,
                f"roast_{safe_id(batch_id)}_{roaster}",
            )
            consume_interval[batch_id][roaster] = model.new_optional_fixed_size_interval_var(
                start_var,
                DC,
                present,
                f"consume_{safe_id(batch_id)}_{roaster}",
            )

            allowed = _allowed_starts(d, batch_id, roaster)
            allowed_starts[(batch_id, roaster)] = allowed
            start_choice[batch_id][roaster] = {}
            for start_value in allowed:
                x_var = model.NewBoolVar(f"x_{safe_id(batch_id)}_{roaster}_{start_value}")
                start_choice[batch_id][roaster][start_value] = x_var
                for minute in range(start_value, min(SL, start_value + duration)):
                    running_terms[roaster][minute].append(x_var)

                if batch_sku[batch_id] == "PSC":
                    completion_time = start_value + duration
                    if completion_time < SL:
                        if roaster == "R3" and d["allow_r3_flex"]:
                            pass
                        else:
                            output_line = d["roaster_can_output"][roaster][0]
                            rc_completion_terms[output_line][completion_time].append(x_var)

            var_counts["start_choice"] += len(start_choice[batch_id][roaster])

        var_counts["start"] += 1
        var_counts["lit"] += len(lit[batch_id])
        var_counts["roast_intervals"] += len(roast_interval[batch_id])
        var_counts["consume_intervals"] += len(consume_interval[batch_id])

    skipped: dict[Any, Any] = {}
    for batch_id in mto_batches:
        skipped_var = model.NewBoolVar(f"skipped_{safe_id(batch_id)}")
        skipped[batch_id] = skipped_var
        model.Add(
            sum(lit[batch_id][roaster] for roaster in sched_elig[batch_id]) + skipped_var == 1
        )
        constraint_counts["C1_C2"] += 1
        var_counts["skipped"] += 1

    for batch_id in all_batches:
        earliest = _earliest_start(d, batch_id)
        active_expr = sum(lit[batch_id].values())
        start_expr_terms: list[Any] = []
        for roaster in sched_elig[batch_id]:
            choices = list(start_choice[batch_id][roaster].values())
            if choices:
                model.Add(sum(choices) == lit[batch_id][roaster])
            else:
                model.Add(lit[batch_id][roaster] == 0)
            constraint_counts["START_LINK"] += 1
            start_expr_terms.extend(
                start_value * choice
                for start_value, choice in start_choice[batch_id][roaster].items()
            )
        model.Add(start[batch_id] == sum(start_expr_terms) + earliest * (1 - active_expr))
        constraint_counts["START_LINK"] += 1

    for job_id in d["jobs"]:
        tard[job_id] = model.NewIntVar(0, SL, f"tard_{job_id}")
    var_counts["tard"] = len(tard)

    for pair in feasible_gc_pairs:
        pair_label = f"{pair[0]}_{pair[1]}"
        pool_size = _max_restocks(pair, d)
        restock_slots[pair] = []
        for idx in range(pool_size):
            present = model.NewBoolVar(f"rstP_{pair_label}_{idx}")
            start_var = model.NewIntVar(0, SL - RST_DUR, f"rstS_{pair_label}_{idx}")
            interval = model.new_optional_fixed_size_interval_var(
                start_var,
                RST_DUR,
                present,
                f"rstI_{pair_label}_{idx}",
            )
            pipe_interval = model.new_optional_fixed_size_interval_var(
                start_var,
                RST_DUR,
                present,
                f"rstPipe_{pair_label}_{idx}",
            )
            restock_slots[pair].append(
                {
                    "start": start_var,
                    "present": present,
                    "interval": interval,
                    "pipe_interval": pipe_interval,
                }
            )
            var_counts["restock_intervals"] += 1

        for left, right in zip(restock_slots[pair], restock_slots[pair][1:]):
            model.Add(left["present"] >= right["present"])
            constraint_counts["RST_SYM"] += 1
            model.Add(left["start"] <= right["start"]).OnlyEnforceIf(
                [left["present"], right["present"]]
            )
            constraint_counts["RST_SYM"] += 1

    if d["allow_r3_flex"]:
        for batch_id in psc_pool:
            if sched_elig[batch_id] != ["R3"]:
                continue
            left = model.NewBoolVar(f"z_L1_{safe_id(batch_id)}")
            right = model.NewBoolVar(f"z_L2_{safe_id(batch_id)}")
            z_l1[batch_id] = left
            z_l2[batch_id] = right
            model.Add(left + right == lit[batch_id]["R3"])
            constraint_counts["C13"] += 1

            route_choice_l1[batch_id] = {}
            route_choice_l2[batch_id] = {}
            for start_value, x_var in start_choice[batch_id]["R3"].items():
                l1_choice = model.NewBoolVar(f"routeL1_{safe_id(batch_id)}_{start_value}")
                l2_choice = model.NewBoolVar(f"routeL2_{safe_id(batch_id)}_{start_value}")
                route_choice_l1[batch_id][start_value] = l1_choice
                route_choice_l2[batch_id][start_value] = l2_choice

                model.Add(l1_choice <= x_var)
                model.Add(l1_choice <= left)
                model.Add(l1_choice >= x_var + left - 1)
                model.Add(l2_choice <= x_var)
                model.Add(l2_choice <= right)
                model.Add(l2_choice >= x_var + right - 1)
                constraint_counts["ROUTE_LINK"] += 6
                var_counts["route_choice"] += 2

                completion_time = start_value + int(roast_time["PSC"])
                if completion_time < SL:
                    rc_completion_terms["L1"][completion_time].append(l1_choice)
                    rc_completion_terms["L2"][completion_time].append(l2_choice)
        var_counts["z_route"] = len(z_l1) + len(z_l2)

    for roaster in roasters:
        tasks = [batch_id for batch_id in all_batches if roaster in sched_elig[batch_id]]
        node_of = {batch_id: index + 1 for index, batch_id in enumerate(tasks)}
        arcs: list[tuple[int, int, Any]] = []

        empty = model.NewBoolVar(f"empty_{roaster}")
        empty_arc[roaster] = empty
        arcs.append((0, 0, empty))
        var_counts["first_last_arc"] += 1

        for batch_id in tasks:
            first = model.NewBoolVar(f"first_{safe_id(batch_id)}_{roaster}")
            last = model.NewBoolVar(f"last_{safe_id(batch_id)}_{roaster}")
            first_arc[roaster][batch_id] = first
            last_arc[roaster][batch_id] = last
            arcs.append((0, node_of[batch_id], first))
            arcs.append((node_of[batch_id], 0, last))
            arcs.append((node_of[batch_id], node_of[batch_id], lit[batch_id][roaster].Not()))
            var_counts["first_last_arc"] += 2

        for left_batch in tasks:
            left_node = node_of[left_batch]
            left_dur = int(roast_time[batch_sku[left_batch]])
            for right_batch in tasks:
                if left_batch == right_batch:
                    continue
                right_node = node_of[right_batch]
                arc_var = model.NewBoolVar(
                    f"arc_{safe_id(left_batch)}_{safe_id(right_batch)}_{roaster}"
                )
                seq_arc[roaster][(left_batch, right_batch)] = arc_var
                arcs.append((left_node, right_node, arc_var))
                gap = ST if batch_sku[left_batch] != batch_sku[right_batch] else 0
                model.Add(start[right_batch] >= start[left_batch] + left_dur + gap).OnlyEnforceIf(arc_var)
                constraint_counts["C5"] += 1
                var_counts["ordering_arc"] += 1

        model.AddCircuit(arcs)
        constraint_counts["SEQ"] += 1

    for batch_id in all_batches:
        setup_before[batch_id] = {}
        setup_start_choice[batch_id] = {}
        for roaster in sched_elig[batch_id]:
            terms: list[Any] = []
            for predecessor in (candidate for candidate in all_batches if roaster in sched_elig[candidate]):
                if predecessor == batch_id:
                    continue
                if batch_sku[predecessor] != batch_sku[batch_id]:
                    terms.append(seq_arc[roaster][(predecessor, batch_id)])
            if (
                batch_id in mto_batches
                and d["roaster_initial_sku"][roaster] != batch_sku[batch_id]
            ):
                terms.append(first_arc[roaster][batch_id])

            if not terms:
                continue

            setup_var = model.NewBoolVar(f"setup_before_{safe_id(batch_id)}_{roaster}")
            model.Add(setup_var == sum(terms))
            setup_before[batch_id][roaster] = setup_var
            var_counts["setup_before"] += 1
            constraint_counts["SETUP_EXACT"] += 1

            setup_start_choice[batch_id][roaster] = {}
            for start_value, x_var in start_choice[batch_id][roaster].items():
                z_var = model.NewBoolVar(f"setupX_{safe_id(batch_id)}_{roaster}_{start_value}")
                setup_start_choice[batch_id][roaster][start_value] = z_var
                model.Add(z_var <= x_var)
                model.Add(z_var <= setup_var)
                model.Add(z_var >= x_var + setup_var - 1)
                constraint_counts["SETUP_EXACT"] += 3
                var_counts["setup_start_choice"] += 1
                for minute in range(max(0, start_value - ST), start_value):
                    if minute < SL:
                        setup_terms[roaster][minute].append(z_var)

    for batch_id in mto_batches:
        sku = batch_sku[batch_id]
        for roaster in sched_elig[batch_id]:
            if d["roaster_initial_sku"][roaster] != sku:
                model.Add(start[batch_id] >= ST).OnlyEnforceIf(lit[batch_id][roaster])
                constraint_counts["C6"] += 1

    for roaster in roasters:
        intervals = [
            roast_interval[batch_id][roaster]
            for batch_id in all_batches
            if roaster in roast_interval[batch_id]
        ]
        model.AddNoOverlap(intervals)
        constraint_counts["C4"] += 1

    for line_id in lines:
        pipeline_intervals = [
            consume_interval[batch_id][roaster]
            for batch_id in all_batches
            for roaster in sched_elig[batch_id]
            if d["roaster_pipeline"][roaster] == line_id
        ]
        for pair in feasible_gc_pairs:
            if pair[0] != line_id:
                continue
            pipeline_intervals.extend(slot["pipe_interval"] for slot in restock_slots[pair])
        model.AddNoOverlap(pipeline_intervals)
        constraint_counts["C8_C17"] += 1

    all_restock_intervals = [
        slot["interval"]
        for pair in feasible_gc_pairs
        for slot in restock_slots[pair]
    ]
    model.AddNoOverlap(all_restock_intervals)
    constraint_counts["C18"] += 1

    consumption_sets = {
        line_id: set(d["consumption_events"][line_id])
        for line_id in lines
    }
    events_per_line = {line_id: len(consumption_sets[line_id]) for line_id in lines}
    stockout: dict[tuple[str, int], cp_model.IntVar] = {}
    for line_id in lines:
        rc_lb = -events_per_line[line_id]
        for minute in range(SL):
            rc_level[line_id][minute] = model.NewIntVar(rc_lb, MRC, f"rc_{line_id}_{minute}")
            low[line_id][minute] = model.NewBoolVar(f"low_{line_id}_{minute}")
            full[line_id][minute] = model.NewBoolVar(f"full_{line_id}_{minute}")
            var_counts["rc_level"] += 1
            var_counts["low"] += 1
            var_counts["full"] += 1

            completion_expr = sum(rc_completion_terms[line_id][minute])
            consume_now = 1 if minute in consumption_sets[line_id] else 0
            if minute == 0:
                model.Add(
                    rc_level[line_id][minute]
                    == int(d["rc_init"][line_id]) + completion_expr - consume_now
                )
            else:
                model.Add(
                    rc_level[line_id][minute]
                    == rc_level[line_id][minute - 1] + completion_expr - consume_now
                )
            constraint_counts["C10_C11"] += 1

            model.Add(rc_level[line_id][minute] <= SS - 1).OnlyEnforceIf(low[line_id][minute])
            model.Add(rc_level[line_id][minute] >= SS).OnlyEnforceIf(low[line_id][minute].Not())
            model.Add(rc_level[line_id][minute] >= MRC).OnlyEnforceIf(full[line_id][minute])
            model.Add(rc_level[line_id][minute] <= MRC - 1).OnlyEnforceIf(full[line_id][minute].Not())
            constraint_counts["RC_STATE"] += 4

    for line_id in lines:
        for t in sorted(consumption_sets[line_id]):
            so_var = model.NewBoolVar(f"stockout_{line_id}_{t}")
            stockout[(line_id, t)] = so_var
            model.Add(rc_level[line_id][t] >= 0).OnlyEnforceIf(so_var.Not())
            model.Add(rc_level[line_id][t] <= -1).OnlyEnforceIf(so_var)
            var_counts["stockout"] += 1

    for pair in feasible_gc_pairs:
        line_id, sku = pair
        times: list[Any] = [0]
        level_changes: list[Any] = [int(d["gc_init"][pair])]
        actives: list[Any] = [1]

        for batch_id in all_batches:
            if batch_sku[batch_id] != sku:
                continue
            for roaster in sched_elig[batch_id]:
                if d["roaster_pipeline"][roaster] != line_id:
                    continue
                times.append(start[batch_id])
                level_changes.append(-1)
                actives.append(lit[batch_id][roaster])

        for slot in restock_slots[pair]:
            times.append(slot["start"] + RST_DUR)
            level_changes.append(RST_QTY)
            actives.append(slot["present"])

        model.add_reservoir_constraint_with_active(
            times,
            level_changes,
            actives,
            0,
            int(d["gc_capacity"][pair]),
        )
        constraint_counts["C16"] += 1

    for job_id in d["jobs"]:
        due = int(d["job_due"][job_id])
        for batch_id in mto_batches:
            if batch_id[0] != job_id:
                continue
            duration = int(roast_time[batch_sku[batch_id]])
            model.Add(tard[job_id] >= start[batch_id] + duration - due)
            constraint_counts["C12"] += 1

    for roaster in roasters:
        roaster_psc = sorted(
            [batch_id for batch_id in psc_pool if batch_id[0] == roaster],
            key=lambda batch_id: batch_id[1],
        )
        for left, right in zip(roaster_psc, roaster_psc[1:]):
            model.Add(lit[left][roaster] >= lit[right][roaster])
            constraint_counts["SYM"] += 1

    if d["allow_r3_flex"]:
        for minute in range(SL):
            both_var = model.NewBoolVar(f"full_both_{minute}")
            full_both[minute] = both_var
            model.Add(both_var <= full["L1"][minute])
            model.Add(both_var <= full["L2"][minute])
            model.Add(both_var >= full["L1"][minute] + full["L2"][minute] - 1)
            var_counts["full_both"] += 1
            constraint_counts["IDLE_OVER"] += 3

    for roaster in roasters:
        home_line = d["roaster_line"][roaster]
        out_line = d["roaster_can_output"][roaster][0]
        downtime = d["downtime_slots"].get(roaster, set())
        for minute in range(SL):
            running[roaster][minute] = model.NewBoolVar(f"running_{roaster}_{minute}")
            setup_active[roaster][minute] = model.NewBoolVar(f"setup_{roaster}_{minute}")
            idle[roaster][minute] = model.NewBoolVar(f"idle_{roaster}_{minute}")
            over[roaster][minute] = model.NewBoolVar(f"over_{roaster}_{minute}")
            var_counts["running"] += 1
            var_counts["setup_active"] += 1
            var_counts["idle"] += 1
            var_counts["over"] += 1

            model.Add(running[roaster][minute] == sum(running_terms[roaster][minute]))
            model.Add(setup_active[roaster][minute] == sum(setup_terms[roaster][minute]))
            constraint_counts["IDLE_OVER"] += 2

            if minute in downtime:
                model.Add(idle[roaster][minute] == 0)
                model.Add(over[roaster][minute] == 0)
                constraint_counts["IDLE_OVER"] += 2
                continue

            model.Add(idle[roaster][minute] <= low[home_line][minute])
            model.Add(idle[roaster][minute] <= 1 - running[roaster][minute])
            model.Add(idle[roaster][minute] >= low[home_line][minute] - running[roaster][minute])
            constraint_counts["IDLE_OVER"] += 3

            full_condition = (
                full_both[minute]
                if roaster == "R3" and d["allow_r3_flex"]
                else full[out_line][minute]
            )
            model.Add(over[roaster][minute] <= full_condition)
            model.Add(over[roaster][minute] <= 1 - running[roaster][minute])
            model.Add(over[roaster][minute] <= 1 - setup_active[roaster][minute])
            model.Add(
                over[roaster][minute]
                >= full_condition - running[roaster][minute] - setup_active[roaster][minute]
            )
            constraint_counts["IDLE_OVER"] += 4

    dispatch_hint = _try_dispatch_hint(d)
    if dispatch_hint is not None:
        schedule_by_batch = dispatch_hint["schedule_by_batch"]
        tard_hint = dispatch_hint["tardiness"]
        restock_hint = dispatch_hint["restocks_by_pair"]
        by_roaster = dispatch_hint["by_roaster"]

        hint_dropped = 0
        for batch_id in list(schedule_by_batch.keys()):
            hb = schedule_by_batch[batch_id]
            allowed = allowed_starts.get((batch_id, hb.roaster), [])
            if int(hb.start) not in allowed:
                hint_dropped += 1
                logger.warning(
                    "Dropping hint for %s: start=%d on %s collides with downtime "
                    "(forbidden by _allowed_starts).",
                    batch_id, int(hb.start), hb.roaster,
                )
                schedule_by_batch.pop(batch_id)
                for roaster in list(by_roaster):
                    by_roaster[roaster] = [b for b in by_roaster[roaster] if b.batch_id != batch_id]
        if hint_dropped:
            logger.warning("Dispatch hint: dropped %d batches with UPS-colliding starts.", hint_dropped)

        for batch_id in all_batches:
            hinted_batch = schedule_by_batch.get(batch_id)
            if hinted_batch is not None:
                model.add_hint(start[batch_id], int(hinted_batch.start))
            else:
                model.add_hint(start[batch_id], _earliest_start(d, batch_id))

            for roaster in sched_elig[batch_id]:
                is_assigned = int(hinted_batch is not None and hinted_batch.roaster == roaster)
                model.add_hint(lit[batch_id][roaster], is_assigned)
                for start_value, x_var in start_choice[batch_id][roaster].items():
                    is_here = int(
                        hinted_batch is not None
                        and hinted_batch.roaster == roaster
                        and int(hinted_batch.start) == start_value
                    )
                    model.add_hint(x_var, is_here)

            if batch_id in skipped:
                model.add_hint(skipped[batch_id], 0 if hinted_batch is not None else 1)

            if batch_id in z_l1:
                if hinted_batch is None:
                    model.add_hint(z_l1[batch_id], 0)
                    model.add_hint(z_l2[batch_id], 0)
                    for start_value in route_choice_l1[batch_id]:
                        model.add_hint(route_choice_l1[batch_id][start_value], 0)
                        model.add_hint(route_choice_l2[batch_id][start_value], 0)
                else:
                    is_l1 = int(hinted_batch.output_line == "L1")
                    is_l2 = int(hinted_batch.output_line == "L2")
                    model.add_hint(z_l1[batch_id], is_l1)
                    model.add_hint(z_l2[batch_id], is_l2)
                    for start_value in route_choice_l1[batch_id]:
                        active_here = int(int(hinted_batch.start) == start_value)
                        model.add_hint(route_choice_l1[batch_id][start_value], active_here * is_l1)
                        model.add_hint(route_choice_l2[batch_id][start_value], active_here * is_l2)

        for roaster in roasters:
            ordered = by_roaster.get(roaster, [])
            ordered_batches = [batch.batch_id for batch in ordered]
            ordered_pairs = set(zip(ordered_batches, ordered_batches[1:]))
            model.add_hint(empty_arc[roaster], 1 if not ordered_batches else 0)
            for batch_id in (candidate for candidate in all_batches if roaster in sched_elig[candidate]):
                model.add_hint(
                    first_arc[roaster][batch_id],
                    1 if ordered_batches and ordered_batches[0] == batch_id else 0,
                )
                model.add_hint(
                    last_arc[roaster][batch_id],
                    1 if ordered_batches and ordered_batches[-1] == batch_id else 0,
                )
            for (left_batch, right_batch), arc_var in seq_arc[roaster].items():
                model.add_hint(arc_var, 1 if (left_batch, right_batch) in ordered_pairs else 0)

        for batch_id in all_batches:
            hinted_batch = schedule_by_batch.get(batch_id)
            if hinted_batch is None:
                for roaster, setup_var in setup_before[batch_id].items():
                    model.add_hint(setup_var, 0)
                    for z_var in setup_start_choice.get(batch_id, {}).get(roaster, {}).values():
                        model.add_hint(z_var, 0)
                continue

            roaster = hinted_batch.roaster
            ordered = by_roaster.get(roaster, [])
            setup_now = 0
            for index, batch in enumerate(ordered):
                if batch.batch_id != batch_id:
                    continue
                if index == 0:
                    setup_now = int(
                        batch.is_mto and d["roaster_initial_sku"][roaster] != batch.sku
                    )
                else:
                    prev_batch = ordered[index - 1]
                    setup_now = int(prev_batch.sku != batch.sku)
                break
            for candidate_roaster, setup_var in setup_before[batch_id].items():
                hint_value = int(candidate_roaster == roaster and setup_now == 1)
                model.add_hint(setup_var, hint_value)
                for start_value, z_var in setup_start_choice.get(batch_id, {}).get(candidate_roaster, {}).items():
                    active_here = int(
                        candidate_roaster == roaster
                        and setup_now == 1
                        and int(hinted_batch.start) == start_value
                    )
                    model.add_hint(z_var, active_here)

        for job_id, tard_value in tard_hint.items():
            model.add_hint(tard[job_id], int(tard_value))

        for pair, slots in restock_slots.items():
            hinted_restocks = restock_hint.get(pair, [])
            for idx, slot in enumerate(slots):
                if idx < len(hinted_restocks):
                    model.add_hint(slot["present"], 1)
                    model.add_hint(slot["start"], int(hinted_restocks[idx].start))
                else:
                    model.add_hint(slot["present"], 0)
                    model.add_hint(slot["start"], 0)

        running_hint = {roaster: [0] * SL for roaster in roasters}
        setup_hint = {roaster: [0] * SL for roaster in roasters}
        rc_hint = {line_id: [0] * SL for line_id in lines}

        for roaster, ordered in by_roaster.items():
            prev_sku = d["roaster_initial_sku"][roaster]
            for index, batch in enumerate(ordered):
                for minute in range(int(batch.start), min(SL, int(batch.end))):
                    running_hint[roaster][minute] = 1
                needs_setup = False
                if index == 0:
                    needs_setup = bool(batch.is_mto and batch.sku != prev_sku)
                else:
                    needs_setup = bool(batch.sku != prev_sku)
                if needs_setup:
                    for minute in range(max(0, int(batch.start) - ST), int(batch.start)):
                        setup_hint[roaster][minute] = 1
                prev_sku = batch.sku

        for line_id in lines:
            level = int(d["rc_init"][line_id])
            completions: dict[int, int] = {}
            for batch in schedule_by_batch.values():
                if batch.sku == "PSC" and batch.output_line == line_id and int(batch.end) < SL:
                    completions[int(batch.end)] = completions.get(int(batch.end), 0) + 1
            for minute in range(SL):
                if minute in completions:
                    level += completions[minute]
                if minute in consumption_sets[line_id]:
                    level -= 1
                rc_hint[line_id][minute] = level

        for line_id in lines:
            for minute in range(SL):
                level = rc_hint[line_id][minute]
                model.add_hint(rc_level[line_id][minute], level)
                model.add_hint(low[line_id][minute], 1 if level < SS else 0)
                model.add_hint(full[line_id][minute], 1 if level >= MRC else 0)
            for t in consumption_sets[line_id]:
                model.add_hint(stockout[(line_id, t)], 1 if rc_hint[line_id][t] < 0 else 0)

        if d["allow_r3_flex"]:
            for minute in range(SL):
                model.add_hint(
                    full_both[minute],
                    1 if rc_hint["L1"][minute] >= MRC and rc_hint["L2"][minute] >= MRC else 0,
                )

        for roaster in roasters:
            downtime = d["downtime_slots"].get(roaster, set())
            home_line = d["roaster_line"][roaster]
            out_line = d["roaster_can_output"][roaster][0]
            for minute in range(SL):
                run_now = running_hint[roaster][minute]
                setup_now = setup_hint[roaster][minute]
                if minute in downtime:
                    idle_now = 0
                    over_now = 0
                else:
                    idle_now = int(run_now == 0 and rc_hint[home_line][minute] < SS)
                    if roaster == "R3" and d["allow_r3_flex"]:
                        overflow_cond = rc_hint["L1"][minute] >= MRC and rc_hint["L2"][minute] >= MRC
                    else:
                        overflow_cond = rc_hint[out_line][minute] >= MRC
                    over_now = int(run_now == 0 and setup_now == 0 and overflow_cond)

                model.add_hint(running[roaster][minute], run_now)
                model.add_hint(setup_active[roaster][minute], setup_now)
                model.add_hint(idle[roaster][minute], idle_now)
                model.add_hint(over[roaster][minute], over_now)

    revenue_psc = _as_int_coefficient(d["sku_revenue"]["PSC"], "sku_revenue.PSC")
    mto_revenue = sum(
        _as_int_coefficient(
            d["sku_revenue"][batch_sku[batch_id]], f"sku_revenue.{batch_sku[batch_id]}"
        )
        * sum(lit[batch_id][r] for r in sched_elig[batch_id])
        for batch_id in mto_batches
    )
    cost_tardiness = _as_int_coefficient(d["cost_tardiness"], "cost_tardiness")
    cost_setup = _as_int_coefficient(d["cost_setup"], "cost_setup")
    cost_idle = _as_int_coefficient(d["cost_idle"], "cost_idle")
    cost_overflow = _as_int_coefficient(d["cost_overflow"], "cost_overflow")
    cost_stockout = _as_int_coefficient(d["cost_stockout"], "cost_stockout")
    cost_skip_mto = _as_int_coefficient(d["cost_skip_mto"], "cost_skip_mto")

    psc_revenue_expr = sum(
        revenue_psc * lit[batch_id][sched_elig[batch_id][0]]
        for batch_id in psc_pool
    )
    tard_penalty = cost_tardiness * sum(tard.values())
    exact_setup_events = sum(
        setup_var
        for batch_map in setup_before.values()
        for setup_var in batch_map.values()
    )
    setup_penalty = cost_setup * exact_setup_events
    idle_penalty = cost_idle * sum(
        idle[roaster][minute]
        for roaster in roasters
        for minute in range(SL)
    )
    overflow_penalty = cost_overflow * sum(
        over[roaster][minute]
        for roaster in roasters
        for minute in range(SL)
    )
    stockout_penalty = cost_stockout * sum(stockout.values())
    skip_penalty = cost_skip_mto * sum(skipped.values())
    model.Maximize(
        psc_revenue_expr
        + mto_revenue
        - tard_penalty
        - skip_penalty
        - stockout_penalty
        - setup_penalty
        - idle_penalty
        - overflow_penalty
    )

    elapsed = time.perf_counter() - t_build
    model._build_meta = {
        "var_counts": var_counts,
        "constraint_counts": constraint_counts,
        "build_seconds": elapsed,
        "formulation": "interval-based v4 (soft engine-consistent objective, RC timeline, GC reservoir)",
        "notes": [
            "UPS is handled via downtime pre-merge in runner.py (perfect-information mode).",
            "SKU-specific roast durations are enforced.",
            "Setup cost is counted from realized exact roaster transitions.",
            "Objective includes exact safety-idle and overflow-idle minutes.",
            "RC inventory is tracked minute-by-minute; stockouts allowed at c_stock per consume event.",
            "MTO activation is soft; unscheduled MTO penalised at c_skip_mto per batch.",
            "GC inventory remains hard-feasible via active reservoir constraints.",
            "Current C6 first-setup semantics are preserved unchanged.",
        ],
    }
    logger.info("Model v4 built in %.2fs", elapsed)
    logger.info("Variable counts: %s", var_counts)
    logger.info("Constraint counts: %s", constraint_counts)

    vars_dict = {
        "start": start,
        "end": end_expr,
        "lit": lit,
        "roast_interval": roast_interval,
        "consume_interval": consume_interval,
        "start_choice": start_choice,
        "tard": tard,
        "z_l1": z_l1,
        "z_l2": z_l2,
        "restock_slots": restock_slots,
        "setup_before": setup_before,
        "running": running,
        "setup_active": setup_active,
        "rc_level": rc_level,
        "low": low,
        "full": full,
        "idle": idle,
        "over": over,
        "skipped": skipped,
        "stockout": stockout,
    }
    return model, vars_dict
