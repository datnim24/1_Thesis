"""Reactive CP-SAT model builder.

Adapted from CP_SAT_v2/model.py ``build()`` with the following changes:

1. HORIZON uses [0, shift_length] where shift_length = 480 - t0.
2. FIXED INTERVALS for in-progress batches on healthy roasters.
3. DOWNED ROASTER BLOCKING via downtime_slots (already merged in snapshot.py).
4. SOFT STOCKOUT (C10'): RC can go negative; penalty per sub-zero consumption.
5. OBJECTIVE adds stockout penalty term.

Constraint labels use C1'–C19' to distinguish from the deterministic model.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from ortools.sat.python import cp_model

logger = logging.getLogger("reactive_cpsat.model")


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
        for s in range(max(0, down_slot - duration + 1), min(max_start, down_slot) + 1):
            forbidden.add(s)
    return forbidden


def _as_int_coefficient(value: Any, label: str) -> int:
    numeric = float(value)
    rounded = round(numeric)
    if abs(numeric - rounded) > 1e-9:
        raise ValueError(f"CP-SAT coefficient '{label}' must be integral, got {value!r}")
    return int(rounded)


def _earliest_start(d: dict[str, Any], batch_id: Any) -> int:
    if not d["batch_is_mto"][batch_id]:
        return 0
    sku = d["batch_sku"][batch_id]
    release = int(d["job_release"].get(batch_id[0], 0))
    eligible = d["sched_eligible_roasters"][batch_id]
    all_need_setup = all(d["roaster_initial_sku"][r] != sku for r in eligible)
    return max(release, int(d["setup_time"])) if all_need_setup else release


def _allowed_starts(d: dict[str, Any], batch_id: Any, roaster: str) -> list[int]:
    sku = d["batch_sku"][batch_id]
    earliest = _earliest_start(d, batch_id)
    latest = int(d["MS_by_sku"][sku])
    if latest < earliest:
        return []
    downtime = d["downtime_slots"].get(roaster, set())
    if not downtime:
        return list(range(earliest, latest + 1))
    forbidden = _forbidden_starts(downtime, int(d["roast_time_by_sku"][sku]), latest)
    return [s for s in range(earliest, latest + 1) if s not in forbidden]


def _max_restocks(pair: tuple[str, str], d: dict[str, Any]) -> int:
    line_id, sku = pair
    init = int(d["gc_init"][pair])
    qty = int(d["restock_qty"])
    max_possible = sum(
        1 for bid in d["all_batches"]
        if d["batch_sku"][bid] == sku
        and any(d["roaster_pipeline"][r] == line_id for r in d["sched_eligible_roasters"][bid])
    )
    needed = max(0, math.ceil((max_possible - init) / max(qty, 1)))
    hard_cap = max(1, int(d["shift_length"]) // max(1, int(d["restock_duration"])))
    return max(1, min(needed + 2, hard_cap))


def _apply_greedy_hint(
    model, d, all_batches, mto_batches, psc_pool, sched_elig, batch_sku,
    start, lit, start_choice, empty_arc, first_arc, last_arc, seq_arc,
    setup_before, setup_start_choice, tard, restock_slots,
    running, setup_active, rc_level, low, full, full_both,
    idle, over, stockout_at, fixed_intervals,
) -> None:
    """Apply a greedy warm-start hint: schedule PSC batches back-to-back per roaster.

    This gives the solver a trivially feasible starting point to improve from.
    Without it, the large number of start_choice variables makes finding even
    one feasible solution slow.
    """
    SL = int(d["shift_length"])
    ST = int(d["setup_time"])
    SS = int(d["safety_stock"])
    MRC = int(d["max_rc"])
    roasters = d["roasters"]
    lines = d["lines"]
    roast_time = d["roast_time_by_sku"]

    # Build a greedy schedule: assign PSC batches sequentially per roaster
    roaster_next_start: dict[str, int] = {}
    for r in roasters:
        # Account for fixed intervals — start after the longest fixed interval ends
        fi_end = 0
        for fi in fixed_intervals:
            if fi["roaster"] == r:
                fi_end = max(fi_end, int(fi["end"]))
        # Account for UPS downtime
        downtime = d["downtime_slots"].get(r, set())
        earliest = fi_end
        while earliest in downtime and earliest < SL:
            earliest += 1
        roaster_next_start[r] = earliest

    hint_assigned: dict[Any, tuple[str, int]] = {}  # batch_id -> (roaster, start)
    hint_by_roaster: dict[str, list[tuple[Any, int]]] = {r: [] for r in roasters}

    def _snap_to_allowed(bid_: Any, r_: str, target_s: int) -> int | None:
        """Find the smallest allowed start >= target_s for batch on roaster."""
        allowed = sorted(start_choice[bid_].get(r_, {}).keys())
        for a in allowed:
            if a >= target_s:
                return a
        return None

    # Assign MTO batches first (they're mandatory)
    for bid in mto_batches:
        sku = batch_sku[bid]
        dur = int(roast_time[sku])
        best_r = None
        best_s = SL
        for r in sched_elig[bid]:
            s = roaster_next_start[r]
            # Account for setup
            if d["roaster_initial_sku"][r] != sku:
                if not hint_by_roaster[r]:
                    s = max(s, ST)
                elif hint_by_roaster[r][-1]:
                    prev_sku = batch_sku[hint_by_roaster[r][-1][0]]
                    if prev_sku != sku:
                        s += ST
            snapped = _snap_to_allowed(bid, r, s)
            if snapped is not None and snapped + dur <= SL and snapped < best_s:
                best_r, best_s = r, snapped
        if best_r is not None:
            hint_assigned[bid] = (best_r, best_s)
            hint_by_roaster[best_r].append((bid, best_s))
            roaster_next_start[best_r] = best_s + dur

    # Track GC budget per (line, sku) pair — avoid exceeding gc_init without restocks
    DC = int(d["consume_time"])
    RST_DUR = int(d["restock_duration"])
    RST_QTY = int(d["restock_qty"])
    gc_budget: dict[tuple[str, str], int] = {}
    for pair in d["feasible_gc_pairs"]:
        gc_budget[pair] = int(d["gc_init"].get(pair, 0))

    # Track pipeline availability (consume intervals must not overlap per line)
    pipeline_free: dict[str, int] = {line: 0 for line in lines}
    # Account for fixed consume intervals on pipeline
    for fi in fixed_intervals:
        pl = fi.get("pipeline_line", "")
        pr = int(fi.get("pipeline_remaining", 0))
        if pl and pr > 0:
            pipeline_free[pl] = max(pipeline_free[pl], pr)

    # Track restock scheduling state
    hint_restocks: dict[tuple[str, str], list[int]] = {}
    global_restock_end = 0

    def _try_restock(pair_: tuple[str, str]) -> int | None:
        """Schedule a restock for pair_ at the earliest feasible time."""
        nonlocal global_restock_end
        line_ = pair_[0]
        s = max(global_restock_end, pipeline_free.get(line_, 0))
        if s + RST_DUR <= SL:
            hint_restocks.setdefault(pair_, []).append(s)
            gc_budget[pair_] += RST_QTY
            global_restock_end = s + RST_DUR
            pipeline_free[line_] = max(pipeline_free[line_], s + RST_DUR)
            return s
        return None

    # Assign PSC batches, respecting GC limits AND pipeline NoOverlap
    for bid in psc_pool:
        r = bid[0]
        dur = int(roast_time["PSC"])
        pipe_line = d["roaster_pipeline"].get(r, "")

        # Roaster availability AND pipeline availability
        s = max(roaster_next_start[r], pipeline_free.get(pipe_line, 0))
        snapped = _snap_to_allowed(bid, r, s)
        if snapped is None or snapped + dur > SL:
            continue

        # Check GC availability
        gc_pair = (pipe_line, "PSC") if pipe_line else None
        if gc_pair and gc_pair in gc_budget:
            while gc_budget[gc_pair] <= 0:
                if _try_restock(gc_pair) is None:
                    break
            if gc_budget[gc_pair] <= 0:
                continue  # Can't restock, skip
            gc_budget[gc_pair] -= 1

        hint_assigned[bid] = (r, snapped)
        hint_by_roaster[r].append((bid, snapped))
        roaster_next_start[r] = snapped + dur
        if pipe_line:
            pipeline_free[pipe_line] = snapped + DC  # Pipeline busy until consume ends

    # Apply hints to all variables
    for bid in all_batches:
        if bid in hint_assigned:
            h_r, h_s = hint_assigned[bid]
            model.add_hint(start[bid], h_s)
            for r in sched_elig[bid]:
                model.add_hint(lit[bid][r], 1 if r == h_r else 0)
                for sv, x_var in start_choice[bid].get(r, {}).items():
                    model.add_hint(x_var, 1 if (r == h_r and sv == h_s) else 0)
        else:
            earliest = _earliest_start(d, bid)
            model.add_hint(start[bid], earliest)
            for r in sched_elig[bid]:
                model.add_hint(lit[bid][r], 0)
                for sv, x_var in start_choice[bid].get(r, {}).items():
                    model.add_hint(x_var, 0)

    # Circuit hints
    for r in roasters:
        ordered = [(bid, s) for bid, s in hint_by_roaster[r]]
        ordered.sort(key=lambda x: x[1])
        ordered_bids = [bid for bid, _ in ordered]
        ordered_pairs = set(zip(ordered_bids, ordered_bids[1:]))

        model.add_hint(empty_arc[r], 1 if not ordered_bids else 0)
        for bid in (c for c in all_batches if r in sched_elig[c]):
            if bid in first_arc[r]:
                model.add_hint(first_arc[r][bid],
                               1 if ordered_bids and ordered_bids[0] == bid else 0)
            if bid in last_arc[r]:
                model.add_hint(last_arc[r][bid],
                               1 if ordered_bids and ordered_bids[-1] == bid else 0)
        for (lb, rb), arc_var in seq_arc[r].items():
            model.add_hint(arc_var, 1 if (lb, rb) in ordered_pairs else 0)

    # Setup hints
    for bid in all_batches:
        for r, setup_var in setup_before.get(bid, {}).items():
            model.add_hint(setup_var, 0)  # PSC-only schedule has no setups
            for sv, z_var in setup_start_choice.get(bid, {}).get(r, {}).items():
                model.add_hint(z_var, 0)

    # Tardiness hints
    for job_id, tard_var in tard.items():
        model.add_hint(tard_var, 0)

    # Restock hints
    for pair, slots in restock_slots.items():
        hinted_rst = sorted(hint_restocks.get(pair, []))
        for idx, slot in enumerate(slots):
            if idx < len(hinted_rst):
                model.add_hint(slot["present"], 1)
                model.add_hint(slot["start"], hinted_rst[idx])
            else:
                model.add_hint(slot["present"], 0)
                model.add_hint(slot["start"], 0)

    # RC/idle/overflow hints from greedy schedule
    consumption_sets = {line: set(d["consumption_events"][line]) for line in lines}
    completions: dict[str, dict[int, int]] = {line: {} for line in lines}
    for bid, (h_r, h_s) in hint_assigned.items():
        sku = batch_sku[bid]
        if sku == "PSC":
            out_line = d["roaster_can_output"].get(h_r, [None])[0]
            if out_line:
                ct = h_s + int(roast_time[sku])
                if ct < SL:
                    completions[out_line][ct] = completions[out_line].get(ct, 0) + 1
    # Add fixed interval completions
    for fi in fixed_intervals:
        if fi["sku"] == "PSC" and fi.get("output_line"):
            ct = int(fi["end"])
            if 0 <= ct < SL:
                line = fi["output_line"]
                completions[line][ct] = completions[line].get(ct, 0) + 1

    rc_hint: dict[str, list[int]] = {}
    for line in lines:
        level = int(d["rc_init"][line])
        rc_hint[line] = []
        for minute in range(SL):
            level += completions[line].get(minute, 0)
            if minute in consumption_sets[line]:
                level -= 1
            rc_hint[line].append(level)

    running_hint = {r: [0] * SL for r in roasters}
    for bid, (h_r, h_s) in hint_assigned.items():
        dur = int(roast_time[batch_sku[bid]])
        for m in range(h_s, min(SL, h_s + dur)):
            running_hint[h_r][m] = 1
    for fi in fixed_intervals:
        for m in range(int(fi["start"]), min(SL, int(fi["end"]))):
            running_hint[fi["roaster"]][m] = 1

    for line in lines:
        for minute in range(SL):
            level = rc_hint[line][minute]
            model.add_hint(rc_level[line][minute], max(rc_hint[line][minute], -SL))
            model.add_hint(low[line][minute], 1 if level < SS else 0)
            model.add_hint(full[line][minute], 1 if level >= MRC else 0)

    if d["allow_r3_flex"]:
        for minute in range(SL):
            if minute in full_both:
                model.add_hint(
                    full_both[minute],
                    1 if rc_hint["L1"][minute] >= MRC and rc_hint["L2"][minute] >= MRC else 0,
                )

    for r in roasters:
        home_line = d["roaster_line"][r]
        out_line = d["roaster_can_output"][r][0]
        downtime = d["downtime_slots"].get(r, set())
        for minute in range(SL):
            run_now = running_hint[r][minute]
            model.add_hint(running[r][minute], run_now)
            model.add_hint(setup_active[r][minute], 0)
            if minute in downtime:
                model.add_hint(idle[r][minute], 0)
                model.add_hint(over[r][minute], 0)
            else:
                low_now = rc_hint[home_line][minute] < SS
                model.add_hint(idle[r][minute], 1 if (run_now == 0 and low_now) else 0)
                full_now = rc_hint[out_line][minute] >= MRC
                if r == "R3" and d["allow_r3_flex"]:
                    full_now = rc_hint["L1"][minute] >= MRC and rc_hint["L2"][minute] >= MRC
                model.add_hint(over[r][minute], 1 if (run_now == 0 and full_now) else 0)

    # Stockout hints
    for line in lines:
        for minute, neg_var in stockout_at.get(line, {}).items():
            model.add_hint(neg_var, 1 if rc_hint[line][minute] < 0 else 0)

    logger.info(
        "Greedy hint applied: %d batches assigned (%d MTO, %d PSC)",
        len(hint_assigned),
        sum(1 for b in hint_assigned if b in mto_batches),
        sum(1 for b in hint_assigned if b in psc_pool),
    )


def build_reactive(d: dict[str, Any]) -> tuple[cp_model.CpModel, dict[str, Any]]:
    """Build a reactive CP-SAT model over the remaining horizon.

    Returns (model, vars_dict) — same interface as CP_SAT_v2/model.py build().
    """
    t_build = time.perf_counter()

    SL = int(d["shift_length"])
    min_roast = min(int(v) for v in d["roast_time_by_sku"].values())

    # ── Edge case: horizon too short ─────────────────────────────────────
    if SL < min_roast:
        logger.warning("Remaining horizon %d < shortest roast %d — trivial empty schedule", SL, min_roast)
        model = cp_model.CpModel()
        model.Maximize(0)
        model._build_meta = {
            "trivial": True,
            "reason": f"horizon {SL} < min_roast {min_roast}",
        }
        return model, {"trivial": True}

    model = cp_model.CpModel()

    ST = int(d["setup_time"])
    DC = int(d["consume_time"])
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
    fixed_intervals = d.get("fixed_intervals", [])

    stockout_soft = d.get("stockout_soft", False)
    stockout_penalty_cost = int(d.get("stockout_penalty", 1500))

    logger.info(
        "Building reactive model: horizon=%d, batches=%d (MTO=%d, PSC=%d), "
        "fixed=%d, soft_stockout=%s",
        SL, len(all_batches), len(mto_batches), len(psc_pool),
        len(fixed_intervals), stockout_soft,
    )

    # ── Variable containers ──────────────────────────────────────────────
    start: dict[Any, cp_model.IntVar] = {}
    end_expr: dict[Any, Any] = {}
    lit: dict[Any, dict[str, cp_model.IntVar]] = {}
    roast_interval: dict[Any, dict[str, cp_model.IntervalVar]] = {}
    consume_interval: dict[Any, dict[str, cp_model.IntervalVar]] = {}
    start_choice: dict[Any, dict[str, dict[int, cp_model.IntVar]]] = {}
    allowed_starts_cache: dict[tuple[Any, str], list[int]] = {}

    tard: dict[str, cp_model.IntVar] = {}
    z_l1: dict[Any, cp_model.IntVar] = {}
    z_l2: dict[Any, cp_model.IntVar] = {}
    route_choice_l1: dict[Any, dict[int, cp_model.IntVar]] = {}
    route_choice_l2: dict[Any, dict[int, cp_model.IntVar]] = {}

    restock_slots: dict[tuple[str, str], list[dict[str, Any]]] = {}

    first_arc: dict[str, dict[Any, cp_model.IntVar]] = {r: {} for r in roasters}
    last_arc: dict[str, dict[Any, cp_model.IntVar]] = {r: {} for r in roasters}
    seq_arc: dict[str, dict[tuple[Any, Any], cp_model.IntVar]] = {r: {} for r in roasters}
    empty_arc: dict[str, cp_model.IntVar] = {}
    setup_before: dict[Any, dict[str, cp_model.IntVar]] = {}
    setup_start_choice: dict[Any, dict[str, dict[int, cp_model.IntVar]]] = {}

    running: dict[str, dict[int, cp_model.IntVar]] = {r: {} for r in roasters}
    setup_active: dict[str, dict[int, cp_model.IntVar]] = {r: {} for r in roasters}

    # C10': soft stockout — RC can go negative
    rc_lower = -SL if stockout_soft else 0
    rc_level: dict[str, dict[int, cp_model.IntVar]] = {line: {} for line in lines}
    low: dict[str, dict[int, cp_model.IntVar]] = {line: {} for line in lines}
    full: dict[str, dict[int, cp_model.IntVar]] = {line: {} for line in lines}
    full_both: dict[int, cp_model.IntVar] = {}
    idle: dict[str, dict[int, cp_model.IntVar]] = {r: {} for r in roasters}
    over: dict[str, dict[int, cp_model.IntVar]] = {r: {} for r in roasters}

    # Stockout tracking variables (C10')
    stockout_at: dict[str, dict[int, cp_model.IntVar]] = {line: {} for line in lines}

    running_terms: dict[str, list[list]] = {r: [[] for _ in range(SL)] for r in roasters}
    setup_terms: dict[str, list[list]] = {r: [[] for _ in range(SL)] for r in roasters}
    rc_completion_terms: dict[str, list[list]] = {line: [[] for _ in range(SL)] for line in lines}

    # ── Fixed intervals: add RC completion terms for in-progress PSC ─────
    fixed_roast_intervals: dict[str, list[cp_model.IntervalVar]] = {r: [] for r in roasters}
    fixed_consume_intervals: dict[str, list[cp_model.IntervalVar]] = {line: [] for line in lines}

    for fi in fixed_intervals:
        r_id = fi["roaster"]
        sku = fi["sku"]
        fi_start = int(fi["start"])
        fi_end = int(fi["end"])
        fi_dur = fi_end - fi_start
        if fi_dur <= 0 or fi_end > SL:
            continue

        label = f"fixed_{r_id}_{sku}_{fi_start}"

        # C4': Fixed roast interval (non-optional, forced)
        fixed_roast = model.NewFixedSizeIntervalVar(fi_start, fi_dur, f"froast_{label}")
        fixed_roast_intervals[r_id].append(fixed_roast)

        # Fixed running contribution is computed inline in IDLE_OVER section below

        # C8': Fixed consume interval on pipeline
        pipe_line = fi.get("pipeline_line", "")
        pipe_remaining = int(fi.get("pipeline_remaining", 0))
        if pipe_line and pipe_remaining > 0 and pipe_line in lines:
            pipe_end = min(pipe_remaining, SL)
            if pipe_end > 0:
                fixed_pipe = model.NewFixedSizeIntervalVar(0, pipe_end, f"fpipe_{label}")
                fixed_consume_intervals[pipe_line].append(fixed_pipe)

        # RC completion from fixed PSC batches
        if sku == "PSC" and fi.get("output_line"):
            completion_time = fi_end
            if 0 <= completion_time < SL:
                rc_completion_terms[fi["output_line"]][completion_time].append(1)

    # ── Decision variables for schedulable batches ───────────────────────
    for batch_id in all_batches:
        sku = batch_sku[batch_id]
        duration = int(roast_time[sku])
        earliest = _earliest_start(d, batch_id)
        ms = int(d["MS_by_sku"][sku])
        if ms < earliest:
            ms = earliest  # will be infeasible; handled by solver
        start_var = model.NewIntVar(earliest, max(earliest, ms), f"start_{safe_id(batch_id)}")
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
                start_var, duration, present, f"roast_{safe_id(batch_id)}_{roaster}",
            )
            consume_interval[batch_id][roaster] = model.new_optional_fixed_size_interval_var(
                start_var, DC, present, f"consume_{safe_id(batch_id)}_{roaster}",
            )

            allowed = _allowed_starts(d, batch_id, roaster)
            allowed_starts_cache[(batch_id, roaster)] = allowed
            start_choice[batch_id][roaster] = {}
            for sv in allowed:
                x_var = model.NewBoolVar(f"x_{safe_id(batch_id)}_{roaster}_{sv}")
                start_choice[batch_id][roaster][sv] = x_var
                for minute in range(sv, min(SL, sv + duration)):
                    running_terms[roaster][minute].append(x_var)

                if batch_sku[batch_id] == "PSC":
                    ct = sv + duration
                    if ct < SL:
                        if roaster == "R3" and d["allow_r3_flex"]:
                            pass  # handled by route_choice below
                        else:
                            out_line = d["roaster_can_output"][roaster][0]
                            rc_completion_terms[out_line][ct].append(x_var)

    # ── C1'/C2': MTO assignment ──────────────────────────────────────────
    # Hard-assignment by default; when d["mto_soft"] is true, each batch may
    # be left unassigned at the price of c_skip_mto (paid in the objective).
    mto_soft = bool(d.get("mto_soft", False))
    mto_assigned: dict[Any, Any] = {}
    for batch_id in mto_batches:
        eligible = [r for r in sched_elig[batch_id]
                    if len(start_choice[batch_id].get(r, {})) > 0]
        if eligible:
            if mto_soft:
                a_var = model.NewBoolVar(f"assigned_{safe_id(batch_id)}")
                model.Add(sum(lit[batch_id][r] for r in sched_elig[batch_id]) == a_var)
                mto_assigned[batch_id] = a_var
            else:
                model.Add(sum(lit[batch_id][r] for r in sched_elig[batch_id]) == 1)
        else:
            # No feasible start for any roaster — force unassigned.
            # In soft mode this simply triggers the skip penalty.
            logger.warning("MTO batch %s has no feasible starts on any roaster", batch_id)
            for r in sched_elig[batch_id]:
                model.Add(lit[batch_id][r] == 0)

    # ── START_LINK: start-time channeling ────────────────────────────────
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
            start_expr_terms.extend(
                sv * choice for sv, choice in start_choice[batch_id][roaster].items()
            )
        model.Add(start[batch_id] == sum(start_expr_terms) + earliest * (1 - active_expr))

    # ── Tardiness variables ──────────────────────────────────────────────
    for job_id in d["jobs"]:
        tard[job_id] = model.NewIntVar(0, SL, f"tard_{job_id}")

    # ── Restock slots ────────────────────────────────────────────────────
    for pair in feasible_gc_pairs:
        pair_label = f"{pair[0]}_{pair[1]}"
        pool_size = _max_restocks(pair, d)
        restock_slots[pair] = []
        for idx in range(pool_size):
            present = model.NewBoolVar(f"rstP_{pair_label}_{idx}")
            start_var = model.NewIntVar(0, max(0, SL - RST_DUR), f"rstS_{pair_label}_{idx}")
            interval = model.new_optional_fixed_size_interval_var(
                start_var, RST_DUR, present, f"rstI_{pair_label}_{idx}",
            )
            pipe_interval = model.new_optional_fixed_size_interval_var(
                start_var, RST_DUR, present, f"rstPipe_{pair_label}_{idx}",
            )
            restock_slots[pair].append({
                "start": start_var,
                "present": present,
                "interval": interval,
                "pipe_interval": pipe_interval,
            })
        # RST_SYM: symmetry breaking
        for left, right in zip(restock_slots[pair], restock_slots[pair][1:]):
            model.Add(left["present"] >= right["present"])
            model.Add(left["start"] <= right["start"]).OnlyEnforceIf(
                [left["present"], right["present"]]
            )

    # ── C13': R3 flexible routing ────────────────────────────────────────
    if d["allow_r3_flex"]:
        for batch_id in psc_pool:
            if sched_elig[batch_id] != ["R3"]:
                continue
            left = model.NewBoolVar(f"z_L1_{safe_id(batch_id)}")
            right = model.NewBoolVar(f"z_L2_{safe_id(batch_id)}")
            z_l1[batch_id] = left
            z_l2[batch_id] = right
            model.Add(left + right == lit[batch_id]["R3"])

            route_choice_l1[batch_id] = {}
            route_choice_l2[batch_id] = {}
            for sv, x_var in start_choice[batch_id]["R3"].items():
                l1c = model.NewBoolVar(f"routeL1_{safe_id(batch_id)}_{sv}")
                l2c = model.NewBoolVar(f"routeL2_{safe_id(batch_id)}_{sv}")
                route_choice_l1[batch_id][sv] = l1c
                route_choice_l2[batch_id][sv] = l2c
                model.Add(l1c <= x_var)
                model.Add(l1c <= left)
                model.Add(l1c >= x_var + left - 1)
                model.Add(l2c <= x_var)
                model.Add(l2c <= right)
                model.Add(l2c >= x_var + right - 1)

                ct = sv + int(roast_time["PSC"])
                if ct < SL:
                    rc_completion_terms["L1"][ct].append(l1c)
                    rc_completion_terms["L2"][ct].append(l2c)

    # ── SEQ: Circuit constraint (sequencing) per roaster ─────────────────
    for roaster in roasters:
        tasks = [bid for bid in all_batches if roaster in sched_elig[bid]]
        node_of = {bid: i + 1 for i, bid in enumerate(tasks)}
        arcs: list[tuple[int, int, Any]] = []

        empty = model.NewBoolVar(f"empty_{roaster}")
        empty_arc[roaster] = empty
        arcs.append((0, 0, empty))

        for bid in tasks:
            first = model.NewBoolVar(f"first_{safe_id(bid)}_{roaster}")
            last = model.NewBoolVar(f"last_{safe_id(bid)}_{roaster}")
            first_arc[roaster][bid] = first
            last_arc[roaster][bid] = last
            arcs.append((0, node_of[bid], first))
            arcs.append((node_of[bid], 0, last))
            arcs.append((node_of[bid], node_of[bid], lit[bid][roaster].Not()))

        for lb in tasks:
            ln = node_of[lb]
            ld = int(roast_time[batch_sku[lb]])
            for rb in tasks:
                if lb == rb:
                    continue
                rn = node_of[rb]
                arc_var = model.NewBoolVar(f"arc_{safe_id(lb)}_{safe_id(rb)}_{roaster}")
                seq_arc[roaster][(lb, rb)] = arc_var
                arcs.append((ln, rn, arc_var))
                # C5': Sequencing precedence with setup gap
                gap = ST if batch_sku[lb] != batch_sku[rb] else 0
                model.Add(start[rb] >= start[lb] + ld + gap).OnlyEnforceIf(arc_var)

        model.AddCircuit(arcs)

    # ── SETUP_EXACT: setup event detection ───────────────────────────────
    for batch_id in all_batches:
        setup_before[batch_id] = {}
        setup_start_choice[batch_id] = {}
        for roaster in sched_elig[batch_id]:
            terms: list[Any] = []
            for pred in (c for c in all_batches if roaster in sched_elig[c]):
                if pred == batch_id:
                    continue
                if batch_sku[pred] != batch_sku[batch_id]:
                    terms.append(seq_arc[roaster][(pred, batch_id)])
            if (
                batch_id in mto_batches
                and d["roaster_initial_sku"][roaster] != batch_sku[batch_id]
            ):
                terms.append(first_arc[roaster][batch_id])
            if not terms:
                continue

            setup_var = model.NewBoolVar(f"setup_{safe_id(batch_id)}_{roaster}")
            model.Add(setup_var == sum(terms))
            setup_before[batch_id][roaster] = setup_var

            setup_start_choice[batch_id][roaster] = {}
            for sv, x_var in start_choice[batch_id][roaster].items():
                z_var = model.NewBoolVar(f"setupX_{safe_id(batch_id)}_{roaster}_{sv}")
                setup_start_choice[batch_id][roaster][sv] = z_var
                model.Add(z_var <= x_var)
                model.Add(z_var <= setup_var)
                model.Add(z_var >= x_var + setup_var - 1)
                for minute in range(max(0, sv - ST), sv):
                    if minute < SL:
                        setup_terms[roaster][minute].append(z_var)

    # ── C6': First-batch setup delay for MTO ─────────────────────────────
    for batch_id in mto_batches:
        sku = batch_sku[batch_id]
        for roaster in sched_elig[batch_id]:
            if d["roaster_initial_sku"][roaster] != sku:
                model.Add(start[batch_id] >= ST).OnlyEnforceIf(lit[batch_id][roaster])

    # ── C4': NoOverlap on roaster (roasting) with fixed intervals ────────
    for roaster in roasters:
        intervals = [
            roast_interval[bid][roaster]
            for bid in all_batches
            if roaster in roast_interval[bid]
        ]
        intervals.extend(fixed_roast_intervals[roaster])
        model.AddNoOverlap(intervals)

    # ── C8'/C17': NoOverlap on pipeline ──────────────────────────────────
    for line_id in lines:
        pipeline_intervals = [
            consume_interval[bid][r]
            for bid in all_batches
            for r in sched_elig[bid]
            if d["roaster_pipeline"][r] == line_id
        ]
        pipeline_intervals.extend(fixed_consume_intervals[line_id])
        for pair in feasible_gc_pairs:
            if pair[0] != line_id:
                continue
            pipeline_intervals.extend(slot["pipe_interval"] for slot in restock_slots[pair])
        model.AddNoOverlap(pipeline_intervals)

    # ── C18': NoOverlap on all restocks globally ─────────────────────────
    all_restock_intervals = [
        slot["interval"] for pair in feasible_gc_pairs for slot in restock_slots[pair]
    ]
    model.AddNoOverlap(all_restock_intervals)

    # ── C10'/C11': RC inventory balance (soft stockout) ──────────────────
    consumption_sets = {line: set(d["consumption_events"][line]) for line in lines}

    for line_id in lines:
        for minute in range(SL):
            rc_level[line_id][minute] = model.NewIntVar(
                rc_lower, MRC, f"rc_{line_id}_{minute}"
            )
            low[line_id][minute] = model.NewBoolVar(f"low_{line_id}_{minute}")
            full[line_id][minute] = model.NewBoolVar(f"full_{line_id}_{minute}")

            # Completion terms: sum of integer constants (from fixed intervals)
            # and BoolVar references (from decision variables)
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

            # RC_STATE: low/full indicator channeling
            model.Add(rc_level[line_id][minute] <= SS - 1).OnlyEnforceIf(low[line_id][minute])
            model.Add(rc_level[line_id][minute] >= SS).OnlyEnforceIf(low[line_id][minute].Not())
            model.Add(rc_level[line_id][minute] >= MRC).OnlyEnforceIf(full[line_id][minute])
            model.Add(rc_level[line_id][minute] <= MRC - 1).OnlyEnforceIf(full[line_id][minute].Not())

            # C10' soft stockout: penalty when RC < 0 at consumption event
            if stockout_soft and consume_now:
                neg_var = model.NewBoolVar(f"stockout_{line_id}_{minute}")
                stockout_at[line_id][minute] = neg_var
                model.Add(rc_level[line_id][minute] <= -1).OnlyEnforceIf(neg_var)
                model.Add(rc_level[line_id][minute] >= 0).OnlyEnforceIf(neg_var.Not())

    # ── C16': GC reservoir constraint ────────────────────────────────────
    for pair in feasible_gc_pairs:
        line_id, sku = pair
        times: list[Any] = [0]
        level_changes: list[Any] = [int(d["gc_init"][pair])]
        actives: list[Any] = [1]

        for bid in all_batches:
            if batch_sku[bid] != sku:
                continue
            for r in sched_elig[bid]:
                if d["roaster_pipeline"][r] != line_id:
                    continue
                times.append(start[bid])
                level_changes.append(-1)
                actives.append(lit[bid][r])

        for slot in restock_slots[pair]:
            times.append(slot["start"] + RST_DUR)
            level_changes.append(RST_QTY)
            actives.append(slot["present"])

        model.add_reservoir_constraint_with_active(
            times, level_changes, actives, 0, int(d["gc_capacity"][pair]),
        )

    # ── C12': Tardiness definition ───────────────────────────────────────
    for job_id in d["jobs"]:
        due = int(d["job_due"][job_id])
        for bid in mto_batches:
            if bid[0] != job_id:
                continue
            duration = int(roast_time[batch_sku[bid]])
            if mto_soft and bid in mto_assigned:
                # Only enforce tardiness when this batch is actually scheduled.
                model.Add(tard[job_id] >= start[bid] + duration - due).OnlyEnforceIf(mto_assigned[bid])
            elif not mto_soft:
                model.Add(tard[job_id] >= start[bid] + duration - due)

    # ── SYM: PSC pool symmetry breaking ──────────────────────────────────
    for roaster in roasters:
        roaster_psc = sorted(
            [bid for bid in psc_pool if bid[0] == roaster],
            key=lambda b: b[1],
        )
        for left, right in zip(roaster_psc, roaster_psc[1:]):
            model.Add(lit[left][roaster] >= lit[right][roaster])

    # ── IDLE_OVER: idle and overflow detection ───────────────────────────
    if d["allow_r3_flex"]:
        for minute in range(SL):
            both_var = model.NewBoolVar(f"full_both_{minute}")
            full_both[minute] = both_var
            model.Add(both_var <= full["L1"][minute])
            model.Add(both_var <= full["L2"][minute])
            model.Add(both_var >= full["L1"][minute] + full["L2"][minute] - 1)

    for roaster in roasters:
        home_line = d["roaster_line"][roaster]
        out_line = d["roaster_can_output"][roaster][0]
        downtime = d["downtime_slots"].get(roaster, set())
        for minute in range(SL):
            running[roaster][minute] = model.NewBoolVar(f"running_{roaster}_{minute}")
            setup_active[roaster][minute] = model.NewBoolVar(f"setup_{roaster}_{minute}")
            idle[roaster][minute] = model.NewBoolVar(f"idle_{roaster}_{minute}")
            over[roaster][minute] = model.NewBoolVar(f"over_{roaster}_{minute}")

            # Include fixed interval running_terms as integer constants
            fixed_running = sum(
                1 for fi in fixed_intervals
                if fi["roaster"] == roaster and fi["start"] <= minute < fi["end"]
            )
            model.Add(
                running[roaster][minute]
                == sum(running_terms[roaster][minute]) + fixed_running
            )
            model.Add(setup_active[roaster][minute] == sum(setup_terms[roaster][minute]))

            if minute in downtime:
                model.Add(idle[roaster][minute] == 0)
                model.Add(over[roaster][minute] == 0)
                continue

            model.Add(idle[roaster][minute] <= low[home_line][minute])
            model.Add(idle[roaster][minute] <= 1 - running[roaster][minute])
            model.Add(idle[roaster][minute] >= low[home_line][minute] - running[roaster][minute])

            full_cond = (
                full_both[minute]
                if roaster == "R3" and d["allow_r3_flex"]
                else full[out_line][minute]
            )
            model.Add(over[roaster][minute] <= full_cond)
            model.Add(over[roaster][minute] <= 1 - running[roaster][minute])
            model.Add(over[roaster][minute] <= 1 - setup_active[roaster][minute])
            model.Add(
                over[roaster][minute]
                >= full_cond - running[roaster][minute] - setup_active[roaster][minute]
            )

    # ── WARM-START HINT: greedy PSC schedule ────────────────────────────
    _apply_greedy_hint(
        model, d, all_batches, mto_batches, psc_pool, sched_elig, batch_sku,
        start, lit, start_choice, empty_arc, first_arc, last_arc, seq_arc,
        setup_before, setup_start_choice, tard, restock_slots,
        running, setup_active, rc_level, low, full, full_both,
        idle, over, stockout_at, fixed_intervals,
    )

    # ── OBJECTIVE ────────────────────────────────────────────────────────
    revenue_psc = _as_int_coefficient(d["sku_revenue"]["PSC"], "sku_revenue.PSC")

    if mto_soft:
        # Revenue only accrues for batches that actually get scheduled.
        mto_revenue_expr = sum(
            _as_int_coefficient(d["sku_revenue"][batch_sku[bid]], f"sku_revenue.{batch_sku[bid]}")
            * mto_assigned[bid]
            for bid in mto_batches if bid in mto_assigned
        )
    else:
        mto_revenue_expr = sum(
            _as_int_coefficient(d["sku_revenue"][batch_sku[bid]], f"sku_revenue.{batch_sku[bid]}")
            for bid in mto_batches
        )
    # Add revenue from fixed MTO intervals (they WILL complete)
    fixed_mto_revenue = 0
    for fi in fixed_intervals:
        if fi.get("is_mto", False):
            fi_sku = fi["sku"]
            fixed_mto_revenue += _as_int_coefficient(d["sku_revenue"][fi_sku], f"sku_revenue.{fi_sku}")
    # Add revenue from fixed PSC intervals
    fixed_psc_revenue = sum(
        revenue_psc for fi in fixed_intervals
        if fi["sku"] == "PSC"
    )

    cost_tardiness = _as_int_coefficient(d["cost_tardiness"], "cost_tardiness")
    cost_setup = _as_int_coefficient(d["cost_setup"], "cost_setup")
    cost_idle = _as_int_coefficient(d["cost_idle"], "cost_idle")
    cost_overflow = _as_int_coefficient(d["cost_overflow"], "cost_overflow")

    psc_revenue_expr = sum(
        revenue_psc * lit[bid][sched_elig[bid][0]]
        for bid in psc_pool
        if sched_elig[bid]
    )
    tard_penalty = cost_tardiness * sum(tard.values()) if tard else 0
    exact_setup_events = sum(
        sv for bm in setup_before.values() for sv in bm.values()
    )
    setup_penalty = cost_setup * exact_setup_events
    idle_penalty = cost_idle * sum(
        idle[r][m] for r in roasters for m in range(SL)
    )
    overflow_penalty = cost_overflow * sum(
        over[r][m] for r in roasters for m in range(SL)
    )

    # C10': Stockout penalty (per consumption event with RC < 0)
    stockout_penalty_expr = 0
    if stockout_soft:
        all_stockout_vars = [
            v for line_dict in stockout_at.values() for v in line_dict.values()
        ]
        if all_stockout_vars:
            stockout_penalty_expr = stockout_penalty_cost * sum(all_stockout_vars)

    # MTO skip penalty — per unscheduled MTO batch. Matches env's c_skip_mto.
    skip_penalty_expr = 0
    if mto_soft:
        c_skip = _as_int_coefficient(d.get("c_skip_mto", 100000), "c_skip_mto")
        num_mto = len(mto_batches)
        assigned_count_expr = sum(mto_assigned[bid] for bid in mto_batches if bid in mto_assigned)
        # (num_mto - sum(assigned)) — also charges forced-skipped batches.
        skip_penalty_expr = c_skip * (num_mto - assigned_count_expr)

    model.Maximize(
        psc_revenue_expr
        + fixed_psc_revenue
        + mto_revenue_expr
        + fixed_mto_revenue
        - tard_penalty
        - setup_penalty
        - idle_penalty
        - overflow_penalty
        - stockout_penalty_expr
        - skip_penalty_expr
    )

    elapsed = time.perf_counter() - t_build
    model._build_meta = {
        "build_seconds": elapsed,
        "formulation": "reactive interval-based (soft stockout, fixed intervals)",
        "horizon": SL,
        "t0": d.get("t0", 0),
        "fixed_intervals": len(fixed_intervals),
        "mto_batches": len(mto_batches),
        "psc_pool": len(psc_pool),
    }
    logger.info("Reactive model built in %.2fs — horizon=%d", elapsed, SL)

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
        "stockout_at": stockout_at,
    }
    return model, vars_dict
