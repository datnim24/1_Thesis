"""Stage 2 MILP model builder for the deterministic roasting benchmark."""

from __future__ import annotations

import logging
import sys
import time
from itertools import combinations

import pulp

from data import load


logger = logging.getLogger("model")


def safe_id(batch_id) -> str:
    """Convert tuple batch identifiers into PuLP-safe ASCII names."""
    return (
        str(batch_id)
        .replace(" ", "")
        .replace("'", "")
        .replace(",", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _count_nested_vars(mapping: dict) -> int:
    return sum(len(inner) for inner in mapping.values())


def _extract_windows(slots: set[int]) -> list[tuple[int, int]]:
    if not slots:
        return []

    ordered = sorted(slots)
    windows: list[tuple[int, int]] = []
    start = ordered[0]
    previous = ordered[0]
    for slot in ordered[1:]:
        if slot == previous + 1:
            previous = slot
            continue
        windows.append((start, previous))
        start = slot
        previous = slot
    windows.append((start, previous))
    return windows


def _relevant_rc_times_for_roaster(
    roaster_id: str,
    rc_times: dict[str, list[int]],
    roaster_can_output: dict[str, list[str]],
) -> list[int]:
    times: set[int] = set()
    for line_id in roaster_can_output[roaster_id]:
        times.update(rc_times[line_id])
    return sorted(times)


def _build_event_prefix_counts(events: list[int], rc_times: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    event_index = 0
    ordered_events = sorted(events)
    for t in rc_times:
        while event_index < len(ordered_events) and ordered_events[event_index] <= t:
            event_index += 1
        counts[t] = event_index
    return counts


def build(d: dict) -> tuple[pulp.LpProblem, dict]:
    build_start = time.perf_counter()
    r3_mode = "flex" if d["allow_r3_flex"] else "fixed"
    logger.info("Building model - R3 mode: %s", r3_mode)
    logger.warning(
        "Change-point-only approximation for idle/overflow-idle active. "
        "idle[r][t] and over[r][t] exist only at consumption events, "
        "not at all 480 time slots. This may undercount idle periods."
    )

    if d["allow_r3_flex"]:
        logger.warning(
            "Stage 2 routing scope is limited to PSC pool batches whose ID starts with R3. "
            "Other PSC batches assigned to R3 contribute to L2 only in this build."
        )

    BIG_M = d["shift_length"]
    prob = pulp.LpProblem("roasting_schedule", pulp.LpMaximize)

    vars = {
        "a": {},
        "x": {},
        "s": {},
        "y": {},
        "tard": {},
        "delta": {},
        "pdelta": {},
        "w": {},
        "idle": {},
        "over": {},
        "rc": {},
        "low": {},
        "full": {},
        "c_complete": {},
    }

    # Local auxiliary variables required by the build prompt but not exposed in vars.
    z_window: dict[tuple, pulp.LpVariable] = {}
    r3_complete_l1: dict[tuple, pulp.LpVariable] = {}
    r3_complete_l2: dict[tuple, pulp.LpVariable] = {}

    var_counts: dict[str, int] = {}
    constraint_counts = {
        "C2": 0,
        "C3": 0,
        "C4/C5": 0,
        "C6": 0,
        "C7": 0,
        "C8": 0,
        "C10": 0,
        "C11": 0,
        "C12": 0,
        "C13": 0,
        "C14": 0,
        "C15": 0,
    }

    all_batches = d["all_batches"]
    psc_pool = d["psc_pool"]
    batch_eligible = d["batch_eligible_roasters"]
    batch_sku = d["batch_sku"]
    roasters = d["roasters"]
    lines = d["lines"]
    roaster_line = d["roaster_line"]
    roaster_pipeline = d["roaster_pipeline"]
    roaster_can_output = d["roaster_can_output"]
    process_time = d["process_time"]
    consume_time = d["consume_time"]
    setup_time = d["setup_time"]

    idle_times = {
        r: [
            t
            for t in d["consumption_events"][roaster_line[r]]
            if t not in d["downtime_slots"][r]
        ]
        for r in roasters
    }
    all_consumption_times = sorted(set().union(*(d["consumption_events"][line_id] for line_id in lines)))
    full_indicator_times = {
        line_id: list(all_consumption_times)
        for line_id in lines
    }

    possible_completion_times = list(range(process_time, d["shift_length"] + 1))
    rc_times = {
        line_id: sorted(set(all_consumption_times) | set(possible_completion_times))
        for line_id in lines
    }
    for line_id in lines:
        logger.debug("Change points for %s: %d", line_id, len(rc_times[line_id]))

    consumption_prefix_counts = {
        line_id: _build_event_prefix_counts(d["consumption_events"][line_id], rc_times[line_id])
        for line_id in lines
    }

    candidate_batches_by_roaster = {
        r: sorted([b for b in all_batches if r in batch_eligible[b]])
        for r in roasters
    }
    pipe_candidates = {
        line_id: sorted(d["pipeline_batches"].get(line_id, []))
        for line_id in sorted(d["pipeline_batches"].keys())
    }

    r3_routable_batches = [
        b for b in psc_pool if b[0] == "R3" and "R3" in batch_eligible[b]
    ]
    fixed_output_pairs = {
        line_id: [
            (b, r)
            for b in psc_pool
            for r in batch_eligible[b]
            if r != "R3" and roaster_can_output[r] == [line_id]
        ]
        for line_id in lines
    }

    # Variable creation.
    phase_start = time.perf_counter()

    for b in psc_pool:
        vars["a"][b] = pulp.LpVariable(f"a_{safe_id(b)}", cat="Binary")
    var_counts["a"] = len(vars["a"])
    logger.debug("Created a variables: %d", var_counts["a"])

    for b in all_batches:
        vars["x"][b] = {}
        for r in batch_eligible[b]:
            vars["x"][b][r] = pulp.LpVariable(f"x_{safe_id(b)}_{r}", cat="Binary")
    var_counts["x"] = sum(len(inner) for inner in vars["x"].values())
    logger.debug("Created x variables: %d", var_counts["x"])

    for b in all_batches:
        vars["s"][b] = pulp.LpVariable(
            f"s_{safe_id(b)}",
            lowBound=0,
            upBound=d["max_start"],
            cat="Continuous",
        )
    var_counts["s"] = len(vars["s"])
    logger.debug("Created s variables: %d", var_counts["s"])

    if d["allow_r3_flex"]:
        for b in r3_routable_batches:
            vars["y"][b] = pulp.LpVariable(f"y_{safe_id(b)}", cat="Binary")
            vars["w"][b] = pulp.LpVariable(f"w_{safe_id(b)}", cat="Binary")
    var_counts["y"] = len(vars["y"])
    var_counts["w"] = len(vars["w"])
    logger.debug("Created y variables: %d", var_counts["y"])
    logger.debug("Created w variables: %d", var_counts["w"])

    for job_id in d["jobs"]:
        vars["tard"][job_id] = pulp.LpVariable(f"tard_{job_id}", lowBound=0, cat="Continuous")
    var_counts["tard"] = len(vars["tard"])
    logger.debug("Created tard variables: %d", var_counts["tard"])

    for r in roasters:
        vars["idle"][r] = {}
        vars["over"][r] = {}
        for t in idle_times[r]:
            vars["idle"][r][t] = pulp.LpVariable(f"idle_{r}_{t}", cat="Binary")
            vars["over"][r][t] = pulp.LpVariable(f"over_{r}_{t}", cat="Binary")
    var_counts["idle"] = _count_nested_vars(vars["idle"])
    var_counts["over"] = _count_nested_vars(vars["over"])
    logger.debug("Created idle variables: %d", var_counts["idle"])
    logger.debug("Created over variables: %d", var_counts["over"])

    for line_id in lines:
        vars["rc"][line_id] = {}
        for t in rc_times[line_id]:
            vars["rc"][line_id][t] = pulp.LpVariable(
                f"rc_{line_id}_{t}",
                lowBound=0,
                upBound=d["max_rc"],
                cat="Continuous",
            )
    var_counts["rc"] = _count_nested_vars(vars["rc"])
    logger.debug("Created rc variables: %d", var_counts["rc"])

    for line_id in lines:
        vars["low"][line_id] = {}
        vars["full"][line_id] = {}
        for t in d["consumption_events"][line_id]:
            vars["low"][line_id][t] = pulp.LpVariable(f"low_{line_id}_{t}", cat="Binary")
        for t in full_indicator_times[line_id]:
            vars["full"][line_id][t] = pulp.LpVariable(f"full_{line_id}_{t}", cat="Binary")
    var_counts["low"] = _count_nested_vars(vars["low"])
    var_counts["full"] = _count_nested_vars(vars["full"])
    logger.debug("Created low variables: %d", var_counts["low"])
    logger.debug("Created full variables: %d", var_counts["full"])

    c_complete_count = 0
    for b in psc_pool:
        for r in batch_eligible[b]:
            relevant_times = _relevant_rc_times_for_roaster(r, rc_times, roaster_can_output)
            for tau in relevant_times:
                key = (b, r, tau)
                vars["c_complete"][key] = pulp.LpVariable(
                    f"c_complete_{safe_id(b)}_{r}_{tau}",
                    cat="Binary",
                )
                c_complete_count += 1
    var_counts["c_complete"] = c_complete_count
    logger.debug("Created c_complete variables: %d", var_counts["c_complete"])
    logger.info("Variable creation complete in %.2fs", time.perf_counter() - phase_start)

    if var_counts["c_complete"] > 100000:
        logger.warning(
            "RC linking var count %d - model may be slow to build and solve.",
            var_counts["c_complete"],
        )

    # Objective function.
    prob += (
        pulp.lpSum(d["sku_revenue"]["PSC"] * vars["a"][b] for b in psc_pool)
        + sum(d["sku_revenue"][batch_sku[b]] for b in d["mto_batches"])
        - d["cost_tardiness"] * pulp.lpSum(vars["tard"][j] for j in d["jobs"])
        - d["cost_idle"]
        * pulp.lpSum(
            vars["idle"][r][t]
            for r in roasters
            for t in idle_times[r]
        )
        - d["cost_overflow"]
        * pulp.lpSum(
            vars["over"][r][t]
            for r in roasters
            for t in idle_times[r]
        )
    ), "Total_Profit"

    # C2
    phase_start = time.perf_counter()
    for b in psc_pool:
        prob += (
            pulp.lpSum(vars["x"][b][r] for r in batch_eligible[b]) == vars["a"][b],
            f"C2_activation_{safe_id(b)}",
        )
        constraint_counts["C2"] += 1
    logger.debug("Constraint count C2: %d", constraint_counts["C2"])
    logger.info("C2 complete in %.2fs", time.perf_counter() - phase_start)

    # C3
    phase_start = time.perf_counter()
    for b in d["mto_batches"]:
        prob += (
            pulp.lpSum(vars["x"][b][r] for r in batch_eligible[b]) == 1,
            f"C3_mto_assign_{safe_id(b)}",
        )
        constraint_counts["C3"] += 1
    logger.debug("Constraint count C3: %d", constraint_counts["C3"])
    logger.info("C3 complete in %.2fs", time.perf_counter() - phase_start)

    # C4/C5
    phase_start = time.perf_counter()
    c45_pair_count = sum(
        len(list(combinations(candidate_batches_by_roaster[r], 2)))
        for r in roasters
    )
    logger.debug("Big-M pairs for C4/C5: %d", c45_pair_count)
    if c45_pair_count > 50000:
        logger.warning("Large C4/C5 pair count %d - model build may be slow.", c45_pair_count)

    for r in roasters:
        for b1, b2 in combinations(candidate_batches_by_roaster[r], 2):
            key = (b1, b2, r)
            vars["delta"][key] = pulp.LpVariable(
                f"delta_{safe_id(b1)}_{safe_id(b2)}_{r}",
                cat="Binary",
            )
            setup_gap = setup_time if batch_sku[b1] != batch_sku[b2] else 0
            reverse_gap = setup_time if batch_sku[b2] != batch_sku[b1] else 0

            prob += (
                vars["s"][b2]
                >= vars["s"][b1]
                + process_time
                + setup_gap
                - BIG_M * (1 - vars["delta"][key])
                - BIG_M * (1 - vars["x"][b1][r])
                - BIG_M * (1 - vars["x"][b2][r]),
                f"C4_A_{safe_id(b1)}_{safe_id(b2)}_{r}",
            )
            prob += (
                vars["s"][b1]
                >= vars["s"][b2]
                + process_time
                + reverse_gap
                - BIG_M * vars["delta"][key]
                - BIG_M * (1 - vars["x"][b1][r])
                - BIG_M * (1 - vars["x"][b2][r]),
                f"C4_B_{safe_id(b1)}_{safe_id(b2)}_{r}",
            )
            constraint_counts["C4/C5"] += 2
    var_counts["delta"] = len(vars["delta"])
    logger.debug("Created delta variables: %d", var_counts["delta"])
    logger.debug("Constraint count C4/C5: %d", constraint_counts["C4/C5"])
    logger.info("C4/C5 complete in %.2fs", time.perf_counter() - phase_start)

    # C6
    phase_start = time.perf_counter()
    for b in all_batches:
        if batch_sku[b] == "PSC":
            continue
        for r in batch_eligible[b]:
            prob += (
                vars["s"][b] >= setup_time - BIG_M * (1 - vars["x"][b][r]),
                f"C6_initsetup_{safe_id(b)}_{r}",
            )
            constraint_counts["C6"] += 1
    logger.debug("Constraint count C6: %d", constraint_counts["C6"])
    logger.info("C6 complete in %.2fs", time.perf_counter() - phase_start)

    # C7
    phase_start = time.perf_counter()
    for r in roasters:
        windows = _extract_windows(d["downtime_slots"][r])
        for window_index, (dstart, dend) in enumerate(windows):
            for b in candidate_batches_by_roaster[r]:
                z_key = (b, r, dstart, window_index)
                z_window[z_key] = pulp.LpVariable(
                    f"z_{safe_id(b)}_{r}_{dstart}",
                    cat="Binary",
                )
                prob += (
                    vars["s"][b] + process_time
                    <= dstart + BIG_M * z_window[z_key] + BIG_M * (1 - vars["x"][b][r]),
                    f"C7_A_{safe_id(b)}_{r}_{dstart}",
                )
                prob += (
                    vars["s"][b]
                    >= (dend + 1)
                    - BIG_M * (1 - z_window[z_key])
                    - BIG_M * (1 - vars["x"][b][r]),
                    f"C7_B_{safe_id(b)}_{r}_{dstart}",
                )
                constraint_counts["C7"] += 2
    logger.debug("Constraint count C7: %d", constraint_counts["C7"])
    logger.info("C7 complete in %.2fs", time.perf_counter() - phase_start)

    # C8
    phase_start = time.perf_counter()
    c8_pair_count = sum(len(list(combinations(pipe_candidates[l], 2))) for l in pipe_candidates)
    logger.debug("Big-M pairs for C8: %d", c8_pair_count)
    for line_id in pipe_candidates:
        line_roasters = [r for r in roasters if roaster_pipeline[r] == line_id]
        for b1, b2 in combinations(pipe_candidates[line_id], 2):
            key = (b1, b2, line_id)
            vars["pdelta"][key] = pulp.LpVariable(
                f"pdelta_{safe_id(b1)}_{safe_id(b2)}_{line_id}",
                cat="Binary",
            )
            assigned_b1 = pulp.lpSum(vars["x"][b1][r] for r in line_roasters if r in vars["x"][b1])
            assigned_b2 = pulp.lpSum(vars["x"][b2][r] for r in line_roasters if r in vars["x"][b2])

            prob += (
                vars["s"][b2]
                >= vars["s"][b1]
                + consume_time
                - BIG_M * (1 - vars["pdelta"][key])
                - BIG_M * (1 - assigned_b1)
                - BIG_M * (1 - assigned_b2),
                f"C8_A_{safe_id(b1)}_{safe_id(b2)}_{line_id}",
            )
            prob += (
                vars["s"][b1]
                >= vars["s"][b2]
                + consume_time
                - BIG_M * vars["pdelta"][key]
                - BIG_M * (1 - assigned_b1)
                - BIG_M * (1 - assigned_b2),
                f"C8_B_{safe_id(b1)}_{safe_id(b2)}_{line_id}",
            )
            constraint_counts["C8"] += 2
    var_counts["pdelta"] = len(vars["pdelta"])
    logger.debug("Created pdelta variables: %d", var_counts["pdelta"])
    logger.debug("Constraint count C8: %d", constraint_counts["C8"])
    logger.info("C8 complete in %.2fs", time.perf_counter() - phase_start)

    # C12
    phase_start = time.perf_counter()
    for job_id in d["jobs"]:
        for b in d["mto_batches"]:
            if b[0] != job_id:
                continue
            prob += (
                vars["tard"][job_id] >= vars["s"][b] + process_time - d["job_due"][job_id],
                f"C12_tard_{job_id}_{safe_id(b)}",
            )
            constraint_counts["C12"] += 1
    logger.debug("Constraint count C12: %d", constraint_counts["C12"])
    logger.info("C12 complete in %.2fs", time.perf_counter() - phase_start)

    # C13
    phase_start = time.perf_counter()
    if d["allow_r3_flex"]:
        for b in r3_routable_batches:
            prob += (
                vars["w"][b] <= vars["x"][b]["R3"],
                f"C13_w_ub_x_{safe_id(b)}",
            )
            prob += (
                vars["w"][b] <= vars["y"][b],
                f"C13_w_ub_y_{safe_id(b)}",
            )
            prob += (
                vars["w"][b] >= vars["x"][b]["R3"] + vars["y"][b] - 1,
                f"C13_w_lb_{safe_id(b)}",
            )
            prob += (
                vars["w"][b] >= 0,
                f"C13_w_nonneg_{safe_id(b)}",
            )
            constraint_counts["C13"] += 4
    logger.debug("Constraint count C13: %d", constraint_counts["C13"])
    logger.info("C13 complete in %.2fs", time.perf_counter() - phase_start)

    # C10/C11
    phase_start = time.perf_counter()
    for b in psc_pool:
        for r in batch_eligible[b]:
            support_var = vars["x"][b][r]
            for tau in _relevant_rc_times_for_roaster(r, rc_times, roaster_can_output):
                c_var = vars["c_complete"][(b, r, tau)]
                prob += (
                    c_var <= support_var,
                    f"C10_c_ub_{safe_id(b)}_{r}_{tau}",
                )
                prob += (
                    vars["s"][b] + process_time <= tau + BIG_M * (1 - c_var),
                    f"C10_c_timeub_{safe_id(b)}_{r}_{tau}",
                )
                prob += (
                    vars["s"][b] + process_time
                    >= (tau + 1)
                    - BIG_M * c_var
                    - BIG_M * (1 - support_var),
                    f"C10_c_timelb_{safe_id(b)}_{r}_{tau}",
                )
                constraint_counts["C10"] += 3

    if d["allow_r3_flex"]:
        for b in r3_routable_batches:
            for tau in rc_times["L1"]:
                key = (b, tau)
                r3_complete_l1[key] = pulp.LpVariable(
                    f"c_r3_l1_{safe_id(b)}_{tau}",
                    cat="Binary",
                )
                prob += (
                    r3_complete_l1[key] <= vars["w"][b],
                    f"C10_r3l1_ub_{safe_id(b)}_{tau}",
                )
                prob += (
                    vars["s"][b] + process_time <= tau + BIG_M * (1 - r3_complete_l1[key]),
                    f"C10_r3l1_timeub_{safe_id(b)}_{tau}",
                )
                prob += (
                    vars["s"][b] + process_time
                    >= (tau + 1)
                    - BIG_M * r3_complete_l1[key]
                    - BIG_M * (1 - vars["w"][b]),
                    f"C10_r3l1_timelb_{safe_id(b)}_{tau}",
                )
                constraint_counts["C10"] += 3

            r3_l2_support = vars["x"][b]["R3"] - vars["w"][b]
            for tau in rc_times["L2"]:
                key = (b, tau)
                r3_complete_l2[key] = pulp.LpVariable(
                    f"c_r3_l2_{safe_id(b)}_{tau}",
                    cat="Binary",
                )
                prob += (
                    r3_complete_l2[key] <= r3_l2_support,
                    f"C10_r3l2_ub_{safe_id(b)}_{tau}",
                )
                prob += (
                    vars["s"][b] + process_time <= tau + BIG_M * (1 - r3_complete_l2[key]),
                    f"C10_r3l2_timeub_{safe_id(b)}_{tau}",
                )
                prob += (
                    vars["s"][b] + process_time
                    >= (tau + 1)
                    - BIG_M * r3_complete_l2[key]
                    - BIG_M * (1 - r3_l2_support),
                    f"C10_r3l2_timelb_{safe_id(b)}_{tau}",
                )
                constraint_counts["C10"] += 3

    for line_id in lines:
        for tau in rc_times[line_id]:
            completed_terms = []
            for b, r in fixed_output_pairs[line_id]:
                completed_terms.append(vars["c_complete"][(b, r, tau)])

            if line_id == "L2":
                for b in psc_pool:
                    if "R3" not in batch_eligible[b]:
                        continue
                    if d["allow_r3_flex"] and b in vars["w"]:
                        completed_terms.append(r3_complete_l2[(b, tau)])
                    else:
                        completed_terms.append(vars["c_complete"][(b, "R3", tau)])
            elif d["allow_r3_flex"]:
                for b in r3_routable_batches:
                    completed_terms.append(r3_complete_l1[(b, tau)])

            prob += (
                vars["rc"][line_id][tau]
                == d["rc_init"][line_id]
                + pulp.lpSum(completed_terms)
                - consumption_prefix_counts[line_id][tau],
                f"C10_balance_{line_id}_{tau}",
            )
            prob += (
                vars["rc"][line_id][tau] >= 0,
                f"C10_lb_{line_id}_{tau}",
            )
            prob += (
                vars["rc"][line_id][tau] <= d["max_rc"],
                f"C11_ub_{line_id}_{tau}",
            )
            constraint_counts["C10"] += 2
            constraint_counts["C11"] += 1

    logger.debug("Constraint count C10: %d", constraint_counts["C10"])
    logger.debug("Constraint count C11: %d", constraint_counts["C11"])
    logger.info("C10/C11 complete in %.2fs", time.perf_counter() - phase_start)

    # C14
    phase_start = time.perf_counter()
    busy_approx = {
        r: pulp.lpSum(vars["x"][b][r] for b in candidate_batches_by_roaster[r])
        for r in roasters
    }
    for line_id in lines:
        for t in d["consumption_events"][line_id]:
            rc_var = vars["rc"][line_id][t]
            prob += (
                rc_var <= d["safety_stock"] - 1 + BIG_M * (1 - vars["low"][line_id][t]),
                f"C14_low_ub_{line_id}_{t}",
            )
            prob += (
                rc_var >= d["safety_stock"] - BIG_M * vars["low"][line_id][t],
                f"C14_low_lb_{line_id}_{t}",
            )
            constraint_counts["C14"] += 2

    # This uses the coarse Stage 2 approximation: assignment on a roaster is used
    # as a proxy for "might be busy" at every tracked change point. It avoids a
    # full time-indexed busy-state model but can undercount true idle periods.
    for r in roasters:
        line_id = roaster_line[r]
        for t in idle_times[r]:
            prob += (
                vars["idle"][r][t] >= vars["low"][line_id][t] - BIG_M * busy_approx[r],
                f"C14_idle_{r}_{t}",
            )
            constraint_counts["C14"] += 1
    logger.debug("Constraint count C14: %d", constraint_counts["C14"])
    logger.info("C14 complete in %.2fs", time.perf_counter() - phase_start)

    # C15
    phase_start = time.perf_counter()
    for line_id in lines:
        for t in full_indicator_times[line_id]:
            prob += (
                vars["rc"][line_id][t] >= d["max_rc"] - BIG_M * (1 - vars["full"][line_id][t]),
                f"C15_full_{line_id}_{t}",
            )
            constraint_counts["C15"] += 1

    for r in roasters:
        for t in idle_times[r]:
            if r == "R3" and d["allow_r3_flex"]:
                prob += (
                    vars["over"][r][t]
                    >= vars["full"]["L1"][t] + vars["full"]["L2"][t] - 1 - BIG_M * busy_approx[r],
                    f"C15_over_R3_{t}",
                )
            elif r == "R3":
                prob += (
                    vars["over"][r][t] >= vars["full"]["L2"][t] - BIG_M * busy_approx[r],
                    f"C15_over_R3_fixed_{t}",
                )
            else:
                output_line = roaster_can_output[r][0]
                prob += (
                    vars["over"][r][t] >= vars["full"][output_line][t] - BIG_M * busy_approx[r],
                    f"C15_over_{r}_{t}",
                )
            constraint_counts["C15"] += 1
    logger.debug("Constraint count C15: %d", constraint_counts["C15"])
    logger.info("C15 complete in %.2fs", time.perf_counter() - phase_start)

    total_vars = len(prob.variables())
    total_constraints = len(prob.constraints)
    logger.info("Model built: %d variables, %d constraints", total_vars, total_constraints)

    prob._build_meta = {
        "var_counts": var_counts,
        "constraint_counts": constraint_counts,
        "total_vars": total_vars,
        "total_constraints": total_constraints,
        "r3_mode": r3_mode,
        "build_seconds": time.perf_counter() - build_start,
    }
    return prob, vars


def _print_build_report(prob: pulp.LpProblem, vars: dict) -> None:
    meta = getattr(prob, "_build_meta", {})
    var_counts = meta.get("var_counts", {})
    constraint_counts = meta.get("constraint_counts", {})

    print("=== MODEL BUILD REPORT ===")
    print(f"R3 mode               : {meta.get('r3_mode', 'unknown')}")
    print("Variables:")
    print(f"  a[b]   PSC activation      : {var_counts.get('a', 0)}  (expected 160)")
    print(f"  x[b][r] assignment         : {var_counts.get('x', 0)}")
    print(f"  s[b]   start times         : {var_counts.get('s', 0)}  (expected 164)")
    print(f"  y[b]   R3 routing          : {var_counts.get('y', 0)}  (expected ~32 / 0)")
    print(f"  tard   tardiness           : {var_counts.get('tard', 0)}  (expected 2)")
    print(f"  delta  roaster ordering    : {var_counts.get('delta', 0)}")
    print(f"  pdelta pipeline ordering   : {var_counts.get('pdelta', 0)}")
    print(f"  w[b]   R3 auxiliary        : {var_counts.get('w', 0)}  (expected ~32 / 0)")
    print(f"  idle                       : {var_counts.get('idle', 0)}")
    print(f"  over                       : {var_counts.get('over', 0)}")
    print(f"  rc                         : {var_counts.get('rc', 0)}")
    print(
        "  low / full indicators      : "
        f"{var_counts.get('low', 0) + var_counts.get('full', 0)}"
    )
    print(f"  c_complete linking         : {var_counts.get('c_complete', 0)}")
    print(f"  Total variables            : {meta.get('total_vars', len(prob.variables()))}")
    print("Constraints:")
    print(f"  C2  PSC activation         : {constraint_counts.get('C2', 0)}")
    print(f"  C3  MTO one-roaster        : {constraint_counts.get('C3', 0)}")
    print(f"  C4/C5 roaster NoOverlap    : {constraint_counts.get('C4/C5', 0)}")
    print(f"  C6  initial SKU setup      : {constraint_counts.get('C6', 0)}")
    print(f"  C7  planned downtime       : {constraint_counts.get('C7', 0)}")
    print(f"  C8  pipeline NoOverlap     : {constraint_counts.get('C8', 0)}")
    print(f"  C10 RC lower bound         : {constraint_counts.get('C10', 0)}")
    print(f"  C11 RC upper bound         : {constraint_counts.get('C11', 0)}")
    print(f"  C12 tardiness              : {constraint_counts.get('C12', 0)}")
    print(f"  C13 R3 routing (w links)   : {constraint_counts.get('C13', 0)}")
    print(f"  C14 safety-idle            : {constraint_counts.get('C14', 0)}")
    print(f"  C15 overflow-idle          : {constraint_counts.get('C15', 0)}")
    print(f"  Total constraints          : {meta.get('total_constraints', len(prob.constraints))}")
    print("=== END BUILD REPORT ===")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    data = load()
    problem, variables = build(data)
    _print_build_report(problem, variables)
