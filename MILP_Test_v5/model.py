"""MILP_Test_v5 model builder.

This keeps the v3 time-indexed scaffold, but upgrades the deterministic model
to the current thesis inventory rules:

- SKU-specific roast durations
- Exact deterministic RC balance with hard [0, max_rc] bounds
- Exact GC silo balance with restock decisions
- Restock blocks the line pipeline
- Shared global restock mutex

The formulation intentionally stays no-UPS.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from typing import Any

import pulp

try:
    from .data import load
except ImportError:
    from data import load


logger = logging.getLogger("model")


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


def build(d: dict[str, Any]) -> tuple[pulp.LpProblem, dict[str, Any]]:
    t_build = time.perf_counter()

    SL = d["shift_length"]
    DC = d["consume_time"]
    ST = d["setup_time"]
    MRC = d["max_rc"]
    SS = d["safety_stock"]
    RST_DUR = d["restock_duration"]
    RST_QTY = d["restock_qty"]

    all_batches = d["all_batches"]
    psc_pool = d["psc_pool"]
    mto_batches = d["mto_batches"]
    batch_sku = d["batch_sku"]
    batch_is_mto = d["batch_is_mto"]
    sched_elig = d["sched_eligible_roasters"]
    roasters = d["roasters"]
    lines = d["lines"]
    r_line = d["roaster_line"]
    r_pipe = d["roaster_pipeline"]
    r_out = d["roaster_can_output"]
    roast_time = d["roast_time_by_sku"]
    max_start_by_sku = d["MS_by_sku"]
    feasible_gc_pairs = list(d["feasible_gc_pairs"])

    logger.info("Building MILP_Test_v5 (time-indexed, deterministic inventory-managed model)")
    logger.info("R3 mode: %s", "flex" if d["allow_r3_flex"] else "fixed")

    prob = pulp.LpProblem("roasting_schedule_v5", pulp.LpMaximize)
    v: dict[str, Any] = {
        "a": {},
        "x_ti": {},
        "z_route": {},
        "tard": {},
        "restock": {},
        "rc": {},
        "gc": {},
        "low": {},
        "full": {},
        "idle": {r: {} for r in roasters},
        "over": {r: {} for r in roasters},
        "setup_ind": {},
    }

    vc: dict[str, int] = {}
    cc = {
        "C2": 0,
        "C3": 0,
        "C4": 0,
        "C5": 0,
        "C8_C17": 0,
        "C10_C11": 0,
        "C12": 0,
        "C13": 0,
        "C14": 0,
        "C15": 0,
        "C16": 0,
        "C18": 0,
        "C19": 0,
        "SYM": 0,
    }

    x_by_roaster_cover: dict[str, dict[int, list[pulp.LpVariable]]] = {
        r: defaultdict(list) for r in roasters
    }
    x_by_roaster_sku_start: dict[str, dict[str, dict[int, list[pulp.LpVariable]]]] = {
        r: defaultdict(lambda: defaultdict(list)) for r in roasters
    }
    x_by_pipeline_cover: dict[str, dict[int, list[pulp.LpVariable]]] = {
        line_id: defaultdict(list) for line_id in lines
    }
    x_by_gc_pair_start: dict[tuple[str, str], dict[int, list[pulp.LpVariable]]] = {
        pair: defaultdict(list) for pair in feasible_gc_pairs
    }
    psc_completion_terms: dict[str, dict[int, list[pulp.LpAffineExpression]]] = {
        line_id: defaultdict(list) for line_id in lines
    }

    forbidden_cache: dict[tuple[str, int], set[int]] = {}
    for roaster in roasters:
        for duration in set(roast_time.values()):
            forbidden_cache[(roaster, duration)] = _forbidden_starts(
                d["downtime_slots"].get(roaster, set()),
                duration,
                SL - duration,
            )

    t_vars = time.perf_counter()

    for batch_id in psc_pool:
        v["a"][batch_id] = pulp.LpVariable(f"a_{safe_id(batch_id)}", cat="Binary")
    vc["a"] = len(v["a"])

    for job_id in d["jobs"]:
        v["tard"][job_id] = pulp.LpVariable(f"tard_{job_id}", lowBound=0)
    vc["tard"] = len(v["tard"])

    for pair in feasible_gc_pairs:
        v["restock"][pair] = {}
        for start in range(0, SL - RST_DUR + 1):
            pair_label = f"{pair[0]}_{pair[1]}"
            v["restock"][pair][start] = pulp.LpVariable(
                f"rst_{pair_label}_{start}",
                cat="Binary",
            )
    vc["restock"] = sum(len(start_map) for start_map in v["restock"].values())

    for batch_id in all_batches:
        v["x_ti"][batch_id] = {}
        sku = batch_sku[batch_id]
        duration = roast_time[sku]
        max_start = max_start_by_sku[sku]
        for roaster in sched_elig[batch_id]:
            v["x_ti"][batch_id][roaster] = {}
            release = d["job_release"].get(batch_id[0], 0) if batch_is_mto[batch_id] else 0
            earliest = max(
                release,
                ST if sku != d["roaster_initial_sku"][roaster] else 0,
            )
            forbidden = forbidden_cache[(roaster, duration)]
            for start in range(earliest, max_start + 1):
                if start in forbidden:
                    continue
                var = pulp.LpVariable(
                    f"xti_{safe_id(batch_id)}_{roaster}_{start}",
                    cat="Binary",
                )
                v["x_ti"][batch_id][roaster][start] = var
                x_by_roaster_sku_start[roaster][sku][start].append(var)
                gc_pair = (r_pipe[roaster], sku)
                x_by_gc_pair_start[gc_pair][start].append(var)
                for slot in range(start, min(start + duration, SL)):
                    x_by_roaster_cover[roaster][slot].append(var)
                for slot in range(start, min(start + DC, SL)):
                    x_by_pipeline_cover[r_pipe[roaster]][slot].append(var)
                if sku == "PSC" and not (roaster == "R3" and d["allow_r3_flex"]):
                    completion = start + duration
                    if completion < SL:
                        psc_completion_terms[r_out[roaster][0]][completion].append(var)
    vc["x_ti"] = sum(
        len(start_map)
        for roaster_map in v["x_ti"].values()
        for start_map in roaster_map.values()
    )

    if d["allow_r3_flex"]:
        r3_pool = [batch_id for batch_id in psc_pool if batch_id[0] == "R3"]
        for batch_id in r3_pool:
            if "R3" not in v["x_ti"][batch_id]:
                continue
            v["z_route"][batch_id] = {"L1": {}, "L2": {}}
            for start, x_var in v["x_ti"][batch_id]["R3"].items():
                z_l1 = pulp.LpVariable(f"zr3_{safe_id(batch_id)}_L1_{start}", cat="Binary")
                z_l2 = pulp.LpVariable(f"zr3_{safe_id(batch_id)}_L2_{start}", cat="Binary")
                v["z_route"][batch_id]["L1"][start] = z_l1
                v["z_route"][batch_id]["L2"][start] = z_l2
                prob += (x_var == z_l1 + z_l2, f"C13_route_split_{safe_id(batch_id)}_{start}")
                cc["C13"] += 1
                completion = start + roast_time["PSC"]
                if completion < SL:
                    psc_completion_terms["L1"][completion].append(z_l1)
                    psc_completion_terms["L2"][completion].append(z_l2)
        vc["z_route"] = sum(
            len(start_map)
            for batch_map in v["z_route"].values()
            for start_map in batch_map.values()
        )
    else:
        vc["z_route"] = 0

    # Precompute consumption event sets for idle/over scoping.
    # low, full, idle, over indicators are only meaningful at slots where
    # RC stock changes (consumption events). Creating them at all 480 slots
    # inflates the LP relaxation penalty by ~4x and makes the root LP unsolvable.
    penalty_slots: dict[str, set[int]] = {
        line_id: set(d["consumption_events"][line_id])
        for line_id in lines
    }

    for line_id in lines:
        v["rc"][line_id] = {}
        v["low"][line_id] = {}
        v["full"][line_id] = {}
        for slot in range(SL):
            v["rc"][line_id][slot] = pulp.LpVariable(
                f"rc_{line_id}_{slot}",
                lowBound=0,
                upBound=MRC,
            )
            # low/full only at consumption event slots - see penalty_slots
            if slot in penalty_slots[line_id]:
                v["low"][line_id][slot] = pulp.LpVariable(f"low_{line_id}_{slot}", cat="Binary")
                v["full"][line_id][slot] = pulp.LpVariable(f"full_{line_id}_{slot}", cat="Binary")
    vc["rc"] = sum(len(slot_map) for slot_map in v["rc"].values())
    vc["low"] = sum(len(slot_map) for slot_map in v["low"].values())
    vc["full"] = sum(len(slot_map) for slot_map in v["full"].values())

    for pair in feasible_gc_pairs:
        pair_label = f"{pair[0]}_{pair[1]}"
        v["gc"][pair] = {}
        for slot in range(SL):
            v["gc"][pair][slot] = pulp.LpVariable(
                f"gc_{pair_label}_{slot}",
                lowBound=0,
                upBound=d["gc_capacity"][pair],
            )
    vc["gc"] = sum(len(slot_map) for slot_map in v["gc"].values())

    for roaster in roasters:
        line_id = r_line[roaster]
        downtime = d["downtime_slots"].get(roaster, set())
        for slot in penalty_slots[line_id]:
            if slot in downtime:
                continue
            v["idle"][roaster][slot] = pulp.LpVariable(f"idle_{roaster}_{slot}", cat="Binary")
            v["over"][roaster][slot] = pulp.LpVariable(f"over_{roaster}_{slot}", cat="Binary")
    vc["idle"] = sum(len(slot_map) for slot_map in v["idle"].values())
    vc["over"] = sum(len(slot_map) for slot_map in v["over"].values())

    logger.info("Variable creation complete in %.2fs", time.perf_counter() - t_vars)
    logger.info("Variable counts: %s", vc)

    assign_expr: dict[Any, dict[str, pulp.LpAffineExpression]] = {}
    for batch_id in all_batches:
        assign_expr[batch_id] = {}
        for roaster in sched_elig[batch_id]:
            assign_expr[batch_id][roaster] = pulp.lpSum(v["x_ti"][batch_id][roaster].values())

    # v3/v4 setup cost carry-over: count one in/out switch for each non-PSC SKU
    # activated on a multi-SKU roaster. This keeps the original MILP cost proxy.
    for roaster in roasters:
        if len(d["roaster_eligible_skus"][roaster]) <= 1:
            continue
        for sku in d["roaster_eligible_skus"][roaster]:
            if sku == "PSC":
                continue
            relevant_batches = [
                batch_id
                for batch_id in all_batches
                if sku == batch_sku[batch_id] and roaster in sched_elig[batch_id]
            ]
            if not relevant_batches:
                continue
            ind = pulp.LpVariable(f"setup_ind_{roaster}_{sku}", cat="Binary")
            v["setup_ind"][(roaster, sku)] = ind
            for batch_id in relevant_batches:
                prob += (
                    ind >= assign_expr[batch_id][roaster],
                    f"setup_lb_{roaster}_{sku}_{safe_id(batch_id)}",
                )
            prob += (
                ind <= pulp.lpSum(assign_expr[batch_id][roaster] for batch_id in relevant_batches),
                f"setup_ub_{roaster}_{sku}",
            )
    vc["setup_ind"] = len(v["setup_ind"])
    setup_cost_expr = 2 * d["cost_setup"] * pulp.lpSum(v["setup_ind"].values())

    mto_revenue = sum(d["sku_revenue"][batch_sku[batch_id]] for batch_id in mto_batches)
    prob += (
        pulp.lpSum(d["sku_revenue"]["PSC"] * v["a"][batch_id] for batch_id in psc_pool)
        + mto_revenue
        - d["cost_tardiness"] * pulp.lpSum(v["tard"].values())
        - setup_cost_expr
        - d["cost_idle"] * pulp.lpSum(
            v["idle"][roaster][slot]
            for roaster in roasters
            for slot in v["idle"][roaster]
        )
        - d["cost_overflow"] * pulp.lpSum(
            v["over"][roaster][slot]
            for roaster in roasters
            for slot in v["over"][roaster]
        )
    ), "Total_Profit"

    t_phase = time.perf_counter()
    for batch_id in psc_pool:
        prob += (
            pulp.lpSum(assign_expr[batch_id][roaster] for roaster in sched_elig[batch_id]) == v["a"][batch_id],
            f"C2_{safe_id(batch_id)}",
        )
        cc["C2"] += 1
    logger.info("C2 complete: %d in %.2fs", cc["C2"], time.perf_counter() - t_phase)

    t_phase = time.perf_counter()
    for batch_id in mto_batches:
        prob += (
            pulp.lpSum(assign_expr[batch_id][roaster] for roaster in sched_elig[batch_id]) == 1,
            f"C3_{safe_id(batch_id)}",
        )
        cc["C3"] += 1
    logger.info("C3 complete: %d in %.2fs", cc["C3"], time.perf_counter() - t_phase)

    t_phase = time.perf_counter()
    psc_by_seed = {
        roaster: sorted((batch_id for batch_id in psc_pool if batch_id[0] == roaster), key=lambda bid: bid[1])
        for roaster in roasters
    }
    for roaster, batches in psc_by_seed.items():
        for left, right in zip(batches, batches[1:]):
            prob += (v["a"][left] >= v["a"][right], f"SYM_a_{roaster}_{right[1]}")
            cc["SYM"] += 1
    logger.info("SYM complete: %d in %.2fs", cc["SYM"], time.perf_counter() - t_phase)

    t_phase = time.perf_counter()
    for roaster in roasters:
        for slot in range(SL):
            covering = x_by_roaster_cover[roaster].get(slot, [])
            if len(covering) > 1:
                prob += (pulp.lpSum(covering) <= 1, f"C4_{roaster}_{slot}")
                cc["C4"] += 1
    logger.info("C4 complete: %d in %.2fs", cc["C4"], time.perf_counter() - t_phase)

    t_phase = time.perf_counter()
    for roaster in roasters:
        start_map = x_by_roaster_sku_start[roaster]
        roaster_skus = sorted(start_map.keys())
        for sku_left in roaster_skus:
            duration_left = roast_time[sku_left]
            for sku_right in roaster_skus:
                if sku_left == sku_right:
                    continue
                for start in sorted(start_map[sku_left]):
                    lhs = start_map[sku_left][start]
                    rhs: list[pulp.LpVariable] = []
                    for next_start in range(start + duration_left, min(start + duration_left + ST, SL)):
                        rhs.extend(start_map[sku_right].get(next_start, []))
                    if rhs:
                        prob += (
                            pulp.lpSum(lhs) + pulp.lpSum(rhs) <= 1,
                            f"C5_{roaster}_{sku_left}_{sku_right}_{start}",
                        )
                        cc["C5"] += 1
    logger.info("C5 complete: %d in %.2fs", cc["C5"], time.perf_counter() - t_phase)

    restock_cover_by_line: dict[str, dict[int, list[pulp.LpVariable]]] = {
        line_id: defaultdict(list) for line_id in lines
    }
    restock_cover_global: dict[int, list[pulp.LpVariable]] = defaultdict(list)
    restock_complete_by_pair: dict[tuple[str, str], dict[int, list[pulp.LpVariable]]] = {
        pair: defaultdict(list) for pair in feasible_gc_pairs
    }
    for pair in feasible_gc_pairs:
        for start, rst_var in v["restock"][pair].items():
            for slot in range(start, min(start + RST_DUR, SL)):
                restock_cover_by_line[pair[0]][slot].append(rst_var)
                restock_cover_global[slot].append(rst_var)
            completion = start + RST_DUR
            if completion < SL:
                restock_complete_by_pair[pair][completion].append(rst_var)

    t_phase = time.perf_counter()
    for line_id in lines:
        for slot in range(SL):
            terms = list(x_by_pipeline_cover[line_id].get(slot, []))
            terms.extend(restock_cover_by_line[line_id].get(slot, []))
            if len(terms) > 1:
                prob += (pulp.lpSum(terms) <= 1, f"C8_C17_{line_id}_{slot}")
                cc["C8_C17"] += 1
    logger.info("C8/C17 complete: %d in %.2fs", cc["C8_C17"], time.perf_counter() - t_phase)

    t_phase = time.perf_counter()
    for slot in range(SL):
        terms = restock_cover_global.get(slot, [])
        if len(terms) > 1:
            prob += (pulp.lpSum(terms) <= 1, f"C18_{slot}")
            cc["C18"] += 1
    logger.info("C18 complete: %d in %.2fs", cc["C18"], time.perf_counter() - t_phase)

    t_phase = time.perf_counter()
    consumption_sets = {
        line_id: set(d["consumption_events"][line_id])
        for line_id in lines
    }
    for line_id in lines:
        for slot in range(SL):
            completion_expr = pulp.lpSum(psc_completion_terms[line_id].get(slot, []))
            consume = 1 if slot in consumption_sets[line_id] else 0
            if slot == 0:
                prob += (
                    v["rc"][line_id][slot] == d["rc_init"][line_id] + completion_expr - consume,
                    f"C10_RC_{line_id}_{slot}",
                )
            else:
                prob += (
                    v["rc"][line_id][slot]
                    == v["rc"][line_id][slot - 1] + completion_expr - consume,
                    f"C10_RC_{line_id}_{slot}",
                )
            cc["C10_C11"] += 1
    logger.info("C10/C11 complete: %d in %.2fs", cc["C10_C11"], time.perf_counter() - t_phase)

    t_phase = time.perf_counter()
    for pair in feasible_gc_pairs:
        for slot in range(SL):
            batch_start_expr = pulp.lpSum(x_by_gc_pair_start[pair].get(slot, []))
            restock_expr = RST_QTY * pulp.lpSum(restock_complete_by_pair[pair].get(slot, []))
            pair_label = f"{pair[0]}_{pair[1]}"
            if slot == 0:
                prob += (
                    v["gc"][pair][slot] == d["gc_init"][pair] + restock_expr - batch_start_expr,
                    f"C16_{pair_label}_{slot}",
                )
            else:
                prob += (
                    v["gc"][pair][slot]
                    == v["gc"][pair][slot - 1] + restock_expr - batch_start_expr,
                    f"C16_{pair_label}_{slot}",
                )
            cc["C16"] += 1
    logger.info("C16 complete: %d in %.2fs", cc["C16"], time.perf_counter() - t_phase)

    t_phase = time.perf_counter()
    # Tight Big-M values for C14/C15 indicator constraints.
    # Using M=MRC=40 produced a very weak LP relaxation: when the LP set
    # low=0.5, both constraints became non-binding (rc could be anything in
    # [0, 39]), so the LP could pay fractional idle penalty at rc=0 without
    # any real constraint pressure. This gave a dual bound of ~-$553K.
    #
    # Tight values derived from rc ∈ [0, MRC]:
    #   low=1 iff rc < SS:
    #     upper bound: when low=0, rc can reach MRC  → M = MRC - (SS-1) = MRC - SS + 1
    #     lower bound: when low=1, rc can reach 0    → M = SS
    #   full=1 iff rc = MRC:
    #     upper bound: when full=0, rc ≤ MRC-1       → M = 1  (tightest possible)
    #     lower bound: when full=0, rc ≥ 0           → M = MRC (unchanged, already tight)
    m_low_ub = MRC - SS + 1
    m_low_lb = SS
    m_full_ub = 1
    m_full_lb = MRC
    for line_id in lines:
        for slot in penalty_slots[line_id]:
            prob += (
                v["rc"][line_id][slot] <= (SS - 1) + m_low_ub * (1 - v["low"][line_id][slot]),
                f"C14_low_ub_{line_id}_{slot}",
            )
            prob += (
                v["rc"][line_id][slot] >= SS - m_low_lb * v["low"][line_id][slot],
                f"C14_low_lb_{line_id}_{slot}",
            )
            cc["C14"] += 2
            prob += (
                v["rc"][line_id][slot] >= MRC - m_full_lb * (1 - v["full"][line_id][slot]),
                f"C15_full_lb_{line_id}_{slot}",
            )
            prob += (
                v["rc"][line_id][slot] <= (MRC - 1) + m_full_ub * v["full"][line_id][slot],
                f"C15_full_ub_{line_id}_{slot}",
            )
            cc["C15"] += 2

    for roaster in roasters:
        line_id = r_line[roaster]
        for slot in v["idle"][roaster]:
            busy_expr = pulp.lpSum(x_by_roaster_cover[roaster].get(slot, []))
            prob += (
                v["idle"][roaster][slot] >= v["low"][line_id][slot] - busy_expr,
                f"C14_idle_{roaster}_{slot}",
            )
            cc["C14"] += 1
            if roaster == "R3" and d["allow_r3_flex"]:
                full_l1 = v["full"]["L1"].get(slot, 0)
                full_l2 = v["full"]["L2"].get(slot, 0)
                prob += (
                    v["over"][roaster][slot]
                    >= full_l1 + full_l2 - 1 - busy_expr,
                    f"C15_over_{roaster}_{slot}",
                )
            else:
                out_line = r_out[roaster][0]
                prob += (
                    v["over"][roaster][slot] >= v["full"][out_line][slot] - busy_expr,
                    f"C15_over_{roaster}_{slot}",
                )
            cc["C15"] += 1
    logger.info(
        "C14/C15 complete: C14=%d C15=%d in %.2fs",
        cc["C14"],
        cc["C15"],
        time.perf_counter() - t_phase,
    )

    t_phase = time.perf_counter()
    for pair in feasible_gc_pairs:
        pair_label = f"{pair[0]}_{pair[1]}"
        cap = d["gc_capacity"][pair]
        for start, rst_var in v["restock"][pair].items():
            prob += (
                v["gc"][pair][start] + RST_QTY <= cap + RST_QTY * (1 - rst_var),
                f"C19_{pair_label}_{start}",
            )
            cc["C19"] += 1
    logger.info("C19 complete: %d in %.2fs", cc["C19"], time.perf_counter() - t_phase)

    t_phase = time.perf_counter()
    for job_id in d["jobs"]:
        due = d["job_due"][job_id]
        for batch_id in mto_batches:
            if batch_id[0] != job_id:
                continue
            sku = batch_sku[batch_id]
            duration = roast_time[sku]
            completion_expr = pulp.lpSum(
                (start + duration) * x_var
                for roaster in v["x_ti"][batch_id]
                for start, x_var in v["x_ti"][batch_id][roaster].items()
            )
            prob += (
                v["tard"][job_id] >= completion_expr - due,
                f"C12_{job_id}_{safe_id(batch_id)}",
            )
            cc["C12"] += 1
    logger.info("C12 complete: %d in %.2fs", cc["C12"], time.perf_counter() - t_phase)

    total_vars = len(prob.variables())
    total_constraints = len(prob.constraints)
    elapsed = time.perf_counter() - t_build
    logger.info(
        "Model v5 built: %d variables, %d constraints in %.2fs",
        total_vars,
        total_constraints,
        elapsed,
    )

    prob._build_meta = {
        "var_counts": vc,
        "constraint_counts": cc,
        "total_vars": total_vars,
        "total_constraints": total_constraints,
        "build_seconds": elapsed,
        "formulation": "time-indexed v5 (deterministic RC+GC+restock)",
        "notes": [
            "No UPS variables are included in v5.",
            "SKU-specific roast durations are enforced.",
            "Setup cost proxy is carried over from v3/v4.",
            "Idle/overflow busy detection tracks roast occupancy, not explicit setup intervals.",
        ],
    }

    return prob, v


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="model.py",
        description="Build-only report for the MILP_Test_v5 model.",
    )
    parser.add_argument("--input-dir", default="Input_data", help="Path to the input CSV directory.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    data = load(input_dir=args.input_dir)
    problem, _vars = build(data)
    meta = problem._build_meta
    print("=== MODEL BUILD REPORT (v5) ===")
    print(f"Formulation       : {meta['formulation']}")
    print(f"Build time        : {meta['build_seconds']:.2f}s")
    print(f"Variables         : {meta['total_vars']}")
    print(f"Constraints       : {meta['total_constraints']}")
    print(f"Variable counts   : {meta['var_counts']}")
    print(f"Constraint counts : {meta['constraint_counts']}")
    for note in meta.get("notes", []):
        print(f"- {note}")
