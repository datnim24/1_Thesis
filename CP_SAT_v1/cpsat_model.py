"""Stage 1 CP-SAT model builder for the deterministic roasting benchmark."""

from __future__ import annotations

from ortools.sat.python import cp_model
import logging
import time


logger = logging.getLogger("cpsat_model")

_LAST_BUILD_STATS: dict | None = None


def safe_id(b) -> str:
    return str(b).replace(" ", "").replace("'", "").replace(",", "_")


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


def _latest_event_index_at_or_before(events: list[int], target_times: list[int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    idx = -1
    for tau in target_times:
        while idx + 1 < len(events) and events[idx + 1] <= tau:
            idx += 1
        mapping[tau] = idx
    return mapping


def _as_int_coefficient(value, label: str) -> int:
    numeric = float(value)
    rounded = round(numeric)
    if abs(numeric - rounded) > 1e-9:
        raise ValueError(f"CP-SAT objective coefficient '{label}' must be integral, got {value!r}")
    return int(rounded)


def _output_lines_for_roaster(d: dict, roaster_id: str) -> list[str]:
    if roaster_id == "R3" and not d["allow_r3_flex"]:
        return ["L2"]
    return list(d["roaster_can_output"][roaster_id])


def _output_lines_for_batch_roaster(d: dict, batch_id, roaster_id: str) -> list[str]:
    if roaster_id != "R3":
        return list(d["roaster_can_output"][roaster_id])
    if d["allow_r3_flex"] and batch_id[0] == "R3":
        return ["L1", "L2"]
    return ["L2"]


def _emit_build_report(stats: dict, cp_vars: dict) -> None:
    width = 58
    var_counts = stats["var_counts"]
    constraint_counts = stats["constraint_counts"]

    print("=== CP-SAT MODEL BUILD REPORT ===")
    print(f"R3 mode               : {stats['r3_mode']}")
    print("Variables:")
    print(f"  start[b]            : {var_counts.get('start', 0)}  (expected 164)")
    print(f"  active[b]           : {var_counts.get('active', 0)}  (expected 160, PSC only)")
    print(f"  assign[b][r]        : {var_counts.get('assign', 0)}")
    print(
        f"  y[b] / w[b]         : {var_counts.get('y', 0)} / {var_counts.get('w', 0)}"
        "  (expected ~32 / 0)"
    )
    print(f"  tard[j]             : {var_counts.get('tard', 0)}  (expected 2)")
    print(f"  roast_interval      : {var_counts.get('roast_interval', 0)}")
    print(f"  consume_interval    : {var_counts.get('consume_interval', 0)}")
    print(f"  n_completed         : {var_counts.get('n_completed', 0)}  (expected ~194)")
    print(f"  completed_by        : {var_counts.get('completed_by', 0)}  (expected ~15,000)")
    print(f"  rc                  : {var_counts.get('rc', 0)}  (expected ~194)")
    print(
        f"  low / full          : {var_counts.get('low', 0)} / {var_counts.get('full', 0)}"
    )
    print(
        f"  idle / over         : {var_counts.get('idle', 0)} / {var_counts.get('over', 0)}"
    )
    print(f"  busy_any            : {var_counts.get('busy_any', 0)}")
    print(
        f"  order (C4/C5)       : {var_counts.get('order', 0)}"
        "  (expected << MILP's 64,729)"
    )
    print(
        f"  Total CP-SAT vars   : {stats.get('total_vars', 0)}"
        "  (target << 501,568 MILP)"
    )
    print("Constraints:")
    print(f"  C2  PSC activation  : {constraint_counts.get('C2', 0)}")
    print(f"  C3  MTO assignment  : {constraint_counts.get('C3', 0)}")
    print(f"  C4  NoOverlap       : {constraint_counts.get('C4', 0)} calls  (expected 5)")
    print(f"  C5  Setup pairs     : {constraint_counts.get('C5', 0)}")
    print(f"  C6  Initial SKU     : {constraint_counts.get('C6', 0)}")
    print(f"  C7  Downtime        : {constraint_counts.get('C7', 0)}")
    print(f"  C8  Pipeline        : {constraint_counts.get('C8', 0)} calls  (expected 2)")
    print(f"  C10/C11 RC balance  : {constraint_counts.get('C10/C11', 0)}")
    print(f"  C12 Tardiness       : {constraint_counts.get('C12', 0)}")
    print(f"  C14 Safety-idle     : {constraint_counts.get('C14', 0)}")
    print(f"  C15 Overflow-idle   : {constraint_counts.get('C15', 0)}")
    print(
        f"  Total constraints   : {stats.get('total_constraints', 0)}"
        "  (target << 1,407,615 MILP)"
    )
    print("=== END BUILD REPORT ===")


def build(d: dict) -> tuple[cp_model.CpModel, dict]:
    global _LAST_BUILD_STATS

    build_start = time.perf_counter()
    r3_mode = "flex" if d["allow_r3_flex"] else "fixed"
    logger.info("Building CP-SAT model - R3 mode: %s", r3_mode)
    if d["allow_r3_flex"]:
        logger.warning(
            "Routing auxiliaries exist only for PSC pool batches whose ID starts with R3. "
            "Other PSC batches assigned to R3 contribute to L2 only in this build."
        )

    model = cp_model.CpModel()
    P = d["process_time"]
    DC = d["consume_time"]
    SL = d["shift_length"]
    MS = d["max_start"]
    ST = d["setup_time"]

    cp_vars = {
        "start": {},
        "active": {},
        "assign": {},
        "y": {},
        "w": {},
        "r3_to_l2": {},
        "tard": {},
        "roast_interval": {},
        "consume_interval": {},
        "n_completed": {},
        "completed_by": {},
        "rc": {},
        "low": {},
        "full": {},
        "idle": {},
        "over": {},
        "busy_any": {},
        "order": {},
    }

    var_counts = {
        "start": 0,
        "active": 0,
        "assign": 0,
        "y": 0,
        "w": 0,
        "r3_to_l2": 0,
        "tard": 0,
        "roast_interval": 0,
        "consume_interval": 0,
        "n_completed": 0,
        "completed_by": 0,
        "rc": 0,
        "low": 0,
        "full": 0,
        "idle": 0,
        "over": 0,
        "busy_any": 0,
        "order": 0,
    }
    constraint_counts = {
        "C2": 0,
        "C3": 0,
        "C4": 0,
        "C5": 0,
        "C6": 0,
        "C7": 0,
        "C8": 0,
        "C10/C11": 0,
        "C12": 0,
        "C14": 0,
        "C15": 0,
    }

    all_batches = d["all_batches"]
    psc_pool = d["psc_pool"]
    mto_batches = d["mto_batches"]
    lines = d["lines"]
    roasters = d["roasters"]
    batch_eligible = d["batch_eligible_roasters"]
    batch_sku = d["batch_sku"]
    roaster_line = d["roaster_line"]
    roaster_pipeline = d["roaster_pipeline"]
    consumption_events = d["consumption_events"]
    downtime_slots = d["downtime_slots"]

    eligible_on_roaster = {
        r: [b for b in all_batches if r in batch_eligible[b]]
        for r in roasters
    }
    active_idle_indices = {
        r: [
            (k, tau)
            for k, tau in enumerate(consumption_events[roaster_line[r]])
            if tau not in downtime_slots[r]
        ]
        for r in roasters
    }

    r3_flex_batches = []
    if d["allow_r3_flex"]:
        r3_flex_batches = [
            b for b in psc_pool if b[0] == "R3" and "R3" in batch_eligible[b]
        ]
    r3_flex_batch_set = set(r3_flex_batches)

    revenue_psc = _as_int_coefficient(d["sku_revenue"]["PSC"], "sku_revenue.PSC")
    revenue_by_batch = {
        b: _as_int_coefficient(d["sku_revenue"][batch_sku[b]], f"sku_revenue.{batch_sku[b]}")
        for b in mto_batches
    }
    cost_tardiness = _as_int_coefficient(d["cost_tardiness"], "cost_tardiness")
    cost_idle = _as_int_coefficient(d["cost_idle"], "cost_idle")
    cost_overflow = _as_int_coefficient(d["cost_overflow"], "cost_overflow")

    phase_start = time.perf_counter()

    for b in all_batches:
        cp_vars["start"][b] = model.NewIntVar(0, MS, f"start_{safe_id(b)}")

    for b in psc_pool:
        cp_vars["active"][b] = model.NewBoolVar(f"active_{safe_id(b)}")

    for b in all_batches:
        cp_vars["assign"][b] = {}
        for r in batch_eligible[b]:
            cp_vars["assign"][b][r] = model.NewBoolVar(f"assign_{safe_id(b)}_{r}")
    logger.debug("assign vars created: %d", sum(len(inner) for inner in cp_vars["assign"].values()))

    if d["allow_r3_flex"]:
        for b in r3_flex_batches:
            cp_vars["y"][b] = model.NewBoolVar(f"y_{safe_id(b)}")
            cp_vars["w"][b] = model.NewBoolVar(f"w_{safe_id(b)}")
            model.AddImplication(cp_vars["w"][b], cp_vars["assign"][b]["R3"])
            model.AddImplication(cp_vars["w"][b], cp_vars["y"][b])
            model.AddBoolOr(
                [
                    cp_vars["w"][b],
                    cp_vars["assign"][b]["R3"].Not(),
                    cp_vars["y"][b].Not(),
                ]
            )

            cp_vars["r3_to_l2"][b] = model.NewBoolVar(f"r3_to_l2_{safe_id(b)}")
            model.AddBoolAnd(
                [
                    cp_vars["assign"][b]["R3"],
                    cp_vars["y"][b].Not(),
                ]
            ).OnlyEnforceIf(cp_vars["r3_to_l2"][b])
            model.AddBoolOr(
                [
                    cp_vars["assign"][b]["R3"].Not(),
                    cp_vars["y"][b],
                ]
            ).OnlyEnforceIf(cp_vars["r3_to_l2"][b].Not())

    for j in d["jobs"]:
        cp_vars["tard"][j] = model.NewIntVar(0, SL, f"tard_{j}")

    roast_end = {}
    for b in all_batches:
        cp_vars["roast_interval"][b] = {}
        roast_end[b] = {}
        for r in batch_eligible[b]:
            end_var = model.NewIntVar(P, SL, f"end_{safe_id(b)}_{r}")
            model.Add(end_var == cp_vars["start"][b] + P).OnlyEnforceIf(cp_vars["assign"][b][r])
            roast_end[b][r] = end_var
            cp_vars["roast_interval"][b][r] = model.NewOptionalIntervalVar(
                cp_vars["start"][b],
                P,
                end_var,
                cp_vars["assign"][b][r],
                f"roast_{safe_id(b)}_{r}",
            )

    consume_end = {}
    for b in all_batches:
        cp_vars["consume_interval"][b] = {}
        consume_end[b] = {}
        for r in batch_eligible[b]:
            end_var = model.NewIntVar(DC, SL, f"cend_{safe_id(b)}_{r}")
            model.Add(end_var == cp_vars["start"][b] + DC).OnlyEnforceIf(cp_vars["assign"][b][r])
            consume_end[b][r] = end_var
            cp_vars["consume_interval"][b][r] = model.NewOptionalIntervalVar(
                cp_vars["start"][b],
                DC,
                end_var,
                cp_vars["assign"][b][r],
                f"consume_{safe_id(b)}_{r}",
            )

    for line_id in lines:
        cp_vars["n_completed"][line_id] = {}
        for k in range(len(consumption_events[line_id])):
            cp_vars["n_completed"][line_id][k] = model.NewIntVar(
                0,
                len(psc_pool),
                f"ncomp_{line_id}_{k}",
            )

    for b in psc_pool:
        for r in batch_eligible[b]:
            output_lines = _output_lines_for_batch_roaster(d, b, r)
            for line_id in output_lines:
                for k, tau_k in enumerate(consumption_events[line_id]):
                    if tau_k < P:
                        continue
                    cp_vars["completed_by"][(b, r, line_id, k)] = model.NewBoolVar(
                        f"cb_{safe_id(b)}_{r}_{line_id}_{k}"
                    )

    for line_id in lines:
        cp_vars["rc"][line_id] = {}
        cp_vars["low"][line_id] = {}
        cp_vars["full"][line_id] = {}
        for k in range(len(consumption_events[line_id])):
            cp_vars["rc"][line_id][k] = model.NewIntVar(0, d["max_rc"], f"rc_{line_id}_{k}")
            cp_vars["low"][line_id][k] = model.NewBoolVar(f"low_{line_id}_{k}")
            cp_vars["full"][line_id][k] = model.NewBoolVar(f"full_{line_id}_{k}")

    for r in roasters:
        cp_vars["idle"][r] = {}
        cp_vars["over"][r] = {}
        cp_vars["busy_any"][r] = {}
        for k, tau_k in active_idle_indices[r]:
            cp_vars["idle"][r][k] = model.NewBoolVar(f"idle_{r}_{k}")
            cp_vars["over"][r][k] = model.NewBoolVar(f"over_{r}_{k}")

    var_counts["start"] = len(cp_vars["start"])
    var_counts["active"] = len(cp_vars["active"])
    var_counts["assign"] = sum(len(inner) for inner in cp_vars["assign"].values())
    var_counts["y"] = len(cp_vars["y"])
    var_counts["w"] = len(cp_vars["w"])
    var_counts["r3_to_l2"] = len(cp_vars["r3_to_l2"])
    var_counts["tard"] = len(cp_vars["tard"])
    var_counts["roast_interval"] = _count_nested_vars(cp_vars["roast_interval"])
    var_counts["consume_interval"] = _count_nested_vars(cp_vars["consume_interval"])
    var_counts["n_completed"] = _count_nested_vars(cp_vars["n_completed"])
    var_counts["completed_by"] = len(cp_vars["completed_by"])
    var_counts["rc"] = _count_nested_vars(cp_vars["rc"])
    var_counts["low"] = _count_nested_vars(cp_vars["low"])
    var_counts["full"] = _count_nested_vars(cp_vars["full"])
    var_counts["idle"] = _count_nested_vars(cp_vars["idle"])
    var_counts["over"] = _count_nested_vars(cp_vars["over"])

    logger.debug("Variable counts after creation:")
    for key in (
        "start",
        "active",
        "assign",
        "y",
        "w",
        "r3_to_l2",
        "tard",
        "roast_interval",
        "consume_interval",
        "n_completed",
        "completed_by",
        "rc",
        "low",
        "full",
        "idle",
        "over",
    ):
        logger.debug("  %s: %d", key, var_counts[key])
    logger.info("Variable creation complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    for b in psc_pool:
        model.Add(
            sum(cp_vars["assign"][b][r] for r in batch_eligible[b]) == cp_vars["active"][b]
        )
        constraint_counts["C2"] += 1
    logger.debug("Constraint count C2: %d", constraint_counts["C2"])
    logger.info("C2 complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    for b in mto_batches:
        model.Add(sum(cp_vars["assign"][b][r] for r in batch_eligible[b]) == 1)
        constraint_counts["C3"] += 1
    logger.debug("Constraint count C3: %d", constraint_counts["C3"])
    logger.info("C3 complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    total_pair_count = 0
    for r in roasters:
        no_overlap_intervals = [cp_vars["roast_interval"][b][r] for b in eligible_on_roaster[r]]
        windows = _extract_windows(downtime_slots[r])
        for dstart, dend in windows:
            no_overlap_intervals.append(
                model.NewFixedSizeIntervalVar(
                    dstart,
                    dend - dstart + 1,
                    f"downtime_{r}_{dstart}_{dend}",
                )
            )
            constraint_counts["C7"] += 1
            logger.debug("C7: %s downtime interval added [%d, %d]", r, dstart, dend)

        model.AddNoOverlap(no_overlap_intervals)
        constraint_counts["C4"] += 1

        if len(d["roaster_eligible_skus"][r]) <= 1:
            continue

        pair_count = 0
        eligible_batches = eligible_on_roaster[r]
        for idx_1 in range(len(eligible_batches)):
            b1 = eligible_batches[idx_1]
            for idx_2 in range(idx_1 + 1, len(eligible_batches)):
                b2 = eligible_batches[idx_2]
                order_key = (b1, b2, r)
                cp_vars["order"][order_key] = model.NewBoolVar(
                    f"ord_{safe_id(b1)}_{safe_id(b2)}_{r}"
                )
                sg = ST if batch_sku[b1] != batch_sku[b2] else 0
                sgr = ST if batch_sku[b2] != batch_sku[b1] else 0
                model.Add(cp_vars["start"][b2] >= cp_vars["start"][b1] + P + sg).OnlyEnforceIf(
                    [
                        cp_vars["assign"][b1][r],
                        cp_vars["assign"][b2][r],
                        cp_vars["order"][order_key],
                    ]
                )
                model.Add(cp_vars["start"][b1] >= cp_vars["start"][b2] + P + sgr).OnlyEnforceIf(
                    [
                        cp_vars["assign"][b1][r],
                        cp_vars["assign"][b2][r],
                        cp_vars["order"][order_key].Not(),
                    ]
                )
                pair_count += 1
                constraint_counts["C5"] += 1

        total_pair_count += pair_count
        logger.debug("C4/C5 pair count for %s: %d", r, pair_count)

    if total_pair_count > 10000:
        logger.warning("C4/C5 pair count: %d - mixed-SKU roaster pair set is large.", total_pair_count)
    var_counts["order"] = len(cp_vars["order"])
    logger.debug("Constraint count C4: %d", constraint_counts["C4"])
    logger.debug("Constraint count C5: %d", constraint_counts["C5"])
    logger.debug("Constraint count C7: %d", constraint_counts["C7"])
    logger.info("C4/C5/C7 complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    # This is a conservative approximation: every non-PSC batch is forced to
    # start after setup time, which is always safe but can exclude some valid
    # schedules where PSC would have run first on the roaster.
    logger.warning(
        "C6 uses conservative first-batch-style constraint: all non-PSC batches start >= "
        "setup_time. Valid but slightly suboptimal if solver could schedule PSC first and "
        "start NDG/BUSTA at slot 0."
    )
    for b in all_batches:
        if batch_sku[b] == "PSC":
            continue
        for r in batch_eligible[b]:
            model.Add(cp_vars["start"][b] >= ST).OnlyEnforceIf(cp_vars["assign"][b][r])
            constraint_counts["C6"] += 1
    logger.debug("Constraint count C6: %d", constraint_counts["C6"])
    logger.info("C6 complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    for line_id in lines:
        pipe_intervals = [
            cp_vars["consume_interval"][b][r]
            for b in all_batches
            for r in batch_eligible[b]
            if roaster_pipeline[r] == line_id
        ]
        model.AddNoOverlap(pipe_intervals)
        constraint_counts["C8"] += 1
        logger.debug("C8 pipeline %s interval count: %d", line_id, len(pipe_intervals))
    logger.debug("Constraint count C8: %d", constraint_counts["C8"])
    logger.info("C8 complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    for line_id in lines:
        events = consumption_events[line_id]
        for k, tau_k in enumerate(events):
            events_so_far = k + 1
            model.Add(
                cp_vars["rc"][line_id][k]
                == d["rc_init"][line_id] + cp_vars["n_completed"][line_id][k] - events_so_far
            )
            model.Add(cp_vars["rc"][line_id][k] >= 0)
            model.Add(cp_vars["rc"][line_id][k] <= d["max_rc"])
            constraint_counts["C10/C11"] += 3

        for k in range(len(events)):
            all_cb_for_lk = [
                cp_vars["completed_by"][(b, r, line_id, k)]
                for b in psc_pool
                for r in batch_eligible[b]
                if (b, r, line_id, k) in cp_vars["completed_by"]
            ]
            model.Add(cp_vars["n_completed"][line_id][k] == sum(all_cb_for_lk))
            constraint_counts["C10/C11"] += 1

    completed_by_count = len(cp_vars["completed_by"])
    logger.debug("completed_by variable count: %d", completed_by_count)
    if completed_by_count > 20000:
        logger.warning(
            "completed_by count: %d - model build may be slow. Consider reducing PSC pool.",
            completed_by_count,
        )

    for (b, r, line_id, k), completed_var in cp_vars["completed_by"].items():
        tau_k = consumption_events[line_id][k]
        if r == "R3" and d["allow_r3_flex"]:
            if b in r3_flex_batch_set and line_id == "L1":
                presence_lit = cp_vars["w"][b]
            elif b in r3_flex_batch_set:
                presence_lit = cp_vars["r3_to_l2"][b]
            else:
                presence_lit = cp_vars["assign"][b][r]
        else:
            presence_lit = cp_vars["assign"][b][r]

        model.AddImplication(presence_lit.Not(), completed_var.Not())
        model.AddImplication(completed_var, presence_lit)
        model.Add(cp_vars["start"][b] + P <= tau_k).OnlyEnforceIf(completed_var)
        model.Add(cp_vars["start"][b] + P >= tau_k + 1).OnlyEnforceIf(
            [presence_lit, completed_var.Not()]
        )
        constraint_counts["C10/C11"] += 4
    logger.debug("Constraint count C10/C11: %d", constraint_counts["C10/C11"])
    logger.info("C10/C11 complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    for j in d["jobs"]:
        for b in mto_batches:
            if b[0] != j:
                continue
            model.Add(cp_vars["tard"][j] >= cp_vars["start"][b] + P - d["job_due"][j])
            constraint_counts["C12"] += 1
    logger.debug("Constraint count C12: %d", constraint_counts["C12"])
    logger.info("C12 complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    # busy_any tracks assignment existence rather than actual time coverage.
    # This matches the MILP approximation and keeps the change-point model small.
    logger.warning(
        "C14 uses assignment-count busy approximation. idle[r][k] may undercount idle periods "
        "when roaster has batches assigned but none covering event time tau_k. Same approximation "
        "as MILP."
    )
    for line_id in lines:
        for k, tau_k in enumerate(consumption_events[line_id]):
            model.Add(cp_vars["rc"][line_id][k] <= d["safety_stock"] - 1).OnlyEnforceIf(
                cp_vars["low"][line_id][k]
            )
            model.Add(cp_vars["rc"][line_id][k] >= d["safety_stock"]).OnlyEnforceIf(
                cp_vars["low"][line_id][k].Not()
            )
            constraint_counts["C14"] += 2

        for r in roasters:
            if roaster_line[r] != line_id:
                continue
            roaster_batches = eligible_on_roaster[r]
            for k, tau_k in active_idle_indices[r]:
                cp_vars["busy_any"][r][k] = model.NewBoolVar(f"busy_{r}_{k}")
                model.AddBoolOr([cp_vars["assign"][b][r] for b in roaster_batches]).OnlyEnforceIf(
                    cp_vars["busy_any"][r][k]
                )
                model.AddBoolAnd(
                    [cp_vars["assign"][b][r].Not() for b in roaster_batches]
                ).OnlyEnforceIf(cp_vars["busy_any"][r][k].Not())
                model.AddBoolAnd(
                    [cp_vars["low"][line_id][k], cp_vars["busy_any"][r][k].Not()]
                ).OnlyEnforceIf(cp_vars["idle"][r][k])
                model.AddBoolOr(
                    [cp_vars["low"][line_id][k].Not(), cp_vars["busy_any"][r][k]]
                ).OnlyEnforceIf(cp_vars["idle"][r][k].Not())
                constraint_counts["C14"] += 4

    var_counts["busy_any"] = _count_nested_vars(cp_vars["busy_any"])
    logger.debug("busy_any variable count: %d", var_counts["busy_any"])
    logger.debug("Constraint count C14: %d", constraint_counts["C14"])
    logger.info("C14 complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    for line_id in lines:
        for k, tau_k in enumerate(consumption_events[line_id]):
            model.Add(cp_vars["rc"][line_id][k] >= d["max_rc"]).OnlyEnforceIf(
                cp_vars["full"][line_id][k]
            )
            model.Add(cp_vars["rc"][line_id][k] <= d["max_rc"] - 1).OnlyEnforceIf(
                cp_vars["full"][line_id][k].Not()
            )
            constraint_counts["C15"] += 2

    l2_to_l1_latest = {}
    if d["allow_r3_flex"]:
        l2_to_l1_latest = _latest_event_index_at_or_before(
            consumption_events["L1"],
            consumption_events["L2"],
        )
        logger.warning(
            "C15 maps each R3/L2 event to the latest available L1 event index at or before the "
            "same time. This is a change-point approximation forced by per-line event indexing."
        )

    for r in roasters:
        home_line = roaster_line[r]
        for k, tau_k in active_idle_indices[r]:
            if r == "R3" and d["allow_r3_flex"]:
                l1_idx = l2_to_l1_latest.get(tau_k, -1)
                if l1_idx < 0:
                    if d["rc_init"]["L1"] == d["max_rc"]:
                        model.AddBoolAnd(
                            [cp_vars["full"]["L2"][k], cp_vars["busy_any"][r][k].Not()]
                        ).OnlyEnforceIf(cp_vars["over"][r][k])
                        model.AddBoolOr(
                            [cp_vars["full"]["L2"][k].Not(), cp_vars["busy_any"][r][k]]
                        ).OnlyEnforceIf(cp_vars["over"][r][k].Not())
                        constraint_counts["C15"] += 2
                    else:
                        model.Add(cp_vars["over"][r][k] == 0)
                        constraint_counts["C15"] += 1
                else:
                    model.AddBoolAnd(
                        [
                            cp_vars["full"]["L1"][l1_idx],
                            cp_vars["full"]["L2"][k],
                            cp_vars["busy_any"][r][k].Not(),
                        ]
                    ).OnlyEnforceIf(cp_vars["over"][r][k])
                    model.AddBoolOr(
                        [
                            cp_vars["full"]["L1"][l1_idx].Not(),
                            cp_vars["full"]["L2"][k].Not(),
                            cp_vars["busy_any"][r][k],
                        ]
                    ).OnlyEnforceIf(cp_vars["over"][r][k].Not())
                    constraint_counts["C15"] += 2
            else:
                output_line = _output_lines_for_roaster(d, r)[0]
                model.AddBoolAnd(
                    [cp_vars["full"][output_line][k], cp_vars["busy_any"][r][k].Not()]
                ).OnlyEnforceIf(cp_vars["over"][r][k])
                model.AddBoolOr(
                    [cp_vars["full"][output_line][k].Not(), cp_vars["busy_any"][r][k]]
                ).OnlyEnforceIf(cp_vars["over"][r][k].Not())
                constraint_counts["C15"] += 2
    logger.debug("Constraint count C15: %d", constraint_counts["C15"])
    logger.info("C15 complete in %.2fs", time.perf_counter() - phase_start)

    phase_start = time.perf_counter()
    psc_revenue_terms = [revenue_psc * cp_vars["active"][b] for b in psc_pool]
    mto_revenue_const = sum(revenue_by_batch[b] for b in mto_batches)
    tard_terms = [cost_tardiness * cp_vars["tard"][j] for j in d["jobs"]]
    idle_terms = [
        cost_idle * cp_vars["idle"][r][k]
        for r in roasters
        for k, tau_k in active_idle_indices[r]
    ]
    over_terms = [
        cost_overflow * cp_vars["over"][r][k]
        for r in roasters
        for k, tau_k in active_idle_indices[r]
    ]

    model.Maximize(
        cp_model.LinearExpr.Sum(psc_revenue_terms)
        + mto_revenue_const
        - cp_model.LinearExpr.Sum(tard_terms)
        - cp_model.LinearExpr.Sum(idle_terms)
        - cp_model.LinearExpr.Sum(over_terms)
    )
    logger.info("Objective complete in %.2fs", time.perf_counter() - phase_start)

    total_vars = len(model.Proto().variables)
    total_constraints = len(model.Proto().constraints)
    logger.info(
        "CP-SAT model built: %d constraints, %d variables",
        total_constraints,
        total_vars,
    )

    _LAST_BUILD_STATS = {
        "r3_mode": r3_mode,
        "var_counts": var_counts,
        "constraint_counts": constraint_counts,
        "total_vars": total_vars,
        "total_constraints": total_constraints,
        "build_time_sec": round(time.perf_counter() - build_start, 3),
    }

    return model, cp_vars


if __name__ == "__main__":
    import sys

    from data import load

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dict = load(input_dir="Input_data")
    _, vars_dict = build(data_dict)
    _emit_build_report(_LAST_BUILD_STATS or {}, vars_dict)
