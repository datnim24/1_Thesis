"""
MILP.py — Time-indexed MILP formulation for the Nestlé roasting scheduling problem.

Uses PuLP (CBC Solver) to build and solve the deterministic model.
Supports re-solve from a disruption time with updated state.
"""

import time as _time
import math
from typing import Dict, List, Optional, Tuple, Any

from data_loader import ShiftData, Roaster, Job, SKU
from gui_state import SolveResult, BatchResult

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("WARNING: pulp not installed. Install with: pip install pulp")


# ──────────────────────────────────────────────
# Batch definition helpers
# ──────────────────────────────────────────────

class BatchDef:
    """Definition of a batch (MTO or PSC pool) before solving."""
    def __init__(self, batch_id: str, sku: str, job_id: Optional[str],
                 is_mto: bool, eligible_roasters: List[str]):
        self.batch_id = batch_id
        self.sku = sku
        self.job_id = job_id
        self.is_mto = is_mto
        self.eligible_roasters = eligible_roasters


def _create_batches(data: ShiftData, t_start: int = 0) -> List[BatchDef]:
    """Create batch definitions for the MILP model."""
    batches = []

    # MTO batches (always active)
    for job in data.jobs:
        for i in range(1, job.required_batches + 1):
            bid = f"MTO_{job.job_id}_b{i}"
            eligible = data.eligible_roasters_by_sku.get(job.sku, [])
            batches.append(BatchDef(bid, job.sku, job.job_id, True, eligible))

    # PSC pool batches (optional, solver decides activation)
    remaining_time = data.shift_length - t_start
    active_roasters = [r for r in data.roasters.values() if r.is_active]
    if not active_roasters:
        return batches

    # Max possible batches across all roasters
    max_batches = sum(remaining_time // r.process_time for r in active_roasters)
    # Give a bit of buffer
    pool_size = max_batches

    eligible_psc = data.eligible_roasters_by_sku.get("PSC", [])
    for i in range(1, pool_size + 1):
        bid = f"PSC_pool_b{i}"
        batches.append(BatchDef(bid, "PSC", None, False, eligible_psc))

    return batches


# ──────────────────────────────────────────────
# Core solver
# ──────────────────────────────────────────────

def solve_milp(
    data: ShiftData,
    t_start: int = 0,
    initial_rc: Optional[Dict[str, int]] = None,
    frozen_batches: Optional[List[Dict]] = None,
    disruptions: Optional[List[Dict]] = None,
    reactive_mode: bool = False,
    time_limit_sec: float = 30.0,
    mip_gap: float = 0.01,
) -> SolveResult:
    """
    Build and solve the MILP model using PuLP.
    """
    if not PULP_AVAILABLE:
        result = SolveResult(status="error", solver_log="pulp not installed")
        return result

    result = SolveResult()
    result.solve_mode = "reactive" if reactive_mode else "deterministic"

    # Extract parameters
    T = data.shift_length                          # 480
    P = data.roasters[list(data.roasters.keys())[0]].process_time  # 15
    delta = data.roasters[list(data.roasters.keys())[0]].consume_time  # 3
    sigma = data.setup_time                        # 5
    max_rc = data.max_rc_per_line                  # 40
    safety = data.safety_stock                     # 20
    max_start = T - P                              # 465

    # Time range for this solve
    time_step = data.time_step
    time_range = list(range(t_start, max_start + 1, time_step))

    # RC initial values
    rc_init = {
        "L1": initial_rc["L1"] if initial_rc else data.initial_rc_l1,
        "L2": initial_rc["L2"] if initial_rc else data.initial_rc_l2,
    }

    # Merge planned downtime + disruptions into downtime slots
    dt_slots = {}
    for rid in data.roasters:
        dt_slots[rid] = set(data.downtime_slots.get(rid, set()))
    if disruptions:
        for d in disruptions:
            rid = d["roaster_id"]
            t0 = d["time_min"]
            dur = d["duration_min"]
            for tt in range(t0, min(t0 + dur, T)):
                dt_slots.setdefault(rid, set()).add(tt)

    # Consumption schedules from t_start onward
    cons_events = {
        "L1": [t for t in data.consumption_schedule_l1 if t >= t_start],
        "L2": [t for t in data.consumption_schedule_l2 if t >= t_start],
    }

    # Create batches
    all_batches = _create_batches(data, t_start)
    mto_batches = [b for b in all_batches if b.is_mto]
    psc_batches = [b for b in all_batches if not b.is_mto]

    # Get roaster and line info
    roaster_ids = [r.roaster_id for r in data.roasters.values() if r.is_active]

    # Revenue map
    rev_map = {sku.name: sku.revenue for sku in data.skus.values()}

    # R3 flexible output?
    r3_flexible = data.solver_config.get("allow_r3_flexible_output", "1") == "1"

    # ─── BUILD MODEL ───

    mdl = pulp.LpProblem("NestleRoastingMILP", pulp.LpMaximize)

    # Solver parameters
    solver_tl = float(data.solver_config.get("time_limit_sec", str(time_limit_sec)))
    solver_gap = float(data.solver_config.get("mip_gap_target", str(mip_gap)))

    # ─── DECISION VARIABLES ───

    # Group variables by batch and line for faster constraint generation
    x_by_batch = {}
    for b in all_batches:
        x_by_batch[b.batch_id] = []

    # x[b, r, t] = 1 if batch b starts on roaster r at time t
    x = {}
    for b in all_batches:
        for r_id in b.eligible_roasters:
            roaster = data.roasters[r_id]
            for t in time_range:
                # C7: Skip if batch would overlap with downtime
                batch_interval = set(range(t, t + P))
                if batch_interval & dt_slots.get(r_id, set()):
                    continue
                # C9: End-of-shift
                if t + P > T:
                    continue
                var = pulp.LpVariable(name=f"x_{b.batch_id}_{r_id}_{t}", cat='Binary')
                x[b.batch_id, r_id, t] = var
                x_by_batch[b.batch_id].append(var)

    # a[b] = 1 if PSC batch b is activated
    a = {}
    for b in psc_batches:
        a[b.batch_id] = pulp.LpVariable(name=f"a_{b.batch_id}", cat='Binary')

    # Tardiness variables for MTO jobs
    tard = {}
    jobs_map = {j.job_id: j for j in data.jobs}
    for j in data.jobs:
        tard[j.job_id] = pulp.LpVariable(name=f"tard_{j.job_id}", lowBound=0, cat='Continuous')

    # R3 routing: y[b, t] = 1 means R3 batch outputs to L1
    y = {}
    if r3_flexible and "R3" in data.roasters:
        for b in all_batches:
            if "R3" in b.eligible_roasters and b.sku == "PSC":
                for t in time_range:
                    if (b.batch_id, "R3", t) in x:
                        y[b.batch_id, t] = pulp.LpVariable(name=f"y_{b.batch_id}_{t}", cat='Binary')

    # RC stock tracking at key time points
    change_points = {"L1": set(), "L2": set()}
    for line in ["L1", "L2"]:
        change_points[line].update(cons_events[line])
    # Add batch completion times
    for t in time_range:
        completion = t + P
        if completion <= T:
            change_points["L1"].add(completion)
            change_points["L2"].add(completion)

    rc_var = {}
    for line in ["L1", "L2"]:
        for t in sorted(change_points[line]):
            lb = -T if reactive_mode else 0  # soft stockout in reactive
            rc_var[line, t] = pulp.LpVariable(
                name=f"rc_{line}_{t}", lowBound=lb, upBound=max_rc, cat='Integer'
            )

    # Stockout variables (reactive mode only)
    stockout_var = {}
    if reactive_mode:
        for line in ["L1", "L2"]:
            for t in cons_events[line]:
                stockout_var[line, t] = pulp.LpVariable(name=f"stockout_{line}_{t}", lowBound=0, cat='Continuous')

    # ─── CONSTRAINTS ───

    # C1: MTO batches must be assigned exactly once
    for b in mto_batches:
        mdl += pulp.lpSum(x_by_batch[b.batch_id]) == 1, f"C1_mto_{b.batch_id}"

    # C2: PSC batches: activation links to assignment
    for b in psc_batches:
        x_sum = pulp.lpSum(x_by_batch[b.batch_id])
        mdl += x_sum == a[b.batch_id], f"C2_psc_{b.batch_id}"

    # C4: Roaster NoOverlap
    print("  Generating C4: Roaster NoOverlap...")
    r_t_vars = {r: {t: [] for t in range(t_start, T)} for r in roaster_ids}
    for (b_id, r_id, t_start_var), v in x.items():
        for t_active in range(t_start_var, min(t_start_var + P, T)):
            if t_active >= t_start:
                r_t_vars[r_id][t_active].append(v)
                
    for r_id in roaster_ids:
        for t in range(t_start, T):
            overlapping = r_t_vars[r_id].get(t, [])
            if len(overlapping) > 1:
                mdl += pulp.lpSum(overlapping) <= 1, f"C4_nooverlap_{r_id}_{t}"

    # C5 + C6: Setup time
    print("  Generating C5: Setup time...")
    batches_by_sku = {}
    for b in all_batches:
        batches_by_sku.setdefault(b.sku, []).append(b)

    sku_r_t_vars = {}
    for (b_id, r_id, t), v in x.items():
        b_sku = next(bb.sku for bb in all_batches if bb.batch_id == b_id)
        sku_r_t_vars.setdefault((b_sku, r_id, t), []).append(v)

    for r_id in roaster_ids:
        for b_sku in batches_by_sku:
            for t in time_range:
                vars_k = sku_r_t_vars.get((b_sku, r_id, t), [])
                if not vars_k:
                    continue
                    
                vars_other_skus = []
                for other_sku in batches_by_sku:
                    if other_sku == b_sku:
                        continue
                    for t2 in range(t + P, min(t + P + sigma, max_start + 1)):
                        vars_other_skus.extend(sku_r_t_vars.get((other_sku, r_id, t2), []))
                        
                if vars_other_skus and vars_k:
                    mdl += pulp.lpSum(vars_k) + pulp.lpSum(vars_other_skus) <= 1, f"C5_setup_{r_id}_{b_sku}_{t}"

    # C6: Initial SKU setup
    for r_id in roaster_ids:
        roaster = data.roasters[r_id]
        init_sku = roaster.initial_last_sku
        for b in all_batches:
            if r_id not in b.eligible_roasters:
                continue
            if b.sku == init_sku:
                continue
            for t in range(t_start, min(t_start + sigma, max_start + 1)):
                key = (b.batch_id, r_id, t)
                if key in x:
                    mdl += x[key] == 0, f"C6_init_setup_{b.batch_id}_{r_id}_{t}"

    # C8: Pipeline NoOverlap
    print("  Generating C8: Pipeline NoOverlap...")
    for line in ["L1", "L2"]:
        pipeline_roasters = data.roaster_ids_by_pipeline.get(line, [])
        pipe_t_vars = {t: [] for t in range(t_start, T)}
        for (b_id, r_id, t_var), v in x.items():
            if r_id in pipeline_roasters:
                for t_active in range(t_var, min(t_var + delta, T)):
                    pipe_t_vars[t_active].append(v)
                    
        for t in range(t_start, T):
            overlapping = pipe_t_vars[t]
            if len(overlapping) > 1:
                mdl += pulp.lpSum(overlapping) <= 1, f"C8_pipeline_{line}_{t}"

    # C10/C11: RC inventory tracking
    print("  Generating C10: RC tracking...")
    psc_completions = {"L1": {t: [] for t in time_range}, "L2": {t: [] for t in time_range}}
    
    for (b_id, r_id, t_var), v in x.items():
        b = next(bb for bb in all_batches if bb.batch_id == b_id)
        if b.sku != "PSC":
            continue
            
        completion = t_var + P
        roaster = data.roasters[r_id]
        
        if completion > T:
            continue
            
        if r_id == "R3" and r3_flexible:
            if (b_id, t_var) in y:
                # PuLP nonlinear restriction workaround:
                # Since v and y are binary, v * y is nonlinear in PuLP unless modeled.
                # However, y implies x! y is only 1 if x is 1. (See C13)
                # Therefore, instead of x * y, we can just use y directly!
                # Since y <= x, and y means L1 output, the contribution to L1 is y.
                # The contribution to L2 is x - y.
                psc_completions["L1"][t_var].append(y[b_id, t_var])
                psc_completions["L2"][t_var].append(v - y[b_id, t_var])
        elif r_id == "R3" and not r3_flexible:
            psc_completions["L2"][t_var].append(v)
        else:
            line_id = roaster.line_id
            psc_completions[line_id][t_var].append(v)

    for line in ["L1", "L2"]:
        sorted_cps = sorted(change_points[line])
        tracked_terms = []
        last_cp = -1
        
        for cp_t in sorted_cps:
            # 1. Accumulate consumption up to cp_t
            if line == "L1":
                cons_count = sum(1 for ev in data.consumption_schedule_l1 if t_start < ev <= cp_t)
            else:
                cons_count = sum(1 for ev in data.consumption_schedule_l2 if t_start < ev <= cp_t)

            # 2. Add production completions that finish exactly in (last_cp, cp_t]
            for t_var in time_range:
                comp_time = t_var + P
                if last_cp < comp_time <= cp_t:
                    if t_var in psc_completions[line]:
                        for term in psc_completions[line][t_var]:
                            tracked_terms.append(term)
            
            last_cp = cp_t

            psc_sum = pulp.lpSum(tracked_terms) if tracked_terms else 0
            mdl += rc_var[line, cp_t] == rc_init[line] + psc_sum - cons_count, f"C10_rc_{line}_{cp_t}"

            if reactive_mode and (line, cp_t) in stockout_var:
                mdl += stockout_var[line, cp_t] >= -rc_var[line, cp_t], f"C10_stockout_{line}_{cp_t}"

    # C12: MTO tardiness
    for j in data.jobs:
        job_batches = [b for b in mto_batches if b.job_id == j.job_id]
        for b in job_batches:
            for r_id in b.eligible_roasters:
                for t in time_range:
                    key = (b.batch_id, r_id, t)
                    if key in x:
                        completion = t + P
                        if completion > j.due_time:
                            lateness = completion - j.due_time
                            mdl += tard[j.job_id] >= lateness * x[key], f"C12_tard_{j.job_id}_{b.batch_id}_{r_id}_{t}"

    # C13: R3 routing
    if r3_flexible and "R3" in data.roasters:
        for b in all_batches:
            if b.sku != "PSC" or "R3" not in b.eligible_roasters:
                continue
            for t in time_range:
                key = (b.batch_id, "R3", t)
                if key in x and (b.batch_id, t) in y:
                    mdl += y[b.batch_id, t] <= x[key], f"C13_y_link_{b.batch_id}_{t}"
                    r3 = data.roasters["R3"]
                    if not r3.can_output_l1:
                        mdl += y[b.batch_id, t] == 0, f"C13_no_l1_{b.batch_id}_{t}"

    # ─── OBJECTIVE ───

    revenue_terms = []
    for b in all_batches:
        rev = rev_map.get(b.sku, 0)
        if b.is_mto:
            revenue_terms.append(rev)
        else:
            revenue_terms.append(rev * a[b.batch_id])

    revenue_expr = pulp.lpSum(revenue_terms)
    tard_cost_expr = data.tardiness_cost * pulp.lpSum(tard.values())
    stockout_cost_expr = data.stockout_cost * pulp.lpSum(stockout_var.values()) if reactive_mode and stockout_var else 0

    mdl += revenue_expr - tard_cost_expr - stockout_cost_expr

    # ─── SOLVE ───

    result.num_variables = len(mdl.variables())
    result.num_constraints = len(mdl.constraints)

    print(f"MILP Model: {result.num_variables} variables, {result.num_constraints} constraints")
    print(f"  MTO batches: {len(mto_batches)}, PSC pool: {len(psc_batches)}")
    print(f"  Time range: [{t_start}, {max_start}]")
    print(f"  Solving with time limit {solver_tl}s, gap {solver_gap}...")

    wall_start = _time.time()
    solver = pulp.PULP_CBC_CMD(timeLimit=solver_tl, gapRel=solver_gap, msg=1)
    status_code = mdl.solve(solver)
    wall_end = _time.time()

    result.solve_time_sec = wall_end - wall_start
    result.status = pulp.LpStatus[mdl.status]

    if status_code != pulp.LpStatusOptimal or result.status.lower() == "infeasible":
        result.status = "infeasible"
        result.solver_log = "No feasible solution found."
        print("MILP: No solution found.")
        return result

    result.mip_gap = 0.0
    result.objective_value = pulp.value(mdl.objective)
    result.best_bound = result.objective_value

    # ─── EXTRACT SOLUTION ───

    result.batches = []
    batch_assignments = [] 

    for b in all_batches:
        assigned = False
        for r_id in b.eligible_roasters:
            for t_val in time_range:
                key = (b.batch_id, r_id, t_val)
                if key in x and x[key].varValue and x[key].varValue > 0.5:
                    roaster = data.roasters[r_id]
                    if r_id == "R3" and r3_flexible and (b.batch_id, t_val) in y:
                        out_l1 = y[b.batch_id, t_val].varValue and y[b.batch_id, t_val].varValue > 0.5
                        output_line = "L1" if out_l1 else "L2"
                    elif r_id == "R3":
                        output_line = "L2"
                    else:
                        output_line = roaster.line_id

                    batch_assignments.append({
                        "batch_id": b.batch_id,
                        "sku": b.sku,
                        "roaster_id": r_id,
                        "start_time": t_val,
                        "job_id": b.job_id,
                        "output_line": output_line,
                    })
                    assigned = True
                    break
            if assigned:
                break

    batch_assignments.sort(key=lambda ba: (ba["roaster_id"], ba["start_time"]))

    last_sku_on_roaster = {r_id: data.roasters[r_id].initial_last_sku for r_id in roaster_ids}

    for ba in batch_assignments:
        prev_sku = last_sku_on_roaster.get(ba["roaster_id"], "PSC")
        setup = sigma if prev_sku != ba["sku"] else 0
        last_sku_on_roaster[ba["roaster_id"]] = ba["sku"]

        result.batches.append(BatchResult(
            batch_id=ba["batch_id"],
            job_id=ba["job_id"],
            sku=ba["sku"],
            roaster_id=ba["roaster_id"],
            start_time=ba["start_time"],
            end_time=ba["start_time"] + P,
            consume_start=ba["start_time"],
            consume_end=ba["start_time"] + delta,
            output_line=ba["output_line"],
            setup_before=setup,
            revenue=rev_map.get(ba["sku"], 0),
        ))

    # ─── COMPUTE KPIs ───

    psc_done = sum(1 for b in result.batches if b.sku == "PSC")
    mto_done = sum(1 for b in result.batches if b.sku != "PSC")
    result.psc_batches_completed = psc_done
    result.mto_batches_completed = mto_done
    result.total_revenue = sum(b.revenue for b in result.batches)

    for j in data.jobs:
        tard_val = tard[j.job_id].varValue or 0.0
        result.job_tardiness[j.job_id] = tard_val
    result.total_tardiness_min = sum(result.job_tardiness.values())
    result.tardiness_cost = data.tardiness_cost * result.total_tardiness_min

    if reactive_mode:
        total_so = sum((v.varValue or 0.0) for v in stockout_var.values())
        result.stockout_minutes = int(total_so)
        result.stockout_cost = data.stockout_cost * total_so

    # ─── COMPUTE RC TRAJECTORY ───

    result.rc_trajectory = _compute_rc_trajectory(data, result.batches, t_start, rc_init)

    # ─── POST-SOLVE: Idle/Overflow penalties ───

    _compute_idle_penalties(data, result, t_start, dt_slots)

    result.net_profit = (
        result.total_revenue
        - result.tardiness_cost
        - result.stockout_cost
        - result.idle_cost
        - result.overflow_idle_cost
    )

    for r_id in roaster_ids:
        busy_slots = sum(b.end_time - b.start_time for b in result.batches if b.roaster_id == r_id)
        available = T - t_start - len(dt_slots.get(r_id, set()))
        result.utilization[r_id] = busy_slots / max(available, 1)

    if disruptions:
        result.disruptions_applied = disruptions

    result.solver_log = (
        f"Status: {result.status}\n"
        f"Solve time: {result.solve_time_sec:.2f}s\n"
        f"MIP gap: {result.mip_gap*100:.2f}%\n"
        f"Objective: ${result.objective_value:,.0f}\n"
        f"Variables: {result.num_variables}\n"
        f"Constraints: {result.num_constraints}\n"
        f"PSC batches: {psc_done}\n"
        f"MTO batches: {mto_done}\n"
    )

    print(f"\n=== MILP Solution ===")
    print(f"Status:     {result.status}")
    print(f"Objective:  ${result.objective_value:,.0f}")
    print(f"Gap:        {result.mip_gap*100:.2f}%")
    print(f"Solve time: {result.solve_time_sec:.2f}s")
    print(f"PSC done:   {psc_done}  |  MTO done: {mto_done}")
    print(f"Tardiness:  {result.total_tardiness_min:.0f} min (${result.tardiness_cost:,.0f})")
    print(f"Revenue:    ${result.total_revenue:,.0f}")
    print(f"Net profit: ${result.net_profit:,.0f}")

    return result


# ──────────────────────────────────────────────
# RC Trajectory computation
# ──────────────────────────────────────────────

def _compute_rc_trajectory(
    data: ShiftData,
    batches: List[BatchResult],
    t_start: int,
    rc_init: Dict[str, int],
) -> Dict[str, List[tuple]]:
    """Compute RC stock level at every minute for each line."""
    T = data.shift_length
    trajectory = {}

    for line in ["L1", "L2"]:
        stock = rc_init[line]
        cons_set = set(
            data.consumption_schedule_l1 if line == "L1"
            else data.consumption_schedule_l2
        )
        completions = {}
        for b in batches:
            if b.sku == "PSC" and b.output_line == line:
                completions.setdefault(b.end_time, 0)
                completions[b.end_time] += 1

        points = [(t_start, stock)]
        for t in range(t_start + 1, T + 1):
            stock += completions.get(t, 0)
            if t in cons_set:
                stock -= 1
            points.append((t, stock))

        trajectory[line] = points

    return trajectory


# ──────────────────────────────────────────────
# Post-solve idle penalty computation
# ──────────────────────────────────────────────

def _compute_idle_penalties(
    data: ShiftData,
    result: SolveResult,
    t_start: int,
    dt_slots: Dict[str, set],
):
    """Compute C14 (safety-idle) and C15 (overflow-idle) penalties post-solve."""
    T = data.shift_length
    roaster_ids = [r for r in data.roasters if data.roasters[r].is_active]

    busy = {r: set() for r in roaster_ids}
    for b in result.batches:
        for t in range(b.start_time, b.end_time):
            busy[b.roaster_id].add(t)
        if b.setup_before > 0:
            for t in range(b.start_time - b.setup_before, b.start_time):
                pass

    rc_at = {"L1": {}, "L2": {}}
    for line in ["L1", "L2"]:
        for t, lvl in result.rc_trajectory.get(line, []):
            rc_at[line][t] = lvl

    idle_min = 0
    over_min = 0

    for r_id in roaster_ids:
        roaster = data.roasters[r_id]
        for t in range(t_start, T):
            if t in dt_slots.get(r_id, set()):
                continue
            if t in busy[r_id]:
                continue
            line = roaster.line_id
            rc_level = rc_at.get(line, {}).get(t, data.max_rc_per_line)
            if rc_level < data.safety_stock:
                idle_min += 1
            if rc_level >= data.max_rc_per_line:
                over_min += 1

    result.idle_minutes = idle_min
    result.idle_cost = data.idle_cost * idle_min
    result.overflow_idle_minutes = over_min
    result.overflow_idle_cost = data.overflow_idle_cost * over_min


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_all_data

    data = load_all_data("../Input_data_sample")
    result = solve_milp(data, reactive_mode=True)

    print("\n=== Schedule ===")
    for b in result.batches[:20]:
        print(f"  {b.batch_id:20s}  {b.sku:6s}  {b.roaster_id}  "
              f"t=[{b.start_time},{b.end_time})  out={b.output_line}  "
              f"setup={b.setup_before}  rev=${b.revenue:,.0f}")
    if len(result.batches) > 20:
        print(f"  ... and {len(result.batches)-20} more batches")
