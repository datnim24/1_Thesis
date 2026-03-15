"""
Auto-mode dispatching heuristic for the coffee roasting simulation.
Improved rule-based scheduler with smarter MTO timing, pipeline coordination,
stockout prevention, look-ahead logic, and decision logging.
"""

from simulation.roaster import (
    RoasterState, ROAST_TIME, SETUP_TIME, CONSUME_TIME,
    SKU_PSC, SKU_NDG, SKU_BUSTA, MAX_SHIFT_TIME
)


# Urgency threshold: below this, prefer PSC; above, switch to MTO
MTO_URGENCY_THRESHOLD = 0.6

# RC stock threshold: if stock drops below this, all eligible roasters focus PSC
RC_CRITICAL_THRESHOLD = 5
RC_LOW_THRESHOLD = 10


def auto_dispatch(engine, current_time):
    """
    Improved auto-dispatch: for each IDLE roaster, decide what to produce.
    Processes roasters in priority order based on need.
    Returns a list of decision log entries.
    """
    decisions = []

    # Sort roasters: prioritize roasters whose line has lower RC stock
    idle_roasters = []
    for rid in ["R1", "R2", "R3", "R4", "R5"]:
        roaster = engine.roasters[rid]
        if roaster.state != RoasterState.IDLE and roaster.pending_sku is None:
            continue
        if roaster.state != RoasterState.IDLE:
            continue
        if roaster.pending_sku:
            continue
        idle_roasters.append(rid)

    # Sort: roasters on low-stock lines first, then by pipeline availability
    def roaster_priority(rid):
        r = engine.roasters[rid]
        line_stock = engine.silos[r.line].stock
        pipe_busy = engine.pipeline_busy[r.pipeline]
        return (pipe_busy, line_stock)

    idle_roasters.sort(key=roaster_priority)

    for rid in idle_roasters:
        roaster = engine.roasters[rid]
        action, reason = _decide(engine, roaster, current_time)
        if action is None:
            decisions.append({
                "time": current_time,
                "roaster": rid,
                "action": "WAIT",
                "reason": reason or "No viable action",
            })
            continue
        sku, target_line = action
        result = engine.schedule_batch(rid, sku, target_line)
        if "error" not in result:
            decisions.append({
                "time": current_time,
                "roaster": rid,
                "action": f"Schedule {sku}" + (f" → {target_line}" if target_line else ""),
                "reason": reason,
            })
        else:
            decisions.append({
                "time": current_time,
                "roaster": rid,
                "action": f"FAILED {sku}",
                "reason": f"{reason} | Error: {result['error']}",
            })

    return decisions


def _decide(engine, roaster, t):
    """
    Improved dispatching decision for a single roaster.
    Returns (action_tuple_or_None, reason_string).
    """
    rid = roaster.id

    # Check if there's enough time for ANY batch
    if t + ROAST_TIME > MAX_SHIFT_TIME:
        return None, "Shift ending — no time for any batch"

    # --- Emergency stockout prevention ---
    psc_action, psc_reason = _emergency_psc_check(engine, roaster, t)
    if psc_action:
        return psc_action, psc_reason

    # --- Step 1: Check MTO ---
    mto_action, mto_reason = _check_mto(engine, roaster, t)
    if mto_action:
        return mto_action, mto_reason

    # --- Step 2: PSC scheduling ---
    psc_action, psc_reason = _schedule_psc(engine, roaster, t)
    return psc_action, psc_reason


def _emergency_psc_check(engine, roaster, t):
    """If any line's RC stock is critically low, prioritize PSC to save it."""
    if SKU_PSC not in roaster.eligible_skus:
        return None, None

    possible_lines = [roaster.line]
    if roaster.id == "R3":
        possible_lines = ["L1", "L2"]

    for line_id in possible_lines:
        silo = engine.silos[line_id]
        if silo.stock <= RC_CRITICAL_THRESHOLD:
            # Emergency!
            if not roaster.needs_setup(SKU_PSC):
                if engine.pipeline_busy[roaster.pipeline] > 0:
                    return None, f"EMERGENCY: {line_id} stock={silo.stock} ≤ {RC_CRITICAL_THRESHOLD}, but pipeline busy — waiting"
            setup = SETUP_TIME if roaster.needs_setup(SKU_PSC) else 0
            if t + setup + ROAST_TIME > MAX_SHIFT_TIME:
                return None, f"EMERGENCY: {line_id} stock critical but no time left in shift"

            target_line = line_id if roaster.id == "R3" else roaster.line
            if not engine.silos[target_line].would_overflow():
                return (SKU_PSC, target_line), f"⚠ EMERGENCY PSC: {line_id} stock={silo.stock} critically low (≤{RC_CRITICAL_THRESHOLD}), immediate replenishment"
            else:
                return None, f"EMERGENCY: {line_id} critical but silo {target_line} would overflow"

    return None, None


def _check_mto(engine, roaster, t):
    """Check if MTO should be prioritized. Returns (action, reason)."""
    eligible_mto = []

    for job_key in ["NDG", "Busta"]:
        job = engine.mto_jobs[job_key]
        if job["remaining"] <= 0:
            continue
        sku = SKU_NDG if job_key == "NDG" else SKU_BUSTA
        if roaster.can_produce(sku):
            eligible_mto.append((job_key, sku, job["remaining"]))

    if not eligible_mto:
        return None, None

    # Calculate MTO urgency
    mto_due = engine.mto_due_date
    time_remaining = max(1, mto_due - t)

    total_mto_time = 0
    for job_key, sku, remaining in eligible_mto:
        eligible_roaster_count = 0
        for rid, r in engine.roasters.items():
            if r.can_produce(sku) and r.state in [RoasterState.IDLE, RoasterState.ROASTING]:
                eligible_roaster_count += 1
        eligible_roaster_count = max(1, eligible_roaster_count)

        setup_time = SETUP_TIME
        sequential_time = remaining * ROAST_TIME + setup_time
        effective_time = sequential_time / min(remaining, eligible_roaster_count)
        total_mto_time += effective_time

    total_mto_time += sum(rem for _, _, rem in eligible_mto) * CONSUME_TIME / 2
    urgency = total_mto_time / time_remaining

    deadline_pressure = False
    for job_key, sku, remaining in eligible_mto:
        setup = SETUP_TIME if roaster.needs_setup(sku) else 0
        latest_start = mto_due - (remaining * ROAST_TIME + setup + (remaining - 1) * CONSUME_TIME)
        if t >= latest_start - 5:
            urgency = max(urgency, 0.9)
            deadline_pressure = True

    past_due = False
    if t > mto_due and any(rem > 0 for _, _, rem in eligible_mto):
        urgency = 1.0
        past_due = True

    if urgency >= MTO_URGENCY_THRESHOLD:
        eligible_mto.sort(key=lambda x: (
            -(1 if x[1] == SKU_BUSTA else 0),
            -x[2],
        ))
        job_key, sku, remaining = eligible_mto[0]

        setup = SETUP_TIME if roaster.needs_setup(sku) else 0
        if t + setup + ROAST_TIME > MAX_SHIFT_TIME:
            return None, f"MTO {job_key} urgent (u={urgency:.2f}) but no time left in shift"

        pipeline = roaster.pipeline
        if not roaster.needs_setup(sku) and engine.pipeline_busy[pipeline] > 0:
            return None, f"MTO {job_key} urgent (u={urgency:.2f}) but pipeline {pipeline} busy — waiting"

        own_line_stock = engine.silos[roaster.line].stock
        if own_line_stock <= RC_LOW_THRESHOLD and SKU_PSC in roaster.eligible_skus:
            other_psc_producers = 0
            for other_rid, other_r in engine.roasters.items():
                if other_rid == roaster.id:
                    continue
                if other_r.line == roaster.line and other_r.state == RoasterState.ROASTING:
                    if other_r.current_sku == SKU_PSC:
                        other_psc_producers += 1
            if other_psc_producers == 0 and urgency < 0.85:
                return None, f"MTO {job_key} not critical enough (u={urgency:.2f}), {roaster.line} stock={own_line_stock} low with no other PSC producers — staying on PSC"

        # Build the reason string
        if past_due:
            reason = f"🔴 OVERDUE: {job_key} past deadline (due={mto_due}), {remaining} remaining — urgency={urgency:.2f}"
        elif deadline_pressure:
            reason = f"🟡 DEADLINE PRESSURE: Switching to {job_key}, {remaining} remaining, urgency={urgency:.2f}, must start now"
        else:
            reason = f"MTO urgency high ({urgency:.2f} ≥ {MTO_URGENCY_THRESHOLD}): scheduling {job_key}, {remaining} remaining, due={mto_due}"

        if setup > 0:
            reason += f" (needs {setup}min setup)"

        return (sku, None), reason

    # Not urgent — explain why
    mto_names = ", ".join(f"{jk}({rem})" for jk, _, rem in eligible_mto)
    return None, None  # Silent — will fall through to PSC


def _schedule_psc(engine, roaster, t):
    """Schedule PSC production with improved routing. Returns (action, reason)."""
    sku = SKU_PSC

    setup = SETUP_TIME if roaster.needs_setup(sku) else 0
    if t + setup + ROAST_TIME > MAX_SHIFT_TIME:
        return None, "Shift ending — no time for PSC batch"

    pipeline = roaster.pipeline
    if not roaster.needs_setup(sku) and engine.pipeline_busy[pipeline] > 0:
        return None, f"Pipeline {pipeline} busy ({engine.pipeline_busy[pipeline]}min) — waiting for pipeline"

    target_line = _smart_route(engine, roaster, t)
    if target_line is None:
        return None, "No viable output line for PSC"

    if engine.silos[target_line].would_overflow():
        if roaster.id == "R3":
            other = "L2" if target_line == "L1" else "L1"
            if not engine.silos[other].would_overflow():
                target_line = other
                reason = f"PSC → {target_line} (primary choice full, rerouted)"
                if setup > 0:
                    reason += f" (needs {setup}min setup)"
                return (sku, target_line), reason
            else:
                return None, f"Both silos near overflow (L1={engine.silos['L1'].stock}, L2={engine.silos['L2'].stock}) — waiting"
        else:
            return None, f"Silo {target_line} near overflow (stock={engine.silos[target_line].stock}/{engine.silos[target_line].max_buffer}) — waiting"

    stock = engine.silos[target_line].stock
    reason = f"PSC → {target_line} (stock={stock}/{engine.silos[target_line].max_buffer})"

    # Add context about MTO status
    mto_status_parts = []
    for jk in ["NDG", "Busta"]:
        job = engine.mto_jobs[jk]
        if job["remaining"] > 0:
            mto_status_parts.append(f"{jk}:{job['remaining']} left")
    if mto_status_parts:
        reason += f" | MTO not urgent yet ({', '.join(mto_status_parts)})"
    else:
        reason += " | All MTO completed ✓"

    if setup > 0:
        reason += f" (needs {setup}min setup)"

    return (sku, target_line), reason


def _smart_route(engine, roaster, t):
    """
    Improved R3 routing: considers consumption rate, active producers,
    and predicted future stock.
    """
    if roaster.id != "R3":
        return roaster.line

    completion_time = t + ROAST_TIME
    if roaster.needs_setup(SKU_PSC):
        completion_time += SETUP_TIME

    best_line = None
    best_need = -999

    for line_id in ["L1", "L2"]:
        silo = engine.silos[line_id]
        stock = silo.stock

        future_consumption = 0
        for tau in silo.consumption_events:
            if t < tau <= completion_time:
                future_consumption += 1

        future_production = 0
        for rid, r in engine.roasters.items():
            if rid == roaster.id:
                continue
            if r.state == RoasterState.ROASTING and r.current_sku == SKU_PSC:
                r_line = r.line
                if rid == "R3" and r.history:
                    last_entry = r.history[-1]
                    if last_entry.get("target_line"):
                        r_line = last_entry["target_line"]
                if r_line == line_id and r.remaining_time <= (completion_time - t):
                    future_production += 1

        predicted_stock = stock + future_production - future_consumption
        need = -predicted_stock

        if best_line is None or need > best_need:
            best_need = need
            best_line = line_id

    return best_line
