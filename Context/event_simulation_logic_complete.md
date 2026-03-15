# Simulation & Reactive Scheduling Loop — Complete Specification
## Cross-Referenced Against `mathematical_model_complete.md`

> **Purpose:** This document specifies the **simulation engine** that wraps around the mathematical model. It defines exactly how time advances, how states evolve, how UPS events are processed, and how each strategy (Dispatching, CP-SAT, DRL) interfaces with the simulation. Every state variable, transition, and edge case is cross-referenced to the mathematical model.
>
> **Relationship to the math model:** The math model (Parts I–V of `mathematical_model_complete.md`) is a **static optimization problem** — given inputs, produce a schedule. This document describes the **dynamic loop** that calls the static model repeatedly as the shift unfolds and disruptions occur.

---

# 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                       SIMULATION ENGINE                              │
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  Clock    │───▶│  State   │───▶│  Event       │───▶│  KPI       │ │
│  │  (t=0..  │    │  Manager │    │  Processor   │    │  Tracker   │ │
│  │   479)   │    │          │    │              │    │            │ │
│  └──────────┘    └────┬─────┘    └──────┬───────┘    └────────────┘ │
│                       │                 │                            │
│                       ▼                 ▼                            │
│              ┌─────────────────────────────────┐                     │
│              │     STRATEGY INTERFACE           │                     │
│              │                                  │                     │
│              │  ┌───────────┐  ┌──────────┐    │                     │
│              │  │Dispatching│  │ CP-SAT   │    │                     │
│              │  │ Heuristic │  │ Re-solve │    │                     │
│              │  └───────────┘  └──────────┘    │                     │
│              │  ┌───────────┐                   │                     │
│              │  │   DRL     │                   │                     │
│              │  │  Agent    │                   │                     │
│              │  └───────────┘                   │                     │
│              └─────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────────┘
```

The simulation engine is **strategy-agnostic** — it maintains state, advances time, generates UPS events, and delegates scheduling decisions to whichever strategy is being evaluated. All three strategies see the same state and the same UPS events (paired comparison).

---

# 2. Simulation State

At every time slot $t$, the simulation maintains the following state. Each component is cross-referenced to the mathematical model.

## 2.1 Clock State

| Variable | Type | Range | Math model reference |
|----------|------|-------|---------------------|
| `t` | int | $[0, 479]$ | $\mathcal{T}$ (§1.1) |

The current time slot. Advances by 1 each iteration of the main loop.

## 2.2 Roaster State (per roaster $r \in \mathcal{R}$)

| Variable | Type | Values | Math model reference |
|----------|------|--------|---------------------|
| `status[r]` | enum | IDLE, RUNNING, SETUP, DOWN | State vector (§9.1 reactive framework) |
| `remaining[r]` | int | $[0, P]$ or $[0, \sigma]$ or $[0, d]$ | Derived from $s_b$, $e_b$ |
| `current_batch[r]` | batch or None | Active batch on roaster | $r_b$ (§4.2) — the inverse mapping |
| `last_sku[r]` | SKU or None | Last SKU processed | Determines setup need (C5, §6.4) |

**State transitions for each roaster:**

```
           batch assigned              batch completes
  IDLE ──────────────────▶ RUNNING ──────────────────▶ IDLE
   │                         │                           │
   │  UPS                    │  UPS                      │  different SKU
   ▼                         ▼                           ▼    assigned
  DOWN ◀─────────────────  DOWN                        SETUP
   │                                                     │
   │  timer expires                     timer expires    │
   ▼                                         ▼           │
  IDLE ◀─────────────────────────────────  IDLE ◀────────┘
```

**Detailed state definitions:**

**IDLE:** Roaster has no batch, no setup, not down. Ready to accept a new batch assignment. This is a **decision point** — the strategy must decide what to do next.

**RUNNING(remaining):** Roaster is actively processing a batch. `remaining` counts down from $P - 1 = 14$ to 0. When `remaining` = 0, batch completes: RC stock increases (if PSC), roaster transitions to IDLE.
- Math model: batch interval $[s_b, s_b + 15)$ is active. Corresponds to roaster NoOverlap constraint (C4).
- Pipeline: the first 3 slots of RUNNING also have the pipeline busy (consume interval). After slot $s_b + 2$, pipeline is free but roaster still RUNNING.

**SETUP(remaining):** Roaster is in setup between two different-SKU batches. `remaining` counts down from $\sigma - 1 = 4$ to 0. When done, transitions to IDLE (another decision point).
- Math model: setup time constraint (C5). Gap of $\sigma = 5$ between consecutive different-SKU batches.
- Pipeline: **NOT occupied** during setup. Other roasters on same line can consume freely.

**DOWN(remaining):** Roaster is unavailable due to UPS. `remaining` counts down from $d - 1$ to 0. When done, transitions to IDLE (decision point).
- Math model: UPS-induced downtime added to $\mathcal{D}'_r$ (§9.1 reactive framework).
- Any batch that was RUNNING when UPS hit is **cancelled** (see §4 UPS Processing).

## 2.3 Pipeline State (per line $l \in \mathcal{L}$)

| Variable | Type | Range | Math model reference |
|----------|------|-------|---------------------|
| `pipeline_busy[l]` | int | $[0, \delta^{con}]$ = $[0, 3]$ | Consume interval $\text{con}(b)$ (§5.2) |
| `pipeline_batch[l]` | batch or None | Which batch is consuming | Links to pipeline NoOverlap (C7) |

`pipeline_busy[l]` counts down from 2 to 0 when a consume is in progress. When it reaches 0, pipeline is free.

**Relationship to C7 (Pipeline NoOverlap):** The simulation enforces NoOverlap by construction — a new batch can only start if `pipeline_busy[pipe(r)] == 0`. The math model enforces this via the NoOverlap constraint on consume intervals. Both representations are equivalent.

## 2.4 RC Inventory State (per line $l \in \mathcal{L}$)

| Variable | Type | Range | Math model reference |
|----------|------|-------|---------------------|
| `rc_stock[l]` | int | $[..., \overline{B}_l]$ | $B_l(t)$ (§5.5) |

Integer batch counter. Increases by 1 when a PSC batch completes on a roaster with $\text{out}(b) = l$. Decreases by 1 at each consumption event $\tau \in \mathcal{E}_l$.

**Relationship to C8/C9:** In deterministic mode, `rc_stock[l] >= 0` is enforced (C8). In reactive mode after UPS, `rc_stock[l]` can go negative — each negative unit is a stockout event tracked for the penalty (§5.6, §7.2).

`rc_stock[l] <= max_buffer` is always enforced (C9) — if stock is at max and a batch would complete, the batch cannot start (blocked by the strategy's feasibility check).

## 2.5 MTO Tracking

| Variable | Type | Math model reference |
|----------|------|---------------------|
| `mto_remaining[j]` | int | $n_j$ minus completed batches of job $j$ |
| `mto_completed_time[j]` | int or None | $\max_{b \in \mathcal{B}_j} e_b$ — for tardiness computation |

Tracks how many MTO batches of each job still need to be scheduled and the latest completion time for tardiness.

**Relationship to C10:** $\text{tard}_j = \max(0, \text{mto\_completed\_time}[j] - 240)$.

## 2.6 Schedule Queue (CP-SAT strategy only)

| Variable | Type | Description |
|----------|------|-------------|
| `schedule_queue[r]` | list of (batch, start_time) | Upcoming planned batches per roaster, ordered by start time |

When CP-SAT solves the model, it produces a full schedule. The simulation stores this as a per-roaster queue. As time advances, batches are "popped" from the queue when their start time arrives. After a UPS re-solve, the queue is **replaced** with the new schedule.

**DRL and Dispatching do not use a schedule queue** — they make decisions one at a time at each decision point.

## 2.7 UPS Event List (Pre-generated)

| Variable | Type | Math model reference |
|----------|------|---------------------|
| `ups_events` | list of (time, roaster, duration) | §6.1 UPS Event Model in Problem Description |

Pre-generated before the simulation starts. Sorted by time. Same list used for all three strategies in the same replication (paired comparison).

## 2.8 KPI Accumulators

| Variable | Type | Math model reference |
|----------|------|---------------------|
| `total_psc_completed` | int | $\sum a_b$ for completed PSC batches (§7 objective) |
| `total_psc_completed_per_line[l]` | int | Per-line breakdown |
| `stockout_count` | int | KPI (§10.2) |
| `stockout_duration` | int | KPI — total minutes with $B_l < 0$ |
| `mto_tardiness` | float | $\sum_j \text{tard}_j$ (§5.4) |
| `resolves_count` | int | KPI — CP-SAT only |
| `total_compute_time` | float | KPI — wall-clock seconds |

---

# 3. Initialization (Before $t = 0$)

## 3.1 Generate UPS Events

```python
ups_events = []
t_next = exponential(1/lambda)  # first UPS arrival time

while t_next < 480:
    roaster = random_choice([R1, R2, R3, R4, R5])  # uniform
    duration = draw_from_distribution(mean=mu)       # e.g., exponential or uniform
    duration = max(1, round(duration))               # at least 1 minute, integer
    ups_events.append((floor(t_next), roaster, duration))
    t_next += exponential(1/lambda)                  # next inter-arrival

# Sort by time (should already be sorted, but ensure)
ups_events.sort(key=lambda x: x[0])
```

> **Example:** $\lambda = 3$, $\mu = 20$, seed = 42:
> ```
> ups_events = [
>     (87,  R3, 14),
>     (203, R1, 26),
>     (341, R5, 18),
> ]
> ```

**Math model cross-ref:** These events are NOT visible to the solver at shift start. They are injected into the simulation during execution. The initial CP-SAT solve sees $\lambda = 0$ effectively — a clean deterministic problem. Each UPS triggers a re-solve with updated state.

## 3.2 Initialize State

```python
t = 0

# Roaster states
for r in [R1, R2, R3, R4, R5]:
    status[r] = IDLE
    remaining[r] = 0
    current_batch[r] = None
    last_sku[r] = PSC   # assume all roasters start "as if" last batch was PSC (or None)

# Pipeline states
for l in [L1, L2]:
    pipeline_busy[l] = 0
    pipeline_batch[l] = None

# RC inventory
rc_stock[L1] = B0_L1    # e.g., 12
rc_stock[L2] = B0_L2    # e.g., 15

# MTO tracking
for j in MTO_jobs:
    mto_remaining[j] = n_j    # e.g., 3 for NDG job, 1 for Busta job
    mto_completed_time[j] = None

# KPI accumulators
total_psc_completed = 0
stockout_count = 0
stockout_duration = 0
# ... etc.

# Pre-compute consumption schedule
E_L1 = [floor(i * rho_L1) for i in range(1, floor(480/rho_L1) + 1)]
E_L2 = [floor(i * rho_L2) for i in range(1, floor(480/rho_L2) + 1)]
```

**Math model cross-ref:**
- `rc_stock` = $B^0_l$ (§2.2 demand parameters)
- `E_L1`, `E_L2` = $\mathcal{E}_l$ (§2.3 consumption schedule)
- `mto_remaining` = $n_j$ (§1.7 MTO jobs)

## 3.3 Generate Initial Schedule (Strategy-Dependent)

**CP-SAT strategy:**
```python
# Solve the FULL deterministic model (Parts I–V of math model)
# Time horizon: [0, 479]
# All batches available
# RC stockout: HARD constraint (B_l(t) >= 0)
# No UPS knowledge

schedule = solve_cpsat_deterministic(
    time_horizon=(0, 479),
    mto_jobs=MTO_jobs,
    psc_pool_size=160,
    consumption_schedule={L1: E_L1, L2: E_L2},
    initial_rc={L1: B0_L1, L2: B0_L2},
    max_buffer={L1: B_bar_L1, L2: B_bar_L2},
    planned_downtime=D,
    w_tard=w_tard,
)

# Extract per-roaster schedule queues
for r in roasters:
    schedule_queue[r] = [(b, s_b) for (b, s_b, r_b) in schedule if r_b == r]
    schedule_queue[r].sort(key=lambda x: x[1])  # sorted by start time
```

**DRL strategy:**
```python
# No initial full solve needed.
# Agent will be called at each decision point.
# Initialize the DRL environment with shift parameters.
drl_env.reset(shift_params)
```

**Dispatching strategy:**
```python
# No initial solve needed.
# Decisions made reactively at each decision point.
```

---

# 4. Main Simulation Loop

## 4.1 Loop Structure

```python
for t in range(0, 480):
    
    # ─── PHASE 1: Check UPS events at this time slot ───
    process_ups_events(t)
    
    # ─── PHASE 2: Advance roaster timers ───
    advance_roaster_states(t)
    
    # ─── PHASE 3: Advance pipeline timers ───
    advance_pipeline_states(t)
    
    # ─── PHASE 4: Process consumption events ───
    process_consumption(t)
    
    # ─── PHASE 5: Process decision points ───
    process_decision_points(t)
```

**The order of phases within each time slot matters.** Let me explain why and trace through the logic carefully.

## 4.2 Phase 1: Process UPS Events

```python
def process_ups_events(t):
    for (t_ups, r, d) in ups_events:
        if t_ups != t:
            continue  # not this slot
        
        # ── Case A: Roaster is RUNNING ──
        if status[r] == RUNNING:
            cancelled_batch = current_batch[r]
            
            # 1. Cancel the batch
            current_batch[r] = None
            
            # 2. RC stock does NOT increase (batch never completed)
            #    Math model: e_b never reached → no +1 to B_l(t)
            
            # 3. Pipeline: if consume was still in progress, release it
            l = pipe(r)
            if pipeline_batch[l] == cancelled_batch:
                pipeline_busy[l] = 0
                pipeline_batch[l] = None
                # Sunk cost: those consume minutes are gone
            
            # 4. Handle the cancelled batch
            if cancelled_batch.type == MTO:
                # Return to MTO remaining count
                j = cancelled_batch.job
                mto_remaining[j] += 1
                # Math model: batch re-enters B^MTO_rem in next re-solve
            else:
                # PSC batch: return to pool (available for re-scheduling)
                # Math model: a_b set back to 0, available in next re-solve
                pass
            
            # 5. Set roaster to DOWN
            status[r] = DOWN
            remaining[r] = d - 1   # will count down to 0
            # Math model: D'_r = D_r ∪ {t, ..., t+d-1}
        
        # ── Case B: Roaster is SETUP ──
        elif status[r] == SETUP:
            # Setup interrupted. Timer resets — must re-setup after UPS ends.
            status[r] = DOWN
            remaining[r] = d - 1
            # The batch that needed setup: not started yet, stays in queue
            # Math model: no batch cancelled (none was RUNNING)
        
        # ── Case C: Roaster is IDLE ──
        elif status[r] == IDLE:
            status[r] = DOWN
            remaining[r] = d - 1
            # No batch affected. Just blocks the roaster for d minutes.
        
        # ── Case D: Roaster is already DOWN ──
        elif status[r] == DOWN:
            # Extend downtime: take the longer remaining time
            remaining[r] = max(remaining[r], d - 1)
            # Math model: union of downtime intervals
        
        # ── Trigger re-planning (CP-SAT strategy only) ──
        if strategy == CPSAT:
            trigger_cpsat_resolve(t)
```

**Math model cross-references:**
- Batch cancellation → §6.2 UPS Impact in Problem Description: "Batch bị hủy hoàn toàn"
- GC lost → §6.2: "GC đã consume cho batch này mất"
- RC not credited → §5.5: $e_b$ never reached, so summation doesn't include this batch
- DOWN state → §9.1: $\mathcal{D}'_r = \mathcal{D}_r \cup [t_0, t_0 + d)$

**Edge case — UPS at the exact moment batch completes:** If `remaining[r] == 0` in RUNNING state, the batch has already completed (Phase 2 processes this). So the UPS hits an IDLE roaster → Case C. The batch is safe.

**Edge case — UPS on roaster whose consume is in progress:** R3 starts batch at $t=50$, consume occupies Line 2 pipeline $[50, 52]$. UPS hits R3 at $t=51$. The batch is cancelled, AND the pipeline is released early (slot 51 instead of 52). Pipeline time for slots 50–51 is sunk cost. R4 or R5 could consume starting $t=52$ (pipeline freed by cancellation).

**Edge case — Two UPS events at the same time slot:** Possible if $\lambda$ is high. Processed sequentially. If both hit different roasters, both are handled independently. If both hit the same roaster (extremely rare with continuous exponential inter-arrival → practically impossible since events are rounded to integer slots), the second extends the duration.

## 4.3 Phase 2: Advance Roaster States

```python
def advance_roaster_states(t):
    for r in roasters:
        
        if status[r] == RUNNING:
            if remaining[r] == 0:
                # ── Batch completes! ──
                b = current_batch[r]
                
                # Credit RC stock (PSC only)
                if b.type == PSC:
                    l_out = output_line(r, b)  # out(b) from §4.5
                    rc_stock[l_out] += 1
                    total_psc_completed += 1
                    total_psc_completed_per_line[l_out] += 1
                    # Math model: B_l(e_b) += 1, where e_b = s_b + 15
                
                # Track MTO completion
                if b.type == MTO:
                    j = b.job
                    mto_remaining[j] -= 1
                    if mto_completed_time[j] is None or t > mto_completed_time[j]:
                        mto_completed_time[j] = t
                    # Math model: e_b recorded for tard_j computation
                
                # Update roaster state
                current_batch[r] = None
                last_sku[r] = b.sku
                status[r] = IDLE    # ← This is a DECISION POINT
                remaining[r] = 0
            else:
                remaining[r] -= 1
        
        elif status[r] == SETUP:
            if remaining[r] == 0:
                status[r] = IDLE    # ← This is a DECISION POINT
                remaining[r] = 0
            else:
                remaining[r] -= 1
        
        elif status[r] == DOWN:
            if remaining[r] == 0:
                status[r] = IDLE    # ← This is a DECISION POINT
                remaining[r] = 0
            else:
                remaining[r] -= 1
        
        # IDLE: no timer to advance
```

**Math model cross-references:**
- Batch completes at $e_b = s_b + P$ → `remaining[r] == 0` in RUNNING state at time $t = e_b$
- RC stock update → $B_l(t)$ increment by 1 (§5.5)
- Output line → $\text{out}(b)$ function (§4.5): depends on $r_b$ and $y_b$
- Setup completes → 5-min gap enforced by C5 (§6.4)
- DOWN completes → roaster returns to available pool

**Important timing detail:** In the math model, $e_b = s_b + P$. A batch starting at $t=0$ has $e_b = 15$. In the simulation, the batch is RUNNING with remaining = 14 at $t=0$, remaining = 13 at $t=1$, ..., remaining = 0 at $t=14$. At $t=14$ (Phase 2), `remaining == 0` → batch completes. RC credited at $t=14$.

**Wait — is this consistent with the math model?** The math model says $e_b = s_b + 15 = 15$, meaning the batch "ends" at slot 15, and $B_l(t)$ counts batches where $e_b \leq t$. So at $t=14$, $e_b = 15 > 14$, batch is NOT counted. At $t=15$, $e_b = 15 \leq 15$, batch IS counted.

**This reveals a potential off-by-one.** Let me reconcile:

- **Math model convention:** Batch starts at $s_b$, ends at $e_b = s_b + P$. The batch occupies the roaster for the interval $[s_b, e_b)$ — **half-open**. The roaster is FREE at slot $e_b$.
- **Simulation convention:** Batch starts at $t = s_b$ with `remaining = P - 1 = 14`. At each time step, `remaining` decrements. When `remaining == 0`, the batch is in its **last active slot**. It completes at the **end** of this slot, and the roaster becomes IDLE at the **start** of slot $s_b + P$.

So the simulation should credit RC and transition to IDLE **at the start of slot $e_b = s_b + P$**, not at the end of slot $s_b + P - 1$. Let me adjust:

```python
# CORRECTED: Initialize remaining = P (not P-1)
# Decrement at START of each slot
# When remaining == 0, batch has completed — roaster is free

def start_batch(r, b, t):
    status[r] = RUNNING
    remaining[r] = P        # 15 slots remaining
    current_batch[r] = b

def advance_roaster_states(t):
    for r in roasters:
        if status[r] == RUNNING:
            remaining[r] -= 1      # decrement first
            if remaining[r] == 0:
                # Batch completes at the start of this slot
                # Roaster occupied slots: [s_b, s_b + P - 1] = [s_b, s_b + 14]
                # Completes: slot s_b + P = t  ← this is e_b
                complete_batch(r, t)
                status[r] = IDLE
```

**This matches the math model:** $e_b = s_b + P$. At slot $e_b$, the batch is complete and RC is credited. The roaster is free to start a new batch at $e_b$ (same SKU) or $e_b + \sigma$ (different SKU).

**Same correction applies to SETUP (remaining = σ = 5) and DOWN (remaining = d).**

## 4.4 Phase 3: Advance Pipeline States

```python
def advance_pipeline_states(t):
    for l in [L1, L2]:
        if pipeline_busy[l] > 0:
            pipeline_busy[l] -= 1
            if pipeline_busy[l] == 0:
                pipeline_batch[l] = None
                # Pipeline is now free
                # Math model: consume interval [s_b, s_b + 3) ended at slot s_b + 3
```

**Math model cross-ref:** Consume interval $\text{con}(b) = [s_b, s_b + \delta^{con}) = [s_b, s_b + 3)$. Pipeline is busy for slots $s_b, s_b+1, s_b+2$. Free at slot $s_b + 3$.

**Simulation:** When batch starts at $t$, set `pipeline_busy[l] = δ_con = 3`. Decrement each slot. Free when `pipeline_busy[l] == 0`.

With the same correction as above: initialize `pipeline_busy = 3`, decrement at start of each slot. Reaches 0 at slot $t + 3$. This matches the math model's half-open interval $[s_b, s_b + 3)$.

## 4.5 Phase 4: Process Consumption Events

```python
def process_consumption(t):
    for l in [L1, L2]:
        if t in E[l]:   # E[l] = precomputed consumption schedule E_l
            rc_stock[l] -= 1
            
            if rc_stock[l] < 0:
                stockout_count += 1
                # Math model: stockout_{l,t} = max(0, -B_l(t))
            
            # Track stockout duration (every slot where stock < 0)
            if rc_stock[l] < 0:
                stockout_duration += 1
```

**Math model cross-ref:** $\mathcal{E}_l$ (§2.3), $B_l(t)$ decrement (§5.5). Stockout tracking (§5.6).

**Edge case — Consumption and batch completion at the same slot:** If $t \in \mathcal{E}_l$ AND a batch completes at $t$ on line $l$, the **order matters**. In the phase ordering above, Phase 2 (batch completion, RC +1) happens **before** Phase 4 (consumption, RC −1). This means the batch output "arrives just in time" to prevent stockout. Is this consistent with the math model?

Math model: $B_l(t) = B^0_l + \sum(e_b \leq t) - |\{\tau \in \mathcal{E}_l : \tau \leq t\}|$. Both the batch completion ($e_b = t$) and consumption ($\tau = t$) are counted at the same time. So $B_l(t) = \text{previous} + 1 - 1 = \text{previous}$. The simulation gives the same result: RC +1 then −1 = net 0. **Consistent.** ✓

## 4.6 Phase 5: Process Decision Points

A **decision point** occurs when a roaster transitions to IDLE. The strategy must decide what to assign to this roaster.

```python
def process_decision_points(t):
    for r in roasters:
        if status[r] == IDLE and roaster_just_became_idle[r]:
            
            decision = strategy.decide(r, t, state)
            
            # decision is one of:
            #   ("START", batch_b)  — start batch b on roaster r at time t
            #   ("WAIT",)          — do nothing this slot, check again next slot
            
            if decision[0] == "START":
                b = decision[1]
                execute_batch_start(r, b, t)
            elif decision[0] == "WAIT":
                pass  # roaster stays IDLE, will re-evaluate next slot
```

**`execute_batch_start` — what happens when a batch starts:**

```python
def execute_batch_start(r, b, t):
    l = pipe(r)    # pipeline line for this roaster
    
    # ── Pre-conditions (MUST all be true — strategy guarantees this) ──
    assert status[r] == IDLE
    assert pipeline_busy[l] == 0                    # C7: pipeline free
    assert b.sku in eligible_skus[r]                # C3: eligibility
    assert t + P - 1 not in planned_downtime[r]     # C6: no overlap with downtime
    # Actually: check all slots [t, t+P-1] ∩ D_r = ∅
    
    # Check setup requirement
    if last_sku[r] is not None and last_sku[r] != b.sku:
        # Need setup — but we're starting the batch now, which means
        # the strategy should have waited for setup to complete first.
        # This means: strategy must issue a SETUP command, wait 5 slots, 
        # THEN start the batch.
        #
        # REVISED: The decision at IDLE can be:
        #   ("START_SAME_SKU", batch)  — immediate start (no setup needed)
        #   ("START_DIFF_SKU", batch)  — enter SETUP first, batch starts after
        #   ("WAIT",)                  — do nothing
        pass
    
    # ── Start the batch ──
    status[r] = RUNNING
    remaining[r] = P                                # 15 slots
    current_batch[r] = b
    
    # ── Occupy the pipeline ──
    pipeline_busy[l] = delta_con                    # 3 slots
    pipeline_batch[l] = b
    
    # ── If PSC, check overflow ──
    # This is a look-ahead: when this batch completes (at t+P),
    # will RC stock exceed max_buffer?
    # Strategy must check this before deciding.
    
    # ── Record for KPIs ──
    b.actual_start = t
    b.actual_roaster = r
```

**Wait — the setup handling needs more detail.** Let me clarify the decision flow:

### Decision Flow When Roaster Becomes IDLE

```
Roaster r becomes IDLE at time t.
last_sku[r] = the SKU of the previous batch (or None at shift start).

Strategy is asked: "What should r do?"

Strategy returns one of:
  (a) START batch b where b.sku == last_sku[r]
      → No setup needed. Batch starts at t (if pipeline free).
      → If pipeline busy: must WAIT until pipeline free, then start.
      
  (b) START batch b where b.sku != last_sku[r]
      → Setup required: 5 min.
      → Roaster enters SETUP state for 5 slots.
      → After setup completes (at t+5), roaster becomes IDLE again.
      → At that point, strategy is asked again — and should start batch b.
      → Pipeline is NOT occupied during setup.
      
  (c) WAIT
      → Roaster stays IDLE. Re-evaluated at next decision point.
```

**The subtlety:** When a roaster needs setup for a different-SKU batch, there are two time costs that may need to be waited out:

1. Setup time (5 min) — roaster-side only, pipeline free
2. Pipeline availability — pipeline must be free when batch actually starts

These can overlap. If setup takes 5 min and pipeline becomes free at minute 3 of setup, the batch starts at minute 5 (setup is the binding constraint). If pipeline is busy until minute 7, the batch starts at minute 7 (pipeline is the binding constraint).

**In the simulation, we model this as:**

```python
# Strategy decides to start batch b (different SKU) on roaster r at time t

if b.sku != last_sku[r] and last_sku[r] is not None:
    # Enter SETUP
    status[r] = SETUP
    remaining[r] = sigma          # 5 slots
    pending_batch[r] = b          # remember what batch to start after setup
    # After 5 slots, roaster becomes IDLE with last_sku updated
    # Decision point fires again — now check pipeline and start

# When SETUP completes:
def on_setup_complete(r, t):
    status[r] = IDLE
    last_sku[r] = pending_batch[r].sku   # ← update last_sku to new SKU
    # Now the batch can start (if pipeline free)
    # Decision point fires → strategy should start pending_batch[r]
```

**Math model cross-ref:** This two-step process (SETUP → IDLE → START) correctly implements C5 (§6.4): $s_{b_2} \geq e_{b_1} + \sigma$ when SKUs differ. The setup gap of $\sigma = 5$ is explicitly simulated as 5 time slots where the roaster is in SETUP state.

### Pipeline Wait Handling

If the strategy decides to start a batch but the pipeline is busy:

```python
# Strategy decides START batch b on roaster r
# But pipeline_busy[pipe(r)] > 0

# Option A: Strategy returns WAIT, and re-evaluates each slot until pipeline free.
# Option B: Strategy "reserves" the batch, and the simulation auto-starts when pipeline frees.

# We use Option A for simplicity and generality (works for all strategies):
# The strategy is called every slot that the roaster is IDLE.
# If pipeline is busy, strategy returns WAIT.
# When pipeline becomes free, strategy's next call will return START.
```

**Math model cross-ref:** The solver handles this automatically — C7 (NoOverlap) ensures no batch starts when pipeline is busy. The simulation achieves the same effect by checking `pipeline_busy[l] == 0` before allowing a START.

---

# 5. CP-SAT Re-Solve Logic

## 5.1 When Is CP-SAT Re-Solve Triggered?

Re-solve is triggered **immediately when a UPS event occurs** (Phase 1 of the main loop). Not on a periodic timer.

```python
def trigger_cpsat_resolve(t):
    # ── Step 1: Snapshot current state ──
    state_snapshot = capture_state(t)
    
    # ── Step 2: Build reduced model ──
    reduced_model = build_reduced_cpsat_model(state_snapshot)
    
    # ── Step 3: Solve ──
    start_time = time.time()
    solution = reduced_model.solve(time_limit=1.0)  # 1 second max
    elapsed = time.time() - start_time
    
    resolves_count += 1
    total_compute_time += elapsed
    
    # ── Step 4: Replace schedule queues ──
    if solution is not None:
        for r in roasters:
            schedule_queue[r] = extract_queue(solution, r, t)
    else:
        # Solver found no solution within time limit
        # Fall back to dispatching heuristic for remaining shift
        strategy = DISPATCHING  # degrade gracefully
```

## 5.2 How the Reduced Model Differs from the Full Model

The full model (Parts I–V of the math model) is for the **initial solve at shift start**. The reduced model at re-solve time $t_0$ has these modifications:

### 5.2.1 Time Horizon

$$\mathcal{T}_{rem} = \{t_0, t_0 + 1, \dots, 479\}$$

Only the remaining slots. This reduces the variable domains.

**Math model:** $s_b \in \mathcal{T}$ (§4.3) → becomes $s_b \in \mathcal{T}_{rem}$.

### 5.2.2 Frozen Batches

Batches that have **already completed** before $t_0$ are **removed from the model entirely**. Their effects (RC stock increases) are baked into the initial RC stock.

Batches that are **currently in progress** (RUNNING with remaining > 0) on other roasters are **fixed**: their start time, roaster, and SKU are known. They are modeled as **fixed intervals** (not decision variables) that still participate in NoOverlap constraints.

```python
# Example at t_0 = 150:
# R3: RUNNING, batch started at t=147, remaining=12 → fixed interval [147, 162)
#     Consume interval: [147, 150) — already past, but still counts in NoOverlap
#     Actually: consume [147, 149] already done. Pipeline already free.
#     Roaster interval [147, 162): partially in past, but future part [150, 162) 
#     still occupies the roaster → model as fixed interval [150, 162) on R3.
# R5: RUNNING, batch started at t=148, remaining=13 → fixed interval [150, 163)
```

**Math model cross-ref:** These become **fixed intervals** in the NoOverlap sets for roasters (C4) and pipelines (C7). The solver cannot move them — only schedule new batches around them.

### 5.2.3 Initial RC Stock (Updated)

$$B^0_l \;\leftarrow\; \text{rc\_stock}[l] \text{ at time } t_0$$

Not the original shift-start value — the **actual current stock** after all production and consumption up to $t_0$.

**Math model cross-ref:** $B^0_l$ in §2.2 → replaced with actual $B_l(t_0)$.

### 5.2.4 Consumption Schedule (Trimmed)

$$\mathcal{E}_l \;\leftarrow\; \{\tau \in \mathcal{E}_l : \tau \geq t_0\}$$

Only future consumption events. Past ones already happened.

### 5.2.5 Downtime (Extended with UPS)

$$\mathcal{D}'_r = \mathcal{D}_r \cup \{t_0, t_0+1, \dots, t_0 + d - 1\} \quad \text{for UPS-affected roaster } r$$

The UPS duration is modeled as additional planned downtime in the reduced model.

**Math model cross-ref:** $\mathcal{D}_r$ in §2.4 → extended with UPS-induced slots.

### 5.2.6 MTO Remaining

$$n_j \;\leftarrow\; \text{mto\_remaining}[j]$$

Only the uncompleted MTO batches need to be scheduled. Already-completed MTO batches are removed.

### 5.2.7 PSC Pool (Reduced)

$$N^{pool}_{rem} = \sum_{r \in \mathcal{R}} \left\lfloor \frac{480 - t_0}{P} \right\rfloor$$

The pool of optional PSC batches is regenerated for the remaining horizon. It's smaller than the initial pool (fewer slots available).

### 5.2.8 Stockout Constraint (Softened)

$$B_l(t) \geq 0 \quad \text{(hard)} \;\;\rightarrow\;\; \text{penalize } \max(0, -B_l(t)) \quad \text{(soft)}$$

In the reactive model, stockout may be unavoidable. The solver minimizes stockout rather than declaring infeasibility.

**Math model cross-ref:** §7.2 reactive mode objective adds $-w^{stock} \cdot \sum \text{stockout}_{l,t}$.

### 5.2.9 Setup State Continuity

If a roaster at $t_0$ was in SETUP (interrupted by UPS) or if its `last_sku` differs from the first batch the solver wants to assign:

$$\text{For roaster } r: \text{ if last\_sku}[r] \neq \text{sku of first batch on } r \text{ in new schedule} \rightarrow s_{b_{first}} \geq t_0 + \sigma$$

The solver must account for setup time from the current SKU state of each roaster.

**Math model cross-ref:** C5 (§6.4) — first batch on each roaster in the reduced model must respect setup with respect to `last_sku[r]`.

### 5.2.10 Complete Re-Solve Input Summary

```
╔═══════════════════════════════════════════════════════════╗
║         CP-SAT RE-SOLVE INPUT at t₀ = 150                ║
╠═══════════════════════════════════════════════════════════╣
║ Time horizon:   [150, 479] = 330 slots                   ║
║ Roaster states:                                          ║
║   R1: IDLE, last_sku=NDG                                 ║
║   R2: IDLE, last_sku=PSC                                 ║
║   R3: RUNNING [147,162), fixed, last_sku=PSC             ║
║   R4: DOWN until t=175, last_sku=PSC                     ║
║   R5: RUNNING [148,163), fixed, last_sku=PSC             ║
║ RC stock:  L1=8, L2=11                                   ║
║ Consumption:  E_L1 = {151, 156, ...}, E_L2 = {152, ...}  ║
║ MTO remaining: j1=1 (NDG), j2=0 (Busta done)            ║
║ PSC pool: 110 optional batches (reduced horizon)         ║
║ Planned downtime: R3 [200,229], R4 [150,174] (UPS)      ║
║ Stockout mode: SOFT (penalty w_stock)                    ║
║ Setup needed: R1 needs 5min if assigned PSC (was NDG)    ║
║ Objective: max PSC_rem − w_tard × tard − w_stock × SO    ║
╚═══════════════════════════════════════════════════════════╝
```

---

# 6. DRL Agent Interface

## 6.1 When Is the DRL Agent Called?

At every **decision point** — whenever a roaster becomes IDLE (batch completed, setup completed, or UPS ended). Not at every time slot.

```python
def drl_decide(r, t, state):
    observation = encode_state(state, t)     # normalize to [0,1] vector
    action_mask = compute_action_mask(r, t, state)
    action = agent.predict(observation, action_mask)
    return decode_action(action, r)
```

## 6.2 Observation Vector

| Index | Feature | Range | Source |
|-------|---------|-------|--------|
| 0 | $t / 479$ | [0, 1] | Normalized current time |
| 1–5 | $\text{status}[R_i]$ encoded | [0, 1] | 0=IDLE, 0.33=SETUP, 0.67=RUNNING, 1.0=DOWN |
| 6–10 | $\text{remaining}[R_i] / P$ | [0, 1] | Normalized remaining timer |
| 11–15 | $\text{last\_sku}[R_i]$ encoded | [0, 1] | 0=PSC, 0.5=NDG, 1.0=Busta |
| 16 | $\text{rc\_stock}[L_1] / \overline{B}_{L_1}$ | [0, 1] | Normalized L1 RC stock |
| 17 | $\text{rc\_stock}[L_2] / \overline{B}_{L_2}$ | [0, 1] | Normalized L2 RC stock |
| 18 | $\text{mto\_remaining\_total} / N^{MTO}$ | [0, 1] | Fraction of MTO batches remaining |
| 19 | $\text{pipeline\_busy}[L_1] / \delta^{con}$ | [0, 1] | L1 pipeline status |
| 20 | $\text{pipeline\_busy}[L_2] / \delta^{con}$ | [0, 1] | L2 pipeline status |

**Total observation dimension: 21** (compact — good for PPO).

## 6.3 Action Space and Masking

Raw action space: $|\mathcal{K}| \times |\mathcal{R}| + 1 = 3 \times 5 + 1 = 16$

| Action ID | Meaning |
|-----------|---------|
| 0 | Start PSC on R1 |
| 1 | Start PSC on R2 |
| 2 | Start PSC on R3 |
| 3 | Start PSC on R4 |
| 4 | Start PSC on R5 |
| 5 | Start NDG on R1 |
| 6 | Start NDG on R2 |
| 7 | Start NDG on R3 (always masked — R3 can't do NDG) |
| 8 | Start NDG on R4 (always masked) |
| 9 | Start NDG on R5 (always masked) |
| 10 | Start Busta on R1 (always masked — R1 can't do Busta) |
| 11 | Start Busta on R2 |
| 12 | Start Busta on R3 (always masked) |
| 13 | Start Busta on R4 (always masked) |
| 14 | Start Busta on R5 (always masked) |
| 15 | WAIT (do nothing this decision point) |

**Action mask computation:**

```python
def compute_action_mask(r, t, state):
    mask = [False] * 16
    
    for action_id in range(15):
        sku = action_id // 5    # 0=PSC, 1=NDG, 2=Busta
        roaster = action_id % 5  # 0=R1, ..., 4=R5
        
        # Only the roaster at this decision point can be assigned
        if roasters[roaster] != r:
            continue
        
        # C3: Eligibility
        if sku_types[sku] not in eligible_skus[r]:
            continue
        
        # Roaster must be IDLE
        if status[r] != IDLE:
            continue
        
        # C7: Pipeline must be free (or will be free after setup)
        l = pipe(r)
        if last_sku[r] != sku_types[sku] and last_sku[r] is not None:
            # Need setup (5 min). Pipeline must be free at t+5.
            # We can't know for sure — allow action, setup will handle waiting.
            pass
        else:
            # No setup. Pipeline must be free NOW.
            if pipeline_busy[l] > 0:
                continue
        
        # C9: RC overflow check (lookahead)
        if sku_types[sku] == PSC:
            l_out = output_line_for(r, sku)
            # Batch completes at t + (setup if needed) + P
            # Check if RC would overflow
            completion_time = t + (sigma if last_sku[r] != sku_types[sku] else 0) + P
            future_rc = rc_stock[l_out] + pending_completions(l_out, t, completion_time) \
                        - consumption_events_between(l_out, t, completion_time) + 1
            if future_rc > max_buffer[l_out]:
                continue
        
        # C6: Downtime check
        start_time = t + (sigma if last_sku[r] != sku_types[sku] else 0)
        if any(slot in planned_downtime[r] for slot in range(start_time, start_time + P)):
            continue
        
        # MTO availability
        if sku_types[sku] in [NDG, BUSTA]:
            relevant_jobs = [j for j in mto_jobs if j.sku == sku_types[sku] and mto_remaining[j] > 0]
            if not relevant_jobs:
                continue
        
        mask[action_id] = True
    
    mask[15] = True  # WAIT is always available
    return mask
```

**Math model cross-ref:** Each mask check corresponds to a constraint:
- Eligibility → C3 (§6.2)
- Pipeline availability → C7 (§6.6)
- RC overflow → C9 (§6.8)
- Downtime → C6 (§6.5)
- MTO availability → C1 (§6.1) — can't schedule MTO that doesn't exist

## 6.4 Reward Function

```python
def compute_reward(prev_state, action, new_state, t):
    reward = 0.0
    
    # +1 for each PSC batch completed this step
    reward += (new_state.total_psc - prev_state.total_psc)
    
    # Penalty for stockout events
    reward -= w_stock * (new_state.stockout_count - prev_state.stockout_count)
    
    # Penalty for MTO tardiness (measured at batch completion)
    reward -= w_tard * (new_state.mto_tardiness - prev_state.mto_tardiness)
    
    # Small negative reward per time step (encourages efficiency)
    reward -= 0.01
    
    return reward
```

**Math model cross-ref:** This reward function is the **per-step decomposition** of the objective function (§7). The cumulative reward over an episode equals the objective value.

---

# 7. Consistency Verification Checklist

| Math Model Element | Simulation Implementation | Consistent? |
|-------------------|--------------------------|-------------|
| $P = 15$ min processing | `remaining[r] = P`, decrements to 0 | ✓ — batch occupies $[s_b, s_b+P)$ |
| $\delta^{con} = 3$ min consume | `pipeline_busy[l] = 3`, decrements to 0 | ✓ — pipeline busy $[s_b, s_b+3)$ |
| Consume concurrent with roast start | Batch start sets both `remaining[r]=P` and `pipeline_busy[l]=3` at same $t$ | ✓ |
| $\sigma = 5$ min setup | SETUP state with `remaining = 5` | ✓ |
| Setup does not occupy pipeline | Pipeline not set when entering SETUP | ✓ |
| Pipeline NoOverlap (C7) | Batch can only START if `pipeline_busy[pipe(r)] == 0` | ✓ — enforced by action mask (DRL) and pre-condition check (CP-SAT/dispatching) |
| Roaster NoOverlap (C4) | Batch can only START if `status[r] == IDLE` | ✓ |
| Planned downtime (C6) | $[s_b, s_b+P-1] \cap \mathcal{D}_r = \emptyset$ checked before start | ✓ |
| RC balance (§5.5) | `rc_stock[l] += 1` at completion, `rc_stock[l] -= 1` at consumption | ✓ |
| $e_b = s_b + P$ timing | Batch RUNNING for $P$ slots, completes at slot $s_b + P$ | ✓ — with corrected remaining = P (not P-1) |
| NDG/Busta not in RC | Only PSC batches credit `rc_stock` | ✓ |
| $\text{out}(b)$ (§4.5) | `output_line(r, b)` function matches cases | ✓ |
| UPS cancels batch | RUNNING → batch removed, RC NOT credited | ✓ |
| UPS: GC lost, restart needed | Re-start = new batch object, new consume, full 15 min | ✓ |
| UPS: scheduler decides restart | Decision point fires after DOWN ends | ✓ |
| Soft stockout in reactive | `rc_stock` can go negative, tracked as penalty | ✓ |
| Hard overflow always | Overflow checked in action mask / CP-SAT constraint | ✓ |
| Paired comparison | Same `ups_events` list for all 3 strategies | ✓ |
| Consumption schedule $\mathcal{E}_l$ | Precomputed `E[l]`, checked each slot | ✓ |
| MTO tardiness (C10) | Tracked at batch completion, $\max(0, e_b - 240)$ | ✓ |

---

# 8. Complete Worked Example — Multi-UPS Scenario

## 8.1 Setup

```
Shift instance (same as math model §3):
  MTO: j1 = NDG × 3, j2 = Busta × 1
  Consumption: ρ_L1 = 5.1, ρ_L2 = 4.8
  Initial RC: L1 = 12, L2 = 15
  Planned downtime: R3 [200, 229]
  
UPS events (λ=2, μ=20, seed=42):
  UPS #1: t=87, R4, duration=22 min (R4 down until t=109)
  UPS #2: t=195, R1, duration=15 min (R1 down until t=210)
```

## 8.2 Trace: Initial Schedule (CP-SAT)

```
CP-SAT solves full [0, 479] deterministic model.
Optimal schedule (simplified, showing first 100 slots):

R1: [NDG 0-15][NDG 15-30][NDG 30-45][SETUP 45-50][PSC 50-65][PSC 65-80][PSC 80-95]...
R2: [Busta 3-18][SETUP 18-23][PSC 23-38][PSC 38-53][PSC 53-68][PSC 68-83]...
R3: [PSC 0-15][PSC 15-30][PSC 30-45][PSC 45-60][PSC 60-75][PSC 75-90]...
R4: [PSC 3-18][PSC 18-33][PSC 33-48][PSC 48-63][PSC 63-78][PSC 78-93]...
R5: [PSC 6-21][PSC 21-36][PSC 36-51][PSC 51-66][PSC 66-81][PSC 81-96]...

Pipeline L1: [R1:0-2][R2:3-5]...[R1:15-17][R2:23-25]...[R1:30-32]...
Pipeline L2: [R3:0-2][R4:3-5][R5:6-8]...[R3:15-17][R4:18-20][R5:21-23]...
```

## 8.3 Trace: UPS #1 at t=87 on R4

```
t=87: UPS EVENT on R4!

State at t=87:
  R1: RUNNING PSC batch [80, 95), remaining=8
  R2: RUNNING PSC batch [83, 98), remaining=11 (started late due to pipeline)
  R3: RUNNING PSC batch [75, 90), remaining=3
  R4: RUNNING PSC batch [78, 93), remaining=6  ← THIS BATCH IS CANCELLED
  R5: RUNNING PSC batch [81, 96), remaining=9
  
  Pipeline L2: free (R4's consume [78,80] ended long ago)
  RC stock: L1 = 8, L2 = 11 (after ~17 consumption events per line)

Processing UPS:
  1. R4 batch [78, 93) CANCELLED. RC L2 does NOT get +1 at t=93.
  2. R4 → DOWN, remaining = 22 (down until t=109)
  3. Trigger CP-SAT re-solve.

CP-SAT RE-SOLVE at t=87:
  Time horizon: [87, 479] = 393 slots
  Frozen in-progress:
    R1: [80, 95) fixed — will complete at t=95
    R2: [83, 98) fixed — will complete at t=98
    R3: [75, 90) fixed — will complete at t=90
    R5: [81, 96) fixed — will complete at t=96
  R4: DOWN until t=109. D'_R4 = {87, ..., 108}
  RC stock: L1=8, L2=11
  Consumption: E_L1 = {87(maybe), 92, 97, ...}, E_L2 = {88, 93, ...}
  MTO remaining: j1=0, j2=0 (all MTO done by slot 45)
  Stockout mode: SOFT
  
  Solver produces new schedule for [87, 479]:
    R4 resumes at t=109: [PSC 109-124][PSC 124-139]...
    Other roasters continue with slight adjustments to absorb R4 gap.
    R3: some batches routed to L2 to compensate for R4 lost batch.
  
  Schedule queues updated. Simulation continues.
```

## 8.4 Trace: UPS #2 at t=195 on R1

```
t=195: UPS EVENT on R1!

State at t=195:
  R1: RUNNING PSC batch [185, 200), remaining=5  ← CANCELLED
  R2: RUNNING PSC batch [188, 203), remaining=8
  R3: RUNNING PSC batch [183, 198), remaining=3
  R4: RUNNING PSC batch [190, 205), remaining=10
  R5: IDLE (just completed batch, decision pending)
  
  R1 batch CANCELLED. RC L1 does NOT get +1.
  R1 → DOWN until t=210.
  
  Note: R3 has planned downtime at [200, 229].
  R3 current batch [183, 198) will complete at t=198, before downtime at 200. ✓
  But R3 cannot start a new batch: 198 + 15 = 213 > 200. Must idle until 230.
  
  DOUBLE IMPACT on Lines:
  - Line 1: R1 down (195-210) + R1 batch lost. Only R2 producing for L1.
  - Line 2: R3 about to enter planned downtime (200-229). Only R4, R5 producing.
  
  Both lines under-producing simultaneously. RC stocks dropping fast.

CP-SAT RE-SOLVE at t=195:
  R1: DOWN until 210. D'_R1 = {195,...,209}
  R3: will complete current batch at 198, then planned downtime {200,...,229}
  RC stock: L1=5, L2=7 (getting low)
  
  Solver decisions:
  - R3 batch completing at 198: route to L1 (y_b = 1) — L1 needs rescue
  - R2 stays on PSC for L1 continuously — no MTO remaining
  - R1 at t=210: immediately start PSC (last_sku=PSC, no setup needed)
  - R3 at t=230: immediately start PSC after downtime
  - R4, R5: maximize throughput for L2
  
  Solver may report: stockout on L1 unavoidable between t=220-230
  (consumption continues but R1 just returned and R2 alone insufficient)
  → stockout_L1 = ~2 events
```

## 8.5 KPI Outcome

```
Final KPIs (CP-SAT strategy, this replication):
  PSC completed: 59 (vs. 66 in no-UPS baseline)
  Throughput loss: 7 batches (≈11%)
  Stockout events: 2 (both on L1 around t=225)
  Stockout duration: 6 minutes
  MTO tardiness: 0 (all MTO done before UPS events)
  Re-solves: 2
  Total compute time: 0.8 seconds

Compare to Dispatching (same UPS events):
  PSC completed: 55 (4 fewer than CP-SAT)
  Stockout events: 5 (dispatching didn't optimize R3 routing)
  Stockout duration: 18 minutes
  → CP-SAT recovers better: +4 batches, −12 min stockout

Compare to DRL (same UPS events):
  PSC completed: 58 (1 fewer than CP-SAT)
  Stockout events: 1 (DRL learned to preemptively route R3 to L1 after UPS#2)
  Stockout duration: 3 minutes
  → DRL nearly matches CP-SAT on throughput, beats on stockout
```

---

# 9. Summary: Simulation ↔ Math Model Mapping

```
Math Model (static)              Simulation (dynamic)
═══════════════════              ═══════════════════════
Sets §1                    →     Defined at initialization
Parameters §2              →     Constants in simulation config
Decision vars §4           →     CP-SAT: solver output → schedule queue
                                 DRL: agent action at decision points
                                 Dispatching: rule output at decision points
Constraints §6 (C1-C11)   →     Enforced by:
                                   C1-C3: action mask / model constraints
                                   C4: status[r] must be IDLE to start
                                   C5: SETUP state (5 min delay)
                                   C6: downtime check before start
                                   C7: pipeline_busy check before start
                                   C8-C9: RC bounds check / soft penalty
                                   C10: tracked at completion
                                   C11: y_b set by solver / agent
Objective §7               →     KPI accumulators (throughput, tardiness, stockout)
RC balance §5.5            →     rc_stock[l] incremented/decremented per event
Reactive framework §9      →     THIS DOCUMENT — the simulation loop itself
```
