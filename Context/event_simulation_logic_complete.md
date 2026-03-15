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

**RUNNING(remaining):** Roaster is actively processing a batch. `remaining` initializes to $P = 15$ and decrements by 1 at the **start** of each slot. When `remaining` reaches 0, the batch completes: RC stock increases (if PSC), roaster transitions to IDLE. A batch starting at slot $s_b$ completes at slot $s_b + P$.
- Math model: batch interval $[s_b, s_b + 15)$ is active. Corresponds to roaster NoOverlap constraint (C4).
- Pipeline: the first 3 slots of RUNNING also have the pipeline busy (consume interval). After slot $s_b + 2$, pipeline is free but roaster still RUNNING.

**SETUP(remaining):** Roaster is in setup between two different-SKU batches. `remaining` initializes to $\sigma = 5$ and decrements by 1 each slot. When `remaining` reaches 0, transitions to IDLE (another decision point).
- Math model: setup time constraint (C5). Gap of $\sigma = 5$ between consecutive different-SKU batches.
- Pipeline: **NOT occupied** during setup. Other roasters on same line can consume freely.

**DOWN(remaining):** Roaster is unavailable due to UPS. `remaining` initializes to $d$ (UPS duration) and decrements by 1 each slot. When `remaining` reaches 0, transitions to IDLE (decision point).
- Math model: UPS-induced downtime added to $\mathcal{D}'_r$ (§9.1 reactive framework).
- Any batch that was RUNNING when UPS hit is **cancelled** (see §4 UPS Processing).

## 2.3 Pipeline State (per line $l \in \mathcal{L}$)

| Variable | Type | Range | Math model reference |
|----------|------|-------|---------------------|
| `pipeline_busy[l]` | int | $[0, \delta^{con}]$ = $[0, 3]$ | Consume interval $\text{con}(b)$ (§5.2) |
| `pipeline_batch[l]` | batch or None | Which batch is consuming | Links to pipeline NoOverlap (C7) |

`pipeline_busy[l]` counts down from 3 (= $\delta^{con}$) to 0 when a consume is in progress. When it reaches 0, pipeline is free.

**Relationship to C7 (Pipeline NoOverlap):** The simulation enforces NoOverlap by construction — a new batch can only start if `pipeline_busy[pipe(r)] == 0`. The math model enforces this via the NoOverlap constraint on consume intervals. Both representations are equivalent.

## 2.4 RC Inventory State (per line $l \in \mathcal{L}$)

| Variable | Type | Range | Math model reference |
|----------|------|-------|---------------------|
| `rc_stock[l]` | int | $[..., \overline{B}_l]$ | $B_l(t)$ (§4.5) |

Integer batch counter. $\overline{B}_l = 40$ batches per line. Increases by 1 when a PSC batch completes on a roaster with $\text{out}(b) = l$. Decreases by 1 at each consumption event $\tau \in \mathcal{E}_l$.

**Safety stock threshold:** $\theta^{SS} = 20$ batches. When `rc_stock[l] < 20`, idle roasters on line $l$ incur a $200/min safety-idle penalty (see `cost.md` §3.4).

**Relationship to C10/C11:** In deterministic mode, `rc_stock[l] >= 0` is enforced (C10). In reactive mode after UPS, `rc_stock[l]` can go negative — each consumption event where `rc_stock[l] < 0` (strictly negative, demand unmet) is a stockout event penalized at $1,500/event (see `cost.md` §3.2). Note: `rc_stock[l] = 0` means last demand was served but stock is empty — this is **not** a stockout event. Stockout duration (total minutes with $B_l \leq 0$, including zero) is tracked as a separate KPI but is **not** in the objective.

`rc_stock[l] <= 40` is always enforced (C11) — if stock is at max and a batch would complete, the batch cannot start (blocked by the strategy's feasibility check). Roasters forced idle by full RC incur a $50/min overflow-idle penalty. For R3 with flexible routing, overflow-idle applies only if **both** lines are at 40.

## 2.5 MTO Tracking

| Variable | Type | Math model reference |
|----------|------|---------------------|
| `mto_remaining[j]` | int | $n_j$ minus completed batches of job $j$ |
| `mto_completed_time[j]` | int or None | $\max_{b \in \mathcal{B}_j} e_b$ — for tardiness computation |

Tracks how many MTO batches of each job still need to be scheduled and the latest completion time for tardiness.

**Relationship to C12 (Tardiness):** $\text{tard}_j = \max(0, \text{mto\_completed\_time}[j] - 240)$. Penalized at $c^{tard} = \$1{,}000$/min.

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
| `total_revenue` | float | $\sum R_{\text{sku}(b)}$ for completed batches (§6 objective) |
| `total_psc_completed` | int | PSC batch count |
| `total_psc_completed_per_line[l]` | int | Per-line breakdown |
| `total_mto_completed` | int | MTO batch count |
| `stockout_event_count` | int | Consumption events where $B_l < 0$ (strictly negative, demand unmet) — **penalized in objective at $1,500/event** |
| `stockout_duration` | int | Total minutes with $B_l \leq 0$ — **KPI only, not in objective** |
| `mto_tardiness` | float | $\sum_j \text{tard}_j$ — penalized at $1,000/min (§6, C12) |
| `safety_idle_minutes` | int | Σ (roaster idle or SETUP minutes when $B_{\ell(r)} < 20$ and not DOWN) — penalized at $200/min |
| `overflow_idle_minutes` | int | Σ (roaster idle minutes when output line RC = 40; R3: both lines = 40) — penalized at $50/min |
| `total_cost` | float | $c^{tard} \cdot \text{tard} + c^{stock} \cdot \text{SO} + c^{idle} \cdot \text{idle} + c^{over} \cdot \text{over}$ |
| `total_profit` | float | `total_revenue - total_cost` — **primary metric** |
| `resolves_count` | int | KPI — CP-SAT only |
| `total_compute_time` | float | KPI — wall-clock seconds |

> **Important:** `stockout_event_count` is penalized in the objective. `stockout_duration` is a separate reporting KPI — correlated but not equivalent. See `cost.md` §3.2 for the precise distinction.

---

# 3. Initialization (Before $t = 0$)

## 3.1 Generate UPS Events

```python
ups_events = []
t_next = exponential(1/lambda)  # first UPS arrival time (Poisson process)

while t_next < 480:
    roaster = random_choice([R1, R2, R3, R4, R5])  # uniform across roasters
    duration = exponential(mean=mu)                  # Exponential(mean=μ) — memoryless, standard MTTR
    duration = min(duration, 3 * mu)                 # cap at 3μ to prevent pathological outliers
    duration = min(duration, 480 - floor(t_next))    # cap at remaining shift time
    duration = max(1, round(duration))               # at least 1 minute, integer
    ups_events.append((floor(t_next), roaster, duration))
    t_next += exponential(1/lambda)                  # next inter-arrival

ups_events.sort(key=lambda x: x[0])
```

> **Distribution choice:** Exponential duration is the standard MTTR model in reliability engineering (memoryless, consistent with Poisson failure framework). The $3\mu$ cap prevents rare extreme values (e.g., $\mu = 30$ → max duration = 90 min, which would consume most of the remaining shift). This is documented as Assumption A5.

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
    last_sku[r] = PSC   # ALL roasters start as if last batch was PSC (confirmed §2.6 of math model)
                        # First PSC batch: no setup. First NDG/Busta: 5 min setup needed.

# Pipeline states
for l in [L1, L2]:
    pipeline_busy[l] = 0
    pipeline_batch[l] = None

# RC inventory
rc_stock[L1] = B0_L1    # e.g., 12
rc_stock[L2] = B0_L2    # e.g., 15
MAX_BUFFER = 40          # hard overflow limit per line (= 20,000 kg / 500 kg)
SAFETY_THRESHOLD = 20    # safety-idle penalty active below this

# KPI accumulators
total_revenue = 0
total_psc_completed = 0
total_mto_completed = 0
stockout_event_count = 0  # consumption events with B_l < 0 (strictly negative, penalized)
stockout_duration = 0     # minutes with B_l <= 0 (KPI only)
mto_tardiness = 0
safety_idle_minutes = 0
overflow_idle_minutes = 0
total_cost = 0
total_profit = 0
# ... etc.

# MTO tracking
for j in MTO_jobs:
    mto_remaining[j] = n_j    # e.g., 3 for NDG job, 1 for Busta job
    mto_completed_time[j] = None

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
    time_horizon=(0, 465),          # s_b ≤ 465 (end-of-shift C9)
    mto_jobs=MTO_jobs,
    psc_pool_size=160,
    consumption_schedule={L1: E_L1, L2: E_L2},
    initial_rc={L1: B0_L1, L2: B0_L2},
    max_buffer={L1: 40, L2: 40},    # C11
    safety_threshold=20,             # θ_SS for idle penalty
    planned_downtime=D,
    initial_sku={r: PSC for r in roasters},  # C6
    # Cost parameters (from cost.md):
    revenue={PSC: 4000, NDG: 7000, BUS: 7000},
    c_tard=1000,                     # $/min MTO tardiness
    c_idle=200,                      # $/min safety-idle
    c_over=50,                       # $/min overflow-idle
    # Stockout: HARD in deterministic (no c_stock needed)
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
            
            # 3. Pipeline: if consume was still in progress, release it immediately
            l = pipe(r)
            if pipeline_batch[l] == cancelled_batch:
                pipeline_busy[l] = 0
                pipeline_batch[l] = None
                # Pipeline freed at t_0. In the reduced re-solve model, the cancelled
                # batch's pipeline interval does NOT appear — it is deleted, not truncated.
                # Other roasters on this line can consume starting at t_0 (if the solver
                # or strategy assigns them). See §5.2.2 for the full interval handling.
            
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
            remaining[r] = d              # decrement-first convention (same as RUNNING/SETUP)
            # Math model: D'_r = D_r ∪ {t, ..., t+d-1}
        
        # ── Case B: Roaster is SETUP ──
        elif status[r] == SETUP:
            # Setup interrupted. Timer resets — must re-setup after UPS ends.
            setup_target_sku[r] = None    # clear stale target — setup was not completed
            status[r] = DOWN
            remaining[r] = d              # decrement-first convention
            # last_sku[r] stays at the OLD SKU (setup never completed)
            # After DOWN ends → IDLE → strategy decides fresh (may re-setup or pick different SKU)
        
        # ── Case C: Roaster is IDLE ──
        elif status[r] == IDLE:
            status[r] = DOWN
            remaining[r] = d              # decrement-first convention
            # No batch affected. Just blocks the roaster for d minutes.
        
        # ── Case D: Roaster is already DOWN ──
        elif status[r] == DOWN:
            # Extend downtime: take the longer remaining time
            remaining[r] = max(remaining[r], d)
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

**Convention:** `remaining` initializes to the full duration ($P$, $\sigma$, or $d$). Each slot, decrement **first**, then check for zero. Batch starting at $s_b$ with `remaining = P = 15` reaches `remaining == 0` at slot $s_b + P$, matching the math model's $e_b = s_b + P$.

```python
def start_batch(r, b, t):
    status[r] = RUNNING
    remaining[r] = P              # 15 (not P-1)
    current_batch[r] = b

def advance_roaster_states(t):
    for r in roasters:
        
        if status[r] == RUNNING:
            remaining[r] -= 1     # decrement FIRST
            if remaining[r] == 0:
                # ── Batch completes at slot t = s_b + P ──
                b = current_batch[r]
                
                # Credit RC stock (PSC only)
                if b.sku == PSC:
                    l_out = output_line(r, b)  # out(b) from §3.5
                    rc_stock[l_out] += 1
                    total_psc_completed += 1
                    total_psc_completed_per_line[l_out] += 1
                    total_revenue += 4000      # R_PSC
                
                # Track MTO completion
                if b.sku in [NDG, BUS]:
                    j = b.job
                    mto_remaining[j] -= 1
                    total_mto_completed += 1
                    total_revenue += 7000      # R_NDG / R_BUS
                    if mto_completed_time[j] is None or t > mto_completed_time[j]:
                        mto_completed_time[j] = t
                
                # Update roaster state
                current_batch[r] = None
                last_sku[r] = b.sku            # last SKU *completed* on this roaster
                status[r] = IDLE               # ← DECISION POINT
                remaining[r] = 0
        
        elif status[r] == SETUP:
            remaining[r] -= 1                  # decrement FIRST
            if remaining[r] == 0:
                # Setup complete — update last_sku to the target SKU NOW
                last_sku[r] = setup_target_sku[r]
                setup_target_sku[r] = None
                status[r] = IDLE               # ← DECISION POINT
                remaining[r] = 0
        
        elif status[r] == DOWN:
            remaining[r] -= 1                  # decrement FIRST
            if remaining[r] == 0:
                status[r] = IDLE               # ← DECISION POINT
                remaining[r] = 0
        
        # IDLE: no timer to advance
```

**Math model cross-references:**
- Batch completes at $e_b = s_b + P$ → `remaining[r]` decremented to 0 at slot $e_b$. ✓
- RC stock update → $B_l(t)$ increment by 1 (§4.5). Revenue credited: $4k PSC, $7k MTO.
- Output line → $\text{out}(b)$ function (§3.5): depends on $r_b$ and $y_b$
- Setup completes → 5-min gap enforced by C5
- DOWN completes → roaster returns to available pool

**Why decrement-first:** A batch starting at $t = 0$ has `remaining = 15`. At $t = 0$: decrement → 14, not zero → still RUNNING. At $t = 14$: decrement → 1, not zero → still RUNNING. At $t = 15$: decrement → 0 → batch completes. This matches $e_b = 0 + 15 = 15$. Same logic applies to SETUP ($\sigma = 5$) and DOWN ($d$).

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
            
            # STOCKOUT EVENT — penalized at $1,500 per event
            # B_l = 0 means last demand served (not stockout). B_l < 0 means demand unmet.
            if rc_stock[l] < 0:
                stockout_event_count += 1
                total_cost += C_STOCK   # $1,500
        
        # STOCKOUT DURATION — KPI only, not in objective
        if rc_stock[l] <= 0:
            stockout_duration += 1
    
    # SAFETY-IDLE tracking — per roaster, per slot
    for r in roasters:
        if status[r] in [IDLE, SETUP] and t not in planned_downtime[r]:
            if rc_stock[line_of(r)] < SAFETY_STOCK:  # < 20
                safety_idle_minutes += 1
                total_cost += C_IDLE   # $200
    
    # OVERFLOW-IDLE tracking — per roaster, per slot
    for r in roasters:
        if status[r] == IDLE and t not in planned_downtime[r]:
            if r == R3:
                # R3 special: overflow-idle only if BOTH lines full
                if rc_stock[L1] >= MAX_BUFFER and rc_stock[L2] >= MAX_BUFFER:
                    overflow_idle_minutes += 1
                    total_cost += C_OVER   # $50
            else:
                out_l = output_line(r)
                if rc_stock[out_l] >= MAX_BUFFER:  # = 40
                    overflow_idle_minutes += 1
                    total_cost += C_OVER   # $50
```

**Stockout penalty is EVENT-BASED:** $c^{stock} = \$1{,}500$ fires only at consumption event times $\tau \in \mathcal{E}_l$ when $B_l(\tau) < 0$ (strictly negative — demand unmet). $B_l = 0$ is not penalized (demand was served). Stockout duration (minutes $B_l \leq 0$, including zero) is tracked as a KPI but **not** penalized in the objective. See `cost.md` §3.2.

**Safety-idle and overflow-idle** are tracked every time slot (per roaster). R3 overflow-idle uses the **both-lines-full** rule — R3 can route to whichever line has room. See `cost.md` §3.3, §3.4.

**Edge case — Consumption and batch completion at the same slot:** If $t \in \mathcal{E}_l$ AND a batch completes at $t$ on line $l$, the **order matters**. In the phase ordering above, Phase 2 (batch completion, RC +1) happens **before** Phase 4 (consumption, RC −1). This means the batch output "arrives just in time" to prevent stockout. Is this consistent with the math model?

Math model: $B_l(t) = B^0_l + \sum(e_b \leq t) - |\{\tau \in \mathcal{E}_l : \tau \leq t\}|$. Both the batch completion ($e_b = t$) and consumption ($\tau = t$) are counted at the same time. So $B_l(t) = \text{previous} + 1 - 1 = \text{previous}$. The simulation gives the same result: RC +1 then −1 = net 0. **Consistent.** ✓

## 4.6 Phase 5: Process Decision Points

A **decision point** occurs when a roaster needs a scheduling decision. This is tracked by a persistent `needs_decision[r]` flag — set when a roaster transitions to IDLE (from RUNNING, SETUP, or DOWN) and when a WAIT decision is made (so the roaster is re-evaluated next slot).

```python
# Initialization (before main loop):
for r in roasters:
    needs_decision[r] = True   # all roasters IDLE at shift start → need first decision

# In Phase 2 — any transition TO IDLE sets the flag:
#   (already handled: status[r] = IDLE triggers needs_decision[r] = True)

def process_decision_points(t):
    for r in [R1, R2, R3, R4, R5]:   # fixed order for determinism
        if status[r] == IDLE and needs_decision[r]:
            
            decision = strategy.decide(r, t, state)
            
            # decision is one of:
            #   ("START", batch_b)  — start batch b on roaster r at time t
            #   ("SETUP", batch_b) — enter SETUP for different-SKU batch
            #   ("WAIT",)          — do nothing this slot, re-evaluate next slot
            
            if decision[0] == "START":
                b = decision[1]
                execute_batch_start(r, b, t)
                needs_decision[r] = False   # decision consumed
            elif decision[0] == "SETUP":
                b = decision[1]
                start_setup(r, b, t)
                needs_decision[r] = False   # setup started
            elif decision[0] == "WAIT":
                needs_decision[r] = True    # stays set → re-evaluate next slot
```

**Why `needs_decision` instead of a one-shot flag:** A roaster that returns WAIT (e.g., pipeline busy) must be re-evaluated every subsequent slot until it can act. A one-shot "just became idle" flag would miss this — the roaster would stall permanently after one WAIT.
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

if b.sku != last_sku[r]:
    # Enter SETUP — roaster transitions to the new SKU
    status[r] = SETUP
    remaining[r] = sigma          # 5 slots (decrement-first convention)
    setup_target_sku[r] = b.sku   # remember what SKU we're setting up FOR
    # After 5 slots, roaster becomes IDLE
    # Decision point fires again — now check pipeline and start

# When SETUP completes (in Phase 2, remaining hits 0):
def on_setup_complete(r, t):
    status[r] = IDLE
    last_sku[r] = setup_target_sku[r]  # roaster is now "in" the new SKU state
    setup_target_sku[r] = None
    # Decision point fires → strategy should start the intended batch
    # (if pipeline free; otherwise WAIT)
```

**Semantics of `last_sku[r]`:** This variable means **"the SKU state the roaster is currently configured for."** It is updated at two points:
1. When a batch **completes** (Phase 2): `last_sku[r] = completed_batch.sku`
2. When a **setup completes**: `last_sku[r] = setup_target_sku[r]`

It is **not** updated when a batch starts — only when it finishes or when setup finishes. This means: during RUNNING, `last_sku[r]` reflects the previous batch's SKU (before the current one). During SETUP, `last_sku[r]` still reflects the old SKU. Only after setup completes does it flip to the new SKU.

**Why this matters for the action mask:** When the strategy is called after setup completes, `last_sku[r]` has already been updated to the new SKU. If the strategy decides to start a batch of that same SKU → no additional setup needed (correct). If the strategy changes its mind and picks a different SKU → another 5-min setup (also correct, but unusual).

**Math model cross-ref:** This two-step process (SETUP → IDLE → START) correctly implements C5: $s_{b_2} \geq e_{b_1} + \sigma$ when SKUs differ. The setup gap of $\sigma = 5$ is explicitly simulated as 5 time slots in SETUP state.

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

### 5.2.2 Frozen Batches and Cancelled Batch Interval Handling

Batches that have **already completed** before $t_0$ are **removed from the model entirely**. Their effects (RC stock increases) are baked into the initial RC stock.

**Cancelled batch** (the one hit by UPS on roaster $r$):
- Removed from the active set. Does NOT contribute to RC stock.
- Its roaster interval disappears (roaster is DOWN, modeled via $\mathcal{D}'_r$).
- Its pipeline interval: **only the portion already consumed matters**. Since consume lasts 3 minutes and is concurrent with roast start, by the time a UPS hits mid-batch, the consume is almost certainly finished (consume ends at $s_b + 3$, UPS arrives at some $t_0 > s_b + 3$ in most cases). The consumed portion is in the past — it does not appear in the reduced model. If UPS hits during the consume itself ($s_b \leq t_0 < s_b + 3$), the pipeline is released at $t_0$ (the simulation sets `pipeline_busy = 0` in Phase 1). In the reduced model, the cancelled batch's pipeline interval simply does not exist.

**In-progress batches on OTHER roasters** (not the one hit by UPS):
These are **fixed** — their start time, roaster, and SKU are known. They require two separate fixed intervals:

1. **Roaster fixed interval:** Only the future portion — $[\max(s_b, t_0), \; e_b)$ — is included in the roaster's NoOverlap set (C4). The past portion is irrelevant.

2. **Pipeline fixed interval:** Only if the consume is **still active** at $t_0$. Consume runs $[s_b, s_b + 3)$.
   - If $t_0 \geq s_b + 3$: consume already finished → **no pipeline fixed interval** (pipeline is free).
   - If $t_0 < s_b + 3$: consume still in progress → add pipeline fixed interval $[t_0, s_b + 3)$ to the line's pipeline NoOverlap set (C8).

```python
# Example at t_0 = 150:
# R3: RUNNING, batch started at t=147, remaining=12, will complete at t=162
#     Consume interval: [147, 150) — already finished at t=150.
#     Roaster fixed interval: [150, 162) in R3's NoOverlap set.
#     Pipeline fixed interval: NONE (consume [147, 150) is in the past).
#     → Line 2 pipeline is FREE at t=150 for new batches.
#
# R5: RUNNING, batch started at t=148, remaining=13, will complete at t=163
#     Consume interval: [148, 151) — partially active! t=150 is within [148, 151).
#     Roaster fixed interval: [150, 163) in R5's NoOverlap set.
#     Pipeline fixed interval: [150, 151) in Line 2's pipeline NoOverlap set.
#     → Line 2 pipeline is BUSY at t=150 (R5 still consuming), free at t=151.
```

**Convention for re-solve start time:** The reduced model's time horizon is $[t_0, 479]$. New batch starts are allowed from $t_0$ onward, subject to pipeline availability. If a frozen batch's consume extends into $[t_0, ...)$, the pipeline fixed interval blocks that slot. There is no "same-slot vs. next-slot" ambiguity — the interval geometry handles it exactly.

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

$$B_l(\tau) \geq 0 \;\text{(hard at consumption events)} \;\;\rightarrow\;\; \text{penalize each event where } B_l(\tau) < 0 \text{ at } c^{stock} = \$1{,}500 \;\text{(soft)}$$

In the reactive model, stockout events may be unavoidable. The solver minimizes the **count** of consumption events with stock ≤ 0, not the duration.

**Math model cross-ref:** §6.2 reactive mode objective uses $c^{stock} \sum_l \text{SO}_l$ where $\text{SO}_l = |\{\tau \in \mathcal{E}_l^{rem} : B_l(\tau) < 0\}|$ (strictly negative).

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
║   R3: RUNNING [150,162) roaster fixed. Pipeline: FREE (consume [147,150) past) ║
║   R4: DOWN until t=175, last_sku=PSC                     ║
║   R5: RUNNING [150,163) roaster fixed. Pipeline: [150,151) fixed (consume active) ║
║ RC stock:  L1=8, L2=11                                   ║
║ Consumption:  E_L1 = {151, 156, ...}, E_L2 = {152, ...}  ║
║ MTO remaining: j1=1 (NDG), j2=0 (Busta done)            ║
║ PSC pool: 110 optional batches (reduced horizon)         ║
║ Planned downtime: R3 [200,229], R4 [150,174] (UPS)      ║
║ Stockout mode: SOFT ($1,500/event at consumption times)   ║
║ Setup needed: R1 needs 5min if assigned PSC (was NDG)    ║
║ Objective: max Revenue − $1k×tard − $1.5k×SO − $200×idle − $50×over ║
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
| 6–10 | $\text{remaining}[R_i] / D_{max}$ | [0, 1] | Normalized remaining timer. $D_{max} = 60$ (cap for all timers: RUNNING max $P=15$, SETUP max $\sigma=5$, DOWN max $3\mu_{max} = 90$ capped to 60). Prevents out-of-range observations for long UPS. |
| 11–15 | $\text{last\_sku}[R_i]$ encoded | [0, 1] | 0=PSC, 0.5=NDG, 1.0=Busta |
| 16 | $\text{rc\_stock}[L_1] / \overline{B}_{L_1}$ | [0, 1] | Normalized L1 RC stock |
| 17 | $\text{rc\_stock}[L_2] / \overline{B}_{L_2}$ | [0, 1] | Normalized L2 RC stock |
| 18 | $\text{mto\_remaining\_total} / N^{MTO}$ | [0, 1] | Fraction of MTO batches remaining |
| 19 | $\text{pipeline\_busy}[L_1] / \delta^{con}$ | [0, 1] | L1 pipeline status |
| 20 | $\text{pipeline\_busy}[L_2] / \delta^{con}$ | [0, 1] | L2 pipeline status |

**Total observation dimension: 21** (compact — good for PPO).

## 6.3 Action Space and Masking

**17 actions** (R3 routing baked into action space):

| Action ID | Meaning |
|-----------|---------|
| 0 | Start PSC on R1 → L1 |
| 1 | Start PSC on R2 → L1 |
| 2 | Start PSC on R3 → **L1** ($y_b = 1$) |
| 3 | Start PSC on R3 → **L2** ($y_b = 0$) |
| 4 | Start PSC on R4 → L2 |
| 5 | Start PSC on R5 → L2 |
| 6 | Start NDG on R1 |
| 7 | Start NDG on R2 |
| 8 | Start Busta on R2 |
| 9–15 | *(permanently masked — ineligible SKU-roaster combos)* |
| 16 | WAIT (do nothing this decision point) |

**Per-roaster decision:** When roaster $r$ becomes IDLE, agent is called for $r$ only. Actions for other roasters are masked. If 2 roasters IDLE at same slot, called sequentially R1→R2→R3→R4→R5.

**Action mask computation:**

```python
def compute_action_mask(r, t, state):
    mask = [False] * 17
    
    for action_id in range(16):  # 0-15: batch start actions
        # Decode action → (sku, roaster, output_line)
        if action_id <= 5:   # PSC actions
            sku = PSC
            if action_id == 0: roaster, out_l = R1, L1
            elif action_id == 1: roaster, out_l = R2, L1
            elif action_id == 2: roaster, out_l = R3, L1  # R3 → L1
            elif action_id == 3: roaster, out_l = R3, L2  # R3 → L2
            elif action_id == 4: roaster, out_l = R4, L2
            elif action_id == 5: roaster, out_l = R5, L2
        elif action_id <= 7:  # NDG actions
            sku = NDG
            roaster = R1 if action_id == 6 else R2
            out_l = None  # MTO doesn't enter RC
        elif action_id == 8:  # Busta action
            sku = BUS
            roaster = R2
            out_l = None
        else:
            continue  # 9-15: permanently masked
        
        # Only the roaster at this decision point
        if roaster != r:
            continue
        
        # C3: Eligibility
        if sku not in eligible_skus[r]:
            continue
        
        # Roaster must be IDLE
        if status[r] != IDLE:
            continue
        
        # C8: Pipeline must be free (or will be free after setup)
        l_pipe = pipe(r)
        setup_needed = (last_sku[r] != sku) and (last_sku[r] is not None)
        if not setup_needed:
            if pipeline_busy[l_pipe] > 0:
                continue
        
        # C11: RC overflow check (PSC only)
        if sku == PSC and out_l is not None:
            completion_time = t + (sigma if setup_needed else 0) + P
            future_consumption = consumption_events_between(out_l, t, completion_time)
            future_completions = pending_completions(out_l, t, completion_time)
            projected_rc = rc_stock[out_l] + future_completions - future_consumption + 1
            if projected_rc > MAX_BUFFER:  # > 40
                continue
        
        # C7: Downtime check
        start_time = t + (sigma if setup_needed else 0)
        if any(slot in planned_downtime[r] for slot in range(start_time, start_time + P)):
            continue
        
        # C9: End-of-shift check
        if start_time + P > 480:  # s_b + 15 > 480 → s_b > 465
            continue
        
        # MTO availability
        if sku in [NDG, BUS]:
            relevant_jobs = [j for j in mto_jobs if j.sku == sku and mto_remaining[j] > 0]
            if not relevant_jobs:
                continue
        
        mask[action_id] = True
    
    mask[16] = True  # WAIT is always available
    return mask
```

**Math model cross-ref:** Each mask check corresponds to a constraint:
- Eligibility → C3
- Pipeline availability → C8 (Pipeline NoOverlap)
- RC overflow → C11 (RC ≤ 40). For R3 actions: checked on specific output line (L1 for action 2, L2 for action 3)
- Downtime → C7
- End-of-shift → C9 ($s_b \leq 465$)
- MTO availability → C1 (must schedule all MTO)

**Helper function — `pending_completions`** (used in overflow check):

```python
def pending_completions(line, t_now, t_future):
    """Count PSC batch completions on `line` in the interval (t_now, t_future]."""
    count = 0
    for r in roasters:
        if status[r] == RUNNING and current_batch[r].sku == PSC:
            batch_out_line = output_line(r, current_batch[r])
            if batch_out_line == line:
                completion_slot = t_now + remaining[r]  # remaining decrements → completion
                if t_now < completion_slot <= t_future:
                    count += 1
    return count
```

> **Note:** For R3, `output_line(R3, batch)` uses the batch's $y_b$ decision to determine L1 or L2.

## 6.4 Reward Function (Profit-Based, from `cost.md` §4.3)

```python
def compute_reward(prev_state, action, new_state, t):
    reward = 0.0
    
    # Revenue: +$ for each batch completed this step
    for batch in newly_completed_batches(prev_state, new_state):
        if batch.sku == PSC:
            reward += 4000   # R_PSC
        elif batch.sku in [NDG, BUS]:
            reward += 7000   # R_NDG / R_BUS
    
    # Tardiness cost: -$1,000 for each new minute of MTO tardiness
    new_tard = new_state.mto_tardiness - prev_state.mto_tardiness
    reward -= 1000 * new_tard
    
    # Stockout cost: -$1,500 per CONSUMPTION EVENT where stock < 0 (strictly negative)
    # B_l = 0 → last demand served, not stockout. B_l < 0 → demand unmet.
    for l in [L1, L2]:
        if t in E[l] and new_state.rc_stock[l] < 0:
            reward -= 1500   # one event penalty
    
    # Overflow-idle cost: -$50 for each roaster blocked by full RC
    for r in roasters:
        if new_state.status[r] == IDLE and r not in downtime_at(t):
            if r == R3:
                # R3 special: overflow-idle only if BOTH lines full
                if new_state.rc_stock[L1] >= 40 and new_state.rc_stock[L2] >= 40:
                    reward -= 50
            else:
                if new_state.rc_stock[output_line(r)] >= 40:
                    reward -= 50
    
    # Safety-idle cost: -$200 for each idle roaster when RC < safety threshold
    for r in roasters:
        if new_state.status[r] in [IDLE, SETUP] and r not in downtime_at(t):
            if new_state.rc_stock[line_of(r)] < 20:
                reward -= 200
    
    return reward
```

**Key properties:**
- All values in **dollars** — no arbitrary weights
- Stockout penalty fires **only at consumption event times** (~94/line/shift), not every slot
- Overflow-idle for R3 uses **both-lines-full** rule
- Cumulative reward across 480 steps ≈ shift profit from objective function
- See `cost.md` §4.3 for detailed justification

---

# 7. Consistency Verification Checklist

| Math Model Element | Simulation Implementation | Consistent? |
|-------------------|--------------------------|-------------|
| $P = 15$ min processing | `remaining[r] = P`, decrements to 0 | ✓ — batch occupies $[s_b, s_b+P)$ |
| $\delta^{con} = 3$ min consume | `pipeline_busy[l] = 3`, decrements to 0 | ✓ — pipeline busy $[s_b, s_b+3)$ |
| Consume concurrent with roast start | Batch start sets both `remaining[r]=P` and `pipeline_busy[l]=3` at same $t$ | ✓ |
| $\sigma = 5$ min setup | SETUP state with `remaining = 5` | ✓ |
| Setup does not occupy pipeline | Pipeline not set when entering SETUP | ✓ |
| Initial SKU = PSC (C6) | `last_sku[r] = PSC` at init. First MTO batch needs setup. | ✓ |
| End-of-shift (C9) | `start_time + P > 480` → masked in action mask | ✓ |
| Pipeline NoOverlap (C8) | Batch can only START if `pipeline_busy[pipe(r)] == 0` | ✓ |
| Roaster NoOverlap (C4) | Batch can only START if `status[r] == IDLE` | ✓ |
| Planned downtime (C7) | $[s_b, s_b+P-1] \cap \mathcal{D}_r = \emptyset$ checked before start | ✓ |
| RC balance (§4.5) | `rc_stock[l] += 1` at completion, `rc_stock[l] -= 1` at consumption | ✓ |
| $e_b = s_b + P$ timing | Batch RUNNING for $P$ slots, completes at slot $s_b + P$ | ✓ — with corrected remaining = P (not P-1) |
| NDG/Busta not in RC | Only PSC batches credit `rc_stock` | ✓ |
| $\text{out}(b)$ (§3.5) | `output_line(r, b)` function matches cases | ✓ |
| R3 routing: 17 actions | Actions 2 (→L1) and 3 (→L2) for R3 | ✓ |
| UPS cancels batch | RUNNING → batch removed, RC NOT credited | ✓ |
| UPS: GC lost, restart needed | Re-start = new batch object, new consume, full 15 min | ✓ |
| UPS: scheduler decides restart | Decision point fires after DOWN ends | ✓ |
| Stockout: event-based penalty (C10) | Penalized at consumption events only, $1,500/event | ✓ — matches `cost.md` §3.2 |
| Stockout duration: KPI only | Tracked separately, NOT in objective/reward | ✓ |
| Hard overflow always (C11) | Overflow checked in action mask, $B_l \leq 40$ | ✓ |
| R3 overflow-idle: both lines (C15) | Reward checks `rc[L1] >= 40 AND rc[L2] >= 40` for R3 | ✓ — matches math model §4.9/C15 |
| Safety-idle (C14) | Reward penalizes idle when $B_l < 20$, $200/min | ✓ |
| Overflow-idle (C15) | Reward penalizes idle when $B_l = 40$, $50/min | ✓ |
| Revenue: PSC $4k, MTO $7k | Reward += revenue at batch completion | ✓ — matches `cost.md` §2 |
| Paired comparison | Same `ups_events` list for all 3 strategies | ✓ |
| Consumption schedule $\mathcal{E}_l$ | Precomputed `E[l]`, checked each slot | ✓ |
| MTO tardiness (C12) | Tracked at batch completion, $\max(0, e_b - 240)$, $1,000/min | ✓ |
| Profit = revenue − costs | `total_profit = total_revenue - total_cost` | ✓ — primary KPI |

---

# 8. Complete Worked Example — Multi-UPS Scenario

> **Note on illustrative values:** RC stock values and timings in this section are **rounded approximations** for readability, not exact model outputs. The initial schedule (§8.2) is labeled "simplified" — some timings (e.g., R2 Busta start) may differ slightly between sub-sections due to rounding. The purpose is to demonstrate the simulation logic and strategy comparison, not to present verified numerical results.

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
  MTO completed: 4/4 (all done before UPS events)
  Stockout events: 2 (both on L1 around t=225)
  Stockout duration: 6 minutes (KPI only)
  MTO tardiness: 0 min
  Safety-idle: ~12 min (R1 IDLE after UPS recovery t=210-213 waiting for pipeline + R1/R2
    SETUP time at shift start [0-5] when L1 stock later dips below 20. DOWN time excluded.)
  Overflow-idle: 0 min
  Re-solves: 2
  Compute time: 0.8 seconds

  PROFIT CALCULATION:
    Revenue:  59 × $4,000 + 4 × $7,000           = $264,000
    Costs:    tard $0
            + stockout 2 × $1,500                  =   $3,000
            + safety-idle 12 × $200                =   $2,400
            + overflow-idle 0 × $50                =       $0
    Total cost:                                    =   $5,400
    PROFIT:   $264,000 − $5,400                    = $258,600

Compare to Dispatching (same UPS events):
  PSC completed: 55 (4 fewer than CP-SAT)
  MTO completed: 4/4
  Stockout events: 5 (dispatching didn't optimize R3 routing)
  Stockout duration: 18 minutes (KPI only)
  MTO tardiness: 0 min
  Safety-idle: ~35 min
  Overflow-idle: 0 min

  PROFIT:
    Revenue:  55 × $4,000 + 4 × $7,000           = $248,000
    Costs:    stockout 5 × $1,500 + idle 35 × $200 = $14,500
    PROFIT:   $248,000 − $14,500                   = $233,500
    → CP-SAT wins by $23,500 (+$16k revenue, −$7.5k costs)

Compare to DRL (same UPS events):
  PSC completed: 58 (1 fewer than CP-SAT)
  MTO completed: 4/4
  Stockout events: 1 (DRL learned to preemptively route R3 to L1 after UPS#2)
  Stockout duration: 3 minutes (KPI only)
  MTO tardiness: 0 min
  Safety-idle: ~15 min
  Overflow-idle: 0 min

  PROFIT:
    Revenue:  58 × $4,000 + 4 × $7,000           = $260,000
    Costs:    stockout 1 × $1,500 + idle 15 × $200 = $4,500
    PROFIT:   $260,000 − $4,500                    = $255,500
    → DRL nearly matches CP-SAT on profit (−$1,500), beats on stockout

STRATEGY RANKING (this replication):
  1st: CP-SAT   $258,600  (best total profit — optimizes remaining horizon exactly)
  2nd: DRL      $255,500  (−$3,100 — fewer stockouts but 1 less PSC batch)
  3rd: Dispatch $233,500  (−$25,100 — myopic R3 routing causes cascading stockouts)
```

---

# 9. Summary: Simulation ↔ Math Model Mapping

```
Math Model (static)              Simulation (dynamic)
═══════════════════              ═══════════════════════
Sets §1                    →     Defined at initialization
Parameters §2              →     Constants in simulation config (incl. cost.md values)
Decision vars §3           →     CP-SAT: solver output → schedule queue
                                 DRL: agent action at decision points (17 actions)
                                 Dispatching: rule output at decision points
Constraints §5 (C1-C15)   →     Enforced by:
                                   C1-C3: activation, eligibility (action mask)
                                   C4-C5: NoOverlap + setup (status[r] + SETUP state)
                                   C6: initial SKU = PSC (init)
                                   C7: downtime check before start
                                   C8: pipeline_busy check before start
                                   C9: end-of-shift s_b ≤ 465 (action mask)
                                   C10: RC ≥ 0 hard/soft (stockout event count)
                                   C11: RC ≤ 40 hard (overflow block)
                                   C12: tardiness tracked at completion
                                   C13: R3 routing via action 2/3
                                   C14: safety-idle tracked per slot
                                   C15: overflow-idle tracked per slot (R3: both lines)
Objective §6               →     Profit = revenue − costs (KPI accumulators)
                                   Revenue: $4k/$7k per completed batch
                                   Costs: $1k/min tard, $1.5k/event stockout,
                                          $200/min idle, $50/min overflow-idle
RC balance §4.5            →     rc_stock[l] incremented/decremented per event
Reactive framework         →     THIS DOCUMENT — the simulation loop itself
```
