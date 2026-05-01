# Cost Structure Specification
## Dynamic Batch Roasting Scheduling — Nestlé Trị An

> **Purpose:** Single source of truth for all monetary values in the objective function. Referenced by `Thesis_Problem_Description_v2.md`, `mathematical_model_complete.md`, and implementation code. All values are in **USD ($)** as proxy costs — not audited factory financials. Relative ratios are what matter for scheduling decisions, not absolute dollar amounts.

---

## 1. Design Philosophy

The objective function is **Maximize Profit** — total revenue minus total costs, all expressed in the same monetary unit. This is cleaner than the old approach (maximize batches minus weighted penalties) because:

1. **Every term has the same unit ($)** — no arbitrary weight calibration needed
2. **RL reward function maps directly to profit** — the agent optimizes what we care about
3. **Costs reflect factory priorities naturally** — stockout ($1,500/event) is much worse than tardiness ($1,000/min) because it stops the packaging line
4. **Easy to interpret and validate** — "this schedule earns $280,000 per shift" is meaningful to factory management

---

## 2. Revenue (Earned Per Completed Batch)

| Symbol | SKU | Value | Unit | Condition |
|--------|-----|-------|------|-----------|
| $R^{PSC}$ | PSC | **$4,000** | per batch | Batch must complete within the shift ($e_b \leq 480$) |
| $R^{NDG}$ | NDG | **$7,000** | per batch | Batch must complete (always active — MTO) |
| $R^{BUS}$ | Busta | **$7,000** | per batch | Batch must complete (always active — MTO) |

**Notes:**
- MTO batches (NDG, Busta) are worth **1.75× more** than PSC per batch — reflects higher value-added for specialty products
- MTO revenue is **fixed** in the objective (all MTO batches must be scheduled, so $R^{MTO} \times n^{MTO}$ is a constant). But it still matters for the RL reward — completing an MTO batch is a +$7,000 reward event
- Only **completed** batches earn revenue. A batch cancelled by UPS mid-roast earns nothing
- Revenue is earned at batch completion time ($t = e_b = s_b + p_k$, where $p_k$ is SKU-dependent: PSC=15, NDG=17, Busta=18)

**Revenue per roaster-minute:**
- PSC: $4,000 / 15 min = **$267/min** of roaster time
- NDG: $7,000 / 17 min = **$412/min** of roaster time
- Busta: $7,000 / 18 min = **$389/min** of roaster time
- This means every minute a roaster is idle costs ~$267–$412 in foregone revenue (opportunity cost)

---

## 3. Costs / Penalties (Incurred Per Violation)

### 3.1 MTO Tardiness Penalty

| Symbol | Value | Unit | Trigger |
|--------|-------|------|---------|
| $c^{tard}$ | **$1,000** | per minute late | MTO job $j$ completes after due date ($e_{b_{last}^j} > 240$) |

**Applies to:** Each MTO job independently. Tardiness = $\text{tard}_j = \max(0, e_{b_{last}^j} - 240)$.

**Rationale:** Late MTO delivery incurs contractual penalties, rush shipping costs, and customer relationship damage. At $1,000/min, a 10-minute delay costs $10,000 — more than the revenue of 1 NDG batch ($7,000). This makes tardiness of ≥7 minutes unprofitable (the batch's revenue doesn't cover its tardiness cost).

**Example:**
- NDG job (3 batches), last batch ends at slot 248 → tardiness = 8 min → cost = $8,000
- Busta job (1 batch), ends at slot 235 → tardiness = 0 → cost = $0

### 3.2 RC Stockout Penalty (Understock)

| Symbol | Value | Unit | Trigger |
|--------|-------|------|---------|
| $c^{stock}$ | **$1,500** | **per consumption event with deficit** | A consumption event at $\tau \in \mathcal{E}_l$ occurs and $B_l(\tau) < 0$ after decrement (strictly negative — demand unmet) |

**Applies to:** Each line independently. Penalized **only at consumption event times** — not every time slot. Between events, negative stock has no additional penalty (the packaging line was already stopped at the event).

**Precise definition:** At each consumption event $\tau$, RC stock is decremented by 1. If $B_l(\tau) < 0$ after the decrement (strictly negative — the consumption event could not be fully served), one stockout event is recorded and penalized at $c^{stock}$. If $B_l(\tau) = 0$, the last batch was served but stock is now empty — this is **not** a stockout event (demand was met), but the packaging line is at risk. The penalty is **per event**, not per minute of duration.

**Boundary clarification:** $B_l = 0$ means "stock depleted but last demand served." $B_l < 0$ means "demand arrived and could not be served." Only the latter is penalized. This is consistent with C10 ($B_l \geq 0$ hard constraint in deterministic mode) — a feasible deterministic solution has $\text{SO}_l = 0$ always.

**Rationale:** Stockout stops the PSC packaging line. Each failed consumption event = one batch of demand unmet. At $1,500 per event, stockout is the **most expensive penalty** — reflecting the factory's top priority (PSC continuity). With ~94 events per shift per line, a sustained stockout of 10 consecutive events costs $15,000.

**In deterministic mode:** Stockout is a **hard constraint** ($B_l(\tau) \geq 0$ at all $\tau \in \mathcal{E}_l$). The penalty is effectively infinite — solver must find a schedule with zero stockout.

**In reactive mode (post-UPS):** Stockout becomes **soft** with penalty $c^{stock}$ per event. The solver minimizes total number of stockout events.

**Distinction: penalty vs. KPI:**
- **Penalty (in objective/reward):** $c^{stock}$ × number of consumption events where $B_l < 0$ (strictly negative). This is what the solver/agent optimizes.
- **KPI (reported, not optimized):** "Stockout duration" = total minutes where $B_l \leq 0$ (includes zero — line stalled waiting for recovery). This is an operational metric for analysis.

**Example:**
- Line 1 RC hits 0 at t=220. Consumption events at t=225, 230 find $B_l < 0$ (stock was already 0, decrement makes it negative).
- Stock recovers at t=232. Event at t=235 finds $B_l > 0$.
- Stockout penalty = 2 events × $1,500 = **$3,000**
- Note: the event at t=220 brought $B_l$ from 1 to 0 — demand was served, so this is **not** a stockout event.
- Stockout duration (KPI) = 12 minutes (t=220 to t=232, includes time at zero)

### 3.3 RC Overstock Penalty (Overflow Idle)

| Symbol | Value | Unit | Trigger |
|--------|-------|------|---------|
| $c^{over}$ | **$50** | per minute per roaster | Roaster is **forced idle** because $B_{\text{out}(r)}(t) = \overline{B}_l = 40$ AND roaster has no MTO batch to run instead |

**Applies to:** Individual roasters, not lines. A roaster is "overflow-idle" when it's ready to start a PSC batch but RC is full on its output line.

**Rationale:** Low penalty ($50/min) because overflow-idle is a scheduling inefficiency, not a crisis. The roaster wastes capacity but the packaging line keeps running. At $50/min, 15 minutes of overflow-idle costs $750 — less than 1 PSC batch revenue ($4,000). This encourages the solver to avoid overflow but doesn't panic over it.

**Distinction from hard overflow:** The hard constraint ($B_l(t) \leq 40$) prevents a roaster from **starting a batch** that would complete when RC is full. The soft penalty penalizes the **idle time** that results from being blocked. Both are needed.

**Example:**
- R4 is IDLE at t=300. RC Line 2 = 40 (full). R4 cannot start PSC.
- R4 stays idle until consumption event at t=305 brings RC to 39.
- R4 starts PSC at t=305 (or later if pipeline busy).
- Overflow-idle cost: 5 min × $50 = $250.

### 3.4 Setup Cost (Per Changeover Event)

| Symbol | Value | Unit | Trigger |
|--------|-------|------|---------|
| $c^{setup}$ | **$800** | per setup event | Roaster begins a setup (SKU transition), i.e., enters SETUP state |

**Applies to:** Each individual setup event on each roaster. Incurred **once** at the moment the roaster enters SETUP state — not per minute. A single PSC→NDG transition costs exactly $800 regardless of whether the setup takes the full 5 minutes or is interrupted by UPS.

**Rationale:** Changeovers consume cleaning materials, operator attention, and QC verification. The $800 lump-sum cost is separate from the **time cost** of setup (which is already penalized indirectly through lost throughput and, when RC is low, via the safety-idle penalty). Together, the time cost and the monetary cost make unnecessary changeovers doubly expensive.

**Interaction with safety-idle penalty:** During the 5-minute setup, the roaster is non-productive. If RC < 20, the safety-idle penalty ($200/min × 5 min = $1,000) fires **in addition to** the $800 setup cost, for a combined penalty of $1,800 per changeover under low stock.

**Example:**
- R2 transitions from Busta → PSC at t=80. Setup cost = **$800** (incurred at t=80).
- If RC Line 1 = 18 during t=80–84: additional safety-idle cost = 5 × $200 = $1,000.
- Total changeover penalty: $800 + $1,000 = $1,800.
- Same transition but RC Line 1 = 25: only $800 setup cost, no safety-idle.

**UPS during setup:** If UPS hits a roaster mid-setup, the $800 has already been incurred (sunk cost). When the roaster recovers from DOWN and must re-setup, a **second** $800 is charged for the new setup event.

### 3.5 Safety-Idle Penalty (Idle While RC Low)

| Symbol | Value | Unit | Trigger |
|--------|-------|------|---------|
| $c^{idle}$ | **$200** | per minute per roaster | Roaster is **idle** (not RUNNING, not DOWN) AND $B_{\text{line}(r)}(t) < \theta^{SS}$ |

**Safety stock threshold:** $\theta^{SS} = 20$ batches (half of max_buffer = 40)

**Applies to:** Individual roasters. "Idle" includes IDLE state and SETUP state (during setup, the roaster is not producing, but RC is depleting — setup under low stock is costly). Does NOT apply during DOWN state (roaster can't help being down).

**Rationale:** When RC is below safety stock, every idle minute risks a future stockout. At $200/min, this is moderate — it nudges the solver to keep roasters productive when stock is low, but doesn't override the cost of a bad decision (e.g., starting an MTO batch just to avoid idle penalty when MTO isn't due yet).

**Relationship to setup time:** During the 5-minute setup after an SKU switch, the roaster is "idle" in the SETUP state. If RC is below 20, each setup minute costs $200. A full PSC→NDG→PSC round-trip setup costs 2 × $800 (setup cost) + 10 min × $200 (safety-idle) = $3,600 — nearly 1 PSC batch revenue. This naturally discourages unnecessary SKU switches when stock is low.

**Does NOT apply:**
- During planned downtime (roaster can't produce — not a scheduling decision)
- During UPS (roaster is DOWN — not controllable)
- When RC ≥ 20 (stock is comfortable — idle is not penalized)

**Example:**
- R1 is in SETUP (PSC→NDG) from t=175 to t=179. RC Line 1 = 18 (below 20).
- Safety-idle cost: 5 min × $200 = $1,000
- Same scenario but RC Line 1 = 22: no penalty ($0)

---

## 4. Complete Objective Function

### 4.1 Deterministic Mode (Initial Schedule)

$$\boxed{\text{Maximize Profit} = \underbrace{\sum_{b: a_b=1} R_{\text{sku}(b)}}_{\text{Revenue}} - \underbrace{c^{tard} \sum_j \text{tard}_j}_{\text{Tardiness}} - \underbrace{c^{setup} \sum_r N^{setup}_r}_{\text{Setup cost}} - \underbrace{c^{idle} \sum_{r,t} \text{idle}_{r,t}}_{\text{Safety-Idle}} - \underbrace{c^{over} \sum_{r,t} \text{over}_{r,t}}_{\text{Overflow-Idle}}}$$

Where:
- $R_{\text{sku}(b)} \in \{R^{PSC}, R^{NDG}, R^{BUS}\} = \{4000, 7000, 7000\}$
- $\text{tard}_j = \max(0, \max_{b \in \mathcal{B}_j} e_b - 240)$
- $N^{setup}_r$ = number of SKU transitions (setup events) on roaster $r$ — including any initial setup if the first batch is not PSC (see §3.4)
- $\text{idle}_{r,t} = 1$ if roaster $r$ is not RUNNING at time $t$ AND not in planned downtime AND $B_{\ell(r)}(t) < 20$
- $\text{over}_{r,t} = 1$ if roaster $r$ is idle because RC on its output line = 40 (see §3.3)

**Notes on deterministic mode:**
- **No stockout penalty** ($c^{stock}$) — stockout is a **hard constraint** ($B_l(\tau) \geq 0$ at all consumption events), so stockout events never occur in a feasible deterministic solution
- **Overflow-idle IS included** — the solver can prevent overflow by pacing batch starts. The $c^{over}$ penalty gives it incentive to avoid scheduling patterns that fill RC to capacity. Without this penalty, the solver might front-load production and create overflow-idle gaps later
- MTO revenue ($R^{NDG}$, $R^{BUS}$) is a constant since all MTO batches are always active — included for completeness and RL reward decomposition

### 4.2 Reactive Mode (Post-UPS Re-solve)

$$\boxed{\text{Max Profit}_{rem} = \sum_{b \in \mathcal{B}_{rem}} R_{\text{sku}(b)} \cdot a_b - c^{tard}\sum_j \text{tard}_j - c^{stock}\sum_{l} \text{SO}_l - c^{setup}\sum_r N^{setup}_r - c^{over}\sum_{r,t} \text{over}_{r,t} - c^{idle}\sum_{r,t} \text{idle}_{r,t}}$$

Where:
- $\text{SO}_l = \left|\{\tau \in \mathcal{E}_l^{rem} : B_l(\tau) < 0\}\right|$ — **count** of consumption events on line $l$ where stock is strictly negative (demand unmet). Event-based, not duration-based.
- $N^{setup}_r$ = setup events on roaster $r$ during remaining shift (including re-setups after UPS recovery)
- $\text{over}_{r,t}$ and $\text{idle}_{r,t}$ as in deterministic mode
- $\mathcal{B}_{rem}$ = remaining unstarted batches only; completed batches frozen

**All five cost terms are active** because UPS can make both stockout and overflow-idle unavoidable, and UPS recovery may force additional setups.

### 4.3 DRL Reward Function (Per-Step Decomposition)

```python
def compute_reward(prev_state, action, new_state, t):
    reward = 0.0
    
    # Revenue: +R for each batch completed this step
    for batch in newly_completed_batches(prev_state, new_state):
        if batch.sku == PSC:
            reward += 4000
        elif batch.sku in [NDG, BUS]:
            reward += 7000
    
    # Tardiness cost: -c_tard for each new minute of tardiness
    new_tard = new_state.total_tardiness - prev_state.total_tardiness
    reward -= 1000 * new_tard
    
    # Setup cost: -c_setup for each roaster that just entered SETUP this step
    for r in roasters:
        if new_state.status[r] == SETUP and prev_state.status[r] != SETUP:
            reward -= 800   # one-time lump-sum per setup event
    
    # Stockout cost: -c_stock per CONSUMPTION EVENT where stock < 0 (strictly negative)
    # B_l = 0 means last demand served but empty; B_l < 0 means demand unmet
    for l in [L1, L2]:
        if t in E[l] and new_state.rc_stock[l] < 0:
            reward -= 1500   # one event penalty
    
    # Overflow-idle cost: -c_over for each roaster blocked by full RC
    for r in roasters:
        if new_state.status[r] == IDLE and r not in downtime_at(t):
            out_l = output_line(r)
            if r == R3:
                # R3 special: overflow-idle only if BOTH lines full
                if new_state.rc_stock[L1] >= 40 and new_state.rc_stock[L2] >= 40:
                    reward -= 50
            else:
                if new_state.rc_stock[out_l] >= 40:
                    reward -= 50
    
    # Safety-idle cost: -c_idle for each idle roaster when RC < safety
    for r in roasters:
        if new_state.status[r] in [IDLE, SETUP] and r not in downtime_at(t):
            if new_state.rc_stock[line_of(r)] < 20:
                reward -= 200
    
    return reward
```

**Key properties:**
- Setup cost fires **once** when a roaster transitions to SETUP — not per minute of setup duration
- Stockout penalty fires **only at consumption event times** (~94 per line per shift), not every slot
- Overflow-idle for R3 uses the **both-lines-full** rule (R3 can route to either line)
- Cumulative reward across all 480 time steps ≈ shift profit from the objective function

---

## 5. Cost Ratio Analysis

| Cost Item | Rate | Impact | Priority Rank |
|-----------|------|--------|---------------|
| **Stockout** | $1,500/event | 3 events ≈ 1 NDG batch revenue wiped | **1st (highest)** |
| **Tardiness** | $1,000/min | 7 min late = 1 NDG batch revenue wiped | **2nd** |
| **Setup** | $800/event | 5 setups = 1 PSC batch revenue wiped | **3rd** |
| **Safety-idle** | $200/min per roaster | 20 min idle = 1 PSC batch revenue wiped | **4th** |
| **Overflow-idle** | $50/min per roaster | 80 min idle = 1 PSC batch revenue wiped | **5th (lowest)** |

**Breakeven analysis:**
- Is it worth a 5-min setup to switch from PSC to NDG?
  - Revenue gain: $7,000 (NDG) − $4,000 (PSC would have produced) = $3,000
  - Setup cost: $800 (lump-sum) + 5 min × $200 (if RC < 20) = $1,800
  - **Net: +$1,200 → YES** (even under low stock, NDG is worth switching to)
  - But: if RC is at 5 batches (critical), 5 min setup → 1 consumption event during setup → potential stockout event at $1,500 → total cost $800 + $1,000 + $1,500 = $3,300 → **net −$300 → reconsider**

- Is it worth routing R3 to Line 1 (rescuing low stock)?
  - Benefit: avoid ~1 stockout event on L1 (consumption event arrives with $B_l < 0$) = $1,500 saved. If L1 stock is critically low, multiple consumption events could stockout → avoiding 3 events = $4,500 saved.
  - Cost: L2 gets no R3 output for 15 min → L2 stock drops by 1 extra batch
  - **Net: strongly positive** whenever L1 is at risk of stockout events

---

## 6. Parameter Sensitivity Notes

These cost values are **starting points**, not final. The thesis should acknowledge:

1. **Absolute values are proxy costs** — not derived from audited factory financials. Real revenue/cost per batch is confidential.
2. **Relative ratios are what drive scheduling decisions.** The key ratios are:
   - $R^{MTO} / R^{PSC} = 1.75$ (MTO batches are more valuable)
   - $c^{stock} / R^{PSC} = 0.375$ per minute (stockout quickly exceeds batch value)
   - $c^{tard} / R^{MTO} = 0.143$ per minute (7 min late wipes out one batch's revenue)
   - $c^{setup} / R^{PSC} = 0.20$ per event (5 setups wipe out one PSC batch)
   - $c^{idle} / R^{PSC} = 0.05$ per minute (mild pressure to stay productive)
3. **Sensitivity analysis:** If time permits, vary $c^{stock}$ and $c^{tard}$ by ±50% and observe whether strategy ranking changes. If CP-SAT beats DRL under all cost structures, the result is robust.

---

## 7. Quick Reference Card

```
╔════════════════════════════════════════════════════════╗
║               COST STRUCTURE AT A GLANCE               ║
╠════════════════════════════════════════════════════════╣
║  REVENUE                                               ║
║    PSC batch completed:    +$4,000                     ║
║    NDG batch completed:    +$7,000                     ║
║    Busta batch completed:  +$7,000                     ║
║                                                        ║
║  PENALTIES                                             ║
║    MTO tardiness:          −$1,000 / min late          ║
║    Setup cost:             −$800 / event (per change)  ║
║    RC stockout:            −$1,500 / event (per line)  ║
║    Overflow-idle:          −$50 / min (per roaster)    ║
║    Safety-idle:            −$200 / min (per roaster)   ║
║       (when RC < 20 batches on roaster's line)         ║
║                                                        ║
║  THRESHOLDS                                            ║
║    Safety stock: θ_SS = 20 batches (half of max 40)    ║
║    Max buffer:   B̄ = 40 batches per line              ║
║                                                        ║
║  PRIORITY: Stockout > Tardiness > Setup > Idle > Over  ║
╚════════════════════════════════════════════════════════╝
```
