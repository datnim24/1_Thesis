# Cost Structure Specification
## Dynamic Batch Roasting Scheduling — Nestlé Trị An

> **Purpose:** Single source of truth for all monetary values in the objective function. Referenced by `Thesis_Problem_Description_v2.md`, `mathematical_model_complete.md`, and implementation code. All values are in **USD ($)** as proxy costs — not audited factory financials. Relative ratios are what matter for scheduling decisions, not absolute dollar amounts.

---

## 1. Design Philosophy

The objective function is **Maximize Profit** — total revenue minus total costs, all expressed in the same monetary unit. This is cleaner than the old approach (maximize batches minus weighted penalties) because:

1. **Every term has the same unit ($)** — no arbitrary weight calibration needed
2. **RL reward function maps directly to profit** — the agent optimizes what we care about
3. **Costs reflect factory priorities naturally** — stockout ($1,500/min) is much worse than tardiness ($1,000/min) because it stops the packaging line
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
- Revenue is earned at batch completion time ($t = e_b = s_b + 15$)

**Revenue per roaster-minute:**
- PSC: $4,000 / 15 min = **$267/min** of roaster time
- NDG/Busta: $7,000 / 15 min = **$467/min** of roaster time
- This means every minute a roaster is idle costs ~$267–$467 in foregone revenue (opportunity cost)

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
| $c^{stock}$ | **$1,500** | per minute per line | $B_l(t) \leq 0$ at a consumption event time $\tau \in \mathcal{E}_l$ |

**Applies to:** Each line independently. Tracked at consumption event times.

**Rationale:** Stockout stops the PSC packaging line entirely. Cost includes: idle packaging labor, energy waste, production plan disruption, potential downstream supply chain penalties. At $1,500/min, stockout is the **most expensive penalty** — reflecting the factory's top priority (PSC continuity).

**Relationship to consumption rate:** With $\rho_l = 5.1$ min/batch, a 5-minute stockout spans approximately 1 consumption event → cost = $1,500 × 5 = $7,500. This is nearly 2× the revenue of one PSC batch ($4,000), making it always worth preventing.

**In deterministic mode:** Stockout is a **hard constraint** ($B_l(t) \geq 0$). The penalty is effectively infinite — the solver must find a schedule with zero stockout.

**In reactive mode (post-UPS):** Stockout becomes **soft** with penalty $c^{stock}$. The solver minimizes total stockout cost rather than declaring infeasibility.

**Example:**
- Line 1 RC stock drops to 0 at t=220, stays at 0 until t=232 (12 minutes) → cost = $1,500 × 12 = $18,000

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

### 3.4 Safety-Idle Penalty (Idle While RC Low)

| Symbol | Value | Unit | Trigger |
|--------|-------|------|---------|
| $c^{idle}$ | **$200** | per minute per roaster | Roaster is **idle** (not RUNNING, not DOWN) AND $B_{\text{line}(r)}(t) < \theta^{SS}$ |

**Safety stock threshold:** $\theta^{SS} = 20$ batches (half of max_buffer = 40)

**Applies to:** Individual roasters. "Idle" includes IDLE state and SETUP state (during setup, the roaster is not producing, but RC is depleting — setup under low stock is costly). Does NOT apply during DOWN state (roaster can't help being down).

**Rationale:** When RC is below safety stock, every idle minute risks a future stockout. At $200/min, this is moderate — it nudges the solver to keep roasters productive when stock is low, but doesn't override the cost of a bad decision (e.g., starting an MTO batch just to avoid idle penalty when MTO isn't due yet).

**Relationship to setup time:** During the 5-minute setup after an SKU switch, the roaster is "idle" in the SETUP state. If RC is below 20, each setup minute costs $200. A full PSC→NDG→PSC round-trip setup costs 10 min × $200 = $2,000 in idle penalty — half the revenue of 1 PSC batch. This naturally discourages unnecessary SKU switches when stock is low.

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

$$\boxed{\text{Maximize Profit} = \underbrace{\sum_{b \in \mathcal{B}: a_b=1} R_{\text{sku}(b)}}_{\text{Total Revenue}} - \underbrace{c^{tard} \sum_{j} \text{tard}_j}_{\text{Tardiness Cost}} - \underbrace{c^{idle} \sum_{r} \sum_{t: B_{\ell(r)}(t) < \theta^{SS}} \text{idle}_{r,t}}_{\text{Safety-Idle Cost}}}$$

Where:
- $R_{\text{sku}(b)} \in \{R^{PSC}, R^{NDG}, R^{BUS}\} = \{4000, 7000, 7000\}$
- $\text{tard}_j = \max(0, \max_{b \in \mathcal{B}_j} e_b - 240)$
- $\text{idle}_{r,t} = 1$ if roaster $r$ is not RUNNING at time $t$ AND not in planned downtime AND $B_{\ell(r)}(t) < 20$

**Notes on deterministic mode:**
- No stockout penalty ($c^{stock}$) — stockout is a **hard constraint** ($B_l(t) \geq 0$), so the solver never encounters it
- No overstock penalty ($c^{over}$) — overflow is a **hard constraint** ($B_l(t) \leq 40$), enforced by not allowing batch starts that would overflow. The resulting idle time falls under safety-idle if RC is also below threshold (unlikely — overflow and understock don't coexist), otherwise it's just unpenalized idle
- MTO revenue ($R^{NDG}$, $R^{BUS}$) is a constant since all MTO batches are always active — but included for completeness and for the RL reward decomposition

### 4.2 Reactive Mode (Post-UPS Re-solve)

$$\boxed{\text{Max Profit}_{rem} = \sum_{b \in \mathcal{B}_{rem}: a_b=1} R_{\text{sku}(b)} - c^{tard} \sum_j \text{tard}_j - c^{stock} \sum_{l,t} \text{stockout}_{l,t} - c^{over} \sum_{r,t} \text{over}_{r,t} - c^{idle} \sum_{r,t} \text{idle}_{r,t}}$$

Where additionally:
- $\text{stockout}_{l,t} = \max(0, -B_l(t))$ — deficit at consumption events (soft)
- $\text{over}_{r,t} = 1$ if roaster $r$ is idle **because** RC on its output line = 40 (forced idle, distinct from safety-idle)
- $\mathcal{B}_{rem}$ = remaining unstarted batches only; completed batches frozen

**All five terms are now active** because UPS can make both stockout and overflow-idle unavoidable.

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
    
    # Stockout cost: -c_stock for each stockout minute this step
    for l in [L1, L2]:
        if new_state.rc_stock[l] < 0 and t in E[l]:
            reward -= 1500
    
    # Overstock-idle cost: -c_over for each roaster blocked by overflow
    for r in roasters:
        if new_state.status[r] == IDLE and new_state.rc_stock[output_line(r)] >= 40:
            reward -= 50
    
    # Safety-idle cost: -c_idle for each idle roaster when RC < safety
    for r in roasters:
        if new_state.status[r] in [IDLE, SETUP]:
            if new_state.rc_stock[line_of(r)] < 20:
                if r not in planned_downtime_at(t) and new_state.status[r] != DOWN:
                    reward -= 200
    
    return reward
```

**Key property:** The cumulative reward across all 480 time steps equals the shift profit from the objective function. No arbitrary normalization needed.

---

## 5. Cost Ratio Analysis

| Cost Item | $/min | $/batch-equivalent | Priority Rank |
|-----------|-------|-------------------|---------------|
| **Stockout** | $1,500/min | 1 min ≈ 0.38 PSC batches lost | **1st (highest)** |
| **Tardiness** | $1,000/min | 7 min late = 1 NDG batch revenue wiped | **2nd** |
| **Safety-idle** | $200/min | 20 min idle = 1 PSC batch revenue wiped | **3rd** |
| **Overstock-idle** | $50/min | 80 min idle = 1 PSC batch revenue wiped | **4th (lowest)** |

**Breakeven analysis:**
- Is it worth a 5-min setup to switch from PSC to NDG?
  - Revenue gain: $7,000 (NDG) − $4,000 (PSC would have produced) = $3,000
  - Setup cost: 5 min × $200 (if RC < 20) = $1,000
  - **Net: +$2,000 → YES** (even under low stock, NDG is worth switching to)
  - But: if RC is at 5 batches (critical), 5 min setup → 1 consumption event during setup → potential stockout at $1,500/min → **reconsider**

- Is it worth routing R3 to Line 1 (rescuing low stock)?
  - Benefit: avoid ~5 min of potential stockout on L1 = $7,500 saved
  - Cost: L2 gets no R3 output for 15 min → L2 stock drops by 1 extra batch
  - **Net: strongly positive** whenever L1 is at risk of stockout

---

## 6. Parameter Sensitivity Notes

These cost values are **starting points**, not final. The thesis should acknowledge:

1. **Absolute values are proxy costs** — not derived from audited factory financials. Real revenue/cost per batch is confidential.
2. **Relative ratios are what drive scheduling decisions.** The key ratios are:
   - $R^{MTO} / R^{PSC} = 1.75$ (MTO batches are more valuable)
   - $c^{stock} / R^{PSC} = 0.375$ per minute (stockout quickly exceeds batch value)
   - $c^{tard} / R^{MTO} = 0.143$ per minute (7 min late wipes out one batch's revenue)
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
║    RC stockout:            −$1,500 / min (per line)    ║
║    Overflow-idle:          −$50 / min (per roaster)    ║
║    Safety-idle:            −$200 / min (per roaster)   ║
║       (when RC < 20 batches on roaster's line)         ║
║                                                        ║
║  THRESHOLDS                                            ║
║    Safety stock: θ_SS = 20 batches (half of max 40)    ║
║    Max buffer:   B̄ = 40 batches per line              ║
║                                                        ║
║  PRIORITY: Stockout > Tardiness > Idle > Overstock     ║
╚════════════════════════════════════════════════════════╝
```
