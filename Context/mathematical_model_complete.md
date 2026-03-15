# Mathematical Model — Complete Specification (v2)
## Dynamic Batch Roasting Scheduling with Shared Pipeline Constraints under Unplanned Disruptions

> **Purpose:** Self-contained mathematical formulation ready for direct CP-SAT/MILP implementation. Every symbol is defined, explained, and illustrated with a concrete example. Cost structure references `cost.md` as the single source of truth for monetary values.
>
> **Solver roles:**
> - **CP-SAT (Google OR-Tools):** Primary solver for both deterministic schedule and reactive re-solve after UPS
> - **MILP (e.g., CPLEX, Gurobi, or OR-Tools MILP):** Solves the same deterministic model as a benchmark — provides LP relaxation lower bound to verify CP-SAT solution quality
> - **DRL (MaskablePPO):** Does not use this model directly — uses the simulation environment. But the model defines the constraints the DRL action mask enforces

---

# PART I — MODEL INPUTS

Everything in this section is **given before solving**. The solver does not decide these values.

---

## 1. Sets

### 1.1 Time Horizon

$$\mathcal{T} = \{0, 1, 2, \dots, 479\}$$

480-minute shift, 1-minute slots. Slot 0 = shift start. Slot 479 = last minute.

### 1.2 Production Lines

$$\mathcal{L} = \{L_1, L_2\}$$

### 1.3 Roasters

$$\mathcal{R} = \{R_1, R_2, R_3, R_4, R_5\}$$

$$\mathcal{R}_{L_1} = \{R_1, R_2\} \qquad \mathcal{R}_{L_2} = \{R_3, R_4, R_5\}$$

### 1.4 SKU Types

$$\mathcal{K} = \{k^{PSC}, k^{NDG}, k^{BUS}\}$$

### 1.5 Roaster Eligibility

| Roaster | $\mathcal{K}_r$ | Reverse: $\mathcal{R}_k$ |
|---------|-----------------|--------------------------|
| $R_1$ | $\{k^{PSC}, k^{NDG}\}$ | $\mathcal{R}_{PSC} = \{R_1..R_5\}$ |
| $R_2$ | $\{k^{PSC}, k^{NDG}, k^{BUS}\}$ | $\mathcal{R}_{NDG} = \{R_1, R_2\}$ |
| $R_3$ | $\{k^{PSC}\}$ | $\mathcal{R}_{BUS} = \{R_2\}$ |
| $R_4$ | $\{k^{PSC}\}$ | |
| $R_5$ | $\{k^{PSC}\}$ | |

### 1.6 Pipeline Mapping

$$\text{pipe}(r) = \begin{cases} L_1 & \text{if } r \in \{R_1, R_2\} \\ L_2 & \text{if } r \in \{R_3, R_4, R_5\} \end{cases}$$

R3 **always** uses Line 2 pipeline regardless of RC output direction.

### 1.7 Line-of-Roaster Mapping (for idle penalty)

$$\ell(r) = \begin{cases} L_1 & \text{if } r \in \{R_1, R_2\} \\ L_2 & \text{if } r \in \{R_3, R_4, R_5\} \end{cases}$$

> Note: $\ell(r) = \text{pipe}(r)$ for all roasters. But $\text{out}(b)$ (output line) can differ from $\ell(r)$ for R3.

### 1.8 MTO Jobs

$$\mathcal{J}^{MTO} = \{j_1, j_2, \dots\}$$

Each job $j$: SKU type $\text{sku}(j)$, batch count $n_j$, eligible roasters $\mathcal{R}_{\text{sku}(j)}$.

> **Example:** $j_1$: NDG × 3 batches, $j_2$: Busta × 1 batch. Total MTO = 4 batches.

### 1.9 MTO Batches

$$\mathcal{B}^{MTO} = \bigcup_{j \in \mathcal{J}^{MTO}} \{b_1^j, b_2^j, \dots, b_{n_j}^j\}$$

Always active ($a_b = 1$ for all $b \in \mathcal{B}^{MTO}$).

### 1.10 PSC Batch Pool

$$\mathcal{B}^{PSC} = \{b^{PSC}_1, \dots, b^{PSC}_{160}\}$$

Pool size: $5 \times \lfloor 480/15 \rfloor = 160$ optional batches. Solver activates as many as needed.

### 1.11 All Batches

$$\mathcal{B} = \mathcal{B}^{MTO} \cup \mathcal{B}^{PSC}$$

---

## 2. Parameters

### 2.1 Timing

| Symbol | Value | Description |
|--------|-------|-------------|
| $P$ | 15 min | Processing time per batch (uniform, all SKUs) |
| $\delta^{con}$ | 3 min | Pipeline consume duration |
| $\sigma$ | 5 min | Setup time between any two different SKUs |
| $D^{MTO}$ | 240 | MTO soft due date (slot 240 = half-shift) |

### 2.2 Inventory

| Symbol | Value | Description |
|--------|-------|-------------|
| $\overline{B}_l$ | **40 batches** | Maximum RC buffer per line (= 20,000 kg / 500 kg per batch) |
| $\theta^{SS}$ | **20 batches** | Safety stock threshold (= $\overline{B}_l / 2$). Idle penalty active below this. |
| $B^0_l$ | input | Initial RC stock per line (batch units) |
| $\rho_l$ | input | PSC consumption rate (minutes per batch) |

### 2.3 Cost Structure (see `cost.md` for full detail)

| Symbol | Value | Unit | Description |
|--------|-------|------|-------------|
| $R^{PSC}$ | $4,000 | per batch | Revenue per completed PSC batch |
| $R^{NDG}$ | $7,000 | per batch | Revenue per completed NDG batch |
| $R^{BUS}$ | $7,000 | per batch | Revenue per completed Busta batch |
| $c^{tard}$ | $1,000 | per min | MTO tardiness penalty |
| $c^{stock}$ | $1,500 | per min per line | RC stockout penalty (reactive mode only) |
| $c^{over}$ | $50 | per min per roaster | Overflow-idle penalty (roaster blocked by full RC) |
| $c^{idle}$ | $200 | per min per roaster | Safety-idle penalty (idle while $B_l < \theta^{SS}$) |

**Priority hierarchy:** $c^{stock} > c^{tard} > c^{idle} > c^{over}$

### 2.4 PSC Consumption Schedule (Precomputed)

$$\mathcal{E}_l = \left\{ \lfloor i \cdot \rho_l \rfloor \;\middle|\; i = 1, 2, \dots, \lfloor 480 / \rho_l \rfloor \right\}$$

> **Example:** $\rho_{L_1} = 5.1$ → $\mathcal{E}_{L_1} = \{5, 10, 15, 20, 25, 30, 35, 40, 45, 51, \dots, 479\}$, ~94 events.

### 2.5 Planned Downtime

$$\mathcal{D}_r \subset \mathcal{T} \quad \forall r \in \mathcal{R}$$

Known windows where roaster $r$ is unavailable.

### 2.6 Initial SKU State

$$\text{sku}^0_r = k^{PSC} \quad \forall r \in \mathcal{R}$$

All roasters start the shift as if their last batch was PSC. Setup is needed if the first batch on a roaster is NDG or Busta.

> **Example:** R1's first batch is NDG → 5-min setup [0, 5) before NDG can start at slot 5.
> R3's first batch is PSC → no setup, can start at slot 0.

### 2.7 Complete Example Instance

```
TIME:          480 slots. P=15, δ_con=3, σ=5. D_MTO=240.
ROASTERS:      R1(L1), R2(L1), R3(L2), R4(L2), R5(L2). All start last_sku=PSC.
MTO:           j1: NDG×3, j2: Busta×1 (4 MTO batches total)
PSC:           ρ_L1=5.1, ρ_L2=4.8. Pool: 160 optional batches.
RC:            B⁰_L1=12, B⁰_L2=15. B̄=40. θ_SS=20.
DOWNTIME:      R3: {200,...,229} (30 min)
COSTS:         R_PSC=$4k, R_NDG=$7k, R_BUS=$7k, c_tard=$1k, c_stock=$1.5k,
               c_over=$50, c_idle=$200
```

---

# PART II — DECISION VARIABLES

---

## 3. Decision Variables

### 3.1 PSC Batch Activation: $a_b$

$$a_b \in \{0, 1\} \quad \forall b \in \mathcal{B}^{PSC}$$

1 = batch scheduled, 0 = not. MTO batches: $a_b = 1$ always (fixed).

> **Example:** Solver activates 62 of 160 PSC pool batches → schedule has 62 + 4 = 66 active batches.

### 3.2 Roaster Assignment: $r_b$

$$r_b \in \mathcal{R}_{\text{sku}(b)} \quad \forall b : a_b = 1$$

> Busta → $R_2$ only. NDG → $\{R_1, R_2\}$. PSC → $\{R_1..R_5\}$.

### 3.3 Start Time: $s_b$

$$s_b \in [0, 465] \quad \forall b : a_b = 1$$

**End-of-shift constraint baked in:** $s_b \leq 480 - P = 465$ ensures every batch completes within the shift. Derived end time: $e_b = s_b + P = s_b + 15$.

> **Example:** Batch at $s_b = 465$ → $e_b = 480$ (completes exactly at shift end). Batch at $s_b = 466$ would end at 481 → not allowed.

### 3.4 R3 Output Routing: $y_b$

$$y_b \in \{0, 1\} \quad \forall b : r_b = R_3, \; a_b = 1$$

$y_b = 1$: RC output → Line 1. $y_b = 0$: RC output → Line 2.

**Experimental factor:** In "fixed R3" mode, $y_b = 0$ for all R3 batches. In "flexible R3" mode, $y_b$ is free.

### 3.5 Output Line Function (Derived)

$$\text{out}(b) = \begin{cases} L_1 & \text{if } r_b \in \{R_1, R_2\} \\ L_1 & \text{if } r_b = R_3 \wedge y_b = 1 \\ L_2 & \text{if } r_b = R_3 \wedge y_b = 0 \\ L_2 & \text{if } r_b \in \{R_4, R_5\} \end{cases}$$

### 3.6 Variable Summary

| Variable | Domain | Count | Description |
|----------|--------|-------|-------------|
| $a_b$ | $\{0,1\}$ | 160 | PSC activation |
| $r_b$ | $\mathcal{R}_b$ | ≤164 | Roaster assignment |
| $s_b$ | $[0, 465]$ | ≤164 | Start time |
| $y_b$ | $\{0,1\}$ | ≤32 | R3 routing |

---

## 4. Auxiliary Variables

### 4.1 Batch End Time

$$e_b = s_b + 15 \quad \forall b : a_b = 1$$

### 4.2 Consume Interval

$$\text{con}(b) = [s_b, \; s_b + 3) \quad \text{on pipeline } \text{pipe}(r_b)$$

### 4.3 Roaster Interval

$$\text{roast}(b) = [s_b, \; s_b + 15) \quad \text{on roaster } r_b$$

### 4.4 MTO Tardiness

$$\text{tard}_j = \max\left(0,\; \max_{b \in \mathcal{B}_j} e_b - D^{MTO}\right) \quad \forall j \in \mathcal{J}^{MTO}$$

> **Example:** Last NDG batch ends at 248 → $\text{tard} = 8$ min → cost = $8 \times \$1{,}000 = \$8{,}000$.

### 4.5 RC Inventory Level

$$B_l(t) = B^0_l + \sum_{\substack{b: a_b=1,\; \text{out}(b)=l,\\ \text{sku}(b)=k^{PSC},\; e_b \leq t}} 1 \;\;-\;\; \left|\{\tau \in \mathcal{E}_l : \tau \leq t\}\right|$$

> Only **PSC** batches credit RC. NDG/Busta output is delivered directly — does NOT enter RC stock.

> **CP-SAT note:** Track $B_l(t)$ only at change points (batch completions + consumption events) — ~150 constraints per line, not 480.

### 4.6 Stockout Variable (Reactive Mode)

$$\text{stockout}_{l,t} = \max(0, -B_l(t)) \quad \forall l, t$$

Zero in deterministic mode (hard constraint forces $B_l(t) \geq 0$).

### 4.7 Roaster Busy Indicator

$$\text{busy}_{r,t} = \begin{cases} 1 & \text{if } \exists\; b : a_b=1, r_b=r, s_b \leq t < e_b \\ 0 & \text{otherwise} \end{cases}$$

Roaster $r$ is actively processing a batch at time $t$.

> **CP-SAT:** Derived from interval variables. For MILP: $\text{busy}_{r,t} = \sum_{b: r_b=r} \sum_{\tau=\max(0,t-14)}^{t} x_{b,r,\tau}$ (if using time-indexed formulation).

### 4.8 Safety-Idle Indicator

$$\text{idle}_{r,t} = \begin{cases} 1 & \text{if } \text{busy}_{r,t} = 0 \;\wedge\; t \notin \mathcal{D}_r \;\wedge\; B_{\ell(r)}(t) < \theta^{SS} \\ 0 & \text{otherwise} \end{cases}$$

Roaster is idle (not running, not in planned downtime) AND RC stock on its line is below safety threshold (20 batches). Includes SETUP time (roaster not "busy" during setup).

> **Example:** R1 in SETUP [45, 50). RC Line 1 = 18 < 20. → $\text{idle}_{R_1, t} = 1$ for $t \in \{45..49\}$. Cost: $5 \times \$200 = \$1{,}000$.

### 4.9 Overflow-Idle Indicator

$$\text{over}_{r,t} = \begin{cases} 1 & \text{if } \text{busy}_{r,t} = 0 \;\wedge\; t \notin \mathcal{D}_r \;\wedge\; B_{\text{out}^*(r)}(t) = \overline{B}_l \\ 0 & \text{otherwise} \end{cases}$$

Where $\text{out}^*(r)$ is the default output line of roaster $r$ (L1 for R1/R2, L2 for R4/R5). For R3 with flexible routing, overflow-idle only applies if **both** lines are at max_buffer (otherwise R3 can route to the non-full line).

> **Example:** R4 idle at t=300. $B_{L_2}(300) = 40$. → $\text{over}_{R_4, 300} = 1$. Cost: $\$50$.

---

# PART III — CONSTRAINTS

---

## 5. Constraints

### C1–C2: Batch Activation

**(C1)** MTO always active: $a_b = 1 \quad \forall b \in \mathcal{B}^{MTO}$

**(C2)** PSC optional: $a_b \in \{0, 1\} \quad \forall b \in \mathcal{B}^{PSC}$

### C3: Roaster Eligibility

$$r_b \in \mathcal{R}_{\text{sku}(b)} \quad \forall b : a_b = 1$$

### C4: Roaster NoOverlap

$$\text{NoOverlap}\left(\{[s_b, s_b+15) : r_b = r, a_b = 1\}\right) \quad \forall r \in \mathcal{R}$$

No two batches simultaneously on the same roaster. 5 separate NoOverlap sets (one per roaster).

### C5: Sequence-Dependent Setup Time

For consecutive batches $b_1, b_2$ on the same roaster with $\text{sku}(b_1) \neq \text{sku}(b_2)$:

$$s_{b_2} \geq e_{b_1} + \sigma = s_{b_1} + 20$$

**CP-SAT:** NoOverlap with transition matrix:
```
transition[k1][k2] = 0 if k1==k2, else 5
```

### C6: Initial SKU Setup

The first batch on each roaster must respect setup from the initial SKU state ($k^{PSC}$):

$$s_{b_{first}^r} \geq \sigma = 5 \quad \text{if } \text{sku}(b_{first}^r) \neq k^{PSC}, \quad \forall r \in \mathcal{R}$$

> **Example:** R1's first batch is NDG → cannot start before slot 5.
> R4's first batch is PSC → can start at slot 0.

**CP-SAT:** Add a phantom fixed interval $[−\sigma, 0)$ with SKU = PSC on each roaster's NoOverlap set with the transition matrix. This automatically enforces setup for different-SKU first batches.

### C7: Planned Downtime

$$[s_b, s_b + P - 1] \cap \mathcal{D}_r = \emptyset \quad \forall r, \forall b : r_b = r, a_b = 1$$

Batch must **fully complete** before downtime starts. No mid-batch pause.

> **Example:** $\mathcal{D}_{R_3} = \{200..229\}$. Last valid start = 185 ($185 + 14 = 199 < 200$). Start at 186 → $186+14=200$ → overlap → VIOLATION.

**CP-SAT:** Add fixed downtime intervals to roaster NoOverlap sets.

### C8: Pipeline NoOverlap (Core Bottleneck)

$$\text{NoOverlap}\left(\{[s_b, s_b+3) : a_b = 1, \text{pipe}(r_b) = l\}\right) \quad \forall l \in \mathcal{L}$$

**Line 1 pipeline:** R1, R2 consume intervals.
**Line 2 pipeline:** R3, R4, R5 consume intervals.

2 separate NoOverlap sets (one per line). Line 2 utilization: 9/15 = 60%.

### C9: End-of-Shift

$$s_b \leq 465 \quad \forall b : a_b = 1$$

Ensures $e_b = s_b + 15 \leq 480$. Baked into the domain of $s_b$.

### C10: RC Inventory Lower Bound (Stockout)

**Deterministic mode (hard):**
$$B_l(t) \geq 0 \quad \forall l \in \mathcal{L}, \; \forall t \in \mathcal{T}$$

**Reactive mode (soft):** Relaxed — stockout tracked by $\text{stockout}_{l,t}$ and penalized at $c^{stock} = \$1{,}500$/min.

> Only needs enforcement at consumption event times $\tau \in \mathcal{E}_l$ (~94 per line).

### C11: RC Inventory Upper Bound (Overflow)

$$B_l(t) \leq \overline{B}_l = 40 \quad \forall l \in \mathcal{L}, \; \forall t \in \mathcal{T}$$

**Hard in all modes.** Overflow is physical impossibility — silo is full, roaster cannot dump output.

> Only needs enforcement at batch completion times $e_b$ for PSC batches on line $l$.

### C12: MTO Tardiness Computation

$$\text{tard}_j \geq e_b - D^{MTO} \quad \forall j \in \mathcal{J}^{MTO}, \forall b \in \mathcal{B}_j$$
$$\text{tard}_j \geq 0$$

Solver minimizes $\text{tard}_j$ through the cost term $c^{tard} \cdot \text{tard}_j$.

### C13: R3 Output Routing

$$y_b \in \{0, 1\} \quad \forall b : r_b = R_3, a_b = 1$$

Links to $B_l(t)$ through $\text{out}(b)$.

### C14: Safety-Idle Detection

$$\text{idle}_{r,t} \geq (1 - \text{busy}_{r,t}) + \text{low}_{l,t} - 1 \quad \forall r, t \notin \mathcal{D}_r$$

Where $\text{low}_{l,t} = 1$ iff $B_{\ell(r)}(t) < \theta^{SS} = 20$:

$$B_{\ell(r)}(t) \leq \theta^{SS} - 1 + M(1 - \text{low}_{l,t})$$
$$B_{\ell(r)}(t) \geq \theta^{SS} - M \cdot \text{low}_{l,t}$$

$\text{idle}_{r,t} = 1$ requires BOTH: roaster not busy AND RC below safety stock.

Since the objective minimizes idle cost, the solver sets $\text{idle}_{r,t} = 0$ whenever it can — the one-sided inequality suffices (no need to force $\text{idle}_{r,t} = 1$).

> **Scope:** Applies only when roaster is not in planned downtime ($t \notin \mathcal{D}_r$). During downtime, idle is not penalized (roaster can't help being down). During UPS-induced DOWN: also not penalized in the model (UPS is uncontrollable).

### C15: Overflow-Idle Detection

$$\text{over}_{r,t} \geq (1 - \text{busy}_{r,t}) + \text{full}_{l,t} - 1 \quad \forall r, t \notin \mathcal{D}_r$$

Where $\text{full}_{l,t} = 1$ iff $B_{\text{out}^*(r)}(t) = \overline{B}_l = 40$:

$$B_l(t) \geq \overline{B}_l - M(1 - \text{full}_{l,t})$$
$$B_l(t) \leq \overline{B}_l - 1 + M \cdot \text{full}_{l,t}$$

> **Note:** Overflow-idle penalty is much smaller ($50/min) than safety-idle ($200/min) and stockout ($1,500/min). It gently discourages scheduling patterns that lead to full RC buffers.

### Constraint Summary

| ID | Constraint | Type | Penalty if violated | Count |
|----|-----------|------|--------------------|----|
| C1 | MTO always active | Hard | — | $|\mathcal{B}^{MTO}|$ |
| C2 | PSC optional | — | — | $|\mathcal{B}^{PSC}|$ |
| C3 | Roaster eligibility | Hard | — | $|\mathcal{B}|$ |
| C4 | Roaster NoOverlap | Hard | — | 5 sets |
| C5 | Setup time (transitions) | Hard | — | via C4 matrix |
| C6 | Initial SKU setup | Hard | — | 5 (one per roaster) |
| C7 | Planned downtime | Hard | — | per roaster |
| C8 | Pipeline NoOverlap | Hard | — | 2 sets |
| C9 | End-of-shift | Hard | — | $|\mathcal{B}|$ |
| C10 | RC ≥ 0 (stockout) | Hard/Soft | $c^{stock}=\$1{,}500$/min | ~94×2 lines |
| C11 | RC ≤ 40 (overflow) | Hard | — | ~66×2 lines |
| C12 | Tardiness computation | Soft | $c^{tard}=\$1{,}000$/min | $|\mathcal{J}^{MTO}|$ |
| C13 | R3 routing | — | — | ≤32 |
| C14 | Safety-idle detection | Soft | $c^{idle}=\$200$/min | 5×480 |
| C15 | Overflow-idle detection | Soft | $c^{over}=\$50$/min | 5×480 |

**Total logical groups: 15.** Expands to ~3,000+ individual constraints when instantiated (dominated by C14/C15 time-indexed indicators, which can be reduced by only tracking at change points).

---

# PART IV — OBJECTIVE FUNCTION

---

## 6. Objective: Maximize Profit

### 6.1 Deterministic Mode

$$\boxed{\text{Maximize:} \quad \underbrace{\sum_{b: a_b=1} R_{\text{sku}(b)}}_{\text{Revenue}} \;-\; \underbrace{c^{tard} \sum_j \text{tard}_j}_{\text{Tardiness}} \;-\; \underbrace{c^{idle} \sum_{r,t} \text{idle}_{r,t}}_{\text{Safety-Idle}} \;-\; \underbrace{c^{over} \sum_{r,t} \text{over}_{r,t}}_{\text{Overflow-Idle}}}$$

**No stockout term** — stockout is a hard constraint in deterministic mode.

**Revenue breakdown:**
- Each activated PSC batch: +$4,000
- Each MTO batch (always active): +$7,000
- MTO revenue is constant ($7,000 × 4 = $28,000 for the example instance), but still included for completeness

> **Example:**
> | Schedule | PSC | MTO tard | Idle(RC<20) | Over-idle | Revenue | Costs | **Profit** |
> |----------|-----|----------|-------------|-----------|---------|-------|-----------|
> | A | 65 | 0 min | 30 min | 0 min | $288k | $6k | **$282k** |
> | B | 68 | 8 min | 10 min | 5 min | $300k | $10.25k | **$289.75k** |
> | C | 60 | 0 min | 80 min | 0 min | $268k | $16k | **$252k** |
>
> Schedule B wins despite 8-min tardiness — the extra 3 PSC batches ($12k) more than offset the tardiness cost ($8k).

### 6.2 Reactive Mode (Post-UPS)

$$\boxed{\text{Max:} \sum_{b \in \mathcal{B}_{rem}} R_{\text{sku}(b)} \cdot a_b \;-\; c^{tard}\sum_j \text{tard}_j \;-\; c^{stock}\sum_{l,t} \text{stockout}_{l,t} \;-\; c^{idle}\sum_{r,t} \text{idle}_{r,t} \;-\; c^{over}\sum_{r,t} \text{over}_{r,t}}$$

All five terms active. Stockout is now soft-penalized at $1,500/min.

### 6.3 DRL Reward (Per-Step Decomposition)

```python
def reward(prev, new, t):
    r = 0
    # Revenue: completed batches this step
    for b in newly_completed(prev, new):
        r += {PSC: 4000, NDG: 7000, BUS: 7000}[b.sku]
    # Tardiness: new minutes of MTO lateness
    r -= 1000 * (new.tardiness - prev.tardiness)
    # Stockout: consumption event with empty RC
    for l in [L1, L2]:
        if new.rc[l] < 0 and t in E[l]:
            r -= 1500
    # Safety-idle: each idle roaster with low RC
    for roaster in R:
        if not new.busy[roaster] and roaster not in downtime_at(t):
            if new.rc[line_of(roaster)] < 20:
                r -= 200
            if new.rc[out_line(roaster)] >= 40 and new.status[roaster] == IDLE:
                r -= 50
    return r
```

Cumulative reward across 480 steps = shift profit.

---

# PART V — MILP BENCHMARK

---

## 7. MILP Role: Deterministic Optimality Verification

### 7.1 Purpose

MILP (Mixed-Integer Linear Programming) solves **the same deterministic model** as CP-SAT — identical sets, parameters, variables, constraints, and objective. It serves as a **solver benchmark**, not a separate methodology:

1. **LP relaxation lower bound:** MILP's linear relaxation provides a provable lower bound on the optimal objective. If CP-SAT's solution value equals (or is very close to) the MILP lower bound, CP-SAT has found a provably optimal (or near-optimal) solution.
2. **Solution quality verification:** On small instances, both solvers should find the same optimal schedule. On larger instances, CP-SAT may find better solutions faster (due to native NoOverlap handling), while MILP provides the bound.
3. **Solve time comparison:** Reported as a supplementary metric — how long each solver takes on the deterministic model.

### 7.2 MILP Does NOT Participate in the Reactive Experiments

The 90-cell × 100-rep factorial experiment (with UPS) uses **only** Dispatching, CP-SAT re-solve, and DRL. MILP is not a reactive strategy — it solves once (deterministic) and provides a quality benchmark.

**MILP experimental scope:** Solve the same deterministic instances that CP-SAT solves at shift start ($\lambda = 0$ case). Compare objective value and solve time. Report LP relaxation gap.

### 7.3 MILP Formulation Notes

The constraints in §5 translate to MILP as follows:

| CP-SAT Feature | MILP Equivalent |
|---------------|-----------------|
| `IntervalVar` + `NoOverlap` (C4, C8) | Time-indexed binary $x_{b,r,t}$ or Big-M disjunctive constraints |
| `OptionalIntervalVar` (C2) | Activation binary $a_b$ with Big-M linking |
| Transition matrix (C5) | Pairwise ordering constraints with Big-M |
| `NewBoolVar` for $y_b$ (C13) | Standard binary variable |
| $\text{idle}_{r,t}$, $\text{over}_{r,t}$ (C14, C15) | Binary indicators with Big-M linearization |

**Expected outcome:** MILP will have more variables (time-indexed binaries dominate) and weaker LP relaxation (Big-M for disjunctive constraints). CP-SAT's native NoOverlap propagation should outperform MILP on solve time. This confirms CP-SAT as the right solver choice for reactive re-solve.

---

# PART VI — DRL ACTION SPACE

---

## 8. DRL Agent Interface

### 8.1 Action Space (17 Actions)

| ID | Action | Eligibility |
|----|--------|-------------|
| 0 | PSC on R1 → L1 | R1 IDLE, pipeline L1 free |
| 1 | PSC on R2 → L1 | R2 IDLE, pipeline L1 free |
| 2 | PSC on R3 → **L1** | R3 IDLE, pipeline L2 free, $B_{L1} < 40$ |
| 3 | PSC on R3 → **L2** | R3 IDLE, pipeline L2 free, $B_{L2} < 40$ |
| 4 | PSC on R4 → L2 | R4 IDLE, pipeline L2 free |
| 5 | PSC on R5 → L2 | R5 IDLE, pipeline L2 free |
| 6 | NDG on R1 | R1 IDLE, pipeline L1 free (or after setup), NDG remaining > 0 |
| 7 | NDG on R2 | R2 IDLE, pipeline L1 free (or after setup), NDG remaining > 0 |
| 8 | Busta on R2 | R2 IDLE, pipeline L1 free (or after setup), Busta remaining > 0 |
| 9–15 | *(permanently masked)* | NDG on R3/R4/R5, Busta on R1/R3/R4/R5 — never valid |
| 16 | WAIT | Always valid |

**Per-roaster decision:** When roaster $r$ becomes IDLE, agent is called for $r$ only. Actions for other roasters are masked. Effective valid actions per call: 1–4.

### 8.2 Observation Vector (21 features)

| Index | Feature | Range |
|-------|---------|-------|
| 0 | $t / 479$ | [0, 1] |
| 1–5 | $\text{status}[R_i]$ | 0=IDLE, 0.33=SETUP, 0.67=RUN, 1.0=DOWN |
| 6–10 | $\text{remaining}[R_i] / 15$ | [0, 1] |
| 11–15 | $\text{last\_sku}[R_i]$ | 0=PSC, 0.5=NDG, 1.0=Busta |
| 16 | $B_{L_1} / 40$ | [0, 1] |
| 17 | $B_{L_2} / 40$ | [0, 1] |
| 18 | MTO remaining / total MTO | [0, 1] |
| 19 | pipeline\_busy$[L_1]$ / 3 | [0, 1] |
| 20 | pipeline\_busy$[L_2]$ / 3 | [0, 1] |

---

# PART VII — OUTPUTS

---

## 9. Model Output: Shift Schedule + Profit Report

### 9.1 Example Schedule (abbreviated)

```
Batch | SKU   | Roaster | Start | End | Pipeline   | RC Out | Setup? | Revenue
------+-------+---------+-------+-----+------------+--------+--------+--------
b_m1  | NDG   | R1      |   5   |  20 | L1 [5,7]   | —      | Yes(5m)| $7,000
b_m2  | NDG   | R1      |  20   |  35 | L1 [20,22] | —      | No     | $7,000
b_m3  | NDG   | R1      |  35   |  50 | L1 [35,37] | —      | No     | $7,000
b_m4  | Busta | R2      |   5   |  20 | L1 [5,7]** | —      | Yes(5m)| $7,000
 ** PIPELINE CONFLICT with b_m1 → corrected: b_m4 starts at t=8
b_m4  | Busta | R2      |   8   |  23 | L1 [8,10]  | —      | Yes(5m)| $7,000
b_p1  | PSC   | R2      |  28   |  43 | L1 [28,30] | L1     | Yes(5m)| $4,000
b_p2  | PSC   | R3      |   0   |  15 | L2 [0,2]   | L2     | No     | $4,000
b_p3  | PSC   | R3      |  15   |  30 | L2 [15,17] | L1     | No     | $4,000
b_p4  | PSC   | R4      |   3   |  18 | L2 [3,5]   | L2     | No     | $4,000
b_p5  | PSC   | R5      |   6   |  21 | L2 [6,8]   | L2     | No     | $4,000
...   (62 PSC + 4 MTO = 66 total batches)
```

### 9.2 Profit Report

```
═══════════════════════════════════════════════════
PROFIT REPORT
═══════════════════════════════════════════════════
REVENUE:
  PSC batches completed: 62 × $4,000 = $248,000
  NDG batches completed:  3 × $7,000 =  $21,000
  Busta completed:        1 × $7,000 =   $7,000
  Total Revenue:                        $276,000

COSTS:
  MTO tardiness:     3 min × $1,000 =    $3,000
  Stockout:          0 min × $1,500 =        $0
  Safety-idle:      25 min × $200   =    $5,000
    (R1 setup 5m + R2 setup 5m + various idle under low stock)
  Overflow-idle:     0 min × $50    =        $0
  Total Costs:                            $8,000

NET PROFIT:  $276,000 − $8,000 = $268,000

MILP LP Bound:       $271,500 (optimality gap: 1.3%)
CP-SAT Solve Time:   0.4 seconds
MILP Solve Time:     12.3 seconds
═══════════════════════════════════════════════════
```

---

# PART VIII — EXPERIMENTAL DESIGN

---

## 10. Factorial Experiment

| Factor | Levels | Count |
|--------|--------|-------|
| UPS rate $\lambda$ | 0, 1, 2, 3, 5 | 5 |
| UPS duration $\mu$ | 10, 20, 30 min | 3 |
| Strategy | Dispatching / CP-SAT / DRL | 3 |
| R3 routing | Fixed / Flexible | 2 |

**90 cells × 100 reps = 9,000 runs** (reactive experiment).

**+ MILP benchmark:** Run on the $\lambda = 0$ instances only (deterministic). Compare CP-SAT vs. MILP on objective value, solve time, LP gap. This is a separate experiment (~200 runs: 100 reps × 2 R3 modes).

**Paired comparison:** Same UPS realizations across all 3 reactive strategies per cell. Differences are purely from strategy.

**KPIs:** Total profit ($), PSC throughput, stockout count/duration, MTO tardiness, compute time per decision.

---

# APPENDIX — Quick Reference

```
╔════════════════════════════════════════════════════════════════════╗
║                    MODEL v2 AT A GLANCE                           ║
╠════════════════════════════════════════════════════════════════════╣
║ Roasters:    5 (R1-R2 on L1, R3-R5 on L2)                       ║
║ Time:        480 slots, s_b ∈ [0, 465]                           ║
║ Roast:       15 min. Consume: 3 min (concurrent). Setup: 5 min.  ║
║ RC:          Aggregate counter, max 40/line, safety 20/line       ║
║ Initial SKU: PSC (setup needed for first MTO batch)               ║
║ Objective:   MAX PROFIT = Revenue − Tard − Stockout − Idle       ║
║ Revenue:     PSC $4k, NDG $7k, Busta $7k per batch               ║
║ Penalties:   Stockout $1.5k/min, Tard $1k/min, Idle $200/min,    ║
║              Over-idle $50/min                                    ║
║ Constraints: 15 groups (C1–C15)                                   ║
║ Solvers:     CP-SAT (primary), MILP (deterministic benchmark)     ║
║ DRL:         17 actions, 21-dim observation, MaskablePPO          ║
║ Experiments: 90 cells × 100 reps = 9,000 runs + MILP benchmark   ║
╚════════════════════════════════════════════════════════════════════╝
```
