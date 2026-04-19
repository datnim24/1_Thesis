# Mathematical Model — Reactive Re-Solve Under Unplanned Disruptions
### Dynamic Batch Roasting Scheduling at Nestlé Trị An

> **Purpose of this document:** This is the **canonical** mathematical formulation for the reactive scheduling problem — invoked after an Unplanned Stoppage (UPS) disrupts the running schedule. This model is solved by CP-SAT (event-triggered re-optimization). The DRL agent and dispatching heuristic operate in the simulation environment, not on this model directly, but must satisfy the same constraints.
>
> **Relationship to the deterministic model:** The deterministic model (`mathematical_model_complete.md`) solves a single optimization at shift start (t=0) with full information and no disruptions. This reactive model solves the **remaining** shift $[t_0, 479]$ given a disrupted state snapshot. It is structurally similar — same factory, same products, same constraint types — but with modified inputs, relaxed C10, a new penalty term, and frozen in-progress operations.
>
> **When is this model invoked?** Each time a UPS event occurs during the simulation. The simulation engine freezes the current state, builds this model from the snapshot, calls CP-SAT, and applies the resulting schedule to the remaining shift. Each re-solve is **stateless** — it does not remember previous re-solves. It optimizes based solely on the current state.

---

# PART I — TRIGGER AND STATE SNAPSHOT

---

## 1. UPS Trigger

An Unplanned Stoppage (UPS) occurs at time $t_0 \in \mathcal{T}$ on a specific roaster $r^{UPS}$. The UPS:

1. **Cancels** any batch currently running on $r^{UPS}$ — the batch is destroyed (GC lost, no RC credit, no revenue). The roaster interval and pipeline interval for this batch are removed from the model entirely.
2. **Blocks** $r^{UPS}$ for a duration $d^{UPS}$ minutes. The roaster cannot start any new batch during $[t_0, t_0 + d^{UPS} - 1]$.
3. **Triggers** a re-solve of the remaining shift $[t_0, 479]$.

> **Key principle:** The re-solve sees only what exists *right now*. Completed batches are gone (their revenue is locked in outside this model). In-progress batches on other roasters are frozen. The cancelled batch is erased. The solver optimizes from this point forward.

---

## 2. State Snapshot at $t_0$

The simulation engine provides the following state to the model builder. Every item is a **given input** — not a decision.

---

### 2.1 Time Parameters

| Symbol | Description |
|--------|-------------|
| $t_0$ | Current time (the moment UPS is detected) |
| $\mathcal{T}^{rem} = \{t_0, t_0+1, \dots, 479\}$ | Remaining time horizon |
| $SL^{rem} = 480 - t_0$ | Remaining minutes in shift |

---

### 2.2 Roaster States

For each roaster $r \in \mathcal{R}$, the simulation provides:

| Symbol | Type | Description |
|--------|------|-------------|
| $\text{status}_r(t_0)$ | enum | One of: IDLE, RUNNING, SETUP, DOWN |
| $\text{remaining}_r(t_0)$ | int $\geq 0$ | Minutes left in current operation (0 if IDLE) |
| $\text{last\_sku}_r(t_0)$ | SKU | The SKU the roaster is currently configured for |
| $\text{setup\_target}_r(t_0)$ | SKU or null | If in SETUP: the SKU being set up for. Null otherwise |

**How each status maps to model inputs:**

**IDLE:** Roaster is free. Available immediately for new batches. Initial SKU = $\text{last\_sku}_r(t_0)$.

**RUNNING:** Roaster is executing a batch that started before $t_0$ and will complete after $t_0$. This batch becomes a **fixed interval** in the model (see §2.5). The roaster is unavailable until the batch completes.

**SETUP:** Roaster is transitioning between SKUs. Setup started before $t_0$ and has $\sigma^{rem}_r = \text{remaining}_r(t_0)$ minutes left. This becomes a **phantom blocking interval** $[t_0, t_0 + \sigma^{rem}_r)$ on the roaster (see §2.6). After setup completes, the roaster's effective initial SKU is $\text{setup\_target}_r(t_0)$, not $\text{last\_sku}_r(t_0)$.

> **Important:** The pipeline is NOT occupied during SETUP — only the roaster is blocked.

**DOWN:** Roaster is already in a UPS or planned downtime. Has $d^{rem}_r = \text{remaining}_r(t_0)$ minutes left. Adds blocking interval $[t_0, t_0 + d^{rem}_r)$.

---

### 2.3 RC Inventory

| Symbol | Description |
|--------|-------------|
| $B_l(t_0)$ | Current RC stock on line $l$ at time $t_0$, in batches. Replaces $B^0_l$ from the deterministic model. |

> **Note:** $B_l(t_0)$ can be any non-negative integer $\in [0, 40]$. In principle, it could be negative if a previous re-solve allowed stockouts — but the simulation convention is to track actual stock (never below 0 physically; "stockout" means demand was unmet, not that stock went negative in the physical silo). For the re-solve model, we allow $B_l(t_0) \geq 0$.

---

### 2.4 Pipeline States

| Symbol | Description |
|--------|-------------|
| $\text{pipe\_busy}_l(t_0)$ | Minutes remaining on the current pipeline consume operation on line $l$. 0 = free. |

If $\text{pipe\_busy}_l(t_0) > 0$, a **fixed pipeline interval** $[t_0, t_0 + \text{pipe\_busy}_l(t_0))$ is added to the pipeline's NoOverlap set. This prevents new batches from starting on that pipeline until the current consume finishes.

---

### 2.4a GC Silo States

| Symbol | Description |
|--------|-------------|
| $G_{l,k}(t_0)$ | Current GC silo level for SKU $k$ on line $l$ at time $t_0$, in batches. Replaces $G^0_{l,k}$ from the deterministic model. |

Available silos: $(L_1, PSC)$, $(L_1, NDG)$, $(L_1, BUS)$, $(L_2, PSC)$.

### 2.4b Restock States

| Symbol | Description |
|--------|-------------|
| $\text{restock\_in\_progress}_l(t_0)$ | Boolean: is a restock currently blocking line $l$'s pipeline? |
| $\text{restock\_timer}_l(t_0)$ | Minutes remaining on the active restock (0 if no restock active on $l$) |
| $\text{restock\_sku}_l(t_0)$ | Which SKU is being restocked on line $l$ (null if no active restock) |
| $\text{restock\_station\_busy}(t_0)$ | Boolean: is the shared restock station occupied by either line? |

If $\text{restock\_in\_progress}_l(t_0) = 1$: a **fixed restock interval** $[t_0, t_0 + \text{restock\_timer}_l(t_0))$ is added to line $l$'s pipeline NoOverlap set AND to the global restock station NoOverlap set. The GC silo credit (+5 batches) is applied when the restock interval completes.

---

### 2.5 In-Progress Batches (Fixed Intervals)

Let $\mathcal{F}$ be the set of batches that are currently RUNNING on roasters other than $r^{UPS}$. For each $f \in \mathcal{F}$:

| Attribute | Value | Description |
|-----------|-------|-------------|
| $r_f$ | roaster | The roaster executing this batch |
| $s_f$ | start time | When this batch started (before $t_0$) |
| $e_f = s_f + p_{k_f}$ | end time | When this batch will complete (SKU-dependent) |
| $\text{sku}_f$ | SKU | Product being roasted |
| $\text{out}_f$ | line | Where RC output goes (if PSC) |
| $\text{con\_end}_f = s_f + \delta^{con}$ | int | When the pipeline consume finishes |

**How fixed batches enter the model:**

1. **Roaster interval:** Fixed interval $[\max(s_f, t_0), \; e_f)$ on roaster $r_f$'s NoOverlap set. The solver cannot schedule anything on $r_f$ during this window.

2. **Pipeline interval:** Fixed interval $[\max(s_f, t_0), \; \text{con\_end}_f)$ on pipeline $\text{pipe}(r_f)$'s NoOverlap set — **only if** $t_0 < \text{con\_end}_f$ (consume is still active). If $t_0 \geq \text{con\_end}_f$, the consume already finished and no pipeline interval is needed.

3. **RC credit:** If $\text{sku}_f = \text{PSC}$ and $\text{out}_f = l$, then batch $f$ will add +1 to $B_l$ at time $e_f$. This is included in the inventory balance as a known future deposit.

4. **MTO tracking:** If $f$ is an MTO batch, it counts toward job completion.

> **Example:** R2 is running a PSC batch that started at $s=173$, so $e=173+p_{PSC}=173+15=188$, $\text{con\_end}=176$. UPS occurs at $t_0=180$.
> - Roaster interval: $[180, 188)$ on R2 — R2 is busy for 8 more minutes.
> - Pipeline interval: $\text{con\_end}=176 < 180 = t_0$ → pipeline consume already finished. **No fixed pipeline interval.**
> - RC credit: +1 to L1 at $t=188$.

---

### 2.6 SETUP Phantom Intervals

If roaster $r$ has $\text{status}_r(t_0) = \text{SETUP}$ with $\sigma^{rem}_r$ minutes remaining:

- **Roaster blocking interval:** $[t_0, \; t_0 + \sigma^{rem}_r)$ added to $r$'s NoOverlap set.
- **Pipeline:** NOT occupied (SETUP is roaster-side only).
- **Post-setup initial SKU:** After the phantom interval ends, the roaster's effective initial SKU for C6' is $\text{setup\_target}_r(t_0)$ — **not** $\text{last\_sku}_r(t_0)$.

> **Example:** R1 is in SETUP at $t_0=180$ with 3 minutes remaining, transitioning from NDG to PSC. $\text{setup\_target}_{R1} = \text{PSC}$.
> - Phantom interval: $[180, 183)$ on R1's NoOverlap set.
> - After $t=183$: R1's initial SKU is PSC. A PSC batch can start at $t=183$ with no additional setup.
> - If the solver wants NDG instead: needs another 5-min setup from PSC → NDG.

---

### 2.7 UPS on a Roaster in SETUP

If UPS hits a roaster that is **currently in SETUP** (not RUNNING a batch):

1. **No batch is cancelled** — there is no in-progress batch to destroy.
2. **SETUP is aborted.** The roaster goes DOWN immediately.
3. **The roaster reverts to its previous SKU:** $\text{last\_sku}_r$ stays at the SKU *before* the setup started (the setup_target is discarded).
4. **When the roaster comes back online** after $d^{UPS}$ minutes, its initial SKU is the old pre-setup SKU.

> **Example:** R2 was transitioning from PSC to NDG (SETUP, 3 min remaining). UPS hits R2 at $t_0$. R2 goes DOWN for 20 minutes.
> - At $t_0 + 20$: R2 is IDLE with $\text{last\_sku} = \text{PSC}$ (reverted, NOT NDG).
> - If the strategy still wants NDG on R2: another 5-min setup is required.
> - Total delay: 20 min (UPS) + 5 min (re-setup) = 25 min before NDG can start.

---

### 2.8 Cancelled Batch

The batch that was RUNNING on $r^{UPS}$ at time $t_0$ (if any) is completely removed:
- No roaster interval in the model
- No pipeline interval in the model
- No RC credit (GC is lost)
- No revenue
- If it was an MTO batch, it goes back into the "remaining MTO" pool and must be re-scheduled

> **Pipeline release:** The pipeline is "released" not by an explicit variable, but by the **absence** of the cancelled batch's consume interval from the model. The solver sees a free pipeline and can schedule new consumes starting at $t_0$ (subject to other fixed pipeline intervals from §2.4 and §2.5).

---

### 2.9 Remaining MTO

$$\mathcal{J}^{MTO}_{rem} \subseteq \mathcal{J}^{MTO}$$

Jobs whose batches are not yet fully completed. For each $j \in \mathcal{J}^{MTO}_{rem}$:

$$n^{rem}_j = n_j - (\text{batches of } j \text{ already completed before } t_0) - (\text{batches of } j \text{ currently in-progress in } \mathcal{F})$$

The cancelled batch (if MTO) is **not** subtracted — it re-enters the pool.

$$\mathcal{B}^{MTO}_{rem} = \bigcup_{j \in \mathcal{J}^{MTO}_{rem}} \{b_1^j, \dots, b_{n^{rem}_j}^j\}$$

---

### 2.10 Remaining Consumption Events

$$\mathcal{E}^{rem}_l = \{\tau \in \mathcal{E}_l : \tau \geq t_0\}$$

Only future consumption events matter for the re-solve.

---

### 2.11 UPS Downtime Set

$$\mathcal{D}'_{r^{UPS}} = \{t_0, t_0+1, \dots, t_0 + d^{UPS} - 1\}$$

This is merged with any existing planned downtime: the effective downtime for the UPS roaster is $\mathcal{D}_r \cup \mathcal{D}'_r$.

For other roasters, $\mathcal{D}'_r = \emptyset$.

---

# PART II — MODIFIED SETS AND PARAMETERS

---

## 3. Sets (Reactive)

| Set | Definition | Change from deterministic |
|-----|-----------|---------------------------|
| $\mathcal{T}^{rem}$ | $\{t_0, \dots, 479\}$ | Reduced horizon |
| $\mathcal{R}$ | $\{R_1, \dots, R_5\}$ | Unchanged |
| $\mathcal{L}$ | $\{L_1, L_2\}$ | Unchanged |
| $\mathcal{B}^{MTO}_{rem}$ | Remaining MTO batches | Reduced (completed removed, cancelled re-added) |
| $\mathcal{B}^{PSC}_{rem}$ | $\{b^{PSC}_1, \dots, b^{PSC}_{N^{PSC}_{rem}}\}$ | Reduced pool (see §3.1) |
| $\mathcal{B}_{rem}$ | $\mathcal{B}^{MTO}_{rem} \cup \mathcal{B}^{PSC}_{rem}$ | All decision batches |
| $\mathcal{F}$ | Fixed in-progress batches | **New** — not in deterministic model |
| $\mathcal{E}^{rem}_l$ | Future consumption events | Filtered from full $\mathcal{E}_l$ |

### 3.1 PSC Pool Size (Reactive)

$$N^{PSC}_{rem} = |\mathcal{R}| \times \left\lfloor \frac{480 - t_0}{P} \right\rfloor$$

> **Example:** $t_0 = 300$. Remaining = 180 min. $N^{PSC}_{rem} = 5 \times \lfloor 180/15 \rfloor = 5 \times 12 = 60$ candidate PSC batches.
>
> This is a theoretical maximum (all 5 roasters running PSC non-stop for the remaining shift). In practice, some roasters are down and some time goes to MTO. But the pool must be large enough to never constrain the solver.

---

## 4. Parameters (Reactive)

All timing parameters ($P$, $\delta^{con}$, $\sigma$, $D^{MTO}$) and cost parameters ($R^{PSC}$, $R^{NDG}$, $R^{BUS}$, $c^{tard}$, $c^{stock}$, $c^{setup}$, $c^{idle}$, $c^{over}$) are **unchanged** from the deterministic model.

**New parameters from state snapshot:**

| Symbol | Source | Description |
|--------|--------|-------------|
| $t_0$ | Simulation clock | Re-solve trigger time |
| $B_l(t_0)$ | Simulation state | Current RC stock per line |
| $\text{sku}^{init}_r$ | Derived (see below) | Effective initial SKU per roaster |
| $\mathcal{D}'_r$ | UPS event | UPS-induced downtime set |
| $\mathcal{F}$ | Simulation state | Fixed in-progress batches |

**Effective initial SKU (per roaster):**

$$\text{sku}^{init}_r = \begin{cases}
\text{setup\_target}_r(t_0) & \text{if } \text{status}_r(t_0) = \text{SETUP} \\
\text{sku}_f & \text{if } \text{status}_r(t_0) = \text{RUNNING and } f \text{ is the in-progress batch} \\
\text{last\_sku}_r(t_0) & \text{otherwise (IDLE or DOWN)}
\end{cases}$$

> **Why RUNNING uses the in-progress batch's SKU:** When the fixed batch completes, the roaster's last_sku becomes the batch's SKU. The first new batch after the fixed one must respect setup against that SKU.

**New cost parameter:**

| Symbol | Value | Description |
|--------|-------|-------------|
| $c^{skip}$ | $50{,}000$ per batch | Penalty for unscheduled MTO batch (see C1') |

> **Rationale for $c^{skip}$:** Must dominate all other terms so the solver never voluntarily skips MTO. $50{,}000 > $ any possible combination of tardiness + idle + overflow for a single batch. Only triggers when MTO is **physically impossible** (e.g., R2 down for rest of shift, Busta cannot be produced).

---

# PART III — DECISION VARIABLES (Reactive)

---

## 5. Decision Variables

The structure is identical to the deterministic model but scoped to remaining batches and the remaining horizon.

| Variable | Domain | Count | Description |
|----------|--------|-------|-------------|
| $a_b$ | $\{0, 1\}$ | $N^{PSC}_{rem}$ | PSC batch activation |
| $a_b$ | $\{0, 1\}$ | $|\mathcal{B}^{MTO}_{rem}|$ | **MTO batch activation (NOW A DECISION — see C1')** |
| $r_b$ | $\mathcal{R}_{\text{sku}(b)}$ | $\leq |\mathcal{B}_{rem}|$ | Roaster assignment |
| $s_b$ | $[t_0, 465]$ | $\leq |\mathcal{B}_{rem}|$ | Start time (lower bound is now $t_0$, not 0) |
| $y_b$ | $\{0, 1\}$ | $\leq N^{PSC}_{rem} / 5$ | R3 output routing (flex mode) |
| $\text{tard}_j$ | $\geq 0$ | $|\mathcal{J}^{MTO}_{rem}|$ | Job tardiness |
| $\text{SO}_l$ | $\geq 0$ integer | 2 | **Stockout event count per line (NEW)** |

---

# PART IV — CONSTRAINTS (Reactive)

> Constraints are numbered C1'–C15' to distinguish from the deterministic model. The prime (') indicates the reactive version.

---

### C1': MTO Activation (Soft)

**Deterministic:** $a_b = 1$ for all MTO batches (hard).

**Reactive:**
$$a_b \in \{0, 1\} \quad \forall b \in \mathcal{B}^{MTO}_{rem}$$

MTO batches are **strongly encouraged** but not forced. An unscheduled MTO batch ($a_b = 0$) incurs a penalty of $c^{skip} = \$50{,}000$ in the objective. The solver will only skip MTO when it is physically impossible to schedule (e.g., Busta required but R2 is down for the rest of the shift).

> **Why soft?** Under severe disruption (multiple UPS, key roaster down), forcing $a_b = 1$ would make the model **infeasible** — no valid schedule exists that completes all MTO. Making it soft ensures the solver always finds a solution, even if that solution includes unmet MTO orders.

---

### C2': PSC Optional

$$a_b \in \{0, 1\} \quad \forall b \in \mathcal{B}^{PSC}_{rem}$$

Unchanged from deterministic.

---

### C3': Roaster Eligibility

$$r_b \in \mathcal{R}_{\text{sku}(b)} \quad \forall b \in \mathcal{B}_{rem} : a_b = 1$$

Unchanged from deterministic.

---

### C4': Roaster NoOverlap (with Fixed Intervals)

$$\text{NoOverlap}\left(\mathcal{I}^{fixed}_r \;\cup\; \{[s_b, s_b + p_{k_b}) : r_b = r, a_b = 1\}\right) \quad \forall r \in \mathcal{R}$$

Where $\mathcal{I}^{fixed}_r$ contains:

1. **In-progress batch interval:** $[\max(s_f, t_0), \; e_f)$ for each $f \in \mathcal{F}$ on roaster $r$.
2. **SETUP phantom interval:** $[t_0, \; t_0 + \sigma^{rem}_r)$ if $\text{status}_r(t_0) = \text{SETUP}$.
3. **UPS/DOWN blocking interval:** $[t_0, \; t_0 + d^{rem}_r)$ if $\text{status}_r(t_0) = \text{DOWN}$ or $r = r^{UPS}$.

These fixed intervals are **not decision variables** — they are pre-computed and added to the NoOverlap set as non-optional intervals. The solver schedules new batches around them.

> **Example:** At $t_0 = 180$:
> - R1 (UPS victim): blocked $[180, 200)$ (20-min repair). $\mathcal{I}^{fixed}_{R1} = \{[180, 200)\}$.
> - R2 (RUNNING PSC, started at 173): $\mathcal{I}^{fixed}_{R2} = \{[180, 188)\}$.
> - R3 (SETUP, 3 min remaining): $\mathcal{I}^{fixed}_{R3} = \{[180, 183)\}$.
> - R4 (IDLE): $\mathcal{I}^{fixed}_{R4} = \emptyset$. Available immediately.
> - R5 (IDLE): $\mathcal{I}^{fixed}_{R5} = \emptyset$.

---

### C5': Setup Time

Identical to deterministic C5:

$$s_{b_2} \geq s_{b_1} + p_{k_{b_1}} + \sigma \quad \text{OR} \quad s_{b_1} \geq s_{b_2} + p_{k_{b_2}} + \sigma$$

for any two batches on the same roaster with different SKUs. Processing time $p_k$ is SKU-dependent: $p_{PSC}=15$, $p_{NDG}=17$, $p_{BUS}=18$.

> **CP-SAT implementation:** Same transition matrix. Fixed intervals from C4' carry a SKU label; the transition matrix applies between fixed intervals and new batches just as it applies between new batches.

---

### C6': Initial SKU Setup (Per-Roaster, State-Dependent)

**Deterministic C6:** $\text{sku}^0_r = k^{PSC}$ for all $r$. First non-PSC batch needs setup.

**Reactive C6':** Each roaster has its own initial SKU from the state snapshot:

$$s_{b_{first}^r} \geq t^{avail}_r + \sigma \quad \text{if } \text{sku}(b_{first}^r) \neq \text{sku}^{init}_r$$

$$s_{b_{first}^r} \geq t^{avail}_r \quad \text{if } \text{sku}(b_{first}^r) = \text{sku}^{init}_r$$

Where:
- $b_{first}^r$ is the first new (non-fixed) batch on roaster $r$
- $t^{avail}_r$ is the earliest time $r$ is available (after all fixed intervals end)
- $\text{sku}^{init}_r$ is the effective initial SKU (from §4)

> **CP-SAT implementation:** Attach a phantom fixed interval with the correct $\text{sku}^{init}_r$ label to each roaster's NoOverlap set, ending at $t^{avail}_r$. The transition matrix automatically enforces setup if the first new batch has a different SKU.

> **Example at $t_0 = 180$:**
> - R1 (DOWN until 200, was running NDG): $\text{sku}^{init}_{R1} = \text{NDG}$, $t^{avail}_{R1} = 200$. First PSC batch on R1 → $s \geq 200 + 5 = 205$. First NDG batch → $s \geq 200$.
> - R3 (SETUP to PSC, done at 183): $\text{sku}^{init}_{R3} = \text{PSC}$, $t^{avail}_{R3} = 183$. First PSC batch → $s \geq 183$. (No additional setup since setup already transitioning to PSC.)
> - R4 (IDLE, last_sku = PSC): $\text{sku}^{init}_{R4} = \text{PSC}$, $t^{avail}_{R4} = t_0 = 180$. First PSC batch → $s \geq 180$.

---

### C7': Planned + UPS Downtime

$$[s_b, s_b + p_{k_b} - 1] \cap \left(\mathcal{D}_r \cup \mathcal{D}'_r\right) = \emptyset \quad \forall r, \forall b : r_b = r, a_b = 1$$

UPS downtime $\mathcal{D}'_r$ is merged with planned downtime $\mathcal{D}_r$.

> **CP-SAT implementation:** Add a fixed downtime interval $[t_0, t_0 + d^{UPS})$ on the UPS roaster. This is equivalent to adding it to C4' as a blocking interval (which is what we already do). No separate C7' encoding needed if blocking intervals are used.

---

### C8': Pipeline NoOverlap (with Fixed Intervals)

$$\text{NoOverlap}\left(\mathcal{P}^{fixed}_l \;\cup\; \{[s_b, s_b + \delta^{con}) : a_b = 1, \text{pipe}(r_b) = l\}\right) \quad \forall l \in \mathcal{L}$$

Where $\mathcal{P}^{fixed}_l$ contains:

1. **Active pipeline consume from in-progress batch:** $[\max(s_f, t_0), \; s_f + \delta^{con})$ for each $f \in \mathcal{F}$ where $\text{pipe}(r_f) = l$ **and** $t_0 < s_f + \delta^{con}$.
2. **Remaining pipeline timer:** $[t_0, \; t_0 + \text{pipe\_busy}_l(t_0))$ if $\text{pipe\_busy}_l(t_0) > 0$ and no in-progress batch accounts for it.

If $t_0 \geq s_f + \delta^{con}$ for an in-progress batch, its consume already finished — **no pipeline fixed interval** is added. The pipeline is free for new batches.

> **Example:** R3 started a batch at $s=178$, so consume window = $[178, 180]$. UPS occurs at $t_0=180$.
> - $\text{con\_end} = 178 + 3 = 181$. Since $t_0 = 180 < 181$: fixed pipeline interval $[180, 181)$ on L2.
> - R4 wants to start at $t=180$: blocked. Must wait until $t=181$.

---

### C9': End-of-Shift

$$s_b \leq 480 - p_{k_b} \quad \forall b \in \mathcal{B}_{rem} : a_b = 1$$

SKU-dependent: PSC batches can start at 465, NDG at 463, Busta at 462.

---

### C10': RC Inventory Lower Bound (SOFT — Stockout Allowed)

**Deterministic C10:** $B_l(\tau) \geq 0$ at all consumption events (hard).

**Reactive C10':** Stockouts are **allowed but penalized.**

$$B_l(\tau) = B_l(t_0) + \underbrace{\sum_{\substack{f \in \mathcal{F}:\\ \text{out}(f)=l,\; e_f \leq \tau}} 1}_{\text{fixed PSC completions}} + \underbrace{\sum_{\substack{b \in \mathcal{B}_{rem}: a_b=1,\\ \text{out}(b)=l,\; \text{sku}(b)=k^{PSC},\; e_b \leq \tau}} 1}_{\text{new PSC completions}} - \underbrace{|\{\tau' \in \mathcal{E}^{rem}_l : \tau' \leq \tau\}|}_{\text{remaining consumption events}}$$

$$\text{SO}_l = |\{\tau \in \mathcal{E}^{rem}_l : B_l(\tau) < 0\}|$$

**Stockout count** $\text{SO}_l$ is the number of consumption events where RC stock is strictly negative. Each such event contributes $c^{stock} = \$1{,}500$ to the objective penalty.

> **Key distinction from deterministic model:** The $B_l(\tau) \geq 0$ hard constraint is **removed**. Instead, $\text{SO}_l$ is counted and penalized. The solver is free to allow stockouts if the cost of preventing them (e.g., sacrificing MTO timeliness) exceeds $\$1{,}500$ per event.

---

### C11': RC Inventory Upper Bound

$$B_l(\tau) \leq \overline{B}_l = 40 \quad \forall \tau \in \mathcal{E}^{rem}_l \cup \{e_b : \text{out}(b)=l, a_b=1\} \cup \{e_f : f \in \mathcal{F}, \text{out}(f)=l\}$$

Still **hard** in reactive mode. A physically full silo cannot accept more coffee.

> **Note:** Check at both consumption events (where stock decreases — trivially satisfied) and batch completion times (where stock increases — this is where violations can occur). The union covers all stock-change points.

---

### C12': Tardiness (Remaining MTO Only)

$$\text{tard}_j \geq e_b - D^{MTO} \quad \forall b \in \mathcal{B}^{MTO}_{rem,j} : a_b = 1$$
$$\text{tard}_j \geq 0$$

Identical structure. Only remaining MTO batches are included. The due date $D^{MTO} = 240$ is unchanged.

> **Note:** For in-progress MTO batches ($f \in \mathcal{F}$ that are MTO), their completion time $e_f$ also contributes to tardiness. Include: $\text{tard}_j \geq e_f - D^{MTO}$ for each in-progress MTO batch $f$ belonging to job $j$.

---

### C13': R3 Routing

$$y_b \in \{0, 1\} \quad \forall b : r_b = R_3, a_b = 1$$

Unchanged.

---

### C14': Safety-Idle Detection (Scoped to Remaining Shift)

$$\text{idle}_{r,t} \geq (1 - \text{busy}_{r,t}) + \text{low}_{\ell(r),t} - 1 \quad \forall r \in \mathcal{R}, \; t \in \mathcal{E}^{rem}_{\ell(r)} \setminus (\mathcal{D}_r \cup \mathcal{D}'_r)$$

**Changes from deterministic:**
- Scoped to $t \geq t_0$ only (remaining shift).
- UPS downtime $\mathcal{D}'_r$ is excluded — a roaster forced down by UPS is not penalized for being idle.
- Planned downtime $\mathcal{D}_r$ still excluded (same as deterministic).

> **Rationale:** UPS is uncontrollable. Penalizing a roaster for being idle while it's physically broken would distort the objective — the solver cannot make it un-broken.

---

### C15': Overflow-Idle Detection (Scoped to Remaining Shift)

Same scoping rules as C14':

**For $R_1, R_2, R_4, R_5$:**
$$\text{over}_{r,t} \geq (1 - \text{busy}_{r,t}) + \text{full}_{\text{out}(r),t} - 1 \quad \forall t \in \mathcal{E}^{rem}_{\ell(r)} \setminus (\mathcal{D}_r \cup \mathcal{D}'_r)$$

**For $R_3$ (flex mode):**
$$\text{over}_{R_3,t} \geq (1 - \text{busy}_{R_3,t}) + \text{full}_{L_1,t} + \text{full}_{L_2,t} - 2 \quad \forall t \in \mathcal{E}^{rem}_{L_2} \setminus (\mathcal{D}_{R_3} \cup \mathcal{D}'_{R_3})$$

---

# PART V — OBJECTIVE FUNCTION (Reactive)

---

## 6. Maximize Remaining-Shift Profit

$$\boxed{\text{Max:} \quad \underbrace{\sum_{b \in \mathcal{B}_{rem}} R_{\text{sku}(b)} \cdot a_b}_{\text{New batch revenue}} + \underbrace{\sum_{f \in \mathcal{F}} R_{\text{sku}(f)}}_{\text{Fixed batch revenue}} - \underbrace{c^{skip} \sum_{b \in \mathcal{B}^{MTO}_{rem}} (1 - a_b)}_{\text{Skipped MTO penalty}} - \underbrace{c^{tard} \sum_j \text{tard}_j}_{\text{Tardiness}} - \underbrace{c^{stock} \sum_l \text{SO}_l}_{\text{Stockout}} - \underbrace{c^{setup} \sum_r N^{setup}_r}_{\text{Setup cost}} - \underbrace{c^{idle} \sum_{r,t} \text{idle}_{r,t}}_{\text{Safety-idle}} - \underbrace{c^{over} \sum_{r,t} \text{over}_{r,t}}_{\text{Overflow-idle}}}$$

**Terms explained:**
1. **New batch revenue:** Revenue from batches the solver decides to schedule. Includes both PSC and MTO.
2. **Fixed batch revenue:** Revenue from in-progress batches that will complete regardless. This is a constant that doesn't affect optimization, but is included for profit reporting consistency.
3. **Skipped MTO penalty:** $c^{skip} = \$50{,}000$ per unscheduled MTO batch. Dominates all other terms — the solver will never voluntarily skip MTO.
4. **Tardiness:** Same as deterministic.
5. **Stockout:** **NEW.** $c^{stock} = \$1{,}500$ per consumption event where $B_l(\tau) < 0$. Not present in deterministic mode (where stockout is a hard constraint).
6. **Setup cost:** $c^{setup} = \$800$ per SKU transition event during the remaining shift. UPS recovery may force re-setups (e.g., setup aborted by UPS → roaster reverts to old SKU → must re-setup after recovery).
7. **Safety-idle:** Same structure, scoped to remaining shift, excludes UPS downtime.
8. **Overflow-idle:** Same structure, scoped to remaining shift, excludes UPS downtime.

**Cost hierarchy (most to least expensive):**
$$c^{skip} \gg c^{stock} > c^{tard} > c^{setup} > c^{idle} > c^{over}$$
$$\$50{,}000 \gg \$1{,}500/\text{event} > \$1{,}000/\text{min} > \$800/\text{event} > \$200/\text{min} > \$50/\text{min}$$

---

## 7. Worked Example — Re-Solve After UPS

```
=== STATE AT t₀ = 180 ===

UPS hits R1 (was running NDG batch b*, started at s=175, e=175+17=192).
  → b* CANCELLED. NDG batch goes back to MTO remaining pool.
  → R1 blocked [180, 200) (20-minute repair).

R2: RUNNING PSC batch (s=173, e=173+15=188). Pipeline consume done (176 < 180).
R3: IDLE. last_sku = PSC. B_L2 = 28.
R4: RUNNING PSC batch (s=177, e=177+15=192). Pipeline consume [177,179], done at 180.
R5: IDLE. last_sku = PSC. 

B_L1(180) = 15.  B_L2(180) = 28.

GC Silos at t₀:
  G_{L1,PSC}(180) = 8    G_{L1,NDG}(180) = 3    G_{L1,BUS}(180) = 5
  G_{L2,PSC}(180) = 6
  Restock station: FREE. No restock in progress on either line.

MTO remaining: j₁ has 1 NDG batch left (2 completed, 1 cancelled = b*).
               j₂ has 0 Busta left (completed).

=== RE-SOLVE MODEL ===

Horizon: [180, 479]. Remaining: 300 minutes.
PSC pool: 5 × ⌊300/p_PSC⌋ = 5 × 20 = 100 candidate batches.
Consumption events remaining: |E^rem_L1| ≈ 37, |E^rem_L2| ≈ 38.

Fixed intervals:
  R2: roaster [180, 188), pipeline NONE (consume done). RC credit: L1 +1 at t=188.
  R4: roaster [180, 192), pipeline [180, 180) = empty (consume just finished). RC credit: L2 +1 at t=192.

Blocking intervals:
  R1: [180, 200) (UPS). sku_init = NDG (was running NDG when UPS hit).

Available immediately: R3 (IDLE), R5 (IDLE).
Available at t=188: R2.
Available at t=192: R4.
Available at t=200: R1.

GC considerations:
  L2 PSC silo at 6 → R3+R4+R5 need ~38 PSC batches remaining.
  Must schedule ~(38-6)/5 ≈ 7 restocks on L2 over 300 min.
  L1 PSC silo at 8 → R1+R2 need ~37 PSC batches remaining.
  Must schedule ~(37-8)/5 ≈ 6 restocks on L1 over 300 min.
  Total ~13 restocks competing for shared station.

=== SOLVER DECISION ===

CP-SAT re-solves in ~0.1–1.0 seconds.
Key decisions:
  - R3 routes output to L1 (y_b = 1) for next 3 batches → helps L1 recover.
  - NDG batch rescheduled to R1 starting at t=205 (after UPS + 5-min setup PSC→NDG).
    → ends at t=205+17=222 < 240 → tard = 0. 
  - R5 starts PSC immediately at t=180.
  - Schedule L2 PSC restock at t=195 (while R4 still mid-roast, R3 mid-roast)
    → pipeline block [195, 210). Minimal idle since roasters in mid-cycle.

Stockout risk: L1 has 15 batches, ~37 events remaining.
  Need ~37 - 15 = 22 more PSC batches on L1 over 300 minutes.
  R2 available at 188, can produce ⌊(479-188)/15⌋ = 19 PSC batches.
  R3 routing 3 batches to L1 = 22. Tight but feasible → SO_L1 = 0.
```

---

# PART VI — CONSTRAINT SUMMARY (Reactive)

| ID | Constraint | Hard/Soft | Penalty | Change from deterministic |
|----|-----------|-----------|---------|---------------------------|
| C1' | MTO activation | **Soft** | $c^{skip}$ = $50k/batch | Was hard ($a_b = 1$) |
| C2' | PSC optional | — | — | Reduced pool |
| C3' | Roaster eligibility | Hard | — | Unchanged |
| C4' | Roaster NoOverlap (duration = $p_k$) | Hard | — | + fixed intervals $\mathcal{I}^{fixed}_r$ |
| C5' | Setup time ($p_k + \sigma$) | Hard | — | Unchanged |
| C6' | Initial SKU | Hard | — | Per-roaster $\text{sku}^{init}_r$ from state |
| C7' | Downtime | Hard | — | $\mathcal{D}_r \cup \mathcal{D}'_r$ |
| C8' | Pipeline NoOverlap | Hard | — | + fixed pipeline intervals |
| C9' | End-of-shift ($s_b \leq 480 - p_k$) | Hard | — | Unchanged |
| C10' | RC ≥ 0 | **Soft** | $c^{stock}$ = $1.5k/event | Was hard |
| C11' | RC ≤ 40 | Hard | — | $B_l(t_0)$ as initial |
| C12' | Tardiness | Soft | $c^{tard}$ = $1k/min | Only remaining MTO |
| C13' | R3 routing | — | — | Unchanged |
| C14' | Safety-idle | Soft | $c^{idle}$ = $200/min | Scoped $t \geq t_0$, excludes $\mathcal{D}'_r$ |
| C15' | Overflow-idle | Soft | $c^{over}$ = $50/min | Scoped $t \geq t_0$, excludes $\mathcal{D}'_r$ |
| **C16'** | **GC silo ≥ 0** | **Hard** | — | $G_{l,k}(t_0)$ as initial; same as deterministic |
| **C17'** | **Restock blocks pipeline (15 min)** | **Hard** | — | + fixed restock interval if in progress at $t_0$ |
| **C18'** | **Shared restock station** | **Hard** | — | + fixed restock interval if station busy at $t_0$ |
| **C19'** | **Restock capacity guard** | **Hard** | — | Unchanged |
| — | Setup cost | Soft | $c^{setup}$ = $800/event | Same as deterministic; UPS may force re-setups |

---

# PART VII — SPECIAL CASES AND EDGE CONDITIONS

---

## 8.1 Multiple Consecutive UPS

Each UPS triggers an independent re-solve. The second re-solve sees the state left by the first re-solve's schedule (partially executed), plus the new UPS. There is no memory of previous re-solves — only the current state matters.

> **Example:** UPS₁ at $t=100$ on R1 → re-solve₁ produces schedule A. At $t=250$, UPS₂ hits R3 → re-solve₂ sees the state at $t=250$ (which reflects schedule A's execution up to that point, plus UPS₂). Re-solve₂ has no knowledge of re-solve₁'s objective value, constraint set, or decisions.

## 8.2 UPS at $t_0 = 0$

If UPS occurs at the very start of the shift, the reactive model degenerates to the deterministic model with one roaster blocked — except C10 is soft and the stockout term is active. This is a valid edge case and requires no special handling.

## 8.3 UPS with No Running Batch

If the UPS roaster was IDLE (not running anything), no batch is cancelled. The model simply adds a blocking interval for the repair duration. This is simpler than the running-batch case.

## 8.4 UPS During SETUP (Detailed)

As described in §2.7:
1. No batch cancelled (no batch was running).
2. SETUP aborted. Roaster goes DOWN. The $800 setup cost already incurred is a **sunk cost**.
3. $\text{last\_sku}_r$ reverts to pre-setup SKU (setup_target discarded).
4. After repair: roaster IDLE with old SKU. New setup needed if different SKU desired — incurring a **second** $800 setup cost.

## 8.5 Endgame — UPS Near Shift End

If $t_0 + d^{UPS} > 465$ (repair extends beyond latest batch start time), the UPS roaster cannot run any more batches this shift. It effectively becomes permanently blocked. The solver handles this naturally — no batches can be assigned to a roaster whose blocking interval covers the entire remaining feasible start window.

---

# QUICK REFERENCE: REACTIVE MODEL AT A GLANCE

```
╔════════════════════════════════════════════════════════════════════════╗
║                 REACTIVE (UPS) MODEL SUMMARY                          ║
╠════════════════════════════════════════════════════════════════════════╣
║ TRIGGER:     UPS at t₀ on roaster r^UPS.                             ║
║ HORIZON:     [t₀, 479]. Remaining = 480 − t₀ minutes.               ║
║ INPUTS:      State snapshot — RC stock, GC silo levels, roaster      ║
║              states, restock state, MTO left.                        ║
║ ROASTING:    p_PSC=15, p_NDG=17, p_BUS=18 min (SKU-dependent).      ║
║ CANCELLED:   Batch on r^UPS destroyed. GC lost. No credit.           ║
║ FIXED:       In-progress batches on other roasters = immovable.      ║
║ SETUP:       Per-roaster initial SKU from snapshot. Phantom blocks.  ║
║ GC SILOS:    G_{l,k}(t₀) as initial. Restock intervals scheduled.   ║
║              Shared station mutex (C18') carries over.               ║
║ C1:          MTO soft ($50k skip penalty). Always feasible.          ║
║ C10:         Stockout soft ($1.5k/event). B_l can go negative.       ║
║ C16:         GC silo ≥ 0 HARD. Cannot roast from empty silo.        ║
║ C14/C15:     UPS downtime excluded from idle penalties.              ║
║ OBJECTIVE:   Revenue − skip − tardiness − stockout − setup − idle    ║
║ SOLVER:      CP-SAT. Target: < 1 second per re-solve.               ║
║ STATELESS:   Each re-solve from scratch. No memory of previous.      ║
╚════════════════════════════════════════════════════════════════════════╝
```
