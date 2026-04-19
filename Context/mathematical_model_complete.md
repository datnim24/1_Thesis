# Mathematical Model — Deterministic Optimization (Best-Case Benchmark)
### Dynamic Batch Roasting Scheduling at Nestlé Trị An

> **Purpose of this document:** This is the **canonical** mathematical formulation for the deterministic scheduling problem — no disruptions, full information, single solve at shift start. Every symbol is defined, explained, and illustrated with concrete examples. This model serves as the benchmark that CP-SAT and MILP solve. For the reactive (post-UPS) formulation, see `UPS_Mathematical_Model.md`.

---

# HOW TO READ THIS DOCUMENT

The model is split into five logical layers:

1. **Inputs (Part I):** What is *given* to the solver — the factory, the machines, the products, the shift.
2. **Decision Variables (Part II):** What the solver *decides* — which batches to run, on which roaster, at what time.
3. **Constraints (Part III):** What the solver *cannot do* — physical and operational rules.
4. **Objective (Part IV):** What the solver is *trying to maximize* — shift profit.
5. **Solver Roles:** How CP-SAT, MILP, and DRL each use the model differently.

---

# PART I — MODEL INPUTS

> Everything here is **fixed before the solver runs**. Think of it as the "factory settings" loaded at the start of each shift.

---

## 1. Sets — The "Objects" in the Problem

A **set** is simply a named collection of objects. The model uses sets to refer to groups of machines, products, jobs, etc., without writing out every member each time.

---

### 1.1 Time Horizon: $\mathcal{T}$

$$\mathcal{T} = \{0, 1, 2, \dots, 479\}$$

**Plain English:** The 8-hour shift is divided into 480 one-minute slots, numbered 0 through 479. Slot 0 is the moment the shift starts (e.g., 6:00 AM). Slot 479 is the last minute (e.g., 1:59 PM).

> **Why minutes, not hours?** Because the shortest meaningful event — the pipeline consume window — lasts 3 minutes. Using minutes gives enough resolution without making the model too large.

**Extra example:**
```
Slot  0 = 6:00 AM  (shift starts)
Slot 60 = 7:00 AM
Slot240 = 10:00 AM (half-shift mark = MTO due date)
Slot479 = 1:59 PM  (last minute of shift)
```

---

### 1.2 Production Lines: $\mathcal{L}$

$$\mathcal{L} = \{L_1, L_2\}$$

Two packaging lines downstream of the roasters. They consume roasted coffee (RC) from a shared buffer (the RC silo). Each line has its own buffer and its own consumption rate.

---

### 1.3 Roasters: $\mathcal{R}$

$$\mathcal{R} = \{R_1, R_2, R_3, R_4, R_5\}$$

Five roasting machines in the factory. They are grouped by which pipeline they use to receive Green Coffee (GC):

$$\mathcal{R}_{L_1} = \{R_1, R_2\} \qquad \mathcal{R}_{L_2} = \{R_3, R_4, R_5\}$$

> **Key fact:** $R_1$ and $R_2$ share the Line 1 pipeline. $R_3$, $R_4$, and $R_5$ share the Line 2 pipeline. Two roasters on the same line **cannot start a batch at the same time** because they would both need to pull GC through the same pipe simultaneously — and the pipe can only handle one flow at a time. This pipeline constraint is the **central bottleneck** of the whole problem.

---

### 1.4 SKU Types: $\mathcal{K}$

$$\mathcal{K} = \{k^{PSC}, k^{NDG}, k^{BUS}\}$$

Three products:
- **PSC (Passata / Standard Coffee):** The main make-to-stock product. Fills the RC buffer. Revenue: $4,000/batch.
- **NDG (Nescafé Dolce Gusto):** Make-to-order. Revenue: $7,000/batch.
- **BUS (Busta):** Make-to-order. Revenue: $7,000/batch.

> **Important distinction:** PSC output goes into the RC silo (buffer stock). NDG and Busta output is delivered directly to their own storage — they do **not** add to the RC silo.

---

### 1.5 Roaster Eligibility

Not every roaster can make every product. The eligibility rules are:

| Roaster | Can produce | Reason |
|---------|-------------|--------|
| $R_1$ | PSC, NDG | Configured for standard + Nescafé |
| $R_2$ | PSC, NDG, **Busta** | The only roaster capable of Busta |
| $R_3$ | PSC only | Dedicated PSC roaster |
| $R_4$ | PSC only | Dedicated PSC roaster |
| $R_5$ | PSC only | Dedicated PSC roaster |

**Consequence:** If there is a Busta order ($k^{BUS}$), it **must** go to $R_2$. There is no alternative. This creates a bottleneck: $R_2$ is the only machine that can serve both NDG and Busta MTO orders, while also running PSC to fill the buffer.

**Extra example — what happens if R2 breaks down (UPS)?**
> Busta order cannot be completed at all. NDG can still run on R1. PSC can still run on R1, R3, R4, R5. This is exactly the type of disruption the thesis models.

---

### 1.6 Pipeline Mapping: $\text{pipe}(r)$

$$\text{pipe}(r) = \begin{cases} L_1 & \text{if } r \in \{R_1, R_2\} \\ L_2 & \text{if } r \in \{R_3, R_4, R_5\} \end{cases}$$

**Plain English:** This function answers the question "which GC pipeline does roaster $r$ use?" It is used in constraint C8 to enforce that no two roasters on the same line start a batch at overlapping times.

**Special case — R3:** R3 always *pulls* GC through the Line 2 pipeline (so `pipe(R3) = L2`), but it can *send* its roasted coffee output to either line's RC buffer. These are two different concepts: pipeline = input; output routing = separate decision variable $y_b$.

---

### 1.7 Line-of-Roaster Mapping: $\ell(r)$

$$\ell(r) = \begin{cases} L_1 & \text{if } r \in \{R_1, R_2\} \\ L_2 & \text{if } r \in \{R_3, R_4, R_5\} \end{cases}$$

This is used only for the **idle penalty** calculation: if R3 is idle and the RC stock on its home line ($L_2$) is low, it gets penalized. Note that `ℓ(r) = pipe(r)` — they are numerically the same, but conceptually separate.

---

### 1.8 MTO Jobs: $\mathcal{J}^{MTO}$

$$\mathcal{J}^{MTO} = \{j_1, j_2, \dots\}$$

Each MTO (Make-To-Order) job is a customer order with a fixed product type and quantity (in batches). Jobs **must** be completed. The solver cannot skip them.

**Example instance (used throughout this document):**
```
j₁: NDG × 3 batches   → must run on R1 or R2, 3 times
j₂: Busta × 1 batch   → must run on R2, 1 time
Total MTO load: 4 batches
```

**Extra example — what "job" vs "batch" means:**
- Job $j_1$ is the *order*: "produce 3 batches of NDG."
- Batches $b_1^{j_1}$, $b_2^{j_1}$, $b_3^{j_1}$ are the three individual 15-minute roasting runs needed to fulfill that order.
- The solver decides *when* and *on which roaster* each batch runs — but all three must be completed.

---

### 1.9 MTO Batches: $\mathcal{B}^{MTO}$

$$\mathcal{B}^{MTO} = \bigcup_{j \in \mathcal{J}^{MTO}} \{b_1^j, b_2^j, \dots, b_{n_j}^j\}$$

The symbol $\bigcup$ means "combine all elements from each set." So this just means: take every batch from every job and put them all in one big collection.

**For our example:**
```
B^MTO = {b₁^j₁, b₂^j₁, b₃^j₁,  ← 3 NDG batches from j₁
          b₁^j₂}                   ← 1 Busta batch from j₂
Total: 4 mandatory batches
```

**Key property:** $a_b = 1$ for ALL MTO batches (activation = 1, meaning they are always scheduled). The solver has no choice about *whether* to run them — only *when* and *where*.

> **Symmetry note:** Batches within the same MTO job (e.g., $b_1^{j_1}$, $b_2^{j_1}$, $b_3^{j_1}$) are **interchangeable** — they have identical SKU, eligibility, and due dates. The solver may explore permutations of these identical batches without improving the schedule. With only 4 MTO batches total, this symmetry is manageable. CP-SAT handles it implicitly; MILP could add symmetry-breaking constraints if needed.

---

### 1.10 PSC Batch Pool: $\mathcal{B}^{PSC}$

$$\mathcal{B}^{PSC} = \{b^{PSC}_1, \dots, b^{PSC}_{160}\}$$

**How the pool works:** The solver does not know in advance how many PSC batches it will run. Instead, a pool of 160 "candidate" PSC batches is created (the theoretical maximum if all 5 roasters ran PSC non-stop for 480 minutes: $5 \times \lfloor 480/15 \rfloor = 160$). The solver then *activates* as many as it needs.

**Analogy:** Imagine 160 blank job tickets. The solver stamps "USE" or "SKIP" on each one. The stamps it marks "USE" become the actual PSC schedule.

**Extra example (illustrative):**
```
If B⁰_L1 = 12 (Line 1 starts with 12 batches of stock) and the line
consumes 1 batch every ρ_L1 minutes:
  → Line 1 needs roughly |E_L1| − B⁰_L1 more PSC deliveries over the shift
  → But with 2 roasters (R1, R2) occasionally running MTO, the solver
     might only activate a fraction of PSC batches on L1 roasters
  → The remaining PSC demand is met by R3 routing output to L1 (via y_b=1)
```

---

### 1.11 All Batches

$$\mathcal{B} = \mathcal{B}^{MTO} \cup \mathcal{B}^{PSC}$$

All batches, mandatory and optional, in one set. Total: up to 164 batches (4 MTO + 160 PSC candidates). In practice, the solver activates ~62–70 PSC batches, so the working schedule has ~66–74 batches.

---

## 2. Parameters — Fixed Numbers

Parameters are numerical values that are **given as inputs** and do not change during solving.

---

### 2.1 Timing Parameters

| Symbol | Value | What it means |
|--------|-------|---------------|
| $p_k$ | SKU-dependent | Processing time per batch. $p_{PSC} = 15$ min, $p_{NDG} = 17$ min, $p_{BUS} = 18$ min |
| $\delta^{con}$ | 3 min | When a batch starts, it "consumes" GC from the pipeline for the first 3 minutes |
| $\sigma$ | 5 min | If a roaster switches from one SKU to a different SKU, it needs 5 minutes of cleanup/setup |
| $D^{MTO}$ | slot 240 | MTO orders are "due" by the halfway point of the shift (10:00 AM if shift starts 6:00 AM) |

**SKU-dependent processing times:**
```
PSC batch:   p_PSC = 15 min → batch at s=0 finishes at e=15
NDG batch:   p_NDG = 17 min → batch at s=0 finishes at e=17
Busta batch: p_BUS = 18 min → batch at s=0 finishes at e=18
```

> **Why different times?** Different bean types and roast profiles require different durations. PSC (standard blend) is the fastest. NDG and Busta (specialty products) require longer roasting for flavor development. This creates **desynchronization** — roasters that start in a neat 3-minute stagger will drift out of alignment after a few batches of mixed SKUs, making pipeline scheduling non-trivial.

**Extra example — why $\delta^{con} = 3$ matters:**
```
Timeline for a PSC batch starting at slot 10:
  [10, 11, 12]     ← GC is being pulled through the pipeline (3 min)
  [10, 11, ..., 24] ← Roaster is busy for all 15 min (p_PSC = 15)
  [25 onward]      ← Roaster is free; batch is done

Timeline for an NDG batch starting at slot 10:
  [10, 11, 12]     ← GC pipeline (3 min, same for all SKUs)
  [10, 11, ..., 26] ← Roaster is busy for 17 min (p_NDG = 17)
  [27 onward]      ← Roaster is free

If R1 starts at slot 10 and R2 tries to start at slot 11 (both on L1 pipeline):
  R1 needs pipeline at slots [10, 11, 12]
  R2 needs pipeline at slots [11, 12, 13]  ← CONFLICT at slots 11 and 12
  → R2 must wait until slot 13 to start.
```

---

### 2.2 Inventory Parameters

| Symbol | Value | What it means |
|--------|-------|---------------|
| $\overline{B}_l$ | 40 batches | The RC silo on each line holds at most 40 batches (= 20,000 kg ÷ 500 kg/batch) |
| $\theta^{SS}$ | 20 batches | "Safety stock" threshold. If stock drops below 20, an idle roaster gets penalized |
| $B^0_l$ | variable | Starting stock at shift begin (differs by shift; a model input from real data) |
| $\rho_l$ | variable | How fast the packaging line consumes RC (minutes per batch consumed) |

**Extra example — $\rho_l$ in practice:**

> **Note:** The values below are **illustrative**. Actual consumption rates are loaded from `shift_parameters.csv` and may change between experiments. The model uses $\rho_l$ as a parameter throughout.

```
Suppose ρ_L1 = 8.1 minutes per batch (illustrative):
  → Line 1 consumes 1 batch of RC every 8.1 minutes
  → Over 480 minutes: ⌊480 / 8.1⌋ = 59 consumption events

Suppose ρ_L2 = 7.8 minutes per batch (illustrative):
  → Line 2 consumes 1 batch every 7.8 minutes
  → Over 480 minutes: ⌊480 / 7.8⌋ = 61 consumption events

Implication: Both lines are demanding — combined ~120 consumption events
per shift means roasters must produce ~120 PSC batches minus initial stock.
```

---

### 2.3 GC Silo Parameters (Green Coffee Input Inventory)

Each production line has dedicated GC silos — one per SKU that can be roasted on that line. GC silos are **finite** and must be **restocked** periodically. Restocking blocks the pipeline.

**Silo Configuration:**

| Line | SKU | Symbol | Capacity (batches) | Initial Level (batches) |
|------|-----|--------|--------------------|------------------------|
| $L_1$ | PSC | $\overline{G}_{L_1, PSC}$ | 40 | $G^0_{L_1, PSC} = 20$ |
| $L_1$ | NDG | $\overline{G}_{L_1, NDG}$ | 10 | $G^0_{L_1, NDG} = 5$ |
| $L_1$ | Busta | $\overline{G}_{L_1, BUS}$ | 10 | $G^0_{L_1, BUS} = 5$ |
| $L_2$ | PSC | $\overline{G}_{L_2, PSC}$ | 40 | $G^0_{L_2, PSC} = 20$ |

> **Why Line 2 only has PSC:** All three Line 2 roasters (R3, R4, R5) can only produce PSC. There is no need for NDG or Busta GC on Line 2. NDG and Busta are exclusively roasted on R1 and R2 (Line 1).

**Restock Parameters:**

| Symbol | Value | What it means |
|--------|-------|---------------|
| $Q^{restock}$ | 5 batches | Each restock operation adds exactly 5 batches worth of GC to a silo |
| $\delta^{restock}$ | 15 min | Duration of a restock operation — pipeline is **blocked** for the full 15 minutes |

**Restock Rules:**
1. **Pipeline blocking:** A restock on line $l$ occupies the pipeline for 15 minutes. During this time, NO roaster on line $l$ can start a new batch (the consume window requires pipeline access). Roasters that are already mid-roast continue normally — they only needed the pipeline for their first 3 minutes.
2. **Shared restock station:** Only **one restock can occur at a time** across both lines. If Line 1 is restocking, Line 2 cannot restock simultaneously, and vice versa. This is a global mutex constraint.
3. **Capacity constraint:** A restock is only allowed if the silo level **plus** $Q^{restock}$ does not exceed the silo capacity: $G_{l,k}(t) + Q^{restock} \leq \overline{G}_{l,k}$.
4. **No direct cost:** Restocking has no monetary cost. The cost is purely the **opportunity cost** of blocking the pipeline for 15 minutes.

**GC Silo Mapping — which silo feeds which roaster:**

$$\text{gc\_silo}(r, k) = (\text{pipe}(r), k)$$

A roaster $r$ producing SKU $k$ consumes from the GC silo on its pipeline's line for that SKU. For example:
```
R1 roasting NDG → consumes from GC silo (L1, NDG)
R2 roasting Busta → consumes from GC silo (L1, BUS)
R3 roasting PSC → consumes from GC silo (L2, PSC)
R4 roasting PSC → consumes from GC silo (L2, PSC)
```

> **Key implication:** R3 always consumes from Line 2's PSC silo (because pipe(R3) = L2), regardless of where R3 routes its RC output. The GC input and RC output are on different sides of the roaster.

**Extra example — why restock timing matters:**
```
Line 2, PSC silo at t=200: G_{L2,PSC} = 3 batches remaining.
R3, R4, R5 each need 1 batch GC per roast cycle.
At current throughput (~1 batch every 5 min across the 3 roasters):
  → 3 batches last ~15 minutes.
  → If no restock by t=215, silo empty → all Line 2 roasters forced idle.

Option A: Restock NOW at t=200.
  Pipeline blocked [200, 215). Roasters in mid-roast continue.
  R4 finishes batch at t=208 → can't start new batch until t=215.
  R5 finishes at t=210 → can't start until t=215.
  After t=215: silo has 3 + 5 = 8 batches. Pipeline free. Resume normal.
  Cost: ~5-7 min idle per roaster (pipeline wait).

Option B: Squeeze 2 more batches, restock at t=210.
  R3 starts at t=200 (silo: 3→2), R4 at t=203 (silo: 2→1), 
  R5 at t=206 (silo: 1→0). Silo EMPTY at t=206.
  Restock at t=206: pipeline blocked [206, 221).
  R3 finishes at t=215 → can't start until t=221. Idle: 6 min.
  R4 finishes at t=218 → can't start until t=221. Idle: 3 min.
  R5 finishes at t=221 → starts immediately. Idle: 0 min.
  But if UPS hits during [206,221] → silo stays at 0, roaster can't restart.

→ Option A is safer. Option B squeezes more batches but risks stockout.
  A smart scheduler (DRL/CP-SAT) evaluates this tradeoff; greedy cannot.
```

---

### 2.4 Cost Structure

| Symbol | Value | Per what | Meaning |
|--------|-------|----------|---------|
| $R^{PSC}$ | $4,000 | batch | Revenue earned when a PSC batch completes |
| $R^{NDG}$ | $7,000 | batch | Revenue earned when an NDG batch completes |
| $R^{BUS}$ | $7,000 | batch | Revenue earned when a Busta batch completes |
| $c^{tard}$ | $1,000 | minute | Penalty for every minute an MTO job finishes late (after slot 240) |
| $c^{stock}$ | $1,500 | event | Penalty per consumption event where the RC silo is empty (reactive mode only) |
| $c^{over}$ | $50 | minute/roaster | Penalty per minute a roaster is idle because the RC silo is full |
| $c^{idle}$ | $200 | minute/roaster | Penalty per minute a roaster is idle while RC stock is below safety threshold |

**Priority order** (most expensive to cheapest):
$$c^{stock} > c^{tard} > c^{idle} > c^{over}$$
$$\$1{,}500/\text{event} > \$1{,}000/\text{min} > \$200/\text{min} > \$50/\text{min}$$

**Extra example — comparing penalty magnitudes:**
```
Scenario A: MTO finishes 5 minutes late.
  Cost = 5 × $1,000 = $5,000

Scenario B: R1 is idle for 25 minutes while RC_L1 < 20 batches.
  Cost = 25 × $200 = $5,000

Scenario C: 2 stockout events on L1 in reactive mode.
  Cost = 2 × $1,500 = $3,000

→ These costs are in the same order of magnitude, so the solver must
  genuinely trade them off — it can't just ignore one category.
```

---

### 2.5 PSC Consumption Schedule: $\mathcal{E}_l$

$$\mathcal{E}_l = \left\{ \lfloor i \cdot \rho_l \rfloor \;\middle|\; i = 1, 2, \dots, \lfloor 480 / \rho_l \rfloor \right\}$$

**Plain English:** This is the pre-computed list of exact time slots when Line $l$'s packaging machine "reaches in" and grabs one batch from the RC silo. It is deterministic and known at shift start.

The symbol $\lfloor \cdot \rfloor$ means "round down to the nearest integer."

**Extra example (illustrative, using $\rho_{L_1} = 8.1$):**
```
i=1:  ⌊1 × 8.1⌋ = ⌊8.1⌋  = 8    ← first draw at slot 8
i=2:  ⌊2 × 8.1⌋ = ⌊16.2⌋ = 16   ← second draw at slot 16
i=3:  ⌊3 × 8.1⌋ = ⌊24.3⌋ = 24
i=4:  ⌊4 × 8.1⌋ = ⌊32.4⌋ = 32
i=5:  ⌊5 × 8.1⌋ = ⌊40.5⌋ = 40
i=6:  ⌊6 × 8.1⌋ = ⌊48.6⌋ = 48
i=7:  ⌊7 × 8.1⌋ = ⌊56.7⌋ = 56
i=8:  ⌊8 × 8.1⌋ = ⌊64.8⌋ = 64   ← gap is always 8 min at this rate
...
i=59: ⌊59 × 8.1⌋ = ⌊477.9⌋ = 477  ← last event

So E_L1 = {8, 16, 24, 32, 40, 48, 56, 64, ...}

Note: with ρ=8.1, the gap alternates between 8 and 9 as floor() shifts.
Total events: ⌊480 / 8.1⌋ = 59 consumption events during the shift.
```

> **Why precompute this?** The model only needs to check inventory at these ~94 moments per line, not at all 480 slots. This makes the constraint count manageable.

---

### 2.6 Planned Downtime: $\mathcal{D}_r$

$$\mathcal{D}_r \subset \mathcal{T} \quad \forall r \in \mathcal{R}$$

A set of time slots when roaster $r$ is scheduled for maintenance and unavailable. This is **known in advance** (unlike UPS, which is unplanned).

**Example:**
```
D_R3 = {200, 201, 202, ..., 229}  (30-minute maintenance window)
→ R3 cannot start any batch that would still be running during slots 200–229.
→ Last valid start for R3: slot 185 (batch runs [185, 199], finishes at 200, 
  which is exactly the start of downtime — but the batch ends BEFORE slot 200 
  begins, so it is safe. See C7.)

Actually: batch at s=185 ends at s+15=200. Downtime starts at 200.
  → [185, 199] ∩ {200,...,229} = ∅ ✓ (valid)
  → Batch at s=186 ends at 201. [186, 200] ∩ {200,...,229} = {200} ✗ (violation)
```

---

### 2.7 Initial SKU State

$$\text{sku}^0_r = k^{PSC} \quad \forall r \in \mathcal{R}$$

At shift start, every roaster is treated as if it last ran a PSC batch. This means:
- A roaster's first batch is **PSC** → no setup needed, can start at slot 0.
- A roaster's first batch is **NDG or Busta** → 5-minute setup required first.

**Extra example:**
```
Suppose R2's first batch of the shift is Busta.
  → Setup occupies slots [0, 4] (5 minutes)
  → Busta batch can start at slot 5 at the earliest
  → Batch runs [5, 19], finishes at slot 20
  → Pipeline is consumed during slots [5, 7]

If instead R2's first batch is PSC:
  → No setup needed
  → PSC batch can start at slot 0
  → Batch runs [0, 14], finishes at slot 15
  → Pipeline consumed during [0, 2]
```

---

### 2.8 Complete Example Instance (Reference)

This instance is used for all examples throughout the rest of this document:

```
SHIFT:       480 minutes (slots 0–479)
             p_PSC = 15 min, p_NDG = 17 min, p_BUS = 18 min
             δ_con = 3 min, σ = 5 min, D_MTO = 240

ROASTERS:    R1 (L1), R2 (L1), R3 (L2), R4 (L2), R5 (L2)
             All start with last_sku = PSC

MTO ORDERS:  j₁: NDG × 3 batches    (must run on R1 or R2)
             j₂: Busta × 1 batch    (must run on R2)
             Total: 4 mandatory batches

PSC:         ρ_L1, ρ_L2 loaded from shift_parameters.csv
             → |E_L1| = ⌊480/ρ_L1⌋ consumption events
             → |E_L2| = ⌊480/ρ_L2⌋ consumption events
             Pool: 5 × ⌊480/p_PSC⌋ = 160 candidate PSC batches

RC STOCK:    B⁰_L1 = 12 batches    B⁰_L2 = 15 batches
             Max = 40 batches/line  Safety threshold = 20 batches/line

GC SILOS:    L1: PSC(cap=40, init=20), NDG(cap=10, init=5), Busta(cap=10, init=5)
             L2: PSC(cap=40, init=20)
             Restock: 5 batches per event, 15-min pipeline block
             Shared restock station (1 restock at a time globally)

DOWNTIME:    D_R3 = {200, ..., 229} (30-minute window)

COSTS:       R_PSC=$4k, R_NDG=$7k, R_BUS=$7k
             c_tard=$1k/min, c_stock=$1.5k/event (reactive only)
             c_setup=$800/event, c_over=$50/min/roaster, c_idle=$200/min/roaster
```

---

# PART II — DECISION VARIABLES

> These are what the solver *chooses*. Everything in Part I was given. Everything here is the solver's answer.

---

## 3. Decision Variables

### 3.1 PSC Batch Activation: $a_b$

$$a_b \in \{0, 1\} \quad \forall b \in \mathcal{B}^{PSC}$$

Binary variable: 1 = this PSC batch slot is used, 0 = it is skipped.

For MTO batches, this is not a decision — they are always 1:
$$a_b = 1 \quad \forall b \in \mathcal{B}^{MTO}$$

**Extra example — why not just decide how many PSC batches to run?**
> Because the solver also needs to decide *which roasters* and *what times*. Activation $a_b$ is coupled to the assignment and start time variables. A deactivated batch ($a_b = 0$) has no roaster, no start time, and contributes nothing to inventory or revenue.

```
Pool has 160 slots: b¹_PSC, b²_PSC, ..., b¹⁶⁰_PSC

If solver activates b¹_PSC through b⁶²_PSC (sets a=1 for 62 of them):
  → 62 PSC batches will be scheduled
  → The remaining 98 batches have a=0 and are simply ignored
  → Total active schedule: 62 PSC + 4 MTO = 66 batches
```

---

### 3.2 Roaster Assignment: $r_b$

$$r_b \in \mathcal{R}_{\text{sku}(b)} \quad \forall b : a_b = 1$$

For each active batch, this variable assigns it to a roaster. The domain is restricted by eligibility.

| Batch SKU | Allowed roasters for $r_b$ |
|-----------|---------------------------|
| PSC | Any of $\{R_1, R_2, R_3, R_4, R_5\}$ |
| NDG | Only $\{R_1, R_2\}$ |
| Busta | Only $\{R_2\}$ |

**Extra example:**
```
Batch b¹^j₁ is an NDG batch (from job j₁).
  → r_{b¹^j₁} ∈ {R1, R2}
  → Solver might assign it to R1.

Batch b¹^j₂ is a Busta batch (from job j₂).
  → r_{b¹^j₂} ∈ {R2}   ← only one option
  → It must go to R2.

Batch b¹_PSC is a PSC batch.
  → r_{b¹_PSC} ∈ {R1, R2, R3, R4, R5}  ← solver has full flexibility
  → Solver might assign it to R4 (which has no MTO load) for efficiency.
```

---

### 3.3 Start Time: $s_b$

$$s_b \in [0, 480 - p_{k_b}] \quad \forall b : a_b = 1$$

The slot at which batch $b$ begins roasting. The upper bound depends on the batch's SKU processing time to ensure every batch fully completes within the shift:

$$e_b = s_b + p_{k_b} \leq 480 \implies s_b \leq 480 - p_{k_b}$$

| SKU | $p_k$ | Max start time $s_b$ |
|-----|--------|---------------------|
| PSC | 15 | 465 |
| NDG | 17 | 463 |
| Busta | 18 | 462 |

**Extra example:**
```
PSC batch: s_b = 465 → runs [465, 479], ends at 480. Fine (exactly at shift end).
NDG batch: s_b = 463 → runs [463, 479], ends at 480. Fine.
NDG batch: s_b = 464 → runs [464, 480], ends at 481. INVALID — exceeds shift.
Busta batch: s_b = 462 → runs [462, 479], ends at 480. Fine.
Busta batch: s_b = 463 → runs [463, 480], ends at 481. INVALID.
```

**Key derived quantity:** $e_b = s_b + p_{k_b}$ (end time). This is not a decision — it is calculated from $s_b$ and the batch's SKU.

---

### 3.4 R3 Output Routing: $y_b$

$$y_b \in \{0, 1\} \quad \forall b : r_b = R_3, \; a_b = 1$$

This variable only exists for batches assigned to R3. It decides where R3 sends its roasted coffee:
- $y_b = 1$ → output goes to Line 1's RC buffer
- $y_b = 0$ → output goes to Line 2's RC buffer

**Experimental design note:** This is one of the two **experimental factors** in the thesis. In the "fixed R3" experiment condition, $y_b = 0$ always (R3 always feeds Line 2). In the "flexible R3" condition, $y_b$ is a free decision variable.

**Extra example:**
```
Suppose at time t=150, L1 has B_L1 = 8 batches (dangerously low, below θ^SS=20)
and L2 has B_L2 = 35 batches (nearly full).

Flexible R3: solver sets y_b = 1 → R3's next batch output goes to L1 → helps Line 1.
Fixed R3:    y_b = 0 forced → output goes to L2 → L2 gets fuller while L1 stays low
             → possible safety-idle penalty on R1, R2 (they're idle but L1 is low)

This is exactly what the experiment tests: does flexible routing significantly 
improve performance under disruption?
```

---

### 3.5 Output Line Function: $\text{out}(b)$

$$\text{out}(b) = \begin{cases} L_1 & \text{if } r_b \in \{R_1, R_2\} \\ L_1 & \text{if } r_b = R_3 \text{ and } y_b = 1 \\ L_2 & \text{if } r_b = R_3 \text{ and } y_b = 0 \\ L_2 & \text{if } r_b \in \{R_4, R_5\} \end{cases}$$

This is **not** a decision variable — it is derived from $r_b$ and $y_b$. It answers: "which line's RC buffer does this batch fill?"

**Critical distinction:**
```
pipe(r) = which pipeline delivers GC INTO the roaster (input side)
out(b)  = which line's buffer the roasted coffee goes TO (output side)

For R3:
  pipe(R3) = L2  ← always pulls GC from Line 2 pipeline
  out(b)   = L1 or L2 depending on y_b ← can send output to either line

For R1:
  pipe(R1) = L1  ← pulls GC from Line 1 pipeline
  out(b)   = L1  ← always sends output to Line 1 buffer (fixed)
```

---

### 3.6 Variable Summary

| Variable | Type | Count | Decided by solver? |
|----------|------|-------|--------------------|
| $a_b$ | Binary {0,1} | 160 | Yes (for PSC pool) |
| $r_b$ | Categorical | ≤164 | Yes |
| $s_b$ | Integer [0, 480−$p_k$] | ≤164 | Yes |
| $y_b$ | Binary {0,1} | ≤32 | Yes (in flexible mode) |

---

## 4. Auxiliary Variables — Derived Quantities

These are not independent decisions. They are calculated from the decision variables to make constraints and the objective function easier to write.

---

### 4.1 Batch End Time: $e_b$

$$e_b = s_b + p_{k_b}$$

A batch of SKU $k$ starting at $s_b$ ends at $s_b + p_k$. PSC ends 15 min later, NDG 17 min, Busta 18 min.

---

### 4.2 Consume Interval: $\text{con}(b)$

$$\text{con}(b) = [s_b, \; s_b + 3)$$

The 3-minute window during which the batch "pulls" GC from the pipeline. It runs **concurrently with the first 3 minutes of roasting** — the roaster is already processing while GC is flowing in.

**Extra example:**
```
PSC batch starts at s_b = 50.
  Roaster is busy: slots [50, 51, ..., 64]  (15 minutes, p_PSC = 15)
  Pipeline is used: slots [50, 51, 52]        (3 minutes only)
  Pipeline is free: from slot 53 onward        ← other roasters on same line can start

NDG batch starts at s_b = 50.
  Roaster is busy: slots [50, 51, ..., 66]  (17 minutes, p_NDG = 17)
  Pipeline is used: slots [50, 51, 52]        (3 minutes — same for all SKUs)
  Pipeline is free: from slot 53 onward

So if R1 starts PSC at slot 50 and R2 wants to start on the same L1 pipeline:
  R2 must wait until slot 53 at the earliest.
```

---

### 4.3 MTO Tardiness: $\text{tard}_j$

$$\text{tard}_j = \max\left(0,\; \max_{b \in \mathcal{B}_j} e_b - D^{MTO}\right)$$

**Plain English:** For each MTO job $j$, look at the latest-finishing batch in that job. If it finishes after slot 240 (10:00 AM), the tardiness is the number of minutes it is late. If all batches finish on time, tardiness is 0. Note: $e_b = s_b + p_{k_b}$, so NDG/Busta batches take 17/18 min respectively.

**Extra example with job j₁ (NDG × 3 batches, p_NDG = 17):**
```
Scenario A: Batches start at slots 178, 195, 213.
  Latest finish = 213 + 17 = 230 < 240 → tard_{j₁} = max(0, 230 - 240) = 0
  → No penalty. Job is on time.

Scenario B: Batches start at slots 178, 195, 241.
  Latest finish = 241 + 17 = 258 > 240 → tard_{j₁} = max(0, 258 - 240) = 18
  → Penalty = 18 × $1,000 = $18,000

Scenario C (disruption): UPS hits R1 at slot 180. 3rd NDG batch must restart.
  3rd batch restarts at slot 210, ends at slot 227. Still < 240.
  → tard_{j₁} = 0. No tardiness despite the disruption.
```

---

### 4.4 RC Inventory Level: $B_l(t)$

$$B_l(t) = B^0_l + \underbrace{\sum_{\substack{b: a_b=1,\; \text{out}(b)=l,\\ \text{sku}(b)=k^{PSC},\; e_b \leq t}} 1}_{\text{PSC batches completed on line } l \text{ by time } t} \;\;-\;\; \underbrace{\left|\{\tau \in \mathcal{E}_l : \tau \leq t\}\right|}_{\text{consumption events up to } t}$$

**Plain English:** Inventory at time $t$ = starting stock + PSC batches completed and sent to line $l$ − total consumption events that have occurred by time $t$.

**Key rules:**
1. Only **PSC** batches add to RC. NDG and Busta don't.
2. Only batches whose output is directed to **line $l$** add to $B_l$.
3. Consumption is deterministic — it happens at the precomputed times in $\mathcal{E}_l$.

**Worked example (Line 1 only, using $\rho_{L_1} = 8.1$ illustratively):**
```
Starting state: B⁰_L1 = 12 batches.
ρ_L1 = 8.1, so consumption events at slots: 8, 16, 24, 32, ...

t=0:   B_L1(0) = 12  (no events yet)
t=8:   Consumption event → B_L1(8) = 12 - 1 = 11
t=15:  A PSC batch on R1 completes (out(b)=L1, e_b=0+15=15) → B_L1 = 11 + 1 = 12
t=16:  Consumption event → B_L1(16) = 12 - 1 = 11
t=17:  An NDG batch on R2 completes (e_b=0+17=17) → NDG, NOT PSC → no RC change
t=24:  Another consumption event → B_L1(24) = 11 - 1 = 10
...

The inventory level zigzags up (when PSC batches complete) and down (consumption).
The solver's job is to schedule PSC batches so B_L1(t) stays above 0 at all times.
Note: NDG/Busta completions do NOT affect RC — only PSC does.
```

---

### 4.5 Stockout Count: $\text{SO}_l$

$$\text{SO}_l = \left|\{\tau \in \mathcal{E}_l : B_l(\tau) < 0\}\right|$$

**Plain English:** Count how many times the packaging line tried to grab a batch from an empty silo ($B_l < 0$ means demand couldn't be met).

> Note: $B_l = 0$ is NOT a stockout. It means the last batch was just consumed and the silo is now empty — but the demand WAS met. Stockout is only when $B_l$ goes **below zero** (strictly negative).

**Extra example:**
```
t=100: Consumption event on L1. B_L1 just before = 1.
  After draw: B_L1(100) = 1 - 1 = 0   → Not a stockout. Demand met.

t=105: Another consumption event. B_L1 just before = 0.
  After draw: B_L1(105) = 0 - 1 = -1  → STOCKOUT. Demand unmet. SO_L1 += 1.
  Penalty: $1,500

t=110: A PSC batch completed at t=108 added 1. B_L1 just before = -1 + 1 = 0.
  After draw: B_L1(110) = 0 - 1 = -1  → Another STOCKOUT. SO_L1 += 1.
  Total penalty so far: $3,000

In reactive mode (post-UPS), the hard constraint B_l ≥ 0 is relaxed, and these 
events are allowed but penalized. In deterministic mode, they are forbidden.
```

---

### 4.6 Roaster Busy Indicator: $\text{busy}_{r,t}$

$$\text{busy}_{r,t} = \begin{cases} 1 & \text{if roaster } r \text{ is processing a batch at slot } t \\ 0 & \text{otherwise} \end{cases}$$

**Extra example:**
```
PSC batch b assigned to R1, starts at s_b = 30, ends at e_b = 30 + 15 = 45.
  busy_{R1, 30} = 1
  busy_{R1, 35} = 1
  busy_{R1, 44} = 1
  busy_{R1, 45} = 0  ← batch ends AT 45, so slot 45 is free

NDG batch b' on R1 starts at s_{b'} = 50, ends at e_{b'} = 50 + 17 = 67:
  busy_{R1, 50} = 1
  busy_{R1, 66} = 1
  busy_{R1, 67} = 0  ← free (NDG takes 17 min, not 15)
```

> **Note:** `busy = 0` does not mean the roaster is ready to start a new batch. If the roaster is in SETUP (transitioning between SKUs), `busy = 0` but it's also not productively roasting.
>
> **Model vs. Simulation distinction:** In this deterministic model, there is no explicit SETUP state — setup is represented as an enforced **time gap** via C5 and C6. During this gap, `busy = 0`, and the safety-idle penalty (C14) applies if RC is low. In the simulation engine and the reactive model (`UPS_Mathematical_Model.md`), SETUP is an explicit roaster state with a countdown timer, because the re-solve must know a roaster's exact in-progress state at the moment of disruption.

---

### 4.7 Safety-Idle Indicator: $\text{idle}_{r,t}$

$$\text{idle}_{r,t} = 1 \quad \text{if: roaster } r \text{ is NOT busy AND not in downtime AND } B_{\ell(r)}(t) < 20$$

**Plain English:** This flag is raised when a roaster is sitting idle at a time when the RC buffer on its line is running low. The idea is: "you're doing nothing, and we're running out of stock — that's a problem."

**Important nuance:** Includes SETUP time. A roaster doing a 5-minute setup is technically not "busy" — so if RC is low during setup, the idle penalty still applies.

**Extra example:**
```
R1 in SETUP, slots [45, 49] (5 minutes between an NDG batch and PSC batch).
B_L1 at these slots = 18 (below θ^SS = 20).

idle_{R1, 45} = 1  (not busy, not in downtime, B < 20)
idle_{R1, 46} = 1
idle_{R1, 47} = 1
idle_{R1, 48} = 1
idle_{R1, 49} = 1

Penalty: 5 slots × $200 = $1,000

The solver "knows" this penalty exists and will try to schedule R1 to avoid
long setup periods when stock is low — for example, by grouping NDG batches
together rather than alternating NDG-PSC-NDG (which triggers two setups).
```

---

### 4.8 Overflow-Idle Indicator: $\text{over}_{r,t}$

$$\text{over}_{r,t} = 1 \quad \text{if: roaster } r \text{ is IDLE AND its output line's RC buffer is full (= 40 batches)}$$

**Plain English:** The roaster wants to run but literally cannot — the silo is full and there's no room to put the roasted coffee.

**R3 is special:** Since R3 can route output to either line, it is overflow-idle only if **both** lines are full.

**Extra example:**
```
t=300. R3 is idle. B_L1 = 40 (full), B_L2 = 37 (has room).
  → R3 can route to L2 → NOT overflow-idle.
  → No penalty.

t=350. R3 is idle. B_L1 = 40, B_L2 = 40 (both full).
  → R3 cannot route anywhere → OVERFLOW-IDLE.
  → Penalty: $50 per minute until one buffer drops below 40.

t=300. R4 is idle. B_L2 = 40 (full). (R4 always outputs to L2.)
  → R4 cannot start a batch → OVERFLOW-IDLE.
  → Penalty: $50 per minute.
  → Note: B_L1 doesn't matter for R4.
```

---

### 4.9 Setup Count Per Roaster: $N^{setup}_r$

$$N^{setup}_r = \left|\{(b_i, b_{i+1}) : b_i, b_{i+1} \text{ consecutive on } r,\; \text{sku}(b_i) \neq \text{sku}(b_{i+1})\}\right| \;+\; \mathbb{1}[\text{sku}(b^r_{\text{first}}) \neq k^{PSC}]$$

**Plain English:** Count the number of SKU transitions on roaster $r$ — every time two consecutive batches have different SKUs, plus the initial setup if the first batch is not PSC. Each such event incurs a lump-sum cost of $c^{setup} = \$800$.

**Extra example:**
```
R2 schedule: [Busta, Busta, PSC, PSC, PSC, NDG, PSC, PSC]
  → Initial: PSC→Busta (first batch ≠ PSC): +1
  → Busta→PSC: +1
  → PSC→NDG: +1
  → NDG→PSC: +1
  → Total N^setup_R2 = 4 setups × $800 = $3,200

R4 schedule: [PSC, PSC, PSC, ..., PSC]  (all same SKU, starts with PSC)
  → N^setup_R4 = 0 setups × $800 = $0
```

> **Implementation note:** In CP-SAT, setup count can be derived from the transition matrix applied to the NoOverlap sequence on each roaster. In MILP, it equals the sum of the binary ordering variables $\delta_{b_1, b_2}$ where $\text{sku}(b_1) \neq \text{sku}(b_2)$ and $b_2$ directly follows $b_1$ on the same roaster, plus any initial-setup indicator.

---

# PART III — CONSTRAINTS

> Rules the solver must obey. Violating any hard constraint produces an infeasible (invalid) schedule.

---

## 5. Constraints

### C1–C2: Batch Activation

- **(C1)** MTO batches are always active: $a_b = 1$ for all MTO batches. Non-negotiable.
- **(C2)** PSC batches are optional: $a_b \in \{0, 1\}$. The solver chooses.

---

### C3: Roaster Eligibility

$$r_b \in \mathcal{R}_{\text{sku}(b)} \quad \forall b : a_b = 1$$

The assigned roaster must be capable of producing the batch's SKU. Prevents, for example, a Busta batch being assigned to R4 (which can only do PSC).

---

### C4: Roaster NoOverlap

$$\text{NoOverlap}\left(\{[s_b, s_b + p_{k_b}) : r_b = r, a_b = 1\}\right) \quad \forall r \in \mathcal{R}$$

**Plain English:** On any single roaster, no two batches can run at the same time. The interval length depends on the batch's SKU: PSC occupies 15 min, NDG 17 min, Busta 18 min.

**Extra example:**
```
R1 has two batches assigned:
  b1 (PSC): s_b1 = 20 → runs [20, 34]  (15 min)
  b2 (NDG): s_b2 = 30 → would run [30, 46]  (17 min)

Overlap: [30, 34] — VIOLATION of C4.

Fix: delay b2 to start at s_b2 = 35 → runs [35, 51] → no overlap.
Note: b1 (PSC, 15 min) finishes at 35, b2 (NDG) starts at 35. But b1 ≠ b2 SKU,
so C5 (setup time) also applies — see below.
```

---

### C5: Sequence-Dependent Setup Time

For any two batches $b_1, b_2$ assigned to the **same roaster** with $\text{sku}(b_1) \neq \text{sku}(b_2)$:

$$s_{b_2} \geq s_{b_1} + p_{k_{b_1}} + \sigma \quad \text{OR} \quad s_{b_1} \geq s_{b_2} + p_{k_{b_2}} + \sigma$$

That is, whichever batch comes second must start at least $p_{k_{b_1}} + \sigma$ minutes after the first one starts (processing time of the first batch plus setup). This is a **disjunctive** constraint — the solver decides the ordering.

> **Implementation note:** In CP-SAT, this is handled via the `NoOverlap` transition matrix: `transition[k1][k2] = σ` when `k1 ≠ k2`, and `0` otherwise. In MILP, it requires Big-M ordering binaries (delta). The mathematical statement above is solver-agnostic.

**Extra example:**
```
R2 runs: NDG batch (b1) then PSC batch (b2).  [solver chose b1 first]
  s_b1 = 5, e_b1 = 5 + 17 = 22.  (NDG takes 17 min)
  → s_b2 ≥ 5 + 17 + 5 = 27  (need 5 extra minutes for setup after NDG finishes)

R2 runs: PSC batch (b3) then PSC batch (b4).  [same SKU]
  s_b3 = 27, e_b3 = 27 + 15 = 42.  (PSC takes 15 min)
  → s_b4 ≥ 42  (same SKU, no setup needed, can start immediately after)

R2 runs: Busta batch (b5) then NDG batch (b6).  [different SKUs]
  s_b5 = 42, e_b5 = 42 + 18 = 60.  (Busta takes 18 min)
  → s_b6 ≥ 42 + 18 + 5 = 65  (setup required)
```

---

### C6: Initial SKU Setup

$$s_{b_{\text{first}}^r} \geq 5 \quad \text{if the first batch on roaster } r \text{ is NOT PSC}$$

Since every roaster "starts the day" in PSC mode, the first batch of a different SKU needs a setup.

**Extra example:**
```
R1's first batch of the shift is NDG.
  → 5-min setup required before roasting begins
  → s_{first batch on R1} ≥ 5
  → Earliest the NDG can start: slot 5

R3's first batch of the shift is PSC.
  → No setup (PSC → PSC, same SKU)
  → s_{first batch on R3} ≥ 0
  → Can start at slot 0
```

---

### C7: Planned Downtime

$$[s_b, s_b + p_{k_b} - 1] \cap \mathcal{D}_r = \emptyset \quad \forall b : r_b = r$$

Every minute of the batch must fall outside the downtime window. Batches cannot be paused mid-run. The batch duration depends on its SKU ($p_{k_b}$).

**Extra example:**
```
D_R3 = {200, ..., 229}. R3 can only produce PSC (p_PSC = 15).

Start at 185: runs [185, 199]. Ends at 200. ✓ (no overlap with {200,...})
Start at 186: runs [186, 200]. Overlap at slot 200. ✗ VIOLATION.
Start at 230: runs [230, 244]. ✓ (starts after downtime ends)

If R1 runs NDG (p_NDG = 17) with D_R1 = {200, ..., 229}:
Start at 183: runs [183, 199]. ✓ (17-min batch: 183+16=199, just clears)
Start at 184: runs [184, 200]. Overlap at 200. ✗ VIOLATION.
```

---

### C8: Pipeline NoOverlap (Core Bottleneck)

$$\text{NoOverlap}\left(\{[s_b, s_b+3) : \text{pipe}(r_b) = l, a_b = 1\}\right) \quad \forall l \in \mathcal{L}$$

**Plain English:** On the same GC pipeline, no two roasters can pull GC at overlapping times. Their 3-minute consume windows must not overlap.

**Extra example:**
```
Line 2 pipeline, serving R3, R4, R5.

Scenario: R3 starts at s=10, R4 starts at s=12.
  R3 consume window: [10, 12]
  R4 consume window: [12, 14]
  Overlap at slot 12: VIOLATION.
  → R4 must start at s=13 or later.

Scenario: R3 starts at s=10, R4 starts at s=13.
  R3 consume window: [10, 12]
  R4 consume window: [13, 15]
  No overlap. ✓

Throughput implication: With 3 roasters on L2, the earliest possible 
staggered starts are s=0, 3, 6 (or 0, 3, 6+setup time).
Maximum throughput on L2: 1 batch start every 3 minutes across the 3 roasters.
```

---

### C9: End-of-Shift

$$s_b \leq 480 - p_{k_b} \quad \forall b : a_b = 1$$

Any batch must complete within the shift. The latest valid start time depends on SKU: PSC can start at 465, NDG at 463, Busta at 462.

---

### C10: RC Inventory Lower Bound (Stockout Prevention)

**Deterministic mode (hard):**
$$B_l(t) \geq 0 \quad \forall t \in \mathcal{E}_l$$

The silo must never go below zero at any consumption event. In deterministic mode, the solver must guarantee this.

**Reactive mode (soft):** This constraint is relaxed. Stockouts are allowed but penalized at $c^{stock} = \$1{,}500$ per event. This reflects real-world disruptions where perfect inventory management is impossible.

---

### C11: RC Inventory Upper Bound (Overflow Prevention)

$$B_l(t) \leq 40 \quad \text{at all batch completion times}$$

Hard in all modes. A silo that is physically full cannot accept more coffee — the batch's output has nowhere to go.

**Extra example:**
```
B_L1 = 39. A PSC batch on R2 completes at t=100 (out(b) = L1).
  After completion: B_L1 = 39 + 1 = 40. ✓ (exactly full — still allowed)

B_L1 = 40. Another PSC batch on R1 completes at t=110 (out(b) = L1).
  Would become: B_L1 = 40 + 1 = 41. ✗ VIOLATION — silo overflow.
  → The solver must have NOT scheduled this PSC batch (a_b = 0), or
    must have routed R3's output to L2 instead, or
    must have delayed the batch until a consumption event drops B_L1 first.
```

---

### C12: MTO Tardiness Computation

$$\text{tard}_j \geq e_b - 240 \quad \forall b \in \mathcal{B}_j, \qquad \text{tard}_j \geq 0$$

This pair of inequalities implements $\text{tard}_j = \max(0, \text{latest finish time} - 240)$ in a linear form the solver can handle. Since the objective penalizes $\text{tard}_j$, the solver will make it as small as possible — which means it will push MTO batches to finish by slot 240.

---

### C13: R3 Output Routing

$$y_b \in \{0, 1\} \quad \forall b : r_b = R_3, a_b = 1$$

Defines the decision variable. Links to $B_l(t)$ via the $\text{out}(b)$ function (Section 3.5).

---

### C14: Safety-Idle Detection

$$\text{idle}_{r,t} \geq (1 - \text{busy}_{r,t}) + \text{low}_{\ell(r),t} - 1$$

Where $\text{low}_{l,t} = 1$ if and only if $B_l(t) < 20$.

**How to read this:** The right side equals 1 only when BOTH conditions hold (not busy AND RC low). Since the objective penalizes $\text{idle}_{r,t}$, the solver sets it to 0 whenever possible — this one-sided inequality is sufficient.

---

### C15: Overflow-Idle Detection

Similar structure to C14, but triggers when a roaster is idle AND RC buffer is full (= 40).

For R3 specifically (because it can route to either line):
$$\text{over}_{R_3,t} \geq (1 - \text{busy}_{R_3,t}) + \text{full}_{L_1,t} + \text{full}_{L_2,t} - 2$$

R3 is overflow-idle only when **both** lines are full (sum of full indicators = 2, so right side = 1). If either line has room, the right side ≤ 0, and $\text{over}_{R_3,t}$ can be 0.

---

### C16: GC Silo Balance (Input Inventory)

$$G_{l,k}(t) = G^0_{l,k} + Q^{restock} \cdot \text{restocks\_completed}_{l,k}(t) - \text{batches\_started}_{l,k}(t)$$
$$G_{l,k}(t) \geq 0 \quad \forall t \text{ where a batch of SKU } k \text{ starts on line } l$$

**Plain English:** Each GC silo tracks how much green coffee is available. It starts at $G^0_{l,k}$, increases by 5 each time a restock completes, and decreases by 1 each time a batch of that SKU starts on a roaster fed by that line's pipeline. The silo level must never go negative — you cannot roast without green coffee.

**Hard constraint in all modes.** Unlike RC stockout (soft in reactive mode), GC stockout is physically impossible — you literally cannot start a roast without beans in the hopper.

**Extra example:**
```
Line 2, PSC silo. G⁰_{L2,PSC} = 20 batches.

t=0:   R3 starts PSC → G = 20 - 1 = 19
t=3:   R4 starts PSC → G = 19 - 1 = 18
t=6:   R5 starts PSC → G = 18 - 1 = 17
...continuing at ~1 batch per 5 min across 3 roasters...
t=80:  G_{L2,PSC} = 4.  Getting low.
t=82:  Restock completes (started at t=67) → G = 4 + 5 = 9.
t=85:  R3 starts PSC → G = 9 - 1 = 8.

Without the restock at t=67: G would hit 0 around t=95.
  → All three Line 2 roasters forced idle until restocked.
```

---

### C17: Restock Pipeline Block

$$\text{NoOverlap}\left(\{[s_b, s_b + \delta^{con}) : \text{pipe}(r_b) = l\} \;\cup\; \{[s^{rst}_i, s^{rst}_i + \delta^{restock}) : \text{line}(rst_i) = l\}\right) \quad \forall l \in \mathcal{L}$$

**Plain English:** Restock intervals participate in the **same NoOverlap constraint** as batch consume intervals on the pipeline. During a 15-minute restock, no roaster on that line can start a new batch (because starting requires 3 minutes of pipeline access, and the pipeline is occupied by restocking).

This is an **extension of C8** — the pipeline NoOverlap set now includes both 3-min batch consume intervals AND 15-min restock intervals.

**Extra example:**
```
Line 2 restock starts at t=100, occupies pipeline [100, 115).

R3 finishes batch at t=105. Wants to start new batch.
  → Needs pipeline for consume at [105, 108).
  → Pipeline blocked by restock until t=115. CANNOT START.
  → R3 idle from t=105 to t=115 (10 min forced idle).

R4 finishes batch at t=112. Same situation.
  → R4 idle from t=112 to t=115 (3 min forced idle).

R5 finishes batch at t=114.
  → R5 idle from t=114 to t=115 (1 min), then can consume at [115, 118).

After t=115: pipeline free. All three roasters compete for pipeline 
in normal 3-min stagger. First come first served.
```

---

### C18: Shared Restock Station (Global Mutex)

$$\text{NoOverlap}\left(\{[s^{rst}_i, s^{rst}_i + \delta^{restock}) : \forall \text{ restock } i \text{ on any line}\}\right)$$

**Plain English:** Only one restock can occur at a time across **both lines**. There is a single restock station (manual big-bag loading) shared by the entire factory. If Line 1 is restocking PSC, Line 2 cannot restock anything until Line 1's restock finishes.

This creates **cross-line coupling** independent of R3 routing. Even if R3 routing is fixed (lines decoupled for RC), the shared restock station couples them for GC supply.

**Extra example:**
```
Line 1 needs PSC restock (silo at 3 batches).
Line 2 needs PSC restock (silo at 2 batches — more urgent).

If Line 2 restocks first: [100, 115) → Line 1 must wait → restocks at [115, 130).
  Line 1 PSC silo may hit 0 before t=115 if consumption is fast.

If Line 1 restocks first: [100, 115) → Line 2 must wait → restocks at [115, 130).
  Line 2 PSC silo may hit 0 before t=115.

→ The scheduler must decide WHO restocks first based on urgency, 
  upcoming UPS risk, and downstream impact. Greedy picks whoever 
  is lower; optimal solver considers future consequences.
```

---

### C19: Restock Capacity Guard

$$G_{l,k}(s^{rst}_i) + Q^{restock} \leq \overline{G}_{l,k} \quad \forall \text{ restock } i \text{ on silo } (l,k)$$

**Plain English:** A restock is only allowed if the silo has room for 5 more batches. If a silo is at 36/40, restocking would push it to 41 — which exceeds capacity. The restock must wait until the silo level drops to at most $\overline{G}_{l,k} - Q^{restock}$.

| Silo | Capacity $\overline{G}$ | Max level to allow restock |
|------|------------------------|---------------------------|
| L1 PSC | 40 | ≤ 35 |
| L1 NDG | 10 | ≤ 5 |
| L1 Busta | 10 | ≤ 5 |
| L2 PSC | 40 | ≤ 35 |

> **Note on NDG/Busta silos:** With capacity 10 and restock size 5, these silos can hold at most 2 restocks worth. Starting at 5, the first restock can happen when the silo drops to 5 (5+5=10, at capacity). The NDG silo must drop to 5 or below before restocking — which happens after 5 NDG batches. With only 3-4 NDG batches per shift, restocking NDG is rare (0-1 times).

---

### Constraint Summary Table

| ID | What it enforces | Hard or Soft? | Penalty if soft |
|----|-----------------|---------------|-----------------|
| C1 | All MTO batches must run | Hard | — |
| C2 | PSC batches are optional | — | — |
| C3 | Roaster must be eligible for SKU | Hard | — |
| C4 | No two batches on same roaster at same time (duration = $p_k$) | Hard | — |
| C5 | $p_k + 5$ min gap when switching SKUs on same roaster | Hard | — |
| C6 | First batch respects initial SKU state | Hard | — |
| C7 | Batch must not overlap planned downtime | Hard | — |
| C8 | No two batch consumes on same pipeline simultaneously | Hard | — |
| C9 | All batches complete within shift ($s_b \leq 480 - p_k$) | Hard | — |
| C10 | RC silo ≥ 0 (no stockout) | Hard (det.) / Soft (reactive) | $1,500/event |
| C11 | RC silo ≤ 40 (no overflow) | Hard | — |
| C12 | Tardiness is correctly computed | Soft | $1,000/min |
| C13 | R3 routing decision is binary | — | — |
| C14 | Safety-idle is detected for penalty | Soft | $200/min/roaster |
| C15 | Overflow-idle is detected for penalty | Soft | $50/min/roaster |
| **C16** | **GC silo ≥ 0 (can't roast from empty silo)** | **Hard** | — |
| **C17** | **Restock blocks pipeline (15 min, in NoOverlap with consumes)** | **Hard** | — |
| **C18** | **Only 1 restock at a time globally (shared station)** | **Hard** | — |
| **C19** | **Restock only if silo has room ($G + 5 \leq$ capacity)** | **Hard** | — |

---

# PART IV — OBJECTIVE FUNCTION

## 6. Maximize Profit

### 6.1 Deterministic Mode

$$\text{Maximize:} \quad \underbrace{\sum_{b: a_b=1} R_{\text{sku}(b)}}_{\text{Revenue}} \;-\; \underbrace{c^{tard} \sum_j \text{tard}_j}_{\text{Tardiness penalty}} \;-\; \underbrace{c^{setup} \sum_r N^{setup}_r}_{\text{Setup cost}} \;-\; \underbrace{c^{idle} \sum_{r,t} \text{idle}_{r,t}}_{\text{Safety-idle penalty}} \;-\; \underbrace{c^{over} \sum_{r,t} \text{over}_{r,t}}_{\text{Overflow-idle penalty}}$$

**No stockout term** in deterministic mode — stockouts are forbidden by hard constraint C10.

**Full worked example:**

```
Suppose the solver produces this schedule:

  62 PSC batches completed → Revenue = 62 × $4,000 = $248,000
   3 NDG batches completed → Revenue =  3 × $7,000 =  $21,000
   1 Busta batch completed → Revenue =  1 × $7,000 =   $7,000
  Total Revenue = $276,000

  MTO tardiness (j₁ NDG): last batch ends at slot 243 → tard = 3 min
  Tardiness cost = 3 × $1,000 = $3,000

  Setup events: R1 (PSC→NDG + NDG→PSC = 2), R2 (PSC→Busta + Busta→PSC = 2)
  Setup cost = 4 × $800 = $3,200

  Safety-idle: R1 idle 10 min + R2 idle 5 min while B_L1 < 20
  Safety-idle cost = 15 min × $200 = $3,000

  Overflow-idle: 0 minutes (RC never hit 40 batches)
  Overflow-idle cost = $0

NET PROFIT = $276,000 − $3,000 − $3,200 − $3,000 − $0 = $266,800
```

**Tradeoff example — should we run more PSC batches even if it causes more tardiness?**
```
Option A: 62 PSC batches, 3 min tardiness, 4 setups
  Profit = $276,000 − $3,000 − $3,200 − $3,000 = $266,800

Option B: 65 PSC batches (3 more), but NDG finishes 10 min late, 4 setups
  Revenue = 65×$4k + 3×$7k + 1×$7k = $292,000
  Tardiness = 10 × $1,000 = $10,000
  Setup cost = 4 × $800 = $3,200
  Safety-idle = 0 (RC is well-stocked now)
  Profit = $292,000 − $10,000 − $3,200 = $278,800 → BETTER

Option C: 60 PSC batches, 0 tardiness, 80 min safety-idle, 4 setups
  Revenue = 60×$4k + $28k = $268,000
  Tardiness = $0, Setup = $3,200, Safety-idle = 80×$200 = $16,000
  Profit = $268,000 − $3,200 − $16,000 = $248,800 → WORSE

→ Option B wins: the 3 extra PSC batches ($12k revenue) outweigh the 
  extra tardiness ($10k penalty) — but only just. This is exactly the 
  kind of tradeoff the objective function is designed to balance.
```

---

### 6.2 Reactive Mode (Post-UPS Re-solve)

After an unplanned stoppage (UPS) at time $t_0$, the solver re-optimizes only the **remaining** shift $[t_0, 479]$:

$$\text{Maximize:} \sum_{b \in \mathcal{B}_{rem}} R_{\text{sku}(b)} \cdot a_b \;-\; c^{tard}\sum_j \text{tard}_j \;-\; c^{stock}\sum_l \text{SO}_l \;-\; c^{setup}\sum_r N^{setup}_r \;-\; c^{idle}\sum_{r,t \geq t_0} \text{idle}_{r,t} \;-\; c^{over}\sum_{r,t \geq t_0} \text{over}_{r,t}$$

Key changes from deterministic mode:
1. **Stockout term added** ($c^{stock} \cdot \text{SO}_l$) — because disruptions may make it impossible to prevent all stockouts.
2. **Setup cost** ($c^{setup} \cdot N^{setup}_r$) — same as deterministic; UPS recovery may force re-setups that would not have occurred otherwise.
3. **$\mathcal{B}_{rem}$** — the batch pool is rebuilt for the remaining shift. Completed batches are removed; the cancelled batch (UPS victim) is removed; in-progress batches become fixed constraints.
4. Revenue is only counted for batches completed **after** $t_0$ and still to be scheduled. Already-completed batches don't re-enter the objective (their revenue is "locked in").

**Extra example — UPS hits R1 at t=180:**
```
State at t=180:
  R1 was running NDG batch b*, started at s=175, would end at s=190.
  → b* is CANCELLED (GC lost, batch must restart from scratch).
  R2 is running PSC batch, started at s=173, ends at s=188.
  R3 completed 8 PSC batches. B_L2 = 28.
  B_L1 = 15 (below θ^SS = 20, so safety-idle pressure exists).

Re-solve at t=180 with horizon [180, 479]:
  - R1 is unavailable until UPS ends, say t=200 (20-min repair).
  - The 3rd NDG batch (originally b*) must be re-scheduled on R1 (after t=200) 
    or on R2 (if R2 is free).
  - Tardiness penalty kicks in if NDG finishes after slot 240.
    → NDG started at t=205 on R1 → ends at t=220 → tard = 0. OK.
  - Stockout risk: L1 is now short. PSC roasters (R1 offline until 200) 
    cannot help for 20 minutes. CP-SAT re-solve should schedule R3 to route 
    to L1 (y_b = 1) as a mitigation.
```

---

# PART V — HOW THE SOLVERS USE THIS MODEL

## CP-SAT (Primary Solver)

- Solves the full model at shift start (deterministic: $t_0 = 0$).
- Re-solves the reduced model whenever a UPS is detected (event-triggered).
- Uses native `NoOverlap` constraints for C4 and C8 — more efficient than MILP's big-M.
- Uses interval variables for batches — each batch is an "object" with a start, end, and machine assignment.

## MILP (Benchmark Only)

- Solves the **same deterministic model** as CP-SAT at shift start (no UPS).
- Uses binary ordering variables and big-M constraints to replicate NoOverlap.
- Provides an LP relaxation lower bound — used to verify CP-SAT solution quality.
- Not used in reactive experiments.

## DRL / MaskablePPO (Learning-Based)

- Does **not** use this mathematical model directly.
- Instead, learns a policy by running episodes in the simulation environment.
- The action mask enforces C3 (eligibility) and C8 (pipeline) in real-time, preventing the agent from selecting physically invalid actions.
- The observation vector (21 features) gives the agent the state information it needs to make decisions.

---

# QUICK REFERENCE: MODEL AT A GLANCE

```
╔════════════════════════════════════════════════════════════════════╗
║                  DETERMINISTIC MODEL SUMMARY                       ║
╠════════════════════════════════════════════════════════════════════╣
║ FACTORY:     5 roasters (R1-R2 on L1, R3-R5 on L2), 2 lines      ║
║ TIME:        480 min slots, s_b ∈ [0, 480 - p_k]                  ║
║ ROASTING:    p_PSC=15, p_NDG=17, p_BUS=18 min (SKU-dependent).   ║
║              Pipeline pull: 3 min concurrent. Setup: 5 min.       ║
║ RC BUFFER:   Max 40 batches/line. Safety threshold: 20 batches.   ║
║ GC SILOS:    L1: PSC(40), NDG(10), Busta(10). L2: PSC(40).       ║
║              Restock: +5 batches, 15-min pipeline block.          ║
║              Shared station: 1 restock at a time globally.        ║
║ PRODUCTS:    PSC → fills RC buffer. NDG/Busta → direct delivery.  ║
║ CONSUMPTION: ρ_l min/batch per line (from CSV). Deterministic.    ║
║ OBJECTIVE:   MAX PROFIT = Revenue − Tardiness − Setup − Idle      ║
║ REVENUE:     PSC $4k/batch, NDG $7k/batch, Busta $7k/batch       ║
║ PENALTIES:   Tardiness $1k/min >                                  ║
║              Safety-idle $200/min > Overflow-idle $50/min         ║
║              (Stockout $1.5k/event — reactive mode only)          ║
║ CONSTRAINTS: 19 groups (C1–C19). Stockout = HARD (B_l ≥ 0).      ║
║              GC silo ≥ 0 = HARD. Restock pipeline block = HARD.  ║
║ VARIABLES:   ~164 start times, ~164 roaster assignments,         ║
║              160 activation binaries, ≤32 R3 routing binaries,   ║
║              restock intervals (decision variables)               ║
║ REACTIVE:    See UPS_Mathematical_Model.md for post-disruption.  ║
╚════════════════════════════════════════════════════════════════════╝
```
