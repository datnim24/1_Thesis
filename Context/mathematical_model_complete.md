# Mathematical Model — Annotated Explanation
### Dynamic Batch Roasting Scheduling at Nestlé Trị An

> **Purpose of this document:** This is a plain-language walkthrough of the full mathematical model. Every symbol is re-explained from scratch, with extra worked examples added wherever the original model was brief. Read this alongside `mathematical_model_complete.md`.

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

---

### 1.10 PSC Batch Pool: $\mathcal{B}^{PSC}$

$$\mathcal{B}^{PSC} = \{b^{PSC}_1, \dots, b^{PSC}_{160}\}$$

**How the pool works:** The solver does not know in advance how many PSC batches it will run. Instead, a pool of 160 "candidate" PSC batches is created (the theoretical maximum if all 5 roasters ran PSC non-stop for 480 minutes: $5 \times \lfloor 480/15 \rfloor = 160$). The solver then *activates* as many as it needs.

**Analogy:** Imagine 160 blank job tickets. The solver stamps "USE" or "SKIP" on each one. The stamps it marks "USE" become the actual PSC schedule.

**Extra example:**
```
If B⁰_L1 = 12 (Line 1 starts with 12 batches of stock) and the line
consumes 1 batch every 5.1 minutes:
  → Line 1 needs roughly 94 more PSC deliveries over the shift
  → But with 2 roasters (R1, R2) occasionally running MTO, the solver
     might only activate ~30 PSC batches on L1 roasters
  → The other PSC demand is met by R3 routing output to L1 (via y_b=1)
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
| $P$ | 15 min | Every batch takes exactly 15 minutes to roast, regardless of SKU |
| $\delta^{con}$ | 3 min | When a batch starts, it "consumes" GC from the pipeline for the first 3 minutes |
| $\sigma$ | 5 min | If a roaster switches from one SKU to a different SKU, it needs 5 minutes of cleanup/setup |
| $D^{MTO}$ | slot 240 | MTO orders are "due" by the halfway point of the shift (10:00 AM if shift starts 6:00 AM) |

**Extra example — why $\delta^{con} = 3$ matters:**
```
Timeline for a batch starting at slot 10:
  [10, 11, 12]     ← GC is being pulled through the pipeline (3 min)
  [10, 11, ..., 24] ← Roaster is busy for all 15 min
  [25 onward]      ← Roaster is free; batch is done

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

```
ρ_L1 = 5.1 minutes per batch
  → Line 1 consumes 1 batch of RC every 5.1 minutes
  → Over 480 minutes: ⌊480 / 5.1⌋ = 94 consumption events

ρ_L2 = 4.8 minutes per batch  (slightly faster line)
  → Line 2 consumes 1 batch every 4.8 minutes
  → Over 480 minutes: ⌊480 / 4.8⌋ = 100 consumption events

Implication: Line 2 is more demanding — it needs more PSC replenishment
from roasters R3, R4, R5 (and possibly R3 rerouted to L1 for help).
```

---

### 2.3 Cost Structure

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

### 2.4 PSC Consumption Schedule: $\mathcal{E}_l$

$$\mathcal{E}_l = \left\{ \lfloor i \cdot \rho_l \rfloor \;\middle|\; i = 1, 2, \dots, \lfloor 480 / \rho_l \rfloor \right\}$$

**Plain English:** This is the pre-computed list of exact time slots when Line $l$'s packaging machine "reaches in" and grabs one batch from the RC silo. It is deterministic and known at shift start.

The symbol $\lfloor \cdot \rfloor$ means "round down to the nearest integer."

**Extra example:**
```
ρ_L1 = 5.1 minutes per batch.

i=1:  ⌊1 × 5.1⌋ = ⌊5.1⌋  = 5   ← first draw at slot 5
i=2:  ⌊2 × 5.1⌋ = ⌊10.2⌋ = 10  ← second draw at slot 10
i=3:  ⌊3 × 5.1⌋ = ⌊15.3⌋ = 15
i=4:  ⌊4 × 5.1⌋ = ⌊20.4⌋ = 20
i=5:  ⌊5 × 5.1⌋ = ⌊25.5⌋ = 25
i=6:  ⌊6 × 5.1⌋ = ⌊30.6⌋ = 30
i=7:  ⌊7 × 5.1⌋ = ⌊35.7⌋ = 35
i=8:  ⌊8 × 5.1⌋ = ⌊40.8⌋ = 40
i=9:  ⌊9 × 5.1⌋ = ⌊45.9⌋ = 45
i=10: ⌊10 × 5.1⌋= ⌊51.0⌋ = 51  ← "gap" shifts by rounding

So E_L1 = {5, 10, 15, 20, 25, 30, 35, 40, 45, 51, 56, 61, 66, ...}

Note: the gap is sometimes 5 and sometimes 6 — this is the floor() effect.
Total events: ⌊480 / 5.1⌋ = 94 consumption events during the shift.
```

> **Why precompute this?** The model only needs to check inventory at these ~94 moments per line, not at all 480 slots. This makes the constraint count manageable.

---

### 2.5 Planned Downtime: $\mathcal{D}_r$

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

### 2.6 Initial SKU State

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

### 2.7 Complete Example Instance (Reference)

This instance is used for all examples throughout the rest of this document:

```
SHIFT:       480 minutes (slots 0–479)
             P = 15 min, δ_con = 3 min, σ = 5 min, D_MTO = 240

ROASTERS:    R1 (L1), R2 (L1), R3 (L2), R4 (L2), R5 (L2)
             All start with last_sku = PSC

MTO ORDERS:  j₁: NDG × 3 batches    (must run on R1 or R2)
             j₂: Busta × 1 batch    (must run on R2)
             Total: 4 mandatory batches

PSC:         ρ_L1 = 5.1 min/batch → 94 consumption events
             ρ_L2 = 4.8 min/batch → 100 consumption events
             Pool: 160 candidate batches

RC STOCK:    B⁰_L1 = 12 batches    B⁰_L2 = 15 batches
             Max = 40 batches/line  Safety threshold = 20 batches/line

DOWNTIME:    D_R3 = {200, ..., 229} (30-minute window)

COSTS:       R_PSC=$4k, R_NDG=$7k, R_BUS=$7k
             c_tard=$1k/min, c_stock=$1.5k/event
             c_over=$50/min/roaster, c_idle=$200/min/roaster
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

$$s_b \in [0, 465] \quad \forall b : a_b = 1$$

The slot at which batch $b$ begins roasting. The upper bound of 465 (not 479) ensures every batch fully completes within the shift:

$$e_b = s_b + 15 \leq 480 \implies s_b \leq 465$$

**Extra example:**
```
s_b = 0   → batch runs [0, 14], ends at slot 15. Fine.
s_b = 465 → batch runs [465, 479], ends at slot 480. Fine (exactly at shift end).
s_b = 466 → batch would run [466, 480], ends at slot 481. INVALID — exceeds shift.
```

**Key derived quantity:** $e_b = s_b + 15$ (end time). This is not a decision — it is calculated from $s_b$.

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
| $s_b$ | Integer [0,465] | ≤164 | Yes |
| $y_b$ | Binary {0,1} | ≤32 | Yes (in flexible mode) |

---

## 4. Auxiliary Variables — Derived Quantities

These are not independent decisions. They are calculated from the decision variables to make constraints and the objective function easier to write.

---

### 4.1 Batch End Time: $e_b$

$$e_b = s_b + 15$$

Trivial: a batch starting at $s_b$ always ends at $s_b + 15$.

---

### 4.2 Consume Interval: $\text{con}(b)$

$$\text{con}(b) = [s_b, \; s_b + 3)$$

The 3-minute window during which the batch "pulls" GC from the pipeline. It runs **concurrently with the first 3 minutes of roasting** — the roaster is already processing while GC is flowing in.

**Extra example:**
```
Batch starts at s_b = 50.
  Roaster is busy: slots [50, 51, ..., 64]  (15 minutes)
  Pipeline is used: slots [50, 51, 52]        (3 minutes only)
  Pipeline is free: from slot 53 onward        ← other roasters on same line can start

So if R1 starts at slot 50 and R2 wants to start on the same L1 pipeline:
  R2 must wait until slot 53 at the earliest.
```

---

### 4.3 MTO Tardiness: $\text{tard}_j$

$$\text{tard}_j = \max\left(0,\; \max_{b \in \mathcal{B}_j} e_b - D^{MTO}\right)$$

**Plain English:** For each MTO job $j$, look at the latest-finishing batch in that job. If it finishes after slot 240 (10:00 AM), the tardiness is the number of minutes it is late. If all batches finish on time, tardiness is 0.

**Extra example with job j₁ (NDG × 3 batches):**
```
Scenario A: Batches finish at slots 195, 210, 230.
  Latest finish = 230 < 240 → tard_{j₁} = max(0, 230 - 240) = max(0, -10) = 0
  → No penalty. Job is on time.

Scenario B: Batches finish at slots 195, 210, 255.
  Latest finish = 255 > 240 → tard_{j₁} = max(0, 255 - 240) = 15
  → Penalty = 15 × $1,000 = $15,000

Scenario C (disruption): UPS hits R1 at slot 180. 3rd NDG batch must restart.
  3rd batch restarts at slot 210, ends at slot 225. Still < 240.
  → tard_{j₁} = 0. No tardiness despite the disruption.
  (This is a "good" reactive schedule — the CP-SAT re-solve found a recovery.)
```

---

### 4.4 RC Inventory Level: $B_l(t)$

$$B_l(t) = B^0_l + \underbrace{\sum_{\substack{b: a_b=1,\; \text{out}(b)=l,\\ \text{sku}(b)=k^{PSC},\; e_b \leq t}} 1}_{\text{PSC batches completed on line } l \text{ by time } t} \;\;-\;\; \underbrace{\left|\{\tau \in \mathcal{E}_l : \tau \leq t\}\right|}_{\text{consumption events up to } t}$$

**Plain English:** Inventory at time $t$ = starting stock + PSC batches completed and sent to line $l$ − total consumption events that have occurred by time $t$.

**Key rules:**
1. Only **PSC** batches add to RC. NDG and Busta don't.
2. Only batches whose output is directed to **line $l$** add to $B_l$.
3. Consumption is deterministic — it happens at the precomputed times in $\mathcal{E}_l$.

**Worked example (Line 1 only, simplified):**
```
Starting state: B⁰_L1 = 12 batches.
ρ_L1 = 5.1, so consumption events at slots: 5, 10, 15, 20, 25, ...

t=0:   B_L1(0) = 12  (no events yet)
t=5:   Consumption event → B_L1(5) = 12 - 1 = 11
t=10:  Another event → B_L1(10) = 11 - 1 = 10
t=15:  A PSC batch on R2 completes (out(b)=L1, e_b=15) → adds 1 FIRST, then:
       Consumption event also at t=15 → subtract 1
       Net: B_L1(15) = 10 + 1 - 1 = 10  ← same level, batch just replaced the draw
t=20:  Another consumption event → B_L1(20) = 10 - 1 = 9
...
t=51:  If another batch completed at t=45 (started at t=30):
       B_L1 = 9 + 1 = 10 after completion, then -1 at t=51 → 9

The inventory level zigzags up (when PSC batches complete) and down (consumption).
The solver's job is to schedule PSC batches so B_L1(t) stays above 0 at all times.
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
Batch b assigned to R1, starts at s_b = 30, ends at e_b = 45.
  busy_{R1, 30} = 1
  busy_{R1, 35} = 1
  busy_{R1, 44} = 1
  busy_{R1, 45} = 0  ← batch ends AT 45, so slot 45 is free

Batch b' on R1 starts at s_{b'} = 50:
  busy_{R1, 45} = 0  (gap between batches: slots 45, 46, 47, 48, 49 are idle)
  busy_{R1, 50} = 1
```

> **Note:** `busy = 0` does not mean the roaster is ready to start a new batch. If the roaster is in SETUP (transitioning between SKUs), `busy = 0` but it's also not IDLE. The model tracks three states: BUSY (running a batch), SETUP (cleaning/reconfiguring), IDLE (genuinely free).

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

$$\text{NoOverlap}\left(\{[s_b, s_b+15) : r_b = r, a_b = 1\}\right) \quad \forall r \in \mathcal{R}$$

**Plain English:** On any single roaster, no two batches can run at the same time.

**Extra example:**
```
R1 has two batches assigned:
  b1: s_b1 = 20 → runs [20, 34]
  b2: s_b2 = 30 → would run [30, 44]

Overlap: [30, 34] — VIOLATION of C4.

Fix: delay b2 to start at s_b2 = 35 → runs [35, 49] → no overlap.
```

---

### C5: Sequence-Dependent Setup Time

$$\text{If } r_{b_1} = r_{b_2} \text{ and } \text{sku}(b_1) \neq \text{sku}(b_2): \quad s_{b_2} \geq s_{b_1} + 20$$

**Plain English:** When switching between different product types on the same roaster, a 5-minute gap is needed on top of the 15-minute roasting time ($15 + 5 = 20$ total minimum gap).

**Extra example:**
```
R2 runs: NDG batch (b1) then PSC batch (b2).
  s_b1 = 5, e_b1 = 20.
  → s_b2 ≥ 20 + 5 = 25  (need 5 extra minutes for setup)

R2 runs: PSC batch (b3) then PSC batch (b4).
  s_b3 = 25, e_b3 = 40.
  → s_b4 ≥ 40  (same SKU, no setup needed, can start immediately after)
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

$$[s_b, s_b + 14] \cap \mathcal{D}_r = \emptyset \quad \forall b : r_b = r$$

Every minute of the batch must fall outside the downtime window. Batches cannot be paused mid-run.

**Extra example:**
```
D_R3 = {200, ..., 229}.

Start at 183: runs [183, 197]. Ends at 197. ✓ (no overlap with {200,...})
Start at 185: runs [185, 199]. Ends at 199. ✓ (barely safe — ends 1 slot before downtime)
Start at 186: runs [186, 200]. Overlap at slot 200. ✗ VIOLATION.
Start at 230: runs [230, 244]. ✓ (starts after downtime ends)
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

$$s_b \leq 465 \quad \forall b : a_b = 1$$

Any batch starting after slot 465 would not finish within the shift. Enforced by restricting the domain of $s_b$.

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

### Constraint Summary Table

| ID | What it enforces | Hard or Soft? | Penalty if soft |
|----|-----------------|---------------|-----------------|
| C1 | All MTO batches must run | Hard | — |
| C2 | PSC batches are optional | — | — |
| C3 | Roaster must be eligible for SKU | Hard | — |
| C4 | No two batches on same roaster at same time | Hard | — |
| C5 | 5-min setup gap when switching SKUs | Hard | — |
| C6 | First batch respects initial SKU state | Hard | — |
| C7 | Batch must not overlap planned downtime | Hard | — |
| C8 | No two batches on same pipeline simultaneously | Hard | — |
| C9 | All batches complete within shift | Hard | — |
| C10 | RC silo ≥ 0 (no stockout) | Hard (det.) / Soft (reactive) | $1,500/event |
| C11 | RC silo ≤ 40 (no overflow) | Hard | — |
| C12 | Tardiness is correctly computed | Soft | $1,000/min |
| C13 | R3 routing decision is binary | — | — |
| C14 | Safety-idle is detected for penalty | Soft | $200/min/roaster |
| C15 | Overflow-idle is detected for penalty | Soft | $50/min/roaster |

---

# PART IV — OBJECTIVE FUNCTION

## 6. Maximize Profit

### 6.1 Deterministic Mode

$$\text{Maximize:} \quad \underbrace{\sum_{b: a_b=1} R_{\text{sku}(b)}}_{\text{Revenue}} \;-\; \underbrace{c^{tard} \sum_j \text{tard}_j}_{\text{Tardiness penalty}} \;-\; \underbrace{c^{idle} \sum_{r,t} \text{idle}_{r,t}}_{\text{Safety-idle penalty}} \;-\; \underbrace{c^{over} \sum_{r,t} \text{over}_{r,t}}_{\text{Overflow-idle penalty}}$$

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

  Safety-idle: R1 idle 10 min + R2 idle 5 min while B_L1 < 20
  Safety-idle cost = 15 min × $200 = $3,000

  Overflow-idle: 0 minutes (RC never hit 40 batches)
  Overflow-idle cost = $0

NET PROFIT = $276,000 − $3,000 − $3,000 − $0 = $270,000
```

**Tradeoff example — should we run more PSC batches even if it causes more tardiness?**
```
Option A: 62 PSC batches, 3 min tardiness
  Profit = $276,000 − $3,000 − $3,000 = $270,000

Option B: 65 PSC batches (3 more), but NDG finishes 10 min late
  Revenue = 65×$4k + 3×$7k + 1×$7k = $292,000
  Tardiness = 10 × $1,000 = $10,000
  Safety-idle = 0 (RC is well-stocked now)
  Profit = $292,000 − $10,000 = $282,000 → BETTER

Option C: 60 PSC batches, 0 tardiness, 80 min safety-idle
  Revenue = 60×$4k + $28k = $268,000
  Tardiness = $0, Safety-idle = 80×$200 = $16,000
  Profit = $268,000 − $16,000 = $252,000 → WORSE

→ Option B wins: the 3 extra PSC batches ($12k revenue) outweigh the 
  extra tardiness ($10k penalty) — but only just. This is exactly the 
  kind of tradeoff the objective function is designed to balance.
```

---

### 6.2 Reactive Mode (Post-UPS Re-solve)

After an unplanned stoppage (UPS) at time $t_0$, the solver re-optimizes only the **remaining** shift $[t_0, 479]$:

$$\text{Maximize:} \sum_{b \in \mathcal{B}_{rem}} R_{\text{sku}(b)} \cdot a_b \;-\; c^{tard}\sum_j \text{tard}_j \;-\; c^{stock}\sum_l \text{SO}_l \;-\; c^{idle}\sum_{r,t \geq t_0} \text{idle}_{r,t} \;-\; c^{over}\sum_{r,t \geq t_0} \text{over}_{r,t}$$

Key changes from deterministic mode:
1. **Stockout term added** ($c^{stock} \cdot \text{SO}_l$) — because disruptions may make it impossible to prevent all stockouts.
2. **$\mathcal{B}_{rem}$** — the batch pool is rebuilt for the remaining shift. Completed batches are removed; the cancelled batch (UPS victim) is removed; in-progress batches become fixed constraints.
3. Revenue is only counted for batches completed **after** $t_0$ and still to be scheduled. Already-completed batches don't re-enter the objective (their revenue is "locked in").

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
║                    MODEL SUMMARY                                   ║
╠════════════════════════════════════════════════════════════════════╣
║ FACTORY:     5 roasters (R1-R2 on L1, R3-R5 on L2), 2 lines      ║
║ TIME:        480 min slots, s_b ∈ [0, 465]                        ║
║ ROASTING:    15 min fixed. Pipeline pull: 3 min concurrent.       ║
║              Setup (SKU change): 5 min.                           ║
║ RC BUFFER:   Max 40 batches/line. Safety threshold: 20 batches.   ║
║ PRODUCTS:    PSC → fills RC buffer. NDG/Busta → direct delivery.  ║
║ OBJECTIVE:   MAX PROFIT = Revenue − Tardiness − Idle penalties    ║
║ REVENUE:     PSC $4k/batch, NDG $7k/batch, Busta $7k/batch       ║
║ PENALTIES:   Stockout $1.5k/event > Tardiness $1k/min >          ║
║              Safety-idle $200/min > Overflow-idle $50/min         ║
║ CONSTRAINTS: 15 groups (C1–C15)                                   ║
║ VARIABLES:   ~164 start times, ~164 roaster assignments,         ║
║              160 activation binaries, ≤32 R3 routing binaries    ║
╚════════════════════════════════════════════════════════════════════╝
```
