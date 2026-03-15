# Mathematical Formulation: Multi-Machine Batch Roasting Scheduling with GC Silo Constraints

## Nestlé Trị An Factory — 8-Hour Shift Scheduling Problem

### Version 2.0 — Revised with Highest-Stock RC Consumption Rule

---

## 1. Sets and Indices

### 1.1 Time and Structure

| Symbol | Definition |
|--------|-----------|
| $\mathcal{T} = \{0, 1, \dots, 479\}$ | Set of time slots (1 slot = 1 minute, 480 slots = 8 hours) |
| $\mathcal{L} = \{L_1, L_2\}$ | Set of production lines |
| $\mathcal{R} = \{R_1, R_2, R_3, R_4, R_5\}$ | Set of roasters |
| $\mathcal{R}_l \subseteq \mathcal{R}$ | Roasters belonging to line $l$: $\mathcal{R}_{L_1} = \{R_1, R_2\}$, $\mathcal{R}_{L_2} = \{R_3, R_4, R_5\}$ |
| $\ell(r) \in \mathcal{L}$ | Line that roaster $r$ draws GC from: $\ell(R_1)=\ell(R_2)=L_1$; $\ell(R_3)=\ell(R_4)=\ell(R_5)=L_2$ |

### 1.2 Products and SKUs

| Symbol | Definition |
|--------|-----------|
| $\mathcal{K}^{RC}$ | Set of all RC-SKUs (roasted coffee SKU identifiers) |
| $\mathcal{K}^{GC}$ | Set of all GC-SKUs (green coffee SKU identifiers) |
| $\mathcal{K}^{RC}_r \subseteq \mathcal{K}^{RC}$ | RC-SKUs compatible with roaster $r$ |

### 1.3 Silos

| Symbol | Definition |
|--------|-----------|
| $\mathcal{G}_l = \{1, \dots, 8\}$ | GC silos of line $l$ (8 per line, 16 total) |
| $\mathcal{S}_l = \{1, 2, 3, 4\}$ | RC silos of line $l$ (4 per line, 8 total). Silo 4 is the **buffer silo** with priority in PSC consumption |
| $\mathcal{S}^{main}_l = \{1, 2, 3\}$ | Main RC silos (non-buffer) |

### 1.4 Jobs and Batches

| Symbol | Definition |
|--------|-----------|
| $\mathcal{J}^{MTO}$ | Set of make-to-order jobs (NDG and Busta tickets), known at shift start |
| $\mathcal{J}^{MTS}$ | Set of "virtual" PSC job slots (make-to-stock) |
| $\mathcal{J} = \mathcal{J}^{MTO} \cup \mathcal{J}^{MTS}$ | All jobs |
| $\mathcal{B}_j$ | Set of batches belonging to job $j$ |
| $\mathcal{B} = \bigcup_{j \in \mathcal{J}} \mathcal{B}_j$ | Set of all batches |
| $\mathcal{B}^{MTO}, \mathcal{B}^{MTS}$ | MTO and MTS batch subsets respectively |

**PSC Batch Pool Pre-generation:**

The number of PSC batches is not known in advance — it is itself a decision variable. To handle this in a fixed-variable formulation, we pre-generate a pool of *potential* PSC batches, each with an activation binary $a_b$ that the solver switches on or off.

**Pool size computation:** For each compatible (roaster $r$, RC-SKU $k$) pair, the theoretical maximum number of batches that could fit in the horizon is:

$$N^{max}_{r,k} = \left\lfloor \frac{480}{\text{proc}_k} \right\rfloor \quad \forall r \in \mathcal{R},\; k \in \mathcal{K}^{RC}_r$$

**Worked example:** Suppose PSC-A has $\text{proc} = 15$ min and is compatible with $R_3, R_4, R_5$:

- Pool for $(R_3, \text{PSC-A})$: $\lfloor 480/15 \rfloor = 32$ potential batches → $b^{R3}_1, \dots, b^{R3}_{32}$
- Pool for $(R_4, \text{PSC-A})$: 32 potential batches → $b^{R4}_1, \dots, b^{R4}_{32}$
- Pool for $(R_5, \text{PSC-A})$: 32 potential batches → $b^{R5}_1, \dots, b^{R5}_{32}$

This creates 96 potential PSC-A batches. If the optimal solution needs only 50, the solver sets $a_b = 1$ for 50 batches and $a_b = 0$ for the remaining 46. The deactivated batches contribute nothing — no variables, no constraints bind.

The upper bound deliberately overestimates (ignores setup time, pipeline waits, downtime). This guarantees the pool never artificially restricts the solution space. The cost is extra binary variables that preprocessing typically fixes to zero.

### 1.5 Pipeline Operation Types

| Symbol | Definition |
|--------|-----------|
| $\mathcal{O} = \{\texttt{consume}, \texttt{replenish}, \texttt{dump}\}$ | Operation types on the shared GC pipeline |

---

## 2. Parameters

### 2.1 Roasting Parameters

| Symbol | Definition | Unit |
|--------|-----------|------|
| $\text{proc}_k$ | Processing (roasting) time for RC-SKU $k$ | minutes (integer, range 13–16) |
| $\text{bs}_k$ | Batch size (RC output) for RC-SKU $k$ | kg |
| $\text{sku}(b)$ | RC-SKU of batch $b$ | — |
| $\text{job}(b)$ | Job that batch $b$ belongs to | — |
| $\sigma$ | Sequence-dependent setup time (uniform for all SKU transitions) | 5 minutes |
| $D^{MTO}$ | Due time for all MTO jobs | 240 (minute 240 = end of hour 4) |

### 2.2 BOM (Bill of Materials)

| Symbol | Definition | Unit |
|--------|-----------|------|
| $\text{bom}_{k,g}$ | Amount of GC-SKU $g$ required per batch of RC-SKU $k$ | kg |
| $\mathcal{K}^{GC}_k = \{g \in \mathcal{K}^{GC} : \text{bom}_{k,g} > 0\}$ | GC-SKUs needed by RC-SKU $k$ (1 to 3 GC-SKUs) | — |
| $W_k = \sum_{g \in \mathcal{K}^{GC}} \text{bom}_{k,g}$ | Total GC weight consumed per batch of SKU $k$ | kg |

### 2.3 GC Silo Parameters

| Symbol | Value | Definition |
|--------|-------|-----------|
| $\overline{C}^{GC}$ | 3,000 kg | Capacity of each GC silo |
| $Q^{rep}$ | 500 kg | Fixed lot size per replenish event |
| $\delta^{con}$ | 2 | Duration of a consume operation (time slots) |
| $\delta^{rep}$ | 2 | Duration of a replenish operation (time slots) |
| $\delta^{dump}(q)$ | $5 + \lceil q/100 \rceil$ | Duration of dumping $q$ kg (time slots) |
| $I^{GC,0}_{l,s}$ | given | Initial GC level of silo $s$ on line $l$ (kg) |
| $\kappa^{GC,0}_{l,s}$ | given | Initial GC-SKU stored in silo $s$ on line $l$ (or $\emptyset$ if empty) |

### 2.4 RC Silo Parameters

| Symbol | Value | Definition |
|--------|-------|-----------|
| $\overline{C}^{RC}$ | 5,000 kg | Capacity of each RC silo |
| $\overline{C}^{RC}_{tot}$ | 20,000 kg | Total RC capacity per line |
| $\theta^{SS}$ | 10,000 kg | Safety stock threshold per line for idle penalty |
| $\theta^{SW}$ | 15,000 kg | Maximum RC level of old SKU at SKU switch / changeover |
| $I^{RC,0}_{l,s}$ | given | Initial RC level of silo $s$ on line $l$ |
| $\kappa^{RC,0}_{l,s}$ | given | Initial RC-SKU in RC silo $s$ on line $l$ |

### 2.5 PSC Consumption Parameters

| Symbol | Definition | Unit |
|--------|-----------|------|
| $\rho_k$ | Consumption rate of RC-SKU $k$ by PSC line | kg per 10 minutes |
| $\mathcal{T}^{PSC} = \{10, 20, 30, \dots, 480\}$ | Consumption event time points (every 10 min), 48 events per shift | — |
| $\text{run}^{RC}_{l,t}$ | The RC-SKU currently consumed by PSC on line $l$ at time $t$ | input schedule |

### 2.6 Planned Downtime & Changeover

| Symbol | Definition |
|--------|-----------|
| $\mathcal{D}_r \subseteq \mathcal{T}$ | Time slots where roaster $r$ is in planned downtime |
| $\text{CO}_l$ | Changeover event for line $l$: tuple $(\text{start}, \text{end}, k^{old}, k^{new})$ or $\emptyset$ |
| $\mathcal{T}^{CO}_l \subseteq \mathcal{T}$ | Time slots during changeover of line $l$ |

### 2.7 Penalty Costs

| Symbol | Value | Definition |
|--------|-------|-----------|
| $c^{short}$ | 600,000 VND/min | PSC shortage penalty per minute of stockout |
| $c^{idle}$ | 100,000 VND/min | Roaster idle penalty per minute (when RC < safety stock) |
| $c^{dump}_{fix}$ | 500,000 VND | Fixed cost per dump event |
| $c^{dump}_{var}$ | 10,000 VND/kg | Variable cost per kg dumped |
| $c^{rep}$ | 100,000 VND | Fixed cost per replenish event |
| $c^{tard}$ | $M^{tard}$ VND/min | NDG/Busta tardiness penalty per minute late (calibrated very large) |

---

## 3. Decision Variables

### 3.1 Batch Scheduling Variables

| Variable | Domain | Meaning |
|----------|--------|---------|
| $x_{b,r,t} \in \{0,1\}$ | $b \in \mathcal{B},\; r \in \mathcal{R},\; t \in \mathcal{T}$ | 1 if batch $b$ starts on roaster $r$ at time $t$ |
| $a_b \in \{0,1\}$ | $b \in \mathcal{B}^{MTS}$ | 1 if PSC potential batch $b$ is activated (actually scheduled) |
| $y_b \in \{0,1\}$ | $b \in \mathcal{B} : \text{assigned to } R_3$ | 1 if R3 batch $b$ sends RC output to Line 1; 0 → Line 2 |

**Derived (auxiliary) scheduling variables:**

| Variable | Definition |
|----------|-----------|
| $\text{start}_b = \sum_{r,t} t \cdot x_{b,r,t}$ | Start time of batch $b$ |
| $\text{end}_b = \text{start}_b + \text{proc}_{\text{sku}(b)}$ | End time of batch $b$ |
| $z^r_b = \sum_{t} x_{b,r,t}$ | 1 if batch $b$ is assigned to roaster $r$ |
| $\text{tard}_j = \max(0,\; \max_{b \in \mathcal{B}_j} \text{end}_b - D^{MTO})$ | Tardiness of MTO job $j$ (minutes) |

### 3.2 GC Silo Operation Variables

| Variable | Domain | Meaning |
|----------|--------|---------|
| $u^{rep}_{l,s,g,t} \in \{0,1\}$ | $l \in \mathcal{L},\; s \in \mathcal{G}_l,\; g \in \mathcal{K}^{GC},\; t \in \mathcal{T}$ | 1 if replenish of GC-SKU $g$ into silo $s$ of line $l$ starts at $t$ |
| $u^{dump}_{l,s,t} \in \{0,1\}$ | $l \in \mathcal{L},\; s \in \mathcal{G}_l,\; t \in \mathcal{T}$ | 1 if dump of silo $s$ on line $l$ starts at $t$ |
| $q^{dump}_{l,s,t} \geq 0$ | continuous | Quantity dumped from silo $(l,s)$ at time $t$ (kg) |
| $n^{dump}_{l,s,t} \in \mathbb{Z}_{\geq 0}$ | integer, range $[0, 30]$ | $\lceil q^{dump}_{l,s,t} / 100 \rceil$, used for dump duration linearization |
| $v^{con}_{b,l,s,g} \geq 0$ | continuous | Amount of GC-SKU $g$ that batch $b$ draws from silo $(l,s)$ |
| $I^{GC}_{l,s,t} \geq 0$ | continuous | GC inventory level of silo $(l,s)$ at end of time slot $t$ |
| $\phi^{GC}_{l,s,g,t} \in \{0,1\}$ | binary | 1 if silo $(l,s)$ is assigned to GC-SKU $g$ at time $t$ |

### 3.3 RC Silo Variables

| Variable | Domain | Meaning |
|----------|--------|---------|
| $I^{RC}_{l,s,t} \geq 0$ | continuous | RC inventory level of RC silo $(l,s)$ at end of time slot $t$ |
| $\phi^{RC}_{l,s,k,t} \in \{0,1\}$ | binary | 1 if RC silo $(l,s)$ holds RC-SKU $k$ at time $t$ |
| $f_{b,l,s} \in \{0,1\}$ | binary | 1 if batch $b$'s RC output fills into RC silo $(l,s)$ |
| $d_{l,s,t} \in \{0,1\}$ | $s \in \mathcal{S}_l,\; t \in \mathcal{T}^{PSC}$ | 1 if silo $s$ is selected for PSC draw on line $l$ at consumption event $t$ |

### 3.4 Penalty Tracking Variables

| Variable | Domain | Meaning |
|----------|--------|---------|
| $\text{short}_{l,t} \geq 0$ | continuous | PSC shortage amount (kg) on line $l$ at consumption event $t$ |
| $\text{idle}_{r,t} \in \{0,1\}$ | binary | 1 if roaster $r$ is idle at time $t$ AND total RC on $\ell(r)$ is below $\theta^{SS}$ |
| $\text{tard}_j \geq 0$ | continuous | Tardiness of MTO job $j$ (minutes late) |

### 3.5 Pipeline Occupation Variables

| Variable | Domain | Meaning |
|----------|--------|---------|
| $p^{con}_{l,t} \in \{0,1\}$ | $l \in \mathcal{L},\; t \in \mathcal{T}$ | 1 if pipeline of line $l$ is performing a consume at slot $t$ |
| $p^{rep}_{l,t} \in \{0,1\}$ | $l \in \mathcal{L},\; t \in \mathcal{T}$ | 1 if pipeline of line $l$ is performing a replenish at slot $t$ |
| $p^{dump}_{l,t} \in \{0,1\}$ | $l \in \mathcal{L},\; t \in \mathcal{T}$ | 1 if pipeline of line $l$ is performing a dump at slot $t$ |

---

## 4. Objective Function

Minimize total penalty cost over the 8-hour shift:

$$
\min \; Z = Z^{short} + Z^{idle} + Z^{dump} + Z^{rep} + Z^{tard}
$$

**PSC Shortage Penalty** — each consumption event where shortage occurs incurs 10 minutes of penalty (the full cycle length):

$$
Z^{short} = c^{short} \times 10 \times \sum_{l \in \mathcal{L}} \sum_{t \in \mathcal{T}^{PSC} \setminus \mathcal{T}^{CO}_l} \mathbb{1}[\text{short}_{l,t} > 0]
$$

> Linearization of the indicator: introduce binary $h_{l,t} \in \{0,1\}$ with $\text{short}_{l,t} \leq M \cdot h_{l,t}$ and $\text{short}_{l,t} \geq \epsilon \cdot h_{l,t}$ (where $\epsilon$ is a small positive constant and $M$ is an upper bound on shortage). Then replace $\mathbb{1}[\cdot]$ with $h_{l,t}$.

**Roaster Idle Penalty** — per minute, excluding downtime and changeover periods:

$$
Z^{idle} = c^{idle} \times \sum_{r \in \mathcal{R}} \sum_{t \in \mathcal{T} \setminus (\mathcal{D}_r \cup \mathcal{T}^{CO}_{\ell(r)})} \text{idle}_{r,t}
$$

**Dump Penalty** — fixed cost per event plus variable cost per kg:

$$
Z^{dump} = \sum_{l \in \mathcal{L}} \sum_{s \in \mathcal{G}_l} \sum_{t \in \mathcal{T}} \left( c^{dump}_{fix} \cdot u^{dump}_{l,s,t} + c^{dump}_{var} \cdot q^{dump}_{l,s,t} \right)
$$

**Replenish Penalty** — fixed cost per event:

$$
Z^{rep} = c^{rep} \times \sum_{l \in \mathcal{L}} \sum_{s \in \mathcal{G}_l} \sum_{g \in \mathcal{K}^{GC}} \sum_{t \in \mathcal{T}} u^{rep}_{l,s,g,t}
$$

**Tardiness Penalty** — per minute late, calibrated very large:

$$
Z^{tard} = c^{tard} \times \sum_{j \in \mathcal{J}^{MTO}} \text{tard}_j
$$

---

## 5. Constraints

### 5.1 Batch Assignment Constraints

**(C1) Each MTO batch is assigned exactly once:**

$$
\sum_{r \in \mathcal{R}} \sum_{t \in \mathcal{T}} x_{b,r,t} = 1 \quad \forall b \in \mathcal{B}^{MTO}
$$

> Every NDG/Busta batch must be scheduled on exactly one roaster at exactly one start time. Non-negotiable.

**(C2) Each MTS (PSC) batch is assigned at most once, linked to activation:**

$$
\sum_{r \in \mathcal{R}} \sum_{t \in \mathcal{T}} x_{b,r,t} = a_b \quad \forall b \in \mathcal{B}^{MTS}
$$

> If the solver decides to activate a PSC batch ($a_b = 1$), it must be assigned to exactly one (roaster, time) pair. If $a_b = 0$, the batch does not exist in the schedule.

**(C3) Roaster compatibility:**

$$
x_{b,r,t} = 0 \quad \forall b \in \mathcal{B},\; r \in \mathcal{R} : \text{sku}(b) \notin \mathcal{K}^{RC}_r,\; \forall t
$$

> Enforces: R1 can roast NDG + PSC. R2 can roast NDG + Busta + PSC. R3, R4, R5 can only roast PSC.

**(C4) Batch must complete within the time horizon:**

$$
x_{b,r,t} = 0 \quad \forall b, r, t : t + \text{proc}_{\text{sku}(b)} > 480
$$

> A batch starting at minute $t$ must finish by minute 480. For PSC-A with $\text{proc} = 15$, the latest possible start is $t = 465$.

**(C5) No overlap on the same roaster:**

$$
\sum_{b \in \mathcal{B}} \sum_{\tau = \max(0,\, t - \text{proc}_{\text{sku}(b)} + 1)}^{t} x_{b,r,\tau} \leq 1 \quad \forall r \in \mathcal{R},\; t \in \mathcal{T}
$$

> At any minute $t$, at most one batch can be "in progress" on roaster $r$. This is the standard time-indexed disjunctive constraint. It works by checking: for every batch $b$ that *could* be processing at time $t$ (i.e., started at some $\tau$ where $\tau \leq t < \tau + \text{proc}$), at most one such assignment exists.

**(C6) Sequence-dependent setup time:**

For every ordered pair of batches $(b_1, b_2)$ where $\text{sku}(b_1) \neq \text{sku}(b_2)$:

$$
\text{start}_{b_2} \geq \text{end}_{b_1} + \sigma - M \cdot (2 - z^r_{b_1} - z^r_{b_2}) \quad \forall r \in \mathcal{R}
$$

where:
- $z^r_b = \sum_t x_{b,r,t}$ indicates batch $b$ is on roaster $r$
- $\sigma = 5$ minutes
- $M$ is a sufficiently large constant (e.g., 480)

> **In plain language:** If $b_1$ and $b_2$ are both assigned to the same roaster $r$ (making $z^r_{b_1} + z^r_{b_2} = 2$, so the big-M term vanishes), and they have different SKUs, then $b_2$ cannot start until $b_1$ ends plus 5 minutes of setup. If same SKU ($\text{sku}(b_1) = \text{sku}(b_2)$), this constraint is not generated — batch $b_2$ can start immediately after $b_1$.

> **Implementation note:** For the time-indexed formulation, an equivalent and computationally tighter approach is to extend the "shadow" of each batch in C5 by $\sigma$ slots when a different-SKU batch follows. This avoids the $O(|\mathcal{B}|^2)$ pairwise constraints. The formulation above is stated for conceptual clarity.

**(C7) Planned downtime — no batch active during downtime:**

$$
x_{b,r,t} = 0 \quad \forall r \in \mathcal{R},\; t \in \mathcal{D}_r,\; \forall b
$$

> No batch may start during planned downtime. Additionally, no batch may be *in progress* during downtime:

$$
x_{b,r,\tau} = 0 \quad \forall r, b, \tau : [\tau, \tau + \text{proc}_{\text{sku}(b)} - 1] \cap \mathcal{D}_r \neq \emptyset
$$

> A batch that would still be roasting when downtime begins is also forbidden.

**(C8) Changeover — no new batch targeting changeover line:**

$$
x_{b,r,t} = 0 \quad \forall r \in \mathcal{R}_l,\; t \in \mathcal{T}^{CO}_l,\; b \in \mathcal{B}^{MTS} : \text{out\_line}(b) = l
$$

> During changeover on line $l$, no PSC batch whose RC output targets line $l$ may start on any roaster of that line.

---

### 5.2 MTO Job Constraints

**(C9) Tardiness computation:**

$$
\text{tard}_j \geq \text{end}_b - D^{MTO} \quad \forall j \in \mathcal{J}^{MTO},\; b \in \mathcal{B}_j
$$

$$
\text{tard}_j \geq 0 \quad \forall j \in \mathcal{J}^{MTO}
$$

> Tardiness of job $j$ is the maximum over all its batches of (end time − due date), floored at zero. Since the objective minimizes $\text{tard}_j$, the solver will naturally set it to the tightest (largest) bound, capturing the $\max$.

---

### 5.3 R3 Output Direction

**(C10) R3 output line assignment:**

$$
y_b \in \{0,1\} \quad \forall b \in \mathcal{B} : z^{R_3}_b = 1
$$

> $y_b = 1$: RC output of R3 batch $b$ goes to Line 1. $y_b = 0$: output goes to Line 2.

Fixed assignments for other roasters (not decision variables):
- $R_1, R_2$ batches: output always → Line 1
- $R_4, R_5$ batches: output always → Line 2

Define the **RC output line** function:

$$
\text{out\_line}(b) = \begin{cases} L_1 & \text{if } z^{R_1}_b = 1 \text{ or } z^{R_2}_b = 1 \\ L_1 & \text{if } z^{R_3}_b = 1 \text{ and } y_b = 1 \\ L_2 & \text{otherwise} \end{cases}
$$

---

### 5.4 GC Pipeline Mutual Exclusion Constraints

These encode the physical bottleneck: each line has one shared pipeline serving all 8 GC silos, and only one operation (consume, replenish, or dump) can use it at any time slot.

**(C11) At most one pipeline operation per line per time slot:**

$$
p^{con}_{l,t} + p^{rep}_{l,t} + p^{dump}_{l,t} \leq 1 \quad \forall l \in \mathcal{L},\; t \in \mathcal{T}
$$

**(C12) Consume occupies pipeline for 2 consecutive slots:**

When a batch starts on a roaster of line $l$ at time $t$, the pipeline of line $l$ is occupied for slots $t$ and $t+1$:

$$
p^{con}_{l,t'} \geq x_{b,r,t} \quad \forall b \in \mathcal{B},\; r \in \mathcal{R}_{L_1},\; t' \in \{t, t+1\}
$$

> **Critical detail — R3 always consumes from Line 2's pipeline**, regardless of where RC output goes:

$$
p^{con}_{L_2, t'} \geq x_{b, R_3, t} \quad \forall b \in \mathcal{B},\; t' \in \{t, t+1\}
$$

> This means: even if R3 sends RC output to Line 1, it still occupies Line 2's GC pipeline during consume.

**(C13) Replenish occupies pipeline for 2 consecutive slots:**

$$
p^{rep}_{l,t'} \geq u^{rep}_{l,s,g,t} \quad \forall l, s \in \mathcal{G}_l, g \in \mathcal{K}^{GC}, t,\; t' \in \{t, t+1\}
$$

**(C14) Dump occupies pipeline for variable duration:**

The dump duration is $\delta^{dump}(q) = 5 + \lceil q/100 \rceil$ slots, where $q = q^{dump}_{l,s,t}$ is a decision variable. This creates a variable-length pipeline occupation.

**Linearization using integer variable $n^{dump}_{l,s,t}$:**

The integer variable $n^{dump}_{l,s,t}$ represents $\lceil q^{dump}_{l,s,t} / 100 \rceil$:

$$
100 \cdot (n^{dump}_{l,s,t} - 1) < q^{dump}_{l,s,t} \leq 100 \cdot n^{dump}_{l,s,t} \quad \forall l, s, t : u^{dump}_{l,s,t} = 1
$$

Rewritten as linear constraints:

$$
q^{dump}_{l,s,t} \leq 100 \cdot n^{dump}_{l,s,t} \quad \forall l, s, t
$$

$$
q^{dump}_{l,s,t} \geq 100 \cdot (n^{dump}_{l,s,t} - 1) + \epsilon \cdot u^{dump}_{l,s,t} \quad \forall l, s, t
$$

$$
n^{dump}_{l,s,t} \leq 30 \cdot u^{dump}_{l,s,t} \quad \forall l, s, t
$$

> The last constraint ensures $n^{dump} = 0$ when no dump occurs.

**Pipeline occupation for the fixed 5-slot portion:**

$$
p^{dump}_{l,t'} \geq u^{dump}_{l,s,t} \quad \forall t' \in \{t, t+1, t+2, t+3, t+4\}
$$

**Pipeline occupation for the variable portion** (using auxiliary binaries $\alpha_{l,s,t,i} \in \{0,1\}$, where $\alpha_{l,s,t,i} = 1$ iff $n^{dump}_{l,s,t} \geq i + 1$):

$$
n^{dump}_{l,s,t} \geq (i+1) \cdot \alpha_{l,s,t,i} \quad \forall i = 0, \dots, 29
$$

$$
n^{dump}_{l,s,t} \leq i + 30 \cdot (1 - \alpha_{l,s,t,i}) \quad \forall i = 0, \dots, 29
$$

$$
p^{dump}_{l,t+5+i} \geq \alpha_{l,s,t,i} \quad \forall i = 0, \dots, 29 : t+5+i \leq 479
$$

> **Computational cost note:** The $\alpha$ binaries add $30 \times |\text{dump events}|$ variables. Since dumps are rare (estimated 2–5 per line per shift) and the $n^{dump}$ domain is small (0–30), modern MIP solvers handle this efficiently. The batch scheduling variables $x_{b,r,t}$ remain the dominant source of computational cost by a large margin.

> **Practical simplification:** Dump events can be restricted to start only at every-5th-minute slots (i.e., $t \in \{0, 5, 10, \dots, 475\}$), reducing the dump variable space by 5× with negligible impact on solution quality. This is a standard time-bucketing technique.

**(C15) At most one batch starts consuming per line within any 2-slot window:**

$$
\sum_{b \in \mathcal{B}} \sum_{r \in \mathcal{R}_l} x_{b,r,t} \leq 1 \quad \forall l \in \mathcal{L},\; t \in \mathcal{T}
$$

$$
\sum_{b \in \mathcal{B}} \sum_{r \in \mathcal{R}_l} x_{b,r,t-1} + \sum_{b \in \mathcal{B}} \sum_{r \in \mathcal{R}_l} x_{b,r,t} \leq 1 \quad \forall l \in \mathcal{L},\; t \in \mathcal{T}, t \geq 1
$$

> Since each consume takes 2 slots, two batches starting at $t$ and $t+1$ on the same line would overlap in pipeline usage. The second constraint prevents this.

---

### 5.5 GC Silo Inventory Balance Constraints

**(C16) GC silo inventory balance:**

$$
I^{GC}_{l,s,t} = I^{GC}_{l,s,t-1} + \underbrace{Q^{rep} \cdot \sum_{g \in \mathcal{K}^{GC}} u^{rep}_{l,s,g,t}}_{\text{replenish in}} - \underbrace{\sum_{b \in \mathcal{B}} \sum_{g \in \mathcal{K}^{GC}} v^{con}_{b,l,s,g} \cdot \delta^{start}_{b,t}}_{\text{consume out}} - \underbrace{q^{dump}_{l,s,t}}_{\text{dump out}} \quad \forall l, s, t
$$

where $\delta^{start}_{b,t} = \sum_{r \in \mathcal{R}} x_{b,r,t}$ equals 1 if batch $b$ starts at time $t$, 0 otherwise.

> **Timing conventions (Assumptions A6):**
> - **Replenish:** 500 kg credited to silo at time slot $t$ (start of operation), even though pipeline is busy for 2 slots.
> - **Dump:** Entire dump quantity $q^{dump}$ deducted at time slot $t$ (start of operation), even though pipeline is busy for $5 + \lceil q/100 \rceil$ slots.
> - **Consume:** All GC drawn at time slot $t$ (start of batch), even though pipeline is busy for 2 slots.
>
> These conventions simplify inventory tracking. The pipeline occupation constraints (C11–C15) separately enforce the correct duration of each operation.

**(C17) GC silo capacity:**

$$
0 \leq I^{GC}_{l,s,t} \leq \overline{C}^{GC} \quad \forall l \in \mathcal{L},\; s \in \mathcal{G}_l,\; t \in \mathcal{T}
$$

**(C18) GC consumed matches BOM exactly:**

$$
\sum_{s \in \mathcal{G}_l} v^{con}_{b,l,s,g} = \text{bom}_{\text{sku}(b), g} \cdot \left(\sum_{r \in \mathcal{R}} \sum_{t \in \mathcal{T}} x_{b,r,t}\right) \quad \forall b \in \mathcal{B},\; g \in \mathcal{K}^{GC}_{\text{sku}(b)},\; l = \ell(\text{roast}_b)
$$

> For MTO batches, the right-hand side equals $\text{bom}_{\text{sku}(b),g} \cdot 1 = \text{bom}_{\text{sku}(b),g}$. For MTS batches, it equals $\text{bom}_{\text{sku}(b),g} \cdot a_b$, which is zero when the batch is not activated — cleanly zeroing out all GC consumption for phantom batches.

**(C19) Can only draw from a silo assigned to the correct GC-SKU:**

$$
v^{con}_{b,l,s,g} \leq \overline{C}^{GC} \cdot \phi^{GC}_{l,s,g,\text{start}_b} \quad \forall b, l, s, g
$$

> You cannot source GC-SKU $g$ from silo $s$ unless silo $s$ contains GC-SKU $g$ at the time of consumption.

**(C20) Cannot draw more than physically available:**

$$
v^{con}_{b,l,s,g} \leq I^{GC}_{l,s,\text{start}_b - 1} \quad \forall b, l, s, g
$$

> The amount drawn from a silo cannot exceed the silo's inventory at the moment before consumption.

**(C21) Sourcing from at most 2 silos per GC-SKU, with emptying condition:**

Introduce $w_{b,s,g} \in \{0,1\}$ = 1 if batch $b$ draws any amount of GC-SKU $g$ from silo $s$:

$$
v^{con}_{b,l,s,g} \leq \overline{C}^{GC} \cdot w_{b,s,g} \quad \forall b, l, s, g
$$

**At most 2 silos per GC-SKU per batch:**

$$
\sum_{s \in \mathcal{G}_l} w_{b,s,g} \leq 2 \quad \forall b \in \mathcal{B},\; g \in \mathcal{K}^{GC}
$$

**If 2 silos are used, at least one must be fully emptied:**

Introduce $e_{b,s,g} \in \{0,1\}$ = 1 if silo $s$ is emptied by batch $b$'s draw of GC-SKU $g$:

$$
I^{GC}_{l,s,\text{start}_b - 1} - v^{con}_{b,l,s,g} \leq \overline{C}^{GC} \cdot (1 - e_{b,s,g}) \quad \forall b, l, s, g
$$

$$
\sum_{s \in \mathcal{G}_l} w_{b,s,g} - 1 \leq \sum_{s \in \mathcal{G}_l} e_{b,s,g} \quad \forall b \in \mathcal{B},\; g \in \mathcal{K}^{GC}
$$

> **In plain language:** If a batch sources GC-1 from 2 silos ($\sum w = 2$), then $\sum e \geq 1$ — at least one silo must be completely emptied. This prevents gratuitous splitting; the only reason to split is to free up a silo for future reassignment.

---

### 5.6 GC Silo SKU Assignment (Single-SKU Rule)

**(C22) Each GC silo holds at most one GC-SKU at any time:**

$$
\sum_{g \in \mathcal{K}^{GC}} \phi^{GC}_{l,s,g,t} \leq 1 \quad \forall l \in \mathcal{L},\; s \in \mathcal{G}_l,\; t \in \mathcal{T}
$$

**(C23) SKU assignment consistency with inventory:**

If a silo has positive inventory, it must be assigned to exactly one SKU. If empty, it is unassigned (free for reassignment):

$$
I^{GC}_{l,s,t} \leq \overline{C}^{GC} \cdot \sum_{g \in \mathcal{K}^{GC}} \phi^{GC}_{l,s,g,t} \quad \forall l, s, t
$$

> This ensures: if $I^{GC} > 0$, then at least one $\phi^{GC}$ must be 1. Combined with C22 (at most one), a non-empty silo has exactly one SKU.

For the reverse direction (empty silo should not have SKU assignment):

$$
\phi^{GC}_{l,s,g,t} \leq \frac{I^{GC}_{l,s,t}}{\epsilon'} \quad \text{(conceptual)}
$$

> **Practical implementation:** This reverse implication is enforced indirectly through the replenish rule (C23b below) rather than through a direct linear constraint, which would require a small epsilon and create numerical issues.

**(C23b) Replenish only into a silo that already holds the same SKU or is empty:**

$$
u^{rep}_{l,s,g,t} \leq \phi^{GC}_{l,s,g,t-1} + \left(1 - \sum_{g' \in \mathcal{K}^{GC}} \phi^{GC}_{l,s,g',t-1}\right) \quad \forall l, s, g, t
$$

> **In plain language:** You can replenish GC-SKU $g$ into silo $s$ only if either (a) silo $s$ already contains GC-SKU $g$, or (b) silo $s$ is currently empty (no SKU assigned). This is the operational enforcement of the no-mixing rule — it is impossible to create a mixed silo through any sequence of decisions.

**(C24) SKU assignment transitions — SKU changes only when silo becomes empty:**

$$
\phi^{GC}_{l,s,g,t-1} = 1 \text{ and } \phi^{GC}_{l,s,g,t} = 0 \implies I^{GC}_{l,s,t-1} = 0 \text{ or } I^{GC}_{l,s,t} = 0
$$

> **Linearization:**

$$
\phi^{GC}_{l,s,g,t-1} - \phi^{GC}_{l,s,g,t} \leq 1 - \frac{I^{GC}_{l,s,t}}{\overline{C}^{GC}} \quad \forall l, s, g, t
$$

> If SKU $g$ is "removed" from silo $s$ ($\phi$ goes from 1 to 0), then the silo must be empty at time $t$.

---

### 5.7 RC Silo Constraints

**(C25) RC silo inventory balance:**

$$
I^{RC}_{l,s,t} = I^{RC}_{l,s,t-1} + \underbrace{\sum_{\substack{b \in \mathcal{B}^{MTS} : \\ \text{end}_b = t,\; \text{out\_line}(b) = l}} \text{bs}_{\text{sku}(b)} \cdot f_{b,l,s}}_{\text{batch output in}} - \underbrace{\text{psc\_draw}_{l,s,t}}_{\text{PSC consumption out}} \quad \forall l, s, t
$$

> Only PSC (MTS) batches fill RC silos. NDG/Busta output is delivered directly and does not enter RC inventory.

**(C26) RC silo capacity:**

$$
0 \leq I^{RC}_{l,s,t} \leq \overline{C}^{RC} \quad \forall l \in \mathcal{L},\; s \in \mathcal{S}_l,\; t \in \mathcal{T}
$$

**(C27) Total RC capacity per line:**

$$
\sum_{s \in \mathcal{S}_l} I^{RC}_{l,s,t} \leq \overline{C}^{RC}_{tot} \quad \forall l \in \mathcal{L},\; t \in \mathcal{T}
$$

**(C28) Each activated PSC batch fills into exactly one RC silo:**

$$
\sum_{s \in \mathcal{S}_l} f_{b,l,s} = a_b \quad \forall b \in \mathcal{B}^{MTS},\; l = \text{out\_line}(b)
$$

$$
f_{b,l,s} = 0 \quad \forall b \in \mathcal{B}^{MTO},\; \forall l, s
$$

> When $a_b = 0$ (batch not activated), no fill assignment occurs. When $a_b = 1$, exactly one RC silo is selected on the output line.

**(C28b) RC silo has room for batch output (look-ahead feasibility):**

$$
I^{RC}_{l,s,\text{end}_b - 1} + \text{bs}_{\text{sku}(b)} \cdot f_{b,l,s} \leq \overline{C}^{RC} \quad \forall b \in \mathcal{B}^{MTS}, l, s
$$

> The silo must have sufficient remaining capacity at the moment the batch finishes roasting. If all silos of the correct SKU are full, the batch cannot start — this is a hard constraint that the scheduler must anticipate.

---

### 5.8 PSC Consumption — Highest-Stock Priority Rule (Revised C29)

**Design decision:** The original FIFO-by-fill-time rule requires tracking *when* each silo was filled — a historical dependency that creates circular coupling between scheduling decisions and consumption order. We replace it with **"Silo 4 first, then highest-stock among silos 1–3"**, which depends only on inventory levels (already modeled as variables). This is operationally equivalent in most scenarios, since the silo filled earliest tends to accumulate the most stock under steady consumption.

> **Assumption to acknowledge:** This rule is a modeling simplification. The actual factory uses silo 4 first, then FIFO-by-fill-time for silos 1–3. In practice, the highest-stock heuristic closely approximates FIFO behavior when consumption is steady, and diverges only when fill patterns are highly irregular within a single consumption cycle. This is stated as a thesis limitation.

At each PSC consumption event $t \in \mathcal{T}^{PSC}$, the PSC line on line $l$ consumes $\rho_k$ kg (where $k = \text{run}^{RC}_{l,t}$) from one silo at a time, draining it completely before moving to the next.

**(C29a) Silo selection — at most one silo is drawn from per consumption event:**

$$
\sum_{s \in \mathcal{S}_l} d_{l,s,t} = 1 \quad \forall l \in \mathcal{L},\; t \in \mathcal{T}^{PSC} \setminus \mathcal{T}^{CO}_l
$$

> Exactly one silo is selected for PSC draw at each consumption event (assuming non-changeover). If the selected silo has less than $\rho_k$, it is fully drained and the remainder becomes shortage (modeled in C30).

> **Note:** This is a simplification — in reality, PSC drains one silo fully then moves to the next within a single 10-minute cycle. Modeling this intra-cycle rollover exactly would require sub-cycle resolution. For the 10-minute granularity, selecting one silo per event and recording any deficit as shortage is a conservative approximation. Alternatively, we can allow drawing from up to 2 silos in a single event (see C29a-extended below).

**(C29a-extended) Allow drawing from up to 2 silos per event (optional refinement):**

$$
\sum_{s \in \mathcal{S}_l} d_{l,s,t} \leq 2 \quad \forall l, t \in \mathcal{T}^{PSC} \setminus \mathcal{T}^{CO}_l
$$

This allows the PSC to finish draining one silo and start on the next within the same 10-minute window, which is what happens physically.

**(C29b) Silo 4 has priority — if silo 4 has stock of the correct SKU, it must be selected:**

$$
d_{l,4,t} \geq \phi^{RC}_{l,4,k,t-1} \cdot \mathbb{1}[I^{RC}_{l,4,t-1} > 0] \quad \forall l, t \in \mathcal{T}^{PSC},\; k = \text{run}^{RC}_{l,t}
$$

> **Linearization:**

$$
d_{l,4,t} \geq \phi^{RC}_{l,4,k,t-1} - M \cdot (1 - \beta_{l,4,t}) \quad \forall l, t
$$

where $\beta_{l,4,t} \in \{0,1\}$ with $I^{RC}_{l,4,t-1} \geq \epsilon \cdot \beta_{l,4,t}$ and $I^{RC}_{l,4,t-1} \leq M \cdot \beta_{l,4,t}$.

> In plain language: if silo 4 holds the current running SKU and has positive stock, it must be included in the draw selection.

**(C29c) Highest-stock priority among silos 1–3 — if silo 4 is empty/wrong SKU, select the silo with highest inventory:**

For each pair of main silos $s_1, s_2 \in \mathcal{S}^{main}_l$ where $s_1 \neq s_2$:

$$
d_{l,s_2,t} \leq d_{l,s_1,t} + (1 - \gamma_{l,s_1,s_2,t}) \quad \forall l, t
$$

where $\gamma_{l,s_1,s_2,t} \in \{0,1\}$ indicates $I^{RC}_{l,s_1,t-1} \geq I^{RC}_{l,s_2,t-1}$:

$$
I^{RC}_{l,s_1,t-1} - I^{RC}_{l,s_2,t-1} \geq -M \cdot (1 - \gamma_{l,s_1,s_2,t}) \quad \forall l, s_1, s_2, t
$$

$$
I^{RC}_{l,s_1,t-1} - I^{RC}_{l,s_2,t-1} \leq M \cdot \gamma_{l,s_1,s_2,t} \quad \forall l, s_1, s_2, t
$$

> **In plain language:** Among silos 1–3, if silo $s_1$ has at least as much stock as silo $s_2$, then $s_2$ cannot be selected unless $s_1$ is also selected. This enforces the "drain highest stock first" priority.

> **Variable count:** For each line and each consumption event: 3 pairwise comparisons × 1 binary each = 3 binaries. Total: $2 \times 48 \times 3 = 288$ additional binaries — negligible.

**(C29d) PSC draw amount from selected silo:**

$$
\text{psc\_draw}_{l,s,t} \leq I^{RC}_{l,s,t-1} \quad \forall l, s, t
$$

$$
\text{psc\_draw}_{l,s,t} \leq M \cdot d_{l,s,t} \quad \forall l, s, t
$$

$$
\sum_{s \in \mathcal{S}_l} \text{psc\_draw}_{l,s,t} + \text{short}_{l,t} = \rho_{\text{run}^{RC}_{l,t}} \quad \forall l, t \in \mathcal{T}^{PSC} \setminus \mathcal{T}^{CO}_l
$$

> The total drawn from all selected silos plus any shortage must equal the consumption demand $\rho_k$.

**(C30) Shortage computation (consolidated):**

$$
\text{short}_{l,t} = \max\left(0,\; \rho_{\text{run}^{RC}_{l,t}} - \sum_{s \in \mathcal{S}_l} \min(I^{RC}_{l,s,t-1} \cdot d_{l,s,t},\; \text{available stock})\right)
$$

> This is already captured by C29d: $\text{short}_{l,t}$ is the slack variable that absorbs any gap between demand and available supply.

$$
\text{short}_{l,t} \geq 0 \quad \forall l, t
$$

$$
\text{short}_{l,t} = 0 \quad \forall l, t \in \mathcal{T}^{CO}_l
$$

---

### 5.9 SKU Switch Constraints

**(C31) RC level threshold before SKU switch:**

Let $t^{sw}_{l}$ be the time at which the first batch of a new RC-SKU $k^{new}$ completes on line $l$, marking the transition from $k^{old}$ to $k^{new}$:

$$
\sum_{s \in \mathcal{S}_l} I^{RC}_{l,s,t^{sw}_l - 1} \cdot \phi^{RC}_{l,s,k^{old},t^{sw}_l - 1} \leq \theta^{SW}
$$

> The total RC of the old SKU across all silos must be $\leq 15{,}000$ kg when the first batch of the new SKU starts filling. This guarantees at least one RC silo (5,000 kg capacity) is available for the new SKU.

> **Linearization note:** The product $I^{RC} \cdot \phi^{RC}$ is bilinear. Introduce auxiliary variable $H_{l,s,t} \geq 0$:

$$
H_{l,s,t} \leq I^{RC}_{l,s,t}
$$

$$
H_{l,s,t} \leq \overline{C}^{RC} \cdot \phi^{RC}_{l,s,k^{old},t}
$$

$$
H_{l,s,t} \geq I^{RC}_{l,s,t} - \overline{C}^{RC} \cdot (1 - \phi^{RC}_{l,s,k^{old},t})
$$

Then: $\sum_s H_{l,s,t^{sw}_l - 1} \leq \theta^{SW}$.

---

### 5.10 Changeover Constraints

**(C32) RC level threshold before changeover:**

$$
\sum_{s \in \mathcal{S}_l} I^{RC}_{l,s, \text{CO}_l.\text{start} - 1} \leq \theta^{SW} \quad \forall l : \text{CO}_l \neq \emptyset
$$

**(C33) No new batch starts during changeover:**

Already covered by C8.

**(C34) PSC consumption of new SKU resumes 2 hours after changeover ends:**

$$
\text{psc\_draw}_{l,s,t} = 0 \quad \forall t \in [\text{CO}_l.\text{end},\; \text{CO}_l.\text{end} + 120),\; \forall s \in \mathcal{S}_l
$$

> The PSC line needs 120 minutes to convert to the new SKU after changeover completion.

**(C35) No shortage or idle penalty during changeover:**

Handled by excluding $\mathcal{T}^{CO}_l$ from the penalty sums in Section 4.

---

### 5.11 Idle Penalty Constraints

**(C36) Roaster busy indicator:**

$$
\text{busy}_{r,t} = \sum_{b \in \mathcal{B}} \sum_{\tau = \max(0, t - \text{proc}_{\text{sku}(b)} + 1)}^{t} x_{b,r,\tau} \quad \forall r, t
$$

> $\text{busy}_{r,t} = 1$ if roaster $r$ is actively processing a batch at time $t$; 0 otherwise.

**(C37) RC below safety stock indicator:**

Introduce $\text{low}_{l,t} \in \{0,1\}$:

$$
\sum_{s \in \mathcal{S}_l} I^{RC}_{l,s,t} \leq \theta^{SS} + M \cdot (1 - \text{low}_{l,t}) \quad \forall l, t
$$

$$
\sum_{s \in \mathcal{S}_l} I^{RC}_{l,s,t} \geq \theta^{SS} + \epsilon - M \cdot \text{low}_{l,t} \quad \forall l, t
$$

> $\text{low}_{l,t} = 1$ when total RC on line $l$ is strictly below 10,000 kg.

**(C38) Idle penalty activation:**

$$
\text{idle}_{r,t} \geq (1 - \text{busy}_{r,t}) + \text{low}_{\ell(r),t} - 1 \quad \forall r \in \mathcal{R},\; t \notin \mathcal{D}_r \cup \mathcal{T}^{CO}_{\ell(r)}
$$

> $\text{idle}_{r,t} = 1$ requires *both* conditions: roaster is not busy *and* RC is below safety stock. Since the objective minimizes idle penalty, the solver sets $\text{idle}_{r,t} = 0$ whenever it can — this one-sided inequality suffices.

> **Setup time and idle:** During the 5-minute setup after an SKU change, $\text{busy}_{r,t} = 0$. If RC is also below safety stock, idle penalty applies. This correctly penalizes unnecessary SKU transitions when inventory is low.

---

## 6. Model Size and Complexity

### 6.1 Variable Count Estimates

For a typical instance: $|\mathcal{B}| \approx 100$ batches, $|\mathcal{K}^{GC}| \approx 8$, 480 time slots.

| Variable type | Approximate count | Type |
|--------------|------------------|------|
| $x_{b,r,t}$ | $100 \times 5 \times 480 = 240{,}000$ | Binary |
| $a_b$ | $\sim 80$ | Binary |
| $y_b$ | $\sim 30$ | Binary |
| $u^{rep}_{l,s,g,t}$ | $2 \times 8 \times 8 \times 480 = 61{,}440$ | Binary |
| $u^{dump}_{l,s,t}$ | $2 \times 8 \times 480 = 7{,}680$ | Binary |
| $n^{dump}_{l,s,t}$ | $7{,}680$ | Integer |
| $q^{dump}_{l,s,t}$ | $7{,}680$ | Continuous |
| $v^{con}_{b,l,s,g}$ | $100 \times 8 \times 8 = 6{,}400$ | Continuous |
| $I^{GC}_{l,s,t}$ | $2 \times 8 \times 480 = 7{,}680$ | Continuous |
| $\phi^{GC}_{l,s,g,t}$ | $2 \times 8 \times 8 \times 480 = 61{,}440$ | Binary |
| $I^{RC}_{l,s,t}$ | $2 \times 4 \times 480 = 3{,}840$ | Continuous |
| $d_{l,s,t}$ (PSC draw) | $2 \times 4 \times 48 = 384$ | Binary |
| $\gamma$ (pairwise) | $2 \times 3 \times 48 = 288$ | Binary |
| Pipeline vars ($p^{con}, p^{rep}, p^{dump}$) | $3 \times 2 \times 480 = 2{,}880$ | Binary |
| Other (penalty, busy, low, etc.) | $\sim 10{,}000$ | Mixed |

**Total: approximately 400,000–500,000 variables** (majority binary).

### 6.2 Recommended Solution Approach

| Approach | Strengths | Weaknesses |
|----------|-----------|-----------|
| **Pure MIP** (Gurobi/CPLEX) | Globally optimal; strong LP relaxation for time-indexed models; proven theory | Huge model size; dump linearization adds complexity |
| **Constraint Programming** (OR-Tools CP-SAT) | Natural fit for no-overlap, pipeline exclusion, sequence-dependent setup; variable-duration dump handled natively via interval variables; internally uses LNS (a metaheuristic) | Weaker lower bounds; harder to prove optimality for very large instances |
| **Standalone Metaheuristic** (GA, SA, ALNS) | Can find solutions quickly; no solver license needed | No optimality guarantee; feasibility repair for GC silo constraints is extremely hard to design; no quality bound without a separate lower bound |
| **Hybrid CP + Simulation** | CP handles scheduling; simulation evaluates RC dynamics | Implementation complexity; requires custom integration |

**Primary recommendation: OR-Tools CP-SAT.**

Rationale for this thesis:
1. Pipeline mutual exclusion with variable-duration dumps maps directly to **interval variables** with `NoOverlap` constraints — no linearization needed.
2. Sequence-dependent setup is a first-class CP concept.
3. CP-SAT internally employs **Large Neighborhood Search (LNS)** — itself a state-of-the-art metaheuristic — combined with SAT-based constraint propagation. This gives metaheuristic-level speed with exact constraint enforcement.
4. For instances where CP-SAT cannot prove optimality within the time limit, it reports the best solution found and the remaining optimality gap.

**On metaheuristics:** A standalone metaheuristic (e.g., ALNS as per Ropke & Pisinger, 2006) could be included as a **benchmark comparison**. However, it faces a fundamental challenge: designing neighborhood operators that maintain feasibility across the interlocking GC silo constraints (single-SKU, pipeline mutex, BOM matching, split-sourcing with emptying condition) is a research problem in itself. CP-SAT avoids this entirely through native propagation.

**Recommendation:** Use CP-SAT as the primary solver. If desired, compare against a simple priority-rule heuristic (e.g., earliest-due-date first for MTO, then fill-to-safety-stock for PSC) as a baseline to demonstrate the value of optimization.

The MIP formulation in this document serves as the **formal mathematical reference** — it defines the problem precisely and enables theoretical analysis. The CP implementation encodes the same constraints using interval variables and global constraints.

---

## 7. Illustrative Toy Example

### 7.1 Instance Setup

A **simplified single-line** instance to demonstrate all key constraint interactions:

| Parameter | Value |
|-----------|-------|
| Time horizon | 60 minutes (slots 0–59) |
| Roasters | $R_1, R_2$ (Line 1 only) |
| GC silos | 3 silos ($G_1, G_2, G_3$), capacity 3,000 kg each |
| RC silos | 2 silos ($S_1, S_2$), capacity 5,000 kg each. $S_2$ acts as buffer silo |
| RC-SKUs | PSC-A (both roasters), NDG-X ($R_1$ only) |
| GC-SKUs | GC-1, GC-2 |
| Setup time $\sigma$ | 5 minutes |
| Safety stock $\theta^{SS}$ | 3,000 kg (scaled for toy) |

**BOM:**

| RC-SKU | GC-1 | GC-2 | Batch size (RC) | Proc. time |
|--------|------|------|----------------|-----------|
| PSC-A  | 300 kg | 200 kg | 400 kg | 15 min |
| NDG-X  | 250 kg | 0 kg   | 300 kg | 13 min |

**Initial state:**

| Silo | SKU | Level |
|------|-----|-------|
| $G_1$ | GC-1 | 1,500 kg |
| $G_2$ | GC-2 | 1,000 kg |
| $G_3$ | empty | 0 kg |
| $S_1$ | PSC-A | 2,000 kg |
| $S_2$ | empty | 0 kg |

**Jobs:**
- **Job 1 (MTO):** NDG-X, 2 batches, due at minute 30
- **PSC pool:** up to 4 potential PSC-A batches ($b^{PSC}_1, b^{PSC}_2, b^{PSC}_3, b^{PSC}_4$)

**PSC consumption:** 200 kg PSC-A every 10 minutes

### 7.2 PSC Pool Pre-generation in Action

For this toy instance:
- $R_1$ compatible with PSC-A: $\lfloor 60/15 \rfloor = 4$ potential batches
- $R_2$ compatible with PSC-A: $\lfloor 60/15 \rfloor = 4$ potential batches
- Total PSC pool: 8 potential batches

The solver will activate some subset. In the solution below, 3 are activated:

| Pool batch | Activated? | Roaster | Start time |
|-----------|-----------|---------|------------|
| $b^{PSC}_{R1,1}$ | $a = 0$ | — | — |
| $b^{PSC}_{R1,2}$ | $a = 1$ | $R_1$ | $t = 31$ |
| $b^{PSC}_{R1,3}$ | $a = 0$ | — | — |
| $b^{PSC}_{R1,4}$ | $a = 0$ | — | — |
| $b^{PSC}_{R2,1}$ | $a = 1$ | $R_2$ | $t = 2$ |
| $b^{PSC}_{R2,2}$ | $a = 1$ | $R_2$ | $t = 19$ |
| $b^{PSC}_{R2,3}$ | $a = 0$ | — | — |
| $b^{PSC}_{R2,4}$ | $a = 0$ | — | — |

The 5 deactivated batches ($a = 0$) have all their $x_{b,r,t} = 0$ and $v^{con} = 0$ — they are invisible in the schedule.

### 7.3 Feasible Schedule

| Time | R1 | R2 | Pipeline (Line 1) | Inventory changes |
|------|----|----|-------------------|-------------------|
| $t=0$ | NDG-X B1 **starts** | idle | **consume** (2 slots): 250 kg GC-1 from $G_1$ | $G_1$: 1500 → 1250 |
| $t=2$ | roasting | PSC-A B1 **starts** | **consume** (2 slots): 300 kg GC-1 from $G_1$, 200 kg GC-2 from $G_2$ | $G_1$: 1250 → 950; $G_2$: 1000 → 800 |
| $t=4$ | roasting | roasting | **free** | — |
| $t=5$ | roasting | roasting | **replenish** $G_3$ with GC-1, +500 kg (2 slots) | $G_3$: 0 → 500 (assigned GC-1) |
| $t=7$ | roasting | roasting | **replenish** $G_1$ with GC-1, +500 kg (2 slots) | $G_1$: 950 → 1450 |
| $t=9$ | roasting | roasting | **free** | — |
| $t=10$ | roasting | roasting | — | **PSC event:** draw 200 kg from $S_1$ → $S_1$: 2000 → 1800 |
| $t=13$ | NDG-X B1 **ends** | roasting | **consume** (2 slots): 250 kg GC-1 from $G_1$ | $G_1$: 1450 → 1200. Same SKU → no setup! |
| | NDG-X B2 **starts** | | | |
| $t=17$ | roasting | PSC-A B1 **ends** | — | RC output 400 kg → $S_1$ (highest stock with PSC-A). $S_1$: 1800 → 2200 |
| $t=19$ | roasting | PSC-A B2 **starts** | **consume** (2 slots): 300 GC-1 from $G_1$, 200 GC-2 from $G_2$ | $G_1$: 1200 → 900; $G_2$: 800 → 600 |
| $t=20$ | roasting | roasting | — | **PSC event:** 200 kg from $S_1$ → $S_1$: 2200 → 2000 |
| $t=26$ | NDG-X B2 **ends** | roasting | — | Job 1 complete. $\text{tard} = \max(0, 26-30) = 0$ |
| | R1 enters **setup** (5 min) | | | NDG-X → PSC-A transition |
| $t=30$ | setup (min 4 of 5) | roasting | — | **PSC event:** 200 kg from $S_1$ → $S_1$: 2000 → 1800 |
| $t=31$ | R1 setup **ends** | roasting | — | — |
| | PSC-A B3 **starts** | | **consume** (2 slots): 300 GC-1 from $G_1$, 200 GC-2 from $G_2$ | $G_1$: 900 → 600; $G_2$: 600 → 400 |
| $t=34$ | roasting | PSC-A B2 **ends** | — | 400 kg → $S_1$: 1800 → 2200 |
| $t=40$ | roasting | idle | — | **PSC event:** 200 kg from $S_1$ → $S_1$: 2200 → 2000 |
| $t=46$ | PSC-A B3 **ends** | idle | — | 400 kg → $S_1$: 2000 → 2400 |

### 7.4 Variable Values for Key Constraints

**C1 — MTO assignment (each batch assigned once):**

| Batch | $x$ values | $\sum x$ |
|-------|-----------|----------|
| NDG-X B1 | $x_{B1, R1, 0} = 1$; all others = 0 | 1 ✓ |
| NDG-X B2 | $x_{B2, R1, 13} = 1$; all others = 0 | 1 ✓ |

**C2 — PSC activation linkage:**

| Batch | $a_b$ | $\sum x$ | Match? |
|-------|-------|----------|--------|
| PSC B1 | 1 | $x_{B1,R2,2} = 1$ → sum = 1 | ✓ |
| PSC B2 | 1 | $x_{B2,R2,19} = 1$ → sum = 1 | ✓ |
| PSC B3 | 1 | $x_{B3,R1,31} = 1$ → sum = 1 | ✓ |
| PSC B4 | 0 | all $x = 0$ → sum = 0 | ✓ |

**C3 — Compatibility:**

NDG-X on $R_1$ ✓ (compatible). Not on $R_2$ (Busta+PSC only — wait, in problem statement R2 does NDG too). Regardless, NDG-X is assigned to $R_1$ which is compatible. ✓

**C5 — No overlap on R1:**

R1 schedule: B1 at [0, 12], B2 at [13, 25], setup [26, 30], B3 at [31, 45].
- At $t = 12$: only B1 active. ✓
- At $t = 13$: only B2 active (B1 ended). ✓
- No two processing windows overlap. ✓

**C5 — No overlap on R2:**

R2 schedule: PSC-B1 at [2, 16], PSC-B2 at [19, 33].
- Gap between B1 end (17) and B2 start (19) = 2 min. ✓ (Same SKU, no setup needed; gap due to pipeline constraint.)

**C6 — Setup time:**

R1: NDG-X B2 ends at $t = 26$. Next batch is PSC-A (different SKU). PSC-A B3 starts at $t = 31$. Gap = $31 - 26 = 5 = \sigma$. ✓

R2: PSC-A B1 → PSC-A B2. Same SKU → no setup required. ✓

**C11 — Pipeline mutual exclusion:**

| Slots | Operation | Conflict check |
|-------|-----------|---------------|
| [0, 1] | Consume for R1 (NDG-X B1) | ✓ free before |
| [2, 3] | Consume for R2 (PSC-A B1) | ✓ no overlap with [0,1] |
| [4] | Free | — |
| [5, 6] | Replenish $G_3$ | ✓ no overlap |
| [7, 8] | Replenish $G_1$ | ✓ no overlap |
| [9–12] | Free | — |
| [13, 14] | Consume for R1 (NDG-X B2) | ✓ |
| [15–18] | Free | — |
| [19, 20] | Consume for R2 (PSC-A B2) | ✓ |
| [21–30] | Free | — |
| [31, 32] | Consume for R1 (PSC-A B3) | ✓ |

No two operations overlap in any time slot. ✓

**C18 — BOM match:**

NDG-X B1: draws 250 kg GC-1 from $G_1$, 0 kg GC-2. BOM requires (250, 0). ✓

PSC-A B1: draws 300 kg GC-1 from $G_1$, 200 kg GC-2 from $G_2$. BOM requires (300, 200). ✓

**C17 — GC capacity:**

$G_1$ peak after replenish at $t=7$: 1450 kg ≤ 3000. ✓

$G_3$ after replenish at $t=5$: 500 kg ≤ 3000. ✓

**C20 — Sufficient GC:**

PSC-A B1 at $t=2$ draws 300 kg GC-1 from $G_1$. $G_1$ level at $t=1$: 1250 kg ≥ 300. ✓

PSC-A B1 draws 200 kg GC-2 from $G_2$. $G_2$ level at $t=1$: 1000 kg ≥ 200. ✓

**C26 — RC capacity:**

$S_1$ peaks at 2400 kg ≤ 5000. ✓

**C9 — Tardiness:**

Job 1: last batch (NDG-X B2) ends at $t = 26 \leq 30$. Tardiness = 0. ✓

**C29 — PSC consumption (highest-stock rule):**

At $t = 10$: $S_1 = 2000$ kg, $S_2 = 0$ kg. Only $S_1$ has correct SKU and stock. $d_{L1,S1,10} = 1$. Draw 200 kg. ✓

At $t = 20$: $S_1 = 2200$ kg, $S_2 = 0$ kg. Same. ✓

**C38 — Idle penalty:**

During R1 setup [26–30], $\text{busy}_{R1,t} = 0$. Total RC = $S_1$ ≈ 2000 kg < $\theta^{SS} = 3000$ kg → $\text{low} = 1$. Therefore $\text{idle}_{R1,t} = 1$ for $t \in \{26,27,28,29,30\}$. Penalty = $5 \times c^{idle}$. This correctly captures the cost of the SKU transition under low inventory.

### 7.5 What Would Go Wrong Without Key Constraints?

**Without C11 (pipeline mutex):** R1 and R2 could both start consuming at $t = 0$ simultaneously. In reality, the shared pipeline can only serve one operation at a time. The constraint forces R2 to wait until $t = 2$, introducing a 2-minute delay that cascades through R2's schedule.

**Without C6 (setup time):** After NDG-X B2 ends at $t = 26$, R1 could start PSC-A at $t = 26$ instead of $t = 31$. This saves 5 minutes of idle time but is physically impossible — the roaster needs cleaning and reconfiguration.

**Without C20 (sufficient GC):** At $t = 31$, $G_1$ has 900 kg of GC-1. If the scheduler tried to draw 1000 kg, the physical system would fail. The constraint ensures: you can only take what's actually there.

**Without C2 (PSC activation):** All 8 potential PSC batches would need to be scheduled, overfilling RC silos and wasting GC. The activation variable lets the solver pick exactly the right number.

**Without C21 (max 2 silos with emptying):** A batch could arbitrarily split GC sourcing across many silos, making silo management chaotic. The constraint ensures splits only happen to *free up* silos for reassignment.

---

## 8. Summary of Constraint Categories

| ID | Category | Type | Key Difficulty |
|----|----------|------|----------------|
| C1–C4 | Batch assignment | Hard | Standard scheduling |
| C5–C6 | Roaster sequencing | Hard | Setup time + pipeline interaction |
| C7–C8 | Downtime / changeover | Hard | Standard |
| C9 | Tardiness | Soft (penalized) | Standard |
| C10 | R3 output direction | Hard | Binary choice per batch |
| C11–C15 | **Pipeline mutual exclusion** | Hard | **Core bottleneck**; variable-duration dump linearization |
| C16–C21 | **GC silo inventory** | Hard | Dynamic SKU assignment; split-sourcing with emptying |
| C22–C24 | **GC silo SKU purity** | Hard | Conditional logic (empty before reassign) |
| C25–C28b | **RC silo inventory** | Hard | Look-ahead feasibility |
| C29–C30 | **PSC consumption** | Hard/Soft | **Highest-stock priority** (revised from FIFO) |
| C31 | SKU switch threshold | Hard | Bilinear linearization |
| C32–C35 | Changeover | Hard | Threshold + timing conditions |
| C36–C38 | Idle penalty | Soft | Conditional (idle AND low stock) |

---

## 9. Notation Quick Reference

| Symbol | Meaning |
|--------|---------|
| $x_{b,r,t}$ | Batch $b$ starts on roaster $r$ at time $t$ |
| $a_b$ | PSC batch $b$ is activated (used) |
| $y_b$ | R3 batch $b$ output goes to Line 1 |
| $z^r_b$ | Batch $b$ is assigned to roaster $r$ |
| $u^{rep}_{l,s,g,t}$ | Replenish GC-SKU $g$ into silo $s$ of line $l$ at time $t$ |
| $u^{dump}_{l,s,t}$ | Dump silo $s$ of line $l$ at time $t$ |
| $n^{dump}_{l,s,t}$ | $\lceil q^{dump}_{l,s,t} / 100 \rceil$ — dump duration helper |
| $q^{dump}_{l,s,t}$ | Quantity dumped (kg) |
| $v^{con}_{b,l,s,g}$ | GC-SKU $g$ drawn by batch $b$ from silo $(l,s)$ |
| $w_{b,s,g}$ | Batch $b$ uses silo $s$ for GC-SKU $g$ |
| $e_{b,s,g}$ | Silo $s$ emptied by batch $b$'s draw of GC-SKU $g$ |
| $f_{b,l,s}$ | Batch $b$ RC output fills RC silo $(l,s)$ |
| $d_{l,s,t}$ | Silo $s$ selected for PSC draw at time $t$ |
| $I^{GC}_{l,s,t}$ | GC inventory of silo $(l,s)$ at time $t$ |
| $I^{RC}_{l,s,t}$ | RC inventory of silo $(l,s)$ at time $t$ |
| $\phi^{GC}_{l,s,g,t}$ | GC silo $(l,s)$ assigned to GC-SKU $g$ at time $t$ |
| $\phi^{RC}_{l,s,k,t}$ | RC silo $(l,s)$ holds RC-SKU $k$ at time $t$ |
| $p^{con}_{l,t}, p^{rep}_{l,t}, p^{dump}_{l,t}$ | Pipeline occupation indicators |
| $\text{short}_{l,t}$ | PSC shortage on line $l$ at time $t$ |
| $\text{idle}_{r,t}$ | Roaster $r$ idle with low RC at time $t$ |
| $\text{tard}_j$ | Tardiness of MTO job $j$ |
| $\text{busy}_{r,t}$ | Roaster $r$ is processing at time $t$ |
| $\text{low}_{l,t}$ | Total RC on line $l$ is below safety stock |

---

## 10. Modeling Assumptions Summary

| ID | Assumption | Risk if violated |
|----|-----------|-----------------|
| A1 | All inputs deterministic, known at shift start | Unplanned events require manual re-planning |
| A2 | GC silo operations discretized to 1-minute slots | Small timing errors vs. physical reality |
| A3 | PSC consumption discrete every 10 minutes | Inaccurate if cycle time varies by SKU |
| A4 | No shelf life / date-based FIFO | Acceptable within 8-hour horizon |
| A5 | NDG/Busta due date is soft with very large penalty | Avoids infeasibility in extreme cases |
| A6 | Dump/replenish/consume quantities credited/debited at operation start | Simplifies inventory balance; duration enforced by pipeline constraints |
| A7 | Uniform 5-minute setup time for all SKU transitions | May underestimate some transitions, overestimate others |
| **A8** | **RC consumption uses highest-stock priority instead of FIFO-by-fill-time** | **May diverge from actual behavior when fill patterns are irregular. Equivalent under steady consumption.** |
