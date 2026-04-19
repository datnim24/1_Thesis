# Surrounding Information & Introduction Reference
## Thesis: Dynamic Batch Roasting Scheduling with Shared Pipeline Constraints under Unplanned Disruptions: A Case Study at Nestlé Trị An

> **Purpose of this file:** Living reference document synthesizing all gathered contextual knowledge — from Q&A sessions, problem description, and mathematical formulation — to guide writing of Chapters 1–6. Every claim is traceable. Items marked ⚠️ remain open. Items marked ✅ are resolved.

---

## 1. Thesis Identity

| Item | Value |
|---|---|
| **University** | Ho Chi Minh City International University, Vietnam National University (IU-VNU) |
| **Level** | Undergraduate thesis (Bachelor of Engineering, Logistics and Supply Chain Management) |
| **Standard** | Full academic rigor — must be defensible before an examination committee |
| **Primary framing** | Methodological comparison of reactive scheduling strategies on a real industrial problem. The scheduling model is a tool, not the contribution — the contribution is the comparative analysis of how different strategies handle disruptions. |
| **Case company** | Nestlé Trị An Manufacturing Facility, Đồng Nai Province, Vietnam |
| **Working title** | **Dynamic Batch Roasting Scheduling with Shared Pipeline Constraints under Unplanned Disruptions: A Case Study at Nestlé Trị An** |

> **Title rationale:** The title names four differentiating elements: (1) "Dynamic Batch Roasting Scheduling" — signals reactive/online scheduling, not just static optimization, (2) "Shared Pipeline Constraints" — names the core physical bottleneck that creates cascading effects, (3) "Unplanned Disruptions" — the stochastic element driving the methodology, and (4) "A Case Study at Nestlé Trị An" — grounded in real industry. The title is problem-forward: specific methods (CP-SAT, DRL, dispatching) are described in the abstract, not the title — keeping flexibility if methods evolve.

---

## 2. Company Background (for Section 2.1)

### What can be stated ✅

- **Company:** Nestlé Trị An — a manufacturing facility of Nestlé Vietnam, located in Trị An, Đồng Nai province.
- **Product focus:** Roasted coffee products, including:
  - **PSC (Pure Soluble Coffee):** The primary high-volume product. PSC is the purest coffee powder — an intermediate product shipped as 500 kg bags to Đồng Nai Factory or Thailand Nescafé plant for manufacturing into finished Nescafé products. PSC is the make-to-stock (MTS) product that drives continuous downstream demand.
  - **NDG (Nescafé Dolce Gusto):** Capsule coffee product. Make-to-order (MTO) with firm delivery windows.
  - **Busta:** A dedicated roasting product for the Starbucks brand. Make-to-order (MTO) with firm delivery windows.
- **Throughput (disclosable):** PSC ships approximately **17–20 tons of pure coffee powder per 8-hour shift**. Output figures for NDG and Busta are classified.
- **Production structure:** Two parallel production lines (Line 1, Line 2), each with dedicated roasters. Five roasters total (R1–R5). R3 is a physical cross-line bridge — always consumes GC from Line 2 pipeline but can route RC output to either line.
- **Staffing on the roasting floor:** **3–4 operators plus 1 contractor** per shift manage the roasting stage. This small crew size means within-shift decisions rely heavily on individual operator experience — and are vulnerable to cognitive overload during disruptions.
- **Shift structure:** 3 shifts per day, 8 hours each, continuous operation.
- **Planning cycle:** Head Office issues demand (POs) → Supply Chain distributes to lines → supervisors hold daily planning meetings (~1 hour) → weekly plan updated daily. Within-shift execution is operator-driven.

### What cannot be stated (confidentiality / data access)

- Exact NDG and Busta production volumes — **classified**.
- Specific VND financial figures from internal P&L — **not available**.
- **UPS data** (MTBF, MTTR per roaster) — **not available from factory**. UPS parameters in thesis are synthetic, based on literature estimates. This is a stated limitation.

### Framing guidance for Section 2.1

Write as: *"a high-volume, continuous-process food manufacturing plant operating a batch roasting stage with a hybrid demand structure (MTO and MTS), a shared pipeline physical constraint, and frequent exposure to unplanned equipment stoppages that disrupt within-shift scheduling."* PSC throughput (17–20 tons/shift) can be stated. Emphasize: 5 roasters × 2 lines × shared pipeline per line × 3–4 operators = both the justification for formal optimization AND the vulnerability to disruptions.

---

## 3. Current Practice (for Section 1.1)

### How within-shift scheduling is done today

The current process is **human-driven and experience-based** at the execution level:

1. **Strategic input:** PO demand data from Head Office → distributed to lines by Factory Supply Chain.
2. **Daily meeting (~1 hour):** Supervisors collectively plan considering line conditions, planned stoppages, and cross-department needs.
3. **Execution layer (within the shift):** Detailed decisions — which roaster runs next, when to start each batch, how to manage RC buffer levels — are made **in real time by operators** based on experience and direct observation. No algorithmic decision support exists at this layer.
4. **Disruption response:** When an unplanned stoppage occurs, operators react ad-hoc. There is no systematic re-planning protocol — the operator assesses the situation, re-prioritizes mentally, and makes adjustments on the fly. This is where the most significant scheduling quality loss occurs.
5. **Update cycle:** Weekly plan revised daily.

### Why this thesis is relevant

> The thesis targets the **within-shift reactive scheduling layer** — specifically, how to (a) generate a good initial schedule and (b) respond systematically when unplanned stoppages disrupt that schedule. The strategic planning layer remains human-managed.

The key insight: operators can produce reasonable schedules under normal conditions, but their **response to disruptions is ad-hoc and uncoordinated**. An unplanned stoppage on one roaster cascades through pipeline contention to other roasters, and the operator must mentally re-sequence multiple roasters simultaneously under time pressure. This is where algorithmic support adds the most value.

### The gap this thesis fills

The gap is the **absence of a systematic reactive scheduling strategy** that can respond to within-shift disruptions — specifically, unplanned equipment stoppages — in a way that minimizes throughput loss and maintains downstream supply continuity. No existing study provides a systematic comparative evaluation of reactive scheduling strategies — optimization-based re-planning versus learning-based policy — for batch roasting operations with shared pipeline constraints under stochastic equipment failures.

---

## 4. Operational Pain Points (for Section 1.2)

### Observed figures

All values below are **rough estimates from on-site observation, interviews, and manual timing during the internship period.** Not statistically validated. Present as qualitative motivation only — use hedged language ("observation suggests," "interviews indicate," "estimated at approximately").

| Observation | Rough Figure | Root cause framing |
|---|---|---|
| Overall speed loss vs. theoretical | ~18% of theoretical throughput | **Dominated by UPS** — unplanned stoppages are the single largest contributor to throughput loss |
| Total roaster stoppage from RC high-stock | ~300 minutes per observed period | RC buffer overflow halts roasters — consequence of uncoordinated scheduling after disruptions |
| UPS frequency (qualitative) | Frequent enough to disrupt most shifts | Operators report that "clean" shifts (zero UPS) are the exception, not the norm |

> **Writing guidance:** The 18% throughput gap and the dominance of UPS as its cause is the primary motivational statistic. Supplement with the 300-minute RC overflow figure as evidence of cascading effects from poor disruption response.

---

## 5. Academic Contribution (for Section 1.2 and 2.2)

### Primary claim ✅

> **A systematic comparative study of four reactive scheduling strategies — dispatching heuristic (operator baseline), tabular Q-Learning, end-to-end MaskedPPO, and RL-based Hyper-Heuristic (tool-based DRL) — for batch roasting with shared pipeline constraints under unplanned stoppages, evaluated on a factorial experimental design across disruption intensities. MILP and CP-SAT serve as deterministic benchmarks establishing the theoretical performance ceiling. The objective is to maximize shift profit (revenue minus penalties for tardiness, stockout, and idle time). The four reactive methods span a complexity spectrum from zero-intelligence rules to structured learning, testing whether added complexity yields proportional performance gains.**

Novelty is in the *systematic comparison of fundamentally different reactive approaches* on a *real industrial constraint structure* with *controlled stochastic disruptions*. Not in algorithmic invention. MILP validates CP-SAT's optimality on the deterministic problem; the four reactive strategies are compared under UPS. The RL-Hyper-Heuristic (RL-HH) component applies the emerging RL-HH paradigm (86 publications 2020–2024, Li et al. 2024) to batch scheduling with shared pipeline constraints — a problem structure not previously addressed in the RL-HH literature.

### Supporting differentiating elements

1. **Pipeline mutex as the propagating constraint:** The shared GC pipeline (60% utilization on Line 2) means a UPS on one roaster cascades into pipeline conflicts for other roasters. This cascading effect is what makes reactive scheduling non-trivial — and what distinguishes this from generic parallel machine rescheduling.

2. **Cross-line R3 routing as reactive tactical decision:** When Line 1 loses throughput due to a UPS on R1 or R2, the scheduler can route R3 output to Line 1 as compensation — but at the cost of Line 2's pipeline capacity. This cross-line tradeoff under disruption is tested as an experimental factor (fixed vs. flexible R3 routing).

3. **End-to-end DRL vs. structured DRL (RL-HH):** The hypothesis is that decomposing the scheduling problem into domain-expert tools (simple heuristics) selected by an RL agent outperforms end-to-end neural network approaches (MaskedPPO), which must learn both scheduling mechanics and timing simultaneously. This tests whether domain knowledge integration yields better performance than brute-force learning.

4. **Method ladder across complexity levels:** Four reactive methods span a clean spectrum: no learning (dispatching) → simple learning (Q-Learning) → deep learning (PPO) → structured learning (RL-HH). The thesis tests whether each added complexity level yields proportional performance improvement — or whether simpler methods suffice.

### What the contribution is NOT

- Not a new scheduling algorithm or solver.
- Not a complexity proof.
- Not a real-time deployed production tool.
- Not an RL algorithm contribution (PPO is used as-is from Stable-Baselines3).
- Not a claim that any strategy is universally superior — the contribution is identifying **when each strategy dominates**.

---

## 6. Methodology Architecture (for Sections 3.1 and 3.2)

### Five-method architecture (4 reactive + 1 deterministic benchmark)

| Method | Type | Implementation | Role in thesis |
|---|---|---|---|
| **MILP** (benchmark) | Exact optimization | PuLP / OR-Tools MILP | Deterministic benchmark — LP relaxation lower bound to verify CP-SAT quality. Does NOT participate in reactive experiments. |
| **CP-SAT** (ceiling) | Constraint programming | Google OR-Tools CP-SAT (Python API) | Deterministic solver — establishes theoretical performance ceiling (~$295k). Does NOT participate in reactive experiments (re-solve time ~2 min on thesis hardware = impractical). |
| Dispatching heuristic | Rule-based | Priority rules: urgency-threshold MTO, most-remaining-batches priority, lowest-stock R3 routing | **Core baseline** — represents current operator practice. Null hypothesis. |
| Q-Learning | Tabular RL | Custom tabular Q-learning, 9,233 Q-table entries, ε-greedy | Simplest RL approach. Tests: does learning help at all vs rules? |
| MaskedPPO | End-to-end DRL | Stable-Baselines3 MaskablePPO, 21 actions, 33-dim observation, action masking | Deep RL — learn everything from scratch. Tests: does deep learning add value? |
| RL-Hyper-Heuristic | Tool-based DRL | Q-learning meta-agent selects from 6 simple heuristic tools | Structured DRL — domain knowledge + RL. Tests: does decomposition beat brute-force learning? |

### Why these four reactive methods (academic justification)

The four strategies represent a **method ladder** across complexity levels:

- **Dispatching heuristic:** Zero learning. Fixed rules. Represents current operator practice — the "null hypothesis."
- **Q-Learning:** Simplest RL. Tabular. No neural network. Tests whether ANY learning improves over fixed rules.
- **MaskedPPO:** End-to-end DRL. Neural network learns everything from raw observations and actions. Tests whether deep learning is needed.
- **RL-Hyper-Heuristic:** Structured DRL. RL selects from domain-expert tools. Decomposes WHAT to do (embedded in tools) from WHEN to do it (learned by RL). Tests whether domain knowledge integration outperforms pure learning. Literature backing: 86 publications 2020–2024 on RL-based Hyper-Heuristics (Li et al. 2024, PeerJ Comput. Sci.).

The research question: **does each added level of complexity yield proportional performance improvement? Under what disruption conditions?**

> **Note on CP-SAT reactive (removed):** CP-SAT event-triggered re-solve was implemented and tested but removed from the reactive comparison. Re-solve time (~2 minutes on thesis hardware, Intel i3-9100F) makes it impractical for real-time reactive scheduling. The implementation remains in the codebase for reference. CP-SAT's role is limited to the deterministic benchmark (Layer A).

### Comparison metrics ✅

| Metric | Definition | Unit |
|---|---|---|
| **Total profit** | Revenue ($4k/PSC, $7k/MTO per batch) minus all costs (primary metric) | $ |
| **PSC throughput** | PSC batches completed per shift | batches |
| **Stockout count** | Consumption events where $B_l < 0$ (strictly negative, demand unmet). Penalized at $1,500/event in objective. | events |
| **Stockout duration** | Total minutes with $B_l \leq 0$ (includes zero — line stalled). KPI only — not in objective. | minutes |
| **MTO tardiness** | $\sum_j \text{tard}_j$ (penalized at $1,000/min in objective) | minutes |
| **Computation time** | Wall-clock time per re-solve / per RL inference | seconds |

> **Important:** Objective/reward optimizes stockout **event count**, not duration. Duration is a separate operational KPI. See `cost.md` for full cost structure.

### Baseline

The dispatching heuristic IS the baseline. All four reactive strategies are compared against each other on the same simulated shift instances with the same UPS realizations (paired comparison). The relevant questions: **how much does each learning method improve over dispatching? Does complexity always help? Under what conditions?**

The CP-SAT deterministic solution (~$295k) serves as the **theoretical ceiling** — the best possible performance with perfect information and no disruptions. The gap between this ceiling and each reactive method's performance measures the "cost of uncertainty."

---

## 7. Validation and Experimental Dataset (for Section 1.4 and Chapter 5)

### Historical data (deterministic validation)

| Parameter | Value |
|---|---|
| **Source** | Nestlé Trị An factory production information system |
| **Period** | One full week — the week with lowest observed UPS frequency |
| **Instances** | 21 shifts (3 shifts/day × 7 days) |
| **Granularity** | ✅ Batch-level — individual batch start times, roaster assignments available |
| **Use** | Deterministic baseline validation — verify model feasibility and throughput on known data |
| **Access** | Formally extracted with approval of internship supervisor and Line Manager |
| **Anonymization** | GC SKU identifiers changed; quantitative parameters reflect real values |

### Simulated experimental data (primary experiments)

| Parameter | Value |
|---|---|
| **Source** | Simulation engine built for this thesis |
| **UPS model** | Exponential inter-arrival (rate λ), parameterized duration (mean μ), uniform roaster selection |
| **UPS parameters** | Synthetic — literature-based MTBF/MTTR estimates, NOT calibrated to plant data |
| **Factorial design** | 5 UPS rates × 3 durations × 3 strategies × 2 R3 routing modes = 90 cells |
| **Replications** | 100 per cell = 9,000 total simulation runs |
| **Controlled randomness** | Same 100 UPS realizations applied to all 3 strategies within each cell (paired comparison) |
| **Estimated compute** | ~2.5 hours on i3-9100F |

---

## 8. Scope and Limitations (for Section 1.4)

### Scope boundaries

| Dimension | In scope | Out of scope |
|---|---|---|
| Planning horizon | Single 8-hour shift (480 min slots) | Multi-shift, weekly, rolling horizon |
| Shift types | Normal production and PSC↔MTO SKU switch | 4-hour changeover, PSC-PSC switch |
| Lines | Both L1 and L2 with R3 cross-line | — |
| Methods compared | Dispatching + Q-Learning + MaskedPPO + RL-HH for reactive; **MILP + CP-SAT as deterministic benchmark** | Metaheuristics (ALNS documented as future work) |
| Objective | **Maximize profit** (revenue − tardiness − stockout − idle costs). All terms in $. See `cost.md`. | Multi-objective Pareto |
| Uncertainty | Unplanned stoppages (UPS) — stochastic | Stochastic demand, yield uncertainty |
| Deliverable | Methodology comparison + simulation framework | Deployed production tool, dashboard |
| Data | Real parameters for model calibration; simulated UPS for experiments | Real-time sensor integration |
| Rescheduling | Reactive (dispatching rules / tabular RL / end-to-end DRL / tool-based RL-HH) | Proactive robust scheduling, rolling horizon |
| GC supply | Finite GC silos with periodic restocking (C16–C19) | GC dump, SKU purity tracking |
| RC tracking | Aggregate batch counter per line | Individual RC silo tracking, fill/draw rules |

### All acknowledged limitations ✅

| ID | Limitation | Implication if violated |
|---|---|---|
| L1 | SKU-dependent roasting times (PSC=15, NDG=17, Busta=18 min) | Real times may vary ±1 min within a SKU due to bean conditions |
| L2 | Uniform 5-min setup time for all SKU pairs | Asymmetric setup times would change optimal sequencing |
| L3 | GC silos finite with fixed restock (5 batch / 15 min); shared station | Restock time may vary in practice; silo capacities are approximate |
| L4 | PSC consumption rate constant throughout shift | Rate changes (line speed adjustment) not captured |
| L5 | UPS parameters synthetic — not calibrated to plant data | Results show relative strategy performance, not absolute plant-specific predictions |
| L6 | UPS affects exactly 1 roaster at a time | Correlated failures (power outage) not modeled |
| L7 | UPS cancels in-progress batch entirely (no pause/resume) | Simplifies state tracking; slightly pessimistic vs. reality if partial recovery is possible |
| L8 | RC tracked as aggregate counter, not individual silos | Acceptable — only 1 PSC SKU per shift, no mixing concern |
| L9 | No cross-shift state carryover | Each shift is independent problem instance |
| L10 | No shelf life / FIFO constraints | Acceptable within 8-hour horizon |
| L11 | Observational figures (18% throughput gap, 300 min stoppage) from limited samples | Used as motivation only; not statistically representative |
| L12 | DRL agent trained on synthetic UPS — transferability to real plant unknown | Would require retraining with real MTBF/MTTR data for deployment |
| L13 | Initial last_sku = PSC for all roasters at shift start | Consistent with factory default; cross-shift carryover not modeled |
| L14 | Cost values ($4k, $7k, $1k, $1.5k, $200, $50) are proxy costs | Relative ratios drive scheduling decisions; absolute values not audited |
| L15 | MILP used for deterministic benchmark only, not reactive re-solve | MILP too slow for real-time re-solve; CP-SAT preferred for disjunctive scheduling |

---

## 9. Design Requirements (for Section 1.3)

The system must satisfy all of the following:

1. Generate a feasible predictive schedule for a full 8-hour shift with no constraint violations (pipeline NoOverlap, RC inventory bounds, setup times, downtime avoidance).
2. React to UPS events: when an unplanned stoppage occurs, the reactive strategy produces a revised decision within practically acceptable time (target: < 1 millisecond for dispatching and RL inference; < 100 milliseconds for RL-HH tool selection + execution).
3. Track RC inventory in batch units for both lines and prevent stockout (deterministic mode) or minimize stockout impact (reactive mode).
4. Handle MTO batch scheduling with soft due-date constraint at slot 240.
5. Support R3 cross-line output routing as a decision variable (and as experimental factor: fixed vs. flexible).
6. Enable controlled experimental comparison: same UPS realizations applied to all four reactive strategies for fair paired comparison.
7. Accept real production parameters as input: roaster eligibility, processing times, setup times, PSC consumption rates, planned downtime windows.

---

## 10. Computing Environment (for Section 3.2)

| Component | Specification |
|---|---|
| Type | VPS (Virtual Private Server), 24/7 operation |
| CPU | Intel Core i3-9100F — 4 cores / 4 threads @ 3.6 GHz base |
| RAM | 16 GB DDR4 @ 2400 MHz |
| GPU | AMD Radeon RX470 4 GB (potentially used for DRL training; otherwise CPU-only) |
| CP-SAT solver | Google OR-Tools CP-SAT (Python API) |
| DRL framework | Stable-Baselines3 with MaskablePPO (Python, Gymnasium) |
| Simulation | Custom Python simulation engine |

**CP-SAT re-solve time:** Target < 1 second per re-solve. Model size ~200 variables — well within i3-9100F capability.

**DRL training time:** Estimated 10,000 episodes × ~0.1 sec/episode = ~15–20 minutes for initial training. Hyperparameter tuning may require multiple training runs.

**Total experiment runtime:** 9,000 simulation runs × ~1 sec/run ≈ 2.5 hours. Very tractable on available hardware.

> **Writing note:** The i3-9100F is a modest, consumer-grade CPU (2018). Acknowledge this when reporting absolute solve times — results will improve on higher-core machines. The *relative* strategy comparison remains valid since all strategies run on identical hardware.

---

## 11. Literature Framework (for Section 2.2)

### Five pillars

**Pillar 1 — Parallel Machine Scheduling with Shared Resources / Hybrid Flow Shop with Limited Buffers**
- Theoretical grounding for the pipeline mutex constraint and buffer-coupled scheduling
- Key references:
  - Bektur & Saraç (2019) — common-server parallel machine scheduling with SDST and machine eligibility
  - Hakimzadeh & Zandieh (2012) — bi-objective HFS with SDST and limited buffers (MILP + metaheuristic). Core structural match: parallel machines + setup + finite buffer. Clean MILP template.
  - Lin et al. (2020) — reentrant HFS with limited buffers and stockers. Useful for centralized buffer (RC silo) modeling.
  - Zheng et al. (2024) — cooperative adaptive GA for reentrant HFS with SDST and limited buffers. Closest match to "multiple parallel roasters + setup + finite buffer."
  - Klanke et al. (2021) — make-and-pack short-term scheduling with finite intermediate buffer and SDST. Discrete-time MILP + decomposition for tractability.
- Broader: SDST surveys (Allahverdi 2015), unrelated parallel machines

**Pillar 2 — Scheduling Under Uncertainty / Reactive Rescheduling**
- Central pillar — the core methodological question of the thesis
- Surveys: Vieira et al. (2003), Herroelen & Leus (2005), Ouelhadj & Petrovic (2009) — rescheduling taxonomy
- Key references:
  - Vin & Ierapetritou (2000) — reactive rescheduling of batch plants under breakdowns and rush orders. **Direct template for the CP-SAT re-solve layer.**
  - Aurich et al. (2017) — simulation-based optimization of 4-stage HFS with SDST and breakdowns. Template for simulation-optimization integration.
  - Gholami et al. (2009) — HFS with SDST + random breakdowns. Bridges setup sequencing and stochastic downtime.
  - Raissi et al. (2019) — stochastic flexible flow shop with PM and buffer holding costs. Shows buffer-cost + stochastic integration.
  - Rooeinfar et al. (2019) — stochastic flexible flow shop with limited buffers and PM downtime. High relevance for combining LB with uncertainty.
- Reactive strategies taxonomy: complete rescheduling, match-up scheduling, right-shift rescheduling

**Pillar 3 — Reinforcement Learning for Production Scheduling**
- One of two primary methods in the thesis
- Foundational: Zhang et al. (2020) — RL for job shop scheduling; Park et al. (2021) — RL with GNN for scheduling
- Action masking: Huang & Ontañón (2022) — invalid action masking for RL
- Target 2020–2025: RL applied to parallel machine or batch scheduling, especially with stochastic disruptions
- MaskablePPO: Stable-Baselines3 documentation + papers using it for constrained scheduling

**Pillar 4 — CP-SAT / MILP for Industrial Scheduling**
- Both primary methods in the thesis (CP-SAT for reactive, MILP for deterministic benchmark)
- Key references:
  - Laborie et al. (2018) — CP Optimizer interval variables and NoOverlap
  - Naderi et al. (2023) — CP vs. MIP for scheduling (empirical evidence of CP advantage on disjunctive structures)
  - Maravelias & Grossmann (2003) — continuous-time STN MILP for multipurpose batch plants. **Strongest "inventory-coupled batch MILP" backbone.** 279 Scopus citations.
  - Wallrath et al. (2023) — batch plant lot-sizing + scheduling with time-bucket MILP. Template for shift-horizon + inventory balances.

**Pillar 5 — Process Systems / Batch Plant Scheduling with Inventory Coupling**
- Conceptual bridge between flow shop literature and process engineering
- Key references:
  - Maravelias & Grossmann (2003) — STN formulation with storage policies and inventory coupling (also in Pillar 4)
  - Nguyen et al. (2025) — HFS with heterogeneous parallel machines and integrated WIP inventory
  - Beldar et al. (2022) — non-identical parallel batch machines with maintenance (MILP + SA + VNS)

### Reference summary table

| Ref | Year | Problem | Method | Disruptions? | Buffers? | Parallel Machines? | Relevance |
|-----|------|---------|--------|-------------|----------|-------------------|-----------|
| Bektur & Saraç | 2019 | Common-server parallel machines, SDST | Exact | — | Server constraint | ✅ | Pipeline mutex |
| Hakimzadeh & Zandieh | 2012 | HFS + SDST + limited buffers | MILP + metaheuristic | — | ✅ | ✅ | Core structural match |
| Zheng et al. | 2024 | Reentrant HFS + SDST + LB | CAGA metaheuristic | — | ✅ | ✅ | Closest to roasting |
| Klanke et al. | 2021 | Make-and-pack + buffer + SDST | Discrete-time MILP + decomposition | — | ✅ | ✅ | MILP tractability template |
| Lin et al. | 2020 | Reentrant HFS + stockers + LB | HHSGA metaheuristic | — | ✅ | ✅ | Central buffer concept |
| Maravelias & Grossmann | 2003 | STN batch plant scheduling | Continuous-time MILP | — | ✅ (storage policies) | ✅ | Inventory-coupled MILP backbone |
| Vin & Ierapetritou | 2000 | Reactive rescheduling batch plants | MILP rescheduling | ✅ (breakdowns, rush) | ✅ (material coupling) | ✅ | **Direct re-solve template** |
| Aurich et al. | 2017 | 4-stage HFS + SDST + breakdowns | Simulation + SA/TS/DE | ✅ (stochastic) | implicit | ✅ | Simulation-optimization loop |
| Raissi et al. | 2019 | Stochastic FFS + PM + buffer costs | Stochastic MILP + metaheuristics | ✅ (PM + stochastic) | ✅ (holding costs) | ✅ | Buffer cost + stochastic |
| Gholami et al. | 2009 | HFS + SDST + breakdowns | GA-based | ✅ (random breakdowns) | — | ✅ | Setup + disruption bridge |
| Rooeinfar et al. | 2019 | Stochastic FFS + LB + PM | Simulation + GA/SA/PSO | ✅ (PM + stochastic) | ✅ | ✅ | LB + uncertainty |
| Nguyen et al. | 2025 | HFS + WIP inventory coupling | Multi-objective model | — | ✅ (WIP) | ✅ | Inventory KPI integration |
| Zhang et al. | 2020 | Job shop scheduling with RL | DQN/PPO | — | — | — | RL for scheduling |
| Huang & Ontañón | 2022 | Action masking for RL | MaskablePPO | — | — | — | Action masking reference |
| Naderi et al. | 2023 | CP vs. MIP for scheduling | CP-SAT, MILP | — | — | ✅ | CP-SAT solver choice justification |
| Laborie et al. | 2018 | CP Optimizer intervals | CP | — | — | ✅ | NoOverlap constraint modeling |

### Checklist

- [x] ≥ 10 references total (16+ identified)
- [x] ≥ 50% published 2019+ (10/16)
- [x] ≥ 2 reactive/dynamic scheduling surveys (Vieira 2003, Ouelhadj 2009, Vin & Ierapetritou 2000)
- [x] ≥ 2 RL for scheduling papers (Zhang 2020, Huang & Ontañón 2022)
- [x] ≥ 1 paper using CP-SAT/OR-Tools (Naderi 2023, Laborie 2018)
- [x] ≥ 1 SDST scheduling reference (Bektur & Saraç 2019, Hakimzadeh & Zandieh 2012)
- [x] ≥ 1 paper on scheduling under breakdowns (Vin & Ierapetritou 2000, Aurich 2017, Gholami 2009)
- [x] ≥ 1 buffer/inventory-coupled scheduling paper (Maravelias & Grossmann 2003, Nguyen 2025)
- [ ] ≥ 1 food/beverage manufacturing scheduling paper (optional — not yet found)

> **Key methodological references:**
> - **Maravelias & Grossmann (2003):** STN MILP backbone for inventory-coupled batch scheduling — strongest formulation template
> - **Vin & Ierapetritou (2000):** Direct template for reactive MILP rescheduling under breakdowns
> - **Hakimzadeh & Zandieh (2012):** Core HFS + SDST + limited buffers structural match
> - **Naderi et al. (2023):** Justifies CP-SAT over MILP for disjunctive scheduling
>
> Full reference details and comparison tables: see `keyref_GPT.md` and `keyref_Perplexity.md` in project files.

---

## 12. Confirmed Design Decisions

| Decision | Detail | Status |
|---|---|---|
| Physical model | GC silos finite (L1: PSC/40, NDG/10, BUS/10; L2: PSC/40), RC aggregate batch counter, pipeline consume (3 min) + restock (15 min), SKU-dependent roast times (PSC=15, NDG=17, BUS=18) | ✅ |
| RC buffer capacity | $\overline{B}_l = 40$ batches per line (20,000 kg / 500 kg per batch) | ✅ |
| RC safety threshold | $\theta^{SS} = 20$ batches (half of max_buffer). Idle penalty active below this. | ✅ |
| Pipeline consume timing | Concurrent with roast start (3 min pipeline busy during first 3 of p_k min roasting). Restock blocks pipeline for 15 min. | ✅ |
| Initial SKU state | $\text{last\_sku} = k^{PSC}$ for all roasters at shift start. Setup needed for first MTO batch. | ✅ |
| End-of-shift | $s_b \leq 465$ — batch must complete within 480-slot shift | ✅ |
| Planned downtime | No mid-batch pause. Batch must complete before downtime. No start if cannot finish. | ✅ |
| UPS behavior | Batch cancelled entirely. GC lost. Must restart (new consume + full roast). Scheduler decides whether to restart. | ✅ |
| UPS on IDLE/SETUP roaster | Roaster goes DOWN. If SETUP, timer resets — must re-setup after UPS ends. | ✅ |
| NDG/Busta RC output | Delivered directly — does NOT enter RC stock | ✅ |
| RC SKU mixing | Not an issue — only 1 PSC SKU per shift, no PSC-PSC switch | ✅ |
| R3 routing | Decision variable per batch; baked into DRL action space (21 actions including restock); also experimental factor (fixed vs. flexible) | ✅ |
| **Objective** | **Maximize profit ($)** — revenue minus costs. See `cost.md` for full structure. | ✅ |
| Revenue | PSC $4,000/batch, NDG $7,000/batch, Busta $7,000/batch | ✅ |
| Tardiness penalty | $c^{tard} = \$1{,}000$/min late (MTO jobs past slot 240) | ✅ |
| Stockout penalty | $c^{stock} = \$1{,}500$/event (per consumption event with $B_l < 0$, strictly negative). $B_l = 0$ is NOT stockout. **Event-based, not per-minute.** | ✅ |
| Safety-idle penalty | $c^{idle} = \$200$/min/roaster (idle when $B_l < 20$, not DOWN) | ✅ |
| Overflow-idle penalty | $c^{over} = \$50$/min/roaster (forced idle at $B_l = 40$). **R3: only when both lines = 40.** | ✅ |
| Overflow-idle in det. mode | Included in deterministic objective (solver paces production to avoid overflow) | ✅ |
| Stockout handling | Hard constraint in deterministic mode; soft penalty ($1,500/event) in reactive mode | ✅ |
| Overflow handling | Hard constraint in all modes (physical impossibility) | ✅ |
| UPS parameters | Synthetic (literature MTBF/MTTR); not calibrated to plant | ✅ |
| Setup time | 5 min between ANY different SKU pair, uniform | ✅ |
| Experimental grid | 5 λ × 3 μ × 3 strategies × 2 R3 modes = 90 cells × 100 reps = 9,000 runs | ✅ |
| MILP benchmark | Same model as CP-SAT, deterministic only, LP bound verification. ~200 runs. | ✅ |
| Strategies (reactive) | Dispatching (baseline) / CP-SAT re-solve / DRL (PPO with MaskablePPO) | ✅ |
| Dispatching spec | Urgency threshold 0.7, most-remaining MTO priority, Busta tie-break, lowest-stock R3 routing | ✅ |
| DRL action space | 21 actions (R3 split into →L1 and →L2, 4 restock actions). Per-roaster decision. Action masking for feasibility. | ✅ |
| CP-SAT re-solve trigger | UPS-only (explicit design choice — not periodic) | ✅ |
| Constraint groups | 19 (C1–C19): activation, eligibility, NoOverlap (p_k), setup, initial SKU, downtime, pipeline, end-of-shift, RC bounds, tardiness, R3 routing, idle detection, GC silo balance, restock pipeline block, shared restock station, restock capacity guard | ✅ |
| Future work candidates | ALNS metaheuristic, rolling horizon decomposition | ✅ |
| Data granularity | Batch-level confirmed — used for deterministic validation | ✅ |
| University / level | Undergraduate, IU-VNU | ✅ |

---

## 13. Draft Sentences Ready for Thesis Writing

Adapt freely — do not copy verbatim.

### Section 1.1 — Background

> "In high-volume food manufacturing, within-shift production scheduling directly determines equipment throughput and downstream process continuity. At Nestlé Trị An, the roasting operation serves a hybrid demand structure: NDG and Busta products are manufactured against firm customer orders (make-to-order), while Pure Soluble Coffee must sustain a continuous downstream packaging process through inventory buffers (make-to-stock). Coordinating both across five roasters and two production lines with a shared physical pipeline constraint constitutes a scheduling problem of substantial complexity."

> "This complexity is compounded by the prevalence of unplanned equipment stoppages (UPS), which are among the most common causes of throughput loss in batch manufacturing operations. When a roaster fails mid-shift, the disruption cascades: the in-progress batch is lost, the pipeline schedule for other roasters on the same line is disrupted, and the downstream inventory buffer begins to deplete. Currently, operators respond to these disruptions based on experience and real-time judgment, without algorithmic support. The absence of a systematic reactive scheduling strategy represents a significant operational gap."

### Section 1.2 — Problem Statement

> "On-site observation during the internship period at Nestlé Trị An indicates that unplanned stoppages are the dominant contributor to throughput loss, with overall production estimated at approximately 18 percent below theoretical capacity. While operators produce reasonable schedules under normal conditions, their response to disruptions is ad-hoc and uncoordinated — a limitation inherent to managing five roasters across two coupled production lines under real-time pressure."

> "This thesis addresses the question: when an unplanned stoppage occurs during a production shift, what is the best strategy for re-scheduling the remaining production horizon? Three fundamentally different approaches are evaluated: a priority-based dispatching heuristic representing current operator practice, an optimization-based re-planning approach using Constraint Programming (CP-SAT), and a learning-based approach using Deep Reinforcement Learning (DRL) trained on simulated disruption scenarios."

### Section 1.3 — Objectives

> "The primary objective is to develop and evaluate a reactive scheduling framework for within-shift batch roasting at Nestlé Trị An that systematically responds to unplanned equipment stoppages. The objective function maximizes shift profit — total revenue from completed batches minus penalties for MTO tardiness, RC stockout events, and roaster idle time under low inventory — all expressed in a common monetary unit to enable direct comparison across strategies and to serve as the reward signal for the DRL agent."

> "The framework is evaluated through four reactive strategies spanning a complexity spectrum — dispatching heuristic (no learning), tabular Q-Learning (simple RL), MaskedPPO (end-to-end DRL), and RL-based Hyper-Heuristic (tool-based DRL) — compared across a factorial experimental design varying disruption frequency and duration. MILP and CP-SAT establish the deterministic performance ceiling. The expected contribution is a systematic identification of the conditions under which each reactive strategy dominates, and whether added complexity (rules → tabular RL → deep RL → structured RL) yields proportional performance gains."

### Section 1.4 — Scope

> "This study is bounded to the within-shift scheduling problem for a single eight-hour shift. The physical model is deliberately simplified — Green Coffee supply is treated as unlimited, Roasted Coffee inventory is tracked as an aggregate batch counter per line, and the pipeline constraint covers consume operations only — to focus analytical depth on the reactive scheduling methodology rather than on the physical constraint modeling."

> "Unplanned stoppages are modeled with exponential inter-arrival times and parameterized durations, based on literature estimates rather than plant-specific calibration. The sensitivity analysis covers a range of disruption intensities (0 to 5 events per shift, 10 to 30 minutes mean duration) to compensate for the lack of calibrated parameters. This is acknowledged as a limitation — results indicate relative strategy performance, not absolute predictions for the Nestlé Trị An facility."

### Section 2.1 — Company Overview

> "Nestlé Trị An is a manufacturing facility of Nestlé Vietnam, located in Trị An, Đồng Nai province. The facility operates a batch coffee roasting stage producing three product families: Pure Soluble Coffee (PSC), Nescafé Dolce Gusto (NDG) capsule coffee, and Busta products for the Starbucks brand. PSC constitutes the primary output, shipping approximately 17 to 20 tons of pure coffee powder per eight-hour shift as an intermediate product to downstream Nestlé facilities."

> "The roasting floor is staffed by three to four operators and one contractor per shift, who collectively manage five roasters across two parallel production lines. Each line has a shared Green Coffee pipeline that serves all roasters on that line — creating a physical coupling that makes scheduling decisions interdependent across roasters on the same line. This constraint structure, combined with the small crew size and the frequency of unplanned equipment failures, motivates the need for systematic reactive scheduling support."

---

## 14. Open Items Tracker

| Item | Owner | Status | Blocking? |
|---|---|---|---|
| Instructor approval of scope | Student | Memo drafted (revised_scope_memo.docx), needs meeting | ⚠️ **Blocks everything** |
| Literature search — reactive scheduling + RL references | Student | ✅ **16+ references identified** from keyref_GPT.md and keyref_Perplexity.md. Need to read full papers and write Ch 2. | ⚠️ Blocks Ch 2 writing |
| RC max_buffer calculation | — | ✅ **Resolved: 40 batches** (20,000 kg / 500 kg per batch) | ✅ Done |
| Initial last_sku at shift start | — | ✅ **Resolved: PSC** for all roasters | ✅ Done |
| Stockout definition | — | ✅ **Resolved: event-based** ($1,500 per consumption event with $B_l < 0$, strictly negative). $B_l = 0$ is not stockout. Duration ($B_l \leq 0$) is KPI only. | ✅ Done |
| Overflow-idle in deterministic mode | — | ✅ **Resolved: included** in deterministic objective | ✅ Done |
| R3 overflow-idle encoding | — | ✅ **Resolved: both lines must be full** for R3 overflow-idle | ✅ Done |
| Cost structure | — | ✅ **Resolved:** cost.md created. Revenue $4k/$7k, penalties $1.5k/$1k/$200/$50. | ✅ Done |
| MILP role | — | ✅ **Resolved:** deterministic benchmark, LP bound verification, not reactive | ✅ Done |
| Dispatching heuristic spec | — | ✅ **Resolved:** urgency threshold 0.7, most-remaining priority, R3 lowest-stock routing | ✅ Done |
| DRL action space | — | ✅ **Resolved:** 21 actions (R3 routing + 4 restock actions), per-roaster decision | ✅ Done |
| End-of-shift constraint | — | ✅ **Resolved: $s_b \leq 465$** | ✅ Done |
| Old weight notation purge | — | ✅ **Resolved:** All $w^{tard}$, $w^{stock}$ replaced with $c^{tard}$, $c^{stock}$ | ✅ Done |
| UPS MTBF/MTTR literature values | Student | Need to find 2–3 reference values from the identified papers | ⚠️ Blocks experimental parametrization |
| Implementation: simulation environment | Not started | — | Blocks all experiments |

### Document inventory ✅

| # | Document | Purpose | Status |
|---|----------|---------|--------|
| 1 | `revised_scope_memo.docx` | Instructor approval | Ready for meeting |
| 2 | `Thesis_Problem_Description_v2.md` | Complete problem description (bilingual VN/EN) | ✅ Updated: MILP, profit objective, dispatching, 17-action DRL |
| 3 | `mathematical_model_complete.md` | Math formulation + I/O | ✅ Updated: 15 constraints, profit objective, MILP benchmark, cost.md refs |
| 4 | `cost.md` | Cost structure (single source of truth for $) | ✅ NEW: revenue, penalties, DRL reward, breakeven analysis |
| 5 | `Surrounding_information_introduction_v2.md` | Writing reference for Ch 1–6 | ✅ Updated: MILP, literature, cost references, resolved items |
| 6 | `event_simulation_logic_complete.md` | Simulation loop spec | ✅ Updated: profit-based reward, 21 actions, GC silo tracking, restock logic, decrement-first timing, needs_decision flag |
| 7 | `audit_resolution.md` | All open items resolved | ✅ Updated |
| 8 | `keyref_GPT.md` | 10 key references from GPT search | Reference file |
| 9 | `keyref_Perplexity.md` | 10 key references from Perplexity search | Reference file |

---

*Next action: student brings scope memo to instructor for approval, then begins implementation (simulation environment → CP-SAT deterministic model → MILP benchmark → dispatching heuristic → UPS simulation → DRL agent → experiments).*
