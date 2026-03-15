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

> **A systematic comparative study of three reactive scheduling strategies (dispatching heuristic, CP-SAT re-optimization, and DRL policy) for batch roasting with shared pipeline constraints under unplanned stoppages, evaluated on a factorial experimental design across disruption intensities, and grounded in the physical system of Nestlé Trị An.**

Novelty is in the *systematic comparison of fundamentally different reactive approaches* on a *real industrial constraint structure* with *controlled stochastic disruptions*. Not in algorithmic invention.

### Supporting differentiating elements

1. **Pipeline mutex as the propagating constraint:** The shared GC pipeline (60% utilization on Line 2) means a UPS on one roaster cascades into pipeline conflicts for other roasters. This cascading effect is what makes reactive scheduling non-trivial — and what distinguishes this from generic parallel machine rescheduling.

2. **Cross-line R3 routing as reactive tactical decision:** When Line 1 loses throughput due to a UPS on R1 or R2, the scheduler can route R3 output to Line 1 as compensation — but at the cost of Line 2's pipeline capacity. This cross-line tradeoff under disruption is tested as an experimental factor (fixed vs. flexible R3 routing).

3. **DRL vs. CP-SAT under increasing disruption intensity:** The hypothesis is that DRL's ability to learn distributional patterns in UPS events gives it an advantage over CP-SAT (which treats each re-solve as a fresh deterministic problem) under high disruption rates, while CP-SAT dominates under low disruption (where its optimality guarantee matters more than learned heuristics).

### What the contribution is NOT

- Not a new scheduling algorithm or solver.
- Not a complexity proof.
- Not a real-time deployed production tool.
- Not an RL algorithm contribution (PPO is used as-is from Stable-Baselines3).
- Not a claim that any strategy is universally superior — the contribution is identifying **when each strategy dominates**.

---

## 6. Methodology Architecture (for Sections 3.1 and 3.2)

### Three-strategy comparative approach

| Strategy | Type | Implementation | Role in thesis |
|---|---|---|---|
| Dispatching heuristic | Rule-based | Priority rules (EDD for MTO, fill-PSC for stock) | Baseline — represents current operator practice |
| CP-SAT re-solve | Optimization-based | Google OR-Tools CP-SAT (Python API) | Optimization-based reactive strategy |
| DRL policy (PPO) | Learning-based | Stable-Baselines3 MaskablePPO (Python) | Learning-based reactive strategy |

### Why these three (academic justification)

The three strategies represent three fundamentally different paradigms for decision-making under uncertainty:

- **Dispatching heuristic:** Zero computation, myopic (no lookahead), zero learning. This is the "null hypothesis" — what happens when you apply simple rules without optimization. It represents current operator practice in simplified form.
- **CP-SAT re-solve:** Full optimization at each decision point, but **stateless** (each re-solve is independent, no memory of past disruptions). Strong on single-instance optimality, but computationally heavier and cannot learn patterns.
- **DRL policy:** Pre-trained on thousands of simulated disruption scenarios. Fast inference (instantaneous at decision time), **stateful** (learns distributional patterns), but no optimality guarantee and black-box decisions.

The research question: **under what disruption conditions does each strategy dominate, and why?**

### Comparison metrics ✅

| Metric | Definition | Unit |
|---|---|---|
| **Total throughput** | PSC batches completed per shift | batches |
| **Stockout count** | Consumption events where $B_l \leq 0$ | events |
| **Stockout duration** | Total minutes with $B_l \leq 0$ | minutes |
| **MTO tardiness** | $\sum_j \text{tard}_j$ | minutes |
| **Computation time** | Wall-clock time per re-solve / per RL inference | seconds |

### Baseline

The dispatching heuristic IS the baseline. All three strategies are compared against each other on the same simulated shift instances with the same UPS realizations (paired comparison). The relevant questions: **how much does CP-SAT improve over dispatching? How much does DRL improve? Under what conditions?**

The 21 historical shift instances from Nestlé Trị An are used for **deterministic model validation** (verify that the CP-SAT schedule is feasible and produces reasonable throughput on known shift data). They are not the primary experimental dataset. The primary experiments use simulated shifts with controlled UPS injection.

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
| Methods compared | Dispatching + CP-SAT re-solve + DRL (PPO) | MILP, metaheuristics (ALNS documented as future work) |
| Uncertainty | Unplanned stoppages (UPS) — stochastic | Stochastic demand, yield uncertainty |
| Deliverable | Methodology comparison + simulation framework | Deployed production tool, dashboard |
| Data | Real parameters for model calibration; simulated UPS for experiments | Real-time sensor integration |
| Rescheduling | Reactive (event-triggered re-solve / DRL action) | Proactive robust scheduling, rolling horizon |
| GC supply | Unlimited (abstracted away) | GC silo inventory, dump, replenish, SKU purity |
| RC tracking | Aggregate batch counter per line | Individual RC silo tracking, fill/draw rules |

### All acknowledged limitations ✅

| ID | Limitation | Implication if violated |
|---|---|---|
| L1 | Uniform 15-min roasting time for all SKUs | Real times vary 13–16 min; pipeline utilization changes slightly |
| L2 | Uniform 5-min setup time for all SKU pairs | Asymmetric setup times would change optimal sequencing |
| L3 | GC supply unlimited — no silo inventory tracking | Acceptable given thesis focus on scheduling methodology, not silo optimization |
| L4 | PSC consumption rate constant throughout shift | Rate changes (line speed adjustment) not captured |
| L5 | UPS parameters synthetic — not calibrated to plant data | Results show relative strategy performance, not absolute plant-specific predictions |
| L6 | UPS affects exactly 1 roaster at a time | Correlated failures (power outage) not modeled |
| L7 | UPS cancels in-progress batch entirely (no pause/resume) | Simplifies state tracking; slightly pessimistic vs. reality if partial recovery is possible |
| L8 | RC tracked as aggregate counter, not individual silos | Acceptable — only 1 PSC SKU per shift, no mixing concern |
| L9 | No cross-shift state carryover | Each shift is independent problem instance |
| L10 | No shelf life / FIFO constraints | Acceptable within 8-hour horizon |
| L11 | Observational figures (18% throughput gap, 300 min stoppage) from limited samples | Used as motivation only; not statistically representative |
| L12 | DRL agent trained on synthetic UPS — transferability to real plant unknown | Would require retraining with real MTBF/MTTR data for deployment |

---

## 9. Design Requirements (for Section 1.3)

The system must satisfy all of the following:

1. Generate a feasible predictive schedule for a full 8-hour shift with no constraint violations (pipeline NoOverlap, RC inventory bounds, setup times, downtime avoidance).
2. React to UPS events: when an unplanned stoppage occurs, produce a revised schedule for the remaining horizon within practically acceptable time (target: < 1 second for CP-SAT re-solve; < 1 millisecond for DRL inference).
3. Track RC inventory in batch units for both lines and prevent stockout (deterministic mode) or minimize stockout impact (reactive mode).
4. Handle MTO batch scheduling with soft due-date constraint at slot 240.
5. Support R3 cross-line output routing as a decision variable (and as experimental factor: fixed vs. flexible).
6. Enable controlled experimental comparison: same UPS realizations applied to all three strategies for fair paired comparison.
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

### Four pillars

**Pillar 1 — Parallel Machine Scheduling with Shared Resources**
- Theoretical grounding for the pipeline mutex constraint
- Key reference: Bektur & Saraç (2019) — common-server parallel machine scheduling with sequence-dependent setup times and machine eligibility
- Broader: SDST surveys (Allahverdi 2015), unrelated parallel machines
- Target: papers modeling shared transfer/pipeline resources in batch scheduling

**Pillar 2 — Scheduling Under Uncertainty / Reactive Rescheduling**
- Central pillar — the core methodological question of the thesis
- Surveys: Vieira et al. (2003) — rescheduling literature review; Herroelen & Leus (2005) — robust/reactive project scheduling; Ouelhadj & Petrovic (2009) — dynamic scheduling in manufacturing
- Reactive strategies taxonomy: complete rescheduling, match-up scheduling, right-shift rescheduling
- Event-triggered vs. periodic re-planning
- Target 2020–2025: reactive scheduling in manufacturing using CP or RL

**Pillar 3 — Reinforcement Learning for Production Scheduling**
- One of two primary methods in the thesis
- Foundational: Zhang et al. (2020) — RL for job shop scheduling; Park et al. (2021) — RL with graph neural networks for scheduling
- Action masking: Huang & Ontañón (2022) — invalid action masking for RL
- Target 2020–2025: RL applied to parallel machine or batch scheduling, especially with stochastic disruptions
- MaskablePPO: Stable-Baselines3 documentation + any papers using it for constrained scheduling

**Pillar 4 — CP-SAT for Industrial Scheduling**
- One of two primary methods in the thesis
- Laborie et al. (2018) — CP Optimizer interval variables and NoOverlap
- Naderi et al. (2023) — CP vs. MIP for scheduling (empirical evidence of CP advantage on disjunctive structures)
- Target: CP-SAT or OR-Tools applied to reactive/online scheduling contexts

### Checklist

- [ ] ≥ 10 references total
- [ ] ≥ 50% published 2020–2025
- [ ] ≥ 1–2 reactive/dynamic scheduling surveys
- [ ] ≥ 1–2 RL for scheduling papers (ideally with action masking or stochastic disruptions)
- [ ] ≥ 1 paper using CP-SAT or OR-Tools for scheduling
- [ ] ≥ 1 SDST scheduling survey or key reference
- [ ] ≥ 1 paper on scheduling under equipment failures/breakdowns
- [ ] ≥ 1 food/beverage manufacturing scheduling paper (optional but nice)

> ⚠️ **Literature search needs to be conducted.** Current confirmed reference: Bektur & Saraç (2019) for pipeline/server constraint. Reactive scheduling and RL references need to be found and evaluated.

---

## 12. Confirmed Design Decisions

| Decision | Detail | Status |
|---|---|---|
| Physical model | GC unlimited, RC aggregate batch counter, pipeline consume-only (3 min), fixed 15 min roast | ✅ |
| Pipeline consume timing | Concurrent with roast start (3 min pipeline busy during first 3 of 15 min roasting) | ✅ |
| Planned downtime | No mid-batch pause. Batch must complete before downtime. No start if cannot finish. | ✅ |
| UPS behavior | Batch cancelled entirely. GC lost. Must restart (new consume + full roast). Scheduler decides whether to restart. | ✅ |
| UPS on IDLE/SETUP roaster | Roaster goes DOWN. If SETUP, timer resets — must re-setup after UPS ends. | ✅ |
| NDG/Busta RC output | Delivered directly — does NOT enter RC stock | ✅ |
| RC SKU mixing | Not an issue — only 1 PSC SKU per shift, no PSC-PSC switch | ✅ |
| R3 routing | Decision variable per batch; also experimental factor (fixed vs. flexible) | ✅ |
| Objective | Maximize PSC throughput − tardiness penalty | ✅ |
| Stockout handling | Hard constraint in deterministic mode; soft penalty in reactive mode | ✅ |
| Overflow handling | Hard constraint in all modes (physical impossibility) | ✅ |
| UPS parameters | Synthetic (literature MTBF/MTTR); not calibrated to plant | ✅ |
| Setup time | 5 min between ANY different SKU pair, uniform | ✅ |
| Experimental grid | 5 λ × 3 μ × 3 strategies × 2 R3 modes = 90 cells × 100 reps = 9,000 runs | ✅ |
| Strategies | Dispatching (baseline) / CP-SAT re-solve / DRL (PPO with MaskablePPO) | ✅ |
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

> "The primary objective is to develop and evaluate a reactive scheduling framework for within-shift batch roasting at Nestlé Trị An that systematically responds to unplanned equipment stoppages. The framework is evaluated through three competing strategies — dispatching heuristic, CP-SAT re-optimization, and DRL policy — compared across a factorial experimental design varying disruption frequency and duration."

> "The expected contribution is a systematic identification of the conditions under which each reactive strategy dominates: specifically, whether the DRL policy's ability to learn disruption patterns provides a measurable advantage over CP-SAT's instance-optimal re-solving under high disruption intensity, and whether either approach justifies the implementation complexity over simple dispatching rules."

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
| Instructor approval of scope | Student | Memo drafted, needs meeting | ⚠️ **Blocks everything** |
| Literature search — reactive scheduling + RL references | Student | Not yet started | ⚠️ Blocks Ch 2 writing |
| RC max_buffer calculation (batch units) | Student | Need to compute from factory kg data | ⚠️ Blocks formulation finalization |
| UPS MTBF/MTTR literature values | Student | Need to find 2–3 reference values | ⚠️ Blocks experimental design parametrization |
| Formulation update for downtime/UPS clarifications | Pending | Minor corrections to simplified_formulation_v1.md | Not blocking |
| Implementation: simulation environment | Not started | — | Blocks all experiments |

---

*Next action: student brings scope memo to instructor for approval, then begins implementation (simulation environment → CP-SAT model → dispatching heuristic → UPS simulation → DRL agent → experiments).*
