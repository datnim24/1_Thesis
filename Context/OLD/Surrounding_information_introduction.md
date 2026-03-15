# Surrounding Information & Introduction Reference
## Thesis: Integrated Batch Roasting Scheduling with Green Coffee Silo Constraints — A Comparative Study of MILP and Constraint Programming at Nestlé Trị An

> **Purpose of this file:** Living reference document synthesizing all gathered contextual knowledge — from Q&A sessions, problem description, and mathematical formulation — to guide writing of Chapters 1, 2, and 3. Every claim is traceable. Items marked ⚠️ remain open. Items marked ✅ are resolved and ready to write from.
>
> **Last updated:** Session 3 — all critical flags resolved (Q1–Q7); thesis title proposed; data granularity confirmed; company details enriched.

---

## 1. Thesis Identity

| Item | Value |
|---|---|
| **University** | Ho Chi Minh City International University, Vietnam National University (IU-VNU) |
| **Level** | Undergraduate thesis |
| **Standard** | Full academic rigor — must be defensible before an examination committee |
| **Primary framing** | Academic model validated on real industrial data. The prototype is a by-product, not the deliverable. |
| **Case company** | Nestlé Trị An Manufacturing Facility, Đồng Nai Province, Vietnam |
| **Working title** | **Integrated Batch Roasting Scheduling with Green Coffee Silo Constraints: A Comparative Study of MILP and Constraint Programming at Nestlé Trị An** |

> **Title rationale:** The title names the three differentiating elements: (1) "Integrated Batch Roasting Scheduling" — the combined MTO+MTS scheduling problem, (2) "Green Coffee Silo Constraints" — the distinguishing physical bottleneck (shared pipeline, SKU purity, variable-duration dumps), and (3) "Comparative Study of MILP and Constraint Programming" — the two-method experimental design. "Nestlé Trị An" grounds it as an industrial case study. The title avoids solver brand names (CPLEX, CP-SAT) which belong in the body text, not the title.

> **Tone implication:** Despite being undergraduate, the thesis must be written to applied OR/IE research standard. Claims must be precise, limitations explicit, and the MILP vs. CP-SAT comparison methodologically rigorous.

---

## 2. Company Background (for Section 2.1)

### What can be stated ✅ (enriched Session 3)

- **Company:** Nestlé Trị An — a manufacturing facility of Nestlé Vietnam, located in Trị An, Đồng Nai province.
- **Product focus:** Roasted coffee products, including:
  - **PSC (Pure Soluble Coffee):** The primary high-volume product. PSC is the purest coffee powder — an intermediate product shipped as 500 kg bags to Đồng Nai Factory or Thailand Nescafé plant for manufacturing into finished Nescafé products. PSC is the make-to-stock (MTS) product that drives continuous downstream demand.
  - **NDG (Nescafé Dolce Gusto):** Capsule coffee product. Make-to-order (MTO) with firm delivery windows.
  - **Busta:** A dedicated roasting line for Starbucks products. Make-to-order (MTO) with firm delivery windows.
- **Throughput (disclosable):** PSC ships approximately **17–20 tons of pure coffee powder per 8-hour shift**. Output figures for NDG and Busta lines are classified.
- **Position in network:** One of a small number of Nestlé plants in Vietnam operating coffee roasting at this scale. Not the sole facility.
- **Production structure:** Two parallel production lines (Line 1, Line 2), each with dedicated GC silos, RC silos, and roasters. Five roasters total (R1–R5). R3 is a physical cross-line bridge — draws GC from Line 2 but can route RC output to either line.
- **Staffing on the roasting floor:** **3–4 operators plus 1 contractor** per shift manage the roasting stage. This small crew size underscores why real-time within-shift decisions rely heavily on individual operator experience — and why algorithmic decision support could reduce cognitive burden.
- **Shift structure:** 3 shifts per day, 8 hours each, continuous operation.
- **Planning cycle:** Head Office issues demand (POs) → Supply Chain distributes to lines → supervisors hold daily planning meetings (~1 hour) considering line conditions, maintenance windows, and cross-department requests (fire testing, compressed air checks, exhaust air tests, etc.) → weekly plan updated daily.

### What cannot be stated (confidentiality / data access)

- Exact NDG and Busta production volumes (tons/day or tons/shift) — **classified**.
- Exact BOM ratios — **anonymized** (GC SKU names changed; proportions are real system values).
- Specific VND financial figures from internal P&L — **not available**. Penalty weights in the model are proxy costs, not audited actuals.

### Framing guidance for Section 2.1

Write as: *"a high-volume, continuous-process food manufacturing plant operating a batch roasting stage with a hybrid demand structure (MTO and MTS) and tightly coupled physical infrastructure constraints."* PSC throughput (17–20 tons/shift) can be stated. Emphasize system complexity: 5 roasters × 2 lines × 16 GC silos × 8 RC silos × 1 shared pipeline per line × 3–4 operators = the justification for formal optimization.

---

## 3. Current Practice (for Section 1.1)

### How within-shift scheduling is done today

The current process is **human-driven and experience-based** at the execution level:

1. **Strategic input:** PO demand data from Head Office → distributed to lines by Factory Supply Chain.
2. **Daily meeting (~1 hour):** Supervisors collectively plan considering line conditions, planned stoppages, and cross-department needs.
3. **Execution layer (within the shift):** Detailed decisions — which roaster runs next, which silo to draw from, when to replenish, how to manage RC buffer levels — are made **in real time by operators** based on experience and direct observation. No algorithmic decision support exists at this layer. The crew of 3–4 operators plus 1 contractor manages all five roasters, both lines' silo systems, and all GC pipeline operations simultaneously.
4. **Update cycle:** Weekly plan revised daily.

### Why this thesis does NOT replace the current system

> The thesis targets the **within-shift scheduling layer only** — roaster assignment, batch sequencing, GC silo lifecycle management, and RC buffer control within one 8-hour shift. The strategic planning layer (PO allocation, weekly scheduling, cross-department coordination) remains entirely human-managed.

The deterministic model cannot handle **unplanned stoppages (UPS)**, which are frequent and cascade unpredictably. This is a deliberate, defensible scope boundary — not a weakness.

### The gap this thesis fills

Supervisors produce a plan at the PO/line level. But the granular within-shift decisions — batch sequencing per roaster, pipeline operation timing, GC silo reassignment sequencing, RC routing for R3 — are left to real-time operator judgment. This is the unoptimized layer. This thesis provides the first formal optimization model for this layer.

---

## 4. Operational Pain Points (for Section 1.2)

### Observed figures

All values below are **rough estimates from on-site observation, interviews, and manual timing of 1–2 sample events during the internship period.** Not statistically validated. No additional data points available beyond these initial observations. Present as qualitative motivation only — use hedged language throughout ("observation suggests," "interviews indicate," "estimated at approximately").

| Observation | Rough Figure | Root cause framing |
|---|---|---|
| Unplanned GC dumps | ~3 per week | GC silo SKU switch not coordinated with batch sequence; dump forced to clear silo for incoming SKU |
| Handling time per dump event | ~2 hours (setup, crew, cleanup) | Actual dump: 5–15 min; overhead dominates — indirect cost of uncoordinated silo management |
| Overall speed loss vs. theoretical | ~18% of theoretical throughput | Dominated by UPS; scheduling-induced stops are a meaningful but unquantified subset |
| Total roaster stoppage from RC high-stock | ~300 minutes per observed period | RC silos filling up halts roasters even when GC is available — MTS not coordinated with roaster capacity |

### Academic gap statement (close of Section 1.2)

> No existing within-shift scheduling tool simultaneously handles: **(1)** make-to-order batch sequencing with hard due dates, **(2)** make-to-stock replenishment maintaining a continuous downstream process, and **(3)** shared-resource GC silo lifecycle management — within a single integrated optimization model, in the context of batch coffee roasting operations.

---

## 5. Academic Contribution (for Section 1.2 and 2.2)

### Primary claim ✅

> **The combined formulation of MTO batch scheduling + MTS inventory replenishment + multi-silo GC lifecycle management as a single CP model, validated on real industrial data from a coffee roasting facility, and benchmarked against a MILP formulation solved with CPLEX.**

Novelty is in the *integration*, the *physical constraint modeling* (shared pipeline), and the *real-data validation* — not in algorithmic invention.

### Supporting differentiating elements

1. **Shared GC pipeline as mutual exclusion:** All GC operations (batch consume, replenish, variable-duration dump) compete for a single physical pipeline per line. Modeled as `NoOverlap` over heterogeneous interval variables. Not found in prior scheduling literature reviewed.

2. **Cross-line roaster (R3) as per-batch binary routing decision:** $\pi_j \in \{0,1\}$ routes RC output to L1 or L2 with no time cost — a cross-line inventory coupling absent from standard parallel-machine models.

3. **SKU switch and changeover pre-conditions as hard constraints:** RC depletion thresholds ($\leq 15{,}000$ kg before switch/changeover) and PSC consumption delay (120 slots after changeover) couple batch sequencing to a future inventory state.

### What the contribution is NOT

- Not a new algorithm — CP-SAT and CPLEX used as-is.
- Not a new complexity proof.
- Not a real-time rescheduling system.
- Not a deployment-ready production tool.

---

## 6. Methodology Architecture (for Sections 3.1 and 3.2) ✅

### Two-model comparative approach

| Phase | Formulation | Solver | Role in thesis |
|---|---|---|---|
| Phase 1 | MILP | IBM ILOG CPLEX (Python API) | Industry-standard baseline; rigorous reference point; expected by reviewers |
| Phase 2 | CP (Constraint Program) | Google OR-Tools CP-SAT (Python) | Proposed method; expected to outperform MILP |
| Comparison | — | Both | Formal 3-metric evaluation on 21 historical instances |

### Why MILP first (academic justification)

MILP is the classical approach for production scheduling and the established baseline in the OR/IE literature. Formulating the problem as a MILP first:
- Establishes a theoretically grounded reference (LP relaxation lower bound)
- Demonstrates command of the classical method
- Creates the necessary comparison anchor to justify choosing CP-SAT
- Is expected by IU-VNU examination committees in an IE thesis

### Why CP-SAT is expected to outperform

The core difficulty is disjunctive: `NoOverlap` constraints over batch intervals (per roaster) and pipeline intervals (per line). MIP LP relaxations are weak on disjunctive constraints — fractional solutions are far from feasible integer solutions, so the branch-and-bound tree is enormous. CP-SAT's dedicated interval propagators prune the search space far more effectively on this specific constraint structure.

This is not a claim that CP-SAT always beats MILP — it is a hypothesis grounded in the structure of this specific problem, tested empirically across 21 instances.

### Comparison metrics ✅ (confirmed)

| Metric | Definition | Unit |
|---|---|---|
| **Total penalty** | Objective value $Z$ at solver termination | VND |
| **Solve time** | Wall-clock time from problem load to best solution | seconds |
| **Scalability** | Degradation of penalty and solve time as instance size grows (e.g., more PSC batches) | curve / table |

> Optimality gap (% gap at time limit) is a natural supplementary metric for the sensitivity analysis chapter but is not part of the confirmed primary 3-metric comparison.

### Baseline for comparison (penalty scoring) ✅ (confirmed — batch-level data available)

**Primary plan (confirmed):** Reconstruct the "current practice" schedule from historical **batch-level** records for the best-performing week (minimum UPS). Compute $Z$ — the model's objective function — applied to that historical schedule. This is the **baseline penalty**.

The optimized model's $Z$ on the same shift instance is then compared directly.

> **Critical framing:** This comparison does not claim the model would have prevented real-world UPS events. It shows that, given the same starting conditions and known demand, the model finds a schedule with lower $Z$ — meaning better coordination of the three sub-problems, absent disruptions.

> **Contingency (no longer needed):** ~~If the system only logs aggregated shift-level totals, the baseline is instead a synthetic greedy heuristic schedule.~~ Data granularity confirmed as batch-level (Session 3, Q3). The primary plan is active. Limitation L9 is retired.

---

## 7. Validation Dataset (for Section 1.4) ✅

| Parameter | Value |
|---|---|
| **Source** | Nestlé Trị An factory production information system |
| **Period** | One full week — the week with lowest observed UPS frequency |
| **Instances** | 21 shifts (3 shifts/day × 7 days) |
| **Shift types** | All three present: normal, SKU-switch, changeover |
| **Access** | Formally extracted with approval of internship supervisor and Line Manager |
| **Anonymization** | GC SKU identifiers changed; all quantitative parameters (rates, sizes, times) are real values |
| **Granularity** | ✅ **Batch-level** — individual batch start times, roaster assignments, silo draws available (confirmed Session 3) |

21 instances covering all three shift types enables:
- Stratified analysis by shift type
- Aggregate statistics (mean, min, max penalty; mean solve time) across MILP and CP-SAT
- Scalability observation across instances of varying complexity
- Direct reconstruction of historical schedule as baseline for penalty comparison

---

## 8. Scope and Limitations (for Section 1.4) ✅

### Scope boundaries

| Dimension | In scope | Out of scope |
|---|---|---|
| Planning horizon | Single 8-hour shift (480 min slots) | Multi-shift, weekly, rolling horizon |
| Shift types | Normal, SKU-switch, changeover | — |
| Lines | Both L1 and L2 with R3 cross-line | — |
| Methods compared | MILP (CPLEX) + CP (CP-SAT) | Metaheuristics, simulation, hybrid methods |
| Deliverable | Mathematical formulation + prototype | Deployed production tool, dashboard |
| Data | Real historical data, 21 shifts, batch-level granularity | Real-time sensor integration |
| Rescheduling | Offline pre-shift planning | Real-time intra-shift rescheduling |
| Cross-shift state | None — each shift solved independently | Rolling horizon, carryover inventory |

### All acknowledged limitations ✅ (complete list, updated Session 3)

| ID | Limitation | Implication if violated |
|---|---|---|
| L1 | Fully deterministic — no breakdowns, no demand uncertainty, no UPS | Real shifts will deviate; human override required |
| L2 | Uniform 5-min setup time — no SKU-pair setup matrix | May under/over-estimate some transitions |
| L3 | RC consumption uses highest-stock heuristic, not true FIFO-by-fill-time | Minor deviation from actual consumption order |
| L4 | No shelf life / date-based FIFO | Acceptable within 8-hour horizon |
| L5 | Offline planning only — no real-time rescheduling | Schedule drifts from reality as UPS occur |
| L6 | RC fill rule simplified — lowest-stock normal silo first | Minor deviation from exact plant rule in edge cases |
| L7 | No cross-shift state carryover | Shift-start silo states must be re-supplied per shift |
| L8 | Nestlé's actual system is far more complex | Model captures a well-defined, operationally relevant subset |
| ~~L9~~ | ~~Baseline penalty may use synthetic greedy schedule if batch-level data unavailable~~ | **Retired** — batch-level data confirmed (Session 3) |
| L10 | Observational figures (3 dumps/week, 300 min stop) from 1–2 samples only; no additional data available | Used as motivation only; not statistically representative |

---

## 9. Design Requirements (for Section 1.3) ✅

The model must satisfy all of the following to be considered valid:

1. Handle both MTO (NDG, Busta with due dates ≤ slot 240) and PSC (MTS) jobs in a single 8-hour shift.
2. Enforce GC pipeline mutual exclusion: at most one operation (consume / replenish / dump) per line per time slot.
3. Enforce GC silo SKU purity: no mixing, empty-before-reassign.
4. Support all three shift types: normal, SKU-switch, changeover.
5. Produce a feasible schedule (zero constraint violations) — infeasibility is a model failure.
6. Accept real production parameters as input: BOM, processing times, batch sizes, silo initial states, PSC consumption rates, planned downtime.
7. Return a schedule within a practically interpretable solve time (specific limit determined in sensitivity analysis).

---

## 10. Computing Environment (for Section 3.2 and sensitivity analysis)

| Component | Specification |
|---|---|
| Type | VPS (Virtual Private Server), 24/7 operation |
| CPU | Intel Core i3-9100F — 4 cores / 4 threads @ 3.6 GHz base |
| RAM | 16 GB DDR4 @ 2400 MHz |
| GPU | AMD Radeon RX470 4 GB (not used — both solvers are CPU-only) |
| Primary solver | Google OR-Tools CP-SAT (Python API) |
| Baseline solver | IBM ILOG CPLEX (Python API) |
| **Time limit** | **1800 seconds (30 minutes) per instance — primary configuration** |

> **Time limit rationale (Session 3):** 1800s is the primary experimental time limit, chosen to give both solvers sufficient time to converge on instances of this size (~400k–500k variables for MILP). However, for practical deployment contexts where carryover optimization across consecutive shifts would be desired, a tighter limit of **600 seconds** fits real-world operational needs (a supervisor cannot wait 30 minutes between shifts). The sensitivity analysis chapter will report results at both 600s and 1800s cutoffs to characterize the quality-time tradeoff.

> **Writing note:** The i3-9100F is a modest, consumer-grade CPU (2018). Acknowledge this when reporting absolute solve times — results will improve on higher-core machines. The *relative* MILP vs. CP-SAT comparison remains valid since both run on identical hardware.

---

## 11. Literature Framework (for Section 2.2 — to be populated in dedicated session)

### Four pillars

**Pillar 1 — Scheduling with Sequence-Dependent Setup Times (SDST)**
- Theoretical grounding for the 5-min uniform setup constraint
- Classic: Allahverdi et al. (2008, 2015) surveys
- Target 2020–2024: SDST in batch / food manufacturing contexts

**Pillar 2 — MILP vs. CP for Disjunctive Scheduling**
- Justification for the two-model comparison and CP-SAT hypothesis
- CP foundations: Baptiste, Le Pape & Nuijten (2001)
- CP-SAT: Perron & Furnon (OR-Tools documentation, 2023)
- MILP for scheduling: Pochet & Wolsey (2006)
- Target 2020–2024: industrial CP-SAT applications

**Pillar 3 — Food and Beverage Manufacturing Scheduling**
- Sector-specific literature where the gap is demonstrated
- Coffee, beverage, dairy batch scheduling; silo/tank constraint papers
- Expected finding: no paper models per-slot shared pipeline mutex as a scheduling constraint

**Pillar 4 — Integrated MTO/MTS Production Planning**
- Hybrid systems; joint due-date + inventory-driven optimization
- Inventory-integrated scheduling, lot-sizing + sequencing
- Target: papers showing MTO/MTS joint formulation as strictly harder than either alone

### Checklist

- [ ] ≥ 10 references total
- [ ] ≥ 50% published 2020–2024
- [ ] ≥ 1 paper with CPLEX applied to a scheduling problem (MILP baseline anchor)
- [ ] ≥ 1 paper with CP-SAT or OR-Tools applied industrially
- [ ] ≥ 1 SDST scheduling survey
- [ ] ≥ 1–2 food/beverage manufacturing scheduling papers
- [ ] ≥ 1 hybrid MTO/MTS paper

> ⚠️ **Deferred:** Literature search to be conducted in a dedicated session. Student has a partial reference list but it is not yet shared. Section 2.2 will be left as placeholder during initial Chapter 1–3 drafting.

---

## 12. Flag and Q&A Resolution Summary

| Item | Resolution | Status |
|---|---|---|
| F1 — Baseline cost | Penalty $Z$ reconstructed from best-week historical batch-level data | ✅ |
| F2 — Solver time limit | 1800s primary; 600s as practical-deployment sensitivity point | ✅ |
| F3 — Literature list | Deferred to dedicated session; student has partial list not yet shared | ⚠️ Open (non-blocking for Ch 1, 3) |
| F4 — Implementation | MILP via CPLEX → CP-SAT via OR-Tools; both to be implemented and compared | ✅ |
| F5 — Rough observational figures | Presented with explicit hedging; never as measured statistics; no additional data available | ✅ |
| F6 — Framing register | Full academic model; prototype is by-product | ✅ |
| Q-A — MILP solver | IBM ILOG CPLEX (Python API) | ✅ |
| Q-B — Data granularity | **Batch-level confirmed** (Session 3). Contingency plan retired; L9 retired. | ✅ Resolved |
| Q-C — Validation instances | 21 shifts, 1 best week, all 3 shift types | ✅ |
| Q-D — University / level | Undergraduate, Ho Chi Minh International University — VNU | ✅ |
| Q-E — Comparison metrics | Total penalty, Solve time, Scalability | ✅ |
| Q1 — Literature search | Student has partial list; deferred to dedicated session | ✅ Direction set |
| Q2 — Thesis title | **Integrated Batch Roasting Scheduling with Green Coffee Silo Constraints: A Comparative Study of MILP and Constraint Programming at Nestlé Trị An** | ✅ Proposed |
| Q3 — Data granularity | Batch-level confirmed | ✅ Resolved |
| Q4 — Company detail permitted | PSC throughput (17–20 tons/shift), crew size (3–4 + 1 contractor), product descriptions (PSC → 500 kg bags to Đồng Nai/Thailand; NDG = capsule coffee; Busta = Starbucks line). NDG/Busta volumes classified. | ✅ Resolved |
| Q5 — Additional observational data | None available beyond initial observations | ✅ Confirmed |
| Q6 — Solver time limit | 1800s primary, 600s for carryover/practical scenario | ✅ Resolved |
| Q7 — System design diagram | Student already has one; not needed from Claude | ✅ Resolved |
| Rolling horizon | Explicitly excluded from scope; listed as future work | ✅ |
| Carryover | Explicitly excluded; each shift solved independently | ✅ |

---

## 13. Draft Sentences Ready for Thesis Writing

Adapt freely — do not copy verbatim.

### Section 1.1 — Background

> "In high-volume food manufacturing, the efficiency of within-shift production scheduling directly determines material utilization, equipment throughput, and downstream process continuity. At Nestlé Trị An, the roasting operation serves a hybrid demand structure: some products are manufactured against firm customer orders with hard delivery windows (make-to-order), while others must sustain a continuous downstream packaging process through inventory buffers (make-to-stock). Managing both simultaneously across five roasters, two production lines, and sixteen Green Coffee silos within a single eight-hour shift constitutes a scheduling problem of substantial combinatorial complexity."

> "Despite this complexity, the within-shift scheduling layer at the roasting stage currently relies on real-time operator judgment rather than algorithmic decision support. A crew of three to four operators and one contractor manages the entire roasting floor — five roasters, two lines of GC silos, and all pipeline operations — in real time. While experienced supervisors manage strategic planning effectively, the coordinated optimization of batch sequencing, silo management, and inventory replenishment — simultaneously, under shared physical resource constraints — exceeds what unaided human judgment can be expected to achieve reliably."

### Section 1.2 — Problem Statement

> "On-site observation and interviews with production personnel during the internship period reveal a set of recurring operational inefficiencies attributable, at least in part, to the absence of an integrated within-shift scheduling model. Unplanned Green Coffee dumps occur approximately three times per week; each event requires an estimated two hours of handling, setup, and cleanup, while the physical dump itself takes only 5 to 15 minutes. Roaster stoppages attributable to Roasted Coffee silo overflow were estimated at approximately 300 minutes over the observed period. Overall throughput was observed to be approximately 18% below theoretical capacity, with unplanned stoppages as the dominant contributor."

> "These symptoms share a structural cause: the three interacting scheduling sub-problems — make-to-order batch sequencing, make-to-stock replenishment, and Green Coffee silo lifecycle management — are resolved through separate, uncoordinated human decisions. No unified optimization framework exists to coordinate them within a single shift."

### Section 1.3 — Objectives

> "The primary objective of this study is to develop and validate an integrated mathematical optimization model for within-shift batch roasting scheduling at Nestlé Trị An — one that simultaneously addresses make-to-order batch sequencing with due-date constraints, make-to-stock inventory replenishment for continuous downstream demand, and Green Coffee silo lifecycle management under a shared pipeline constraint."

> "The expected scientific contribution is a formal demonstration that Constraint Programming (CP), implemented via Google OR-Tools CP-SAT, outperforms Mixed-Integer Linear Programming (MILP), solved with IBM ILOG CPLEX, on this problem class — and a structural explanation of why, grounded in the disjunctive nature of the pipeline mutual exclusion constraints."

### Section 1.4 — Scope

> "This study is bounded to the within-shift scheduling problem for a single eight-hour shift, covering all three operational shift scenarios: normal production, SKU-switch, and line changeover. Two model formulations are developed and compared: a Mixed-Integer Linear Program solved with IBM ILOG CPLEX, serving as the industry-standard baseline, and a Constraint Program solved with Google OR-Tools CP-SAT, serving as the proposed method."

> "The model operates in an offline, pre-shift planning mode and does not perform real-time rescheduling in response to unplanned machine stoppages — a deliberate scope boundary, as unplanned events require human situational awareness that no deterministic model can replicate."

> "Validation is conducted on 21 shift instances drawn from one full production week — the week exhibiting the lowest frequency of unplanned stoppages in the available dataset, selected to approximate near-ideal manual execution. Data were extracted from the factory's production information system at batch-level granularity — including individual batch start times, roaster assignments, and silo state records — with formal approval from the internship supervisor and Line Manager. Green Coffee SKU identifiers have been anonymized; all quantitative parameters reflect actual production values."

### Section 2.1 — Company Overview (new draft sentences)

> "Nestlé Trị An is a manufacturing facility of Nestlé Vietnam, located in Trị An, Đồng Nai province. The facility operates a batch coffee roasting stage that produces three product families: Pure Soluble Coffee (PSC), Nescafé Dolce Gusto (NDG) capsule coffee, and Busta products for the Starbucks brand. PSC constitutes the primary high-volume output — an intermediate product shipped as 500 kg bags to downstream Nestlé facilities in Đồng Nai and Thailand for manufacturing into finished Nescafé products. The PSC roasting operation produces approximately 17 to 20 tons of pure coffee powder per eight-hour shift."

> "The roasting floor is staffed by three to four operators and one contractor per shift, who collectively manage five roasters across two parallel production lines, sixteen Green Coffee silos, eight Roasted Coffee silos, and two shared GC pipelines. This staffing level, combined with the physical constraint density of the system, means that within-shift scheduling decisions rely heavily on individual operator experience and real-time observation."

---

## 14. Open Items Tracker

| Item | Owner | Status | Blocking? |
|---|---|---|---|
| Literature references (Section 2.2) | Student | Partial list exists, not yet shared | ⚠️ Blocks Ch 2 writing |
| System design diagram (Section 3.2) | Student | Already exists | ✅ Not blocking |
| All other items | — | Resolved | ✅ |

---

*Last updated: Session 3. All critical flags resolved. Next action: student provides final instruction to begin Chapter 1–3 drafting (literature section left as placeholder).*
