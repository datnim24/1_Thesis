---
name: paper-scout
description: >
  Find, rank, and extract modeling patterns from academic papers structurally 
  closest to a specific OR/scheduling problem. Use this skill whenever the user 
  asks to find papers, do a literature search, build a literature review, find 
  "similar problems in the literature", find references for a thesis chapter, 
  search for academic precedents for a formulation, or says anything like 
  "what papers are close to my problem" or "how have others solved this". 
  Tuned specifically for multi-machine batch scheduling problems with storage 
  constraints (silos/tanks), shared pipeline bottlenecks, sequence-dependent 
  setups, and integrated inventory management — but generalizes to any OR problem.
---

# Paper Scout — Structural Literature Finder for OR/Scheduling Theses

## Purpose

Find academic papers that are **structurally closest** to your optimization problem — meaning same constraint topology, not just same industry. Produce a scored shortlist, a constraint-by-constraint mapping, and ready-to-use thesis writing material.

"Structurally closest" means papers that share:
- The same type of **resource constraints** (e.g., shared pipeline, finite buffers)
- The same **scheduling complexity** (sequence-dependent setups, parallel machines, no-overlap)
- The same **inventory coupling** (production decisions that affect downstream stock levels)
- Ideally, a **comparable solution method** (MILP, CP-SAT, decomposition, or metaheuristic)

Same industry is a bonus, not a requirement.

---

## Phase 0 — Build the Problem Fingerprint

Before any search, extract or ask for the **structural fingerprint**. If the problem description is already available in context (e.g., a thesis document), extract it directly without asking.

### Fingerprint Fields

| Field | What to capture |
|-------|----------------|
| **Machine environment** | Parallel / unrelated / flexible job shop; how many machines; lines if any |
| **Job types** | MTO (fixed due dates) vs MTS (inventory targets); batch sizes |
| **Setup structure** | Sequence-dependent (matrix) vs family-based vs uniform; magnitude |
| **Storage — upstream** | Tank/silo count, capacity, no-mixing rule, assignment when empty, shared fill/drain |
| **Storage — downstream** | Buffer/RC silos, capacity, fill priority rule, consumption pattern |
| **Shared bottleneck** | What resource is contested (pipeline, valve, crane, conveyor), which operations it gates, duration of each |
| **Demand structure** | Hard due dates / consumption curve / safety stock targets |
| **Objective** | Which cost components (tardiness, shortage, setup, idle, dump, replenish) |
| **Horizon** | Shift-level / day-level / weekly; discretization granularity |

If any field is missing and cannot be inferred from context, ask **once** in a compact checklist, then proceed with stated defaults.

### Pre-Built Fingerprint: Nestlé Trị An Roasting Problem

*(Use this when the user's problem matches the thesis context — skip field extraction)*

```
Machine env     : 5 unrelated parallel roasters across 2 lines
Job types       : MTO (NDG/Busta, due ≤240 min) + MTS (PSC, safety-stock driven)
Setups          : Sequence-dependent, uniform 5 min for any SKU transition
Upstream storage: 8 GC silos/line × 3000 kg, single-SKU, no-mix, 
                  1 shared pipeline/line (mutex: consume 2t | replenish 2t | dump 5+⌈q/100⌉ t)
Downstream store: 4 RC silos/line × 5000 kg, FIFO consumption, buffer-silo priority
Demand          : PSC consumption every 10 min (known rate/SKU), MTO hard due dates
Objective       : min(shortage + idle + dump + replenish + tardiness) in VND
Horizon         : 480 min, 1-min slots
Key novelty     : GC pipeline mutex with variable-duration dump + RC safety-stock 
                  coupling + R3 cross-line output routing
```

---

## Phase 1 — Layered Search Strategy

Run searches in 4 layers. Each layer narrows focus. Stop when you have 30–60 candidate papers.

### Layer A — Structural Core (highest priority)
*Find papers with the exact same constraint topology*

```
"batch scheduling" "storage tanks" "no mixing" MILP formulation
"parallel machine scheduling" "finite buffer" "shared resource" "sequence dependent"
"integrated production scheduling" "inventory constraints" "pipeline" 
"scheduling with storage" "tank assignment" "no mixing rule"
"shared material handling" constraint scheduling "disjunctive"
"batch plant scheduling" "storage constraints" "campaign scheduling"
```

### Layer B — Process Industry Analogs
*Chemical/pharma/food industries share the storage+pipeline structure*

```
"multipurpose batch plant" scheduling "storage tanks" allocation
"chemical batch scheduling" "no-wait" OR "intermediate storage" MILP
"pharmaceutical manufacturing" scheduling "shared resources" "sequence dependent"
"food processing" scheduling silo "changeover" "mixed integer"
"continuous casting" scheduling OR "steel plant" "ladle" "shared crane" scheduling
"brewery" OR "winery" scheduling "tank allocation" mathematical programming
```

### Layer C — Method Analogs
*Papers using the same solution approach, regardless of industry*

```
"CP-SAT" OR "constraint programming" scheduling "inventory" "no-overlap" interval
"time-indexed formulation" scheduling "storage capacity" binary variables
"large neighborhood search" scheduling "resource constraints" "batch"
"Benders decomposition" "production scheduling" "inventory"
"rolling horizon" batch scheduling "sequence dependent setups"
```

### Layer D — Domain-Specific (bonus)
*Coffee/roasting — lower structural match expectation*

```
"coffee roasting" scheduling optimization
"roasting" scheduling "batch" "makespan"
```

### Search Databases (in priority order)
1. **Google Scholar** — broadest coverage; use for initial sweep
2. **Scopus / Web of Science** — for citation counts and forward-chaining
3. **ScienceDirect** — Computers & OR, EJOR, IJPE, CIE
4. **IEEE Xplore** — if CP/SAT or computational methods are focal
5. **arXiv (cs.AI, math.OC)** — preprints for recent CP-SAT work

### Citation Chaining Protocol

For each top-ranked paper found:
1. **Backward chain**: scan its reference list for papers it cites in its "related work" section
2. **Forward chain**: use Google Scholar's "Cited by N" to find papers published after it that build on it
3. Look specifically for papers by the **same authors** — they often have a sequence of papers refining the same model

---

## Phase 2 — Scoring Rubric (25 points)

Score every candidate before ranking. Only include papers scoring **≥15/25** in the shortlist (lower threshold than original — some great papers may be missing one dimension).

### Rubric Dimensions

| # | Dimension | 0 | 1–2 | 3–4 | 5 | Weight note |
|---|-----------|---|-----|-----|---|-------------|
| 1 | **Storage realism** | No storage | Finite buffer, no rules | Finite + no-mixing OR assignment | Finite + no-mixing + dynamic reassignment | Heaviest |
| 2 | **Shared bottleneck** | None | Shared but simple (1 type) | Mutex with duration | Mutex + variable duration + multiple operation types | Critical for novelty |
| 3 | **Scheduling richness** | Single machine | Parallel + time windows | + Sequence-dep setup | + Multi-line + cross-routing decisions | |
| 4 | **Demand structure** | Makespan only | Due dates only | Due dates + safety stock | Due dates + consumption curve + shortage penalty | |
| 5 | **Method transferability** | No model detail | Conceptual only | Partial formulation | Full model (vars + constraints + solver) usable in 1 semester | |

### Interpretation

| Score | Action |
|-------|--------|
| 20–25 | **Tier 1** — Deep dive, cite as "closest prior work", adapt formulation |
| 15–19 | **Tier 2** — Deep dive if <3 in Tier 1, else skim |
| 10–14 | **Tier 3** — Skim list only; cite for one specific aspect |
| <10   | Discard unless it's the *only* paper on a key sub-problem |

---

## Phase 3 — Output Format

### A) Top 5 Papers — Deep Dive

For each paper, provide all 7 fields below. Do not omit any.

**1. Citation**
Full APA format + DOI. Flag if open-access available.

**2. Rubric Score**
Breakdown: `Storage(x/5) | Bottleneck(x/5) | Scheduling(x/5) | Demand(x/5) | Method(x/5) = Total/25`

**3. Problem Match Map** (table)

| Aspect | This Paper | Your Problem | Match Quality |
|--------|-----------|-------------|---------------|
| Machine env | ... | 5 unrelated parallel, 2 lines | ✓ / ~ / ✗ |
| Setup rule | ... | seq-dep, 5 min uniform | ✓ / ~ / ✗ |
| Storage (upstream) | ... | GC silos, no-mix, shared pipeline | ✓ / ~ / ✗ |
| Storage (downstream) | ... | RC silos, FIFO, buffer silo | ✓ / ~ / ✗ |
| Shared bottleneck | ... | Pipeline mutex, 3 ops, var. duration | ✓ / ~ / ✗ |
| Demand | ... | MTO due dates + MTS safety stock | ✓ / ~ / ✗ |
| Objective | ... | 5-component penalty (VND) | ✓ / ~ / ✗ |

*Legend: ✓ = direct match, ~ = partial/analog, ✗ = absent*

**4. Formulation Summary**
- Key decision variables (types)
- How they encode: no-mixing / tank assignment / shared resource / consumption
- Objective decomposition
- Solver: MILP (Gurobi/CPLEX) / CP (CP-SAT/ILOG) / decomposition / metaheuristic

**5. Gap Analysis**
*What this paper has that you need* → extractable insight
*What your problem has that this paper lacks* → your novelty contribution

**6. Adaptation Plan**
Concrete steps to adapt their formulation to your context. Be specific about which variables/constraints transfer directly vs. need modification.

**7. Citable Contribution**
≤30-word summary of what you'd cite this paper for in your thesis.

---

### B) Next 5–10 Papers — Skim List

For each: Citation + Score + 2-line relevance note + which chapter section to cite it in.

---

### C) Search Query Log

List every query used, which database, and how many results it returned. This allows reproducibility for the thesis methodology section ("Literature search was conducted using the following queries...").

---

### D) Method Map

Based on what the top papers use, provide:

| If you choose... | Must-read papers | Why |
|-----------------|-----------------|-----|
| Pure MILP (Gurobi/CPLEX) | Paper X, Paper Y | They encode [constraint type] you'll need |
| CP-SAT (OR-Tools) | Paper A, Paper B | They use interval variables for [your bottleneck type] |
| Decomposition / Rolling Horizon | Paper C | They handle [your horizon length] efficiently |
| Metaheuristic baseline | Paper D | Standard benchmark for this problem class |

---

### E) Thesis Positioning (Chapter 2 Ready)

Generate two outputs:

**E1 — Gap Statement** (1 paragraph, academic tone)
"Prior work on [problem class] has addressed [X] and [Y]. However, [specific gap]. In particular, no study has simultaneously modeled [your novelty 1] and [your novelty 2] within a [your horizon/context] setting..."

**E2 — Chapter 2 Outline** (section headers only)
```
2.1 Multi-machine batch scheduling: problem classification
2.2 Sequence-dependent setups: formulations and computational approaches
2.3 Storage-constrained scheduling: tanks, silos, no-mixing rules
2.4 Shared bottleneck resources: pipeline/material-handling constraints
2.5 Integrated production scheduling and inventory management
2.6 Solution methods: MILP vs CP vs decomposition (for this problem class)
2.7 Research gap and contribution of this thesis
```
For each section, suggest 2–3 papers from the shortlist to anchor it.

---

## Phase 4 — Quality Checks

Before finalizing output, verify:

- [ ] Every cited paper has a confirmed DOI or verifiable source (do not hallucinate citations)
- [ ] At least one paper exists for each of: storage/no-mixing, shared bottleneck, CP/MILP method
- [ ] The gap statement is specific — it names the exact constraint combination that prior work has not addressed together
- [ ] The match map has at least 3 ✓ entries for Tier 1 papers
- [ ] No paper in the Top 5 is cited purely for "same industry" without structural match

### Hallucination Prevention Protocol

**Critical**: Academic paper hallucination is a severe risk. Apply these rules strictly:

1. **Never fabricate a paper** — if you cannot confirm a paper exists, say "I believe a paper on this topic may exist but cannot confirm the citation; please verify on Google Scholar"
2. **Flag uncertainty explicitly**: if a paper is retrieved from memory rather than a live search, prefix with ⚠️ *[Unverified — confirm DOI]*
3. **Prefer web search results** over memory for specific papers — always use the web search tool to verify titles, authors, and years before citing
4. When in doubt: provide the **search query that would find the paper** rather than a fabricated citation

---

## Execution Flow

```
START
│
├── Extract problem fingerprint from context (or ask once)
│
├── Phase 1: Run layered searches (A → B → C → D)
│   └── Collect 30–60 candidates
│
├── Phase 2: Score with rubric (25-pt)
│   └── Filter: keep ≥15/25
│
├── Phase 3: Rank and structure output
│   ├── Top 5: full deep dive
│   ├── Next 5–10: skim list  
│   ├── Query log
│   ├── Method map
│   └── Thesis positioning
│
├── Phase 4: Quality checks
│   └── Verify citations, check gap specificity
│
└── Deliver output
```

---

## Usage Notes

- Run Phase 1 searches **before** writing the output — do not rely on memory alone for citations
- If web search is available, use it for every Tier 1 paper to verify the DOI and abstract
- When a student asks "find papers similar to my problem", also flag papers that are **instructive negatives** — papers that tried a similar approach and hit limitations your formulation avoids
- If fewer than 5 papers score ≥15/25, lower the threshold to 12 and note this in the output: "Literature on this exact constraint combination is sparse — this is itself evidence of novelty"
