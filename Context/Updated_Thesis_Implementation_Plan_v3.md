# Updated Thesis Implementation Plan v3

> **Version:** v3 (April 2026)
> **Previous:** v2 (March 2026)
> **Status:** Active — reflects current thesis direction after PPO training cycles C1–C18 and Q-Learning results.

---

## CHANGELOG v2 → v3

### Major Changes

| # | What changed | v2 | v3 | Why |
|---|---|---|---|---|
| 1 | **Reactive CP-SAT removed from reactive comparison** | 3 reactive methods: Reactive CP-SAT, Q-Learning, DRL | 4 reactive methods: Dispatching, Q-Learning, MaskedPPO, RL-HH | CP-SAT re-solve takes ~2 min on i3-9100F — impractical. Compensating solve time too complex. CP-SAT kept as deterministic ceiling only. |
| 2 | **RL-Hyper-Heuristic (RL-HH) added** | Not in plan | New Phase 6: Tool-based DRL — Q-learning agent selects from simple heuristic tools | 86 papers 2020–2024 (Li et al. 2024 survey). Hot trend. Novel application to batch roasting. Avoids gradient death. |
| 3 | **Dispatching upgraded from "optional" to core baseline** | "optional supporting baseline" | Core method #1 in reactive comparison | Represents operator practice. Cannot be optional — it's the null hypothesis. |
| 4 | **Q-Learning reframed from "weak baseline" to competitive method** | "lightweight RL baseline, expected weakest, motivates DRL" | Competitive RL method. $296k non-UPS ≈ CP-SAT. Test if deep learning needed. | Empirical data proved Q-Learning near-optimal on deterministic. Old framing wrong. |
| 5 | **Method ladder expanded** | Exact → Simple RL → Deep RL (3 levels) | No Learning → Simple RL → End-to-end DRL → Structured DRL (4 levels) | Cleaner academic progression. Tests "does complexity add value?" |
| 6 | **PPO architectural fix identified** | "MaskedPPO as primary DRL" | MaskedPPO with separate policy/value networks recommended | 18 training cycles identified shared backbone → gradient death as root cause. |
| 7 | **Phase numbering reorganized** | Phases 0–7 (8 phases) | Phases 0–8 (9 phases) | Added Phase 6 (RL-HH). Renumbered Phase 7 (experiments) → Phase 8. |

### Minor Changes
- Updated file plan to include `PPOmask/` and `rl_hh/` directories
- Updated chapter structure to include RL-HH in Chapters 3–5
- Updated experiment design: Block B now has 4 methods instead of 3
- Added gradient death analysis as explicit Chapter 5 contribution
- Updated milestone sequence to 9 milestones

### Unchanged
- Phase 0 (Lock contract) — completed, no changes
- Phase 1 (Build env) — completed, no changes
- Phase 2 (Deterministic solvers MILP + CP-SAT) — completed, no changes
- Layer 1–2 architecture (Input_data, env/) — no changes
- Layer 4 (result_schema, verify, plot) — no changes
- Experiment Block A (MILP vs CP-SAT deterministic) — no changes
- Cost structure, math model, simulation engine spec — no changes

---

## Thesis direction now

The thesis evaluates scheduling strategies across two layers:

1. **Layer A — Deterministic Benchmark:** Compare **MILP vs CP-SAT** on the no-UPS problem. CP-SAT establishes the theoretical performance ceiling.
2. **Layer B — Reactive Scheduling:** Compare **4 reactive strategies** under UPS — from zero intelligence to structured intelligence:
   - Dispatching Heuristic (no learning — operator baseline)
   - Q-Learning (tabular RL — simplest learning)
   - MaskedPPO (end-to-end DRL — learn everything from scratch)
   - RL-Hyper-Heuristic (tool-based DRL — learn to select domain-expert tools)

CP-SAT does NOT participate in reactive experiments. Its re-solve time (~2 minutes on thesis hardware) makes it impractical for real-time reactive scheduling. It serves only as the deterministic ceiling against which all reactive methods are measured.

---

## 1. Revised thesis contribution

### 1.1 What the thesis claims

> This thesis develops a validated scheduling model and simulation environment for Nestlé Trị An's batch roasting process, then evaluates scheduling approaches in two layers: (i) deterministic scheduling quality through MILP vs CP-SAT without disruption, and (ii) reactive scheduling performance through four strategies — dispatching heuristic, tabular Q-Learning, end-to-end MaskedPPO, and RL-based Hyper-Heuristic — under unplanned stoppages. The comparison spans a spectrum from zero-intelligence rules to structured learning, testing whether added complexity yields proportional performance gains.

### 1.2 What the contribution is not

- Not a new MILP, CP-SAT, Q-learning, PPO, or hyper-heuristic algorithm.
- Not a plant deployment project.
- Not a predictive maintenance thesis.
- Not a claim that any single method is universally superior.

### 1.3 Why this framing is strong

The four-method reactive comparison forms a clean **method ladder**:

```
Level 0: Dispatching     — No learning. Rules only. Operator baseline.
Level 1: Q-Learning      — Simplest RL. Tabular. No neural network.
Level 2: MaskedPPO       — End-to-end DRL. Neural network. Learn from scratch.
Level 3: RL-HH           — Structured DRL. RL selects domain-expert tools.
```

Each level adds complexity. The thesis tests: **does each added level of complexity yield proportional performance improvement?**

Expected findings:
- **Finding #1 (Layer A):** CP-SAT outperforms MILP dramatically on disjunctive scheduling (NoOverlap native vs Big-M).
- **Finding #2 (Layer B):** Method ranking under disruption, with analysis of WHY (gradient death for PPO, tool decomposition for RL-HH, etc.).
- **Finding #3:** Gradient death analysis — 18 PPO training cycles identifying shared backbone as root cause.
- **Finding #4:** Whether tool-based decomposition (RL-HH) outperforms end-to-end learning (PPO) for this problem structure.

---

## 2. Document alignment

All documents must reflect the v3 direction:

| Document | Changes needed |
|---|---|
| `Thesis_Problem_Description_v2.md` | Update §9 methods table: remove CP-SAT reactive strategy, add RL-HH, upgrade dispatching to core, reframe Q-Learning |
| `Surrounding_information_introduction_v2.md` | Update §5 contribution, §6 methodology table, §7 comparison metrics |
| `event_simulation_logic_complete.md` | **No changes needed** — simulation engine is strategy-agnostic. §5 CP-SAT re-solve logic remains as reference (code still exists). |
| `mathematical_model_complete.md` | **No changes needed** — model unchanged. |
| `UPS_Mathematical_Model.md` | **No changes needed** — reactive model unchanged (still used by Q-Learning and RL-HH environments). |
| `cost.md` | **No changes needed** — cost structure unchanged. |

---

## 3. Chapter structure

### Chapter 3 — Methodology

Five methodological blocks:

#### 3.1 Problem formulation
- Physical system, assumptions, deterministic math model, reactive extension, cost structure

#### 3.2 Deterministic solution methods
- MILP formulation and role (LP bound benchmark)
- CP-SAT formulation and role (practical solver, theoretical ceiling)

#### 3.3 Simulation environment design
- Time-stepped logic, state definition, transition rules, UPS handling, GC/RC inventory, restock

#### 3.4 Reactive control methods
- Dispatching Heuristic (rule-based baseline)
- Q-Learning (tabular RL)
- MaskedPPO (end-to-end DRL, with gradient death analysis)
- RL-Hyper-Heuristic (tool-based DRL, literature: RL-HH, 86 papers 2020–2024)

#### 3.5 Experimental design
- Block A: MILP vs CP-SAT (deterministic, no UPS)
- Block B: 4 reactive methods (under UPS, paired seeds, factorial design)

### Chapter 4 — Implementation

#### 4.1 Input and data layer
#### 4.2 Mathematical model implementation
#### 4.3 Environment implementation
#### 4.4 Deterministic solvers: MILP and CP-SAT
#### 4.5 Dispatching Heuristic
#### 4.6 Q-Learning
#### 4.7 MaskedPPO (including gradient death analysis and architectural fixes)
#### 4.8 RL-Hyper-Heuristic (tool design, meta Q-table, training)
#### 4.9 Result schema, verification, and visualization

### Chapter 5 — Results and Discussion

#### 5.1 Environment and model validation
#### 5.2 Deterministic comparison: MILP vs CP-SAT (Finding #1)
#### 5.3 Reactive comparison: 4-method comparison under UPS (Finding #2)
#### 5.4 PPO gradient death analysis (Finding #3)
#### 5.5 End-to-end vs structured DRL analysis (Finding #4)
#### 5.6 Sensitivity / robustness analysis
#### 5.7 Practical implications and future work

---

## 4. Implementation architecture

### Layer 1 — Canonical inputs and model contract (unchanged)

Core artifacts: `Input_data/`, math model docs, cost spec.

### Layer 2 — Core environment (unchanged)

Core artifacts: `env/data_bridge.py`, `env/simulation_state.py`, `env/simulation_engine.py`, `env/kpi_tracker.py`, `env/export.py`, `env/ups_generator.py`

### Layer 3 — Solvers / controllers (UPDATED)

| Folder | Method | Role in thesis |
|---|---|---|
| `MILP_Test_v5/` | MILP | Deterministic benchmark, LP bound |
| `CP_SAT_v2/` | CP-SAT | Deterministic solver, theoretical ceiling |
| `dispatch/` | Dispatching Heuristic | **Core baseline** — operator practice (upgraded from optional) |
| `q_learning/` | Q-Learning | Tabular RL — simplest learning approach |
| `PPOmask/` | MaskedPPO | End-to-end DRL — learn from scratch |
| `rl_hh/` | RL-Hyper-Heuristic | **NEW** — tool-based DRL, Q-learning meta-agent |

### Layer 4 — Evaluation and reporting (unchanged)

Core artifacts: `result_schema.py`, `verify_result.py`, `plot_result.py`, `run_experiment.py`, `Reactive_GUI.py`

---

## 5. Dependency tree (UPDATED)

```text
Input_data/
   ↓
Deterministic math model + Reactive math model + cost specification
   ↓
env/   ← canonical execution engine (strategy-agnostic)
   ↓
├── MILP deterministic solver (benchmark, LP bound)
├── CP-SAT deterministic solver (theoretical ceiling)
│
├── Dispatching heuristic (reactive baseline — no learning)
├── Q-Learning (reactive — tabular RL)
├── MaskedPPO (reactive — end-to-end DRL)
└── RL-HH (reactive — tool-based DRL, uses simple heuristic tools)
   ↓
result_schema.py / verify_result.py / plot_result.py
   ↓
run_experiment.py / analysis.py / thesis tables and figures
```

---

## 6. Build order (UPDATED phases)

### Phase 0 — Lock the thesis contract ✅ DONE
Unchanged from v2. All model docs frozen.

---

### Phase 1 — Build and validate the environment ✅ DONE
Unchanged from v2. env/ package validated.

---

### Phase 2 — Deterministic solver layer ✅ DONE
Unchanged from v2. MILP_Test_v5/ and CP_SAT_v2/ implemented and compared.

---

### Phase 3 — Dispatching Heuristic ✅ DONE

> **v2 change:** Was "Phase 3: Reactive CP-SAT." CP-SAT reactive removed from reactive comparison. Dispatching moved up from "optional" to core Phase 3.

#### Objective
Implement the rule-based baseline representing current operator practice.

#### What it is
Priority-based dispatching: urgency-threshold MTO scheduling (>70% time pressure), most-remaining-batches job priority, lowest-stock R3 routing, overflow/downtime checks.

#### Why it matters
This is the **null hypothesis** — what happens with zero intelligence. All learning methods must beat this to justify their existence.

**Run dir:** `dispatch/`

---

### Phase 4 — Q-Learning ✅ DONE

> **v2 change:** Was positioned as "weak baseline, expected to lose." Now reframed based on empirical evidence ($296k non-UPS ≈ CP-SAT $295k).

#### Objective
Implement tabular RL with careful state discretization.

#### Key empirical results (non-UPS)
- 1.4M episodes, 8.3h training
- Last 1000 avg: $296,270 (near-optimal!)
- Q-table: 9,233 entries only
- Restock L2-PSC = argmax in 34% of states — **fully learned restock**

#### Why Q-Learning works here
- No gradient death (no neural network, no shared backbone)
- Q(state, action) updates independently per entry
- Careful state discretization: 10 fields × few values = tractable tabular
- 1.4M episodes = sufficient visitation for convergence

#### Remaining work
- **Test under UPS** — critical experiment. Non-UPS proven, UPS unknown.
- If Q-Learning degrades under UPS → state discretization may not capture UPS dynamics
- If Q-Learning holds under UPS → "simple RL sufficient" finding

**Run dir:** `q_learning/`

---

### Phase 5 — MaskedPPO ⏳ IN PROGRESS (C19 next)

> **v2 change:** Added gradient death analysis. Added architectural fix recommendation (separate networks).

#### Objective
Implement end-to-end DRL with action masking for hard constraint feasibility.

#### Architecture
- 33-feature observation (27 base + 6 context one-hot)
- 21 discrete actions (9 batch start + 4 restock + 7 reserved + WAIT)
- MaskablePPO (sb3-contrib) with action masking for C3, C7, C8, C9, C11, C16, C18, C19
- Per-roaster decision: agent called when roaster becomes IDLE
- Reward: incremental profit ($) + RC maintenance bonus + danger zone penalty

#### Current status: 18 training cycles completed
- Best eval: $127,900 (C12, seed 300, 7 deterministic restocks)
- Core bottleneck: **gradient death** — shared policy/value backbone causes value convergence → advantage ≈ 0 → policy frozen
- Gradient lifespan correlates with eval quality (C12: 4h gradient → $128k)

#### Next steps
- **C19: Separate policy/value networks** — `net_arch={"pi":[256,256],"vf":[256,256]}`
  - Hypothesis: value convergence no longer kills policy gradient
  - Test: seed 42 (prove seed-independence), 4h run
  - Success criteria: approx_kl > 1e-4 at 4h while explained_variance > 0.5

#### Thesis contribution from PPO
Even if PPO underperforms Q-Learning, the **gradient death analysis** (18 cycles, systematic hypothesis testing, root cause identification) is a standalone contribution.

**Run dir:** `PPOmask/`

---

### Phase 6 — RL-Hyper-Heuristic 🆕 NEW

> **v2:** Did not exist. **v3:** New phase — tool-based DRL.

#### Objective
Implement RL agent that selects from a set of simple, domain-expert heuristic tools.

#### Literature backing
- **RL-based Hyper-Heuristic (RL-HH):** 86 papers 2020–2024 (Li et al. 2024, PeerJ Comput. Sci.)
- **Key references:** Karimi-Mamaghan et al. (2023, EJOR), Panzer et al. (2024, IJPR), PetriRL (2026, arXiv)
- **Concept:** RL selects WHICH heuristic to apply; heuristic decides HOW to execute

#### Tool design (6 simple heuristic tools)
Each tool = 3–5 lines of logic, solving one sub-problem:

| Tool | Name | Logic |
|---|---|---|
| 0 | PSC_lowest_RC | Rang PSC, output cho line có RC thấp nhất |
| 1 | Restock_lowest_GC | Restock GC silo thấp nhất nếu pipeline free |
| 2 | MTO_urgent | Rang MTO nếu urgency > threshold |
| 3 | PSC_same_SKU | Rang PSC trên roaster cùng SKU (tránh setup) |
| 4 | R3_balance | R3 route sang line có RC thấp hơn |
| 5 | WAIT | Không làm gì |

#### Meta-agent design
- **Algorithm:** Q-Learning (tabular — no neural network, no gradient death)
- **State:** Discretized meta-state (~432 entries): time_bucket × rc_level × gc_level × mto_remaining × ups_count
- **Action:** Select tool {0, 1, 2, 3, 4, 5}
- **Q-table size:** ~432 × 6 = ~2,600 entries (trivial)
- **Training time:** ~1–2h on i3-9100F (vs 8h+ for PPO)

#### Why this approach avoids PPO's problems
1. **No gradient death** — tabular Q-learning, no neural network
2. **Small action space** — 6 tools vs 21 raw actions
3. **Domain knowledge embedded** — tools encode operator expertise
4. **Interpretable** — "agent chose Tool 1 (restock) because GC was low" = explainable
5. **Fast training** — Q-table converges in ~1–2h vs PPO's 8h+ with uncertain outcome

#### What to implement
1. Tool definitions (6 heuristic functions)
2. Meta-state discretization
3. Meta Q-table + ε-greedy training loop
4. Evaluation script
5. Integration with simulation engine

#### Estimated effort
~100–150 lines of code. All infrastructure (env, result_schema, etc.) already exists.

**Run dir:** `rl_hh/`

---

### Phase 7 — Unified result pipeline ✅ DONE
Unchanged from v2. result_schema.py, verify_result.py, plot_result.py implemented.

---

### Phase 8 — Final experiments ⏳ CURRENT

> **v2 change:** Was Phase 7. Block B now has 4 methods instead of 3. Reactive CP-SAT removed.

#### Experiment Block A — Deterministic benchmark (unchanged)

**Methods:** MILP vs CP-SAT
**Conditions:** No UPS, same instances, same parameters
**Metrics:** Objective value, runtime, LP gap, constraint feasibility
**Purpose:** Finding #1 — CP-SAT outperforms MILP on disjunctive scheduling

#### Experiment Block B — Reactive strategy comparison (UPDATED)

**Methods (4):**
1. Dispatching Heuristic (baseline — no learning)
2. Q-Learning (tabular RL)
3. MaskedPPO (end-to-end DRL)
4. RL-Hyper-Heuristic (tool-based DRL)

**UPS scenario design:**
- UPS rate λ: low (1) / medium (3) / high (5)
- UPS mean duration μ: short (10) / medium (20) / long (40)
- R3 routing mode: fixed / flexible (optional factor)

**Paired random seeds:** All 4 methods face identical UPS realizations per cell.

**Metrics:**
- Total profit (primary)
- PSC throughput
- Stockout event count + duration
- MTO tardiness
- Compute time per decision
- Restock count

**Statistical tests:** Paired t-test or Wilcoxon signed-rank. 100 replications per cell.

**Factorial size:** 3λ × 3μ × 4 methods = 36 cells × 100 reps = 3,600 runs (reduced from v2's 9,000 — fewer cells, same power).

---

## 7. Chapter 5 evidence structure (UPDATED)

### 5.1 Model and environment validation
Constraint tests, event logic, trace verification.

### 5.2 Deterministic comparison: MILP vs CP-SAT
Finding #1: CP-SAT advantage on disjunctive scheduling.

### 5.3 Reactive comparison: 4-method comparison under UPS
Finding #2: Method ranking across disruption levels. Heat map: method × λ × μ.

### 5.4 PPO gradient death analysis
Finding #3: 18 cycles, root cause (shared backbone), architectural fix (separate networks). Table: gradient lifespan vs eval quality.

### 5.5 End-to-end vs structured DRL
Finding #4: PPO (learn everything) vs RL-HH (learn to select tools). Analysis: why decomposition helps/hurts.

### 5.6 Sensitivity analysis
Cost parameter variation ±50%. Does strategy ranking change?

### 5.7 Practical implications
What factory would realistically use. What remains research-only.

---

## 8. File plan (UPDATED)

### 8.1 Core inputs and model docs (unchanged)
- `Input_data/`
- `Context/` (all .md model docs)
- `cost.md`

### 8.2 Environment (unchanged)
- `env/` package

### 8.3 Deterministic solvers (unchanged)
- `MILP_Test_v5/`
- `CP_SAT_v2/`

### 8.4 Reactive controllers (UPDATED)
- `dispatch/dispatching_heuristic.py`
- `q_learning/q_strategy.py`, `q_learning_train.py`, `q_learning_run.py`
- `PPOmask/Engine/` (roasting_env, action_spec, observation_spec, reward_spec, mask_spec, ppo_strategy)
- `PPOmask/train_maskedppo.py`, `evaluate_maskedppo.py`
- **`rl_hh/tools.py`** 🆕 (6 heuristic tool definitions)
- **`rl_hh/meta_agent.py`** 🆕 (Q-learning meta-agent, meta-state, training loop)
- **`rl_hh/rl_hh_strategy.py`** 🆕 (strategy interface for simulation engine)
- **`rl_hh/train_rl_hh.py`** 🆕 (training script)
- **`rl_hh/evaluate_rl_hh.py`** 🆕 (evaluation script)

### 8.5 Reporting utilities (unchanged)
- `result_schema.py`, `verify_result.py`, `plot_result.py`, `run_experiment.py`, `Reactive_GUI.py`

---

## 9. Milestone sequence (UPDATED)

| # | Milestone | Status |
|---|---|---|
| 1 | Freeze math model + simulation logic + inputs | ✅ Done |
| 2 | Finish and validate environment | ✅ Done |
| 3 | Finish deterministic MILP + CP-SAT comparison | ✅ Done |
| 4 | Finish dispatching heuristic | ✅ Done |
| 5 | Finish Q-Learning — train + eval (non-UPS done, UPS pending) | ⏳ UPS test needed |
| 6 | Finish MaskedPPO — C19 separate networks test | ⏳ Next run |
| 7 | **Finish RL-HH — implement tools + meta Q-table + train** | 🆕 New |
| 8 | Run full experiments Block A + Block B | ⏳ After M5–M7 |
| 9 | Write Chapters 3–5 from evidence | ⏳ After M8 |

---

## 10. Thesis narrative (UPDATED)

1. The roasting problem is modeled rigorously (19 constraints, profit objective).
2. The model is embedded in a validated simulation environment.
3. On the deterministic no-UPS problem, MILP and CP-SAT are compared → **CP-SAT wins** (Finding #1).
4. CP-SAT establishes the theoretical performance ceiling (~$295k).
5. Four reactive strategies are compared under controlled UPS scenarios:
   - Dispatching (rules only — operator baseline)
   - Q-Learning (simplest RL — surprisingly competitive)
   - MaskedPPO (end-to-end DRL — limited by gradient death)
   - RL-HH (tool-based DRL — structured intelligence)
6. The thesis identifies **under which conditions each method dominates** (Finding #2).
7. The gradient death phenomenon is analyzed in depth (Finding #3).
8. End-to-end vs structured DRL comparison provides insight into when domain knowledge decomposition helps (Finding #4).
9. The thesis concludes not by declaring a universal winner, but by mapping the **method-complexity vs performance-gain tradeoff** across disruption intensities.

---

## 11. Method ladder — the academic argument

```
Complexity    Method              What it learns       What it's given
──────────────────────────────────────────────────────────────────────
None          Dispatching         Nothing              Fixed rules
Low           Q-Learning          Action values         State discretization
High          MaskedPPO           Everything            Raw obs + actions
Structured    RL-HH               Tool selection        Tools + domain knowledge
──────────────────────────────────────────────────────────────────────
                                                        
Research question: Does moving DOWN this table always improve performance?
Or does structured decomposition (RL-HH) beat brute-force learning (PPO)?
```

This is easier to defend than v2's "exact optimization → simple RL → deep RL" because it includes the **structured intelligence** level that tests whether domain knowledge + RL outperforms pure RL.
