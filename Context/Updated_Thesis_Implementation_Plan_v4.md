# Updated Thesis Implementation Plan v4

> **Version:** v4 (April 2026)
> **Previous:** v3 (April 2026)
> **Status:** Active — reflects pivoted thesis direction. PPO/MaskedPPO removed. Method comparison restructured around Q-Learning (Zhang 2007 lineage) baseline → Paeng's Modified DDQN (2021) key-reference comparison → RL-HH (Dueling DDQN) innovation.

---

## CHANGELOG v3 → v4

### Major Changes

| # | What changed | v3 | v4 | Why |
|---|---|---|---|---|
| 1 | **MaskedPPO removed entirely from thesis** | Phase 5: end-to-end DRL with gradient death analysis | Removed. PPOmask/ code archived but not part of thesis comparison. | After 18 training cycles with persistent gradient death, PPO cannot deliver a defensible result. The gradient death analysis becomes a Chapter 6 future-work / cautionary note rather than a primary finding. Resources redirected to a stronger comparison structure. |
| 2 | **Primary key reference changed to Paeng et al. (2021), IEEE Access** | Ren & Liu (2024) primary | Paeng et al. (2021) primary; Ren & Liu (2024) supporting | Paeng's problem is structurally identical to ours: UPMSP with sequence-dependent family setups, total-tardiness objective minimized by DDQN, with explicit Q-Learning baseline (LBF-Q from Zhang 2007). Ren & Liu retained as architectural justification for Dueling DDQN over standard DDQN (85% vs 80% on-time completion). |
| 3 | **Method ladder restructured to 3 RL tiers + 1 rule baseline** | Dispatching → Q-Learning → MaskedPPO → RL-HH (4 reactive) | Dispatching → Q-Learning → Paeng DDQN → RL-HH (4 reactive, but cleaner architectural progression) | The new ladder mirrors the literature lineage exactly: tabular Q-Learning (Zhang 2007 / Luo 2020 baseline) → DDQN with parameter sharing (Paeng 2021 = key reference) → Dueling DDQN with grouped network selecting dispatching tools (this thesis innovation, following Ren & Liu 2024). Each step is one architectural rung. |
| 4 | **Q-Learning reframed as Zhang 2007 / LBF-Q lineage baseline** | "Tabular RL — simplest learning. $296k non-UPS." | Tabular Q-Learning baseline, methodologically aligned with LBF-Q (Zhang 2007, EJOR Q1) and the discretized-state Q-Learning baseline that Luo (2020) and Paeng (2021) both beat with deep methods. | Ties our baseline to a published lineage. Replicates Luo's Section 6.4 / Paeng's Table 3 experimental design exactly. The expected outcome (DDQN beats tabular Q-Learning) is a literature-confirmed pattern (38/45 instances in Luo, ~85% better in Paeng). |
| 5 | **Paeng's Modified DDQN added as middle-tier reactive method** | Did not exist | New phase: implement Paeng-style DDQN with parameter sharing for our UPMSP problem. Direct comparison against tabular Q-Learning (lower) and Dueling DDQN RL-HH (upper). | This is the methodological inheritance from the primary key reference. Implementing Paeng's architecture on our problem provides (a) direct evidence that DDQN beats tabular Q-Learning on our problem class (replicating Paeng's headline result), and (b) the apples-to-apples comparison that justifies the next architectural step (dueling + RL-HH). |
| 6 | **R3 routing factor removed as experimental factor (kept fixed)** | Fixed vs. Flexible R3 routing | Flexible only | Reduces factorial size; cross-line R3 routing is part of the action space, not a treatment. |
| 7 | **Phase numbering reorganized** | Phases 0–8 (9 phases, with PPO at Phase 5) | Phases 0–7 (8 phases). PPO phase removed; Paeng DDQN inserted as Phase 5; RL-HH renumbered Phase 6. | Cleaner build order. PPOmask/ folder retained as archived code but not active. |

### Minor Changes
- Removed all references to PPOmask/ from the active build pipeline
- Updated chapter structure to integrate Paeng DDQN in Chapters 3–5
- Updated experiment design: Block B retains 4 methods (Dispatching, Q-Learning, Paeng DDQN, RL-HH) — same total count as v3 but with PPO replaced by Paeng DDQN
- Removed gradient death analysis from primary findings; relegated to Chapter 6 future-work note
- Updated milestone sequence to 8 milestones
- Updated file plan: added `paeng_ddqn/` directory; PPOmask/ marked as archived

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
2. **Layer B — Reactive Scheduling:** Compare **4 reactive strategies** under UPS, organized as a clean architectural progression:
   - **Dispatching Heuristic** — operator baseline, no learning
   - **Tabular Q-Learning** — lineage of Zhang (2007) LBF-Q and the tabular baselines used by Luo (2020) and Paeng (2021)
   - **Paeng's Modified DDQN** — primary key-reference comparison method (Paeng et al. 2021, IEEE Access). Same architecture (dimension-invariant state with parameter sharing, DDQN training) adapted to our UPMSP-SDFST problem
   - **RL-Hyper-Heuristic (Dueling DDQN selecting from 5 dispatching tools)** — the thesis innovation, one architectural step beyond Paeng (dueling + tool-selection action space)

CP-SAT does NOT participate in reactive experiments. Its re-solve time (~2 minutes on thesis hardware) makes it impractical for real-time reactive scheduling. It serves only as the deterministic ceiling against which all reactive methods are measured.

**Method-comparison logic:** Each rung of the ladder is one architectural step beyond the previous one. The literature establishes that deep DDQN methods beat tabular Q-Learning by large margins on UPMSP-SDFST problems (Luo 2020: 38/45 instances; Paeng 2021: ~85% improvement). The thesis tests whether (a) this pattern reproduces on our problem with shared GC pipeline, and (b) whether the further architectural step from standard DDQN to Dueling DDQN with tool-selection action space yields additional gains under UPS disruptions.

---

## 1. Revised thesis contribution

### 1.1 What the thesis claims

> This thesis develops a validated scheduling model and simulation environment for Nestlé Trị An's batch roasting process, then evaluates scheduling approaches in two layers: (i) deterministic scheduling quality through MILP vs CP-SAT without disruption, and (ii) reactive scheduling performance through four strategies under unplanned stoppages — dispatching heuristic (operator baseline), tabular Q-Learning (Zhang 2007 LBF-Q lineage), Modified DDQN with parameter sharing (Paeng et al. 2021 architecture adapted to our problem), and a proposed RL-based Hyper-Heuristic (Dueling DDQN selecting from five dispatching tools). The comparison spans a literature-aligned architectural progression — discretized-state tabular RL → continuous-state DDQN → Dueling DDQN with tool-selection action space — testing whether each step yields proportional performance gains on a UPMSP problem with shared pipeline constraints under stochastic equipment failures.

### 1.2 What the contribution is not

- Not a new MILP, CP-SAT, Q-learning, DDQN, or hyper-heuristic algorithm.
- Not a plant deployment project.
- Not a predictive maintenance thesis.
- Not a claim that any single method is universally superior.

### 1.3 Why this framing is strong

The four-method reactive comparison forms a clean **architectural ladder** aligned with the published RL-HH literature lineage:

```
Level 0: Dispatching     — No learning. Rules only. Operator baseline.
Level 1: Q-Learning      — Tabular RL. Discretized state. Zhang 2007 / Luo 2020 baseline.
Level 2: Paeng DDQN      — Continuous state. Parameter-sharing DDQN. Primary key reference.
Level 3: RL-HH (Dueling) — Dueling DDQN selecting from 5 tools. Thesis innovation.
```

Each level adds one architectural element. The thesis tests: **does each architectural step yield proportional performance improvement on our UPMSP-SDFST problem with shared GC pipeline under UPS?**

Expected findings:
- **Finding #1 (Layer A):** CP-SAT outperforms MILP dramatically on disjunctive scheduling (NoOverlap native vs Big-M).
- **Finding #2 (Layer B):** Deep DDQN methods outperform tabular Q-Learning under UPS (replicating Luo 2020 / Paeng 2021 pattern on a new problem class).
- **Finding #3:** Whether Dueling DDQN with tool-selection action space (RL-HH) outperforms standard DDQN with direct allocation (Paeng-style) on this problem — testing the architectural hypothesis from Ren & Liu (2024).
- **Finding #4:** Sensitivity of method ranking to UPS intensity (low / medium / high λ × short / medium / long μ).

---

## 2. Document alignment

All documents must reflect the v4 direction:

| Document | Changes needed |
|---|---|
| `Thesis_Problem_Description_v3.md` → v4 | Update §9 methods table: remove MaskedPPO, add Paeng's Modified DDQN as third reactive method. Update §9.2 strategy descriptions accordingly. |
| `Surrounding_information_introduction_v3.md` → v4 | Update §5 contribution claim, §6 methodology table, §7 comparison metrics. Replace MaskedPPO references with Paeng DDQN. Update key references: Paeng (2021) primary, Ren & Liu (2024) supporting. |
| `RLHH_Papersv2.md` → v3 | Promote Paeng (2021) IEEE Access to primary key reference. Demote Ren & Liu (2024) to supporting role (architectural justification for Dueling). Update Q1 verification table accordingly. |
| `RL_HH_Philosophy.md` → v3 | Update §1 literature foundation to put Paeng (2021) front. Remove Section "Why NOT PPO for Meta-Agent" (PPO no longer in scope). Update §5.1 architectural justification. |
| `event_simulation_logic_complete.md` | **No changes needed** — simulation engine is strategy-agnostic. |
| `mathematical_model_complete.md` | **No changes needed** — model unchanged. |
| `UPS_Mathematical_Model.md` | **No changes needed** — reactive model unchanged. |
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
- 3.4.1 Dispatching Heuristic (rule-based baseline)
- 3.4.2 Tabular Q-Learning (Zhang 2007 LBF-Q lineage; Luo 2020 Section 6.4 / Paeng 2021 Table 3 baseline experimental design)
- 3.4.3 Modified DDQN with parameter sharing (Paeng et al. 2021 architecture adapted to UPMSP-SDFST with shared GC pipeline)
- 3.4.4 RL-Hyper-Heuristic (Dueling DDQN selecting from 5 dispatching tools — thesis innovation, justified by Ren & Liu 2024 D3QN > DDQN result)

#### 3.5 Experimental design
- Block A: MILP vs CP-SAT (deterministic, no UPS)
- Block B: 4 reactive methods compared under UPS (paired seeds, factorial design)

### Chapter 4 — Implementation

#### 4.1 Input and data layer
#### 4.2 Mathematical model implementation
#### 4.3 Environment implementation
#### 4.4 Deterministic solvers: MILP and CP-SAT
#### 4.5 Dispatching Heuristic
#### 4.6 Tabular Q-Learning (with state discretization mirroring Luo 2020 SOM-style approach)
#### 4.7 Paeng's Modified DDQN (state representation, parameter-sharing network, DDQN training algorithm, reward function)
#### 4.8 RL-Hyper-Heuristic (5 tool definitions, Dueling DDQN with grouped network, training procedure)
#### 4.9 Result schema, verification, and visualization

### Chapter 5 — Results and Discussion

#### 5.1 Environment and model validation
#### 5.2 Deterministic comparison: MILP vs CP-SAT (Finding #1)
#### 5.3 Reactive comparison: 4-method comparison under UPS (Finding #2)
   - 5.3.1 Tabular Q-Learning vs. Paeng DDQN — replicating the literature-confirmed pattern (Luo 2020: 38/45 instances; Paeng 2021: ~85%) on our problem class
   - 5.3.2 Paeng DDQN vs. Dueling DDQN RL-HH — testing the architectural step from Ren & Liu (2024)
#### 5.4 Sensitivity / robustness analysis (UPS intensity grid)
#### 5.5 Method ranking heat map across (λ, μ) cells
#### 5.6 Practical implications and future work

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
| `dispatch/` | Dispatching Heuristic | **Core baseline** — operator practice (null hypothesis) |
| `q_learning/` | Tabular Q-Learning | Discretized-state RL baseline — Zhang 2007 / Luo 2020 lineage |
| `paeng_ddqn/` | **Modified DDQN (Paeng 2021 arch)** 🆕 | **Primary key-reference comparison method** — DDQN with parameter sharing |
| `rl_hh/` | RL-Hyper-Heuristic | **Thesis innovation** — Dueling DDQN selecting from 5 dispatching tools |
| `PPOmask/` | MaskedPPO | **ARCHIVED** — not part of thesis comparison. Kept for reference. |

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
├── Tabular Q-Learning (reactive — Zhang 2007 LBF-Q lineage)
├── Paeng's Modified DDQN (reactive — primary key reference, parameter-sharing DDQN)
└── RL-HH (reactive — Dueling DDQN selecting from 5 tools, thesis innovation)
   ↓
result_schema.py / verify_result.py / plot_result.py
   ↓
run_experiment.py / analysis.py / thesis tables and figures
```

> **Note:** PPOmask/ is archived (not active in the dependency tree). Code retained for reference but not built upon for this thesis.

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

> **v3 → v4 change:** Unchanged. Phase remains the operator-baseline rule-based scheduler.

#### Objective
Implement the rule-based baseline representing current operator practice.

#### What it is
Priority-based dispatching: urgency-threshold MTO scheduling (>70% time pressure), most-remaining-batches job priority, lowest-stock R3 routing, overflow/downtime checks.

#### Why it matters
This is the **null hypothesis** — what happens with zero intelligence. All learning methods must beat this to justify their existence.

**Run dir:** `dispatch/`

---

### Phase 4 — Tabular Q-Learning ✅ DONE

> **v3 → v4 change:** Reframed from "competitive method" to "lineage-aligned tabular baseline." Same code; revised positioning. Tabular Q-Learning now serves the same role in our thesis as LBF-Q (Zhang 2007) does in Paeng (2021) Table 3 and as SOM-discretized Q-Learning does in Luo (2020) Section 6.4 — a discretized-state RL baseline that the deep methods are expected to beat.

#### Objective
Implement tabular RL with careful state discretization. Establishes the architectural floor for value-based RL on this problem.

#### Lineage justification
- **Zhang, Z., Zheng, L., & Weng, M. X. (2007).** Dynamic parallel machine scheduling with mean weighted tardiness objective by Q-Learning. *Int. J. Adv. Manuf. Technol.*, 34(9–10), 968–980 — the original Q-Learning baseline for parallel machine scheduling that Paeng (2021) and Luo (2020) both benchmark against.
- **Luo (2020) Section 6.4 / Table 8:** DQN beats tabular Q-Learning with SOM discretization on 38/45 instances (84.4% win rate) at DDT=1.0.
- **Paeng et al. (2021) Table 3:** DDQN beats LBF-Q on all 8 datasets by 73–86%.

Our tabular Q-Learning replicates this baseline role exactly — discretized state, hash-table Q-values, ε-greedy exploration. The expected outcome (deep DDQN beats this tabular baseline under UPS) is a literature-confirmed pattern we aim to reproduce on a new problem class.

#### Key empirical results (non-UPS, already obtained)
- 1.4M episodes, 8.3h training
- Last 1000 avg: $296,270 (near-optimal — within 1% of CP-SAT $295k ceiling on the deterministic problem)
- Q-table: 9,233 entries — hash table grows on visited states only
- Restock L2-PSC = argmax in 34% of states — restock fully learned

#### Why tabular Q-Learning works on the deterministic problem
- No gradient death (no neural network, no shared backbone)
- Q(state, action) updates independently per entry
- Careful state discretization: 10 fields × few values = tractable tabular
- 1.4M episodes = sufficient visitation for convergence on the deterministic problem

#### What Block B will test
- **Test under UPS** — critical experiment. Non-UPS proven; UPS unknown.
- Hypothesis (from literature): Q-Learning with discretized state will degrade more under UPS than continuous-state DDQN methods, because the discretization bins do not capture roaster DOWN status without state-space explosion (see `RL_HH_Philosophy.md` §2.2).
- Expected outcome: Paeng DDQN and RL-HH both beat tabular Q-Learning on UPS-loaded cells, replicating Luo and Paeng's literature results on our problem class.

**Run dir:** `q_learning/`

---

### Phase 5 — Paeng's Modified DDQN 🆕 (replaces v3 Phase 5 MaskedPPO)

> **v3 → v4 change:** MaskedPPO removed entirely. Paeng's Modified DDQN inserted as the primary key-reference comparison method.

#### Objective
Implement a DDQN-based reactive scheduler with Paeng's parameter-sharing architecture, adapted to our UPMSP-SDFST problem with shared GC pipeline.

#### Why Paeng (2021) is the primary key reference
**Paeng, B., Park, I.-B., & Park, J. (2021).** Deep Reinforcement Learning for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups. *IEEE Access*, 9, 101390–101401. https://doi.org/10.1109/ACCESS.2021.3097254

| Dimension | Paeng (2021) | Our thesis |
|---|---|---|
| Topology | Unrelated Parallel Machines | Unrelated Parallel Machines (5 roasters, 2 lines) |
| Setups | Sequence-dependent family setups | Sequence-dependent SKU setups (5 min + $800) |
| Demand | MTO with due-dates | Mixed MTS (PSC) + MTO (NDG, Busta) |
| Objective | Total tardiness | Multi-component profit |
| Q-Learning baseline | LBF-Q (Zhang 2007) | Tabular Q-Learning (this thesis Phase 4) |
| Proposed method | DDQN with parameter sharing | DDQN with parameter sharing (Phase 5 — this phase) + Dueling DDQN RL-HH (Phase 6) |
| Disruption test | Stochastic processing/setup times (Table 5 robustness) | UPS events (cancels running batch, GC lost) |

The first four rows are structurally aligned — Paeng's problem class is the closest published match to ours. The disruption model differs (continuous noise vs. discrete UPS events) but the methodological architecture transfers cleanly.

#### Architecture (following Paeng 2021)

**State representation** (dimension-invariant via family-based binning):
- Adapted from Paeng's 5 matrices + 1 vector format to our SKU-grouped structure
- Roaster-grouped features: status, timer, last_sku per roaster
- SKU-grouped features: in-progress count, completion progress, due-date proximity (NDG, Busta)
- System-level features: RC buffer levels per line, GC silo levels, pipeline saturation, restock station status

**Action**: tuple (SKU, target_roaster_pool) following Paeng's (family, machine_setup) structure, adapted to roaster eligibility constraints (NDG → R1/R2, Busta → R2, PSC → R1–R5)

**Network**: Parameter-sharing fully-connected blocks (one shared block per SKU, concatenated with context vector, fed into Q-output head). Following Paeng's Figure 2.

**Training**: DDQN (Double DQN with target network), Huber loss, RMSProp, ε-greedy with linear decay, target network soft update τ = 0.005

#### What this phase contributes to the thesis
1. **Replicates Paeng's headline result on a new problem class** — DDQN beats tabular Q-Learning under disruption (expected pattern from literature)
2. **Establishes the apples-to-apples comparison baseline for RL-HH** — same DDQN training algorithm, same simulation engine, same reward function. The architectural difference between Phase 5 and Phase 6 is precisely (a) standard vs. dueling head, and (b) direct allocation vs. tool-selection action space
3. **Provides the methodological inheritance citation** — Chapter 4 can state "this implementation follows Paeng et al. (2021), with modifications to handle MTS demand and UPS disruptions absent in their formulation"

#### Estimated effort
~600–800 LOC (network architecture, replay buffer, training loop, strategy interface, evaluation script). Dependency: PyTorch CPU.
Estimated training time: 3–5 hours on i3-9100F.

**Run dir:** `paeng_ddqn/`

---

### Phase 6 — RL-Hyper-Heuristic (Dueling DDQN selecting dispatching tools) 🆕

> **v3 → v4 change:** Renumbered from v3 Phase 6 (was after MaskedPPO Phase 5). Architecture refined: meta-agent is now Dueling DDQN (not tabular Q-Learning), following Ren & Liu (2024) D3QN pattern. Tool count finalized at 5.

#### Objective
Implement an RL agent that selects from a set of simple, domain-expert dispatching tools at each rescheduling point. The agent uses a Dueling DDQN with grouped network architecture.

#### Architectural step beyond Paeng (2021)
This phase is one architectural rung beyond Phase 5. Two changes:
1. **Standard DDQN → Dueling DDQN** — separates state-value V(s) from action-advantage A(s,a). Justified by Ren & Liu (2024): D3QN convergence 0.85 vs. DDQN 0.80 vs. DQN 0.70 on dynamic FJSP with machine breakdowns.
2. **Direct allocation → Tool-selection action space** — agent selects from 5 dispatching tools instead of (SKU, roaster) tuple. Each tool encodes domain expertise (PSC throughput, GC restock, MTO deadline, setup avoidance, wait). Following the RL-HH paradigm (Luo 2020, Panzer 2024, Ren & Liu 2024).

#### Literature backing
- **Primary algorithm reference:** Ren & Liu (2024), *Scientific Reports* — D3QN architecture, grouped network, CDR-based action space
- **Primary application reference:** Panzer, Bender, Gronau (2024), *International Journal of Production Research* — DRL-HH for production control with disruptions, tool-based action space defense
- **Survey:** Li et al. (2024), *PeerJ Computer Science* — RL-HH taxonomy with D3QN classified as state-of-the-art value-based DRL-HH

#### Meta-agent design (full details: see `RL_HH_Philosophy.md`)
- **Algorithm:** Dueling DDQN
- **State:** 33 normalized continuous features (NO discretization)
- **Action:** 5 dispatching tools — PSC_THROUGHPUT, GC_RESTOCK, MTO_DEADLINE, SETUP_AVOID, WAIT
- **Network:** Grouped architecture (Roaster / Inventory-Flow / Context groups → merged → Dueling V/A heads)
- **~20,000 parameters, CPU-trainable**

#### What to implement
1. Tool definitions (5 heuristic functions)
2. Tool masking layer (compute feasibility mask before agent selects)
3. Dueling DDQN network (grouped, PyTorch)
4. Replay buffer + DDQN training algorithm
5. Strategy interface for simulation engine
6. Training script (UPS enabled from episode 1)
7. Evaluation script

#### Estimated effort
~400–500 LOC. Dependency: PyTorch CPU.
Estimated training time: 3–5 hours on i3-9100F.

**Run dir:** `rl_hh/`

---

### Phase 7 — Unified result pipeline ✅ DONE
Unchanged from v3. result_schema.py, verify_result.py, plot_result.py implemented.

---

### Phase 8 — Final experiments ⏳ CURRENT

> **v3 → v4 change:** Block B method list updated. MaskedPPO removed; Paeng's Modified DDQN inserted. R3 routing factor removed (kept flexible — part of action space).

#### Experiment Block A — Deterministic benchmark (unchanged)

**Methods:** MILP vs CP-SAT
**Conditions:** No UPS, same instances, same parameters
**Metrics:** Objective value, runtime, LP gap, constraint feasibility
**Purpose:** Finding #1 — CP-SAT outperforms MILP on disjunctive scheduling

#### Experiment Block B — Reactive strategy comparison (UPDATED)

**Methods (4):**
1. Dispatching Heuristic (rule baseline — no learning)
2. Tabular Q-Learning (Zhang 2007 LBF-Q lineage — discretized state)
3. **Paeng's Modified DDQN** (primary key reference — DDQN with parameter sharing)
4. RL-Hyper-Heuristic (thesis innovation — Dueling DDQN selecting from 5 tools)

**UPS scenario design:**
- UPS rate λ: low (1) / medium (3) / high (5) events per shift
- UPS mean duration μ: short (10) / medium (20) / long (40) minutes
- R3 routing: flexible only (removed as factor — part of action space)

**Paired random seeds:** All 4 methods face identical UPS realizations per cell.

**Metrics:**
- Total profit (primary)
- PSC throughput
- Stockout event count + duration
- MTO tardiness
- Compute time per decision
- Restock count

**Statistical tests:** Pairwise t-test or Wilcoxon signed-rank at α = 0.05, following Luo (2020) Section 6.4 / Paeng (2021) Table 4 protocol. 50 replications per cell.

**Factorial size:** 3λ × 3μ × 4 methods = 36 cells × 50 reps = **1,800 runs total** (reduced from v3's 3,600 — same statistical power per method, just one fewer factor).

#### Comparison structure for Chapter 5

The 4-method comparison is reported as three nested contrasts:

1. **Rules vs. Learning:** Dispatching vs. {Q-Learning, Paeng DDQN, RL-HH} — does any RL beat operator practice?
2. **Tabular vs. Deep:** Q-Learning vs. {Paeng DDQN, RL-HH} — does deep value-based RL beat discretized-state RL? (Replicates Luo 2020 / Paeng 2021 finding on our problem class)
3. **Standard DDQN vs. Dueling DDQN RL-HH:** Paeng DDQN vs. RL-HH — does dueling architecture + tool-selection action space yield further gains? (Tests Ren & Liu 2024 hypothesis on a new problem class)

---

## 7. Chapter 5 evidence structure (UPDATED)

### 5.1 Model and environment validation
Constraint tests, event logic, trace verification.

### 5.2 Deterministic comparison: MILP vs CP-SAT
Finding #1: CP-SAT advantage on disjunctive scheduling.

### 5.3 Reactive comparison: 4-method comparison under UPS
Finding #2: Method ranking across disruption levels. Heat map: method × λ × μ.

Reported as three nested contrasts:
- 5.3.1 Rules vs. Learning (Dispatching vs. {Q-Learning, Paeng DDQN, RL-HH})
- 5.3.2 Tabular vs. Deep (Q-Learning vs. Paeng DDQN) — replicates Luo 2020 / Paeng 2021 pattern on our problem class
- 5.3.3 Standard DDQN vs. Dueling DDQN RL-HH (Paeng DDQN vs. RL-HH) — tests Ren & Liu 2024 architectural hypothesis

### 5.4 Sensitivity analysis
Cost parameter variation ±50%. Does method ranking change?

### 5.5 Practical implications
What factory would realistically use. What remains research-only.

### 5.6 Future work and limitations (revised)
- End-to-end DRL (PPO/SAC) is omitted from this thesis; gradient death observations from preliminary trials are documented as a future-work caution
- Real plant calibration of UPS parameters (MTBF/MTTR) for transferability assessment

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
- **`paeng_ddqn/network.py`** 🆕 (parameter-sharing DDQN, PyTorch)
- **`paeng_ddqn/replay_buffer.py`** 🆕
- **`paeng_ddqn/agent.py`** 🆕 (DDQN training algorithm with target network, Huber loss)
- **`paeng_ddqn/paeng_strategy.py`** 🆕 (strategy interface)
- **`paeng_ddqn/train_paeng.py`** 🆕 (training script, UPS enabled from ep 1)
- **`paeng_ddqn/evaluate_paeng.py`** 🆕 (greedy eval, result_schema output)
- **`rl_hh/tools.py`** (5 dispatching tool functions)
- **`rl_hh/tool_mask.py`** (compute feasibility mask)
- **`rl_hh/network.py`** (Dueling DDQN, grouped architecture, PyTorch)
- **`rl_hh/replay_buffer.py`**
- **`rl_hh/meta_agent.py`** (Dueling DDQN training)
- **`rl_hh/rl_hh_strategy.py`** (strategy interface)
- **`rl_hh/train_rl_hh.py`** (training script)
- **`rl_hh/evaluate_rl_hh.py`** (evaluation script)
- `PPOmask/` — **ARCHIVED**, retained for reference, not part of thesis

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
| 5 | Finish Tabular Q-Learning — train + eval (non-UPS done, UPS pending) | ⏳ UPS test needed |
| 6 | **Implement Paeng's Modified DDQN — train + eval** 🆕 | ⏳ Next |
| 7 | **Finish RL-HH — implement 5 tools + Dueling DDQN + train** 🆕 | ⏳ After M6 |
| 8 | Run full experiments Block A + Block B | ⏳ After M5–M7 |
| 9 | Write Chapters 3–5 from evidence | ⏳ After M8 |

---

## 10. Thesis narrative (UPDATED)

1. The roasting problem is modeled rigorously (19 constraints, profit objective).
2. The model is embedded in a validated simulation environment.
3. On the deterministic no-UPS problem, MILP and CP-SAT are compared → **CP-SAT wins** (Finding #1).
4. CP-SAT establishes the theoretical performance ceiling (~$295k).
5. Four reactive strategies are compared under controlled UPS scenarios — forming a literature-aligned architectural progression:
   - Dispatching (rules only — operator baseline / null hypothesis)
   - Tabular Q-Learning (Zhang 2007 LBF-Q lineage — discretized-state RL baseline)
   - **Paeng's Modified DDQN (primary key reference — DDQN with parameter sharing)**
   - **RL-HH with Dueling DDQN selecting from 5 tools (thesis innovation — one architectural step beyond Paeng)**
6. The thesis identifies **under which UPS conditions each method dominates** (Finding #2).
7. The Q-Learning vs. Paeng DDQN contrast tests whether the literature-confirmed "deep beats tabular" pattern (Luo 2020: 38/45 instances; Paeng 2021: ~85%) reproduces on our problem class with shared GC pipeline.
8. The Paeng DDQN vs. RL-HH contrast tests the architectural hypothesis from Ren & Liu (2024) — whether dueling architecture + tool-selection action space yields further gains over standard DDQN with direct allocation.
9. The thesis concludes not by declaring a universal winner, but by mapping the **architectural-step vs. performance-gain tradeoff** across UPS intensities.

---

## 11. Method ladder — the academic argument

```
Architectural step      Method              State            Q-network              Action space
───────────────────────────────────────────────────────────────────────────────────────────
None                    Dispatching         —                —                      Direct (rules)
Discretized state RL    Tabular Q-Learning  Discretized      Hash-table Q           Direct (21 actions)
Continuous state RL     Paeng DDQN          Continuous       DDQN + param-share     Direct (SKU, target)
Dueling + tool space    RL-HH (this thesis) Continuous       Dueling DDQN + group   Tool-selection (5 tools)
───────────────────────────────────────────────────────────────────────────────────────────
                                                              
Each row adds one architectural element. The thesis tests:
  - Does each step yield proportional performance improvement on our UPMSP-SDFST problem?
  - Does the literature-confirmed pattern (deep beats tabular: Luo 2020, Paeng 2021) reproduce on our problem class with shared GC pipeline?
  - Does the further architectural step (Dueling DDQN + tool-selection, motivated by Ren & Liu 2024)
    yield additional gains over standard DDQN with direct allocation (Paeng 2021)?
```

This is a cleaner academic argument than v3 because:
1. **Each step is a defined architectural element** (discretized → continuous, standard Q → DDQN → Dueling DDQN, direct → tool-selection action space)
2. **Each step has a published precedent** (Zhang 2007 / Luo 2020 → Paeng 2021 → Ren & Liu 2024 → this thesis)
3. **The expected result pattern is literature-confirmed** for the lower three rungs; only the top rung is the thesis's empirical contribution on a novel problem class
