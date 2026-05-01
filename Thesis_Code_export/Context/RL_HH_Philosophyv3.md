# RL-HH Philosophy: Design Rationale for Nestlé Trị An Batch Roasting

> **Version:** 3.0 (April 2026)
> **Previous:** v2.0 (Dueling DDQN meta-agent — STRUCTURE PRESERVED, FRAMING UPDATED)
> **Change v2 → v3:**
>   - Primary literature anchor changed from Ren & Liu (2024) DFJSP to **Paeng et al. (2021), IEEE Access — UPMSP-SDFST** (problem-class match: parallel machines + sequence-dependent setups + DDQN with parameter sharing). Ren & Liu (2024) retained as **architectural justification** for the dueling step beyond Paeng's standard DDQN.
>   - Section 5.5 ("Why NOT PPO for Meta-Agent") removed entirely. PPO is no longer in thesis scope.
>   - Section 2 ("Why Not Tabular Q-Learning for UPS") preserved — the state aliasing argument now also justifies why our **tabular Q-Learning baseline** (Phase 4) is expected to be beaten by both Paeng's Modified DDQN (Phase 5) and the RL-HH Dueling DDQN (Phase 6) under UPS, replicating Luo (2020) Section 6.4 / Paeng (2021) Table 3 patterns.
> **Purpose:** Canonical reference for RL-based Hyper-Heuristic design decisions.

---

## 1. Literature Foundation

### 1.1 Survey Context

**Li et al. (2024)** — *A review of reinforcement learning based hyper-heuristics.* PeerJ Comput. Sci., 10, e2141. DOI: 10.7717/peerj-cs.2141

- First comprehensive survey: 80+ papers, 6 databases.
- Taxonomy: Value-Based (TRL-HH: Q-table/bandit; DRL-HH: DQN/DDQN/D3QN) vs Policy-Based (PPO/DPPO).
- Q-table = dominant for discrete, small problems. **DRL-HH = recommended for complex/dynamic environments** (p.15-16): *"TRL methods lack scalability and are limited to low-dimensional problems... DRL synergistically harnesses the strengths of both RL and DL, integrating neural network into RL to address the challenges associated with perception and decision-making in complex systems."*
- D3QN (Dueling Double DQN) highlighted as state-of-art value-based DRL-HH (Table 1, Tu et al. 2023).

### 1.2 Core Papers

| # | Authors | Year | Journal | RL Algo | Action space | DOI |
|---|---------|------|---------|---------|--------------|-----|
| ★1 (problem class) | **Paeng, Park, Park** | 2021 | **IEEE Access** | **DDQN + parameter sharing** | (family, setup) tuple — direct allocation | 10.1109/ACCESS.2021.3097254 |
| ★2 (architecture) | **Ren & Liu** | 2024 | Sci. Reports | **D3QN (Dueling DDQN)** | 7 CDRs — RL-HH | 10.1038/s41598-024-79593-8 |
| ★3 (paradigm) | **Luo, S.** | 2020 | Applied Soft Computing | DQN + Double + soft target | 6 CDRs — RL-HH | 10.1016/j.asoc.2020.106208 |
| 4 | Karimi-Mamaghan et al. | 2023 | EJOR | Q-Learning | 3-5 LLHs — RL-HH | 10.1016/j.ejor.2022.03.054 |
| 5 | Panzer, Bender, Gronau | 2024 | IJPR | DQN | 7 LLHs — RL-HH | 10.1080/00207543.2023.2233641 |
| 6 | Zhang Z.-Q. et al. | 2023 | ESWA | Q-Learning | 6 LLHs — RL-HH | 10.1016/j.eswa.2023.121050 |
| 7 (lineage origin) | **Zhang, Zheng, Weng** | 2007 | IJAMT | **Tabular Q-Learning** (lineage origin) | direct allocation | — |

### 1.3 Key Papers for Two-Pillar Inheritance

The thesis innovation rests on two literature pillars converging in our problem class.

#### Pillar 1 — Problem-Class Precedent: Paeng et al. (2021), IEEE Access

**Why this paper is the primary problem-class anchor:**
- Problem: **Unrelated Parallel Machine Scheduling Problem with Sequence-Dependent Family Setups (UPMSP-SDFST)** — structurally identical to ours: parallel roasters + sequence-dependent SKU setups + total tardiness objective.
- Real-world context: **semiconductor wafer preparation in South Korea** — paralleling our Nestlé Trị An batch roasting context (real industrial deployment, not synthetic benchmark).
- Algorithm: **DDQN with parameter-sharing network** — dimension-invariant state representation (5 matrices + 1 vector binned by family), action space (family, setup) tuple, parameter-sharing 3-hidden-layer block per family (64 → 32 → 16, ReLU).
- **Result that we aim to replicate on our problem class:** DDQN beats LBF-Q (tabular Q-Learning lineage of Zhang 2007) on **all 8/8 datasets by ~85%** (Table 3). This is the literature-confirmed pattern our Phase 4 vs. Phase 5 contrast tests.
- **Robustness test (Table 5):** Tested under stochastic processing/setup times (Uniform[0.8x, 1.2x] noise on each parameter), 30 random seeds. Paeng's DDQN achieved both lowest mean AND lowest standard deviation. This is the published precedent for our paired-seed UPS robustness testing in Block B.
- **Directly inherits to our Phase 5 (paeng_ddqn/):** parameter-sharing network architecture, dimension-invariant state design, DDQN training algorithm (Huber loss, target sync, RMSProp), paired-seed disruption testing protocol.

#### Pillar 2 — Architectural Justification: Ren & Liu (2024), Sci. Reports

**Why this paper justifies the Dueling step beyond Paeng:**
- Problem: Dynamic FJSP with **machine breakdowns** + new job insertions + processing time changes — disruption-based scheduling closer in spirit to our UPS scenario than Paeng's stochastic-times robustness test.
- Algorithm: **Dueling Double DQN (D3QN)** — value-based, handles continuous state, discrete action (7 CDRs).
- State: 8 continuous features normalized [0,1], grouped into machine-level and job-level.
- Network: **Grouped architecture** (not fully connected) — machine features and job features in separate sub-networks for first 3 layers, then merged. Faster convergence than fully connected.
- **Result that justifies our Phase 5 → Phase 6 step:** D3QN convergence **0.85 on-time rate** vs DDQN **0.80** vs DQN **0.70**. Outperformed all single dispatching rules. **The 5-percentage-point gap between D3QN and DDQN is the empirical evidence that the dueling architecture step yields measurable gains over Paeng's standard DDQN on disruption-based scheduling.**
- **Directly inherits to our Phase 6 (rl_hh/):** Dueling V/A decomposition, grouped network architecture (Roaster / Inventory-Flow / Context groups), [0,1] feature normalization philosophy.

#### Pillar 3 — RL-HH Paradigm: Luo (2020), Applied Soft Computing

**Why this paper provides the tool-selection action space:**
- Problem: DFJSP with new job insertions (Poisson arrivals).
- Algorithm: DQN + Double + soft target update (predecessor of Ren & Liu 2024 architecture).
- **Action space: 6 Composite Dispatching Rules (CDRs)** — the RL-HH paradigm template our 5-tool action space inherits. Each CDR combines a job-selection criterion with a machine-assignment criterion.
- **Result that we use as expected pattern:** DQN beats tabular Q-Learning (with SOM-discretized state, Section 6.4 / Table 8) on **38/45 instances (84.4% win rate)** at DDT = 1.0. Pairwise t-tests at 5% significance confirm dominance. Author's interpretation: *"compulsive discretization of continuous state features is too rough... fails to accurately distinguish different production statuses."*
- This 38/45 result is the **second piece of literature evidence** (alongside Paeng 2021's 8/8 wins) establishing the expected pattern for our Block B comparison: deep value-based RL beats discretized-state tabular Q-Learning under disruption.
- **Directly inherits to our Phase 4 (q_learning/) and Phase 6 (rl_hh/):** tabular Q-Learning baseline experimental design (Section 6.4 protocol — paired t-tests at 5%), composite-rule action space template, [0,1] feature-normalization philosophy.

### 1.4 Synthesis: How the Three Pillars Combine in This Thesis

```
ARCHITECTURAL LADDER                   LITERATURE PRECEDENT
───────────────────────────────────────────────────────────────────
Phase 3: Dispatching                   Operator practice (no learning)
Phase 4: Tabular Q-Learning            Zhang 2007 → LBF-Q baseline in
                                       Paeng 2021 → SOM-Q-Learning baseline
                                       in Luo 2020 (38/45 lost to DQN)
Phase 5: Paeng's Modified DDQN  ←──── PAENG 2021 ★1 problem-class precedent
                                       (parameter sharing, dimension-invariant
                                       state, DDQN training)
Phase 6: RL-HH (Dueling DDQN ←──────── REN & LIU 2024 ★2 architectural step
        + 5 tools)                     (dueling V/A) + LUO 2020 ★3 paradigm
                                       (RL selecting from CDRs)
───────────────────────────────────────────────────────────────────
```

Each rung adds one architectural element with a published precedent. The thesis innovation = combining all three inheritances and applying them to a problem class with a shared bottleneck resource (GC pipeline) absent in all three precedents.

---

## 2. Why Deep Value-Based RL (Both Phase 5 and Phase 6) Beats Tabular Q-Learning Under UPS

This section supports two architectural decisions: (a) why tabular Q-Learning is the right baseline for our Phase 4 (it intentionally fails on UPS for the same reasons LBF-Q failed in Paeng 2021 and SOM-Q-Learning failed in Luo 2020 Section 6.4), and (b) why both Paeng's Modified DDQN (Phase 5) and the RL-HH Dueling DDQN (Phase 6) are expected to beat tabular Q-Learning under UPS.

### 2.1 The State Aliasing Problem

Q-Learning raw (non-UPS) uses 10 discretized fields → 4,156 visited states → $296k profit. But this discretization does NOT include roaster DOWN status.

Under UPS, two situations map to the SAME discretized state but need DIFFERENT actions:

```
Situation A: R3 IDLE, t=200, rc_L2=12, gc_l2_psc=9, all roasters healthy
  → Best: PSC_THROUGHPUT (route R3→L2, business as usual)

Situation B: R3 IDLE, t=200, rc_L2=12, gc_l2_psc=9, R4+R5 DOWN
  → Best: PSC_THROUGHPUT route R3→L2 CRITICAL (only L2 producer!)
  → Or: GC_RESTOCK (gc=9 means R3 depletes silo in 9 batches, no R4/R5 to share)
```

Tabular maps both to same state → same Q-values → same action choice → wrong under UPS. This is exactly the failure mode Luo (2020) Section 6.4 identifies for SOM-Q-Learning: *"compulsive discretization of continuous state features is too rough... fails to accurately distinguish different production statuses."*

### 2.2 Adding UPS Features to Tabular Explodes Visit Requirements

```
Per-roaster DOWN flags: 2^5 = 32 combinations
Current visited states: 4,156
With DOWN flags: 4,156 × 32 = ~133,000 states × 5 actions = 665,000 Q-entries
Visit requirement: ~100 visits per entry → 66.5M episodes → 300h+ on i3
→ NOT PRACTICAL
```

This explosion is why Paeng (2021) abandoned LBF-Q (linear basis functions over discretized features) in favor of DDQN with continuous state. The same explosion is why Luo (2020) abandoned SOM-discretized Q-Learning in favor of DQN with 7 continuous features. Our thesis follows the same architectural progression.

### 2.3 Continuous-State DDQN Solves This Without Discretization

Both Phase 5 (Paeng's Modified DDQN) and Phase 6 (RL-HH Dueling DDQN) take full continuous state as input. The neural network sees `roaster_status[R4]=DOWN, timer=15/60` directly. Generalizes from training episodes with various DOWN combinations to ALL combinations. No discretization needed.

**Phase 5 input:** Paeng-style state representation (5 matrices + 1 vector, family-based binning).
**Phase 6 input:** Full 33-feature continuous vector (see §5.2).

Both expected to beat the tabular Phase 4 baseline by large margins under UPS, replicating the literature pattern (Luo: 38/45 instances; Paeng: ~85%).

---

## 3. Design Principles (from Literature)

### P1: Each tool = 1 complete dispatching rule → 1 concrete env action

- Luo (2020) Section 4.1: each CDR is a deterministic mapping from state to a (job, machine) assignment.
- Ren & Liu (2024) CDR1-7: Each CDR integrates job selection + machine assignment in one function.

### P2: Tools represent orthogonal scheduling objectives

- Standard RL-HH design principle (Karimi-Mamaghan 2023, Panzer 2024, Ren & Liu 2024): no single dispatching rule achieves optimal performance across all measured criteria; the agent learns when to invoke which.

### P3: Tool count = 5-7

Across 5 papers: range 4-7, median 6. Our 5 tools (PSC_THROUGHPUT, GC_RESTOCK, MTO_DEADLINE, SETUP_AVOID, WAIT) sit at the lower bound.

### P4: Tool masking before selection

- If tool output is infeasible → tool masked → agent cannot select it.
- Implementation: each tool function returns either an action_id ∈ {0..20} or `None` (infeasible). Mask = bool[5].

### P5: Value-based DRL for discrete tool selection under dynamic disruptions

- **Paeng et al. (2021):** DDQN beats LBF-Q on 8/8 datasets by ~85% on UPMSP-SDFST with stochastic processing/setup times. Confirms value-based deep RL beats tabular/linear-approximation Q-Learning on parallel-machine scheduling under disruption.
- **Ren & Liu (2024):** D3QN + 7 CDRs under machine breakdowns → outperformed DQN, DDQN, all single rules. D3QN: 0.85 > DDQN: 0.80 > DQN: 0.70 on-time rate. Confirms the dueling step yields measurable gains.
- **Luo (2020):** DQN + 6 CDRs beats tabular Q-Learning on 38/45 instances (84.4%). Confirms tool-selection paradigm beats tabular RL with discretized state.
- **Li et al. p.15:** *"DRL dynamically adjusts to evolving environments"*

### P6: Grouped network architecture

- Ren 2024 Fig.1(b): machine-level and job-level features in separate sub-networks → faster convergence, less interference.
- Paeng 2021 Figure 2: parameter-sharing per-family blocks → drastic parameter reduction, dimension-invariance.
- Both papers converge on the same insight: structured grouping of input features beats flat fully-connected for scheduling problems with natural feature groupings.

### P7: Reward = incremental profit ($)

Same reward as Tabular Q-Learning (Phase 4) and Paeng's Modified DDQN (Phase 5) → direct comparison across all three RL methods. Ensures the architectural ladder comparison is apples-to-apples.

---

## 4. The 5 Tools

### 4.1 Tool Definitions

| ID | Name | Concern | Action Logic | Analog |
|----|------|---------|-------------|--------|
| 0 | PSC_THROUGHPUT | Maximize output | Start PSC. R3: route to lower-RC line. | SPT, CDR1 |
| 1 | GC_RESTOCK | Secure supply | Restock most critical silo (lowest gc/cap). | **Novel** |
| 2 | MTO_DEADLINE | Meet due dates | Start MTO if eligible. Most-remaining, Busta>NDG. | EDD, CDR2 |
| 3 | SETUP_AVOID | Minimize changeover | Continue same SKU as last batch. | FIFO |
| 4 | WAIT | Conservative fallback | Action 20. Always feasible. | CDR6 |

### 4.2 Why Exactly 5

**Cannot remove any:**
- No THROUGHPUT → no PSC revenue
- No RESTOCK → must learn restock from scratch (the empirical evidence from Phase 4 Q-Learning shows restock is the argmax in 34% of states — encoding it as a tool removes one entire learning sub-problem from the agent's burden)
- No MTO_DEADLINE → $1k/min tardiness
- No SETUP_AVOID → $800/event waste
- No WAIT → overflow at RC=40

**Cannot add useful ones:**
- R3 routing: handled inside THROUGHPUT (argmin RC)
- Silo selection: handled inside RESTOCK (argmin gc/cap)
- UPS recovery: not a separate tool — UPS changes which tools are feasible + Dueling DDQN sees DOWN status → correct tool selection EMERGES from state-dependent Q-values

### 4.3 Novel: GC_RESTOCK

No JSSP/FJSP paper has inventory replenishment as a tool. Backed by Q-Learning data: Restock L2-PSC = argmax in 34% of states.

---

## 5. Meta-Agent: Dueling DDQN

### 5.1 Why Dueling DDQN

The meta-agent architecture is determined by two literature inheritances:

**From Paeng (2021) — DDQN training algorithm:**
- **Double Q**: Reduces overestimation when tools have similar Q-values. Paeng adopts DDQN over plain DQN for stability on UPMSP-SDFST.
- **Target network with periodic sync**: Standard DDQN training, replay buffer 50,000 transitions.
- **Huber loss + RMSProp**: Same as Paeng (2021) Algorithm 2.

**From Ren & Liu (2024) — Dueling architectural step:**
- **Dueling**: Decomposes Q(s,a) = V(s) + A(s,a). Many states have similar value regardless of tool choice → dueling learns V(s) faster. Empirically: D3QN convergence 0.85 vs. DDQN 0.80 on Ren & Liu's DFJSP with breakdowns.

The combined Dueling DDQN sits exactly one architectural step beyond Paeng (2021): same training algorithm, but with the V/A head decomposition added. This step is the entire architectural difference between our Phase 5 (paeng_ddqn/) and Phase 6 (rl_hh/) implementations.

### 5.2 State Input: Full 33 Features (NO Discretization)

```
Features 0:      t / 479
Features 1-5:    roaster_status[R1..R5]     ← SEES DOWN STATUS
Features 6-10:   roaster_timer[R1..R5] / 60  ← SEES DOWN REMAINING TIME
Features 11-15:  last_sku[R1..R5]
Features 16-17:  rc_stock[L1,L2] / 40
Feature 18:      mto_remaining / N_MTO
Features 19-20:  pipeline_busy[L1,L2] / 15
Features 21-24:  gc_stock (4 silos, normalized)
Feature 25:      restock_station_busy
Feature 26:      restock_timer / 15
Features 27-32:  context one-hot [RESTOCK,R1,R2,R3,R4,R5]
```

### 5.3 Network Architecture (Grouped, following Ren 2024)

```
Group A — Roaster (15 features: 5 status + 5 timer + 5 last_sku)
  → Dense(64) → ReLU → Dense(32) → ReLU

Group B — Inventory/Flow (11 features: RC, pipeline, GC, restock, mto)
  → Dense(64) → ReLU → Dense(32) → ReLU

Group C — Context (6 features: one-hot)
  → Dense(16) → ReLU

Merge: concat(32 + 32 + 16) = 80
  → Dense(128) → ReLU

Dueling:
  V_head → Dense(64) → ReLU → Dense(1)
  A_head → Dense(64) → ReLU → Dense(5)
  Q = V + A - mean(A)
```

~20,000 parameters. CPU-trainable.

### 5.4 Training

```
Replay buffer:     50,000 transitions
Batch size:        128
Learning rate:     0.0005
Discount γ:        0.99
ε:                 1.0 → 0.05 linear over 70% budget
Target net:        soft update τ = 0.005
Episodes:          200k-500k
UPS:               ENABLED from episode 1 (λ=3, μ=20)
Reward:            incremental profit ($)
Framework:         PyTorch CPU
Est. time:         3-5h on i3-9100F
```

---

## 6. Integration

```
env → rl_hh_strategy.decide(state, roaster)
   → build 33-feature vector from state
   → compute tool_mask (run each tool, None = masked)
   → Dueling DDQN forward → 5 Q-values
   → mask infeasible (Q = -∞)
   → ε-greedy (train) or argmax (eval)
   → selected tool → action_id
   → return to env
```

Interface: `decide(state, roaster) → action_id ∈ {0..20}`

---

## 7. File Plan

```
rl_hh/
├── tools.py            # 5 tool functions
├── tool_mask.py        # compute_tool_mask → bool[5]
├── network.py          # DuelingDDQN (PyTorch, grouped arch)
├── replay_buffer.py    # Circular buffer, 50k
├── meta_agent.py       # Agent: select, store, train_step
├── rl_hh_strategy.py   # Strategy interface for env
├── train_rl_hh.py      # Training loop (UPS from ep 1)
├── evaluate_rl_hh.py   # Greedy eval, result_schema output
└── configs.py          # All hyperparams
```

~400-500 LOC. Dependency: PyTorch CPU.
