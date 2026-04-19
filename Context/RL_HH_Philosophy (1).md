# RL-HH Philosophy: Design Rationale for Nestlé Trị An Batch Roasting

> **Version:** 2.0 (April 2026)  
> **Previous:** v1.0 (tabular Q-Learning meta-agent — SUPERSEDED)  
> **Change:** Meta-agent upgraded from tabular Q-Learning to Dueling DDQN. Designed for UPS from the start. No tabular phase.  
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

| # | Authors | Year | Journal | RL Algo | # Tools | DOI |
|---|---------|------|---------|---------|---------|-----|
| 1 | Karimi-Mamaghan et al. | 2023 | EJOR | Q-Learning | 3-5 | 10.1016/j.ejor.2022.03.054 |
| 2 | Panzer, Bender, Gronau | 2024 | IJPR | DQN | 7 | 10.1080/00207543.2023.2233641 |
| 3 | Lassoued et al. (PetriRL) | 2026 | arXiv | MaskablePPO | 6 | 2601.11189 |
| 4 | Zhang et al. | 2023 | ESWA | Q-Learning | 6 | 10.1016/j.eswa.2023.121050 |
| 5 | Ren & Liu | 2024 | Sci. Reports | **D3QN** | 7 | 10.1038/s41598-024-79593-8 |

### 1.3 Key Paper for Algorithm Choice: Ren & Liu 2024

**Why this paper is the primary algorithm reference:**
- Problem: Dynamic FJSP with **machine breakdowns** + new job insertions + processing time changes — closest to our UPS scenario.
- Algorithm: **Dueling Double DQN (D3QN)** — value-based, handles continuous state, discrete action (7 CDRs).
- State: 8 continuous features normalized [0,1], grouped into machine-level and job-level.
- Network: **Grouped architecture** (not fully connected) — machine features and job features in separate sub-networks for first 3 layers, then merged. Faster convergence than fully connected.
- Result: D3QN convergence 0.85 on-time rate vs DDQN 0.80 vs DQN 0.70. Outperformed all single dispatching rules.
- **Directly validates:** deep value-based RL selecting dispatching rules under machine breakdowns.

---

## 2. Why Not Tabular Q-Learning for UPS

### 2.1 The State Aliasing Problem

Q-Learning raw (non-UPS) uses 10 discretized fields → 4,156 visited states → $296k profit. But this discretization does NOT include roaster DOWN status.

Under UPS, two situations map to the SAME discretized state but need DIFFERENT tools:

```
Situation A: R3 IDLE, t=200, rc_L2=12, gc_l2_psc=9, all roasters healthy
  → Best: PSC_THROUGHPUT (route R3→L2, business as usual)

Situation B: R3 IDLE, t=200, rc_L2=12, gc_l2_psc=9, R4+R5 DOWN
  → Best: PSC_THROUGHPUT route R3→L2 CRITICAL (only L2 producer!)
  → Or: GC_RESTOCK (gc=9 means R3 depletes silo in 9 batches, no R4/R5 to share)
```

Tabular maps both to same state → same Q-values → same tool → wrong under UPS.

### 2.2 Adding UPS Features to Tabular Explodes Visit Requirements

```
Per-roaster DOWN flags: 2^5 = 32 combinations
Current visited states: 4,156
With DOWN flags: 4,156 × 32 = ~133,000 states × 5 tools = 665,000 Q-entries
Visit requirement: ~100 visits per entry → 66.5M episodes → 300h+ on i3
→ NOT PRACTICAL
```

### 2.3 DQN Solves This Without Discretization

DQN takes full 33-feature continuous state. Neural network sees `roaster_status[R4]=DOWN, timer=15/60` directly. Generalizes from training episodes with various DOWN combinations to ALL combinations. No discretization needed.

---

## 3. Design Principles (from Literature)

### P1: Each tool = 1 complete dispatching rule → 1 concrete env action

- PetriRL Eq.12: *"Each heuristic h_k ∈ H is a deterministic mapping h_k : S → A"*
- Ren CDR1-7: Each CDR integrates job selection + machine assignment in one function.

### P2: Tools represent orthogonal scheduling objectives

- PetriRL §4.3.3 cites Kaban 2012: *"no single dispatching rule achieves optimal performance across all measured criteria"*

### P3: Tool count = 5-7

Across 5 papers: range 4-7, median 6.

### P4: Tool masking before selection

- PetriRL Eq.14: *"the Petri net pre-filters the invalid action"*
- If tool output is infeasible → tool masked → agent cannot select it.

### P5: Value-based DRL for discrete tool selection under dynamic disruptions

- Ren 2024: D3QN + 7 CDRs under machine breakdowns → outperformed DQN, DDQN, all single rules.
- Li et al. p.15: *"DRL dynamically adjusts to evolving environments"*

### P6: Grouped network architecture

- Ren 2024 Fig.1(b): machine-level and job-level features in separate sub-networks → faster convergence, less interference.

### P7: Reward = incremental profit ($)

Same reward as Q-Learning raw and MaskedPPO → direct comparison across all methods.

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
- No RESTOCK → must learn restock from scratch (PPO's failure)
- No MTO_DEADLINE → $1k/min tardiness
- No SETUP_AVOID → $800/event waste
- No WAIT → overflow at RC=40

**Cannot add useful ones:**
- R3 routing: handled inside THROUGHPUT (argmin RC)
- Silo selection: handled inside RESTOCK (argmin gc/cap)
- UPS recovery: not a separate tool — UPS changes which tools are feasible + DQN sees DOWN status → correct tool selection EMERGES from state-dependent Q-values

### 4.3 Novel: GC_RESTOCK

No JSSP/FJSP paper has inventory replenishment as a tool. Backed by Q-Learning data: Restock L2-PSC = argmax in 34% of states.

---

## 5. Meta-Agent: Dueling DDQN

### 5.1 Why Dueling DDQN

Following Ren & Liu 2024:
- **Double Q**: Reduces overestimation when tools have similar Q-values.
- **Dueling**: Decomposes Q(s,a) = V(s) + A(s,a). Many states have similar value regardless of tool → dueling learns V(s) faster.
- **Value-based**: No gradient death (no policy/value backbone conflict like PPO).

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

### 5.5 Why NOT PPO for Meta-Agent

- Li et al. p.17-18: *"policy-based methods require more samples... inferior to value-based in sample efficiency... susceptible to training instabilities"*
- PPO already failed end-to-end (gradient death, 18 cycles). Value-based avoids this entirely.
- Action space = 5 discrete tools → perfect for Q-value estimation.

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
