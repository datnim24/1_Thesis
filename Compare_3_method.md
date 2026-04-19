# Comparison of Three RL Methods for Nestle Tri An Roasting Scheduling

**Problem**: Reactive scheduling of 5 coffee roasters across a 480-minute shift, managing PSC (make-to-stock), NDG and BUSTA (make-to-order) production, RC/GC inventory, and pipeline logistics.

**Shared simulation engine**: All three methods use the same `env/simulation_engine.py` with identical cost parameters, constraints, and KPI calculation.

---

## 1. Q-Learning (Tabular)

### 1.1 Architecture Overview

Q-Learning uses a **tabular, per-roaster** formulation. Each roaster makes independent decisions using a shared Q-table, while restock is delegated to a hand-crafted heuristic.

```
State (tuple) ──→ Q-table lookup ──→ argmax Q(s,a) ──→ action
                    │
                    └── ε-greedy: with probability ε, pick random valid action
```

### 1.2 State Representation

Each state is a **discrete tuple** per roaster:

```
(
  "roaster",              # marker string
  roaster_id,             # "R1"-"R5"
  time_bin,               # 0-7 (piecewise bins around MTO due window)
  last_sku,               # "PSC", "NDG", "BUSTA"
  pipe_busy,              # 0 or 1 (pipeline occupancy for this roaster's line)
  rc_home,                # 0-4 (5-bin: empty / low / safety / high / overflow)
  rc_other,               # 0-4 (only for R3 which is flexible across lines)
  mto_max_urgency,        # 0-4 (done / comfortable / watch / urgent / overdue)
  gc_min_eligible,        # 0-4 (GC stock bin for eligible SKUs)
  setup_flag,             # 0 or 1 (just completed setup?)
)
```

**Discretization functions:**
- `_bin_time()`: Piecewise bins anchored on MTO due times (slot 240). Cutoffs at `[due/2, due-60, due-20, due, due+60, SL-90, SL-30]` → 7-8 bins. The bins are denser near the due window to give the agent fine-grained time awareness when MTO urgency peaks.
- `_bin_rc()`: 5 levels based on safety stock (20) and max RC (40): `{0: empty (rc<=0), 1: <safety/2, 2: <safety, 3: <max, 4: >=max}`
- `_bin_gc()`: 5 levels by capacity ratio: `{0: empty, 1: 1 unit, 2: <restock_qty, 3: <2×restock_qty, 4: full}`
- `_bin_urgency()`: Escalation based on slack time vs roast duration: `{0: done, 1: comfortable, 2: watch, 3: urgent, 4: overdue}`

**Total state space**: ~1,000-10,000 unique states visited during training (out of a theoretical ~200k combinations). The tabular approach keeps this tractable by discretizing aggressively.

### 1.3 Action Space

21 discrete actions shared with PPO:

| Actions | Description |
|---------|-------------|
| 0-5 | PSC on R1→L1, R2→L1, R3→L1, R3→L2, R4→L2, R5→L2 |
| 6-8 | NDG on R1, NDG on R2, BUSTA on R2 |
| 9-12 | Restock L1(PSC), L1(NDG), L1(BUSTA), L2(PSC) |
| 13-19 | Reserved / mapped to WAIT |
| 20 | WAIT (always valid) |

**Per-roaster masking**: Before Q-lookup, actions are filtered to only those feasible for the current roaster (GC stock check, pipeline availability, time horizon, SKU eligibility).

### 1.4 Restock Mechanism — Heuristic Delegation

**Q-Learning does NOT learn restock.** Restock is delegated entirely to a `DispatchingHeuristic`:

```python
# Triggered when:
# - Idle roasters exist for a feasible silo
# - GC stock critically low (<2 restock batches)
# - RC stock below safety with idle capacity
# - MTO urgency high

# Decision: deterministic rule-based selection of best restock target
```

**Why**: Including restock in the Q-table would multiply the state space by ~4x (one dimension per restock target). Delegation keeps the Q-table focused on roasting decisions where tabular learning is most effective.

### 1.5 Training Process

**Standard Q-learning update:**
```
Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') - Q(s,a)]
```

**Episode flow:**
1. Initialize fresh `SimulationEngine` + random seed
2. For each slot (0 to 479):
   - Process UPS events, timers, consumption, idle penalties
   - **Restock decision** → heuristic (not Q-learning)
   - **Per-roaster decisions** → Q-learning (ε-greedy)
   - Collect `(state, action, reward, next_state)` transitions
3. End of shift: penalize skipped MTO ($50k/batch)
4. Update Q-table from collected transitions

**Reward**: Incremental profit delta per slot: `kpi.net_profit(after) - kpi.net_profit(before)`

**Exploration**: ε-greedy with linear decay from 1.0 → 0.05 over 70% of training budget.

### 1.6 Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| α (learning rate) | 0.05 | Q-table update rate |
| γ (discount) | 0.99 | Long-horizon planning |
| ε (exploration) | 1.0 → 0.05 | Linear decay over 70% of budget |
| Training time | 8.3h (30,000s) | ~1.43M episodes |
| Training speed | ~200 ep/s | Fast (no neural network) |
| Q-table size | 9,233 entries | Very compact |

### 1.7 Strengths & Weaknesses

**Strengths:**
- Fast training (~200 ep/s, no GPU needed)
- Interpretable: can inspect Q(s,a) for any state
- Stable convergence (no gradient death, no entropy collapse)
- Restock heuristic is reliable and well-tuned
- Balanced L1/L2 stock management

**Weaknesses:**
- Discretization loses information (e.g., rc=9 and rc=11 both map to same bin)
- Cannot represent continuous state relationships
- Restock not learned — relies on hand-crafted rules
- State space grows exponentially with new features
- No generalization between similar states

---

## 2. MaskablePPO (Neural Network Policy)

### 2.1 Architecture Overview

MaskablePPO uses a **neural network** to directly map continuous observations to action probabilities, with action masking to enforce feasibility.

```
Observation [33] ──→ Policy Network [256,256] ──→ Masked Softmax ──→ Action
                     Value Network  [256,256] ──→ V(s) estimate
```

### 2.2 Observation Space — 33 Continuous Features

**Base features (27 dims):**

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | time | [0,1] | Current slot / shift_length |
| 1-5 | roaster_status[R1-R5] | {0, 0.33, 0.67, 1.0} | IDLE/RUNNING/SETUP/DOWN |
| 6-10 | remaining_timer[R1-R5] | [0,1] | Normalized time remaining |
| 11-15 | last_sku[R1-R5] | {0, 0.5, 1.0} | PSC/NDG/BUSTA |
| 16 | rc_stock_L1 / max_rc | [0,1] | L1 RC inventory level |
| 17 | rc_stock_L2 / max_rc | [0,1] | L2 RC inventory level |
| 18 | mto_remaining / total | [0,1] | Fraction of MTO still needed |
| 19-20 | pipeline_mode[L1,L2] | {0, 0.5, 1.0} | FREE/CONSUME/RESTOCK |
| 21-24 | gc_stock[4 pairs] / capacity | [0,1] | GC silo fill levels |
| 25 | restock_busy | {0,1} | Restock truck in use? |
| 26 | restock_timer / duration | [0,1] | Remaining restock time |

**Context features (6 dims):** One-hot encoding of which decision this is:
| Index | Context |
|-------|---------|
| 27 | RESTOCK decision |
| 28-32 | R1/R2/R3/R4/R5 roaster decision |

### 2.3 Action Space — 21 Discrete Actions with Masking

Same 21 actions as Q-Learning. The key difference: **hard action masking** via `sb3-contrib` `ActionMasker`.

```python
def action_masks() -> np.ndarray[bool, (21,)]:
    # Returns True for each feasible action
    # Checks: GC stock, pipeline availability, time horizon, SKU eligibility
    # WAIT (action 20) always valid
```

If the agent selects a masked (infeasible) action → episode terminates with $50k penalty. This never happens in practice because SB3's masked sampling prevents it.

**Critical difference from Q-Learning**: PPO learns BOTH roasting AND restock decisions. Actions 13-16 are restock actions that the neural network learns to invoke, rather than delegating to a heuristic.

### 2.4 Reward Structure — Multi-Layer Shaping

**Layer 1 — Base incremental profit:**
```
reward += kpi.net_profit(after_action) - kpi.net_profit(before_action)
```

**Layer 2 — RC maintenance bonus** (dense per-slot shaping):
```
For each line:
  if rc_stock >= safety_stock (20):  reward += $50 per slot
  elif rc_stock < safety_stock/2 (10):
    danger_frac = 1.0 - rc / 10
    reward -= $150 × danger_frac per slot  (at rc=0: -$150)
```

**Layer 3 — Hard constraint violations:**
```
RC < 0 or GC < 0 or RC > 40 → reward -= $50,000 + episode terminates
```

**Layer 4 — End-of-shift penalties:**
```
Skipped MTO: -$50,000 per unfinished batch
Completion bonus: +$100,000 if full shift completed without violation
```

### 2.5 Network Architecture — Separate Policy/Value

```
Policy Network (Wπ):
  Input [33] → Linear(256) → ReLU → Linear(256) → ReLU → Linear(21) → logits

Value Network (Wv):
  Input [33] → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1) → V(s)
```

**Why separate**: With shared weights, value function convergence freezes the shared features, killing the policy gradient (`approx_kl → 0`). Separate networks allow the value function to converge independently while the policy continues learning. This was discovered after 19 failed training cycles.

**Total parameters**: ~153,000 (vs ~80,000 shared).

### 2.6 VecNormalize — Reward Normalization

```python
VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
```

Normalizes rewards to running unit variance using exponential moving average. This prevents large reward magnitudes ($50k-$500k range) from swamping small per-action advantages ($1k-$5k range). Combined with separate networks, this was the key fix that enabled PPO to maintain gradient signal throughout training.

### 2.7 Training Process

**PPO update loop (per rollout of 4096 steps × 4 envs = 16,384 samples):**
1. Collect 16,384 transitions with current policy
2. Compute advantages using GAE (λ=0.95, γ=0.99)
3. For 3 epochs (not 10 — reduces over-optimization):
   - Split into 128-sample minibatches
   - Compute ratio π_new/π_old
   - Clip ratio to [1-0.3, 1+0.3]
   - Update policy: maximize clipped surrogate objective + entropy bonus
   - Update value: minimize MSE on returns
   - **Early stop if KL > 0.02** (target_kl)
4. Log to TensorBoard

**Gradient steps per rollout**: 16,384 / 128 × 3 = 384 (reduced from 640+ in early cycles to prevent over-optimization).

### 2.8 Key Hyperparameters (Final C23 Configuration)

| Parameter | Value | Notes |
|-----------|-------|-------|
| learning_rate | 5e-4 (linear decay) | Decays to 5% of base over training |
| gamma | 0.99 | Discount factor |
| n_steps | 4,096 | Rollout length per env |
| batch_size | 128 | Minibatch size |
| n_epochs | 3 | Epochs per rollout update |
| gae_lambda | 0.95 | GAE parameter |
| ent_coef | 0.05 | Entropy bonus weight |
| clip_range | 0.3 | PPO clipping range |
| target_kl | 0.02 | Early stopping threshold |
| vf_coef | 0.5 | Value function loss weight |
| n_envs | 4 | Parallel environments |
| net_arch | {pi:[256,256], vf:[256,256]} | Separate networks |
| normalize_reward | True | VecNormalize on rewards |
| rc_maintenance_bonus | 50.0 | Dense restock shaping |
| completion_bonus | 100,000 | Full shift completion reward |
| Training time | 12h | ~84k episodes |
| Training speed | ~2 ep/s | Slow (neural network + 4 parallel envs) |

### 2.9 Training Journey — 23 Cycles of Investigation

The path to a working PPO configuration required **23 training cycles** over ~5 days, each testing a specific hypothesis:

| Phase | Cycles | Key Finding |
|-------|--------|-------------|
| Baseline | C1-C3 | Entropy collapses → policy freezes at $54.9k |
| Resume attempts | C4-C6 | Resumed policies are permanently frozen |
| Structural fix | C7-C8 | n_epochs=3 prevents over-optimization → $95.9k |
| Reward shaping | C9-C10 | Danger zone penalty + fresh starts → violation rate drops to 84% |
| Lucky seed | C12 | Seed 300 kept gradient alive 4h → first deterministic restock |
| Architecture | C15-C19 | Separate networks = 35x stronger gradient (but still died) |
| **Breakthrough** | **C20** | **Separate nets + VecNormalize = gradient alive indefinitely** |
| Correction | C22-C23 | MTO skip penalty + tardiness fix → $377.6k Grade A |

**Core discovery**: PPO's policy gradient dies when `approx_kl → 0`, caused by value function convergence freezing advantage estimates. The fix: separate policy/value networks + reward normalization.

### 2.10 Strengths & Weaknesses

**Strengths:**
- Learns both roasting AND restock (no hand-crafted heuristic)
- Continuous state representation (no discretization loss)
- Highest profit of all three methods ($377.6k)
- Can potentially generalize to unseen scenarios
- Gradient stayed alive for 12h+ with final configuration

**Weaknesses:**
- Extremely slow to develop (23 cycles, ~100+ hours of compute)
- Sensitive to architecture choices (shared vs separate networks is critical)
- Requires reward shaping (5 layers of bonuses/penalties)
- Imbalanced L1/L2 stock management (L1 hovers at 2-5, L2 at 19-22)
- Skips 2 MTO batches (1 NDG + 1 BUSTA) despite $50k skip penalty
- Not interpretable (neural network black box)

---

## 3. RL-HH (Reinforcement Learning Hyper-Heuristic)

### 3.1 Architecture Overview

RL-HH is a **two-tier** system: a Dueling DDQN meta-agent selects among 5 pre-programmed heuristic tools, which then produce the actual roaster/restock actions.

```
Observation [33] ──→ Dueling DDQN ──→ Q-values for 5 tools ──→ Selected tool
                                                                    │
                                                    Tool executes deterministic logic
                                                                    │
                                                                    ↓
                                                              Action (roaster/restock)
```

### 3.2 The Five Heuristic Tools

**Tool 0 — PSC_THROUGHPUT:**
```
For given roaster: immediately start PSC batch.
R3 special: routes PSC to line with lower RC stock (load balancing).
Returns: PSC action_id, or None if infeasible.
```

**Tool 1 — GC_RESTOCK:**
```
At restock decision point: find GC silo with lowest stock/capacity ratio.
Select restock action that refills the most depleted silo.
Returns: restock action_id, or None if no restock needed.
```

**Tool 2 — MTO_DEADLINE:**
```
Select MTO SKU with most remaining batches (deadline pressure).
Tie-break: BUSTA > NDG (BUSTA more constrained, only R2 can produce it).
Returns: MTO action_id, or None if MTO complete.
```

**Tool 3 — SETUP_AVOID:**
```
Continue same SKU as last batch to avoid $800 setup + 5 min changeover.
If same SKU still needed → return same-SKU action.
Otherwise: None.
```

**Tool 4 — WAIT:**
```
Always feasible. Returns WAIT action.
Used when no other tool produces valid output.
```

**Tool masking**: Before Q-value argmax, tools that cannot produce a valid action are masked out. Tool 4 (WAIT) is always valid.

### 3.3 State Representation

**Same 33-dim observation as PPO** (reuses `observation_spec.py`):
- 27 base features (time, roaster status/timers/SKUs, RC/GC stocks, pipeline, restock)
- 6 context features (one-hot for RESTOCK or R1-R5)

### 3.4 Network Architecture — Dueling DDQN with Feature Grouping

```
Group A — Roaster block:
  obs[1:16] (15 dims) → Linear(64) → ReLU → Linear(32) → ReLU → [32]

Group B — Inventory block:
  obs[16:27] (11 dims) → Linear(64) → ReLU → Linear(32) → ReLU → [32]

Group C — Context + time block:
  obs[0] + obs[27:33] (7 dims) → Linear(16) → ReLU → [16]

Merge:
  concat([32] + [32] + [16] = 80) → Linear(128) → ReLU → [128]

Dueling streams:
  Value:     [128] → Linear(64) → ReLU → Linear(1)  → V(s)
  Advantage: [128] → Linear(64) → ReLU → Linear(5)  → A(s,a)

Output:
  Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
```

**Why feature grouping**: Separates roaster-specific, inventory, and context features for cleaner gradient flow. Each group learns specialized representations before merging.

**Why Dueling**: Separates state value (V) from action advantage (A). Many states have similar values regardless of action chosen (e.g., late shift with all MTO done) — dueling architecture handles this efficiently.

**Parameters**: ~30,000 (much smaller than PPO's 153,000).

### 3.5 Training Process — Double DQN

**DDQN update (per episode):**
1. Run episode using ε-greedy tool selection
2. Store transitions in replay buffer (50,000 capacity)
3. Sample 128-transition minibatch
4. Online network selects best tool: `a* = argmax_a Q_online(s', a)`
5. Target network evaluates: `Q_target(s', a*)`
6. Target: `y = r + γ × Q_target(s', a*) × (1 - done)`
7. Loss: Smooth L1 between `Q_online(s, a)` and `y`
8. Soft target update: `θ_target ← 0.005 × θ_online + 0.995 × θ_target`
9. Repeat 4 gradient steps per episode

**Fast episode runner** (`fast_loop.py`): Bypasses Gymnasium overhead for ~50 ep/s (vs PPO's ~2 ep/s).

### 3.6 Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| learning_rate | 5e-4 | Adam optimizer |
| gamma | 0.99 | Discount factor |
| batch_size | 128 | Replay sampling |
| buffer_size | 50,000 | Experience replay capacity |
| epsilon | 1.0 → 0.05 | Linear decay over 70% of training |
| tau | 0.005 | Soft target update rate |
| grad_clip | 10.0 | Gradient clipping |
| trains_per_ep | 4 | Gradient steps per episode |
| Training time | 5h | ~300k episodes |
| Training speed | ~17 ep/s | Medium (small network + fast_loop) |

### 3.7 Strengths & Weaknesses

**Strengths:**
- Interpretable: tool selection ratios reveal strategy (e.g., "PSC_THROUGHPUT 7.7%, WAIT 81.6%")
- Guaranteed valid actions (tools produce feasible outputs by construction)
- Completes ALL MTO batches (NDG=5, BUSTA=5) — no skipping
- Small network, fast training
- Leverages domain expertise through heuristic tools
- Scalable: adding a new roaster doesn't change the 5-tool action space

**Weaknesses:**
- Limited by heuristic quality (can't discover strategies beyond the 5 tools)
- High idle cost ($138,800 — 40% of revenue!) suggests inefficient roaster utilization
- WAIT dominates at 81.6% — agent is very conservative
- Performance ceiling bounded by tool capabilities
- Requires domain expert to design good tools

---

## 4. Head-to-Head Performance Comparison

### 4.1 Non-UPS Results (No Disruptions)

| Metric | Q-Learning | MaskablePPO (C23) | RL-HH* |
|--------|-----------|-------------------|--------|
| **Net Profit** | **$336,000** | **$377,600** | $340,800* |
| Revenue | — | $527,000 | $486,000 |
| Total Costs | — | $149,400 | $145,200 |
| PSC | ~100 | 117 | 104 |
| NDG | 5 | 4 | 5 |
| BUSTA | 5 | 4 | 5 |
| **MTO Complete** | **10/10** | **8/10** | **10/10** |
| MTO Skip Penalty | $0 | $100,000 | $0 |
| Tardiness | 0 min | 0 min | 0 min |
| Tard Cost | $0 | $0 | $0 |
| Setup Events | — | 5 | 8 |
| Setup Cost | — | $4,000 | $6,400 |
| Restocks | — | 17 | 25 |
| Stockouts | 0 | 0 | 0 |
| Idle Cost | — | $42,400 | $138,800 |
| Violations | 0 | 0 | 0 |
| Training Time | 8.3h | 12h | 5h |
| Training Episodes | 1.43M | 84k | 300k |

*\*RL-HH was evaluated with UPS (lambda=3.0, mu=20). Non-UPS result not available — likely ~$380-420k without disruptions.*

### 4.2 Cost Breakdown Analysis

**PPO's costs:**
- $100,000 MTO skip (skipped 2 batches) — this is PPO's biggest weakness
- $42,400 idle (L1 under safety stock most of the shift)
- $4,000 setups (5 changeovers)
- Total costs: $149,400

**RL-HH's costs:**
- $0 MTO skip (completes all 10 batches!)
- $138,800 idle (694 min — very high, roasters sitting idle while RC below safety)
- $6,400 setups (8 changeovers)
- Total costs: $145,200

**Q-Learning**: $336,000 profit with complete MTO, balanced RC management, heuristic restock.

### 4.3 Production Efficiency

| Metric | Q-Learning | PPO | RL-HH |
|--------|-----------|-----|-------|
| PSC batches | ~100 | **117** | 104 |
| Total batches | ~110 | **125** | 114 |
| MTO completion | 100% | **80%** | 100% |
| Roaster utilization | balanced | L2-heavy | conservative |
| RC balance (L1 vs L2) | balanced | **imbalanced** (L1:3-5, L2:19-22) | likely balanced |

**PPO produces the most PSC** (117 vs 104 vs ~100) but **skips 2 MTO batches** (-$100k penalty). If PPO completed all MTO, its profit would be ~$477.6k.

**RL-HH completes ALL MTO** (5 NDG + 5 BUSTA) but has high idle time due to conservative WAIT strategy (81.6% of decisions are WAIT).

**Q-Learning balances everything** — all MTO done, reasonable PSC count, balanced L1/L2.

### 4.4 Interpretability Comparison

**Q-Learning** — Most interpretable:
- Can look up Q(state, action) for any state
- State bins are human-readable (e.g., "time_bin=3, rc_home=low, mto_urgency=urgent")
- Decision rationale: "Q-value for NDG=$47k vs PSC=$42k → pick NDG"

**RL-HH** — Interpretable at tool level:
- Tool selection ratios tell the strategy: "81.6% WAIT, 7.7% PSC, 5.6% SETUP_AVOID"
- Each tool has clear, named logic
- But: "why Tool 2 at this state?" requires examining DDQN Q-values

**PPO** — Least interpretable:
- 153k neural network weights
- Decision: "logit[action=3] = 2.7 > logit[action=20] = 2.3 → pick PSC on R3→L2"
- No semantic explanation for why

### 4.5 Robustness & Generalization

**Q-Learning**: Trained on non-UPS. Fixed discretization may not transfer well to different shift lengths, roaster configurations, or UPS patterns. New state bins would require retraining.

**PPO**: Trained on non-UPS. Continuous features could theoretically generalize, but the reward shaping (rc_bonus, danger zone, completion bonus) is tuned for this specific problem. The learned restock timing is optimized for 480-slot shifts.

**RL-HH**: Trained with UPS (lambda=3). Tool-based architecture is inherently more robust — tools encode domain rules that work across configurations. Adding a new roaster doesn't change the 5-tool action space. Best theoretical generalization.

---

## 5. Key Architectural Differences

### 5.1 Decision Scope

```
Q-Learning:
  Roaster decisions → Q-table (learned)
  Restock decisions → DispatchingHeuristic (hand-coded)

PPO:
  Roaster decisions → Neural network (learned)
  Restock decisions → Neural network (learned)  ← unified

RL-HH:
  Roaster decisions → Heuristic tools (hand-coded, SELECTED by DDQN)
  Restock decisions → GC_RESTOCK tool (hand-coded, SELECTED by DDQN)
```

### 5.2 Learning Target

| | Q-Learning | PPO | RL-HH |
|---|---|---|---|
| Learns | Q(state, action) | π(action\|obs) + V(obs) | Q(state, tool) |
| Output | argmax over 21 actions | sampled from 21-dim softmax | argmax over 5 tools |
| What's learned | Direct action value | Action probability + state value | Tool selection value |
| Domain knowledge | In state bins + restock heuristic | In reward shaping + obs design | In 5 tool implementations |

### 5.3 Constraint Enforcement

| | Q-Learning | PPO | RL-HH |
|---|---|---|---|
| Invalid actions | Pre-filtered before Q-lookup | Hard mask + termination penalty | Tools guarantee valid output |
| RC violation | Soft (reward signal) | Hard (episode terminates, -$50k) | Soft (via tool design) |
| GC violation | Soft (reward signal) | Hard (episode terminates, -$50k) | Prevented by GC_RESTOCK tool |

### 5.4 Sample Efficiency

| | Q-Learning | PPO | RL-HH |
|---|---|---|---|
| Episodes to converge | ~500k | ~50k | ~200k |
| Wall-clock time | 8.3h | 12h | 5h |
| Episodes per second | ~200 | ~2 | ~17 |
| Gradient steps | N/A (tabular) | ~6,000 | ~1.2M |

Q-Learning is fastest per-episode but needs the most episodes. PPO is slowest per-episode but needs the fewest. RL-HH is in between.

---

## 6. Summary

### 6.1 Winner by Metric

| Metric | Winner |
|--------|--------|
| **Highest profit** | **PPO ($377.6k)** |
| MTO completion | Q-Learning & RL-HH (both 100%) |
| Lowest idle cost | PPO ($42.4k) |
| Lowest total cost | RL-HH ($145.2k) |
| RC stock balance | Q-Learning (both lines balanced) |
| Interpretability | Q-Learning |
| Robustness to UPS | RL-HH |
| Training speed | Q-Learning (8.3h, simple) |
| Development effort | Q-Learning (simplest to implement) |
| Scalability | RL-HH (constant 5-tool space) |

### 6.2 Recommendation

**For this specific problem (non-UPS):** PPO achieves the highest profit but requires extensive tuning (23 cycles). Q-Learning is the pragmatic choice — competitive profit with minimal development effort and full MTO completion.

**For UPS/reactive scheduling:** RL-HH's tool-based architecture provides the best framework for handling disruptions, with interpretable decisions and guaranteed valid actions.

**For thesis contribution:** The PPO investigation documents a significant finding — the **gradient death problem** in PPO for discrete scheduling, caused by shared policy/value networks + unnormalized rewards. The fix (separate networks + VecNormalize) is a generalizable contribution applicable beyond this specific problem.
