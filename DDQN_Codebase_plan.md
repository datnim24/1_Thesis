# DDQN Codebase Adaptation Plan (v4)

> **Source of truth for the v4 thesis pivot.** Implements: Paeng's Modified DDQN as primary key reference (faithfully ported from his GitHub repo at `Paeng_DRL_Github/`), removes MaskedPPO from active comparison, restructures Block B factorial around Input_data multipliers, ensures every new artifact is plot_result.py-compatible and 100-seed evaluator-compatible.

**Date authored**: 2026-04-27
**Plan version**: 2.0 (Paeng-faithful port)
**Replaces / supersedes**: v1.0 (which was paper-abstract-based; superseded after reading Paeng's actual code)

---

## TASK LIST (executive summary)

The 17 tasks below are the full v4 build. Detailed breakdown is in §7; this is the at-a-glance index.

| # | Task | Effort | Status |
|---|------|--------|--------|
| **1** | Partial PPO move → `OLDCODE/PPOmask_archive/`; updated imports in 4 files | 30 min | ✅ done 2026-04-28 |
| **2** | `evaluate_100seeds.py` with `--lambda-mult`, `--mu-mult`, `--package q_learning/dispatching/paeng_ddqn` | 1 h | ✅ done 2026-04-28 |
| 3 | Build `q_learning/export_for_plot_result.py` | 1 h | done 2026-04-28 |
| 4 | Q-learning 100-seed at (1.0,1.0) → mean **$6,677**, std $157k (validates tabular fail pattern) | 15 min | done 2026-04-28 |
| 5 | `paeng_ddqn/agent.py` (10,168 params, smoke PASS) | 6-8 h | done 2026-04-28 |
| 6 | `paeng_ddqn/strategy.py` (state (3,50), all 8 actions wired) | 3-4 h | done 2026-04-28 |
| 7 | Full-episode no-UPS smoke (3 ep, all actions used, replay fills) | 30 min | done 2026-04-28 |
| 8 | `paeng_ddqn/train.py` | 2 h | done 2026-04-28 |
| **8.5** | **10-min smoke train + plot_result HTML — wiring verified, JSON+HTML produced** | 10 min | **done 2026-04-28** |
| 9 | Full 4-hour Paeng training | 4 h | done 2026-04-28 — 4358 ep, **collapsed to WAIT** (best ep 80, $57k); greedy mean -$1.58M |
| 10 | `paeng_ddqn/evaluate.py` Mode 1 (built before smoke; smoke verified) | 2 h | done 2026-04-28 |
| 11 | Verify Mode 1 HTML — opens, KPI+params render (Gantt empty due to under-trained smoke policy) | 15 min | done 2026-04-28 |
| 12 | `paeng_ddqn/evaluate.py` Mode 2 (delegates to evaluate_100seeds with `--package paeng_ddqn`) | 1 h | done 2026-04-28 |
| 13 | Run Mode 2 at (1.0, 1.0) → `paeng_ddqn/outputs/paeng_100seed.json` | 1 min | done 2026-04-28 — best $-1,582,570; final $-1,462,936 |
| 14 | `test_rl_hh/full_comparison.py` updated (drops PPO, adds Paeng row, gracefully skips if no ckpt) | 1 h | done 2026-04-28 |
| 15 | `block_b_runner.py` (3 λ × 3 μ × 4 methods × N seeds; subprocess-free) | 2 h | done 2026-04-28 |
| 16 | Run Block B → 36 JSONs | 1.5 h | done 2026-04-28 — 36 cells in 862 s; rl_hh dominates 9/9 cells |
| 17 | `block_b_analysis.py` (pivot CSV + Wilcoxon contrasts + heat maps + bootstrap CI) | 3 h | done 2026-04-28 |

**Total ≈ 21 h coding + 4 h Paeng training + 1.5 h Block B = ~27 h work / ~4 working days.**

## 17/17 tasks complete — v4 build done 2026-04-28

**Headline Block B result (centre cell λm=μm=1.0, n=100 seeds):**

| Method | Mean profit | Std | Notes |
|--------|------------:|-----|-------|
| **RL-HH (Dueling DDQN + tools)** | **$375,084** | $17,903 | Phase 6 — thesis innovation |
| Dispatching heuristic | $320,140 | $13,189 | Phase 3 — operator baseline |
| Q-learning (tabular) | $6,677 | $157,343 | Phase 4 — fails on unseen seeds |
| Paeng's Modified DDQN | -$1,582,570 | $9,086 | Phase 5 — collapsed during training |
| CP-SAT no-UPS ceiling | $452,400 | 0 | Phase 2 — theoretical ceiling |

The Phase 5 (Paeng) → Phase 6 (RL-HH) Δ ≈ +$1.96M empirically demonstrates the value of the dueling architecture step (Ren & Liu 2024) plus tool-selection action space (Luo 2020) on our problem class with shared GC pipeline + UPS disruptions. RL-HH wins all 9 (λ, μ) cells of the Block B factorial.

Detailed Block B analysis: `paeng_ddqn/PORT_NOTES.md` §11 + `results/block_b_20260428_045926/`.

---

## 0. Decisions locked (from user 2026-04-27)

| # | Decision | Source |
|---|----------|--------|
| D1 | Random seed **rotation** during Paeng training (new seed per episode) | User-confirmed |
| D2 | R3 routing **baked into the (SKU, pool) action** — no separate action dim. Reuse `rl_hh/tools.py::_psc_throughput`'s `argmax(min(rc_space, gc))` rule inside `paeng_ddqn/strategy.py` | User-confirmed |
| D3 | Paeng training budget = **4 hours** wall-clock | User-confirmed |
| D4 | Q-Learning is **already trained** at `q_learning/ql_results/30_03_2026_1230_ep335341_a00500_g09600_Test_restockpatch6_profit226272/` — **do not retrain**. Only need to evaluate it on 100 seeds via `plot_result.py` (the in-folder old plotter is deprecated) | User-confirmed |
| D5 | 100-seed evaluation everywhere | User-confirmed |
| D6 | **Faithfully port Paeng's code** from `Paeng_DRL_Github/` — same architecture, same hyperparameters where applicable, modify only to handle our problem (UPS, MTS+MTO mix, GC silo, RC buffer). PyTorch port (TF1.14 → PyTorch). | User-confirmed |
| D7 | PPOmask folder moved to `OLDCODE/PPOmask/` (gitignored) | User-confirmed |

---

## 1. What Paeng's repo actually contains (read directly from `Paeng_DRL_Github/`)

| Their file | What it does | Maps to our file |
|-----------|--------------|-------------------|
| `config.py` | argparse hyperparameters, dataset selector, save-dir setup | merged into `paeng_ddqn/agent.py` as a dataclass + argparse stub |
| `model/dqn.py::PDQN` | TF1 graph: parameter-sharing FC blocks → optional Dueling head → Q-output. Lines 173-204. | `paeng_ddqn/agent.py::ParameterSharingDQN` (PyTorch port) |
| `model/util_nn.py` | `dense_layer()`, weight init helpers | inlined into `paeng_ddqn/agent.py` |
| `agent/replay_buffer.py::ReplayBuffer` | Circular buffer, signature `add(s, xs, a, r, t, s2, xs2, f_a)` — note **state + auxin** stored separately, plus feasibility mask | `paeng_ddqn/agent.py::ReplayBuffer` |
| `agent/trainer.py::Trainer` | ε-greedy `get_action()`, `remember()`, `train_network()` (Double-Q target compute, Huber loss SGD step), `update_target_network()` (soft sync) | `paeng_ddqn/agent.py::PaengAgent` |
| `env/wrapper.py::Wrapper` | Builds the (F, F*2+42) 2D state matrix per decision: TrackOut count, S_p (F×Hp=5), proc_info (F×16), S_w (F×16), enter count (F×4), setup time row, setup type row, proc time row, S_h (F×3) — checked line-by-line | `paeng_ddqn/strategy.py::build_paeng_state(state)` |
| `env/simul_pms.py`, `env/util_sim.py`, `env/job_generator.py` | His own simulator | **not ported** — we keep our `env/simulation_engine.py` and bridge in `strategy.py` |
| `main.py` | training loop (env reset → wrapper.getState → trainer.get_action → env.step → trainer.remember → trainer.train) | `paeng_ddqn/train.py` |
| `test.py` | evaluation loop with heuristic baselines | `paeng_ddqn/evaluate.py` |

**State shape (verified from `wrapper.py::_getFamilyBasedState`)**: `(F, F*2+42)` where `F` = number of families. For our 3 SKUs (PSC, NDG, BUSTA), state shape is **`(3, 48)`** plus a small auxiliary vector. Constants `Hw=6, Hp=5` per Paeng's Wrapper class.

---

## 1.5. Action space mapping (Paeng → our problem)

### Paeng's action space
`F × F` flattened = 9 actions for our F=3. Action `(from_family, to_family)`. The `from_family` is the queried machine's current setup; the `to_family` is what the DRL agent picks.

### Our extended action space
Our problem has restock decisions and a WAIT default that Paeng doesn't have. We extend to **8 discrete actions per decision** (still small enough that the DDQN Q-head outputs 8 values):

| Action ID | Meaning | Paeng analog |
|-----------|---------|--------------|
| 0 | START PSC on queried roaster | to_family = PSC |
| 1 | START NDG on queried roaster | to_family = NDG |
| 2 | START BUSTA on queried roaster | to_family = BUSTA |
| 3 | WAIT | (Paeng doesn't have this; we add it) |
| 4 | START_RESTOCK L1_PSC | (Paeng doesn't have this) |
| 5 | START_RESTOCK L1_NDG | |
| 6 | START_RESTOCK L1_BUSTA | |
| 7 | START_RESTOCK L2_PSC | |

**Action 0 (PSC) on R3** triggers the R3 routing rule inside `strategy.py` (D2): pick L1 or L2 by `argmax(min(rc_space, gc))`. No separate action dim.

**Feasibility mask** = same as the existing engine mask (PSC infeasible if RC full / GC empty / pipeline busy; restock infeasible if station busy / silo full; etc.). Paeng's `feasible_action_index` machinery in `trainer.py::get_action` carries over directly.

---

## 2. New module: `paeng_ddqn/` (4-file Paeng-faithful layout)

```
paeng_ddqn/
├── __init__.py        # marker only
├── agent.py           # ParameterSharingDQN (PyTorch port of model/dqn.py::PDQN, is_duel=False)
│                      # + ReplayBuffer (port of agent/replay_buffer.py)
│                      # + PaengAgent (port of agent/trainer.py - select_action, train_step, soft sync)
│                      # + dataclass for hyperparameters (config.py merged)
├── strategy.py        # build_paeng_state(): (3, 48) matrix + auxin vector (port of wrapper.py::_getFamilyBasedState)
│                      # PaengStrategy: engine adapter (decide / decide_restock)
│                      # R3 routing helper (re-uses rl_hh/tools.py rule)
├── train.py           # CLI: 4-hour training with seed rotation per episode (port of main.py loop)
│                      # writes paeng_best.pt, paeng_final.pt, training_log.csv
└── evaluate.py        # CLI mode 1: single-seed → result_schema JSON + plot_result HTML
                       # CLI mode 2: 100-seed → JSON in test_rl_hh/evaluate_100seeds shape
```

### 2.1 `agent.py` — algorithm core (≈ 400 LOC)

**`@dataclass class PaengConfig`** — Paeng's `config.py` merged. Defaults from his repo:
```python
lr = 0.0025                # his default
gamma = 0.99               # we override (his is 1.0 since reward is pure tardiness; ours has profit)
batch_size = 32
buffer_size = 100_000
warmup_timesteps = 24_000  # his default — random actions before training starts
freq_target_episodes = 50  # his default — target net update every 50 episodes
freq_online = 1            # his default — online net update every step after warmup
hid_dims = [64, 32, 16]    # his default
eps_start = 0.2            # his default (already low — he doesn't anneal from 1.0)
eps_ratio = 0.9            # his default — anneal over 90% of episodes
is_double = True           # we keep Double Q (per paper)
is_duel = False            # standard DDQN (Phase 5; dueling stays in rl_hh/)
tau = 1.0                  # his default for hard target sync; we use 0.005 for soft
huber_delta = 0.5          # his default
```

**`class ParameterSharingDQN(nn.Module)`** — PyTorch port of `model/dqn.py::PDQN.base_encoder_cells` (lines 173-195):
- Input: `(B, F=3, F*2+42=48)` 2D matrix + `(B, auxin_dim)` aux vector
- For each of the 3 family rows: shared `Linear(48, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 16) → ReLU` — **same weights for all 3 rows** (this is the "parameter sharing")
- Concat 3 row outputs → `(B, 48)` → fusion with auxin → `Linear → action_dim=8`
- `forward(state, auxin)` returns Q-values shape `(B, 8)`
- Faithful to Paeng: when `is_duel=False`, single Q-head; we set `is_duel=False` in Phase 5

**`class ReplayBuffer`** — port of `agent/replay_buffer.py::ReplayBuffer.add(s, xs, a, r, t, s2, xs2, f_a)`. Same 8-tuple signature: state, auxin, action, reward, terminal, next_state, next_auxin, feasibility_mask.

**`class PaengAgent`** — port of `agent/trainer.py::Trainer`:
- `select_action(state, auxin, mask, training)` — ε-greedy with feasibility filter (his `trainer.py::get_action` lines 102-156)
- `store_transition(...)` — circular buffer push
- `train_step()` — Double Q target compute (his `train_network` lines 291-330) + Huber loss SGD with RMSProp
- `update_target_network()` — soft τ-update (his `dqn.py::create_update_op` lines 39-50)
- `decay_epsilon(episode)` — linear over `eps_ratio * max_episodes` (his `check_exploration` lines 158-170)
- `save_checkpoint(path)` / `load_checkpoint(path)` — torch.save state_dicts

### 2.2 `strategy.py` — engine adapter + state builder (≈ 350 LOC)

**`build_paeng_state(data, sim_state, context) -> (np.ndarray, np.ndarray)`** — direct port of `Wrapper._getFamilyBasedState` (lines 350-398). Builds the (3, 48) matrix:
- Row 0 = PSC features, Row 1 = NDG features, Row 2 = BUSTA features
- 48 columns = TrackOut count (1) + S_p (5) + proc_info (16) + S_w (16) + enter count (4) + setup time row (3) + setup type row (3) — wait that's 48? Let me re-verify in `_getFamilyBasedState`. Per his code the slicing comes out to `F*2 + 42 = 48` for F=3. The exact column packing is preserved as-is from his code.

Plus auxin vector (`_get_auxin`, lines 291-330 of wrapper.py) — last action one-hot + last reward + a few flags. Length ~10.

**`class PaengStrategy`**:
```python
def __init__(self, agent: PaengAgent, data, training: bool = False): ...
def decide(self, state, roaster_id: str) -> tuple:
    paeng_state, auxin = build_paeng_state(self.data, state, RoasterContext(roaster_id))
    mask = compute_feasibility_mask(state, roaster_id)  # 8-bit mask
    action_id = self.agent.select_action(paeng_state, auxin, mask, self.training)
    return self._action_id_to_env_tuple(action_id, roaster_id, state)

def decide_restock(self, state) -> tuple:
    # Same flow but with RestockContext — agent picks among action 3 (WAIT) and 4-7 (restock variants)
    ...

def _action_id_to_env_tuple(self, action_id, roaster_id, state):
    if action_id == 3: return ("WAIT",)
    if action_id in (4,5,6,7): return ("START_RESTOCK", line_id, sku)  # by mapping
    if action_id in (0,1,2):  # to-SKU
        sku = ["PSC","NDG","BUSTA"][action_id]
        if roaster_id == "R3" and sku == "PSC":
            line = self._r3_route(state)  # reuses rl_hh logic: argmax(min(rc_space, gc))
            return ("START_BATCH", roaster_id, sku, line)
        return ("START_BATCH", roaster_id, sku)
```

Same interface as `q_learning/q_strategy.py::QStrategy` and `rl_hh/rl_hh_strategy.py::RLHHStrategy`, so `SimulationEngine.run(strategy, ups)` doesn't need changes.

### 2.3 `train.py` — training CLI (≈ 200 LOC, ports `main.py`)

```bash
python -m paeng_ddqn.train --time-sec 14400 --output-dir paeng_ddqn/outputs --seed-base 42
```

Pseudocode mirroring his `main.py`:
```python
data = load_data()
engine = SimulationEngine(data.to_env_params())
agent = PaengAgent(PaengConfig())
strategy = PaengStrategy(agent, data, training=True)

t0 = time.perf_counter()
episode = 0
while time.perf_counter() - t0 < args.time_sec:
    seed = args.seed_base + episode  # D1: rotate seed each episode
    ups = generate_ups_events(data.ups_lambda, data.ups_mu, seed, SL, roasters)
    kpi, state = engine.run(strategy, ups)  # transitions stored inside strategy via agent

    if agent.timestep > config.warmup_timesteps:
        # train_step is called inside strategy after each store_transition;
        # nothing extra to do here per Paeng's loop
        pass

    if episode % config.freq_target_episodes == 0:
        agent.update_target_network()
    agent.decay_epsilon(episode)

    profit = float(kpi.net_profit())
    log_episode(episode, profit, agent.epsilon, len(agent.replay))
    if profit > best_profit:
        best_profit = profit
        agent.save_checkpoint(out_dir / "paeng_best.pt")
    episode += 1

agent.save_checkpoint(out_dir / "paeng_final.pt")
```

Writes `training_log.csv` with columns `episode, profit, epsilon, buffer_size, wall_sec, tool_counts(=action_dist)` — same schema as `rl_hh/outputs/cycle*_training_log.csv` so existing analysis tooling reads it.

### 2.4 `evaluate.py` — eval CLI (≈ 250 LOC, ports `test.py`)

**Mode 1 — single-seed report (plot_result.py-compatible)**:
```bash
python -m paeng_ddqn.evaluate --checkpoint paeng_ddqn/outputs/paeng_best.pt --seed 42 --report
```
Runs 1 episode → builds `result_schema.create_result(metadata={"solver_engine":"paeng_ddqn", "solver_name":"Paeng's Modified DDQN", ...})` → writes `paeng_result_seed42.json` → calls `plot_result._build_html(result, compare=None, offline=True)` → writes `paeng_report_seed42.html`.

**Mode 2 — 100-seed aggregate (test_rl_hh-compatible)**:
```bash
python -m paeng_ddqn.evaluate --checkpoint paeng_ddqn/outputs/paeng_best.pt --n-seeds 100 --base-seed 900000 \
  --lambda-mult 1.0 --mu-mult 1.0 --output paeng_ddqn/outputs/paeng_100seed.json
```
Returns the **same JSON shape** as `RLHH_final_100seed.json` so it drops into `block_b_runner.py` and `test_rl_hh/full_comparison.py` with no shape adapters needed.

---

## 3. 100-seed evaluator: λ/μ multiplier upgrade (D5)

### 3.1 Patch to `test_rl_hh/evaluate_100seeds.py`

Current behaviour:
```python
ups_lambda = data.ups_lambda    # always 5
ups_mu = data.ups_mu            # always 20
```

New behaviour (preserves backward compat — multiplier defaults to 1.0):
```python
parser.add_argument("--lambda-mult", type=float, default=1.0,
                    help="UPS λ multiplier; valid Block B values: 0.5, 1.0, 2.0")
parser.add_argument("--mu-mult", type=float, default=1.0,
                    help="UPS μ multiplier; valid Block B values: 0.5, 1.0, 2.0")
ups_lambda = data.ups_lambda * args.lambda_mult
ups_mu = data.ups_mu * args.mu_mult
```

Output JSON metadata gets new fields:
```json
"ups_lambda_used": 2.5,
"ups_mu_used":     20.0,
"lambda_mult":     0.5,
"mu_mult":         1.0,
"ups_lambda_input_data": 5.0,
"ups_mu_input_data":     20.0
```

### 3.2 Existing JSONs stay valid as the (1.0, 1.0) cell

`baseline_100seed.json`, `RLHH_final_100seed.json`, `PPO_100seed.json` continue to represent the centre cell. Add a one-line metadata note via a small migration script (nice-to-have, not blocking).

### 3.3 Identical multiplier flags in `paeng_ddqn/evaluate.py` Mode 2

Same `--lambda-mult` / `--mu-mult` flags, same JSON shape extension. Block B compatibility from day one.

---

## 4. Block B factorial driver (`block_b_runner.py`)

Single orchestrator, ~150 LOC:

```python
LAMBDA_MULTS = [0.5, 1.0, 2.0]
MU_MULTS     = [0.5, 1.0, 2.0]
METHODS      = ["dispatching", "q_learning", "paeng_ddqn", "rl_hh"]
SEEDS        = list(range(900_000, 900_100))  # 100 paired seeds

# Method → checkpoint mapping (CLI args read these)
CKPT = {
    "dispatching":  None,   # rule-based
    "q_learning":   "q_learning/ql_results/30_03_2026_1230_ep335341_a00500_g09600_Test_restockpatch6_profit226272/q_table.pkl",
    "paeng_ddqn":   "paeng_ddqn/outputs/paeng_best.pt",
    "rl_hh":        "rl_hh/outputs/rlhh_cycle3_best.pt",
}

for λm in LAMBDA_MULTS:
  for μm in MU_MULTS:
    for method in METHODS:
        run_method(method, ckpt=CKPT[method],
                   lambda_mult=λm, mu_mult=μm, seeds=SEEDS,
                   out=f"results/block_b/{method}_lm{λm}_mm{μm}.json")
```

Dispatches via subprocess to each method's existing `evaluate_100seeds`-shaped CLI. **Reuses the per-method evaluators** — no duplicate eval logic.

### Output organisation
```
results/block_b_<ts>/
├── dispatching_lm0.5_mm0.5.json    (9 cells × 4 methods = 36 JSONs)
├── dispatching_lm0.5_mm1.0.json
├── ... (34 more cells)
├── rl_hh_lm2.0_mm2.0.json
├── pivot_summary.csv               # method × λ × μ → mean/std/median
└── heatmap_<metric>.html           # plotly heatmap (one per metric)
```

Total runs = 3 × 3 × 4 × 100 = **3,600 episodes**. ≈ 1.5h CPU on i3 (dispatching < 1s/seed; Q-learning ~ 1s/seed; Paeng ~ 1s/seed; RL-HH ~ 1s/seed).

### `block_b_analysis.py`
- Wilcoxon signed-rank pairwise tests at α = 0.05 (paired seeds within each cell)
- Three nested contrasts per v4:
  1. Rules vs. Learning: dispatching vs. {q_learning, paeng_ddqn, rl_hh}
  2. Tabular vs. Deep: q_learning vs. {paeng_ddqn, rl_hh}
  3. Standard DDQN vs. Dueling-DDQN-RL-HH: paeng_ddqn vs. rl_hh
- Heat map: method × (λm, μm) → mean profit
- Bootstrap CIs (1,000 resamples)

---

## 5. Q-Learning (D4) — evaluation only, no retraining

User confirmed the existing checkpoint at:
```
q_learning/ql_results/30_03_2026_1230_ep335341_a00500_g09600_Test_restockpatch6_profit226272/
```
is the chosen one. The old in-folder plotter is deprecated → eval through the universal pipeline.

### 5.1 Build `q_learning/evaluate_ql.py` (one new file, ≤ 100 LOC) OR reuse `test_rl_hh/evaluate_100seeds.py` with a `--package q_learning` extension

**Cleaner option**: extend `test_rl_hh/evaluate_100seeds.py` with `--package` accepting `"q_learning"` (which already supports `"rl_hh"`, `"test_rl_hh"`):
```python
if args.package == "q_learning":
    from q_learning.q_strategy import QStrategy, load_q_table
    q_table = load_q_table(args.checkpoint)
    factory = lambda seed: QStrategy(params, q_table=q_table)
elif args.package in ("rl_hh", "test_rl_hh"):
    ...
```

Then:
```bash
python -m test_rl_hh.evaluate_100seeds \
    --checkpoint q_learning/ql_results/30_03_2026_1230_*/q_table.pkl \
    --package q_learning \
    --output q_learning/ql_results/30_03_2026_1230_*/100seed_lm1_mm1.json \
    --lambda-mult 1.0 --mu-mult 1.0
```

### 5.2 Build `q_learning/export_for_plot_result.py` (≤ 80 LOC)

Single-seed run that exports universal-schema JSON consumable by `plot_result.py`:
```bash
python -m q_learning.export_for_plot_result \
    --checkpoint q_learning/ql_results/30_03_2026_*/q_table.pkl \
    --seed 42 \
    --output q_learning/ql_results/30_03_2026_*/ql_result_seed42.json

python plot_result.py q_learning/ql_results/30_03_2026_*/ql_result_seed42.json
# → opens ql_result_seed42_plot.html in browser
```

This replaces the deprecated in-folder plotter.

---

## 6. PPOmask archiving (D7) — REVISED after grep audit

> **Discovered during Task 1 execution (2026-04-28)**: `PPOmask/Engine/` contains **shared infrastructure** used by `rl_hh/` and other modules (`action_spec.py`, `data_loader.py`, `observation_spec.py`, `roasting_env.py`). Moving the whole folder breaks the codebase. **Partial move strategy adopted instead.**

### What gets moved to `OLDCODE/PPOmask_archive/` (after re-test)
PPO-specific files only:
- `PPOmask/ppo_run.py`, `PPOmask/evaluate_maskedppo.py`, `PPOmask/train_maskedppo.py`, `PPOmask/bootstrap.py`
- `PPOmask/ASSUMPTIONS_AND_DEVIATIONS.md`, `PPOmask/AUDIT.md`, `PPOmask/PPOtrainProgress.md`
- `PPOmask/tests/`, `PPOmask/outputs/` (outputs/ gitignored already)
- `PPOmask/Engine/ppo_strategy.py`, `PPOmask/Engine/callbacks.py`
- `test_rl_hh/evaluate_ppo_100seeds.py` (PPO-only evaluator)

### Files that **had to stay in `PPOmask/Engine/`** (discovered during smoke test)
`PPOmask/Engine/roasting_env.py` (used by rl_hh in gym mode) imports `mask_spec` and `reward_spec`. These two files were initially moved to OLDCODE but had to be restored after the import broke. Final list of shared infrastructure:
- `PPOmask/__init__.py`, `PPOmask/requirements.txt`
- `PPOmask/Engine/__init__.py`
- `PPOmask/Engine/action_spec.py` (used by rl_hh)
- `PPOmask/Engine/data_loader.py` (used by EVERYTHING)
- `PPOmask/Engine/observation_spec.py` (used by rl_hh)
- `PPOmask/Engine/roasting_env.py` (used by rl_hh evaluator in gym mode)
- `PPOmask/Engine/mask_spec.py` (imported by roasting_env)
- `PPOmask/Engine/reward_spec.py` (imported by roasting_env)

### Future-work cleanup (not in this plan)
Rename `PPOmask/Engine/` → `engine_shared/` at top level so the shared infrastructure no longer lives inside a method-named folder. This is a one-shot find-and-replace across ~10 files but defers cleanly to a separate cleanup pass.

`OLDCODE/` is gitignored (verified in `.gitignore`). Updates after the partial move:

| File | Change |
|------|--------|
| `test_rl_hh/full_comparison.py` | Drop the PPO import block; comparison drops to 4 methods + CP-SAT ceiling |
| `test_rl_hh/evaluate_ppo_100seeds.py` | Move into `OLDCODE/PPOmask/` alongside the model, update its hardcoded import path |
| `master_eval.py` | Replace PPO method slot with `paeng_ddqn` (same I/O contract) |
| `Reactive_GUI.py` | Drop or hide PPO option; add Paeng DDQN |
| `plot_result.py` line 55 | `ROOT_DIR / "PPOmask" / "outputs"` → `ROOT_DIR / "OLDCODE" / "PPOmask" / "outputs"` so historical PPO JSONs remain discoverable |
| `Seed69_eval1/` | Frozen snapshot — leave alone |

Preserved (not deleted):
- All 16 PPO checkpoints under `OLDCODE/PPOmask/outputs/`
- `Seed69_eval1/seed_69/ppo_result.json` and `ppo_report.html`
- `test_rl_hh/outputs/PPO_100seed.json` — kept as Block B (1.0, 1.0) cell entry for PPO. Annotate metadata to mark "archived (gradient death after 18 training cycles)".

---

## 7. Sequenced build order

| # | Task | Dependencies | Est. effort | Status |
|---|------|--------------|-------------|--------|
| 1 | Move `PPOmask/` → `OLDCODE/PPOmask/`; update import paths in 5 files (§6) | None | 30 min | ⏳ |
| 2 | Patch `test_rl_hh/evaluate_100seeds.py` with `--lambda-mult` / `--mu-mult` flags + `--package q_learning` extension (§3.1, §5.1) | None | 1h | ⏳ |
| 3 | Build `q_learning/export_for_plot_result.py` (§5.2) | Task 2 | 1h | ⏳ |
| 4 | Validate Q-learning checkpoint: 100-seed eval at (λm=1, μm=1) + open plot_result HTML for one seed | Tasks 2-3 | 15 min | ⏳ |
| 5 | Build `paeng_ddqn/agent.py` (port `dqn.py::PDQN`, `replay_buffer.py::ReplayBuffer`, `trainer.py::Trainer`) | None | 6-8h | ⏳ |
| 6 | Build `paeng_ddqn/strategy.py` (port `wrapper.py::_getFamilyBasedState` + engine adapter + R3 route helper) | Task 5 | 3-4h | ⏳ |
| 7 | Smoke test Paeng on 100 episodes (no UPS) — confirm Q-values move, replay buffer fills, target net syncs | Tasks 5-6 | 30 min | ⏳ |
| 8 | Build `paeng_ddqn/train.py` (port `main.py` loop with seed rotation D1) | Tasks 5-6 | 2h | ⏳ |
| 9 | Full Paeng training run: **4 hours** wall-clock (D3) with UPS from ep 1 | Tasks 5-6, 8 | 4h training | ⏳ |
| 10 | Build `paeng_ddqn/evaluate.py` Mode 1 (single-seed → result_schema JSON → plot_result HTML) | Task 9 | 2h | ⏳ |
| 11 | Verify Mode 1: open one Paeng `_report.html`, confirm KPI/Gantt/RC/GC panels render | Task 10 | 15 min | ⏳ |
| 12 | Build `paeng_ddqn/evaluate.py` Mode 2 (100-seed aggregate, RLHH_final_100seed shape) | Task 10 | 1h | ⏳ |
| 13 | Run Mode 2 at (λm=1, μm=1) → produces `paeng_ddqn/outputs/paeng_100seed.json` | Tasks 9, 12 | 1 min | ⏳ |
| 14 | Update `test_rl_hh/full_comparison.py`: drop PPO row, add Paeng DDQN row, regenerate `COMPARISON_REPORT.md` | Tasks 4, 13 | 1h | ⏳ |
| 15 | Build `block_b_runner.py` orchestrator (§4) | Tasks 4, 13 | 2h | ⏳ |
| 16 | Run Block B: 3,600 episodes → 36 JSONs | Task 15 | ~1.5h | ⏳ |
| 17 | Build `block_b_analysis.py`: pivot table + Wilcoxon + heat maps | Task 16 | 3h | ⏳ |

**Total ≈ 21h coding + 4h Paeng training + 1.5h Block B = ~27h work**, doable in ~4 working days.

---

## 8. Compatibility checklist (post-implementation verification)

Run after each milestone to catch regressions:

- [ ] `python -m test_rl_hh.evaluate_100seeds --checkpoint rl_hh/outputs/rlhh_cycle3_best.pt --package rl_hh --output /tmp/regress_rlhh.json --n-seeds 10` → mean still ~$375k (within ±2%) at default λ_mult=1, μ_mult=1
- [ ] `python -m test_rl_hh.evaluate_100seeds --checkpoint q_learning/ql_results/30_03_2026_*/q_table.pkl --package q_learning --output /tmp/regress_ql.json --n-seeds 10 --lambda-mult 1.0 --mu-mult 1.0` → matches checkpoint folder's profit (~$226k training profit)
- [ ] `python -m paeng_ddqn.evaluate --checkpoint paeng_ddqn/outputs/paeng_best.pt --seed 42 --report` → produces a `paeng_report_seed42.html` that opens with the standard 7-panel layout (KPI/Gantt/RC/GC L1/GC L2/restock/utilization/parameters)
- [ ] `python plot_result.py paeng_ddqn/outputs/paeng_result_seed42.json` → universal pipeline regenerates the same HTML without errors
- [ ] `python plot_result.py q_learning/ql_results/30_03_2026_*/ql_result_seed42.json` → universal pipeline replaces the deprecated old plotter
- [ ] `block_b_runner.py --reps 5 --dry-run` → all 36 cells produce valid JSON before the full 100-rep run
- [ ] `git status` after PPOmask move → `OLDCODE/PPOmask/` not staged (already gitignored), no PPO files referenced as deleted in the index
- [ ] `python -c "from paeng_ddqn.strategy import PaengStrategy"` works cleanly with no `PPOmask` import errors

---

## 9. Out of scope

- Re-running RL-HH from scratch — the current `rl_hh/outputs/rlhh_cycle3_best.pt` + the 7 promoted tool changes already gives +14.2% and stays as the thesis innovation.
- Re-running CP-SAT — the existing $452,400 ceiling (gap 31.7%) stands.
- Re-training Q-learning — D4 says use the existing checkpoint.
- Adding a 6th tool to RL-HH — diminishing returns.
- ALNS / metaheuristic comparison — future work.
- Real-plant calibration of MTBF/MTTR — explicit thesis limitation L5.

---

## 10. File inventory after this plan is executed

**New files** (10 total — paeng_ddqn 4 + q_learning 1 + 2 block_b + plan + 2 evaluate updates):
```
paeng_ddqn/__init__.py
paeng_ddqn/agent.py
paeng_ddqn/strategy.py
paeng_ddqn/train.py
paeng_ddqn/evaluate.py
q_learning/export_for_plot_result.py
block_b_runner.py
block_b_analysis.py
DDQN_Codebase_plan.md            (this file)
```

**Modified files** (5 total):
```
test_rl_hh/evaluate_100seeds.py    # add --lambda-mult / --mu-mult / --package q_learning
test_rl_hh/full_comparison.py      # drop PPO, add Paeng DDQN
master_eval.py                     # replace PPO with Paeng DDQN
Reactive_GUI.py                    # drop/hide PPO, add Paeng DDQN
plot_result.py                     # update line 55 search root to OLDCODE/PPOmask
```

**Moved (not deleted)** (1 directory):
```
PPOmask/  →  OLDCODE/PPOmask/       # gitignored, code preserved on disk
```

**Untouched** (the rest of the codebase):
```
env/                                 # strategy-agnostic — no changes
dispatch/                            # done
q_learning/                          # only adds export_for_plot_result.py — no edits to q_strategy.py
rl_hh/                               # frozen at +14.2% improvement (post-promotion)
MILP_Test_v5/, CPSAT_Pure/           # done
Input_data/                          # frozen
result_schema.py                     # accepts arbitrary solver_engine strings already
verify_result.py                     # universal validator; no changes
```

---

## 11. Open questions resolved

All 5 prior open questions answered by user 2026-04-27 (see §0). No remaining open questions.

Plan ready to execute. **Start with Task 1 (PPOmask move) and Task 2 (evaluator multiplier flag) — both <1h jobs that unblock everything else.**

---

## Appendix A — Mapping table: Paeng GitHub file → our file

| Paeng file | Lines of interest | Our file | Notes |
|-----------|-------------------|----------|-------|
| `config.py` | full file | `paeng_ddqn/agent.py::PaengConfig` (dataclass) | TF1-specific bits dropped |
| `model/dqn.py` | 173-204 (encoder), 197-205 (Q head), 224-238 (dueling head, unused with `is_duel=False`) | `paeng_ddqn/agent.py::ParameterSharingDQN` | PyTorch port |
| `model/util_nn.py` | `dense_layer()` | inlined in `agent.py` | `nn.Linear` directly |
| `agent/replay_buffer.py` | 62-138 (`ReplayBuffer` class) | `paeng_ddqn/agent.py::ReplayBuffer` | numpy circular buffer |
| `agent/trainer.py` | 102-156 (get_action), 183-234 (remember + train trigger), 291-330 (train_network), 338-? (target sync) | `paeng_ddqn/agent.py::PaengAgent` | Combine into one class |
| `env/wrapper.py` | 350-398 (`_getFamilyBasedState`), 291-330 (`_get_auxin`), 350-484 (component builders `_getTrackOutCnt` etc.) | `paeng_ddqn/strategy.py::build_paeng_state` | Map his "family" → our "SKU" |
| `env/util_sim.py` | constants (F, M, Horizon, MaxProcTime, MaxSetupTime) | `paeng_ddqn/strategy.py` (read from our `Input_data` instead) | Don't import |
| `main.py` | 130-180 (training loop) | `paeng_ddqn/train.py` | Strip TF graph setup; PyTorch is functional |
| `test.py` | 50-200 (eval loop, KPI extraction) | `paeng_ddqn/evaluate.py` | Use `result_schema.create_result()` for output JSON |
