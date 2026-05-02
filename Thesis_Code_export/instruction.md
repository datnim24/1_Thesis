# How to Run

All commands assume your working directory is `Thesis_Code_export/`. Every run
writes a single folder under `Results/` named
`<YYYYMMDD_HHMMSS>_<Method>_<RunName>/`.

---

## 1. Setup

```bash
# Optional: create venv
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

`Input_data/` ships with the canonical configuration (5 roasters, 2 lines,
8-hour shift). Edit the CSVs there to override constants — no code changes needed.

---

## 2. Train one method

Each method exposes a `train.py` invoked as a Python module. All accept a
`--name` (used in the output folder) and most accept `--time` (wall-clock
budget in seconds) or `--seed`.

| Method      | Command                                                                        |
|-------------|--------------------------------------------------------------------------------|
| CP-SAT      | `python -m cpsat_pure.runner --name MyCPSAT --time 600 --seed 42`              |
| Q-Learning  | `python -m q_learning.train --name MyQL --time 600`                            |
| RL-HH       | `python -m rl_hh.train --name MyRLHH cycle --cycle 1 --time-sec 600`           |
| paeng_ddqn_v2 | `python -m paeng_ddqn_v2.train --name V2Smoke --time-sec 60 --seed-base 42`   |

Examples:

```bash
# 5-minute Q-learning smoke
python -m q_learning.train --name QL_Smoke --time 300

# 1-hour RL-HH cycle starting from a previous best
python -m rl_hh.train --name RLHH_Cycle2 cycle \
    --cycle 2 --time-sec 3600 \
    --warm-start Results/<previous>_RLHH_Cycle1/rlhh_cycle1_best.pt
```

Each run produces (inside its `Results/` folder):
- `*_best.pt` / `*_final.pt` — model checkpoints (gitignored)
- `result.json` (or per-cycle aggregate JSONs) — schema-compatible payload
- `training_log.csv` — episode-level metrics (gitignored)
- `report.html` — interactive Plotly dashboard (Q-learning + CP-SAT)

---

## 3. Evaluate a single seed

`evaluate.py` loads a checkpoint and runs one greedy episode through
`SimulationEngine`, writing `result.json` + `report.html` to a new
`Results/<ts>_Eval_<Method>_<Name>/` folder.

```bash
# Q-Learning single-seed evaluation
python -m q_learning.evaluate --file Results/<ts>_QLearning_X/q_table_X.pkl \
    --seed 42 --html

# RL-HH single-seed eval (universal-schema export + HTML report)
python -m rl_hh.evaluate --checkpoint Results/<ts>_RLHH_X/rlhh_cycle1_best.pt \
    --name SingleSeed_C1 --seed 42

# RL-HH multi-seed aggregate (lighter than full 100-seed)
python -m rl_hh.evaluate --checkpoint <pt> --name AggSmoke \
    --aggregate --n-episodes 20 --base-seed 900000
```

---

## 4. Generate a report (`plot_result.py`)

Renders an interactive Plotly dashboard from any `result.json` produced by
the methods above.

```bash
python evaluation/plot_result.py Results/<ts>_<Method>_<Name>/result.json
# Output: report.html alongside the result.json
```

`plot_result.py` searches `Results/` for the latest result if no path is given.
To compare two runs in the same dashboard:

```bash
python evaluation/plot_result.py result.json --compare other_result.json
```

---

## 5. Verify a result (constraint check)

Validates 18 hard constraints (C1–C12, GC-*, RST-*, SKU-DUR, etc.) against a
schedule. Exits 0 if all pass.

```bash
python evaluation/verify_result.py Results/<ts>_<Method>_<Name>/result.json --verbose
```

---

## 6. Run a 100-seed evaluation

`scripts/eval_100seeds.py` evaluates a method across 100 paired UPS seeds and
writes an `aggregate.json` with profit mean/std/median/p25/p75 + per-seed records.

```bash
# Dispatching baseline (no checkpoint needed)
python scripts/eval_100seeds.py --method dispatching --reps 100 --name Disp

# Q-Learning checkpoint
python scripts/eval_100seeds.py --method q_learning \
    --checkpoint Results/<ts>_QLearning_X/q_table_X.pkl \
    --reps 100 --name QL_C5

# RL-HH checkpoint
python scripts/eval_100seeds.py --method rl_hh \
    --checkpoint Results/<ts>_RLHH_X/rlhh_cycle3_best.pt \
    --reps 100 --name RLHH_C3
```

Output: `Results/<ts>_100SeedEval_<Name>_<method>/aggregate.json`.

---

## 7. Compare methods

Combine multiple `aggregate.json` files into one comparison markdown +
Plotly boxplot.

```bash
python scripts/compare_methods.py --name MyComparison \
    Results/<ts>_100SeedEval_Disp_dispatching/aggregate.json \
    Results/<ts>_100SeedEval_QL_C5_q_learning/aggregate.json \
    Results/<ts>_100SeedEval_RLHH_C3_rl_hh/aggregate.json
```

Output: `Results/<ts>_Compare_MyComparison/comparison.md` + `comparison_boxplot.html`.

---

## 8. Run Block-B factorial (3λ × 3μ × N methods)

```bash
python evaluation/block_b_runner.py --name BlockB_Run1 --reps 100 \
    --methods dispatching q_learning rl_hh \
    --checkpoint-q-learning <pkl> --checkpoint-rl-hh <pt>

# After cells finish:
python evaluation/block_b_analysis.py \
    --input-dir Results/<ts>_BlockB_BlockB_Run1/
```

---

## 9. Master comparison

Sequentially trains/evaluates every non-skipped method under one shared run
name, then writes a consolidated `Master_Evaluation.md` and copies each
method's `result.json` + `report.html` into a single folder.

```bash
# 10-minute smoke per method
python evaluation/master_eval.py --name FullRun --time 600 --seed 42

# Skip RL-HH (long-running)
python evaluation/master_eval.py --name Quick --time 120 --skip rlhh

# Only CP-SAT and Dispatch
python evaluation/master_eval.py --name CompareRules --time 600 --only cpsat ql
```

Output: `Results/<ts>_MasterEval_<Name>/`.

---

## 10. Output convention recap

```
Results/<YYYYMMDD_HHMMSS>_<Method>_<RunName>/
```

Method labels are stable across the codebase:

| Label             | Source                                |
|-------------------|---------------------------------------|
| `CPSAT`           | `cpsat_pure.runner`                   |
| `QLearning`       | `q_learning.train`                    |
| `RLHH`            | `rl_hh.train`                         |
| `PaengDDQNv2`     | `paeng_ddqn_v2.train` (stub)          |
| `Eval_<Method>`   | per-method `evaluate.py` single-seed  |
| `100SeedEval`     | `scripts/eval_100seeds.py`            |
| `MasterEval`      | `evaluation/master_eval.py`           |
| `BlockB`          | `evaluation/block_b_runner.py`        |
| `Compare`         | `scripts/compare_methods.py`          |

The convention is enforced by `evaluation.result_schema.make_run_dir(method, run_name)`.
