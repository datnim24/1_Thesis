# Thesis — Hot Coffee Roasting Scheduling under UPS Disruptions

Reactive scheduling for a continuous-roasting line that produces both
make-to-stock (MTS) and make-to-order (MTO) batches across paired roasters
(R1–R5) feeding two packaging lines (L1, L2). Random Unscheduled Power Stops
(UPS) interrupt roasting mid-batch; the controller must keep packaging lines
fed (PSC restocks), respect MTO due dates (NDG, BUSTA), and minimize total
cost (tardiness + setup + stockout + idle + overflow).

This export is the **clean, reproducible artifact** for the thesis. The full
working tree (with experiments, scratch outputs, and legacy code) lives in
the parent repository.

## Methods compared

| Folder              | Method                        | Role                                  |
|---------------------|-------------------------------|---------------------------------------|
| `cpsat_pure/`       | CP-SAT (UPS-as-downtime)      | Optimal-with-perfect-information ceiling |
| `dispatch/`         | Urgency-first dispatching     | Rule-based baseline                   |
| `q_learning/`       | Tabular Q-learning            | Classical RL baseline                 |
| `rl_hh/`            | Dueling DDQN hyper-heuristic  | Canonical RL method                   |
| `paeng_ddqn_v2/`    | Period-based parameter-sharing DDQN | Paeng 2021 port (FBS state, dueling) |

## Layout

```
Thesis_Code_export/
├── README.md                  ← you are here
├── instruction.md             ← step-by-step run guide
├── requirements.txt
│
├── Context/                   ← thesis docs (problem statement, math model, papers)
├── Input_data/                ← roasters, jobs, SKUs, shift parameters (CSVs)
│
├── env/                       ← simulation engine + action/observation specs
├── dispatch/                  ← baseline dispatching heuristic
├── cpsat_pure/                ← CP-SAT solver + CLI runner
├── q_learning/                ← train.py + evaluate.py
├── rl_hh/                     ← train.py + evaluate.py (Dueling DDQN)
├── paeng_ddqn_v2/             ← train.py + evaluate.py (stubs; in dev)
│
├── evaluation/                ← result_schema, verify_result, plot_result,
│                                master_eval, block_b_runner, block_b_analysis
├── scripts/                   ← eval_100seeds.py, compare_methods.py
│
└── Results/                   ← every run lands here (gitignored)
    └── <YYYYMMDD_HHMMSS>_<Method>_<RunName>/
```

## Output convention

Every method writes to:

```
Results/<YYYYMMDD_HHMMSS>_<Method>_<RunName>/
```

Method labels: `CPSAT`, `Dispatch`, `QLearning`, `RLHH`, `PaengDDQNv2`,
`MasterEval`, `100SeedEval`, `BlockB`, `Eval_<MethodName>`, `Compare`.

Examples:
```
Results/20260501_223152_QLearning_TestQL1/
Results/20260502_203432_RLHH_RLHHWorking1/
Results/20260503_203233_100SeedEval_Test100Seed_rl_hh/
Results/20260504_102452_Eval_QLearning_TestQL1/
```

This is wired through `evaluation.result_schema.make_run_dir(method, run_name)`,
which every CLI entry point calls.

## Quick start

```bash
pip install -r requirements.txt

# 60-second smoke training of all methods + comparison report
python evaluation/master_eval.py --name FirstRun --time 60 --seed 42 --skip paengv2

# Open the report
xdg-open Results/<latest>_MasterEval_FirstRun/Master_Evaluation.md
```

See [instruction.md](instruction.md) for full per-method commands and the
100-seed / Block-B factorial workflows.
