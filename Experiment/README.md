# Experiment: CP-SAT Rolling Horizon vs Real-Time RL

## Purpose

Addresses the question: *"Why not use a rolling-horizon exact solver instead of RL?"*

This experiment quantifies the trade-off between solution quality and solver latency
for a rolling-horizon CP-SAT approach (2+2+2+2 hours, 5-min budget per window) vs
the proposed RL-HH method.

## Experiment design

- **Seed**: 69 (same seed used for the CP-SAT oracle reference run)
- **UPS**: λ=5.0 events/shift, μ=20.0 min (same as main comparison)
- **Full-horizon CP-SAT** (480 min) is solved with increasing time budgets:
  `5s, 15s, 30s, 60s, 120s, 300s`
- This simulates the per-window budget of a **4-window rolling horizon** (2+2+2+2 hours)

## How to run

```bash
cd "c:\Local Folder\CODE\1_Thesis"
python Experiment/cpsat_budget_experiment.py
```

Outputs written to `Experiment/results/`:

| File | Contents |
|------|----------|
| `budget_results.json` | Raw numbers per budget |
| `budget_chart.png` | Quality vs budget + rolling overhead chart |
| `report.html` | Self-contained HTML report with table + embedded chart |

Runtime: ~15–30 minutes (6 CP-SAT solves, up to 300s each).

## Key findings (pre-run reference)

| Method | Net Profit | Decision latency |
|--------|----------:|-----------------|
| CP-SAT full (oracle, seed 69) | $443,400 | ~8 h offline |
| RL-HH (100-seed mean) | $359,096 | < 1 ms |
| Dispatching heuristic (100-seed mean) | $326,554 | < 1 ms |

Rolling 4×5-min overhead = **20 min frozen** per 480-min shift (4.2%).
With λ=5 UPS re-solves: up to **45 min frozen** (9.4%).

## Reply draft

See [`reply_draft.md`](reply_draft.md) for a structured response to the advisor's questions.

## Files

```
Experiment/
├── README.md                     ← this file
├── cpsat_budget_experiment.py    ← main experiment script
├── reply_draft.md                ← draft reply to advisor
└── results/                      ← auto-created on first run
    ├── budget_results.json
    ├── budget_chart.png
    └── report.html
```

## Notes

- This experiment **does not modify any existing code** — it only imports from
  `CPSAT_Pure` and `env.ups_generator` (read-only).
- The CP-SAT model used here is the same full-horizon v3 solver — we vary only
  the time budget, not the horizon length, to keep the comparison honest.
- A true rolling-horizon would additionally suffer from inter-window coupling loss
  (RC inventory, GC overflow, MTO spanning windows), making results *worse* than
  shown here.
