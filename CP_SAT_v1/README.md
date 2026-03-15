# CP-SAT Pipeline

This folder contains the OR-Tools CP-SAT implementation for the Nestle Tri An roasting thesis. It runs in parallel to the MILP baseline and reuses the same `data.py` interface and input CSV structure.

## Files

- `data.py`: copied from `MILP_Test_v2/data.py`, unchanged
- `cpsat_model.py`: builds the CP-SAT model only
- `cpsat_solver.py`: solves the model, logs progress, extracts JSON-safe results
- `cpsat_main.py`: CLI entry point and report printing
- `Input_data/`: copied input CSVs used by `data.load()`

## Why CP-SAT

CP-SAT is the primary exact solver path for this thesis because it handles:

- interval scheduling natively through `NoOverlap`
- conditional constraints through `OnlyEnforceIf`
- integer maximization with strong propagation
- incumbent logging through a solution callback

Compared with the MILP version, the CP-SAT pipeline avoids Big-M logic in the core scheduling constraints and is intended to scale better for reactive re-solves.

## Current behavior

- `--verbose` enables full CP-SAT internal search logs.
- The solver also emits a heartbeat every 5 minutes while it is still running.
- The heartbeat log includes elapsed time and remaining solver time.

Example heartbeat line:

```text
CP-SAT solve still running - elapsed: 300s, remaining solver time: 900s
```

## Working directory

All commands below assume the shell is already in:

```powershell
C:\1_Local_Folder\Code\1_Thesis\CP_SAT_v1
```

If you run from the repository root instead, prefix commands with `CP_SAT_v1\`.

## Basic commands

Syntax check:

```powershell
python -m py_compile data.py cpsat_model.py cpsat_solver.py cpsat_main.py
```

Build report only:

```powershell
python cpsat_model.py
```

Show CLI help:

```powershell
python cpsat_main.py --help
```

LP-bound only:

```powershell
python cpsat_main.py --lp-only --time-limit 30
```

Full solve with CP-SAT verbose logs:

```powershell
python cpsat_main.py --time-limit 60 --verbose
```

Recommended practical first run on the current model:

```powershell
python cpsat_main.py --time-limit 120 --num-workers 4 --verbose
```

Full solve without the printed table:

```powershell
python cpsat_main.py --time-limit 60 --verbose --no-report
```

Save results to JSON:

```powershell
python cpsat_main.py --time-limit 60 --output-json cpsat_result.json
```

Fixed R3 routing:

```powershell
python cpsat_main.py --r3-flex 0 --time-limit 60
```

Parallel search:

```powershell
python cpsat_main.py --time-limit 60 --num-workers 4 --verbose
```

Debug logging from Python modules:

```powershell
python cpsat_main.py --time-limit 60 --log-level DEBUG
```

## Inputs and overrides

By default, the CLI loads data from:

```powershell
Input_data
```

You can override solver-relevant settings without editing CSV files:

```powershell
python cpsat_main.py --time-limit 120 --gap 0.005 --r3-flex 1 --disruptions 0
```

## Output schema

The CP-SAT results dict is designed to match the MILP result structure for shared fields:

- `status`
- `solve_time`
- `net_profit`
- `total_revenue`
- `total_costs`
- `psc_count`
- `ndg_count`
- `busta_count`
- `revenue_psc`
- `revenue_ndg`
- `revenue_busta`
- `tardiness_min`
- `tard_cost`
- `idle_min`
- `idle_cost`
- `over_min`
- `over_cost`
- `allow_r3_flex`
- `schedule`
- `lp_bound`
- `gap_pct`

Additional CP-SAT-specific fields:

- `solver_engine`
- `best_bound`
- `num_incumbents`
- `solution_history`

## Notes

- `cpsat_model.py` does not solve the model.
- `cpsat_solver.py` does not print anything; it uses `logging` only.
- `cpsat_main.py` is the only CP-SAT module that prints reports to stdout.
- If the model is infeasible, `cpsat_main.py` exits with code `1` and logs the failure.
