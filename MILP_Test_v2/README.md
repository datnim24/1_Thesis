# MILP_Test_v2

Deterministic MILP baseline for the Nestle Tri An roasting scheduling thesis.

This folder contains the Stage 1-3 implementation:

- `data.py` loads CSV input files and builds derived sets
- `model.py` builds the PuLP MILP model
- `solver.py` calls CBC or HiGHS and extracts a JSON-safe results dict
- `main.py` is the CLI entry point
- `Input_data/` contains the CSV instance used by the solver

## Model Summary

This model schedules 5 roasters across 2 lines over one 8-hour shift:

- Line 1: `R1`, `R2`
- Line 2: `R3`, `R4`, `R5`
- SKUs: `PSC` (make-to-stock), `NDG`, `BUSTA` (make-to-order)
- Shared pipeline constraint per line: no overlapping 3-minute GC pulls
- Process time: 15 minutes
- Setup time: 5 minutes when switching SKU
- MTO due time: slot 240
- Latest valid batch start: slot 465
- Planned downtime: `R3` unavailable on slots 200-229 inclusive

Objective:

- maximize revenue
- minus tardiness cost
- minus safety-idle cost
- minus overflow-idle cost

Important modeling notes in the current build:

- `idle[r][t]` and `over[r][t]` are tracked only at change points, not all 480 minutes
- the inventory block uses a count-based reformulation
- inventory is tracked on the model's consumption-event time grid, not at every possible completion minute
- `R3` always uses the `L2` pipeline; in flex mode its PSC output may go to `L1` or `L2`

## File Layout

```text
MILP_Test_v2/
|-- README.md
|-- data.py
|-- model.py
|-- solver.py
|-- main.py
|-- debug.lp
`-- Input_data/
    |-- roasters.csv
    |-- skus.csv
    |-- jobs.csv
    |-- shift_parameters.csv
    |-- planned_downtime.csv
    |-- manual_disruptions_template.csv
    |-- solver_config.csv
    `-- README.txt
```

For CSV-specific notes, see `Input_data/README.txt`.

## How To Run

Open PowerShell and go to this folder:

```powershell
cd C:\1_Local_Folder\Code\1_Thesis\MILP_Test_v2
```

### 1. Validate Input Data

```powershell
python data.py
```

This prints the data validation report and confirms:

- roaster count
- PSC pool size
- MTO batch count
- consumption events per line
- R3 downtime expansion

### 2. Build The MILP Only

```powershell
python model.py
```

This prints the model build report, including:

- variable counts by type
- constraint counts by group
- total variables
- total constraints

### 3. Solve The Baseline Instance

```powershell
python main.py
```

This runs:

1. `data.load()`
2. `model.build()`
3. `solver.solve()`
4. the printed schedule and profit report

## Example Commands

Run with a longer solve limit:

```powershell
python main.py --time-limit 300
```

Run with a tighter MIP gap target:

```powershell
python main.py --time-limit 300 --gap 0.005
```

Run with verbose CBC output:

```powershell
python main.py --time-limit 300 --verbose
```

Run with verbose solver output and detailed Python logging:

```powershell
python main.py --time-limit 300 --verbose --log-level DEBUG
```

Write the results dict to JSON:

```powershell
python main.py --time-limit 300 --verbose --output-json results.json
```

Suppress the printed report and only write JSON:

```powershell
python main.py --time-limit 300 --no-report --output-json results.json
```

Run the fixed `R3` experiment:

```powershell
python main.py --r3-flex 0 --time-limit 300 --output-json results_r3_fixed.json
```

Run from the parent folder without changing directories:

```powershell
python MILP_Test_v2\main.py --time-limit 300 --output-json MILP_Test_v2\results.json
```

## CLI Flags

- `--input-dir PATH` override the input folder. Default: `./Input_data`
- `--solver {CBC,HiGHS}` override solver choice from `solver_config.csv`
- `--time-limit INT` solver time limit in seconds
- `--gap FLOAT` relative MIP gap target
- `--r3-flex {0,1}` override fixed or flexible `R3` output mode
- `--disruptions {0,1}` load disruption rows only when enabled
- `--verbose` show raw CBC or HiGHS output
- `--log-level {DEBUG,INFO,WARNING,ERROR}` set Python logging verbosity
- `--output-json PATH` save the results dict to JSON
- `--no-report` suppress the printed schedule and profit table

## Outputs

`main.py` can produce:

- terminal logging from `data`, `model`, `solver`, and `main`
- a printed deterministic schedule and profit report
- an optional JSON file with:
  - status
  - solve time
  - objective value
  - LP bound
  - gap
  - revenue and cost breakdown
  - active schedule entries

## Practical Notes

- `--time-limit` applies only to the solver phase, not data loading or model build
- `--verbose` passes through raw CBC output; CBC can stay quiet for long periods during presolve or root processing
- the full production instance is still large, so total wall-clock time can be much longer than the solver time limit alone
- temporary smoke-test files in this folder are not required for normal runs

## Recommended First Runs

Start with:

```powershell
python data.py
python model.py
python main.py --time-limit 120 --output-json results120.json
```

If you want to inspect solver behavior:

```powershell
python main.py --time-limit 120 --verbose --log-level INFO
```
