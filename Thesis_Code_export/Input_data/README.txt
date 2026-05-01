INPUT_DATA — ENVIRONMENT SINGLE SOURCE OF TRUTH
===============================================
Version: 3.0 (env-centric)

This folder is the single source of truth for the simulation / reinforcement-learning
environment. The environment loads parameters and master data from these CSVs directly.
It does not load scheduling or model data via MILP_Test_v3.data (or any parallel legacy
data bundle for the old MILP stack).

ARCHITECTURE (HIGH LEVEL)
-------------------------
- CP-SAT is not part of the environment architecture. The env evolves state through
  discrete time (roasting tasks, setup, downtime, disruptions) and explicit GC restock
  tasks; there is no in-env CP-SAT or MILP solve step tied to each step.

- Roasting duration is SKU-specific: column roast_time_min in skus.csv (per SKU).

- Green-coffee (GC) silo capacities and initial stocks are parameters in
  shift_parameters.csv (gc_capacity_* and gc_initial_*).

- GC restock behaviour is parameterized in shift_parameters.csv:
  restock_duration_min and restock_qty_batches.

- The environment schedules and executes both roasting tasks and restock tasks (restock
  consumes roaster or logistics time according to the implementation; parameters above
  define duration and batch-units delivered per completed restock).

FILES
-----
roasters.csv               — Machine capabilities, pipeline mapping, output routing flags
skus.csv                   — Product types, revenue, RC credit, due times, batch size,
                             roast_time_min (SKU-specific roast duration)
jobs.csv                   — MTO customer orders (mandatory batches)
shift_parameters.csv       — Timing, RC/GC inventory, costs, UPS defaults, GC silos, restock
planned_downtime.csv       — Pre-scheduled unavailability windows per roaster
manual_disruptions_template.csv — Schema for GUI-injected UPS events (empty by default)
solver_config.csv          — PuLP solver selection (used by offline MILP tooling if present;
                             not part of env core architecture)

LOADER CONVENTIONS
------------------

1. SOLVER SELECTION (OFFLINE MILP / PuLP)
   solver_config.csv: solver_name = CBC
   PuLP's free built-in CBC solver is used by default for standalone MILP runs. To use
   HiGHS (faster):
     pip install highspy
     Set solver_name = HiGHS in solver_config.csv.
   Do NOT use CPLEX unless a valid license is available on the target machine.

2. STOCKOUT COST — PER EVENT, NOT PER MINUTE
   shift_parameters.csv: stockout_cost_per_event_per_line = 1500
   This is a FIXED PENALTY per consumption event where B_l(tau) < 0 (strictly negative).
   It is NOT a per-minute rate. In reactive / learning modes the objective or reward
   logic applies this per qualifying event. In deterministic mode with hard RC
   constraints, stockout is typically infeasible or treated as a hard guard.
   Applied in reactive mode only where documented by the env; align with your run mode.

3. PSC BATCH POOL SIZE (DERIVED FROM SKU ROAST TIME)
   shift_parameters.csv: psc_pool_size_per_roaster = 32
   For PSC, use roast_time_min from skus.csv (15) with shift_length_min:
     floor(shift_length_min / roast_time_min) = floor(480 / 15) = 32 batches per roaster
     per shift when the pool is defined that way.
   Total PSC pool = 5 roasters × 32 = 160 candidate batches when that convention applies.
   Other SKUs use their own roast_time_min for duration; pool sizing rules in code should
   use skus.csv, not a single global process_time_min, unless a parameter explicitly
   duplicates a derived constant for backward compatibility.

4. PLANNED DOWNTIME — end_min IS INCLUSIVE
   planned_downtime.csv: start_min and end_min are BOTH INCLUSIVE slot indices.
   Example: R3, start_min=200, end_min=229 → blocked slots {200, 201, ..., 229} = 30 minutes.
   Slot 230 is free. The roaster set D_r = {start_min, start_min+1, ..., end_min}.
   Loader must enforce: for any roast of duration T minutes on roaster r starting at s_b,
     slots {s_b, s_b+1, ..., s_b+T-1} must not intersect D_r (use T = roast_time_min for
     the batch's SKU).
   Example with T=15: last valid start before downtime at 200 is s_b <= 200 - 15 = 185.

5. MANUAL DISRUPTIONS — CONDITIONAL LOADING
   manual_disruptions_template.csv is EMPTY BY DEFAULT (header only, no data rows).
   The loader must NOT read disruption events from this file unless BOTH conditions hold:
     a) enable_disruptions = 1 in solver_config.csv (or the env/GUI equivalent flag)
     b) The GUI has explicitly submitted disruption events for this run.
   In deterministic baseline mode (disruptions disabled), treat disruptions as empty
   regardless of file content so benchmarks are not contaminated by example rows.

6. LINE ENDINGS
   All CSVs use Unix line endings (LF only). Open with:
     open(filepath, newline='', encoding='utf-8')
   and read with csv.DictReader. This handles both LF and CRLF transparently.

7. PRIORITY FIELD (REMOVED FROM jobs.csv)
   The priority column has been removed from jobs.csv. The model does not use job
   priority in the objective — all MTO batches are mandatory and tardiness is penalized
   uniformly at tardiness_cost_per_min unless extended later.

KEY MODEL CONSTANTS (ILLUSTRATIVE; VERIFY AGAINST shift_parameters.csv AND skus.csv)
------------------------------------------------------------------------------------
MTO due date (default):    default_due_time_min in skus.csv (e.g. 240) or policy default
Max batch start time:      shift_length_min - roast_time_min (SKU-specific; e.g. 480-15
                           for PSC)
Pipeline consume:          env-specific (often concurrent slots at roast start; see code)
GC silos / restock:        gc_* and restock_* rows in shift_parameters.csv

R3 SPECIAL RULES
----------------
- pipe(R3) = L2 always (R3 always pulls GC from Line 2 pipeline where that rule applies)
- out(R3 batch) = L1 if y_b=1, L2 if y_b=0 (output routing decision variable)
- When allow_r3_flexible_output=0: fix y_b=0 for all R3 batches (fixed mode experiment)
- When allow_r3_flexible_output=1: y_b is a free binary variable (flexible mode experiment)
- R3 overflow-idle: triggered ONLY when BOTH B_L1 = max_rc_batches_per_line AND
  B_L2 = max_rc_batches_per_line (40 when max_rc_batches_per_line = 40)

COST PRIORITY HIERARCHY (shift_parameters.csv)
----------------------------------------------
stockout ($1500/event) > tardiness ($1000/min) > idle ($200/min/roaster) >
overflow-idle ($50/min/roaster)

Values above match current shift_parameters.csv; if you change costs, update this line
for documentation consistency.
