CSV INPUT PACK — Nestlé Trị An MILP Solver
==========================================
Version: 2.0 (fixed)
Model reference: mathematical_model_complete.md (v2)

FILES
-----
roasters.csv               — Machine capabilities, pipeline mapping, output routing flags
skus.csv                   — Product types, revenue, RC credit flag, due times
jobs.csv                   — MTO customer orders (mandatory batches)
shift_parameters.csv       — All timing, inventory, and cost parameters
planned_downtime.csv       — Pre-scheduled unavailability windows per roaster
manual_disruptions_template.csv — Schema for GUI-injected UPS events (empty by default)
solver_config.csv          — PuLP solver selection and run-time options

LOADER CONVENTIONS
------------------

1. SOLVER SELECTION
   solver_config.csv: solver_name = CBC
   PuLP's free built-in CBC solver is used by default. To use HiGHS (faster):
     pip install highspy
     Set solver_name = HiGHS in solver_config.csv.
   Do NOT use CPLEX unless a valid license is available on the target machine.

2. STOCKOUT COST — PER EVENT, NOT PER MINUTE
   shift_parameters.csv: stockout_cost_per_event_per_line = 1500
   This is a FIXED PENALTY per consumption event where B_l(tau) < 0 (strictly negative).
   It is NOT a per-minute rate. The MILP objective term is:
     c_stock * sum_l( count of tau in E_l where B_l(tau) < 0 )
   Applied in reactive mode only. In deterministic mode, B_l >= 0 is a hard constraint.

3. PSC BATCH POOL SIZE
   shift_parameters.csv: psc_pool_size_per_roaster = 32
   Total PSC pool = 5 roasters × 32 = 160 candidate batches.
   Formula: floor(shift_length_min / process_time_min) = floor(480 / 15) = 32.
   The solver generates activation binary a_b in {0,1} for each of these 160 slots.

4. PLANNED DOWNTIME — end_min IS INCLUSIVE
   planned_downtime.csv: start_min and end_min are BOTH INCLUSIVE slot indices.
   Example: R3, start_min=200, end_min=229 → blocked slots {200, 201, ..., 229} = 30 minutes.
   Slot 230 is free. The roaster set D_r = {start_min, start_min+1, ..., end_min}.
   Loader must enforce: for any batch b on roaster r,
     the set {s_b, s_b+1, ..., s_b+14} must have empty intersection with D_r.
   Last valid start before downtime: s_b such that s_b + 14 < start_min
     → s_b <= start_min - 15 = 200 - 15 = 185.

5. MANUAL DISRUPTIONS — CONDITIONAL LOADING
   manual_disruptions_template.csv is EMPTY BY DEFAULT (header only, no data rows).
   The loader must NOT read disruption events from this file unless BOTH conditions hold:
     a) enable_disruptions = 1 in solver_config.csv
     b) The GUI has explicitly submitted disruption events for this solve run.
   In deterministic baseline mode (enable_disruptions = 0), treat disruptions as empty
   regardless of what the file contains. This prevents accidental contamination of the
   deterministic benchmark results with example/test disruption data.

6. LINE ENDINGS
   All CSVs use Unix line endings (LF only). Open with:
     open(filepath, newline='', encoding='utf-8')
   and read with csv.DictReader. This handles both LF and CRLF transparently.

7. PRIORITY FIELD (REMOVED)
   The priority column has been removed from jobs.csv. The model does not use job
   priority in the objective function — all MTO batches are mandatory and tardiness
   is penalized uniformly at tardiness_cost_per_min regardless of job identity.
   If job-level priority weighting is added in future, re-introduce this column.

KEY MODEL CONSTANTS (derived, not stored in CSVs)
--------------------------------------------------
MTO due date:          slot 240  (= shift_length_min / 2)
Max batch start time:  slot 465  (= shift_length_min - process_time_min = 480 - 15)
Pipeline consume:      slots [s_b, s_b+2] inclusive (3 minutes concurrent with roast start)
Big-M value for MILP:  480 (= shift_length_min; safe upper bound for all time differences)

R3 SPECIAL RULES
----------------
- pipe(R3) = L2 always (R3 always pulls GC from Line 2 pipeline, constraint C8)
- out(R3 batch) = L1 if y_b=1, L2 if y_b=0 (output routing decision variable)
- When allow_r3_flexible_output=0: fix y_b=0 for all R3 batches (fixed mode experiment)
- When allow_r3_flexible_output=1: y_b is a free binary variable (flexible mode experiment)
- R3 overflow-idle: triggered ONLY when BOTH B_L1 = 40 AND B_L2 = 40

COST PRIORITY HIERARCHY
-----------------------
stockout ($1500/event) > tardiness ($1000/min) > idle ($200/min/roaster) > overflow-idle ($50/min/roaster)
