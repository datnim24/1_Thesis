# Q-Learning for Reactive Roasting Scheduling

This package trains a tabular Q-learning policy for the per-roaster roasting
decisions in the thesis simulation.

Restock is not Q-learned. The global GC restock layer is controlled by
`DispatchingHeuristic`, and evaluation now uses that exact same restock policy
path through `QStrategy.decide_restock()`.

## Quick Start

All commands run from the project root (`1_Thesis/`).

```bash
# Train for one hour
python -m q_learning.q_learning_train --time 3600 --name first_try

# Evaluate the latest Q-table and generate the interactive report
python -m q_learning.q_learning_run

# Launch the GUI
python Reactive_GUI.py
```

After evaluation, the generated HTML report is opened automatically in the
default browser.

## Training and Evaluation Consistency

The current Q-learning design is:

- Q-learning trains only the per-roaster action policy.
- Restock decisions are handled by `DispatchingHeuristic`.
- Runtime evaluation delegates `QStrategy.decide_restock()` to that same
  dispatch heuristic.

This matters because the learned roaster policy should see the same restock
behavior during training and evaluation. Older versions used different restock
logic in those two phases, which caused large train/eval drift.

## Restock Policy

### Input-driven reorder points

The restock trigger is derived automatically from `Input_data/`, not from a
hard-coded GC fraction such as `frac <= 0.25`.

For each feasible GC silo pair `(line, sku)`, define:

- `m(line, sku)` = number of roasters that use that pipeline line and can roast
  that SKU
- `p_sku` = roast time of that SKU, in minutes
- `L_restock` = restock duration, in minutes

The maximum deterministic GC depletion rate is:

`d_max(line, sku) = m(line, sku) / p_sku`

The exact lead-time-demand reorder point is:

`ROP_exact(line, sku) = ceil(L_restock * d_max(line, sku))`

The implemented slot-based trigger is:

`ROP_impl(line, sku) = ceil((L_restock + 1) * d_max(line, sku))`

`ROP_impl` is used online because the simulator reviews decisions in discrete
time slots. `ROP_exact` is still reported for thesis traceability.

### Required data fields

The reorder-point rule is computed from the existing parameter layer loaded by
`env.data_bridge.get_sim_params()`:

- `roast_time_by_sku` from `Input_data/skus.csv`
- `restock_duration` from `Input_data/shift_parameters.csv`
- `feasible_gc_pairs` from `Input_data/shift_parameters.csv`
- `roasters`, `R_pipe`, and `R_elig_skus` from `Input_data/roasters.csv`

No extra input file is required.

### Decision rule

`DispatchingHeuristic.decide_restock()` applies the following logic:

1. Reject infeasible actions using `SimulationEngine.can_start_restock(...)`.
2. Only consider silo pairs with `GC(line, sku) <= ROP_impl(line, sku)`.
3. Require real demand justification:
   - `NDG` and `BUSTA` are restocked only if remaining MTO demand exists.
   - `PSC` is restocked only if future line consumption still matters and RC is
     low enough to justify more PSC production.
4. If multiple candidates qualify, rank them by:
   - larger `ROP_impl - current_GC`
   - higher MTO due pressure
   - lower GC stock ratio
   - more currently affected eligible roasters
   - lower remaining MTO slack

If no candidate qualifies, the heuristic returns `("WAIT",)`.

### Why this is better than the old fraction rule

The old `frac <= 0.25` trigger was ad hoc and brittle:

- it ignored SKU-specific roast times
- it ignored how many roasters could deplete that silo
- it ignored the configured restock duration
- it did not adapt automatically when `Input_data/` changed

The reorder-point rule is stronger academically because it is an explicit
deterministic lead-time-demand calculation tied to the actual thesis inputs.

## Retraining Note

Two cases matter:

- If only `QStrategy.decide_restock()` changes so that it delegates to the
  already-existing dispatch restock policy, then retraining is not strictly
  required.
- If `DispatchingHeuristic` itself changes, such as replacing the old fraction
  rule with the reorder-point algorithm, then the training-time restock
  dynamics have changed. Existing Q-tables are no longer perfectly matched, so
  retraining is recommended.

Because the current repo now uses the reorder-point dispatch policy, retraining
old Q-tables is recommended.

## State and Action Formulation

The learned table covers only roaster decisions.

Each roaster decision state is a compressed tuple containing:

- time bin
- current roaster id
- current last SKU
- binary pipeline-busy flag for that roaster's pipeline
- RC stock bin for the home line
- RC stock bin for the other line (`R3` only, else `0`)
- max MTO urgency across eligible MTO SKUs
- minimum GC bin across the roaster's eligible silo pairs
- just-finished-setup flag

The action space remains:

- `("PSC", "L1")`
- `("PSC", "L2")`
- `("NDG",)`
- `("BUSTA",)`
- `("WAIT",)`

Invalid actions are masked by the environment.

Restock actions still exist in the environment interface, but they are handled
by `DispatchingHeuristic`, not by the Q-table.

## Outputs

Each training run creates a folder under `q_learning/ql_results/` containing:

- `q_table_<name>.pkl`
- `training_log_<name>.pkl`
- `meta.json`
- `<result_folder>.html`
- `plots/` with interactive chart pages used by the main report

`q_learning/q_table.pkl` and `q_learning/training_log.pkl` are also updated as
the latest copies for GUI and convenience scripts.

## File Overview

- `q_strategy.py`: runtime strategy wrapper for greedy Q-table decisions
- `q_learning_train.py`: Q-learning training loop
- `q_learning_run.py`: evaluation plus interactive HTML reporting
- `ql_results/`: saved runs and reports
