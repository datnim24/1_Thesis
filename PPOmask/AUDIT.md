# PPOmask Repo Audit

## Input Data (`Input_data/`)

| File | Contents |
|------|----------|
| `shift_parameters.csv` | 31 rows: shift length (480 min), RC init/max/safety, GC capacities/init, consume rates, all cost coefficients, UPS params, restock config, violation_penalty, episode_termination_on_violation |
| `roasters.csv` | 5 roasters (R1-R5), 2 lines (L1, L2). R1/R2 on L1, R3/R4/R5 on L2. R3 has flexible output (L1 or L2). R2 is the only roaster that can process BUSTA. |
| `skus.csv` | 3 SKUs: PSC ($4k, 15 min, credits RC), NDG ($7k, 17 min, MTO), BUSTA ($7k, 18 min, MTO) |
| `jobs.csv` | 2 MTO jobs: J1 (5x NDG, due 240), J2 (5x BUSTA, due 240) |
| `planned_downtime.csv` | R4 maintenance: slots 200-229 (30 min) |
| `solver_config.csv` | Solver settings, R3 flexible output enabled |

## Existing Code

### Canonical Engine (`env/`)
- **`simulation_engine.py`**: SimulationEngine class. `run(strategy, ups_events)` -> (KPITracker, SimulationState). Phase order per slot: UPS -> roaster timers -> pipeline timers -> RC consumption -> stockout tracking -> idle penalties -> restock decisions -> roaster decisions -> snapshot.
- **`data_bridge.py`**: `get_sim_params(input_dir)` -> flat dict with all simulation parameters. The single canonical data loader.
- **`simulation_state.py`**: SimulationState dataclass with rc_stock, gc_stock, status, remaining, pipeline_busy, restock_busy, etc. BatchRecord, RestockRecord, UPSEvent dataclasses.
- **`kpi_tracker.py`**: KPITracker dataclass. `net_profit() = total_revenue - tard_cost - setup_cost - stockout_cost - idle_cost - over_cost`.
- **`export.py`**: `export_run()` writes JSON/CSV for schedule, trace, KPIs, UPS, cancelled, restocks.

### Q-Learning (`q_learning/`)
- **`q_strategy.py`**: QStrategy class with `decide(state, roaster_id)` and `decide_restock(state)`. Uses ACTION_MAP (17 actions), discretized state, greedy Q-table lookup. Restock delegated to DispatchingHeuristic.
- **`q_learning_train.py`**: CLI training with episodes/time budget, epsilon decay, checkpoint saving.
- **`q_learning_run.py`**: Evaluation runner: loads model, runs episodes, generates HTML report, auto-opens browser.

### Old PPO Attempt (`OLDCODE/PPOmask/`)
- **Architecture was solid**: action_spec (21 actions), observation_spec (27+6=33 features), mask_spec (delegates to engine constraints), env_adapter (stepwise wrapper over SimulationEngine), roasting_env (Gymnasium wrapper), drl_strategy (strategy class for GUI).
- **What failed**: No hard constraint enforcement. RC could go negative without episode termination. Invalid actions silently fell back to WAIT. Agent learned to ignore inventory management and converged to garbage policy.
- **Reusable components**: action_spec, observation_spec, mask_spec, reward_spec, drl_strategy are all correct. env_adapter needs hard constraint injection.

### Other
- **`Reactive_GUI.py`**: Strategy dashboard with Dispatching, Q-Learning, Manual modes. Uses SteppableEngine wrapping SimulationEngine.
- **`result_schema.py`**: `create_result()` builds canonical result dict for all solvers.
- **`plot_result.py`**: Universal HTML plotter. Gantt, RC, GC, pipeline, restock, waterfall, utilization plots. Searches `_RESULT_SEARCH_ROOTS` for result files.

## Key Design Decision

This rebuild reuses OLDCODE components but adds **hard constraint enforcement with episode termination** as the #1 priority. The violation_penalty (50000) and episode_termination_on_violation (1) parameters are loaded from Input_data/shift_parameters.csv.
