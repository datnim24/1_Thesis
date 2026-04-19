# env/ - Pure Simulation Environment

This package is the strategy-agnostic simulation environment for the roasting
problem.

## Architecture

- `env/` owns state transitions, KPI accounting, and result export.
- Strategy logic stays outside `env/`.
- `dispatch/` contains the dispatch heuristic.
- `q_learning/` contains the Q-learning wrapper and training scripts.
- All parameters are loaded from `Input_data/` through `data_bridge.py`.
- `SimulationEngine.run(schedule=...)` is intentionally disabled until a real
  planner-to-env execution path exists.

## Key Files

| File | Purpose |
|---|---|
| `data_bridge.py` | Load all simulation parameters from `Input_data/` |
| `simulation_state.py` | Dataclasses for batches, restocks, UPS, and full state |
| `simulation_engine.py` | Slot-by-slot environment dynamics |
| `kpi_tracker.py` | Profit and KPI accumulation |
| `export.py` | Result export |
| `ups_generator.py` | UPS scenario generation |

## Inventory Layers

The environment tracks two inventory layers:

1. GC inventory in finite silos per feasible `(line, sku)` pair
2. RC inventory in downstream PSC buffers per line

All capacities, initials, roast times, and restock parameters are loaded from
`Input_data/`. They are not hard-coded in the environment.

### GC logic

- GC is consumed at batch start.
- GC is added only when a restock completes.
- Cancelled started batches do not refund GC.

### Restock mechanics

- A restock blocks the target line pipeline for the full restock duration.
- Only one restock can be active plant-wide at a time.
- UPS does not interrupt an active restock.

The environment does not prescribe a restock policy. It only exposes the
global restock decision point through `decide_restock(state)` and validates
feasibility through `SimulationEngine.can_start_restock(...)`.

The current default strategy stack uses an input-driven reorder-point policy in
`dispatch/dispatching_heuristic.py`. Q-learning delegates its runtime restock
decisions to that same dispatch policy so training and evaluation stay aligned.

## Slot Phase Order

Each slot runs in this order:

1. UPS events
2. Roaster timers
3. Pipeline and restock timers
4. RC consumption and stockout tracking
5. Idle and overflow penalties
6. Global restock decision point
7. Per-roaster decision points

## Action Protocol

The environment accepts these strategy actions:

- `("PSC", output_line)`
- `("NDG",)`
- `("BUSTA",)`
- `("WAIT",)`
- `("START_RESTOCK", line_id, sku)`
