# dispatch/ — Dispatching Heuristic

This package contains the **urgency-first dispatching heuristic** for the
Nestlé Trị An batch roasting simulation.

## Architecture

- `dispatch/` is **external to `env/`** — the environment does not import
  from this package.
- The heuristic consumes `env` state and params but makes decisions
  independently.
- No CP-SAT or MILP dependencies.

## Strategy Interface

The `DispatchingHeuristic` class implements two methods:

### `decide(state, roaster_id) -> tuple`

Per-roaster decision. Returns one of:
- `("PSC", output_line)` — start PSC batch
- `("NDG",)` / `("BUSTA",)` — start MTO batch
- `("WAIT",)` — do nothing

### `decide_restock(state) -> tuple`

Global restock decision. Returns one of:
- `("START_RESTOCK", line_id, sku)` — start GC restock
- `("WAIT",)` — do nothing

## Heuristic Rules

### Roasting

1. Compute MTO urgency across all eligible jobs
2. If urgency >= 0.7, prioritize MTO (BUSTA first, then NDG)
3. Otherwise, run PSC to protect RC stock
4. R3 routes PSC to the line with lower RC stock

### Restock

1. If a needed batch cannot start because GC is 0 and restock is feasible,
   trigger `START_RESTOCK`
2. MTO shortage triggers higher-priority restock than PSC shortage
3. Priority order:
   - SKU needed immediately by urgent MTO on an idle eligible roaster
   - PSC on a line at RC risk
   - Lowest fractional GC silo
4. If no feasible batch and no feasible restock, WAIT

### GC Awareness

All batch feasibility checks now include GC availability:
- The roaster's home-line GC silo for the batch SKU must have >= 1 unit
- If GC is 0, the heuristic will attempt to trigger a restock before
  the batch can proceed
