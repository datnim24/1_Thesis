# Assumptions & Deviations from Build Prompt

## Observation Space: 33 dims (not 27)

**Deviation**: The build prompt specifies 27 features. We use 33: 27 base features + 6 context one-hot vector.

**Reason**: The agent receives a single Discrete(21) action space for both RESTOCK and ROASTER decisions. Without the context one-hot, the agent cannot distinguish which decision type it is making. The 6-dim one-hot encodes: RESTOCK, R1, R2, R3, R4, R5. This was already implemented in the OLDCODE and is essential for correct policy learning.

## Data Loader: Wraps data_bridge (not OLDCODE's standalone loader)

**Deviation**: The build prompt suggested wrapping `env/data_bridge.py::get_sim_params()`. We do this instead of keeping the OLDCODE's 420-line standalone loader.

**Reason**: `get_sim_params()` already loads all canonical parameters. We only add PPO-specific parameters (`violation_penalty`, `episode_termination_on_violation`) by reading them directly from `shift_parameters.csv`. This eliminates ~370 lines of duplicated code (data_contract.py, input_schema.py, pathing.py).

## RC Negativity = Hard Termination

**Assumption**: When RC stock goes below 0 on any line (from packaging consumption events), the episode terminates immediately with `-violation_penalty` reward.

**Justification**: The core failure of the old PPO attempt was that the agent could allow RC to go negative without consequence. RC negativity means the agent failed to maintain inventory — the most critical task. Termination forces the agent to learn feasibility before optimizing profit.

**Note**: The canonical SimulationEngine allows RC to go negative (counting a stockout event with cost). Our hard termination is an additional layer on top, not a replacement of the engine's stockout tracking.

## DummyVecEnv Default (not SubprocVecEnv)

**Deviation**: The build prompt suggests SubprocVecEnv. We default to DummyVecEnv.

**Reason**: SubprocVecEnv on Windows requires all objects to be picklable via `spawn` (not fork). While `SimulationEngine` is likely picklable, DummyVecEnv is simpler and more reliable. SubprocVecEnv is available via `--subproc` CLI flag for users who want it.

## Restock Decisions: Every Slot (not just when restock_busy==0)

**Assumption**: A restock decision frame is enqueued every slot, but the mask ensures only valid actions (including WAIT) are available. If the only valid action is WAIT, the frame is automatically skipped.

**Justification**: This matches the OLDCODE pattern and ensures consistent decision framing. The mask handles feasibility (restock_busy > 0 → all restock actions masked → only WAIT → frame auto-skipped).

## Pipeline/Restock Mutex Violations

**Assumption**: Pipeline and restock mutex violations are primarily prevented by action masking (the engine's `can_start_batch()` and `can_start_restock()` already check these). The hard constraint enforcement layer checks for GC/RC negative and overflow but does NOT re-check pipeline mutex explicitly, because the canonical engine never allows mutex violations through its constraint-checking API.

## Cancelled Batches

**Assumption**: Cancelled batches (from UPS events) yield no revenue and no RC credit. GC is NOT refunded (already consumed at batch start). This matches the canonical engine behavior.

## Reward Verification

**Assumption**: `sum(step_rewards) == final_net_profit` within epsilon=1.0 for episodes that complete without violations. For violation-terminated episodes, the reward sum includes the violation penalty, so it differs from the engine's net_profit.

## Training Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| timesteps | 500,000 | Sufficient for initial convergence with 480-slot episodes |
| n_steps | 2,048 | Standard PPO rollout length |
| batch_size | 256 | Balance between noise and compute |
| learning_rate | 3e-4 | Standard PPO default |
| ent_coef | 0.01 | Moderate exploration |
| clip_range | 0.2 | Standard PPO default |
| violation_penalty | 50,000 | ~12.5x PSC batch revenue, makes violations catastrophic |
