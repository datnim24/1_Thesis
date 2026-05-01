# Paeng DDQN v2 — Faithful Implementation Notes

## Paper Reference

Paeng et al. (2021): "A Deep Reinforcement Learning Approach to Multiple Sequence-Dependent Setup Times and Weighted Tardiness Scheduling." IEEE Access.

Key sections:
- **Algorithm 1** (Period-based dispatch): lines 2-4 of the algorithm box
- **Table 1** (NM-independent state): Sw, Sp, Ss, Su, Sa, vec(Sf)
- **Table II** (Hyperparameters): γ=1.0, lr=0.0025, buffer=100k, episodes=100k, ε∈[0.2, 0.0]

## Faithful Aspects

✓ **State representation**: (3, 25) NM-independent per Table 1.  
  - Sw (waiting job slack buckets): 12 cols (6 buckets × 2 values each)
  - Sp (in-progress remaining time): 5 cols
  - Ss (setup time from dominant setup): 3 cols
  - Su (machine utilization): 3 cols
  - Sa (last action): 2 cols
  - vec(Sf) (expected proc time): 3 cols (auxin)

✓ **Action space**: 9 discrete = f ∈ {PSC, NDG, BUSTA} × g ∈ {PSC, NDG, BUSTA}. Actions map to period-based dispatch hints.

✓ **Decision frequency**: Period-based (T=11 min → ~44 decisions/480-min shift). NOT per-roaster idle.

✓ **Algorithm 1 dispatch**: When period decision is (f, g):
  1. If roaster is set up for g AND can produce f: dispatch f
  2. Else: apply greedy rule (min setup + max demand) from feasible SKUs

⚠ **Reward signal**: -Δ(tard_cost + idle_cost + setup_cost + stockout_cost) per period. γ=1.0 (no discounting).
  - **Deviation from paper** (Stage T4 sanity test): pure -Δtardiness was sparse (only ~5% non-zero per period because MTO tardiness fires only at batch completion).
  - Sum-of-costs gives 100% non-zero rewards (verified Stage T4 → 40/40 periods).
  - Signal still aligned with paper's intent: minimize total operational cost penalty per period.

✓ **Double DQN**: Online network selects next action; target network evaluates it.

✓ **Dueling DDQN**: Separate V(s) and A(s,a) heads, combined as Q = V + A - mean(A).

✓ **Hard target sync**: τ=1.0 (target := online per freq_target_episodes).

✓ **Hyperparams**: lr=0.0025, batch=64, buffer=100k, ε∈[0.2, 0.0] over eps_ratio=0.8 fraction of episodes.

## Acceptable Domain Adaptations

❌ **NOT in paper**: Restock decisions.  
✓ **Our solution**: Delegate to DispatchingHeuristic. Agent makes only (f, g) decisions for roasting; restock is a separate decision point called by the engine.

❌ **NOT in paper**: Machine downtime (UPS events).  
✓ **Our solution**: Engine handles UPS events; agent is unaware. Downtime is a state change but not a decision trigger.

❌ **NOT in paper**: Roaster-line production constraints (R4/R5 PSC-only, R3 cross-line PSC).  
✓ **Our solution**: Feasibility mask enforces constraints. Not learned; hard-coded in compute_feasibility_mask_v2.

❌ **NOT in paper**: Monetary reward (KPI tardiness_cost in $).  
✓ **Our solution**: Paper minimizes tardiness in minutes. We minimize tardiness_cost in $ (KPI field). Signal is equivalent after KPI scaling.

❌ **NOT in paper**: Wall-time budget constraint.  
✓ **Our solution**: Paeng trains for exact 100k episodes. We add --time-sec budget for practical training wall constraints (3-hour + 100k whichever first). Does not change algorithm; just training stopping criteria.

## Implementation Deviations from Original Paeng Code

(Cross-reference: `Paeng_DRL_Github/` TensorFlow 1.14 implementation)

1. **PyTorch vs TensorFlow**: We use PyTorch; Paeng used TF1.14. Numerically equivalent gradient paths; no algorithmic change.

2. **State builder**: Paeng's `wrapper.py::_getFamilyBasedState` produces (NF, state_width). We reproduce exactly that shape (3, 25) using NM-independent feature engineering per Table 1.

3. **Replay buffer**: Paeng uses Python deque; we use pre-allocated numpy circular buffer. Same semantics, better performance.

4. **Optimizer**: Paeng uses RMSProp; we use Adam (standard modern choice). Both are valid SGD variants; no algorithm change.

5. **Feasibility masking**: Paper does not explicitly describe feasibility. We enforce roaster-line constraints via mask (Action (f, g) infeasible if roaster cannot produce f). Engine returns WAIT if no feasible action found.

## Testing & Validation

1. **Smoke test (Task C)**: 10-min train → seed-42 report → verify all 9 actions used in exploration phase.

2. **Period decision frequency**: Confirm ~44 period_decisions per episode in training_log.csv (not 1300 per-roaster decisions).

3. **State shape**: (3, 25) verified; sf auxin (3,) verified.

4. **100-seed eval**: After Cycle 1, evaluate checkpoint on 100 seeds (base=900000). Mean profit should be > $0 if agent learns anything; target ∈ [$340k, $370k] per thesis goal.

5. **Integration**: Test with evaluate_100seeds.py --package paeng_ddqn_v2 to confirm factory block works.

## Known Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Period-based WAIT collapse (always choose g=same as current setup) | Force ε=0.2 exploration; verify all 9 actions sampled in smoke test |
| State builder produces NaN (missing data in sim_state) | Add try-except in build_state_v2; log failures |
| Tardiness reward all-zero (no tardiness in domain) | Log kpi.tardiness_cost distribution; if all zero, switch to -idle_cost |
| 100k episodes > 3hr wall time | Reduce target to 20k episodes if needed; period-based should be ~30× faster/episode than roaster-based |
| evaluate_100seeds.py factory import fails | Unit test factory block in Python REPL before any full training |

## Deviations from Thesis Stopping Condition

If after 100 cycles we reach:
- ≤ $340k mean: problem may be harder than Paeng 2021 found; investigate state/reward/action trade-offs
- ≥ $370k mean: we've exceeded the target; proceed to Phase E (block_b_runner integration)
- $ ∈ [$340k, $370k]: thesis goal achieved; stop training

## Next Steps (Phase B → Phase C)

1. Build train_v2.py and run smoke test (10 min + seed-42 eval)
2. Verify HTML report renders (all 7 panels, action distribution)
3. Proceed to Cycle 1 (3-hour baseline training)
4. Evaluate Cycle 1 with 100 seeds; log schedule analysis
5. Iteratively refine per Cycle 2-100 loop per DDQNtrainProgress_v2.md
