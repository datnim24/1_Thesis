"""Paeng et al. (2021) IEEE Access paper faithful rebuild.

This module implements the Paeng DDQN algorithm exactly as described in:
  "A Deep Reinforcement Learning Approach to Multiple Sequence-Dependent
   Setup Times and Weighted Tardiness Scheduling"
  IEEE Access, 2021

Key paper references:
  - Algorithm 1: Period-based dispatch with agent hint
  - Table 1: NM-independent state representation (3 families, 25 columns)
  - Table II: Hyperparameters (γ=1.0, lr=0.0025, buffer=100k, episodes=100k)

Domain adaptations (acceptable per plan):
  - Restock decisions delegated to DispatchingHeuristic (paper has no restock)
  - UPS downtime handled by engine (paper has no machine disruptions)
  - Roaster-line production constraints (R4/R5 PSC-only, R3 cross-line PSC)
  - Tardiness-cost reward in monetary terms (KPI's tardiness_cost field)
  - Training wall-time budget in addition to episode count

Modules:
    agent_v2   — PaengConfigV2, ParameterSharingDQNV2, ReplayBufferV2, PaengAgentV2
    strategy_v2 — build_state_v2, PaengStrategyV2 (period-based period adapter, state builder)
    train_v2   — CLI training loop (100k episode target or 3-hour wall budget)
    evaluate_v2 — single-seed report (Mode 1) + 100-seed aggregate (Mode 2)
"""
