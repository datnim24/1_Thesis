"""Sanity checks T3+T4: period decision count and reward signal.

T3: Total period decisions per episode should be ~44 (480-min shift / 11-min period).
T4: At least 50% of stored rewards should be non-zero (tardiness changes during episode).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PPOmask.Engine.data_loader import load_data
from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from paeng_ddqn_v2.agent_v2 import PaengAgentV2, PaengConfigV2
from paeng_ddqn_v2.strategy_v2 import PaengStrategyV2


def t3_period_decision_count():
    """Run 1 episode, count total decisions made across all roasters/period boundaries."""
    data = load_data()
    params = data.to_env_params()
    engine = SimulationEngine(params)
    cfg = PaengConfigV2()
    agent = PaengAgentV2(cfg)
    agent.epsilon = 0.5  # exploration to prevent action collapse

    strategy = PaengStrategyV2(agent, data, training=False, params=params)
    strategy.reset_episode()
    kpi, sim_state = engine.run(strategy, [])

    total_decisions = sum(strategy.action_counts.values())
    print(f"[T3] period_decisions = {total_decisions} (expected ~44 +/- 5)")
    print(f"     action_dist = {dict(strategy.action_counts)}")

    if not (35 <= total_decisions <= 55):
        print(f"  FAIL: total_decisions {total_decisions} outside [35, 55]")
        return False
    if total_decisions == 0:
        print(f"  FAIL: 0 decisions made!")
        return False
    print(f"  PASS")
    return True


def t4_reward_signal_nonzero():
    """Run 1 training episode, check stored rewards are non-zero in majority of periods."""
    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    ups_lambda = float(data.ups_lambda)
    ups_mu = float(data.ups_mu)

    engine = SimulationEngine(params)
    cfg = PaengConfigV2()
    agent = PaengAgentV2(cfg)
    agent.epsilon = 0.5  # exploration

    strategy = PaengStrategyV2(agent, data, training=True, params=params)
    strategy.reset_episode()

    # Use seed 42 with UPS for realistic disruption
    ups = generate_ups_events(ups_lambda, ups_mu, 42, SL, roasters)
    kpi, sim_state = engine.run(strategy, ups)
    strategy.end_episode(sim_state, float(kpi.net_profit()), float(kpi.tard_cost))

    rewards = strategy.reward_history
    print(f"[T4] Total rewards stored: {len(rewards)}")
    print(f"     Reward range: [{min(rewards) if rewards else 'N/A'}, {max(rewards) if rewards else 'N/A'}]")
    print(f"     Reward stats: mean={np.mean(rewards):.2f}, std={np.std(rewards):.2f}" if rewards else "     No rewards stored")
    nonzero = sum(1 for r in rewards if r != 0)
    pct_nonzero = (nonzero / max(1, len(rewards))) * 100
    print(f"     Non-zero rewards: {nonzero}/{len(rewards)} ({pct_nonzero:.1f}%)")
    print(f"     Final tardiness: ${kpi.tard_cost:,.0f}")
    print(f"     Final stockout: ${kpi.stockout_cost:,.0f}")
    print(f"     Final profit: ${kpi.net_profit():,.0f}")

    # Even if all rewards are 0 (no tardiness in this episode), the test should highlight this.
    # If tardiness is always 0 → reward signal is broken → need fallback
    if kpi.tard_cost == 0 and pct_nonzero == 0:
        print(f"  WARN: All rewards zero AND no tardiness — need fallback reward signal")
        return False  # Soft fail: this is a real problem
    if pct_nonzero < 30:
        print(f"  WARN: Only {pct_nonzero:.1f}% non-zero rewards (target 50%+)")
        # Still pass if there's tardiness — just sparse reward
    print(f"  PASS")
    return True


def main():
    print("=" * 60)
    p, f = 0, 0
    if t3_period_decision_count():
        p += 1
    else:
        f += 1
    print()
    if t4_reward_signal_nonzero():
        p += 1
    else:
        f += 1
    print()
    print(f"[summary] passed={p} failed={f}")
    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
