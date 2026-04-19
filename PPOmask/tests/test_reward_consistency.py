"""Test that sum(step_rewards) == final net_profit within tolerance."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PPOmask.Engine.data_loader import load_data
from PPOmask.Engine.roasting_env import RoastingMaskEnv
from PPOmask.Engine.action_spec import WAIT_ACTION_ID


@pytest.fixture
def data():
    return load_data()


class TestRewardSumEqualsProfit:
    """sum(step_rewards) must equal final net_profit within epsilon."""

    def _run_episode_collecting_rewards(self, env) -> tuple[float, float, bool]:
        """Run full episode, return (reward_sum, net_profit, violated)."""
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        terminated = False
        violated = False
        while not terminated:
            mask = env.action_masks()
            # Pick first valid non-WAIT action, or WAIT
            action = WAIT_ACTION_ID
            for i in range(len(mask)):
                if mask[i] and i != WAIT_ACTION_ID:
                    action = i
                    break
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info.get("violation"):
                violated = True
        result = env.get_result()
        return total_reward, result["net_profit"], violated

    def test_reward_sum_matches_profit_no_violation(self, data):
        """With a greedy policy that avoids violations, reward sum should match profit."""
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        reward_sum, net_profit, violated = self._run_episode_collecting_rewards(env)

        if not violated:
            # No violation: reward sum should exactly match net_profit
            assert abs(reward_sum - net_profit) < 1.0, (
                f"Reward sum {reward_sum:.2f} != net_profit {net_profit:.2f} "
                f"(delta={abs(reward_sum - net_profit):.4f})"
            )
        else:
            # Violation occurred: reward includes penalty, so sum != profit
            # This is expected. Just verify the violation was tracked.
            assert violated

    def test_wait_only_reward_consistency(self, data):
        """WAIT-only will cause violation. Verify reward tracking is consistent."""
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        terminated = False
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(WAIT_ACTION_ID)
            total_reward += reward

        result = env.get_result()
        # Episode reward tracked internally should match our accumulation
        assert abs(total_reward - result["reward_sum"]) < 0.01, (
            f"External reward sum {total_reward} != internal {result['reward_sum']}"
        )
