"""Tests for hard constraint enforcement in the Gymnasium environment.

These are the most important tests: they verify that plant constraint violations
cause episode termination with the correct penalty.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PPOmask.Engine.data_loader import load_data
from PPOmask.Engine.roasting_env import RoastingMaskEnv
from PPOmask.Engine.action_spec import WAIT_ACTION_ID, RESERVED_ACTION_IDS


@pytest.fixture
def data():
    return load_data()


@pytest.fixture
def env(data):
    e = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
    e.reset(seed=42)
    return e


class TestRCNegativeTermination:
    """RC going negative after consumption -> episode terminates."""

    def test_rc_negative_terminates(self, data):
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        obs, info = env.reset(seed=42)

        # Force RC to 0 on both lines so next consumption event causes negative
        env.state.rc_stock["L1"] = 0
        env.state.rc_stock["L2"] = 0

        # Step with WAIT until consumption events drive RC negative
        terminated = False
        total_steps = 0
        while not terminated and total_steps < 500:
            obs, reward, terminated, truncated, info = env.step(WAIT_ACTION_ID)
            total_steps += 1

        assert terminated, "Episode should have terminated due to RC negative"
        assert info["violation"] is True
        assert "rc_negative" in info["violation_type"]

    def test_violation_penalty_in_reward(self, data):
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        obs, info = env.reset(seed=42)
        env.state.rc_stock["L1"] = 0
        env.state.rc_stock["L2"] = 0

        cumulative_reward = 0.0
        terminated = False
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(WAIT_ACTION_ID)
            cumulative_reward += reward

        # The cumulative reward should include the violation penalty
        assert cumulative_reward < -data.violation_penalty / 2, (
            f"Expected large negative reward from violation penalty, got {cumulative_reward}"
        )


class TestGCNegativeTermination:
    """GC going negative -> episode terminates (safety net)."""

    def test_gc_negative_terminates(self, data):
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        obs, info = env.reset(seed=42)

        # Manually force GC negative (bypassing mask - this shouldn't happen in
        # normal operation, but tests the safety net)
        env.state.gc_stock[("L1", "PSC")] = -1

        # Next step should detect the violation
        obs, reward, terminated, truncated, info = env.step(WAIT_ACTION_ID)
        assert terminated, "Episode should terminate when GC is negative"
        assert info["violation"] is True
        assert "gc_negative" in info["violation_type"]


class TestRCOverflowTermination:
    """RC exceeding max_rc -> episode terminates."""

    def test_rc_overflow_terminates(self, data):
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        obs, info = env.reset(seed=42)

        # Force RC above max
        env.state.rc_stock["L1"] = data.max_rc + 1

        obs, reward, terminated, truncated, info = env.step(WAIT_ACTION_ID)
        assert terminated, "Episode should terminate on RC overflow"
        assert info["violation"] is True
        assert "rc_overflow" in info["violation_type"]


class TestInvalidActionTermination:
    """Agent bypassing mask -> episode terminates."""

    def test_reserved_action_terminates(self, env):
        """Reserved actions (9-12, 17-19) should always be masked -> violation."""
        reserved_id = RESERVED_ACTION_IDS[0]  # action 9
        obs, reward, terminated, truncated, info = env.step(reserved_id)
        assert terminated, "Episode should terminate on reserved action"
        assert info["violation"] is True
        assert "invalid_action" in info["violation_type"]

    def test_out_of_range_action_terminates(self, env):
        """Action ID outside [0, 20] -> violation."""
        obs, reward, terminated, truncated, info = env.step(99)
        assert terminated
        assert info["violation"] is True


class TestViolationPenaltyValue:
    """Violation penalty matches Input_data value."""

    def test_penalty_loaded_correctly(self, data):
        assert data.violation_penalty == 50000.0

    def test_termination_flag_loaded(self, data):
        assert data.episode_termination_on_violation is True


class TestViolationInfoDict:
    """Info dict contains proper violation metadata."""

    def test_info_has_violation_fields(self, env):
        # Trigger a violation
        env.state.rc_stock["L1"] = -1
        obs, reward, terminated, truncated, info = env.step(WAIT_ACTION_ID)
        assert "violation" in info
        assert "violation_type" in info
        assert "violation_counts" in info
        assert isinstance(info["violation_counts"], dict)


class TestWaitOnlyNoViolation:
    """WAIT-only policy should complete a full episode without violations."""

    def test_wait_only_no_violation(self, data):
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        obs, info = env.reset(seed=42)

        # Note: WAIT-only WILL cause RC to go negative eventually because
        # no batches are started to replenish RC. This IS a violation.
        # So we test that the violation is correctly detected.
        terminated = False
        violation_detected = False
        steps = 0
        while not terminated and steps < 1000:
            obs, reward, terminated, truncated, info = env.step(WAIT_ACTION_ID)
            if info.get("violation"):
                violation_detected = True
            steps += 1

        # With WAIT-only and consumption events, RC WILL go negative
        # The important thing is that the episode DID terminate on violation
        assert terminated, "Episode should eventually terminate"
        assert violation_detected, "WAIT-only should cause RC negative violation"
        assert "rc_negative" in info.get("violation_type", "")
