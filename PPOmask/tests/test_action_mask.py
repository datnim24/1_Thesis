"""Tests for action masking correctness."""
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
from PPOmask.Engine.action_spec import (
    ACTION_COUNT,
    RESERVED_ACTION_IDS,
    RESTOCK_ACTION_IDS,
    ROASTER_ACTION_IDS,
    WAIT_ACTION_ID,
)


@pytest.fixture
def data():
    return load_data()


@pytest.fixture
def env(data):
    e = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
    e.reset(seed=42)
    return e


class TestReservedActions:
    """Reserved actions (9-12, 17-19) must always be masked."""

    def test_reserved_always_masked(self, env):
        mask = env.action_masks()
        for action_id in RESERVED_ACTION_IDS:
            assert not mask[action_id], f"Reserved action {action_id} should be masked"


class TestWaitAlwaysValid:
    """WAIT (20) must always be unmasked."""

    def test_wait_always_unmasked(self, env):
        mask = env.action_masks()
        assert mask[WAIT_ACTION_ID], "WAIT should always be unmasked"


class TestMaskSize:
    """Mask must be exactly ACTION_COUNT elements."""

    def test_mask_shape(self, env):
        mask = env.action_masks()
        assert mask.shape == (ACTION_COUNT,)
        assert mask.dtype == bool


class TestEmptyGCSiloMasking:
    """Empty GC silo -> affected batch-start actions masked."""

    def test_empty_gc_masks_psc(self, data):
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        env.reset(seed=42)

        # Empty all GC silos
        for pair in env.state.gc_stock:
            env.state.gc_stock[pair] = 0

        mask = env.action_masks()
        # All batch-start actions (0-8) should be masked since GC is empty
        for action_id in range(9):
            if action_id not in RESERVED_ACTION_IDS:
                assert not mask[action_id], (
                    f"Action {action_id} should be masked with empty GC"
                )


class TestRestockMasking:
    """Restock station busy -> all restock actions masked."""

    def test_restock_busy_masks_restocks(self, data):
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        env.reset(seed=42)

        # Make restock station busy
        env.state.restock_busy = 10

        mask = env.action_masks()
        for action_id in RESTOCK_ACTION_IDS:
            assert not mask[action_id], (
                f"Restock action {action_id} should be masked when station is busy"
            )


class TestMaskConsistency:
    """Mask should be consistent across multiple calls without state changes."""

    def test_mask_deterministic(self, env):
        mask1 = env.action_masks()
        mask2 = env.action_masks()
        np.testing.assert_array_equal(mask1, mask2)
