"""End-to-end smoke test: env construction + short training + eval."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestEnvironmentConstruction:
    """Verify environment can be constructed and reset."""

    def test_env_creates(self):
        from PPOmask.Engine.data_loader import load_data
        from PPOmask.Engine.roasting_env import RoastingMaskEnv

        data = load_data()
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        obs, info = env.reset(seed=42)
        assert obs.shape[0] == 33  # 27 base + 6 context
        assert obs.dtype == np.float32
        assert "violation" in info

    def test_env_step(self):
        from PPOmask.Engine.data_loader import load_data
        from PPOmask.Engine.roasting_env import RoastingMaskEnv
        from PPOmask.Engine.action_spec import WAIT_ACTION_ID

        data = load_data()
        env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
        obs, info = env.reset(seed=42)
        obs2, reward, terminated, truncated, info2 = env.step(WAIT_ACTION_ID)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)


class TestSmokeTraining:
    """Verify that a tiny training run completes without crash."""

    def test_smoke_train(self):
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        from PPOmask.Engine.data_loader import load_data
        from PPOmask.Engine.roasting_env import RoastingMaskEnv

        data = load_data()

        def mask_fn(env):
            return env.unwrapped.action_masks()

        def make_env():
            env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
            env = Monitor(env)
            env = ActionMasker(env, mask_fn)
            return env

        vec_env = DummyVecEnv([make_env])
        model = MaskablePPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, verbose=0)
        model.learn(total_timesteps=128)
        vec_env.close()

    def test_model_save_load(self, tmp_path):
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        from PPOmask.Engine.data_loader import load_data
        from PPOmask.Engine.roasting_env import RoastingMaskEnv

        data = load_data()

        def mask_fn(env):
            return env.unwrapped.action_masks()

        def make_env():
            env = RoastingMaskEnv(data=data, scenario_seed=42, ups_lambda=0, ups_mu=0)
            env = Monitor(env)
            env = ActionMasker(env, mask_fn)
            return env

        vec_env = DummyVecEnv([make_env])
        model = MaskablePPO("MlpPolicy", vec_env, n_steps=64, batch_size=32, verbose=0)
        model.learn(total_timesteps=64)

        save_path = tmp_path / "test_model"
        model.save(str(save_path))
        assert (save_path.with_suffix(".zip")).exists()

        loaded = MaskablePPO.load(str(save_path))
        assert loaded is not None
        vec_env.close()
