"""PPO strategy class implementing the canonical engine strategy interface.

Plugs a trained MaskablePPO policy into SimulationEngine.run() and Reactive_GUI.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from env.simulation_engine import SimulationEngine

from .action_spec import ACTION_BY_ID, WAIT_ACTION_ID
from .mask_spec import compute_action_mask
from .observation_spec import ObservationContext, build_observation


_PPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_model_path(model_path: str | Path | None = None, prefer_best: bool = True) -> Path:
    """Resolve a model checkpoint path. Auto-discovers latest if not specified."""
    if model_path is not None:
        path = Path(model_path)
        if not path.is_absolute() and not path.exists():
            path = _PPO_ROOT / path
        if path.suffix != ".zip":
            candidate = path.with_suffix(".zip")
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return path.resolve()

    # Auto-discover: search PPOmask/outputs/**/checkpoints/ for latest
    outputs_dir = _PPO_ROOT / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"No outputs directory found: {outputs_dir}")

    if prefer_best:
        best_candidates = sorted(
            outputs_dir.glob("**/best_model.zip"),
            key=lambda p: p.stat().st_mtime,
        )
        if best_candidates:
            return best_candidates[-1].resolve()

    # Fall back to any .zip checkpoint
    all_zips = sorted(
        outputs_dir.glob("**/*.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    if all_zips:
        return all_zips[-1].resolve()

    raise FileNotFoundError("No checkpoint .zip found under PPOmask/outputs/")


class PPOStrategy:
    """Plug a trained MaskablePPO policy into the canonical engine strategy API."""

    def __init__(self, data, model: MaskablePPO, deterministic: bool = True) -> None:
        self.data = data
        self.model = model
        self.deterministic = deterministic
        self.engine = SimulationEngine(data.to_env_params())
        self.predict_calls = 0
        self.total_inference_ms = 0.0
        self.invalid_action_count = 0

    @classmethod
    def load(
        cls,
        data,
        model_path: str | Path | None = None,
        deterministic: bool = True,
        prefer_best: bool = True,
    ) -> "PPOStrategy":
        resolved = resolve_model_path(model_path, prefer_best=prefer_best)
        model = MaskablePPO.load(str(resolved))
        strategy = cls(data=data, model=model, deterministic=deterministic)
        strategy.model_path = resolved
        return strategy

    def _predict_action_id(self, state, context: ObservationContext) -> int:
        observation = build_observation(self.data, state, context)
        mask = compute_action_mask(self.data, self.engine, state, context)
        started = time.perf_counter()
        action, _ = self.model.predict(
            observation,
            deterministic=self.deterministic,
            action_masks=mask,
        )
        self.total_inference_ms += (time.perf_counter() - started) * 1000.0
        self.predict_calls += 1
        action_id = int(np.asarray(action).item())
        if action_id < 0 or action_id >= len(mask) or not mask[action_id]:
            self.invalid_action_count += 1
            return WAIT_ACTION_ID
        return action_id

    def decide_restock(self, state):
        action_id = self._predict_action_id(state, ObservationContext(kind="RESTOCK"))
        action = ACTION_BY_ID[action_id]
        if action_id == WAIT_ACTION_ID:
            return ("WAIT",)
        if action.env_action[0] != "START_RESTOCK":
            return ("WAIT",)
        return action.env_action

    def decide(self, state, roaster_id: str):
        action_id = self._predict_action_id(
            state, ObservationContext(kind="ROASTER", roaster_id=roaster_id),
        )
        action = ACTION_BY_ID[action_id]
        if action_id == WAIT_ACTION_ID:
            return ("WAIT",)
        if action.roaster_id != roaster_id:
            self.invalid_action_count += 1
            return ("WAIT",)
        return action.env_action

    def on_ups(self, state, event) -> None:
        pass

    def summary(self) -> dict[str, float | int | str]:
        model_path = getattr(self, "model_path", None)
        return {
            "model_path": str(model_path) if model_path is not None else "",
            "predict_calls": int(self.predict_calls),
            "total_inference_ms": float(self.total_inference_ms),
            "mean_inference_ms": float(self.total_inference_ms / max(1, self.predict_calls)),
            "invalid_action_count": int(self.invalid_action_count),
        }
