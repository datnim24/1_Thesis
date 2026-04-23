"""Gymnasium environment wrapping the canonical SimulationEngine with hard constraint enforcement.

Merges the adapter logic (decision queue, phase stepping) and the Gym wrapper into one file.
Key difference from OLDCODE: violations terminate the episode with a large penalty.
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from env.simulation_engine import SimulationEngine
from env.simulation_state import SimulationState, UPSEvent
from env.ups_generator import generate_ups_events

from .action_spec import ACTION_BY_ID, ACTION_COUNT, WAIT_ACTION_ID
from .mask_spec import compute_action_mask
from .observation_spec import OBS_TOTAL_DIM, ObservationContext, build_observation
from .reward_spec import incremental_profit, violation_reward


@dataclass(frozen=True)
class DecisionFrame:
    context: ObservationContext
    slot: int
    sequence: int


class RoastingMaskEnv(gym.Env):
    """Gymnasium Env wrapping the canonical SimulationEngine with hard constraint enforcement.

    Hard constraint violations -> episode termination + violation_penalty reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data,
        scenario_seed: int | None = None,
        ups_events: list[UPSEvent] | None = None,
        ups_lambda: float | None = None,
        ups_mu: float | None = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.params = data.to_env_params()
        self.engine = SimulationEngine(self.params)

        self._default_scenario_seed = scenario_seed
        self._scenario_rng = np.random.default_rng(scenario_seed)
        self._explicit_ups_events = ups_events
        self._ups_lambda = data.ups_lambda if ups_lambda is None else float(ups_lambda)
        self._ups_mu = data.ups_mu if ups_mu is None else float(ups_mu)

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_TOTAL_DIM,), dtype=np.float32,
        )

        # Mutable state (reset each episode)
        self._decision_queue: deque[DecisionFrame] = deque()
        self.state: SimulationState | None = None
        self.kpi = None
        self.ups_events: list[UPSEvent] = []
        self._ups_by_time: dict[int, list[UPSEvent]] = {}
        self._slot_index = 0
        self._carry_reward = 0.0
        self._episode_reward = 0.0
        self.invalid_action_count = 0
        self._sequence = 0
        self._terminated = False

        # Violation tracking
        self._violated = False
        self._violation_type = ""
        self._violation_counts: dict[str, int] = defaultdict(int)

        # Action count tracking (for diagnostic protocol)
        self._action_counts: dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            scenario_seed = seed
        elif self._explicit_ups_events is not None:
            scenario_seed = 0
        else:
            # C26 fix: draw a fresh UPS seed each episode so PPO sees broad
            # UPS diversity during training instead of replaying 8 fixed seeds.
            # Reproducibility preserved via _scenario_rng seeded from constructor.
            scenario_seed = int(self._scenario_rng.integers(0, 2**31 - 1))
        self.state = self.engine._initialize_state()
        self.kpi = self.engine._make_kpi_tracker()
        self._slot_index = 0
        self._decision_queue.clear()
        self.invalid_action_count = 0
        self._sequence = 0
        self._terminated = False
        self._violated = False
        self._violation_type = ""
        self._violation_counts = defaultdict(int)
        self._action_counts = defaultdict(int)
        self._episode_reward = 0.0
        self._carry_reward = 0.0

        self.ups_events = self._make_ups_events(scenario_seed)
        self._ups_by_time = {}
        for event in self.ups_events:
            self._ups_by_time.setdefault(event.t, []).append(event)

        self._carry_reward = self._advance_until_decision()
        obs = self._terminal_observation() if self._terminated else self._observation()
        return obs.astype(np.float32), self._info()

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("Cannot call step() after termination.")

        reward = self._carry_reward
        self._carry_reward = 0.0

        action_reward = self._apply_action(int(action))
        reward += action_reward

        # Check if violation occurred during action application
        if self._terminated:
            self._episode_reward += reward
            return self._terminal_observation(), reward, True, False, self._info()

        if self._decision_queue:
            self._episode_reward += reward
            return self._observation(), reward, False, False, self._info()

        assert self.state is not None
        self.state.trace.append(self.engine.get_state_snapshot(self.state))
        self._slot_index += 1
        reward += self._advance_until_decision()
        self._episode_reward += reward

        obs = self._terminal_observation() if self._terminated else self._observation()
        return obs.astype(np.float32), reward, self._terminated, False, self._info()

    def action_masks(self) -> np.ndarray:
        if self._terminated:
            mask = np.zeros(ACTION_COUNT, dtype=bool)
            mask[WAIT_ACTION_ID] = True
            return mask
        return compute_action_mask(
            self.data, self.engine, self.state, self.current_frame.context,
        )

    # ------------------------------------------------------------------
    # Decision queue
    # ------------------------------------------------------------------

    @property
    def current_frame(self) -> DecisionFrame:
        if not self._decision_queue:
            raise RuntimeError("No active decision frame.")
        return self._decision_queue[0]

    def _enqueue_decisions_for_current_slot(self) -> None:
        assert self.state is not None
        assert not self._decision_queue
        # Always offer a restock decision first
        self._sequence += 1
        self._decision_queue.append(
            DecisionFrame(
                context=ObservationContext(kind="RESTOCK"),
                slot=self._slot_index,
                sequence=self._sequence,
            )
        )
        # Then per-roaster decisions for idle roasters needing a decision
        for roaster_id in self.data.roasters:
            if self.state.status[roaster_id] != "IDLE":
                continue
            if not self.state.needs_decision[roaster_id]:
                continue
            self._sequence += 1
            self._decision_queue.append(
                DecisionFrame(
                    context=ObservationContext(kind="ROASTER", roaster_id=roaster_id),
                    slot=self._slot_index,
                    sequence=self._sequence,
                )
            )

        # If only a restock frame with no viable actions, skip it
        if len(self._decision_queue) == 1:
            restock_only = self._decision_queue[0]
            restock_mask = compute_action_mask(
                self.data, self.engine, self.state, restock_only.context,
            )
            if restock_mask.sum() == 1 and restock_mask[WAIT_ACTION_ID]:
                self._decision_queue.clear()

    # ------------------------------------------------------------------
    # Core simulation stepping
    # ------------------------------------------------------------------

    def _advance_until_decision(self) -> float:
        """Advance the simulation slot-by-slot until a decision point or termination."""
        assert self.state is not None
        assert self.kpi is not None

        accumulated_reward = 0.0
        while self._slot_index < self.data.shift_length:
            self.state.t = self._slot_index
            before = float(self.kpi.net_profit())
            prior_mto_completed = self.kpi.ndg_completed + self.kpi.busta_completed

            # Phase 1: UPS events
            for event in self._ups_by_time.get(self._slot_index, []):
                self.engine._process_ups(self.state, event, None, self.kpi)

            # Phase 2: Roaster timers
            self.engine._step_roaster_timers(self.state, self.kpi)

            # Phase 3: Pipeline & restock timers
            self.engine._step_pipeline_and_restock_timers(self.state, self.kpi)

            # Phase 4: RC consumption events
            self.engine._process_consumption_events(self.state, self.kpi)

            # >>> HARD CONSTRAINT CHECK after consumption <<<
            violated, reason = self._check_violations("post_consumption")
            if violated:
                penalty = violation_reward(self.data.violation_penalty)
                accumulated_reward += incremental_profit(before, float(self.kpi.net_profit()))
                accumulated_reward += penalty
                self._register_violation(reason)
                if self.data.episode_termination_on_violation:
                    self._terminated = True
                    # Penalize skipped MTO at early termination too
                    before_skip = float(self.kpi.net_profit())
                    self.engine._penalize_skipped_mto(self.state, self.kpi)
                    accumulated_reward += incremental_profit(before_skip, float(self.kpi.net_profit()))
                    return accumulated_reward

            # Phase 5: Stockout duration tracking
            self.engine._track_stockout_duration(self.state, self.kpi)

            # Phase 6: Idle/overflow penalties
            self.engine._accrue_idle_penalties(self.state, self.kpi)

            accumulated_reward += incremental_profit(before, float(self.kpi.net_profit()))

            # C27: Dense per-MTO-batch completion bonus. Paid at the slot where
            # the NDG/BUSTA batch actually finishes, converting the sparse
            # end-of-shift skip penalty into direct credit-assignable reward.
            # Does NOT modify kpi.net_profit (reporting unchanged).
            if self.data.mto_completion_bonus > 0.0:
                mto_delta = (
                    self.kpi.ndg_completed + self.kpi.busta_completed
                    - prior_mto_completed
                )
                if mto_delta > 0:
                    accumulated_reward += mto_delta * self.data.mto_completion_bonus

            # RC maintenance shaping reward (dense signal for keeping RC healthy)
            if self.data.rc_maintenance_bonus > 0.0:
                for lid in self.data.lines:
                    rc = self.state.rc_stock.get(lid, 0)
                    if rc >= self.data.safety_stock:
                        accumulated_reward += self.data.rc_maintenance_bonus
                    elif rc < self.data.safety_stock // 2:
                        # Danger zone: escalating penalty as RC approaches zero
                        # At rc=0: penalty = -3x bonus. At rc=safety/2: penalty = 0
                        danger_frac = 1.0 - (rc / max(1, self.data.safety_stock // 2))
                        accumulated_reward -= self.data.rc_maintenance_bonus * 3.0 * danger_frac

            # Phase 7+8: Decision enqueueing
            self._enqueue_decisions_for_current_slot()
            if self._decision_queue:
                return accumulated_reward

            self.state.trace.append(self.engine.get_state_snapshot(self.state))
            self._slot_index += 1

        self._terminated = True
        # End-of-shift: penalize skipped MTO batches (same as SimulationEngine.run)
        before_skip = float(self.kpi.net_profit())
        self.engine._penalize_skipped_mto(self.state, self.kpi)
        accumulated_reward += incremental_profit(before_skip, float(self.kpi.net_profit()))
        # Completion bonus: reward for surviving the full shift without violation
        if not self._violated and self.data.completion_bonus > 0.0:
            accumulated_reward += self.data.completion_bonus
        return accumulated_reward

    def _apply_action(self, action_id: int) -> float:
        """Apply the agent's chosen action. Returns incremental reward."""
        assert self.kpi is not None
        assert self.state is not None

        mask = self.action_masks()

        # HARD CONSTRAINT: Invalid action bypassed mask -> terminate
        if action_id < 0 or action_id >= len(mask) or not mask[action_id]:
            self.invalid_action_count += 1
            self._register_violation(f"invalid_action_{action_id}")
            if self.data.episode_termination_on_violation:
                self._terminated = True
                return violation_reward(self.data.violation_penalty)
            # Soft mode: penalize but continue with WAIT
            action_id = WAIT_ACTION_ID

        # Track the effective action applied this step
        self._action_counts[action_id] += 1

        frame = self._decision_queue.popleft()
        before = float(self.kpi.net_profit())
        action = ACTION_BY_ID[action_id]

        if frame.context.kind == "RESTOCK":
            if action_id != WAIT_ACTION_ID:
                _, line_id, sku = action.env_action
                self.engine._start_restock(self.state, line_id, sku, self.kpi)
        else:
            assert frame.context.roaster_id is not None
            self.engine._apply_action(
                self.state, frame.context.roaster_id, action.env_action, self.kpi,
            )

        after = float(self.kpi.net_profit())
        action_reward = incremental_profit(before, after)

        # HARD CONSTRAINT CHECK after action application (belt+suspenders)
        violated, reason = self._check_violations("post_action")
        if violated:
            self._register_violation(reason)
            if self.data.episode_termination_on_violation:
                self._terminated = True
                # Penalize skipped MTO at early termination
                before_skip = float(self.kpi.net_profit())
                self.engine._penalize_skipped_mto(self.state, self.kpi)
                skip_cost = incremental_profit(before_skip, float(self.kpi.net_profit()))
                return action_reward + violation_reward(self.data.violation_penalty) + skip_cost

        return action_reward

    # ------------------------------------------------------------------
    # Hard constraint enforcement
    # ------------------------------------------------------------------

    def _check_violations(self, phase: str) -> tuple[bool, str]:
        """Check plant invariants. Returns (violated, reason_string)."""
        assert self.state is not None
        lines = self.data.lines

        # RC negative (most critical -- the failure mode of the old code)
        for lid in lines:
            if self.state.rc_stock[lid] < 0:
                return True, f"rc_negative_{lid}"

        # GC negative (safety net)
        for pair, qty in self.state.gc_stock.items():
            if qty < 0:
                return True, f"gc_negative_{pair[0]}_{pair[1]}"

        # RC overflow
        for lid in lines:
            if self.state.rc_stock[lid] > self.data.max_rc:
                return True, f"rc_overflow_{lid}"

        return False, ""

    def _register_violation(self, reason: str) -> None:
        """Record a violation for logging/TensorBoard tracking."""
        self._violated = True
        self._violation_type = reason
        self._violation_counts[reason] += 1

    # ------------------------------------------------------------------
    # Observations & info
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        assert self.state is not None
        return build_observation(self.data, self.state, self.current_frame.context).astype(np.float32)

    def _terminal_observation(self) -> np.ndarray:
        return np.zeros(OBS_TOTAL_DIM, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        assert self.kpi is not None
        assert self.state is not None
        payload: dict[str, Any] = {
            "reward_sum": float(self._episode_reward),
            "invalid_action_count": int(self.invalid_action_count),
            "net_profit": float(self.kpi.net_profit()),
            "terminated": self._terminated,
            "violation": self._violated,
            "violation_type": self._violation_type,
            "violation_counts": dict(self._violation_counts),
        }
        if self._terminated:
            kpi_dict = self.kpi.to_dict()
            payload["kpi_psc_count"] = int(kpi_dict.get("psc_count", 0))
            payload["kpi_stockout_events"] = int(
                sum(int(v) for v in kpi_dict.get("stockout_events", {}).values())
            )
            payload["episode_action_counts"] = {
                int(k): int(v) for k, v in self._action_counts.items()
            }
        elif self._decision_queue:
            frame = self.current_frame
            payload["decision_context"] = {
                "kind": frame.context.kind,
                "roaster_id": frame.context.roaster_id,
                "slot": frame.slot,
                "sequence": frame.sequence,
            }
        return payload

    def get_result(self) -> dict[str, Any]:
        """Return full result dict for evaluation/export."""
        assert self.kpi is not None
        assert self.state is not None
        return {
            "reward_sum": float(self._episode_reward),
            "invalid_action_count": int(self.invalid_action_count),
            "kpi": self.kpi.to_dict(),
            "net_profit": float(self.kpi.net_profit()),
            "violation": self._violated,
            "violation_type": self._violation_type,
            "violation_counts": dict(self._violation_counts),
            "ups_events": [
                {"t": int(e.t), "roaster_id": e.roaster_id, "duration": int(e.duration)}
                for e in self.ups_events
            ],
        }

    # ------------------------------------------------------------------
    # UPS event generation
    # ------------------------------------------------------------------

    def _make_ups_events(self, scenario_seed: int | None) -> list[UPSEvent]:
        if self._explicit_ups_events is not None:
            return list(self._explicit_ups_events)
        if scenario_seed is None:
            scenario_seed = 0
        return generate_ups_events(
            lambda_rate=self._ups_lambda,
            mu_mean=self._ups_mu,
            seed=int(scenario_seed),
            shift_length=self.data.shift_length,
            roasters=list(self.data.roasters),
        )
