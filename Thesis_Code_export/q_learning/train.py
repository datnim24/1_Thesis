"""Q-learning training (single-shift PSC restocking + MTO dispatch).

Combines q_strategy + q_learning_train: state discretization, Q-table updates,
training loop. Outputs land in ``Results/<YYYYMMDD_HHMMSS>_QLearning_<RunName>/``
via ``evaluation.result_schema.make_run_dir``.

Usage:
    python -m q_learning.train --name SmokeTest --episodes 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.data_bridge import get_sim_params
from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from dispatch.dispatching_heuristic import DispatchingHeuristic

# ============================================================================
# Q-STRATEGY (state discretization, Q-table wrapper, action space)
# ============================================================================


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

WAIT_ACTION = 16

RESTOCK_ACTION_BY_PAIR: dict[tuple[str, str], int] = {
    ("L1", "PSC"): 9,
    ("L1", "NDG"): 10,
    ("L1", "BUSTA"): 11,
    ("L2", "PSC"): 12,
}

ACTION_MAP: dict[int, tuple] = {
    0: ("PSC", "L1"),
    1: ("PSC", "L1"),
    2: ("PSC", "L1"),
    3: ("PSC", "L2"),
    4: ("PSC", "L2"),
    5: ("PSC", "L2"),
    6: ("NDG",),
    7: ("NDG",),
    8: ("BUSTA",),
    9: ("START_RESTOCK", "L1", "PSC"),
    10: ("START_RESTOCK", "L1", "NDG"),
    11: ("START_RESTOCK", "L1", "BUSTA"),
    12: ("START_RESTOCK", "L2", "PSC"),
    WAIT_ACTION: ("WAIT",),
}

ROASTER_ACTIONS: dict[str, list[int]] = {
    "R1": [0, 6, WAIT_ACTION],
    "R2": [1, 7, 8, WAIT_ACTION],
    "R3": [2, 3, WAIT_ACTION],
    "R3_fixed": [3, WAIT_ACTION],
    "R4": [4, WAIT_ACTION],
    "R5": [5, WAIT_ACTION],
}

RESTOCK_ACTION_ORDER = (
    ("L1", "PSC"),
    ("L1", "NDG"),
    ("L1", "BUSTA"),
    ("L2", "PSC"),
)


def get_roaster_actions(roaster_id: str, allow_r3_flex: bool = True) -> list[int]:
    if roaster_id == "R3":
        return ROASTER_ACTIONS["R3"] if allow_r3_flex else ROASTER_ACTIONS["R3_fixed"]
    return ROASTER_ACTIONS[roaster_id]


def get_restock_actions(params: dict) -> list[int]:
    feasible = {tuple(pair) for pair in params.get("feasible_gc_pairs", [])}
    actions = [
        RESTOCK_ACTION_BY_PAIR[pair]
        for pair in RESTOCK_ACTION_ORDER
        if pair in feasible
    ]
    actions.append(WAIT_ACTION)
    return actions


# ---------------------------------------------------------------------------
# State discretisation
# ---------------------------------------------------------------------------

def _bin_time(slot: int, params: dict) -> int:
    """Piecewise time bins that emphasise the MTO due window and late shift."""
    shift = max(1, int(params.get("SL", 480)))
    due_times = sorted(int(v) for v in params.get("job_due", {}).values())
    first_due = due_times[0] if due_times else shift // 2
    raw_cutoffs = [
        first_due // 2,
        first_due - 60,
        first_due - 20,
        first_due,
        first_due + 60,
        shift - 90,
        shift - 30,
    ]
    cutoffs = []
    for cutoff in raw_cutoffs:
        if 0 < cutoff < shift and cutoff not in cutoffs:
            cutoffs.append(cutoff)
    cutoffs.sort()

    bucket = 0
    for cutoff in cutoffs:
        if slot >= cutoff:
            bucket += 1
    return bucket


def _bin_rc(rc_value: int, params: dict) -> int:
    """Adaptive 5-bin RC discretisation anchored on safety/max buffer levels."""
    max_rc = max(1, int(params.get("max_rc", 40)))
    safety = max(1, int(params.get("safety_stock", max_rc // 2)))
    if rc_value <= 0:
        return 0
    if rc_value <= max(1, safety // 2):
        return 1
    if rc_value <= safety:
        return 2
    if rc_value < max_rc:
        return 3
    return 4


def _bin_gc(stock: int, cap: int) -> int:
    """Adaptive 5-bin GC discretisation aligned with the restock batch size."""
    cap = max(1, int(cap))
    restock_qty = 5
    if stock <= 0:
        return 0
    if stock == 1:
        return 1
    if stock < restock_qty:
        return 2
    if stock < min(cap, 2 * restock_qty):
        return 3
    return 4


def _bin_timer(remaining: int) -> int:
    """Coarse timer bins for pipeline occupancy."""
    if remaining <= 0:
        return 0
    if remaining <= 3:
        return 1
    if remaining <= 9:
        return 2
    return 3


def _encode_pipeline_mode(mode: str) -> int:
    return {"FREE": 0, "CONSUME": 1, "RESTOCK": 2}.get(str(mode), 0)


def _mto_remaining_for_sku(state: Any, params: dict, sku: str) -> int:
    total = 0
    for job_id, job_sku in params["job_sku"].items():
        if job_sku == sku:
            total += int(state.mto_remaining.get(job_id, 0))
    return total


def _job_due_for_sku(params: dict, sku: str) -> int:
    dues = [
        int(params["job_due"][job_id])
        for job_id, job_sku in params["job_sku"].items()
        if job_sku == sku
    ]
    if not dues:
        return int(params.get("SL", 480))
    return min(dues)


def _bin_urgency(slack: int, roast_time: int, sigma: int) -> int:
    """5-bin urgency scale: done / comfortable / watch / urgent / overdue."""
    if slack < 0:
        return 4
    if slack <= roast_time:
        return 3
    if slack <= 2 * roast_time + sigma:
        return 2
    return 1


def _roaster_mto_urgency(state: Any, params: dict, roaster_id: str, sku: str) -> int:
    remaining = _mto_remaining_for_sku(state, params, sku)
    if remaining <= 0:
        return 0
    due = _job_due_for_sku(params, sku)
    roast_time = int(params["roast_time_by_sku"][sku])
    sigma = int(params.get("sigma", 0))
    setup = sigma if state.last_sku.get(roaster_id) != sku else 0
    required = remaining * roast_time + setup
    slack = due - int(state.t) - required
    return _bin_urgency(slack, roast_time, sigma)


def _global_mto_urgency(state: Any, params: dict, sku: str) -> int:
    remaining = _mto_remaining_for_sku(state, params, sku)
    if remaining <= 0:
        return 0
    due = _job_due_for_sku(params, sku)
    roast_time = int(params["roast_time_by_sku"][sku])
    sigma = int(params.get("sigma", 0))
    eligible_roasters = [rid for rid in params["roasters"] if sku in params["R_elig_skus"][rid]]
    setup = 0 if any(state.last_sku.get(rid) == sku for rid in eligible_roasters) else sigma
    required = remaining * roast_time + setup
    slack = due - int(state.t) - required
    return _bin_urgency(slack, roast_time, sigma)


def _gc_bin_for_pair(state: Any, params: dict, line_id: str, sku: str) -> int:
    pair = (line_id, sku)
    cap = int(params.get("gc_capacity", {}).get(pair, 1))
    stock = int(state.gc_stock.get(pair, 0))
    return _bin_gc(stock, cap)


def _idle_roaster_exists(state: Any, params: dict, line_id: str, sku: str) -> int:
    for roaster_id in params["roasters"]:
        if params["R_pipe"][roaster_id] != line_id:
            continue
        if sku not in params["R_elig_skus"][roaster_id]:
            continue
        if state.status.get(roaster_id) == "IDLE":
            return 1
    return 0


def _restock_pressure_psc(state: Any, params: dict, line_id: str, idle_flag: int, gc_bin: int) -> int:
    if not idle_flag or gc_bin > 2:
        return 0
    rc = int(state.rc_stock.get(line_id, 0))
    safety = int(params.get("safety_stock", 20))
    if rc <= max(1, safety // 2):
        return 2
    if rc < safety:
        return 1
    return 0


def _restock_pressure_mto(urgency_bin: int, idle_flag: int, gc_bin: int) -> int:
    if not idle_flag or urgency_bin <= 0 or gc_bin > 2:
        return 0
    if urgency_bin >= 3:
        return 2
    return 1


def should_trigger_restock_decision(state: Any, params: dict) -> bool:
    """Return True only when the restock layer is materially decision-relevant.

    This keeps the tabular learner focused on inventory-management states with
    real pressure instead of creating a noisy restock decision on every idle
    minute where `WAIT` is obviously correct.
    """
    mto_ndg = _global_mto_urgency(state, params, "NDG")
    mto_busta = _global_mto_urgency(state, params, "BUSTA")

    idle_l1_psc = _idle_roaster_exists(state, params, "L1", "PSC")
    idle_l1_ndg = _idle_roaster_exists(state, params, "L1", "NDG")
    idle_l1_busta = _idle_roaster_exists(state, params, "L1", "BUSTA")
    idle_l2_psc = _idle_roaster_exists(state, params, "L2", "PSC")

    gc_l1_psc = _gc_bin_for_pair(state, params, "L1", "PSC")
    gc_l1_ndg = _gc_bin_for_pair(state, params, "L1", "NDG")
    gc_l1_busta = _gc_bin_for_pair(state, params, "L1", "BUSTA")
    gc_l2_psc = _gc_bin_for_pair(state, params, "L2", "PSC")

    pressure_flags = [
        _restock_pressure_psc(state, params, "L1", idle_l1_psc, gc_l1_psc),
        _restock_pressure_psc(state, params, "L2", idle_l2_psc, gc_l2_psc),
        _restock_pressure_mto(mto_ndg, idle_l1_ndg, gc_l1_ndg),
        _restock_pressure_mto(mto_busta, idle_l1_busta, gc_l1_busta),
    ]
    scarcity_flags = [
        gc_l1_psc <= 2,
        gc_l1_ndg <= 2 and mto_ndg > 0,
        gc_l1_busta <= 2 and mto_busta > 0,
        gc_l2_psc <= 2,
    ]
    return any(pressure_flags) or any(scarcity_flags)


def discretize_roaster_state(
    state: Any,
    roaster_id: str,
    params: dict,
    just_set_up: bool = False,
) -> tuple:
    """Compressed hashable state for per-roaster roasting decisions."""
    time_bin = _bin_time(int(state.t), params)
    own_last_sku = state.last_sku.get(roaster_id, "PSC")

    pipe_line = params["R_pipe"][roaster_id]
    pipe_busy = 1 if int(state.pipeline_busy.get(pipe_line, 0)) > 0 else 0

    home_line = params["R_line"][roaster_id]
    rc_home = _bin_rc(int(state.rc_stock.get(home_line, 0)), params)
    if roaster_id == "R3" and params.get("allow_r3_flex", False):
        other_line = "L1" if home_line == "L2" else "L2"
        rc_other = _bin_rc(int(state.rc_stock.get(other_line, 0)), params)
    else:
        rc_other = 0

    eligible = set(params["R_elig_skus"][roaster_id])
    mto_ndg = _roaster_mto_urgency(state, params, roaster_id, "NDG") if "NDG" in eligible else 0
    mto_busta = _roaster_mto_urgency(state, params, roaster_id, "BUSTA") if "BUSTA" in eligible else 0
    mto_max_urgency = max(mto_ndg, mto_busta)

    gc_bins = [
        _gc_bin_for_pair(state, params, pipe_line, sku)
        for sku in eligible
    ]
    gc_min_eligible = min(gc_bins) if gc_bins else 0

    setup_flag = 1 if just_set_up else 0

    return (
        "roaster",
        roaster_id,
        time_bin,
        own_last_sku,
        pipe_busy,
        rc_home,
        rc_other,
        mto_max_urgency,
        gc_min_eligible,
        setup_flag,
    )


def discretize_restock_state(state: Any, params: dict) -> tuple:
    """Hashable state for the global restock decision layer.

    The restock formulation observes:
      - time
      - both line RC levels
      - both line pipeline mode/timer
      - remaining MTO pressure (NDG, BUSTA)
      - GC levels for all feasible silos
      - idle-roaster availability for each feasible silo pair
    """
    time_bin = _bin_time(int(state.t), params)

    rc_l1 = _bin_rc(int(state.rc_stock.get("L1", 0)), params)
    rc_l2 = _bin_rc(int(state.rc_stock.get("L2", 0)), params)

    mode_l1 = _encode_pipeline_mode(state.pipeline_mode.get("L1", "FREE"))
    mode_l2 = _encode_pipeline_mode(state.pipeline_mode.get("L2", "FREE"))
    timer_l1 = _bin_timer(int(state.pipeline_busy.get("L1", 0)))
    timer_l2 = _bin_timer(int(state.pipeline_busy.get("L2", 0)))

    mto_ndg = _global_mto_urgency(state, params, "NDG")
    mto_busta = _global_mto_urgency(state, params, "BUSTA")

    gc_l1_psc = _gc_bin_for_pair(state, params, "L1", "PSC")
    gc_l1_ndg = _gc_bin_for_pair(state, params, "L1", "NDG")
    gc_l1_busta = _gc_bin_for_pair(state, params, "L1", "BUSTA")
    gc_l2_psc = _gc_bin_for_pair(state, params, "L2", "PSC")

    idle_l1_psc = _idle_roaster_exists(state, params, "L1", "PSC")
    idle_l1_ndg = _idle_roaster_exists(state, params, "L1", "NDG")
    idle_l1_busta = _idle_roaster_exists(state, params, "L1", "BUSTA")
    idle_l2_psc = _idle_roaster_exists(state, params, "L2", "PSC")
    pressure_l1_psc = _restock_pressure_psc(state, params, "L1", idle_l1_psc, gc_l1_psc)
    pressure_l1_ndg = _restock_pressure_mto(mto_ndg, idle_l1_ndg, gc_l1_ndg)
    pressure_l1_busta = _restock_pressure_mto(mto_busta, idle_l1_busta, gc_l1_busta)
    pressure_l2_psc = _restock_pressure_psc(state, params, "L2", idle_l2_psc, gc_l2_psc)

    return (
        "restock",
        time_bin,
        rc_l1,
        rc_l2,
        mode_l1,
        timer_l1,
        mode_l2,
        timer_l2,
        mto_ndg,
        mto_busta,
        gc_l1_psc,
        gc_l1_ndg,
        gc_l1_busta,
        gc_l2_psc,
        pressure_l1_psc,
        pressure_l1_ndg,
        pressure_l1_busta,
        pressure_l2_psc,
    )


# Backward-compatible alias used by training/eval imports.
discretize_state = discretize_roaster_state


# ---------------------------------------------------------------------------
# Action masking helpers
# ---------------------------------------------------------------------------

def get_valid_roaster_actions(
    engine: SimulationEngine,
    state: Any,
    roaster_id: str,
    allow_r3_flex: bool = True,
) -> list[int]:
    mask = engine._compute_action_mask(state, roaster_id)
    return [
        action_id
        for action_id in get_roaster_actions(roaster_id, allow_r3_flex)
        if mask.get(ACTION_MAP[action_id], False)
    ]


def get_valid_restock_actions(
    engine: SimulationEngine,
    state: Any,
    params: dict,
) -> list[int]:
    valid: list[int] = []
    for action_id in get_restock_actions(params):
        action_tuple = ACTION_MAP[action_id]
        if action_tuple == ("WAIT",):
            valid.append(action_id)
            continue
        _, line_id, sku = action_tuple
        if engine.can_start_restock(state, line_id, sku):
            valid.append(action_id)
    if WAIT_ACTION not in valid:
        valid.append(WAIT_ACTION)
    return valid


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_q_table(q_table: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_q_table(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class QStrategy:
    """Greedy Q-table strategy for roasting plus dispatch restock control."""

    def __init__(
        self,
        params: dict,
        q_table_path: str | None = None,
        q_table: dict | None = None,
    ):
        self.params = params
        if q_table is not None:
            self.q_table = q_table
        elif q_table_path is not None:
            resolved = Path(q_table_path)
            if not resolved.is_absolute():
                resolved = _PROJECT_ROOT / resolved
            self.q_table = load_q_table(str(resolved))
        else:
            self.q_table = {}

        self._allow_flex = params.get("allow_r3_flex", True)
        self._engine: SimulationEngine | None = None
        self._prev_status: dict[str, str] = {r: "IDLE" for r in params["roasters"]}
        self._just_set_up: dict[str, bool] = {r: False for r in params["roasters"]}
        self._restock_helper = DispatchingHeuristic(params)

    def _get_engine(self) -> SimulationEngine:
        if self._engine is None:
            self._engine = SimulationEngine(self.params)
        return self._engine

    def _sync_status_flags(self, state: Any) -> None:
        for roaster_id in self.params["roasters"]:
            cur = state.status[roaster_id]
            if self._prev_status[roaster_id] == "SETUP" and cur == "IDLE":
                self._just_set_up[roaster_id] = True
            elif cur == "RUNNING":
                self._just_set_up[roaster_id] = False
            self._prev_status[roaster_id] = cur

    def _select_action(self, state_key: tuple, valid_actions: list[int]) -> int:
        if not valid_actions:
            return WAIT_ACTION
        q_values = [(action_id, self.q_table.get((state_key, action_id), 0.0)) for action_id in valid_actions]
        return max(q_values, key=lambda item: item[1])[0]

    def decide(self, state: Any, roaster_id: str) -> tuple:
        self._sync_status_flags(state)
        s = discretize_roaster_state(
            state,
            roaster_id,
            self.params,
            just_set_up=self._just_set_up[roaster_id],
        )
        engine = self._get_engine()
        valid_actions = get_valid_roaster_actions(engine, state, roaster_id, self._allow_flex)
        chosen = self._select_action(s, valid_actions)
        return ACTION_MAP[chosen]

    def decide_restock(self, state: Any) -> tuple:
        """Delegate restock to DispatchingHeuristic for train/eval consistency."""
        self._sync_status_flags(state)
        return self._restock_helper.decide_restock(state)

# ============================================================================
# TRAINING LOOP
# ============================================================================


# ---------------------------------------------------------------------------
# Hyper-parameters (defaults)
# ---------------------------------------------------------------------------

ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
NUM_EPISODES = 20_000

# Robust-mode scenario grid (only used with --robust flag)
ROBUST_LAMBDA = [0, 1, 2, 5, 10, 20]
ROBUST_MU = [5, 10, 20, 30]


# ---------------------------------------------------------------------------
# Result folder naming
# ---------------------------------------------------------------------------

def _make_result_dir(name: str, episodes: int, alpha: float, gamma: float) -> Path:
    """Create the run output dir under Results/<ts>_QLearning_<name>/."""
    from evaluation.result_schema import make_run_dir
    run_name = name or f"ep{episodes}_a{alpha:.4f}_g{gamma:.4f}".replace(".", "")
    return make_run_dir("QLearning", run_name)


def _rename_with_profit(result_dir: Path, final_profit: float) -> Path:
    new_path = result_dir.parent / (result_dir.name + f"_profit{int(round(final_profit))}")
    try:
        result_dir.rename(new_path)
        return new_path
    except OSError:
        return result_dir


# ---------------------------------------------------------------------------
# Manual episode runner - gives direct KPI access for slot-aligned rewards
# ---------------------------------------------------------------------------

def _train_one_episode(
    params: dict,
    engine: SimulationEngine,
    q_table: dict,
    epsilon: float,
    ups_events: list,
    allow_flex: bool,
) -> tuple[list[tuple], float]:
    """Run one shift with epsilon-greedy decisions and slot-aligned rewards.

    Returns (transitions, episode_profit) where each transition is:
    (state, action_id, reward_delta, next_state, next_valid_actions)

    Only per-roaster roasting decisions are Q-learned.
    Restock decisions are delegated to the dispatching heuristic so
    training matches evaluation and the Q-table stays roaster-only.
    """
    state = engine._initialize_state()
    kpi = engine._make_kpi_tracker()

    ups_by_time: dict[int, list] = defaultdict(list)
    for event in sorted(ups_events, key=lambda e: (e.t, e.roaster_id, e.duration)):
        ups_by_time[event.t].append(event)

    transitions: list[tuple] = []
    prev_state_disc: tuple | None = None
    prev_action: int | None = None
    pending_reward: float = 0.0

    # Track SETUP->IDLE so the state can distinguish "freshly configured"
    # from "already producing this SKU".
    prev_status = {r: "IDLE" for r in engine.roasters}
    just_set_up = {r: False for r in engine.roasters}
    restock_helper = DispatchingHeuristic(params)

    def _close_pending(next_state: tuple, next_valid_actions: list[int]) -> None:
        nonlocal pending_reward
        if prev_state_disc is None:
            return
        transitions.append(
            (prev_state_disc, prev_action, pending_reward, next_state, tuple(next_valid_actions))
        )
        pending_reward = 0.0

    def _choose_action(state_key: tuple, valid_actions: list[int]) -> int:
        if not valid_actions:
            return WAIT_ACTION
        if random.random() < epsilon:
            return random.choice(valid_actions)
        q_vals = [(action_id, q_table.get((state_key, action_id), 0.0)) for action_id in valid_actions]
        return max(q_vals, key=lambda item: item[1])[0]

    for slot in range(params["SL"]):
        state.t = slot
        slot_profit_before = kpi.net_profit()

        for event in ups_by_time.get(slot, []):
            engine._process_ups(state, event, None, kpi)

        engine._step_roaster_timers(state, kpi)
        engine._step_pipeline_and_restock_timers(state, kpi)
        engine._process_consumption_events(state, kpi)
        engine._track_stockout_duration(state, kpi)
        engine._accrue_idle_penalties(state, kpi)

        for roaster_id in engine.roasters:
            cur = state.status[roaster_id]
            if prev_status[roaster_id] == "SETUP" and cur == "IDLE":
                just_set_up[roaster_id] = True
            elif cur == "RUNNING":
                just_set_up[roaster_id] = False
            prev_status[roaster_id] = cur

        # Restock handled by heuristic, not by Q-learning.
        # This matches the engine.run() restock phase and keeps the
        # Q-table focused on roaster decisions only.
        engine._process_restock_decision_point(state, restock_helper, kpi)

        for roaster_id in engine.roasters:
            if state.status[roaster_id] != "IDLE" or not state.needs_decision[roaster_id]:
                continue

            s_new = discretize_roaster_state(
                state,
                roaster_id,
                params,
                just_set_up=just_set_up[roaster_id],
            )
            valid_actions = get_valid_roaster_actions(engine, state, roaster_id, allow_flex)
            _close_pending(s_new, valid_actions)

            action_idx = _choose_action(s_new, valid_actions)
            action_tuple = ACTION_MAP[action_idx]
            engine._apply_action(state, roaster_id, action_tuple, kpi)

            if state.status[roaster_id] == "RUNNING":
                just_set_up[roaster_id] = False
            prev_status[roaster_id] = state.status[roaster_id]

            prev_state_disc = s_new
            prev_action = action_idx

        pending_reward += kpi.net_profit() - slot_profit_before

    if prev_state_disc is not None:
        transitions.append((prev_state_disc, prev_action, pending_reward, None, ()))

    episode_profit = kpi.net_profit()
    reward_sum = sum(reward for _, _, reward, _, _ in transitions)
    if abs(reward_sum - episode_profit) > 1e-9:
        raise RuntimeError(
            f"Training reward mismatch: transitions sum to {reward_sum}, "
            f"but episode profit is {episode_profit}"
        )

    return transitions, episode_profit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train tabular Q-learning agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m q_learning.q_learning_train --name first_try\n"
            "  python -m q_learning.q_learning_train --time 3600 --name long_run\n"
            "  python -m q_learning.q_learning_train --episodes 5000 --alpha 0.05 --name quick\n"
        ),
    )
    parser.add_argument("--name", "-n", type=str, default="", help="Run name (used in folder and Q-table filename)")
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=None,
        help=f"Max episodes (default: {NUM_EPISODES} if no --time)",
    )
    parser.add_argument(
        "--time",
        "-t",
        type=float,
        default=None,
        help="Time limit in seconds (preferred over --episodes)",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA, help=f"Learning rate (default: {ALPHA})")
    parser.add_argument("--gamma", type=float, default=GAMMA, help=f"Discount factor (default: {GAMMA})")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N episodes (0=disabled)",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Train on a grid of UPS scenarios instead of the env default (lambda, mu)",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip automatic evaluation after training")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Episodes per scenario for auto-eval (default: 50)",
    )
    args = parser.parse_args(argv)

    use_time = args.time is not None
    if use_time:
        max_episodes = 10_000_000
        time_budget = args.time
    else:
        max_episodes = args.episodes if args.episodes is not None else NUM_EPISODES
        time_budget = float("inf")

    params = get_sim_params()
    q_table: dict = defaultdict(float)
    alpha = args.alpha
    gamma = args.gamma
    allow_flex = params.get("allow_r3_flex", True)
    run_name = args.name

    robust_mode = args.robust
    env_lambda = params.get("ups_lambda", 2.0)
    env_mu = params.get("ups_mu", 20.0)
    if robust_mode:
        lambda_options = ROBUST_LAMBDA
        mu_options = ROBUST_MU
    else:
        lambda_options = [env_lambda]
        mu_options = [env_mu]

    # Create result dir up-front so checkpoints have a destination.
    result_dir = _make_result_dir(run_name, args.episodes or NUM_EPISODES, alpha, gamma)
    print(f"  Output dir: {result_dir}")

    rewards_log: list[float] = []
    t_start = time.perf_counter()

    stop_reason = "episodes"
    if use_time:
        print(f"Training Q-learning: time budget {time_budget:.0f}s  (name={run_name or '(none)'})")
    else:
        print(f"Training Q-learning: {max_episodes} episodes  (name={run_name or '(none)'})")
    if robust_mode:
        print(f"  UPS mode: ROBUST - lambda={lambda_options}, mu={mu_options}")
    else:
        print(f"  UPS mode: env default - lambda={env_lambda}, mu={env_mu}")
    print(f"  alpha={alpha}  gamma={gamma}  linear epsilon decay {EPSILON_START} -> {EPSILON_END}")

    ep = 0
    while ep < max_episodes:
        elapsed_now = time.perf_counter() - t_start
        if use_time and elapsed_now >= time_budget:
            stop_reason = f"time limit ({time_budget:.0f}s)"
            break

        if use_time:
            progress = min(1.0, elapsed_now / (0.7 * time_budget))
        else:
            progress = min(1.0, ep / (0.7 * max_episodes))
        epsilon = max(EPSILON_END, EPSILON_START - progress * (EPSILON_START - EPSILON_END))

        lam = random.choice(lambda_options)
        mu = random.choice(mu_options)
        seed = random.randint(0, 2**31 - 1)
        ups_events = generate_ups_events(lam, mu, seed)

        engine = SimulationEngine(params)
        transitions, episode_profit = _train_one_episode(
            params, engine, q_table, epsilon, ups_events, allow_flex,
        )

        for s, a, r, s_next, next_valid_actions in transitions:
            old_q = q_table[(s, a)]
            if s_next is not None and next_valid_actions:
                max_next_q = max(
                    (q_table.get((s_next, a2), 0.0) for a2 in next_valid_actions),
                    default=0.0,
                )
            else:
                max_next_q = 0.0
            q_table[(s, a)] = old_q + alpha * (r + gamma * max_next_q - old_q)

        rewards_log.append(episode_profit)
        ep += 1

        if ep % 500 == 0:
            recent = rewards_log[-500:]
            avg = sum(recent) / len(recent)
            elapsed = time.perf_counter() - t_start
            eps_per_sec = ep / elapsed
            remaining = ""
            if use_time:
                left = max(0, time_budget - elapsed)
                remaining = f"  {left:.0f}s left"
            print(
                f"  ep {ep:>6} | "
                f"eps={epsilon:.4f} | "
                f"avg500=${avg:>10,.0f} | "
                f"Q-size={len(q_table):>8,} | "
                f"{eps_per_sec:.1f} ep/s{remaining}"
            )

        if args.checkpoint_interval > 0 and ep % args.checkpoint_interval == 0:
            ckpt_dir = result_dir / "_checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tag = f"_{run_name}" if run_name else ""
            save_q_table(dict(q_table), str(ckpt_dir / f"q_table_ckpt_{ep}{tag}.pkl"))

    total_episodes = ep
    elapsed = time.perf_counter() - t_start
    final_avg = sum(rewards_log[-1000:]) / max(1, min(1000, len(rewards_log)))

    if use_time:
        final_progress = min(1.0, elapsed / (0.7 * time_budget))
    else:
        final_progress = min(1.0, total_episodes / (0.7 * max_episodes))
    final_epsilon = max(EPSILON_END, EPSILON_START - final_progress * (EPSILON_START - EPSILON_END))

    qtable_filename = f"q_table_{run_name}.pkl" if run_name else "q_table.pkl"
    save_q_table(dict(q_table), str(result_dir / qtable_filename))

    log_filename = f"training_log_{run_name}.pkl" if run_name else "training_log.pkl"
    with open(result_dir / log_filename, "wb") as f:
        pickle.dump(rewards_log, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "name": run_name,
        "episodes": total_episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_schedule": "linear (reaches eps_end at 70% of budget)",
        "final_epsilon": round(final_epsilon, 6),
        "state_formulation": "roaster_tabular_with_heuristic_restock",
        "ups_mode": "robust" if robust_mode else "env_default",
        "ups_lambda": lambda_options if robust_mode else env_lambda,
        "ups_mu": mu_options if robust_mode else env_mu,
        "elapsed_seconds": round(elapsed, 1),
        "stop_reason": stop_reason,
        "q_table_entries": len(q_table),
        "final_avg_profit_1000": round(final_avg, 2),
        "q_table_file": qtable_filename,
        "training_log_file": log_filename,
        "timestamp": datetime.now().isoformat(),
    }
    with open(result_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    latest_q = Path(__file__).resolve().parent / "q_table.pkl"
    save_q_table(dict(q_table), str(latest_q))
    latest_log = Path(__file__).resolve().parent / "training_log.pkl"
    with open(latest_log, "wb") as f:
        pickle.dump(rewards_log, f, protocol=pickle.HIGHEST_PROTOCOL)

    result_dir = _rename_with_profit(result_dir, final_avg)

    print(f"\nTraining complete ({stop_reason})")
    print(f"  Episodes: {total_episodes:,}  |  Wall time: {elapsed:.1f}s  |  {total_episodes / elapsed:.1f} ep/s")
    print(f"  Q-table: {len(q_table):,} entries")
    print(f"  Final epsilon: {final_epsilon:.4f}")
    print(f"  Final avg profit (last 1000): ${final_avg:,.0f}")
    print(f"  Result folder: {result_dir}")
    print(f"  Latest copy:   {latest_q}")

    if not args.no_eval:
        q_table_path = str(result_dir / qtable_filename)
        print(f"\n{'=' * 72}")
        print("AUTO-EVALUATION: generating report...")
        print(f"{'=' * 72}\n")
        from q_learning.evaluate import main as run_main

        run_main(["--file", q_table_path])


if __name__ == "__main__":
    main()
