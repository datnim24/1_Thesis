"""Q-table strategy wrapper compatible with SimulationEngine.

Runtime uses:
  1. tabular Q-value lookup for per-roaster roasting decisions
  2. the dispatch-layer restock heuristic for the global restock layer
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.simulation_engine import SimulationEngine
from dispatch.dispatching_heuristic import DispatchingHeuristic


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
