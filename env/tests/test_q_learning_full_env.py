"""Regression tests for Q-learning and restock-policy integration."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dispatch.dispatching_heuristic import DispatchingHeuristic
from env.data_bridge import get_sim_params
from env.simulation_engine import SimulationEngine
from q_learning.q_learning_train import _train_one_episode
from q_learning.q_strategy import QStrategy, discretize_roaster_state


def make_params():
    return get_sim_params()


def _eligible_count(params: dict, line_id: str, sku: str) -> int:
    return sum(
        1
        for roaster_id in params["roasters"]
        if params["R_pipe"][roaster_id] == line_id
        and sku in params["R_elig_skus"][roaster_id]
    )


def test_roaster_state_reflects_gc_inventory():
    params = make_params()
    engine = SimulationEngine(params)
    state = engine._initialize_state()

    altered = copy.deepcopy(state)
    altered.gc_stock[("L1", "NDG")] = 0
    altered.gc_stock[("L1", "BUSTA")] = 0

    base = discretize_roaster_state(state, "R2", params)
    changed = discretize_roaster_state(altered, "R2", params)

    assert base != changed


def test_qstrategy_decide_restock_delegates_to_dispatch(monkeypatch):
    params = make_params()
    strategy = QStrategy(params, q_table={})
    engine = SimulationEngine(params)
    state = engine._initialize_state()
    sentinel = ("START_RESTOCK", "L1", "PSC")

    monkeypatch.setattr(strategy._restock_helper, "decide_restock", lambda current_state: sentinel)

    assert strategy.decide_restock(state) == sentinel


def test_reorder_points_are_computed_from_input_parameters():
    params = make_params()
    heuristic = DispatchingHeuristic(params)
    report = heuristic.get_reorder_point_report()

    assert set(report) == {f"{line}_{sku}" for line, sku in params["feasible_gc_pairs"]}

    for line_id, sku in params["feasible_gc_pairs"]:
        key = f"{line_id}_{sku}"
        expected_count = _eligible_count(params, line_id, sku)
        expected_rate = expected_count / params["roast_time_by_sku"][sku]
        expected_exact = math.ceil(params["restock_duration"] * expected_rate)
        expected_impl = math.ceil((params["restock_duration"] + 1) * expected_rate)

        assert report[key]["roast_time"] == params["roast_time_by_sku"][sku]
        assert report[key]["restock_duration"] == params["restock_duration"]
        assert report[key]["eligible_roaster_count"] == expected_count
        assert report[key]["max_depletion_rate"] == expected_rate
        assert report[key]["ROP_exact"] == expected_exact
        assert report[key]["ROP_impl"] == expected_impl


def test_reorder_point_changes_when_roast_time_changes():
    params = make_params()
    base = DispatchingHeuristic(params).reorder_point_impl[("L2", "PSC")]

    changed_params = copy.deepcopy(params)
    changed_params["roast_time_by_sku"]["PSC"] = 10
    changed = DispatchingHeuristic(changed_params).reorder_point_impl[("L2", "PSC")]

    assert changed > base


def test_reorder_point_changes_when_restock_duration_changes():
    params = make_params()
    base = DispatchingHeuristic(params).reorder_point_impl[("L2", "PSC")]

    changed_params = copy.deepcopy(params)
    changed_params["restock_duration"] = params["restock_duration"] + 5
    changed = DispatchingHeuristic(changed_params).reorder_point_impl[("L2", "PSC")]

    assert changed > base


def test_reorder_point_depends_on_eligible_roaster_count():
    params = make_params()
    base_heuristic = DispatchingHeuristic(params)
    base_count = base_heuristic.eligible_roaster_count[("L1", "NDG")]
    base_rop = base_heuristic.reorder_point_impl[("L1", "NDG")]

    changed_params = copy.deepcopy(params)
    changed_params["R_elig_skus"]["R1"] = [sku for sku in changed_params["R_elig_skus"]["R1"] if sku != "NDG"]
    changed_heuristic = DispatchingHeuristic(changed_params)
    changed_count = changed_heuristic.eligible_roaster_count[("L1", "NDG")]
    changed_rop = changed_heuristic.reorder_point_impl[("L1", "NDG")]

    assert changed_count == base_count - 1
    assert changed_rop < base_rop


def test_dispatch_restock_uses_rop_impl_not_fixed_fraction():
    params = make_params()
    heuristic = DispatchingHeuristic(params)
    engine = SimulationEngine(params)
    state = engine._initialize_state()

    for job_id in params["jobs"]:
        state.mto_remaining[job_id] = 0

    state.rc_stock["L1"] = params["max_rc"]
    state.rc_stock["L2"] = 0
    state.gc_stock[("L1", "PSC")] = params["gc_capacity"][("L1", "PSC")]
    state.gc_stock[("L2", "PSC")] = 9

    assert heuristic.reorder_point_impl[("L2", "PSC")] < 9
    assert heuristic.decide_restock(state) == ("WAIT",)

    state.gc_stock[("L2", "PSC")] = heuristic.reorder_point_impl[("L2", "PSC")]
    assert heuristic.decide_restock(state) == ("START_RESTOCK", "L2", "PSC")


def test_mto_only_silos_not_restocked_without_remaining_mto_demand():
    params = make_params()
    heuristic = DispatchingHeuristic(params)
    engine = SimulationEngine(params)
    state = engine._initialize_state()

    for job_id in params["jobs"]:
        state.mto_remaining[job_id] = 0

    state.gc_stock[("L1", "NDG")] = 0
    state.gc_stock[("L1", "BUSTA")] = 0
    state.gc_stock[("L1", "PSC")] = params["gc_capacity"][("L1", "PSC")]
    state.gc_stock[("L2", "PSC")] = params["gc_capacity"][("L2", "PSC")]
    state.rc_stock["L1"] = params["max_rc"]
    state.rc_stock["L2"] = params["max_rc"]

    assert heuristic.decide_restock(state) == ("WAIT",)


def test_training_emits_only_roaster_transitions():
    params = make_params()
    engine = SimulationEngine(params)
    transitions, _ = _train_one_episode(
        params=params,
        engine=engine,
        q_table=defaultdict(float),
        epsilon=0.0,
        ups_events=[],
        allow_flex=params.get("allow_r3_flex", True),
    )

    assert transitions
    assert all(state_key[0] == "roaster" for state_key, *_ in transitions)


def test_readme_documents_reorder_points_and_retraining():
    ql_readme = (ROOT / "q_learning" / "README.md").read_text(encoding="utf-8")
    env_readme = (ROOT / "env" / "README.md").read_text(encoding="utf-8")

    assert "ROP_exact(line, sku) = ceil(L_restock * d_max(line, sku))" in ql_readme
    assert "ROP_impl(line, sku) = ceil((L_restock + 1) * d_max(line, sku))" in ql_readme
    assert "retraining is recommended" in ql_readme.lower()
    assert "training and evaluation" in ql_readme.lower()
    assert "input-driven reorder-point policy" in env_readme
