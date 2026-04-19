"""Regression coverage for the environment audit fixes."""

from __future__ import annotations

import copy
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from dispatch.dispatching_heuristic import DispatchingHeuristic
from env.data_bridge import get_sim_params
from env.export import export_run
from env.simulation_engine import SimulationEngine
from env.simulation_state import UPSEvent
from q_learning.q_learning_train import _train_one_episode
from result_schema import load_default_parameters, reconstruct_gc_trajectory
from verify_result import _load_any_result, verify


def _cancelled_entry(
    batch_id,
    sku: str,
    roaster: str,
    start: int,
    end: int,
    cancel_time: int,
) -> dict:
    return {
        "batch_id": str(batch_id),
        "sku": sku,
        "roaster": roaster,
        "start": start,
        "end": end,
        "status": "cancelled",
        "cancel_time": cancel_time,
    }


def _restock_entry(line_id: str, sku: str, start: int, end: int, qty: int = 5) -> dict:
    return {
        "line_id": line_id,
        "sku": sku,
        "start": start,
        "end": end,
        "qty": qty,
    }


class _ActionBias(dict):
    """State-agnostic Q-value shim that prefers MTO actions."""

    def get(self, key, default=None):
        _, action_idx = key
        if action_idx == 8:
            return 30.0
        if action_idx in {6, 7}:
            return 20.0
        if action_idx == 16:
            return -10.0
        return 0.0


class _ProbeStrategy:
    """Small deterministic policy used to ensure tardiness is exercised."""

    def __init__(self, params: dict):
        self.params = params
        self._restock_helper = DispatchingHeuristic(params)

    def decide_restock(self, state):
        return self._restock_helper.decide_restock(state)

    def decide(self, state, roaster_id: str):
        if roaster_id == "R1":
            return ("NDG",)
        if roaster_id == "R2":
            return ("BUSTA",)
        if roaster_id == "R3":
            return ("PSC", "L2")
        if roaster_id == "R4":
            return ("PSC", "L2")
        if roaster_id == "R5":
            return ("PSC", "L2")
        return ("WAIT",)


def _make_small_mto_params() -> dict:
    params = copy.deepcopy(get_sim_params())
    params["job_batches"] = {"J1": 1, "J2": 1}
    params["job_due"] = {"J1": 10, "J2": 10}
    params["mto_batches"] = [("J1", 0), ("J2", 0)]
    params["all_batches"] = list(params["mto_batches"]) + list(params["psc_pool"])
    params["batch_sku"] = {
        ("J1", 0): "NDG",
        ("J2", 0): "BUSTA",
        **{batch_id: "PSC" for batch_id in params["psc_pool"]},
    }
    params["batch_is_mto"] = {
        ("J1", 0): True,
        ("J2", 0): True,
        **{batch_id: False for batch_id in params["psc_pool"]},
    }
    return params


def test_q_learning_rewards_sum_to_episode_profit():
    params = _make_small_mto_params()
    probe_engine = SimulationEngine(params)
    probe_kpi, _ = probe_engine.run(_ProbeStrategy(params), [])
    assert probe_kpi.tard_cost > 0, "Regression needs a tardiness-bearing episode"

    random.seed(7)
    engine = SimulationEngine(params)
    transitions, episode_profit = _train_one_episode(
        params=params,
        engine=engine,
        q_table=_ActionBias(),
        epsilon=0.0,
        ups_events=[],
        allow_flex=params["allow_r3_flex"],
    )

    reward_sum = sum(reward for _, _, reward, _, _ in transitions)
    assert transitions, "Episode should emit at least one learning transition"
    assert reward_sum == pytest.approx(episode_profit, abs=1e-9)


def test_cancelled_psc_batch_still_consumes_gc():
    params = load_default_parameters()
    cancelled = [
        _cancelled_entry(("R1", 0), "PSC", "R1", start=5, end=20, cancel_time=8),
    ]

    traj = reconstruct_gc_trajectory([], [], params, cancelled)
    init_stock = params["gc_init"]["L1_PSC"]

    assert traj["L1_PSC"][4] == init_stock
    assert traj["L1_PSC"][5] == init_stock - 1
    assert traj["L1_PSC"][-1] == init_stock - 1
    assert traj["events"]["L1_PSC"]["cancelled_batch_starts"] == [5]


def test_cancelled_busta_batch_still_consumes_gc():
    params = load_default_parameters()
    cancelled = [
        _cancelled_entry(("J2", 0), "BUSTA", "R2", start=12, end=30, cancel_time=18),
    ]

    traj = reconstruct_gc_trajectory([], [], params, cancelled)
    init_stock = params["gc_init"]["L1_BUSTA"]

    assert traj["L1_BUSTA"][11] == init_stock
    assert traj["L1_BUSTA"][12] == init_stock - 1
    assert traj["L1_BUSTA"][-1] == init_stock - 1
    assert traj["events"]["L1_BUSTA"]["cancelled_batch_starts"] == [12]


def test_cancelled_batch_and_restock_reconstruct_gc_correctly():
    params = load_default_parameters()
    cancelled = [
        _cancelled_entry(("R1", 0), "PSC", "R1", start=5, end=20, cancel_time=8),
    ]
    restocks = [
        _restock_entry("L1", "PSC", start=12, end=27, qty=5),
    ]

    traj = reconstruct_gc_trajectory([], restocks, params, cancelled)
    init_stock = params["gc_init"]["L1_PSC"]

    assert traj["L1_PSC"][26] == init_stock - 1
    assert traj["L1_PSC"][27] == init_stock - 1 + 5
    assert traj["L1_PSC"][-1] == init_stock - 1 + 5
    assert traj["events"]["L1_PSC"]["cancelled_batch_starts"] == [5]
    assert traj["events"]["L1_PSC"]["restock_starts"] == [12]
    assert traj["events"]["L1_PSC"]["restock_completions"] == [27]


def test_verifier_passes_on_exported_cancelled_run(tmp_path):
    params = get_sim_params()
    strategy = DispatchingHeuristic(params)
    engine = SimulationEngine(params)
    ups_events = [UPSEvent(t=1, roaster_id="R3", duration=10)]

    kpi, state = engine.run(strategy, ups_events)
    assert state.cancelled_batches, "Regression needs at least one cancelled batch"

    paths = export_run(
        kpi.to_dict(),
        state,
        params,
        ups_events,
        output_dir=str(tmp_path),
        run_id="audit_cancelled",
    )

    result, validation_errors = _load_any_result(paths["result_json"])
    assert not validation_errors

    gc_traj = reconstruct_gc_trajectory(
        result["schedule"],
        result.get("restocks", []),
        result["parameters"],
        result.get("cancelled_batches", []),
    )
    assert gc_traj["L2_PSC"][-1] == int(result["kpi"]["gc_final"]["L2_PSC"])

    ok, lines = verify(result)
    assert ok, "\n".join(lines)


def test_run_schedule_argument_is_disabled():
    params = get_sim_params()
    engine = SimulationEngine(params)

    with pytest.raises(NotImplementedError, match="disabled"):
        engine.run(DispatchingHeuristic(params), [], schedule={"R1": []})
