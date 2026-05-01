"""Unit tests for paeng_ddqn_v2 state builder + feasibility mask.

Self-contained runner — no pytest dependency. Each test is a function returning True/False.
Run via:  python -m paeng_ddqn_v2.tests.test_state_builder
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PPOmask.Engine.data_loader import load_data
from env.simulation_engine import SimulationEngine
from env.simulation_state import SimulationState
from paeng_ddqn_v2.strategy_v2 import (
    build_state_v2,
    compute_feasibility_mask_v2,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _DummyStrat:
    def __init__(self):
        self.kpi_ref = None
    def reset_episode(self): pass
    def end_episode(self, *_a, **_k): pass
    def decide(self, _s, _r): return ("WAIT",)
    def decide_restock(self, _s): return ("WAIT",)


def make_baseline_state():
    """Run one full episode with WAIT-only strategy → end-of-shift sim_state."""
    data = load_data()
    params = data.to_env_params()
    engine = SimulationEngine(params)
    kpi, sim_state = engine.run(_DummyStrat(), [])
    return data, params, sim_state, kpi


def make_early_state():
    """Construct a t=0 sim_state with all roasters IDLE, full MTO remaining."""
    data = load_data()
    params = data.to_env_params()
    s = SimulationState()
    s.t = 0
    for r in ("R1", "R2", "R3", "R4", "R5"):
        s.status[r] = "IDLE"
        s.remaining[r] = 0
        s.current_batch[r] = None
        s.last_sku[r] = "PSC"
        s.setup_target_sku[r] = None
    s.rc_stock = {"L1": 40, "L2": 40}
    s.gc_stock = {(line, sku): 5 for line in ("L1", "L2") for sku in ("PSC", "NDG", "BUSTA")}
    s.pipeline_busy = {"L1": 0, "L2": 0}
    s.pipeline_mode = {"L1": "FREE", "L2": "FREE"}
    s.mto_remaining = dict(params.get("job_batches", {}))
    s.mto_tardiness = {jid: 0.0 for jid in params.get("job_batches", {})}
    return data, params, s


# ---------------------------------------------------------------------------
# Stage T1: build_state_v2
# ---------------------------------------------------------------------------

def t_shape_and_dtype(baseline):
    data, params, sim_state, _ = baseline
    state, sf = build_state_v2(data, sim_state, params)
    assert state.shape == (3, 35), f"expected (3,35), got {state.shape}"
    assert state.dtype == np.float32
    assert sf.shape == (3,), f"expected (3,), got {sf.shape}"
    assert sf.dtype == np.float32


def t_no_nan_or_inf(baseline):
    data, params, sim_state, _ = baseline
    state, sf = build_state_v2(data, sim_state, params)
    assert np.all(np.isfinite(state)), "state contains NaN or Inf"
    assert np.all(np.isfinite(sf)), "sf contains NaN or Inf"


def t_bounded_values(baseline):
    data, params, sim_state, _ = baseline
    state, sf = build_state_v2(data, sim_state, params)
    assert state.min() >= -1.1, f"state.min() = {state.min()}"
    assert state.max() <= 1.1, f"state.max() = {state.max()}"
    assert sf.min() >= 0.0
    assert sf.max() <= 1.01


def t_su_fractions_sum_le_1(baseline):
    data, params, sim_state, _ = baseline
    state, _ = build_state_v2(data, sim_state, params)
    for f_idx in range(3):
        s = state[f_idx, 20:23].sum()
        assert s <= 1.0 + 1e-5, f"Su row {f_idx} sums to {s}"


def t_sa_one_hot_integrity(baseline):
    data, params, sim_state, _ = baseline
    state, _ = build_state_v2(data, sim_state, params, last_action_id=4)
    assert state[1, 23] == 1.0, f"state[1,23]={state[1,23]}"
    assert state[1, 24] == 1.0, f"state[1,24]={state[1,24]}"
    for f in range(3):
        if f != 1:
            assert state[f, 23] == 0.0
            assert state[f, 24] == 0.0


def t_sa_zeros_when_no_last_action(baseline):
    data, params, sim_state, _ = baseline
    state, _ = build_state_v2(data, sim_state, params, last_action_id=None)
    assert state[:, 23:25].sum() == 0.0


def t_sf_normalization(baseline):
    data, params, sim_state, _ = baseline
    _, sf = build_state_v2(data, sim_state, params)
    assert abs(sf.max() - 1.0) < 1e-5, f"sf.max()={sf.max()}"
    assert sf.min() > 0.0


def t_slack_bucket_active_jobs(early):
    """At t=0, NDG MTO jobs should have positive slack → buckets 2..5."""
    data, params, sim_state = early
    state, _ = build_state_v2(data, sim_state, params)
    job_sku = params.get("job_sku", {})
    if any(s == "NDG" for s in job_sku.values()):
        ndg_neg = state[1, 0:2].sum()
        ndg_pos = state[1, 2:6].sum()
        assert ndg_pos > 0.0, f"NDG should have non-neg slack at t=0; pos buckets={ndg_pos}"
        assert ndg_neg == 0.0, f"NDG should have no neg slack at t=0; got {ndg_neg}"


# ---------------------------------------------------------------------------
# Stage T2: compute_feasibility_mask_v2
# ---------------------------------------------------------------------------

def t_action_pscpsc_feasible_at_t0(early):
    """At t=0, all roasters last_sku=PSC + PSC demand → action 0 (PSC→PSC) feasible."""
    data, params, sim_state = early
    sim_state.rc_stock["L1"] = 0
    sim_state.rc_stock["L2"] = 0
    mask = compute_feasibility_mask_v2(data, sim_state, params)
    assert mask[0], "(PSC->PSC) should be feasible at t=0 with PSC demand"


def t_action_psc_to_busta_only_via_r2(early):
    """(from=PSC, to=BUSTA) requires SOME roaster with last_sku=PSC AND can produce BUSTA.
    Only R2 can produce BUSTA; if R2.last_sku=PSC and BUSTA demand exists → action feasible.
    """
    data, params, sim_state = early
    # R2 is set up as PSC (default) and can produce BUSTA. BUSTA demand exists at t=0.
    mask = compute_feasibility_mask_v2(data, sim_state, params)
    job_sku = params.get("job_sku", {})
    if any(s == "BUSTA" for s in job_sku.values()):
        # action 2 = from_setup=PSC * 3 + to_dispatch=BUSTA → a=2
        assert mask[2], "(PSC->BUSTA) should be feasible: R2 last=PSC, can do BUSTA, demand exists"


def t_action_busta_from_set_infeasible_initially(early):
    """from=BUSTA setup requires some roaster with last_sku=BUSTA. At t=0 all are PSC."""
    data, params, sim_state = early
    # Force all roasters to PSC setup
    for r in ("R1", "R2", "R3", "R4", "R5"):
        sim_state.last_sku[r] = "PSC"
    mask = compute_feasibility_mask_v2(data, sim_state, params)
    # Actions 6,7,8 require from=BUSTA → infeasible since no roaster has last_sku=BUSTA
    assert not mask[6] and not mask[7] and not mask[8], "from=BUSTA actions need a BUSTA-setup roaster"


def t_no_demand_no_feasibility(early):
    """No demand anywhere → mask is all-False, then failsafe sets all-True (engine WAITs)."""
    data, params, sim_state = early
    sim_state.mto_remaining = {jid: 0 for jid in sim_state.mto_remaining}
    sim_state.rc_stock = {"L1": 40, "L2": 40}
    params_nd = dict(params)
    params_nd["consume_events"] = {"L1": [], "L2": []}
    for line in ("L1", "L2"):
        cap = params.get("gc_capacity", {}).get((line, "PSC"), 100)
        sim_state.gc_stock[(line, "PSC")] = cap
    mask = compute_feasibility_mask_v2(data, sim_state, params_nd)
    # Failsafe: all-True (since no demand means engine produces WAITs anyway)
    assert mask.all(), f"expected failsafe all-True when no demand, got {mask}"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

T1_TESTS = [
    ("shape_and_dtype",       t_shape_and_dtype),
    ("no_nan_or_inf",         t_no_nan_or_inf),
    ("bounded_values",        t_bounded_values),
    ("su_fractions_sum_le_1", t_su_fractions_sum_le_1),
    ("sa_one_hot_integrity",  t_sa_one_hot_integrity),
    ("sa_zeros_no_last",      t_sa_zeros_when_no_last_action),
    ("sf_normalization",      t_sf_normalization),
]

T2_TESTS = [
    ("psc_psc_feasible",       t_action_pscpsc_feasible_at_t0),
    ("psc_to_busta_via_r2",    t_action_psc_to_busta_only_via_r2),
    ("from_busta_infeasible",  t_action_busta_from_set_infeasible_initially),
    ("no_demand_failsafe",     t_no_demand_no_feasibility),
]


def main():
    print("[T1] State-builder tests")
    baseline = make_baseline_state()
    early    = make_early_state()
    p, f = 0, 0
    for name, fn in T1_TESTS:
        try:
            fn(baseline)
            print(f"  PASS  {name}")
            p += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc(limit=2)
            f += 1
    try:
        t_slack_bucket_active_jobs(early)
        print(f"  PASS  slack_bucket_active_jobs")
        p += 1
    except Exception:
        print(f"  FAIL  slack_bucket_active_jobs")
        traceback.print_exc(limit=2)
        f += 1

    print(f"\n[T2] Feasibility-mask tests")
    for name, fn in T2_TESTS:
        try:
            # fresh early-state per test (sim_state mutated in tests)
            fn(make_early_state())
            print(f"  PASS  {name}")
            p += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc(limit=2)
            f += 1

    print(f"\n[summary]  passed={p}  failed={f}")
    return 0 if f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
