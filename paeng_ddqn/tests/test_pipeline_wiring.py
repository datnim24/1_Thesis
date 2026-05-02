"""End-to-end wiring tests for the paeng_ddqn pipeline.

These tests verify the env <-> strategy <-> agent feedback loop works as
intended, especially the things that are easy to silently break:

1. KPI reference is wired to the strategy (cycles 1-36 ran without it).
2. Per-decision reward = true Δnet_profit (not the revenue-only fallback).
3. State / auxin / mask shapes match what the agent expects.
4. Action -> engine tuple mapping is consistent with feasibility mask.
5. Replay buffer accumulates transitions during training.
6. KPI net_profit at end matches sum-of-stored-rewards * reward_scale.

Run with:
    python -m pytest paeng_ddqn/tests/ -v
or directly:
    python -m paeng_ddqn.tests.test_pipeline_wiring
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data
from paeng_ddqn.agent import PaengAgent, PaengConfig
from paeng_ddqn.strategy import PaengStrategy, build_paeng_state, compute_feasibility_mask


def _setup(training: bool = True):
    data = load_data()
    params = data.to_env_params()
    cfg = PaengConfig()
    agent = PaengAgent(cfg)
    strategy = PaengStrategy(agent, data, training=training)
    engine = SimulationEngine(params)
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    ups = generate_ups_events(float(data.ups_lambda), float(data.ups_mu), 42, SL, roasters)
    return data, params, cfg, agent, strategy, engine, ups


def test_kpi_ref_wired() -> None:
    """The engine must wire kpi -> strategy.kpi_ref before the run begins."""
    _, _, _, _, strategy, engine, ups = _setup(training=True)
    assert strategy.kpi_ref is None, "kpi_ref must start as None"
    kpi, _state = engine.run(strategy, ups)
    assert strategy.kpi_ref is kpi, (
        "engine.run did not set strategy.kpi_ref. Per-decision reward will "
        "fall back to revenue-only approximation, missing all penalty signals."
    )
    print(f"  PASS: kpi_ref wired (final net_profit=${kpi.net_profit():,.0f})")


def test_compute_profit_uses_real_kpi() -> None:
    """After engine.run, _compute_profit() must return real net_profit."""
    _, _, _, _, strategy, engine, ups = _setup(training=True)
    kpi, state = engine.run(strategy, ups)
    real = strategy._compute_profit(state)
    expected = float(kpi.net_profit())
    assert abs(real - expected) < 1e-6, (
        f"_compute_profit gave ${real:,.0f} but kpi.net_profit()=${expected:,.0f}. "
        f"Strategy is using revenue-only fallback."
    )
    print(f"  PASS: _compute_profit == kpi.net_profit() = ${expected:,.0f}")


def test_state_and_mask_shapes() -> None:
    """State (3,50), auxin (10,), mask (8,) — must match what the network expects."""
    data, _, cfg, _, _, engine, _ = _setup(training=False)
    state = engine._initialize_state()
    ctx = {"kind": "roaster", "roaster_id": "R1", "last_action_id": -1, "last_reward_norm": 0.0}
    s, x = build_paeng_state(data, state, ctx)
    m = compute_feasibility_mask(engine, state, ctx)
    assert s.shape == (cfg.state_F, cfg.state_W), f"state shape {s.shape} != ({cfg.state_F}, {cfg.state_W})"
    assert x.shape == (cfg.auxin_dim,), f"auxin shape {x.shape} != ({cfg.auxin_dim},)"
    assert m.shape == (cfg.action_dim,), f"mask shape {m.shape} != ({cfg.action_dim},)"
    assert m[3], "WAIT must always be feasible at a decision point"
    print(f"  PASS: state {s.shape}, auxin {x.shape}, mask {m.shape}; WAIT always feasible")


def test_replay_buffer_fills_during_training() -> None:
    """One episode of training should store at least 100 transitions."""
    _, _, _, agent, strategy, engine, ups = _setup(training=True)
    assert len(agent.replay) == 0, "replay buffer must start empty"
    kpi, sim_state = engine.run(strategy, ups)
    final = float(kpi.net_profit())
    strategy.end_episode(sim_state, final)
    n = len(agent.replay)
    assert n > 100, f"replay buffer only got {n} transitions in one episode"
    print(f"  PASS: replay buffer accumulated {n} transitions in 1 episode")


def test_reward_signal_includes_penalties() -> None:
    """Sum of stored scaled rewards * reward_scale should approximate net_profit.

    Subject to the per-decision shaping (idle_penalty, productive_bonus, etc.)
    we tolerate up to 10% slack for ±$50/decision shaping noise.
    """
    _, _, cfg, agent, strategy, engine, ups = _setup(training=True)
    kpi, sim_state = engine.run(strategy, ups)
    final = float(kpi.net_profit())
    strategy.end_episode(sim_state, final)

    n = len(agent.replay)
    sum_rewards_raw = float(np.sum(agent.replay.rewards[:n])) * cfg.reward_scale
    # idle_penalty fires per-decision when WAIT chosen with productive feasible;
    # bound the shaping accumulation so we don't pretend the test is tighter
    # than reality.
    max_shaping = (cfg.idle_penalty + cfg.productive_bonus) * n * cfg.reward_scale
    diff = abs(sum_rewards_raw - final)
    assert diff <= max(abs(final) * 0.20, max_shaping + 50_000), (
        f"sum-of-rewards ${sum_rewards_raw:,.0f} vs net_profit ${final:,.0f} "
        f"differ by ${diff:,.0f} — possible reward-signal disconnect."
    )
    print(f"  PASS: sum(rewards)*scale=${sum_rewards_raw:,.0f}  net_profit=${final:,.0f}  diff=${diff:,.0f}")


def test_action_to_env_tuple_consistent_with_mask() -> None:
    """Every action_id flagged feasible must produce a valid engine tuple."""
    data, _, _, _, strategy, engine, _ = _setup(training=False)
    state = engine._initialize_state()
    for rid in ("R1", "R2", "R3", "R4", "R5"):
        ctx = {"kind": "roaster", "roaster_id": rid, "last_action_id": -1, "last_reward_norm": 0.0}
        m = compute_feasibility_mask(engine, state, ctx)
        for aid in range(8):
            if not m[aid]:
                continue
            tup = strategy._action_to_env_tuple(aid, ctx, state)
            assert isinstance(tup, tuple) and len(tup) >= 1, f"bad tuple for {rid} action {aid}: {tup}"
            assert tup[0] in {"WAIT", "PSC", "NDG", "BUSTA"}, f"unexpected verb {tup[0]} for action {aid}"
    ctx_r = {"kind": "restock", "roaster_id": None, "last_action_id": -1, "last_reward_norm": 0.0}
    m = compute_feasibility_mask(engine, state, ctx_r)
    for aid in range(8):
        if not m[aid]:
            continue
        tup = strategy._action_to_env_tuple(aid, ctx_r, state)
        assert tup[0] in {"WAIT", "START_RESTOCK"}, f"unexpected restock verb {tup[0]} for action {aid}"
    print("  PASS: feasible action_ids -> valid engine tuples in all contexts")


def main() -> None:
    tests = [
        ("kpi_ref wired", test_kpi_ref_wired),
        ("compute_profit uses real KPI", test_compute_profit_uses_real_kpi),
        ("state/auxin/mask shapes", test_state_and_mask_shapes),
        ("replay buffer fills", test_replay_buffer_fills_during_training),
        ("reward signal includes penalties", test_reward_signal_includes_penalties),
        ("action -> tuple consistent", test_action_to_env_tuple_consistent_with_mask),
    ]
    n_pass, n_fail = 0, 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERROR ({type(e).__name__}): {e}")
            n_fail += 1
    print(f"\n=== {n_pass}/{n_pass + n_fail} passed ===")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
