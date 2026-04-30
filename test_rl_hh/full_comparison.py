"""Detailed 100-seed comparison: dispatching baseline, Q-learning, PPO, RL-HH,
plus CP-SAT no-UPS ceiling.

Same seeds (900000-900099, lambda=5, mu=20) for all 4 simulation-based methods.
CP-SAT no-UPS is a single deterministic solve (one number = ceiling).

Output: results/full_comparison_<ts>/comparison_100seed.json + a markdown report.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data
# PPO archived to OLDCODE/PPOmask_archive/ on 2026-04-28 (v4 plan).
# Paeng DDQN takes PPO's slot in the comparison once paeng_ddqn/ is built.

from dispatch.dispatching_heuristic import DispatchingHeuristic
from q_learning.q_strategy import QStrategy, load_q_table
from rl_hh.meta_agent import DuelingDDQNAgent
from rl_hh.rl_hh_strategy import RLHHStrategy

from CPSAT_Pure.runner import run_pure_cpsat


# ---------------------------------------------------------------------------
# Per-seed evaluation runner
# ---------------------------------------------------------------------------

def _run_strategy_once(engine: SimulationEngine, strategy, ups_events) -> dict:
    """Run one episode and extract KPI dict."""
    kpi, state = engine.run(strategy, ups_events)
    k = kpi.to_dict()
    return {
        "net_profit": float(kpi.net_profit()),
        "revenue": float(k["total_revenue"]),
        "tard_cost": float(k["tard_cost"]),
        "setup_cost": float(k["setup_cost"]),
        "stockout_cost": float(k["stockout_cost"]),
        "idle_cost": float(k["idle_cost"]),
        "psc_count": int(k["psc_count"]),
        "ndg_count": int(k.get("ndg_count", 0)),
        "busta_count": int(k.get("busta_count", 0)),
        "setup_events": int(k["setup_events"]),
        "restock_count": int(k["restock_count"]),
        "idle_min": int(k["idle_min"]),
        "stockout_duration_l1": int(k["stockout_duration"].get("L1", 0)) if isinstance(k.get("stockout_duration"), dict) else 0,
        "stockout_duration_l2": int(k["stockout_duration"].get("L2", 0)) if isinstance(k.get("stockout_duration"), dict) else 0,
    }


def evaluate_strategy(name: str, factory, seeds: list[int], data, params, ups_lambda, ups_mu) -> dict:
    """Evaluate a strategy factory over the seed set. factory(seed) -> strategy instance."""
    engine = SimulationEngine(params)
    SL = int(params["SL"])
    roasters = list(params["roasters"])

    per_seed = []
    t0 = time.perf_counter()
    for seed in seeds:
        strategy = factory(seed)
        ups = generate_ups_events(ups_lambda, ups_mu, seed, SL, roasters)
        kpi = _run_strategy_once(engine, strategy, ups)
        kpi["seed"] = seed
        per_seed.append(kpi)
    elapsed = time.perf_counter() - t0
    print(f"  {name}: {len(seeds)} seeds in {elapsed:.1f}s")
    return _aggregate(name, per_seed, elapsed)


def _aggregate(name: str, per_seed: list[dict], wall_sec: float) -> dict:
    profits = np.array([s["net_profit"] for s in per_seed])
    return {
        "method": name,
        "n_seeds": len(per_seed),
        "wall_sec": round(wall_sec, 1),
        "profit_mean": round(float(profits.mean()), 2),
        "profit_std": round(float(profits.std()), 2),
        "profit_min": round(float(profits.min()), 2),
        "profit_max": round(float(profits.max()), 2),
        "profit_median": round(float(np.median(profits)), 2),
        "profit_p25": round(float(np.percentile(profits, 25)), 2),
        "profit_p75": round(float(np.percentile(profits, 75)), 2),
        "mean_revenue": round(float(np.mean([s["revenue"] for s in per_seed])), 2),
        "mean_tard_cost": round(float(np.mean([s["tard_cost"] for s in per_seed])), 2),
        "mean_setup_cost": round(float(np.mean([s["setup_cost"] for s in per_seed])), 2),
        "mean_stockout_cost": round(float(np.mean([s["stockout_cost"] for s in per_seed])), 2),
        "mean_idle_cost": round(float(np.mean([s["idle_cost"] for s in per_seed])), 2),
        "mean_psc": round(float(np.mean([s["psc_count"] for s in per_seed])), 2),
        "mean_ndg": round(float(np.mean([s["ndg_count"] for s in per_seed])), 2),
        "mean_busta": round(float(np.mean([s["busta_count"] for s in per_seed])), 2),
        "mean_setup_events": round(float(np.mean([s["setup_events"] for s in per_seed])), 2),
        "mean_restock_count": round(float(np.mean([s["restock_count"] for s in per_seed])), 2),
        "mean_idle_min": round(float(np.mean([s["idle_min"] for s in per_seed])), 2),
        "mean_stockout_duration_l1": round(float(np.mean([s["stockout_duration_l1"] for s in per_seed])), 2),
        "mean_stockout_duration_l2": round(float(np.mean([s["stockout_duration_l2"] for s in per_seed])), 2),
        "n_perfect_seeds": int(sum(1 for s in per_seed if s["tard_cost"] == 0 and s["stockout_cost"] == 0)),
        "n_tardy_seeds": int(sum(1 for s in per_seed if s["tard_cost"] > 0)),
        "n_stockout_seeds": int(sum(1 for s in per_seed if s["stockout_cost"] > 0)),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = _ROOT / "results" / f"full_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    data = load_data()
    params = data.to_env_params()
    ups_lambda = data.ups_lambda
    ups_mu = data.ups_mu

    seeds = list(range(900_000, 900_100))

    # Model paths (PPO archived to OLDCODE/PPOmask_archive/ on 2026-04-28; Paeng DDQN added v4)
    ql_table_path = _ROOT / "q_learning" / "ql_results" / "31_03_2026_0919_ep1433449_a00500_g09900_Q_learn_nonUPS_overnight_profit296270" / "q_table_Q_learn_nonUPS_overnight.pkl"
    rlhh_ckpt_path = _ROOT / "rl_hh" / "outputs" / "rlhh_cycle3_best.pt"
    paeng_ckpt_path = _ROOT / "paeng_ddqn" / "outputs" / "paeng_best.pt"

    # Pre-load expensive things
    print("Loading models...")
    q_table = load_q_table(str(ql_table_path))
    rlhh_agent = DuelingDDQNAgent()
    rlhh_agent.load_checkpoint(str(rlhh_ckpt_path))
    rlhh_agent.epsilon = 0.0

    paeng_agent = None
    if paeng_ckpt_path.exists():
        from paeng_ddqn.agent import PaengAgent
        paeng_agent = PaengAgent.from_checkpoint(paeng_ckpt_path)
        paeng_agent.epsilon = 0.0
    else:
        print(f"  [skip] paeng_ddqn checkpoint not found at {paeng_ckpt_path} — running 4-method comparison instead of 5.")

    results = {}
    n_methods = 5 if paeng_agent is not None else 4
    idx = 1

    # 1. Dispatching baseline
    print(f"\n[{idx}/{n_methods}] Dispatching heuristic baseline...")
    results["dispatching"] = evaluate_strategy(
        "Dispatching heuristic",
        lambda s: DispatchingHeuristic(params),
        seeds, data, params, ups_lambda, ups_mu,
    )
    idx += 1

    # 2. Q-learning
    print(f"\n[{idx}/{n_methods}] Q-learning best...")
    results["q_learning"] = evaluate_strategy(
        "Q-learning (tabular)",
        lambda s: QStrategy(params, q_table=q_table),
        seeds, data, params, ups_lambda, ups_mu,
    )
    idx += 1

    # 3. Paeng DDQN (v4 — replaces PPO slot)
    if paeng_agent is not None:
        from paeng_ddqn.strategy import PaengStrategy
        print(f"\n[{idx}/{n_methods}] Paeng's Modified DDQN...")
        results["paeng_ddqn"] = evaluate_strategy(
            "Paeng's Modified DDQN",
            lambda s: PaengStrategy(paeng_agent, data, training=False),
            seeds, data, params, ups_lambda, ups_mu,
        )
        idx += 1

    # 4. RL-HH (with promoted tools)
    print(f"\n[{idx}/{n_methods}] RL-HH (Dueling DDQN + improved tools)...")
    results["rl_hh"] = evaluate_strategy(
        "RL-HH (Dueling DDQN)",
        lambda s: RLHHStrategy(rlhh_agent, data, training=False),
        seeds, data, params, ups_lambda, ups_mu,
    )
    idx += 1

    # 5. CP-SAT no-UPS ceiling (one solve, deterministic)
    print(f"\n[{idx}/{n_methods}] CP-SAT no-UPS ceiling (single deterministic solve)...")
    t0 = time.perf_counter()
    cp_result = run_pure_cpsat(time_limit_sec=900, ups_events=None, num_workers=8)
    cp_elapsed = time.perf_counter() - t0
    results["cpsat_ceiling"] = {
        "method": "CP-SAT no-UPS ceiling",
        "n_seeds": 1,
        "wall_sec": round(cp_elapsed, 1),
        "profit_mean": round(float(cp_result["net_profit"]), 2),
        "profit_std": 0.0,
        "profit_min": round(float(cp_result["net_profit"]), 2),
        "profit_max": round(float(cp_result["net_profit"]), 2),
        "profit_median": round(float(cp_result["net_profit"]), 2),
        "mean_revenue": round(float(cp_result.get("revenue", 0)), 2),
        "mean_tard_cost": round(float(cp_result.get("tard_cost", 0)), 2),
        "mean_setup_cost": round(float(cp_result.get("setup_cost", 0)), 2),
        "mean_stockout_cost": round(float(cp_result.get("stockout_cost", 0)), 2),
        "mean_idle_cost": round(float(cp_result.get("idle_cost", 0)), 2),
        "mean_psc": int(cp_result.get("psc_count", 0)),
        "mean_ndg": int(cp_result.get("ndg_count", 0)),
        "mean_busta": int(cp_result.get("busta_count", 0)),
        "mean_setup_events": int(cp_result.get("setup_events", 0)),
        "mean_restock_count": int(cp_result.get("restock_count", 0)),
        "mip_gap_pct": cp_result.get("gap_pct"),
        "best_bound": cp_result.get("best_bound"),
        "solver_status": cp_result.get("status"),
    }
    print(f"  CP-SAT: profit ${results['cpsat_ceiling']['profit_mean']:,.0f}, gap={results['cpsat_ceiling']['mip_gap_pct']}%")

    # Save aggregated result
    summary_path = out_dir / "comparison_100seed.json"
    with open(summary_path, "w") as f:
        # Drop per_seed from json for compactness; save separately
        compact = {}
        per_seed_dump = {}
        for k, v in results.items():
            if "per_seed" in v:
                per_seed_dump[k] = v["per_seed"]
                compact[k] = {kk: vv for kk, vv in v.items() if kk != "per_seed"}
            else:
                compact[k] = v
        json.dump(compact, f, indent=2)
    with open(out_dir / "per_seed.json", "w") as f:
        json.dump(per_seed_dump, f, indent=2)

    print(f"\nResults saved to {summary_path}")
    print(f"\n{'=' * 70}")
    print(f"{'Method':<35} {'Mean':>12} {'Median':>12} {'Std':>12}")
    print(f"{'=' * 70}")
    for k, v in results.items():
        print(f"{v['method']:<35} ${v['profit_mean']:>11,.0f} ${v['profit_median']:>11,.0f} ${v['profit_std']:>11,.0f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
