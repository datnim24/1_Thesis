"""Evaluate the PPO checkpoint on the same 100 seeds RLHH was evaluated on.

Uses the canonical SimulationEngine + PPOStrategy path, mirroring how
test_rl_hh.evaluate_100seeds runs RL-HH. This makes the comparison
apples-to-apples.

Usage:
    python -m test_rl_hh.evaluate_ppo_100seeds \
        --model-path PPOmask/outputs/20260424_004458_seed69_pipeline_s427008_v2/checkpoints/final_model.zip \
        --output test_rl_hh/outputs/ppo_100seed.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data
from PPOmask.Engine.ppo_strategy import PPOStrategy, resolve_model_path
from sb3_contrib import MaskablePPO


def evaluate_ppo_100_seeds(model_path: str, seeds: list[int]) -> dict:
    data = load_data()
    params = data.to_env_params()
    engine = SimulationEngine(params)
    shift_length = int(params["SL"])
    roasters = list(params["roasters"])
    ups_lambda = data.ups_lambda
    ups_mu = data.ups_mu

    resolved = resolve_model_path(model_path)
    print(f"Loading PPO model: {resolved}")
    model = MaskablePPO.load(str(resolved))

    profits: list[float] = []
    per_seed: list[dict] = []
    t0 = time.perf_counter()

    for seed in seeds:
        strategy = PPOStrategy(data=data, model=model, deterministic=True)
        ups = generate_ups_events(ups_lambda, ups_mu, seed, shift_length, roasters)
        kpi, state = engine.run(strategy, ups)
        p = float(kpi.net_profit())
        k = kpi.to_dict()
        profits.append(p)
        per_seed.append({
            "seed": seed,
            "net_profit": round(p, 2),
            "psc": int(k["psc_count"]),
            "ndg": int(k.get("ndg_count", 0)),
            "busta": int(k.get("busta_count", 0)),
            "tard_cost": round(float(k["tard_cost"]), 2),
            "setup_events": int(k["setup_events"]),
            "setup_cost": round(float(k["setup_cost"]), 2),
            "stockout_cost": round(float(k["stockout_cost"]), 2),
            "idle_min": int(k["idle_min"]),
            "idle_cost": round(float(k["idle_cost"]), 2),
            "restock_count": int(k["restock_count"]),
        })

    elapsed = time.perf_counter() - t0
    arr = np.array(profits)
    return {
        "model": str(resolved),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "profit_mean": round(float(arr.mean()), 2),
        "profit_std": round(float(arr.std()), 2),
        "profit_min": round(float(arr.min()), 2),
        "profit_max": round(float(arr.max()), 2),
        "profit_median": round(float(np.median(arr)), 2),
        "mean_psc": round(float(np.mean([s["psc"] for s in per_seed])), 2),
        "mean_setup_events": round(float(np.mean([s["setup_events"] for s in per_seed])), 2),
        "mean_setup_cost": round(float(np.mean([s["setup_cost"] for s in per_seed])), 2),
        "mean_idle_min": round(float(np.mean([s["idle_min"] for s in per_seed])), 2),
        "mean_idle_cost": round(float(np.mean([s["idle_cost"] for s in per_seed])), 2),
        "mean_tard_cost": round(float(np.mean([s["tard_cost"] for s in per_seed])), 2),
        "mean_stockout_cost": round(float(np.mean([s["stockout_cost"] for s in per_seed])), 2),
        "mean_restock_count": round(float(np.mean([s["restock_count"] for s in per_seed])), 2),
        "eval_wall_sec": round(elapsed, 1),
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser(description="100-seed eval for PPO checkpoint")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-seed", type=int, default=900_000)
    parser.add_argument("--n-seeds", type=int, default=100)
    args = parser.parse_args()

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    result = evaluate_ppo_100_seeds(args.model_path, seeds)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  PPO model: {result['model']}")
    print(f"  Seeds:     {result['n_seeds']}")
    print(f"  Profit:    mean=${result['profit_mean']:,.0f}  std=${result['profit_std']:,.0f}")
    print(f"             min=${result['profit_min']:,.0f}  max=${result['profit_max']:,.0f}")
    print(f"             median=${result['profit_median']:,.0f}")
    print(f"  Mean idle: {result['mean_idle_min']:.1f} min (${result['mean_idle_cost']:,.0f})")
    print(f"  Mean setup: {result['mean_setup_events']:.1f} events (${result['mean_setup_cost']:,.0f})")
    print(f"  Mean restocks: {result['mean_restock_count']:.1f}")
    print(f"  Mean PSC:  {result['mean_psc']:.1f}")
    print(f"  Mean tard cost: ${result['mean_tard_cost']:,.0f}")
    print(f"  Mean stockout cost: ${result['mean_stockout_cost']:,.0f}")
    print(f"  Wall: {result['eval_wall_sec']}s")
    print(f"{'=' * 60}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
