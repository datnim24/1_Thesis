"""100-seed evaluation harness for RL-HH / test_rl_hh comparison.

Runs a checkpoint against a fixed set of 100 seeds using the SAME tool module
the checkpoint was trained with. Resolves the package by --package flag so that
rl_hh/outputs/rlhh_cycle3_best.pt is evaluated with rl_hh.tools, and
test_rl_hh/outputs/*.pt is evaluated with test_rl_hh.tools.

Usage:
    python -m test_rl_hh.evaluate_100seeds \
        --checkpoint rl_hh/outputs/rlhh_cycle3_best.pt \
        --package rl_hh \
        --output test_rl_hh/outputs/baseline_cycle3_100seed.json
"""

from __future__ import annotations

import argparse
import importlib
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


TOOL_NAMES = ["PSC_THROUGHPUT", "GC_RESTOCK", "MTO_DEADLINE", "SETUP_AVOID", "WAIT"]


def evaluate_100_seeds(
    checkpoint_path: str | Path,
    package: str,
    seeds: list[int],
    ups_lambda: float | None = None,
    ups_mu: float | None = None,
) -> dict:
    """Evaluate a checkpoint across a fixed seed set using SimulationEngine.run().

    package: "rl_hh" or "test_rl_hh" — selects which tools/network to use.
    """
    meta_agent_mod = importlib.import_module(f"{package}.meta_agent")
    strategy_mod = importlib.import_module(f"{package}.rl_hh_strategy")

    data = load_data()
    params = data.to_env_params()
    if ups_lambda is None:
        ups_lambda = data.ups_lambda
    if ups_mu is None:
        ups_mu = data.ups_mu
    engine = SimulationEngine(params)
    shift_length = int(params["SL"])
    roasters = list(params["roasters"])

    agent = meta_agent_mod.DuelingDDQNAgent()
    agent.load_checkpoint(checkpoint_path)
    agent.epsilon = 0.0

    profits: list[float] = []
    per_seed: list[dict] = []
    tool_counts_total = [0] * 5

    t0 = time.perf_counter()
    for seed in seeds:
        strategy = strategy_mod.RLHHStrategy(agent, data, training=False)
        ups = generate_ups_events(ups_lambda, ups_mu, seed, shift_length, roasters)
        kpi, state = engine.run(strategy, ups)

        p = float(kpi.net_profit())
        k = kpi.to_dict()
        profits.append(p)
        per_seed.append({
            "seed": seed,
            "net_profit": round(p, 2),
            "revenue": round(float(k["total_revenue"]), 2),
            "tard_cost": round(float(k["tard_cost"]), 2),
            "setup_cost": round(float(k["setup_cost"]), 2),
            "stockout_cost": round(float(k["stockout_cost"]), 2),
            "idle_cost": round(float(k["idle_cost"]), 2),
            "psc_count": int(k["psc_count"]),
            "ndg_count": int(k.get("ndg_count", 0)),
            "busta_count": int(k.get("busta_count", 0)),
            "setup_events": int(k["setup_events"]),
            "restock_count": int(k["restock_count"]),
            "idle_min": int(k["idle_min"]),
        })
        for tid, cnt in strategy.tool_counts.items():
            if tid < 5:
                tool_counts_total[tid] += cnt

    elapsed = time.perf_counter() - t0

    arr = np.array(profits)
    total_tools = max(1, sum(tool_counts_total))
    tool_dist = {
        TOOL_NAMES[i]: round(tool_counts_total[i] / total_tools, 4)
        for i in range(5)
    }

    return {
        "checkpoint": str(checkpoint_path),
        "package": package,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "profit_mean": round(float(arr.mean()), 2),
        "profit_std": round(float(arr.std()), 2),
        "profit_min": round(float(arr.min()), 2),
        "profit_max": round(float(arr.max()), 2),
        "profit_median": round(float(np.median(arr)), 2),
        "profit_p25": round(float(np.percentile(arr, 25)), 2),
        "profit_p75": round(float(np.percentile(arr, 75)), 2),
        "mean_tard_cost": round(float(np.mean([s["tard_cost"] for s in per_seed])), 2),
        "mean_setup_cost": round(float(np.mean([s["setup_cost"] for s in per_seed])), 2),
        "mean_stockout_cost": round(float(np.mean([s["stockout_cost"] for s in per_seed])), 2),
        "mean_idle_cost": round(float(np.mean([s["idle_cost"] for s in per_seed])), 2),
        "mean_psc": round(float(np.mean([s["psc_count"] for s in per_seed])), 2),
        "mean_setup_events": round(float(np.mean([s["setup_events"] for s in per_seed])), 2),
        "mean_restock_count": round(float(np.mean([s["restock_count"] for s in per_seed])), 2),
        "mean_idle_min": round(float(np.mean([s["idle_min"] for s in per_seed])), 2),
        "tool_distribution": tool_dist,
        "eval_wall_sec": round(elapsed, 1),
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser(description="100-seed eval for RL-HH checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--package", choices=["rl_hh", "test_rl_hh"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-seed", type=int, default=900_000)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--ups-lambda", type=float, default=None)
    parser.add_argument("--ups-mu", type=float, default=None)
    args = parser.parse_args()

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    print(f"Evaluating {args.checkpoint} (package={args.package}) on {len(seeds)} seeds...")
    result = evaluate_100_seeds(
        args.checkpoint, args.package, seeds,
        ups_lambda=args.ups_lambda, ups_mu=args.ups_mu,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Checkpoint: {result['checkpoint']}")
    print(f"  Package:    {result['package']}")
    print(f"  Seeds:      {result['n_seeds']}")
    print(f"  Profit:     mean=${result['profit_mean']:,.0f}  std=${result['profit_std']:,.0f}")
    print(f"              min=${result['profit_min']:,.0f}  max=${result['profit_max']:,.0f}")
    print(f"              median=${result['profit_median']:,.0f}")
    print(f"  Mean idle:       {result['mean_idle_min']:.1f} min (${result['mean_idle_cost']:,.0f})")
    print(f"  Mean setup:      {result['mean_setup_events']:.1f} events (${result['mean_setup_cost']:,.0f})")
    print(f"  Mean restocks:   {result['mean_restock_count']:.1f}")
    print(f"  Mean PSC:        {result['mean_psc']:.1f}")
    print(f"  Mean tard cost:  ${result['mean_tard_cost']:,.0f}")
    print(f"  Tool dist:       {result['tool_distribution']}")
    print(f"  Wall: {result['eval_wall_sec']}s")
    print(f"{'=' * 60}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
