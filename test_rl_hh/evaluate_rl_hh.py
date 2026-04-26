"""Evaluation script for trained RL-HH Dueling DDQN.

Supports two modes:
  1. Gymnasium env evaluation (consistent with training)
  2. SimulationEngine evaluation (consistent with Q-learning / dispatching baselines)

Outputs: result-schema-compatible JSON with profit stats, KPI breakdown,
and tool selection distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from PPOmask.Engine.action_spec import WAIT_ACTION_ID
from PPOmask.Engine.data_loader import load_data
from PPOmask.Engine.roasting_env import RoastingMaskEnv

from . import configs as C
from .meta_agent import DuelingDDQNAgent
from .rl_hh_strategy import RLHHStrategy
from .tools import ToolKit


# ------------------------------------------------------------------
# Gymnasium-based evaluation (consistent with training)
# ------------------------------------------------------------------

def evaluate_gym(
    agent: DuelingDDQNAgent,
    data,
    n_episodes: int = 100,
    ups_lambda: float | None = None,
    ups_mu: float | None = None,
    base_seed: int = 900_000,
) -> dict:
    """Run greedy evaluation using the Gym env (same as training)."""
    if ups_lambda is None:
        ups_lambda = data.ups_lambda
    if ups_mu is None:
        ups_mu = data.ups_mu
    env = RoastingMaskEnv(data, ups_lambda=ups_lambda, ups_mu=ups_mu)
    toolkit = ToolKit(env.engine, data.to_env_params())

    profits: list[float] = []
    kpis: list[dict] = []
    tool_counts_total: dict[int, int] = defaultdict(int)
    violations = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        done = env._terminated

        while not done:
            context = env.current_frame.context
            roaster_id = context.roaster_id
            tool_outputs, tool_mask = toolkit.compute_all(env.state, roaster_id)

            tool_id = agent.select_tool(obs, tool_mask, training=False)
            tool_counts_total[tool_id] += 1

            action_id = tool_outputs[tool_id]
            if action_id is None:
                action_id = WAIT_ACTION_ID

            obs, reward, terminated, truncated, step_info = env.step(action_id)
            done = terminated or truncated

        net_profit = float(env.kpi.net_profit())
        profits.append(net_profit)
        kpis.append(env.kpi.to_dict())
        if env._violated:
            violations += 1

    return _compile_results(profits, kpis, tool_counts_total, violations, n_episodes)


# ------------------------------------------------------------------
# SimulationEngine-based evaluation (comparable to baselines)
# ------------------------------------------------------------------

def evaluate_engine(
    agent: DuelingDDQNAgent,
    data,
    n_episodes: int = 100,
    ups_lambda: float | None = None,
    ups_mu: float | None = None,
    base_seed: int = 900_000,
) -> dict:
    """Run greedy evaluation using SimulationEngine.run() + RLHHStrategy."""
    if ups_lambda is None:
        ups_lambda = data.ups_lambda
    if ups_mu is None:
        ups_mu = data.ups_mu
    params = data.to_env_params() if hasattr(data, "to_env_params") else data
    engine = SimulationEngine(params)
    shift_length = int(params["SL"])
    roasters = list(params["roasters"])

    profits: list[float] = []
    kpis: list[dict] = []
    tool_counts_total: dict[int, int] = defaultdict(int)

    for ep in range(n_episodes):
        strategy = RLHHStrategy(agent, data, training=False)
        ups = generate_ups_events(ups_lambda, ups_mu, base_seed + ep, shift_length, roasters)
        kpi, state = engine.run(strategy, ups)

        profits.append(float(kpi.net_profit()))
        kpis.append(kpi.to_dict())
        for tid, cnt in strategy.tool_counts.items():
            tool_counts_total[tid] += cnt

    return _compile_results(profits, kpis, tool_counts_total, 0, n_episodes)


# ------------------------------------------------------------------
# Result compilation
# ------------------------------------------------------------------

def _compile_results(
    profits: list[float],
    kpis: list[dict],
    tool_counts: dict[int, int],
    violations: int,
    n_episodes: int,
) -> dict:
    total_tool_selections = max(1, sum(tool_counts.values()))
    tool_dist = {
        C.TOOL_NAMES[tid]: round(cnt / total_tool_selections, 4)
        for tid, cnt in sorted(tool_counts.items())
        if tid < len(C.TOOL_NAMES)
    }

    return {
        "method": "RL-HH (Dueling DDQN)",
        "n_episodes": n_episodes,
        "profit_mean": round(float(np.mean(profits)), 2),
        "profit_std": round(float(np.std(profits)), 2),
        "profit_min": round(float(np.min(profits)), 2),
        "profit_max": round(float(np.max(profits)), 2),
        "violations": violations,
        "psc_count_mean": round(float(np.mean([k["psc_count"] for k in kpis])), 1),
        "ndg_count_mean": round(float(np.mean([k.get("ndg_count", 0) for k in kpis])), 1),
        "busta_count_mean": round(float(np.mean([k.get("busta_count", 0) for k in kpis])), 1),
        "stockout_cost_mean": round(float(np.mean([k["stockout_cost"] for k in kpis])), 2),
        "setup_cost_mean": round(float(np.mean([k["setup_cost"] for k in kpis])), 2),
        "tard_cost_mean": round(float(np.mean([k["tard_cost"] for k in kpis])), 2),
        "idle_cost_mean": round(float(np.mean([k["idle_cost"] for k in kpis])), 2),
        "restock_count_mean": round(float(np.mean([k["restock_count"] for k in kpis])), 1),
        "tool_distribution": tool_dist,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL-HH")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--ups-lambda", type=float, default=None, help="UPS lambda (default: from input data)")
    parser.add_argument("--ups-mu", type=float, default=None, help="UPS mu (default: from input data)")
    parser.add_argument("--mode", choices=["gym", "engine"], default="gym")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--base-seed", type=int, default=900_000)
    args = parser.parse_args()

    data = load_data(args.input_dir)
    agent = DuelingDDQNAgent()
    agent.load_checkpoint(args.checkpoint)
    agent.epsilon = 0.0  # greedy

    print(f"Evaluating {args.checkpoint} ({args.mode} mode, {args.n_episodes} episodes)...")
    t0 = time.perf_counter()

    if args.mode == "gym":
        results = evaluate_gym(
            agent, data, args.n_episodes,
            args.ups_lambda, args.ups_mu, args.base_seed,
        )
    else:
        results = evaluate_engine(
            agent, data, args.n_episodes,
            args.ups_lambda, args.ups_mu, args.base_seed,
        )

    elapsed = time.perf_counter() - t0
    results["eval_wall_sec"] = round(elapsed, 1)

    print(f"\n{'='*60}")
    print(f"  Method:           {results['method']}")
    print(f"  Episodes:         {results['n_episodes']}")
    print(f"  Profit:           {results['profit_mean']:.1f} +/- {results['profit_std']:.1f}")
    print(f"  PSC count:        {results['psc_count_mean']:.1f}")
    print(f"  Stockout cost:    {results['stockout_cost_mean']:.1f}")
    print(f"  Setup cost:       {results['setup_cost_mean']:.1f}")
    print(f"  Tard cost:        {results['tard_cost_mean']:.1f}")
    print(f"  Restock count:    {results['restock_count_mean']:.1f}")
    print(f"  Violations:       {results['violations']}")
    print(f"  Tool distribution:")
    for tool_name, frac in results["tool_distribution"].items():
        print(f"    {tool_name:20s} {frac*100:5.1f}%")
    print(f"  Wall time:        {elapsed:.1f}s")
    print(f"{'='*60}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
