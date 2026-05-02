"""100-seed evaluation harness for the v4 method-comparison set.

Single entry-point used by Block B and by ad-hoc method evaluation. Resolves the
strategy via --package: rl_hh / test_rl_hh / q_learning / dispatching / paeng_ddqn.

UPS λ and μ are read from Input_data and scaled by --lambda-mult / --mu-mult so
the same script drives all 9 cells of the (3 λ × 3 μ) factorial.

Usage examples:

    # RL-HH (existing)
    python -m test_rl_hh.evaluate_100seeds \
        --checkpoint rl_hh/outputs/rlhh_cycle3_best.pt \
        --package rl_hh \
        --output rl_hh/outputs/rlhh_100seed.json

    # Q-learning checkpoint (new)
    python -m test_rl_hh.evaluate_100seeds \
        --checkpoint q_learning/ql_results/30_03_2026_1230_*/q_table.pkl \
        --package q_learning \
        --output q_learning/ql_results/30_03_2026_1230_*/ql_100seed.json

    # Dispatching baseline (no checkpoint needed)
    python -m test_rl_hh.evaluate_100seeds \
        --package dispatching \
        --output results/dispatching_100seed.json

    # Block B cell (λ × 0.5, μ × 1.0)
    python -m test_rl_hh.evaluate_100seeds \
        --checkpoint rl_hh/outputs/rlhh_cycle3_best.pt \
        --package rl_hh \
        --lambda-mult 0.5 --mu-mult 1.0 \
        --output results/block_b/rl_hh_lm0.5_mm1.0.json
"""

from __future__ import annotations

import argparse
import hashlib
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
SUPPORTED_PACKAGES = ("rl_hh", "test_rl_hh", "q_learning", "dispatching", "paeng_ddqn", "paeng_ddqn_v2")


# ---------------------------------------------------------------------------
# Per-package factory: takes (data, params, checkpoint_path) → strategy factory
# (the factory is called once per seed, returns a fresh strategy instance)
# ---------------------------------------------------------------------------

def _build_factory(package: str, data, params, checkpoint_path: str | Path | None):
    """Return (factory, tool_count_extractor). factory(seed) → strategy instance."""
    if package in ("rl_hh", "test_rl_hh"):
        meta_agent_mod = importlib.import_module(f"{package}.meta_agent")
        strategy_mod = importlib.import_module(f"{package}.rl_hh_strategy")
        agent = meta_agent_mod.DuelingDDQNAgent()
        agent.load_checkpoint(checkpoint_path)
        agent.epsilon = 0.0

        def factory(_seed: int):
            return strategy_mod.RLHHStrategy(agent, data, training=False)

        def extract_tool_counts(strategy) -> list[int]:
            counts = [0] * 5
            for tid, cnt in strategy.tool_counts.items():
                if tid < 5:
                    counts[tid] += cnt
            return counts

        return factory, extract_tool_counts

    if package == "q_learning":
        ql_mod = importlib.import_module("q_learning.q_strategy")
        q_table = ql_mod.load_q_table(checkpoint_path)

        def factory(_seed: int):
            return ql_mod.QStrategy(params, q_table=q_table)

        def extract_tool_counts(_strategy) -> list[int]:
            return [0] * 5  # Q-learning has no tool layer

        return factory, extract_tool_counts

    if package == "dispatching":
        from dispatch.dispatching_heuristic import DispatchingHeuristic

        def factory(_seed: int):
            return DispatchingHeuristic(params)

        def extract_tool_counts(_strategy) -> list[int]:
            return [0] * 5

        return factory, extract_tool_counts

    if package == "paeng_ddqn":
        # Will be available after Phase 5 build (paeng_ddqn/ task 5-9).
        paeng_strategy_mod = importlib.import_module("paeng_ddqn.strategy")
        paeng_agent_mod = importlib.import_module("paeng_ddqn.agent")
        agent = paeng_agent_mod.PaengAgent.from_checkpoint(checkpoint_path)

        def factory(_seed: int):
            return paeng_strategy_mod.PaengStrategy(agent, data, training=False)

        def extract_tool_counts(_strategy) -> list[int]:
            return [0] * 5  # Paeng has no tool layer either

        return factory, extract_tool_counts

    if package == "paeng_ddqn_v2":
        # Faithful rebuild of Paeng 2021 paper.
        v2_strategy_mod = importlib.import_module("paeng_ddqn_v2.strategy_v2")
        v2_agent_mod = importlib.import_module("paeng_ddqn_v2.agent_v2")
        agent = v2_agent_mod.PaengAgentV2.from_checkpoint(checkpoint_path)
        agent.epsilon = 0.0

        def factory(_seed: int):
            return v2_strategy_mod.PaengStrategyV2(agent, data, training=False, params=params)

        def extract_tool_counts(_strategy) -> list[int]:
            return [0] * 5  # Paeng v2 has no tool layer

        return factory, extract_tool_counts

    raise ValueError(f"Unknown package: {package}; supported: {SUPPORTED_PACKAGES}")


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def evaluate_100_seeds(
    package: str,
    seeds: list[int],
    checkpoint_path: str | Path | None = None,
    lambda_mult: float = 1.0,
    mu_mult: float = 1.0,
) -> dict:
    """Evaluate any supported method across a fixed seed set.

    UPS λ/μ are read from Input_data and scaled by the multipliers.
    """
    data = load_data()
    params = data.to_env_params()
    ups_lambda_input = float(data.ups_lambda)
    ups_mu_input = float(data.ups_mu)
    ups_lambda = ups_lambda_input * lambda_mult
    ups_mu = ups_mu_input * mu_mult

    engine = SimulationEngine(params)
    shift_length = int(params["SL"])
    roasters = list(params["roasters"])

    factory, extract_tool_counts = _build_factory(package, data, params, checkpoint_path)

    profits: list[float] = []
    per_seed: list[dict] = []
    tool_counts_total = [0] * 5

    t0 = time.perf_counter()
    for seed in seeds:
        strategy = factory(seed)
        ups = generate_ups_events(ups_lambda, ups_mu, seed, shift_length, roasters)
        # Block-B paired-seed receipt: hash the UPS event tuples so the
        # aggregator can assert per-(cell, seed) identity across methods.
        ups_repr = repr(tuple((int(e.t), str(e.roaster_id), int(e.duration)) for e in ups)).encode()
        ups_hash = hashlib.sha256(ups_repr).hexdigest()[:16]
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
            "ups_event_hash": ups_hash,
            "n_ups_events": len(ups),
        })
        for tid, cnt in enumerate(extract_tool_counts(strategy)):
            tool_counts_total[tid] += cnt

    elapsed = time.perf_counter() - t0
    arr = np.array(profits)
    total_tools = max(1, sum(tool_counts_total))
    tool_dist = {
        TOOL_NAMES[i]: round(tool_counts_total[i] / total_tools, 4)
        for i in range(5)
    }

    return {
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "package": package,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "ups_lambda_input_data": round(ups_lambda_input, 4),
        "ups_mu_input_data":     round(ups_mu_input, 4),
        "lambda_mult":           round(lambda_mult, 4),
        "mu_mult":               round(mu_mult, 4),
        "ups_lambda_used":       round(ups_lambda, 4),
        "ups_mu_used":           round(ups_mu, 4),
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
    parser = argparse.ArgumentParser(description="100-seed eval for any supported package")
    parser.add_argument("--checkpoint", default=None,
                        help="Required for rl_hh / test_rl_hh / q_learning / paeng_ddqn; ignored for dispatching")
    parser.add_argument("--package", choices=SUPPORTED_PACKAGES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-seed", type=int, default=900_000)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--lambda-mult", type=float, default=1.0,
                        help="UPS λ multiplier; Block B values: 0.5, 1.0, 2.0")
    parser.add_argument("--mu-mult", type=float, default=1.0,
                        help="UPS μ multiplier; Block B values: 0.5, 1.0, 2.0")
    # Backward-compat aliases (preserved so old scripts still work)
    parser.add_argument("--ups-lambda", type=float, default=None,
                        help="DEPRECATED: prefer --lambda-mult. If set, OVERRIDES Input_data λ entirely.")
    parser.add_argument("--ups-mu", type=float, default=None,
                        help="DEPRECATED: prefer --mu-mult. If set, OVERRIDES Input_data μ entirely.")
    args = parser.parse_args()

    if args.package != "dispatching" and args.checkpoint is None:
        parser.error(f"--checkpoint is required for --package {args.package}")

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    # Backward compat: if old --ups-lambda/--ups-mu are passed, convert to multipliers
    if args.ups_lambda is not None or args.ups_mu is not None:
        data = load_data()
        if args.ups_lambda is not None:
            args.lambda_mult = float(args.ups_lambda) / float(data.ups_lambda)
        if args.ups_mu is not None:
            args.mu_mult = float(args.ups_mu) / float(data.ups_mu)

    print(f"Evaluating package={args.package}")
    if args.checkpoint:
        print(f"  checkpoint: {args.checkpoint}")
    print(f"  seeds:      {args.base_seed}..{args.base_seed + args.n_seeds - 1} ({args.n_seeds} total)")
    print(f"  multipliers: lambda x {args.lambda_mult}, mu x {args.mu_mult}")

    result = evaluate_100_seeds(
        package=args.package,
        seeds=seeds,
        checkpoint_path=args.checkpoint,
        lambda_mult=args.lambda_mult,
        mu_mult=args.mu_mult,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Package:    {result['package']}")
    if result.get('checkpoint'):
        print(f"  Checkpoint: {result['checkpoint']}")
    print(f"  Seeds:      {result['n_seeds']}")
    print(f"  UPS:        lambda={result['ups_lambda_used']:.2f} (x{result['lambda_mult']}), mu={result['ups_mu_used']:.2f} (x{result['mu_mult']})")
    print(f"  Profit:     mean=${result['profit_mean']:,.0f}  std=${result['profit_std']:,.0f}")
    print(f"              min=${result['profit_min']:,.0f}  max=${result['profit_max']:,.0f}")
    print(f"              median=${result['profit_median']:,.0f}")
    print(f"  Mean idle:       {result['mean_idle_min']:.1f} min (${result['mean_idle_cost']:,.0f})")
    print(f"  Mean setup:      {result['mean_setup_events']:.1f} events (${result['mean_setup_cost']:,.0f})")
    print(f"  Mean restocks:   {result['mean_restock_count']:.1f}")
    print(f"  Mean PSC:        {result['mean_psc']:.1f}")
    print(f"  Mean tard cost:  ${result['mean_tard_cost']:,.0f}")
    if result['package'] in ("rl_hh", "test_rl_hh"):
        print(f"  Tool dist:       {result['tool_distribution']}")
    print(f"  Wall: {result['eval_wall_sec']}s")
    print(f"{'=' * 60}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
