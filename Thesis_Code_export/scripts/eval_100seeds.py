"""Unified 100-seed evaluation harness for all supported methods.

Resolves the strategy via --method: rl_hh / q_learning / dispatching /
paeng_ddqn_v2. UPS lambda and mu are read from Input_data and scaled by
``--lambda-mult`` / ``--mu-mult`` so the same script drives all 9 cells of
the (3 lambda x 3 mu) Block-B factorial.

Output convention: Results/<YYYYMMDD_HHMMSS>_100SeedEval_<RunName>/aggregate.json

Usage:
    python scripts/eval_100seeds.py --method dispatching --reps 100 --name DispBaseline
    python scripts/eval_100seeds.py --method rl_hh --checkpoint <pt> --reps 100 --name RLHH_C1
    python scripts/eval_100seeds.py --method q_learning --checkpoint <pkl> --reps 50 --name QL_smoke
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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from env.data_loader import load_data
from evaluation.result_schema import make_run_dir


TOOL_NAMES = ["PSC_THROUGHPUT", "GC_RESTOCK", "MTO_DEADLINE", "SETUP_AVOID", "WAIT"]
SUPPORTED_METHODS = ("rl_hh", "q_learning", "dispatching", "paeng_ddqn_v2")


# ---------------------------------------------------------------------------
# Per-method strategy factory
# ---------------------------------------------------------------------------

def _build_factory(method: str, data, params, checkpoint_path: str | Path | None):
    """Return (factory, tool_count_extractor). factory(seed) -> strategy instance."""
    if method == "rl_hh":
        train_mod = importlib.import_module("rl_hh.train")
        agent = train_mod.DuelingDDQNAgent()
        agent.load_checkpoint(checkpoint_path)
        agent.epsilon = 0.0

        def factory(_seed: int):
            return train_mod.RLHHStrategy(agent, data, training=False)

        def extract_tool_counts(strategy) -> list[int]:
            counts = [0] * 5
            for tid, cnt in strategy.tool_counts.items():
                if tid < 5:
                    counts[tid] += cnt
            return counts

        return factory, extract_tool_counts

    if method == "q_learning":
        train_mod = importlib.import_module("q_learning.train")
        q_table = train_mod.load_q_table(checkpoint_path)

        def factory(_seed: int):
            return train_mod.QStrategy(params, q_table=q_table)

        def extract_tool_counts(_strategy) -> list[int]:
            return [0] * 5

        return factory, extract_tool_counts

    if method == "dispatching":
        from dispatch.dispatching_heuristic import DispatchingHeuristic

        def factory(_seed: int):
            return DispatchingHeuristic(params)

        def extract_tool_counts(_strategy) -> list[int]:
            return [0] * 5

        return factory, extract_tool_counts

    if method == "paeng_ddqn_v2":
        train_mod = importlib.import_module("paeng_ddqn_v2.train")
        agent = train_mod.PaengAgentV2.from_checkpoint(checkpoint_path)
        agent.epsilon = 0.0

        def factory(_seed: int):
            return train_mod.PaengStrategyV2(agent, data, training=False, params=params)

        def extract_tool_counts(_strategy) -> list[int]:
            return [0] * 5  # paeng_v2 has no tool layer

        return factory, extract_tool_counts

    raise ValueError(f"Unknown method: {method}; supported: {SUPPORTED_METHODS}")


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
    """Evaluate a method across a fixed seed set; return aggregate dict."""
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
        # Receipt for paired-seed verification across cells/methods.
        ups_repr = repr(tuple((int(e.t), str(e.roaster_id), int(e.duration)) for e in ups)).encode()
        ups_hash = hashlib.sha256(ups_repr).hexdigest()[:16]
        kpi, _state = engine.run(strategy, ups)

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified 100-seed evaluator.")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Required for rl_hh / q_learning / paeng_ddqn_v2; ignored for dispatching.")
    parser.add_argument("--name", required=True, help="Run name (output folder suffix).")
    parser.add_argument("--reps", type=int, default=100, help="Number of seeds (default 100).")
    parser.add_argument("--base-seed", type=int, default=900_000)
    parser.add_argument("--lambda-mult", type=float, default=1.0)
    parser.add_argument("--mu-mult", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.method != "dispatching" and args.checkpoint is None:
        parser.error(f"--checkpoint is required for --method {args.method}")

    seeds = list(range(args.base_seed, args.base_seed + args.reps))
    out_dir = make_run_dir("100SeedEval", f"{args.name}_{args.method}")
    print(f"[100seed] Output -> {out_dir}")
    print(f"[100seed] Method={args.method}  seeds={args.base_seed}..{args.base_seed + args.reps - 1}")
    if args.checkpoint:
        print(f"[100seed] Checkpoint: {args.checkpoint}")

    result = evaluate_100_seeds(
        package=args.method,
        seeds=seeds,
        checkpoint_path=args.checkpoint,
        lambda_mult=args.lambda_mult,
        mu_mult=args.mu_mult,
    )

    out_path = out_dir / "aggregate.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"  Method:     {result['package']}")
    print(f"  Seeds:      {result['n_seeds']}")
    print(f"  Profit:     mean=${result['profit_mean']:,.0f}  std=${result['profit_std']:,.0f}")
    print(f"              median=${result['profit_median']:,.0f}")
    print(f"  Wall:       {result['eval_wall_sec']}s")
    print(f"{'=' * 60}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
