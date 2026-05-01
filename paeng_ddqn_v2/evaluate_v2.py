"""Paeng DDQN v2 evaluator — single-seed reports and 100-seed aggregates.

Usage:
    # Single-seed report with HTML dashboard
    python -m paeng_ddqn_v2.evaluate_v2 --checkpoint ... --seed 42 --report --output result.json

    # 100-seed aggregate eval
    python -m paeng_ddqn_v2.evaluate_v2 --checkpoint ... --n-seeds 100 --base-seed 900000 \
        --lambda-mult 1.0 --mu-mult 1.0 --output 100seed.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data
from result_schema import create_result
from q_learning.q_learning_run import _batch_to_result_entry, _match_cancel_time

from paeng_ddqn_v2.agent_v2 import PaengAgentV2
from paeng_ddqn_v2.strategy_v2 import PaengStrategyV2


def _json_safe(obj):
    """Convert tuple keys (e.g. gc_capacity[(line, sku)]) to underscore strings for JSON."""
    if isinstance(obj, dict):
        return {
            ("_".join(str(p) for p in k) if isinstance(k, tuple)
             else str(k) if not isinstance(k, (str, int, float, bool)) and k is not None
             else k): _json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        try:
            return sorted(obj)
        except TypeError:
            return [_json_safe(x) for x in obj]
    return obj


def _build_metadata(checkpoint_path: Path) -> dict:
    """Mirror v1 paeng_ddqn metadata builder for plot_result compatibility."""
    metadata = {
        "solver_engine": "paeng_ddqn_v2",
        "solver_name": "Paeng DDQN v2 (3,35 state, faithful to 2021 paper + domain extension)",
        "status": "Completed",
        "input_dir": "Input_data",
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_path": str(checkpoint_path),
    }
    summary_path = Path(checkpoint_path).parent / "training_summary.json"
    if summary_path.exists():
        try:
            tsum = json.loads(summary_path.read_text(encoding="utf-8"))
            metadata["notes"] = (
                f"Paeng DDQN v2: {tsum.get('episodes', '?')} episodes, "
                f"best train profit ${tsum.get('best_profit', '?')}, "
                f"final epsilon {tsum.get('final_epsilon', '?')}"
            )
            metadata["training_run_dir"] = str(Path(checkpoint_path).parent)
        except Exception:
            pass
    return metadata


def evaluate_one_seed(
    checkpoint_path: Path,
    seed: int,
    lambda_mult: float = 1.0,
    mu_mult: float = 1.0,
) -> tuple[dict, dict, object]:
    """Run single-seed evaluation; return (universal-schema result dict, kpi_dict, sim_state)."""
    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    base_lam = float(data.ups_lambda)
    base_mu = float(data.ups_mu)
    lam = base_lam * lambda_mult
    mu = base_mu * mu_mult

    agent = PaengAgentV2.from_checkpoint(checkpoint_path)
    agent.epsilon = 0.0  # greedy for evaluation
    strategy = PaengStrategyV2(agent, data, training=False, params=params)

    engine = SimulationEngine(params)
    ups_events = generate_ups_events(lam, mu, seed, SL, roasters)
    kpi, state = engine.run(strategy, ups_events)

    used_events: set[tuple[int, str, int]] = set()
    metadata = _build_metadata(Path(checkpoint_path))
    metadata["lambda_mult"] = round(lambda_mult, 4)
    metadata["mu_mult"] = round(mu_mult, 4)
    metadata["ups_lambda_used"] = round(lam, 4)
    metadata["ups_mu_used"] = round(mu, 4)
    metadata["notes"] = (metadata.get("notes", "") + "  "
                        + "action_dist=" + json.dumps(dict(strategy.action_counts))).strip()

    result = create_result(
        metadata=metadata,
        experiment={
            "lambda_rate": lam, "mu_mean": mu, "seed": seed,
            "scenario_label": "paeng_v2_single",
        },
        kpi=kpi.to_dict(),
        schedule=[_batch_to_result_entry(b, params) for b in state.completed_batches],
        cancelled_batches=[
            _batch_to_result_entry(b, params, "cancelled",
                                   _match_cancel_time(b, ups_events, used_events))
            for b in state.cancelled_batches
        ],
        ups_events=[
            {"t": int(e.t), "roaster_id": e.roaster_id, "duration": int(e.duration)}
            for e in ups_events
        ],
        parameters=params,
        restocks=[
            {"line_id": rst.line_id, "sku": rst.sku,
             "start": int(rst.start), "end": int(rst.end), "qty": int(rst.qty)}
            for rst in state.completed_restocks
        ],
    )
    return result, kpi.to_dict(), state


def main():
    parser = argparse.ArgumentParser(description="Paeng DDQN v2 evaluator")
    parser.add_argument("--checkpoint", required=True, help="Path to paeng_v2_best.pt")
    parser.add_argument("--seed", type=int, default=42, help="Single seed for Mode 1")
    parser.add_argument("--n-seeds", type=int, default=None, help="Number of seeds for Mode 2")
    parser.add_argument("--base-seed", type=int, default=900000, help="Base seed for Mode 2")
    parser.add_argument("--lambda-mult", type=float, default=1.0, help="UPS rate multiplier")
    parser.add_argument("--mu-mult", type=float, default=1.0, help="UPS duration multiplier")
    parser.add_argument("--report", action="store_true", help="Generate HTML report (Mode 1 only)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.n_seeds is None:
        # Mode 1: single-seed
        result, kpi_dict, _ = evaluate_one_seed(
            Path(args.checkpoint), args.seed, args.lambda_mult, args.mu_mult
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(result), f, indent=2)
        print(f"[paeng-eval-v2] JSON: {output_path}")

        if args.report:
            from plot_result import _build_html
            html_path = output_path.parent / f"{output_path.stem}_report.html"
            html_str = _build_html(result, None, offline=True)
            html_path.write_text(html_str, encoding="utf-8")
            print(f"[paeng-eval-v2] HTML: {html_path}")

    else:
        # Mode 2: aggregate
        profits: list[float] = []
        per_seed: list[dict] = []
        for i in range(args.n_seeds):
            seed = args.base_seed + i
            _, kpi_dict, _ = evaluate_one_seed(
                Path(args.checkpoint), seed, args.lambda_mult, args.mu_mult
            )
            p = float(kpi_dict["net_profit"])
            profits.append(p)
            per_seed.append({"seed": seed, "net_profit": p, **kpi_dict})
            if (i + 1) % 20 == 0:
                print(f"  [{i + 1}/{args.n_seeds}] seeds done")

        arr = np.array(profits)
        aggregate = {
            "package": "paeng_ddqn_v2",
            "n_seeds": args.n_seeds,
            "base_seed": args.base_seed,
            "lambda_mult": args.lambda_mult,
            "mu_mult": args.mu_mult,
            "profit_mean": float(arr.mean()),
            "profit_std": float(arr.std()),
            "profit_median": float(np.median(arr)),
            "profit_p25": float(np.percentile(arr, 25)),
            "profit_p75": float(np.percentile(arr, 75)),
            "profit_min": float(arr.min()),
            "profit_max": float(arr.max()),
            "individual_results": per_seed,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(aggregate), f, indent=2)

        print(f"\n[paeng-eval-v2] 100-seed aggregate:")
        print(f"  Mean ${aggregate['profit_mean']:,.0f} ± ${aggregate['profit_std']:,.0f}")
        print(f"  Median ${aggregate['profit_median']:,.0f}, [min ${aggregate['profit_min']:,.0f}, max ${aggregate['profit_max']:,.0f}]")
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
