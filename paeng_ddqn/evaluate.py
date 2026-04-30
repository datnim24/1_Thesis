"""Paeng's Modified DDQN — evaluator (two modes).

Mode 1 — single-seed report-emitting (plot_result.py-compatible):
    python -m paeng_ddqn.evaluate \
        --checkpoint paeng_ddqn/outputs/paeng_best.pt \
        --seed 42 --report \
        --output paeng_ddqn/outputs/paeng_seed42_result.json

Mode 2 — 100-seed aggregate (RLHH_final_100seed shape):
    python -m paeng_ddqn.evaluate \
        --checkpoint paeng_ddqn/outputs/paeng_best.pt \
        --n-seeds 100 --base-seed 900000 \
        --lambda-mult 1.0 --mu-mult 1.0 \
        --output paeng_ddqn/outputs/paeng_100seed.json

The Mode 2 evaluator delegates to ``test_rl_hh.evaluate_100seeds``'s
centralized factory (which already supports `--package paeng_ddqn` from
Task 2). This single-source-of-truth pattern is why our 100-seed JSON
shape is identical across rl_hh, q_learning, dispatching, and paeng_ddqn.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
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

from paeng_ddqn.agent import PaengAgent
from paeng_ddqn.strategy import PaengStrategy

# Reuse the q_learning result-builder helpers (clean, schema-compatible)
from q_learning.q_learning_run import _batch_to_result_entry, _match_cancel_time


# ---------------------------------------------------------------------------
# Universal-schema JSON output (recursive normalizer for tuple-keyed dicts)
# ---------------------------------------------------------------------------

def _json_safe(obj):
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


# ---------------------------------------------------------------------------
# Mode 1: single-seed report
# ---------------------------------------------------------------------------

def _build_metadata(checkpoint_path: Path, training_started_at: str | None = None) -> dict:
    metadata: dict = {
        "solver_engine": "paeng_ddqn",
        "solver_name": "Paeng's Modified DDQN",
        "status": "Completed",
        "input_dir": "Input_data",
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_path": str(checkpoint_path),
    }
    if training_started_at:
        metadata["training_started_at"] = training_started_at
    # Try to read training summary if it sits next to the checkpoint
    summary_path = checkpoint_path.parent / "training_summary.json"
    if summary_path.exists():
        try:
            tsum = json.loads(summary_path.read_text(encoding="utf-8"))
            metadata["notes"] = (
                f"Paeng DDQN: {tsum.get('episodes', '?')} episodes, "
                f"best train profit ${tsum.get('best_profit', '?')}, "
                f"final epsilon {tsum.get('final_epsilon', '?')}"
            )
            metadata["training_run_dir"] = str(checkpoint_path.parent)
        except Exception:
            pass
    return metadata


def evaluate_one_seed(
    checkpoint_path: Path,
    seed: int,
    lambda_mult: float = 1.0,
    mu_mult: float = 1.0,
) -> tuple[dict, "object"]:
    """Run a single seed; return (universal-schema result dict, sim_state)."""
    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    base_lam = float(data.ups_lambda)
    base_mu = float(data.ups_mu)
    lam = base_lam * lambda_mult
    mu = base_mu * mu_mult

    agent = PaengAgent.from_checkpoint(checkpoint_path)
    agent.epsilon = 0.0  # greedy for evaluation
    strategy = PaengStrategy(agent, data, training=False)

    engine = SimulationEngine(params)
    ups_events = generate_ups_events(lam, mu, seed, SL, roasters)
    kpi, state = engine.run(strategy, ups_events)

    # Build schema-compatible result via reused q_learning helpers
    used_events: set[tuple[int, str, int]] = set()
    metadata = _build_metadata(checkpoint_path)
    metadata["lambda_mult"] = round(lambda_mult, 4)
    metadata["mu_mult"] = round(mu_mult, 4)
    metadata["ups_lambda_used"] = round(lam, 4)
    metadata["ups_mu_used"] = round(mu, 4)
    # Action distribution as a "tool"-style summary
    metadata["notes"] = (metadata.get("notes", "") + "  "
                        + "action_dist=" + json.dumps(dict(strategy.action_counts))).strip()

    result = create_result(
        metadata=metadata,
        experiment={
            "lambda_rate": lam, "mu_mean": mu, "seed": seed,
            "scenario_label": "paeng_single",
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
    return result, state


def write_report(result: dict, json_path: Path, html_path: Path | None = None,
                 open_browser: bool = False) -> None:
    """Write JSON + (optionally) plot_result HTML."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)
    print(f"  JSON: {json_path}")

    if html_path is not None:
        from plot_result import _build_html
        html = _build_html(result, None, offline=True)
        html_path.write_text(html, encoding="utf-8")
        print(f"  HTML: {html_path}")
        if open_browser:
            try:
                webbrowser.open(html_path.resolve().as_uri())
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Mode 2: 100-seed aggregate (delegates to test_rl_hh.evaluate_100seeds)
# ---------------------------------------------------------------------------

def evaluate_n_seeds_aggregate(
    checkpoint_path: Path,
    n_seeds: int,
    base_seed: int,
    lambda_mult: float,
    mu_mult: float,
) -> dict:
    """Reuse the centralized 100-seed evaluator with --package paeng_ddqn."""
    from test_rl_hh.evaluate_100seeds import evaluate_100_seeds
    seeds = list(range(base_seed, base_seed + n_seeds))
    return evaluate_100_seeds(
        package="paeng_ddqn",
        seeds=seeds,
        checkpoint_path=str(checkpoint_path),
        lambda_mult=lambda_mult,
        mu_mult=mu_mult,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Paeng DDQN checkpoint (single-seed or 100-seed).")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42, help="Mode 1 seed.")
    parser.add_argument("--report", action="store_true", help="Mode 1: also write plot_result HTML.")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open the HTML in browser.")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="Mode 2: if set, run aggregate over n seeds.")
    parser.add_argument("--base-seed", type=int, default=900_000)
    parser.add_argument("--lambda-mult", type=float, default=1.0)
    parser.add_argument("--mu-mult", type=float, default=1.0)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    out = Path(args.output)
    if not out.is_absolute():
        out = _ROOT / out

    if args.n_seeds is None:
        # Mode 1
        print(f"[paeng-eval] Mode 1: seed={args.seed}, lambda x{args.lambda_mult}, mu x{args.mu_mult}")
        result, _ = evaluate_one_seed(ckpt, args.seed, args.lambda_mult, args.mu_mult)
        kpi = result["kpi"]
        print(f"  Net profit:  ${kpi.get('net_profit', 0):,.0f}")
        print(f"  PSC: {kpi.get('psc_count', 0)}, NDG: {kpi.get('ndg_count', 0)}, BUSTA: {kpi.get('busta_count', 0)}")
        print(f"  Tard:        ${kpi.get('tard_cost', 0):,.0f}")
        print(f"  Stockout:    ${kpi.get('stockout_cost', 0):,.0f}")
        print(f"  Setup:       {kpi.get('setup_events', 0)} events (${kpi.get('setup_cost', 0):,.0f})")
        print(f"  Restocks:    {kpi.get('restock_count', 0)}")
        html_path = out.with_suffix("").with_name(out.stem + "_report.html") if args.report else None
        write_report(result, out, html_path, open_browser=(not args.no_open and args.report))
    else:
        # Mode 2
        print(f"[paeng-eval] Mode 2: n_seeds={args.n_seeds}, base_seed={args.base_seed}, "
              f"lambda x{args.lambda_mult}, mu x{args.mu_mult}")
        agg = evaluate_n_seeds_aggregate(
            ckpt, args.n_seeds, args.base_seed, args.lambda_mult, args.mu_mult,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(_json_safe(agg), f, indent=2)
        print(f"  Mean profit: ${agg['profit_mean']:,.0f}  std=${agg['profit_std']:,.0f}")
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
