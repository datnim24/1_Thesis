"""Interactive single-episode runner for a trained MaskablePPO policy.

Usage:
    # Default: latest trained model's best checkpoint
    python -m PPOmask.ppo_run

    # Specify a model
    python -m PPOmask.ppo_run --model PPOmask/outputs/PPO_20260409/checkpoints/best_model.zip

    # Multiple episodes
    python -m PPOmask.ppo_run --episodes 10

    # With UPS disruptions
    python -m PPOmask.ppo_run --ups-lambda 2 --ups-mu 15
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from result_schema import create_result

from PPOmask.Engine.data_loader import load_data
from PPOmask.Engine.ppo_strategy import PPOStrategy, resolve_model_path

_PPO_ROOT = Path(__file__).resolve().parent
_RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a trained MaskablePPO policy.")
    p.add_argument("--model", default=None, help="Checkpoint path.")
    p.add_argument("--episodes", "-e", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ups-lambda", type=float, default=None)
    p.add_argument("--ups-mu", type=float, default=None)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-open-report", action="store_true")
    p.add_argument("--input-dir", default=None)
    return p.parse_args(argv)


def _run_one_episode(
    strategy: PPOStrategy,
    params: dict,
    ups_events: list,
    seed: int,
) -> tuple:
    engine = SimulationEngine(params)
    t0 = time.perf_counter()
    kpi, state = engine.run(strategy, ups_events)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return kpi, state, ups_events, elapsed_ms


def _batch_to_entry(batch, params: dict) -> dict:
    return {
        "batch_id": str(batch.batch_id),
        "sku": batch.sku,
        "roaster": batch.roaster,
        "start": int(batch.start),
        "end": int(batch.end),
        "output_line": batch.output_line,
        "is_mto": batch.is_mto,
        "pipeline": params["R_pipe"][batch.roaster],
        "status": "completed",
        "setup": False,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    deterministic = not args.stochastic
    data = load_data(args.input_dir)
    params = data.to_env_params()
    model_path = resolve_model_path(args.model)

    print(f"[ppo_run] Model: {model_path}")
    print(f"[ppo_run] Episodes: {args.episodes}, deterministic={deterministic}")

    strategy = PPOStrategy.load(data=data, model_path=model_path, deterministic=deterministic)

    ups_lam = args.ups_lambda if args.ups_lambda is not None else data.ups_lambda
    ups_mu_val = args.ups_mu if args.ups_mu is not None else data.ups_mu
    case_label = "ups" if ups_lam > 0 else "no_ups"

    all_results: list[dict] = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        ups_events = generate_ups_events(
            lambda_rate=ups_lam, mu_mean=ups_mu_val, seed=seed,
            shift_length=data.shift_length, roasters=list(data.roasters),
        )
        # Fresh strategy for each episode to reset counters
        strat = PPOStrategy.load(data=data, model_path=model_path, deterministic=deterministic)
        kpi, state, ups, elapsed_ms = _run_one_episode(strat, params, ups_events, seed)

        profit = kpi.net_profit()
        print(f"  Episode {ep+1}/{args.episodes} (seed={seed}): profit=${profit:,.0f} "
              f"psc={kpi.psc_completed} time={elapsed_ms:.0f}ms "
              f"invalid={strat.invalid_action_count}")

        result = create_result(
            metadata={
                "solver_engine": "simulation",
                "solver_name": "MaskablePPO",
                "status": "Completed",
                "input_dir": str(data.input_dir),
                "timestamp": datetime.now().isoformat(),
            },
            experiment={
                "lambda_rate": ups_lam, "mu_mean": ups_mu_val,
                "seed": seed, "scenario_label": case_label,
            },
            kpi=kpi.to_dict(),
            schedule=[_batch_to_entry(b, params) for b in state.completed_batches],
            cancelled_batches=[
                {"batch_id": str(b.batch_id), "sku": b.sku, "roaster": b.roaster,
                 "start": int(b.start), "end": int(b.end), "status": "cancelled"}
                for b in state.cancelled_batches
            ],
            ups_events=[
                {"t": int(e.t), "roaster_id": e.roaster_id, "duration": int(e.duration)}
                for e in ups
            ],
            parameters=params,
            restocks=[
                {"line_id": r.line_id, "sku": r.sku, "start": int(r.start),
                 "end": int(r.end), "qty": int(r.qty)}
                for r in state.completed_restocks
            ],
        )
        all_results.append(result)

    # Save and generate report for the best episode
    best_idx = max(range(len(all_results)), key=lambda i: all_results[i]["kpi"]["net_profit"])
    best_result = all_results[best_idx]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _PPO_ROOT / "outputs" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save result JSON
    json_path = run_dir / f"ppo_result_{timestamp}.json"
    json_path.write_text(json.dumps(best_result, indent=2, default=str), encoding="utf-8")

    # Generate HTML report using plot_result
    html_path = run_dir / f"ppo_result_{timestamp}_plot.html"
    try:
        from plot_result import _build_html, _load_any_result
        result_data = _load_any_result(json_path)
        html = _build_html(result=result_data, compare=None, offline=False)
        html_path.write_text(html, encoding="utf-8")
    except Exception as exc:
        print(f"[ppo_run] Warning: Could not generate HTML plot: {exc}")
        html_path = None

    # Copy to root/results/
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dst_json = _RESULTS_DIR / f"ppo_result_{timestamp}.json"
    shutil.copy2(json_path, dst_json)
    if html_path and html_path.exists():
        dst_html = _RESULTS_DIR / f"ppo_result_{timestamp}_plot.html"
        shutil.copy2(html_path, dst_html)

    print(f"\n[ppo_run] Best episode profit: ${best_result['kpi']['net_profit']:,.0f}")
    print(f"[ppo_run] JSON: {json_path}")
    if html_path and html_path.exists():
        print(f"[ppo_run] HTML: {html_path}")

    # Strategy summary
    summary = strategy.summary()
    print(f"[ppo_run] Inference: {summary['predict_calls']} calls, "
          f"{summary['mean_inference_ms']:.3f} ms/call, "
          f"{summary['invalid_action_count']} invalid")

    # Auto-open HTML
    if not args.no_open_report and html_path and html_path.exists():
        try:
            webbrowser.open(html_path.resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
