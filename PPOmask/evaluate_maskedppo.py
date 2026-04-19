"""Evaluate a trained MaskablePPO checkpoint and generate reports.

Usage:
    python -m PPOmask.evaluate_maskedppo --model-path PPOmask/outputs/PPO_20260409/checkpoints/best_model.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
import time
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.export import export_run
from result_schema import create_result
from sb3_contrib import MaskablePPO

from PPOmask.Engine.data_loader import load_data
from PPOmask.Engine.ppo_strategy import resolve_model_path
from PPOmask.Engine.roasting_env import RoastingMaskEnv

_PPO_ROOT = Path(__file__).resolve().parent
_RESULTS_DIR = PROJECT_ROOT / "results"
PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO checkpoint.")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--run-dir", default=None, help="Override output dir.")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic-runs", type=int, default=3)
    p.add_argument("--stochastic-runs", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ups-lambda", type=float, default=None)
    p.add_argument("--ups-mu", type=float, default=None)
    p.add_argument("--no-open-report", action="store_true")
    p.add_argument("--offline-report", action="store_true")
    return p.parse_args(argv)


def _json_safe(obj):
    """Recursively convert tuple dict-keys to strings so json.dumps never fails."""
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def _batch_to_result_entry(batch, params: dict) -> dict:
    sku = batch.sku
    roaster = batch.roaster
    pipe_line = params["R_pipe"][roaster]
    needs_setup = False
    return {
        "batch_id": str(batch.batch_id),
        "sku": sku,
        "roaster": roaster,
        "start": int(batch.start),
        "end": int(batch.end),
        "output_line": batch.output_line,
        "is_mto": batch.is_mto,
        "pipeline": pipe_line,
        "status": "completed",
        "setup": needs_setup,
    }


def run_episode(
    model: MaskablePPO,
    data,
    seed: int,
    deterministic: bool,
    export_dir: Path,
    run_name: str,
    ups_lambda: float | None,
    ups_mu: float | None,
) -> dict:
    env = RoastingMaskEnv(
        data=data, scenario_seed=seed,
        ups_lambda=ups_lambda, ups_mu=ups_mu,
    )
    obs, _ = env.reset(seed=seed)
    terminated = False
    reward_sum = 0.0
    inference_ms = 0.0
    decisions = 0

    while not terminated:
        mask = env.action_masks()
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        inference_ms += (time.perf_counter() - t0) * 1000.0
        decisions += 1
        obs, reward, terminated, truncated, _ = env.step(int(action))
        reward_sum += reward

    result = env.get_result()
    kpi = result["kpi"]

    # Verify reward consistency
    profit_delta = abs(reward_sum - result["net_profit"])
    if profit_delta > 2.0 and not result.get("violation", False):
        print(
            f"[eval] WARNING: reward/profit mismatch on {run_name}: "
            f"reward_sum={reward_sum:.2f}, net_profit={result['net_profit']:.2f}, delta={profit_delta:.2f}"
        )

    # Export raw data via canonical export
    assert env.state is not None and env.kpi is not None
    export_dir.mkdir(parents=True, exist_ok=True)
    export_paths = export_run(
        env.kpi.to_dict(), env.state, env.params,
        env.ups_events, output_dir=str(export_dir), run_id=run_name,
    )

    # Build result_schema-compliant JSON
    lam = ups_lambda if ups_lambda is not None else data.ups_lambda
    mu = ups_mu if ups_mu is not None else data.ups_mu
    case_label = "ups" if lam > 0 else "no_ups"
    schema_result = create_result(
        metadata={
            "solver_engine": "simulation",
            "solver_name": "MaskablePPO",
            "status": "Completed",
            "input_dir": str(data.input_dir),
            "timestamp": datetime.now().isoformat(),
        },
        experiment={
            "lambda_rate": lam, "mu_mean": mu,
            "seed": seed, "scenario_label": case_label,
        },
        kpi=env.kpi.to_dict(),
        schedule=[_batch_to_result_entry(b, env.params) for b in env.state.completed_batches],
        cancelled_batches=[
            {"batch_id": str(b.batch_id), "sku": b.sku, "roaster": b.roaster,
             "start": int(b.start), "end": int(b.end), "status": "cancelled"}
            for b in env.state.cancelled_batches
        ],
        ups_events=[
            {"t": int(e.t), "roaster_id": e.roaster_id, "duration": int(e.duration)}
            for e in env.ups_events
        ],
        parameters=env.params,
        restocks=[
            {"line_id": r.line_id, "sku": r.sku, "start": int(r.start),
             "end": int(r.end), "qty": int(r.qty)}
            for r in env.state.completed_restocks
        ],
    )

    # Save schema result JSON
    schema_json_path = export_dir / f"{run_name}_schema_result.json"
    schema_json_path.write_text(json.dumps(_json_safe(schema_result), indent=2, default=str), encoding="utf-8")
    export_paths["schema_result_json"] = str(schema_json_path)

    return {
        "run_name": run_name,
        "seed": seed,
        "deterministic": deterministic,
        "reward_sum": reward_sum,
        "net_profit": result["net_profit"],
        "psc_throughput": kpi.get("psc_count", 0),
        "mto_tardiness_min": sum(float(v) for v in kpi.get("tardiness_min", {}).values()),
        "stockout_count": sum(int(v) for v in kpi.get("stockout_events", {}).values()),
        "stockout_duration": sum(int(v) for v in kpi.get("stockout_duration", {}).values()),
        "cancelled_batches": len(env.state.cancelled_batches),
        "restock_count": int(kpi.get("restock_count", 0)),
        "setup_events": int(kpi.get("setup_events", 0)),
        "inference_time_ms": inference_ms,
        "mean_inference_ms": inference_ms / max(1, decisions),
        "invalid_action_count": int(result["invalid_action_count"]),
        "violation": result.get("violation", False),
        "violation_type": result.get("violation_type", ""),
        "violation_counts": result.get("violation_counts", {}),
        "export_paths": export_paths,
        "schema_result": schema_result,
    }


def _eval_grade(mean_profit: float, violation_rate: float) -> str:
    if mean_profit >= 280_000 and violation_rate < 0.05:
        return "A"
    if mean_profit >= 240_000 and violation_rate < 0.10:
        return "B"
    if mean_profit >= 200_000 and violation_rate < 0.20:
        return "C"
    if mean_profit >= 150_000 and violation_rate < 0.40:
        return "D"
    return "F"


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"episodes": 0}
    profits = [r["net_profit"] for r in records]
    psc_counts = [r.get("psc_throughput", 0) for r in records]
    stockouts = [r.get("stockout_count", 0) for r in records]
    violations = sum(1 for r in records if r.get("violation"))
    viol_rate = violations / len(records)
    mean_profit = statistics.mean(profits)
    return {
        "episodes": len(records),
        "mean_profit": mean_profit,
        "std_profit": statistics.stdev(profits) if len(profits) > 1 else 0.0,
        "min_profit": min(profits),
        "max_profit": max(profits),
        "violations": violations,
        "violation_rate": viol_rate,
        "mean_psc_count": statistics.mean(psc_counts) if psc_counts else 0.0,
        "mean_stockouts": statistics.mean(stockouts) if stockouts else 0.0,
        "mean_inference_ms": statistics.mean(r["mean_inference_ms"] for r in records),
        "grade": _eval_grade(mean_profit, viol_rate),
    }


def _fmt(value) -> str:
    v = float(value)
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    return f"{v:,.2f}"


def _build_summary_html(summary: dict, run_dir: Path) -> str:
    css = """
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #F6F8FB; color: #22303C; max-width: 1400px; margin: 0 auto; padding: 20px; }
    h1 { color: #0D47A1; border-bottom: 2px solid #0D47A1; padding-bottom: 10px; }
    table { border-collapse: collapse; width: 100%; background: white; margin-bottom: 20px; }
    th { background: #22303C; color: white; padding: 8px 12px; border: 1px solid #D7DCE2; }
    td { padding: 8px 12px; border: 1px solid #D7DCE2; }
    .note { background: #E3F2FD; border: 1px solid #BBDEFB; border-radius: 8px; padding: 12px; margin: 12px 0; }
    .violation { color: #C62828; font-weight: bold; }
    """
    parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>PPO Evaluation</title><style>{css}</style></head><body>",
        f"<h1>MaskablePPO Evaluation Report</h1>",
        f"<div class='note'>Model: {escape(summary.get('model_path', 'N/A'))}<br>"
        f"Generated: {escape(summary.get('generated_at', ''))}</div>",
    ]

    # Summary table
    for mode in ("deterministic", "stochastic"):
        agg = summary.get(mode, {})
        parts.append(f"<h2>{mode.title()} Evaluation ({agg.get('episodes', 0)} episodes)</h2>")
        viol_count = agg.get('violations', 0)
        viol_class = ' class="violation"' if viol_count > 0 else ''
        parts.append(
            f"<table><tr><th>Mean Profit</th><th>Min</th><th>Max</th><th>Violations</th><th>Inference (ms)</th></tr>"
            f"<tr><td>${_fmt(agg.get('mean_profit', 0))}</td><td>${_fmt(agg.get('min_profit', 0))}</td>"
            f"<td>${_fmt(agg.get('max_profit', 0))}</td>"
            f"<td{viol_class}>{viol_count}</td>"
            f"<td>{_fmt(agg.get('mean_inference_ms', 0))}</td></tr></table>"
        )

    # Per-run table
    parts.append("<h2>Per-Run Details</h2>")
    parts.append(
        "<table><tr><th>Run</th><th>Mode</th><th>Seed</th><th>Net Profit</th><th>PSC</th>"
        "<th>Tardiness</th><th>Stockouts</th><th>Setups</th><th>Restocks</th><th>Violation</th></tr>"
    )
    for r in summary.get("runs", []):
        mode = "Det" if r["deterministic"] else "Stoch"
        viol = f"<span class='violation'>{r['violation_type']}</span>" if r.get("violation") else "-"
        parts.append(
            f"<tr><td>{escape(r['run_name'])}</td><td>{mode}</td><td>{r['seed']}</td>"
            f"<td>${_fmt(r['net_profit'])}</td><td>{r.get('psc_throughput', 0)}</td>"
            f"<td>{_fmt(r.get('mto_tardiness_min', 0))}</td><td>{r.get('stockout_count', 0)}</td>"
            f"<td>{r.get('setup_events', 0)}</td><td>{r.get('restock_count', 0)}</td>"
            f"<td>{viol}</td></tr>"
        )
    parts.append("</table>")

    # Generate per-run dashboard links
    parts.append("<h2>Detailed Dashboards</h2>")
    try:
        from plot_result import _build_html as build_plot_dashboard, _load_any_result
        for r in summary.get("runs", []):
            schema_path = r.get("export_paths", {}).get("schema_result_json")
            if schema_path and Path(schema_path).exists():
                dash_name = f"{r['run_name']}_dashboard.html"
                dash_path = run_dir / "plots" / dash_name
                dash_path.parent.mkdir(parents=True, exist_ok=True)
                result_data = _load_any_result(Path(schema_path))
                html = build_plot_dashboard(result=result_data, compare=None, offline=False)
                dash_path.write_text(html, encoding="utf-8")
                rel = dash_path.relative_to(run_dir).as_posix()
                parts.append(f"<p><a href='{rel}' target='_blank'>{escape(r['run_name'])} dashboard</a></p>")
    except Exception as exc:
        parts.append(f"<p>Could not generate dashboards: {escape(str(exc))}</p>")

    parts.append(f"<div style='margin-top:30px;color:#607D8B;font-size:12px;'>Generated by PPOmask/evaluate_maskedppo.py</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    data = load_data(args.input_dir)
    model_path = resolve_model_path(args.model_path)

    # Resolve run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
    else:
        run_dir = model_path.parent.parent  # checkpoints -> run_dir
        if run_dir.name == "outputs" or not run_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = _PPO_ROOT / "outputs" / f"eval_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    model = MaskablePPO.load(str(model_path))
    det_runs = 1 if args.smoke else args.deterministic_runs
    stoch_runs = 1 if args.smoke else args.stochastic_runs

    ups_lam = args.ups_lambda if args.ups_lambda is not None else data.ups_lambda
    ups_mu_val = args.ups_mu if args.ups_mu is not None else data.ups_mu

    print(f"[eval] Model: {model_path}")
    print(f"[eval] Output: {run_dir}")
    print(f"[eval] {det_runs} deterministic + {stoch_runs} stochastic episodes")

    records: list[dict] = []
    for i in range(det_runs):
        seed = args.seed + i
        print(f"  Deterministic episode {i+1}/{det_runs} (seed={seed})...")
        records.append(run_episode(
            model, data, seed, True, eval_dir, f"deterministic_{seed}", ups_lam, ups_mu_val,
        ))

    for i in range(stoch_runs):
        seed = args.seed + 10_000 + i
        print(f"  Stochastic episode {i+1}/{stoch_runs} (seed={seed})...")
        records.append(run_episode(
            model, data, seed, False, eval_dir, f"stochastic_{seed}", ups_lam, ups_mu_val,
        ))

    # Build summary
    det_records = [r for r in records if r["deterministic"]]
    stoch_records = [r for r in records if not r["deterministic"]]
    det_agg = aggregate(det_records)
    stoch_agg = aggregate(stoch_records)
    all_agg = aggregate(records)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "deterministic": det_agg,
        "stochastic": stoch_agg,
        "all": all_agg,
        "runs": records,
    }

    # ── eval_summary.json (diagnostic protocol required path) ──────────
    eval_summary = {
        "generated_at": summary["generated_at"],
        "model_path": str(model_path),
        "grade": det_agg.get("grade", "F"),
        "n_episodes_deterministic": det_agg.get("episodes", 0),
        "n_episodes_stochastic": stoch_agg.get("episodes", 0),
        "mean_profit": det_agg.get("mean_profit", 0.0),
        "std_profit": det_agg.get("std_profit", 0.0),
        "min_profit": det_agg.get("min_profit", 0.0),
        "max_profit": det_agg.get("max_profit", 0.0),
        "violation_rate": det_agg.get("violation_rate", 1.0),
        "mean_psc_count": det_agg.get("mean_psc_count", 0.0),
        "mean_stockouts": det_agg.get("mean_stockouts", 0.0),
        "mean_inference_ms": det_agg.get("mean_inference_ms", 0.0),
        "stochastic_mean_profit": stoch_agg.get("mean_profit", 0.0),
        "stochastic_violation_rate": stoch_agg.get("violation_rate", 1.0),
    }
    eval_summary_path = eval_dir / "eval_summary.json"
    eval_summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    print(f"[eval] eval_summary.json: grade={eval_summary['grade']} mean_profit=${eval_summary['mean_profit']:,.0f}")

    # ── eval_episodes.csv (diagnostic protocol required path) ──────────
    csv_path = eval_dir / "eval_episodes.csv"
    csv_fields = [
        "episode", "seed", "deterministic", "net_profit", "psc_count",
        "stockouts", "violation", "violation_type", "inference_ms",
        "invalid_action_count", "restock_count", "setup_events",
        "cancelled_batches", "mto_tardiness_min",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for ep_idx, r in enumerate(records):
            writer.writerow({
                "episode": ep_idx + 1,
                "seed": r["seed"],
                "deterministic": int(r["deterministic"]),
                "net_profit": r["net_profit"],
                "psc_count": r.get("psc_throughput", 0),
                "stockouts": r.get("stockout_count", 0),
                "violation": int(r.get("violation", False)),
                "violation_type": r.get("violation_type", ""),
                "inference_ms": r.get("inference_time_ms", 0.0),
                "invalid_action_count": r.get("invalid_action_count", 0),
                "restock_count": r.get("restock_count", 0),
                "setup_events": r.get("setup_events", 0),
                "cancelled_batches": r.get("cancelled_batches", 0),
                "mto_tardiness_min": r.get("mto_tardiness_min", 0.0),
            })
    print(f"[eval] eval_episodes.csv: {len(records)} rows -> {csv_path}")

    # ── legacy summary JSON (kept for HTML report generation) ──────────
    summary_json_path = eval_dir / "eval_summary_full.json"
    summary_json_path.write_text(
        json.dumps(_json_safe(summary), indent=2, default=str), encoding="utf-8",
    )

    # Generate HTML report
    html = _build_summary_html(summary, run_dir)
    html_path = run_dir / f"{run_dir.name}_report.html"
    html_path.write_text(html, encoding="utf-8")

    # Copy to root/results/ for plot_result.py discovery
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Copy best deterministic result JSON
    if det_records:
        best = max(det_records, key=lambda r: r["net_profit"])
        src_json = best["export_paths"].get("schema_result_json")
        if src_json and Path(src_json).exists():
            dst_json = _RESULTS_DIR / f"ppo_result_{timestamp}.json"
            shutil.copy2(src_json, dst_json)
            print(f"[eval] Result JSON copied to {dst_json}")

    # Copy HTML report
    dst_html = _RESULTS_DIR / f"ppo_result_{timestamp}_plot.html"
    shutil.copy2(html_path, dst_html)
    print(f"[eval] HTML report copied to {dst_html}")

    # Print summary
    print(f"\n[eval] Results:")
    print(f"  Deterministic: mean=${det_agg.get('mean_profit', 0):,.0f} grade={det_agg.get('grade','?')} violations={det_agg.get('violations', 0)}")
    print(f"  Stochastic:    mean=${stoch_agg.get('mean_profit', 0):,.0f} violations={stoch_agg.get('violations', 0)}")

    # Auto-open
    if not args.no_open_report:
        try:
            webbrowser.open(html_path.resolve().as_uri())
        except Exception:
            pass

    return summary


if __name__ == "__main__":
    main()
