"""Evaluate a trained Q-learning policy and generate an interactive HTML report.

Usage (from project root):
    python -m q_learning.q_learning_run
    python -m q_learning.q_learning_run --file q_learning/ql_results/.../q_table_run.pkl
    python -m q_learning.q_learning_run --episodes 50
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from env.data_bridge import get_sim_params
from env.simulation_engine import SimulationEngine
from dispatch.dispatching_heuristic import DispatchingHeuristic
from env.ups_generator import generate_ups_events, generate_experiment_seeds
from plot_result import (
    _build_gantt as _plotly_gantt,
    _build_gc_plot as _plotly_gc_plot,
    _build_pipeline_timeline as _plotly_pipeline_timeline,
    _build_rc_plot as _plotly_rc_plot,
    _build_restock_timeline as _plotly_restock_timeline,
)
from result_schema import create_result

from q_learning.q_strategy import QStrategy, load_q_table

_CPSAT_AVAILABLE = False

_QL_DIR = Path(__file__).resolve().parent
_RESULTS_ROOT = _QL_DIR / "ql_results"
_PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_latest_result() -> tuple[Path | None, Path | None, dict | None]:
    """Walk ql_results/ and return (q_table_path, log_path, meta) for the
    most-recently-created result folder.  Falls back to q_learning/*.pkl."""
    if _RESULTS_ROOT.is_dir():
        folders = sorted(
            [d for d in _RESULTS_ROOT.iterdir() if d.is_dir() and d.name != "_checkpoints"],
            key=lambda d: d.stat().st_ctime,
            reverse=True,
        )
        for folder in folders:
            pkls = list(folder.glob("q_table*.pkl"))
            if pkls:
                q_path = pkls[0]
                logs = list(folder.glob("training_log*.pkl"))
                log_path = logs[0] if logs else None
                meta_f = folder / "meta.json"
                meta = json.loads(meta_f.read_text()) if meta_f.exists() else None
                return q_path, log_path, meta

    fallback_q = _QL_DIR / "q_table.pkl"
    fallback_log = _QL_DIR / "training_log.pkl"
    if fallback_q.exists():
        return fallback_q, (fallback_log if fallback_log.exists() else None), None
    return None, None, None


def _resolve_file_arg(file_arg: str | None):
    """Return (q_path, log_path, meta, result_dir)."""
    if file_arg:
        p = Path(file_arg)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.exists():
            p2 = _QL_DIR / file_arg
            if p2.exists():
                p = p2
        if not p.exists():
            print(f"File not found: {file_arg}")
            return None, None, None, None
        result_dir = p.parent
        logs = list(result_dir.glob("training_log*.pkl"))
        log_path = logs[0] if logs else None
        meta_f = result_dir / "meta.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else None
        return p, log_path, meta, result_dir

    q_path, log_path, meta = _find_latest_result()
    if q_path is None:
        print("No Q-table found. Train first with: python -m q_learning.q_learning_train")
        return None, None, None, None
    result_dir = q_path.parent
    return q_path, log_path, meta, result_dir


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_strategy(strategy_factory, params, ups_events_list):
    results = []
    for ups_events in ups_events_list:
        engine = SimulationEngine(params)
        strategy = strategy_factory()
        kpi, _ = engine.run(strategy, ups_events)
        results.append((kpi.net_profit(), kpi.to_dict()))
    return results


def _run_single(strategy_factory, params, ups_events):
    engine = SimulationEngine(params)
    strategy = strategy_factory()
    return engine.run(strategy, ups_events)


def _generate_scenarios(lambda_rate, mu_mean, seeds):
    return [generate_ups_events(lambda_rate, mu_mean, s) for s in seeds]


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0 else int(math.ceil(value - 0.5))


def _format_rate(value: float) -> str:
    rounded = round(float(value))
    return str(int(rounded)) if abs(float(value) - rounded) < 1e-9 else f"{value:g}"


def _build_case_definitions(params: dict) -> list[tuple[str, float, float]]:
    env_lam = float(params.get("ups_lambda", 2.0))
    env_mu = float(params.get("ups_mu", 20.0))
    return [
        ("case1", float(max(0, _round_half_up(env_lam * 0.5))), float(max(1, _round_half_up(env_mu * 0.5)))),
        ("case2", env_lam, env_mu),
        ("case3", float(max(0, _round_half_up(env_lam * 1.5))), float(max(1, _round_half_up(env_mu * 1.5)))),
    ]


def _downsample_xy(x_values, y_values, max_points: int = 4000):
    x_arr = np.asarray(x_values)
    y_arr = np.asarray(y_values)
    if len(x_arr) <= max_points:
        return x_arr, y_arr
    idx = np.linspace(0, len(x_arr) - 1, max_points, dtype=int)
    return x_arr[idx], y_arr[idx]


def _batch_to_result_entry(batch, params: dict, status: str = "completed", cancel_time: int | None = None) -> dict[str, Any]:
    entry = {
        "batch_id": str(batch.batch_id),
        "sku": batch.sku,
        "roaster": batch.roaster,
        "start": int(batch.start),
        "end": int(batch.end),
        "output_line": batch.output_line,
        "is_mto": bool(batch.is_mto),
        "pipeline": params["R_pipe"].get(batch.roaster),
        "pipeline_start": int(batch.start),
        "pipeline_end": int(batch.start) + int(params["DC"]),
        "status": status,
        "setup": "No",
    }
    if cancel_time is not None:
        entry["cancel_time"] = int(cancel_time)
    return entry


def _match_cancel_time(batch, ups_events, used_events: set[tuple[int, str, int]]) -> int | None:
    for event in sorted(ups_events, key=lambda e: (int(e.t), e.roaster_id, int(e.duration))):
        key = (int(event.t), event.roaster_id, int(event.duration))
        if key in used_events or event.roaster_id != batch.roaster:
            continue
        if int(batch.start) <= int(event.t) < int(batch.end):
            used_events.add(key)
            return int(event.t)
    return None


def _state_to_result(strategy_name: str, params: dict, kpi, state, ups_events, case_label: str, lam: float, mu: float, seed: int):
    used_events: set[tuple[int, str, int]] = set()
    return create_result(
        metadata={
            "solver_engine": "simulation",
            "solver_name": strategy_name,
            "status": "Completed",
            "input_dir": "Input_data",
            "timestamp": datetime.now().isoformat(),
        },
        experiment={"lambda_rate": lam, "mu_mean": mu, "seed": seed, "scenario_label": case_label},
        kpi=kpi.to_dict(),
        schedule=[_batch_to_result_entry(batch, params) for batch in state.completed_batches],
        cancelled_batches=[
            _batch_to_result_entry(batch, params, "cancelled", _match_cancel_time(batch, ups_events, used_events))
            for batch in state.cancelled_batches
        ],
        ups_events=[{"t": int(e.t), "roaster_id": e.roaster_id, "duration": int(e.duration)} for e in ups_events],
        parameters=params,
        restocks=[
            {"line_id": rst.line_id, "sku": rst.sku, "start": int(rst.start), "end": int(rst.end), "qty": int(rst.qty)}
            for rst in state.completed_restocks
        ],
    )


def _fig_training_curve(rewards):
    arr = np.asarray(rewards, dtype=float)
    window = min(500, len(arr))
    smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
    raw_x, raw_y = _downsample_xy(np.arange(1, len(arr) + 1), arr)
    smooth_x, smooth_y = _downsample_xy(np.arange(window, len(arr) + 1), smoothed)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw_x, y=raw_y, mode="lines", name="Episode profit",
                             line=dict(color="rgba(33,150,243,0.28)", width=1)))
    fig.add_trace(go.Scatter(x=smooth_x, y=smooth_y, mode="lines", name=f"Rolling avg ({window})",
                             line=dict(color="#0D47A1", width=2.5)))
    fig.update_layout(template="plotly_white", title="Q-Learning Training Curve",
                      height=340, margin=dict(l=60, r=20, t=50, b=40),
                      xaxis_title="Episode", yaxis_title="Net Profit ($)", hovermode="x unified")
    return fig


def _fig_convergence(rewards):
    from q_learning.q_learning_train import EPSILON_END, EPSILON_START
    n = len(rewards)
    eps_curve = np.array([max(EPSILON_END, EPSILON_START - min(1.0, i / (0.7 * n)) * (EPSILON_START - EPSILON_END)) for i in range(n)])
    window = min(500, n)
    smoothed = np.convolve(np.asarray(rewards, dtype=float), np.ones(window) / window, mode="valid")
    eps_x, eps_y = _downsample_xy(np.arange(1, n + 1), eps_curve)
    profit_x, profit_y = _downsample_xy(np.arange(window, n + 1), smoothed)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Exploration Decay", f"Profit Convergence ({window}-ep rolling)"))
    fig.add_trace(go.Scatter(x=eps_x, y=eps_y, mode="lines", line=dict(color="#C62828", width=2), name="Epsilon"), row=1, col=1)
    fig.add_trace(go.Scatter(x=profit_x, y=profit_y, mode="lines", line=dict(color="#1565C0", width=2), name="Smoothed profit"), row=1, col=2)
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=60, r=20, t=60, b=40), showlegend=False)
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=1, col=2)
    fig.update_yaxes(title_text="Epsilon", row=1, col=1)
    fig.update_yaxes(title_text="Smoothed Profit ($)", row=1, col=2)
    return fig


def _fig_strategy_comparison(summary):
    colors = {"Dispatching": "#1565C0", "Q-Learning": "#2E7D32", "CP-SAT": "#EF6C00"}
    fig = go.Figure()
    for strat in next(iter(summary.values())).keys():
        means = [float(np.mean(summary[sc][strat])) for sc in summary]
        stds = [float(np.std(summary[sc][strat])) for sc in summary]
        fig.add_trace(go.Bar(x=list(summary.keys()), y=means, name=strat, marker_color=colors.get(strat, "#607D8B"),
                             error_y=dict(type="data", array=stds, visible=True)))
    fig.update_layout(template="plotly_white", title="Strategy Comparison Across UPS Cases",
                      barmode="group", height=340, margin=dict(l=60, r=20, t=50, b=40),
                      xaxis_title="Scenario", yaxis_title="Net Profit ($)")
    return fig


def _fig_profit_boxplot(summary, scenario_key):
    colors = {"Dispatching": "#1565C0", "Q-Learning": "#2E7D32", "CP-SAT": "#EF6C00"}
    fig = go.Figure()
    for strat, profits in summary[scenario_key].items():
        fig.add_trace(go.Box(y=profits, name=strat, boxmean=True, marker_color=colors.get(strat, "#607D8B")))
    fig.update_layout(template="plotly_white", title=f"Profit Distribution - {scenario_key}",
                      height=340, margin=dict(l=60, r=20, t=50, b=40), yaxis_title="Net Profit ($)")
    return fig


def _save_fig(fig: go.Figure, path: Path):
    pio.write_html(fig, str(path), full_html=True, include_plotlyjs=True, config=_PLOTLY_CONFIG)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _summary_table_html(summary: dict[str, dict[str, list[float]]]) -> str:
    strats = list(next(iter(summary.values())).keys())
    parts = ["<table class='data'><tr><th>Scenario</th>"]
    for strat in strats:
        parts.append(f"<th>{strat} mean</th><th>{strat} std</th>")
    parts.append("</tr>")
    for scenario, data in summary.items():
        parts.append(f"<tr><td>{scenario}</td>")
        for strat in strats:
            parts.append(f"<td>${np.mean(data[strat]):,.0f}</td><td>${np.std(data[strat]):,.0f}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _meta_table_html(meta: dict | None) -> str:
    if not meta:
        return ""
    display_keys = [
        ("name", "Run Name"), ("episodes", "Episodes"), ("elapsed_seconds", "Training Time"),
        ("alpha", "Alpha"), ("gamma", "Gamma"), ("epsilon_start", "Epsilon Start"),
        ("epsilon_end", "Epsilon End"), ("epsilon_schedule", "Epsilon Schedule"),
        ("final_epsilon", "Final Epsilon"), ("state_formulation", "State Formulation"),
        ("ups_mode", "UPS Mode"), ("ups_lambda", "UPS Lambda"), ("ups_mu", "UPS Mu"),
        ("q_table_entries", "Q-Table Entries"), ("final_avg_profit_1000", "Avg Profit (last 1000)"),
        ("stop_reason", "Stop Reason"), ("timestamp", "Trained At"),
    ]
    rows = ["<table class='meta-table'>"]
    for key, label in display_keys:
        value = meta.get(key, "-")
        if key == "elapsed_seconds" and isinstance(value, (int, float)):
            mins, secs = divmod(int(value), 60)
            hours, mins = divmod(mins, 60)
            value = f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s"
        elif key == "final_avg_profit_1000" and isinstance(value, (int, float)):
            value = f"${value:,.0f}"
        rows.append(f"<tr><td>{label}</td><td>{value}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def _scenario_table_html(scenarios: list[tuple[str, float, float]]) -> str:
    parts = ["<table class='data'><tr><th>Case</th><th>Lambda</th><th>Mu</th><th>Definition</th></tr>"]
    for label, lam, mu in scenarios:
        rule = "Current Input_data values"
        if label == "case1":
            rule = "50% of Input_data values, rounded"
        elif label == "case3":
            rule = "150% of Input_data values, rounded"
        parts.append(f"<tr><td>{label}</td><td>{_format_rate(lam)}</td><td>{_format_rate(mu)}</td><td>{rule}</td></tr>")
    parts.append("</table>")
    return "".join(parts)


def _sim_params_html(sim_params: dict) -> str:
    rows = ["<table class='data'><tr><th>Parameter</th><th>Value</th></tr>"]
    pairs = [
        ("Shift Length", f"{sim_params.get('SL', '-')} min"),
        ("Setup Time", f"{sim_params.get('sigma', '-')} min"),
        ("Consume Duration", f"{sim_params.get('DC', '-')} min"),
        ("Max RC (buffer)", f"{sim_params.get('max_rc', '-')} batches"),
        ("Safety Stock", f"{sim_params.get('safety_stock', '-')} batches"),
        ("RC Init L1", f"{sim_params.get('rc_init', {}).get('L1', '-')} batches"),
        ("RC Init L2", f"{sim_params.get('rc_init', {}).get('L2', '-')} batches"),
        ("UPS Lambda (Input_data)", _format_rate(sim_params.get("ups_lambda", 0))),
        ("UPS Mu (Input_data)", f"{_format_rate(sim_params.get('ups_mu', 0))} min"),
        ("Restock Duration", f"{sim_params.get('restock_duration', '-')} min"),
        ("Restock Qty", f"{sim_params.get('restock_qty', '-')} batches"),
        ("Roasters", ", ".join(sim_params.get("roasters", []))),
        ("Lines", ", ".join(sim_params.get("lines", []))),
    ]
    for pair, capacity in sorted(sim_params.get("gc_capacity", {}).items()):
        pairs.append((f"GC {pair[0]}_{pair[1]}", f"init={sim_params.get('gc_init', {}).get(pair, '-')}, cap={capacity}"))
    for job_id in sim_params.get("jobs", []):
        pairs.append((f"MTO Job {job_id}", f"{sim_params.get('job_sku', {}).get(job_id, '?')} x{sim_params.get('job_batches', {}).get(job_id, '?')}, due={sim_params.get('job_due', {}).get(job_id, '?')}"))
    for label, value in pairs:
        rows.append(f"<tr><td>{label}</td><td>{value}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def _case_kpi_table(case_results: dict[str, dict[str, Any]]) -> str:
    rows = ["<table class='data'><tr><th>Strategy</th><th>Net Profit</th><th>Revenue</th><th>Costs</th><th>Setup</th><th>Tardiness</th><th>Stockout</th><th>Idle</th><th>Restocks</th></tr>"]
    for strategy, payload in case_results.items():
        kpi = payload["result"]["kpi"]
        rows.append(
            f"<tr><td>{strategy}</td><td>${kpi.get('net_profit', 0):,.0f}</td><td>${kpi.get('total_revenue', 0):,.0f}</td>"
            f"<td>${kpi.get('total_costs', 0):,.0f}</td><td>{kpi.get('setup_events', 0)} (${kpi.get('setup_cost', 0):,.0f})</td>"
            f"<td>${kpi.get('tard_cost', 0):,.0f}</td><td>${kpi.get('stockout_cost', 0):,.0f}</td>"
            f"<td>{kpi.get('idle_min', 0):,.0f} min</td><td>{kpi.get('restock_count', 0)}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _append_chart(parts: list[str], title: str, fig: go.Figure | None, slug: str, plot_dir: Path) -> None:
    if fig is None:
        return
    chart_path = plot_dir / f"{slug}.html"
    _save_fig(fig, chart_path)
    rel_src = chart_path.relative_to(plot_dir.parent).as_posix()
    frame_height = int(fig.layout.height) if getattr(fig.layout, "height", None) else 360
    parts.append(
        "<section class='chart-card'>"
        f"<h3>{title}</h3>"
        f"<iframe class='chart-frame' src='{rel_src}' title='{title}' loading='lazy' "
        f"style='height:{frame_height + 24}px;'></iframe>"
        f"<div class='chart-link'><a href='{rel_src}' target='_blank' rel='noopener'>Open chart in new tab</a></div>"
        "</section>"
    )


def _build_html(title: str, meta: dict | None, scenarios: list[tuple[str, float, float]],
                summary: dict, figures: dict[str, go.Figure], case_demo_results: dict[str, dict[str, Any]],
                eval_episodes: int, sim_params: dict, plot_dir: Path) -> str:
    css = """
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #F6F8FB; color: #22303C; max-width: 1440px; margin: 0 auto; padding: 20px 24px 40px; }
    h1 { color: #0D47A1; border-bottom: 2px solid #0D47A1; padding-bottom: 10px; }
    h2 { margin-top: 32px; }
    .note { background: #E3F2FD; border: 1px solid #BBDEFB; border-radius: 8px; padding: 12px 14px; }
    .meta-table, table.data { border-collapse: collapse; width: 100%; background: white; }
    .meta-table td, table.data td, table.data th { border: 1px solid #D7DCE2; padding: 8px 12px; }
    .meta-table td:first-child { font-weight: 700; width: 240px; }
    table.data th { background: #22303C; color: white; }
    table.data td:nth-child(n+2), table.data th:nth-child(n+2) { text-align: right; }
    table.data td:first-child, table.data th:first-child { text-align: left; }
    .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }
    .chart-card, .strategy-block { background: white; border: 1px solid #D7DCE2; border-radius: 10px; padding: 14px; }
    .strategy-block { margin-top: 16px; }
    .chart-frame { width: 100%; min-height: 260px; border: 0; border-radius: 8px; background: white; }
    .chart-link { margin-top: 8px; font-size: 12px; }
    .chart-link a { color: #1565C0; text-decoration: none; }
    .chart-link a:hover { text-decoration: underline; }
    .footer { margin-top: 36px; padding-top: 12px; border-top: 1px solid #D7DCE2; color: #607D8B; font-size: 12px; }
    """
    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>", f"<title>{title}</title><style>{css}</style></head><body>", f"<h1>{title}</h1>",
             "<div class='note'>Interactive charts are generated with Plotly. UPS cases are derived from the current <code>Input_data</code> values loaded by <code>get_sim_params()</code>.</div>"]
    if meta:
        parts.extend(["<h2>Training Parameters</h2>", _meta_table_html(meta)])
    parts.extend(["<h2>UPS Evaluation Cases</h2>", _scenario_table_html(scenarios), f"<h2>Evaluation Summary ({eval_episodes} seeds per case)</h2>", _summary_table_html(summary), "<h2>Overview Charts</h2><div class='chart-grid'>"])
    for slug, label in (("strategy_comparison", "Strategy Comparison"), ("profit_boxplot", "Profit Distribution"), ("training_curve", "Training Curve"), ("convergence", "Convergence")):
        _append_chart(parts, label, figures.get(slug), slug, plot_dir)
    parts.append("</div>")
    for case_label, lam, mu in scenarios:
        case_payload = case_demo_results[case_label]
        parts.extend([f"<h2>{case_label} Demo Timelines</h2>", f"<div class='note'>Seed 42, lambda={_format_rate(lam)}, mu={_format_rate(mu)}. The same UPS sample is used for every strategy in this case.</div>", _case_kpi_table(case_payload)])
        for strategy, payload in case_payload.items():
            prefix = f"{case_label}_{strategy.lower().replace(' ', '_').replace('-', '_')}"
            parts.append(f"<div class='strategy-block'><h3>{strategy}</h3><div class='chart-grid'>")
            for chart_key, chart_title in (("schedule", "Schedule"), ("pipeline", "Pipeline Usage"), ("rc", "RC Inventory Over Time"), ("gc_l1", "GC Inventory L1"), ("gc_l2", "GC Inventory L2"), ("restock", "Restock Timeline")):
                _append_chart(parts, f"{strategy} {chart_title}", payload["figures"].get(chart_key), f"{prefix}_{chart_key}", plot_dir)
            parts.append("</div></div>")
    parts.extend(["<h2>Simulation Input Parameters</h2>", _sim_params_html(sim_params), f"<div class='footer'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by q_learning_run.py</div></body></html>"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Q-learning vs baselines and generate an interactive HTML report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m q_learning.q_learning_run\n"
            "  python -m q_learning.q_learning_run --file q_learning/ql_results/.../q_table_run.pkl\n"
            "  python -m q_learning.q_learning_run --episodes 50\n"
        ),
    )
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Path to Q-table .pkl (default: latest in ql_results/)")
    parser.add_argument("--episodes", "-e", type=int, default=100,
                        help="UPS realisations per case (default: 100)")
    parser.add_argument("--cpsat", action="store_true",
                        help="Include CP-SAT (TL=2s) in comparison if available")
    args = parser.parse_args(argv)

    params = get_sim_params()
    q_path, log_path, meta, result_dir = _resolve_file_arg(args.file)
    if q_path is None:
        return

    q_table = load_q_table(str(q_path))
    print(f"Loaded Q-table: {q_path}")
    print(f"  {len(q_table):,} entries")
    if meta:
        print(f"  Training: {meta.get('episodes','-')} ep, alpha={meta.get('alpha','-')}, "
              f"gamma={meta.get('gamma','-')}, name={meta.get('name','-')}")

    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for stale in plot_dir.glob("*"):
        if stale.suffix.lower() in {".html", ".png"}:
            stale.unlink(missing_ok=True)

    strategy_specs: list[tuple[str, Any]] = [
        ("Dispatching", lambda: DispatchingHeuristic(params)),
        ("Q-Learning", lambda: QStrategy(params, q_table=q_table)),
    ]
    if args.cpsat and _CPSAT_AVAILABLE:
        strategy_specs.append(("CP-SAT (2s)", lambda: CPSATStrategy(params, time_limit=2.0)))

    rewards = None
    if log_path and log_path.exists():
        with open(log_path, "rb") as f:
            rewards = pickle.load(f)

    scenarios = _build_case_definitions(params)
    seeds = generate_experiment_seeds(args.episodes, base_seed=0)
    summary: dict[str, dict[str, list[float]]] = {}
    print("\nScenario definitions from Input_data:")
    for label, lam, mu in scenarios:
        print(f"  {label}: lambda={_format_rate(lam)}, mu={_format_rate(mu)}")
    print(f"\nEvaluating over {args.episodes} seeds per case...")

    for label, lam, mu in scenarios:
        print(f"\n  --- {label} (lam={_format_rate(lam)}, mu={_format_rate(mu)}) ---")
        ups_list = _generate_scenarios(lam, mu, seeds)
        case_summary: dict[str, list[float]] = {}
        for strategy_name, strategy_factory in strategy_specs:
            started = time.perf_counter()
            runs = _run_strategy(strategy_factory, params, ups_list)
            elapsed = time.perf_counter() - started
            profits = [run[0] for run in runs]
            case_summary[strategy_name] = profits
            print(f"  {strategy_name:<12} avg=${np.mean(profits):>10,.0f}  std=${np.std(profits):>8,.0f}  ({elapsed:.1f}s)")
        summary[label] = case_summary

    figures: dict[str, go.Figure] = {
        "strategy_comparison": _fig_strategy_comparison(summary),
        "profit_boxplot": _fig_profit_boxplot(summary, "case2"),
    }
    if rewards:
        print("\nBuilding interactive training charts...")
        figures["training_curve"] = _fig_training_curve(rewards)
        figures["convergence"] = _fig_convergence(rewards)

    print("\nBuilding interactive demo timelines...")
    case_demo_results: dict[str, dict[str, Any]] = {}
    for label, lam, mu in scenarios:
        ups_events = generate_ups_events(lam, mu, 42)
        case_results: dict[str, Any] = {}
        for strategy_name, strategy_factory in strategy_specs:
            kpi, state = _run_single(strategy_factory, params, ups_events)
            result = _state_to_result(strategy_name, params, kpi, state, ups_events, label, lam, mu, seed=42)
            case_results[strategy_name] = {
                "result": result,
                "figures": {
                    "schedule": _plotly_gantt(result),
                    "pipeline": _plotly_pipeline_timeline(result),
                    "rc": _plotly_rc_plot(result),
                    "gc_l1": _plotly_gc_plot(result, "L1"),
                    "gc_l2": _plotly_gc_plot(result, "L2"),
                    "restock": _plotly_restock_timeline(result),
                },
            }
        case_demo_results[label] = case_results

    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    strats = list(next(iter(summary.values())).keys())
    header = f"{'Scenario':<12}" + "".join(f" {s:>16}" for s in strats)
    print(header)
    print("-" * len(header))
    for sc_label, data in summary.items():
        row = f"{sc_label:<12}"
        for s in strats:
            row += f"{f'${np.mean(data[s]):,.0f}':>18}"
        print(row)

    report_name = result_dir.name if result_dir != _QL_DIR else "q_learning_report"
    html = _build_html(
        title=f"Q-Learning Report: {report_name}",
        meta=meta,
        scenarios=scenarios,
        summary=summary,
        figures=figures,
        case_demo_results=case_demo_results,
        eval_episodes=args.episodes,
        sim_params=params,
        plot_dir=plot_dir,
    )
    html_path = result_dir / f"{report_name}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\nInteractive HTML report: {html_path}")
    print(f"Interactive chart files: {plot_dir}")
    try:
        webbrowser.open(html_path.resolve().as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()
