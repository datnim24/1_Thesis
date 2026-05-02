"""Run CP-SAT for a fixed time budget with full logging on, then plot the
incumbent-over-time and gap-over-time curves.

This is the V4Evaluation.md §10.2 anytime curve — the cleanest "RL beats
unfinished CP-SAT" framing in the literature (Wheatley 2024) when overlaid
against an RL method's per-decision wall-time.

Usage::

    python scripts/cpsat_anytime_curve.py --time 600
    python scripts/cpsat_anytime_curve.py --time 600 --seed 900046 --output-dir output/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plotly.graph_objects as go
import plotly.io as pio

from CPSAT_Pure import solver as cp_solver
from CPSAT_Pure.runner import run_pure_cpsat
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data


def _setup_logging(log_path: Path) -> None:
    """Configure CP-SAT logger to print to stdout AND tee to a file."""
    fmt = "%(asctime)s [%(name)s] %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")

    cp_logger = logging.getLogger("cpsat_solver_v2")
    cp_logger.setLevel(logging.INFO)
    cp_logger.handlers.clear()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    cp_logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    cp_logger.addHandler(fh)

    runner_logger = logging.getLogger("cpsat_pure_runner")
    runner_logger.setLevel(logging.INFO)
    runner_logger.handlers.clear()
    runner_logger.addHandler(sh)
    runner_logger.addHandler(fh)


def build_anytime_html(history: list[dict],
                      time_limit: float,
                      seed: int,
                      n_ups: int,
                      final_obj: float,
                      final_bound: float,
                      final_gap_pct: float | None,
                      status: str,
                      output_html: Path) -> None:
    """Render two Plotly subplots (profit-over-time, gap-over-time) into a
    self-contained HTML."""

    if not history:
        output_html.write_text(
            "<html><body><h2>No incumbents found in the budget.</h2></body></html>",
            encoding="utf-8",
        )
        return

    times = [h["elapsed_s"] for h in history]
    objs = [h["obj"] for h in history]
    bounds = [h["bound"] for h in history]
    gaps = [h["gap_pct"] if h["gap_pct"] is not None else None for h in history]

    # --- Plot 1: profit (incumbent + bound) over time ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=times, y=objs, mode="lines+markers",
        name="Incumbent (best feasible)",
        line=dict(color="#2E7D32", width=3, shape="hv"),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="t=%{x:.2f}s<br>Profit: $%{y:,.0f}<extra>Incumbent</extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=times, y=bounds, mode="lines+markers",
        name="Best bound (LP relaxation upper bound)",
        line=dict(color="#C62828", width=2, shape="hv", dash="dash"),
        marker=dict(size=6, symbol="diamond"),
        hovertemplate="t=%{x:.2f}s<br>Bound: $%{y:,.0f}<extra>Bound</extra>",
    ))
    fig1.update_layout(
        title=(f"CP-SAT incumbent + best-bound vs wall time<br>"
               f"<sub>seed={seed} · {n_ups} UPS events · "
               f"final status={status} · final obj=${final_obj:,.0f} · final bound=${final_bound:,.0f} · "
               f"final gap={final_gap_pct:.2f}%</sub>"),
        xaxis_title="Wall-clock seconds",
        yaxis_title="Profit ($)",
        height=500,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        hovermode="x unified",
        margin=dict(l=80, r=20, t=80, b=60),
    )
    # Highlight the time-limit
    fig1.add_vline(x=time_limit, line_dash="dot", line_color="#666",
                   annotation_text=f"time limit {time_limit:.0f}s",
                   annotation_position="top right")

    # --- Plot 2: optimality gap over time ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=times, y=gaps, mode="lines+markers",
        name="Optimality gap",
        line=dict(color="#1565C0", width=3, shape="hv"),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="t=%{x:.2f}s<br>Gap: %{y:.3f}%<extra></extra>",
    ))
    fig2.update_layout(
        title=("CP-SAT optimality gap vs wall time<br>"
               f"<sub>gap = |bound − obj| / |bound| × 100. Reaching 0% = OPTIMAL proven.</sub>"),
        xaxis_title="Wall-clock seconds",
        yaxis_title="Gap (%)",
        height=500,
        margin=dict(l=80, r=20, t=80, b=60),
        hovermode="x unified",
    )
    fig2.add_vline(x=time_limit, line_dash="dot", line_color="#666",
                   annotation_text=f"time limit {time_limit:.0f}s",
                   annotation_position="top right")
    fig2.update_yaxes(rangemode="tozero")

    # CDN once, then both figures inline
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>'
    fig1_html = pio.to_html(fig1, include_plotlyjs=False, full_html=False,
                            config={"displaylogo": False, "responsive": True})
    fig2_html = pio.to_html(fig2, include_plotlyjs=False, full_html=False,
                            config={"displaylogo": False, "responsive": True})

    # Compose final HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CP-SAT Anytime Curves — {time_limit:.0f}s budget</title>
  {plotly_cdn}
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1300px;
            margin: 0 auto; padding: 24px 30px; background: #FAFAFA; color: #212121; }}
    h1 {{ color: #0D47A1; border-bottom: 3px solid #0D47A1; padding-bottom: 8px; }}
    .meta {{ background: #ECEFF1; padding: 14px 18px; border-radius: 6px; margin: 12px 0; }}
    .chart {{ background: white; padding: 16px; border-radius: 10px;
              border: 1px solid #DDD; margin: 18px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }}
    table {{ border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
    table th, table td {{ border: 1px solid #CFD8DC; padding: 6px 12px; text-align: right; }}
    table th {{ background: #ECEFF1; }}
    table th:first-child, table td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>CP-SAT Anytime Curves — 10-minute budget</h1>
  <div class="meta">
    <b>Run summary:</b> seed={seed} · UPS events fired = {n_ups} · time budget = {time_limit:.0f}s ({time_limit/60:.1f} min)<br>
    <b>Final state:</b> status = {status} · obj = ${final_obj:,.0f} · bound = ${final_bound:,.0f} · gap = {final_gap_pct:.3f}%<br>
    <b>Incumbents found:</b> {len(history)}
  </div>

  <div class="chart">
    <h2>Plot 1 — Profit vs wall time</h2>
    {fig1_html}
    <p><small>Step-function: each step is when CP-SAT found a new incumbent (better feasible
    schedule). The dashed red line is the LP-relaxation upper bound (decreasing as the
    solver tightens it via branching). Solver proves optimality when incumbent meets bound.</small></p>
  </div>

  <div class="chart">
    <h2>Plot 2 — Optimality gap vs wall time</h2>
    {fig2_html}
    <p><small>Gap = (bound − obj) / |bound| × 100. Lower = better. A run that reaches 0%
    has proved the schedule is optimal. Most large-scale UPS instances stay around
    25–35% even at hour-long budgets — this is exactly the V4Evaluation.md §3.2 finding.</small></p>
  </div>

  <div class="chart">
    <h2>Incumbent history table</h2>
    <table>
      <thead><tr><th>#</th><th>Wall time (s)</th><th>Objective</th><th>Best bound</th><th>Gap (%)</th></tr></thead>
      <tbody>
"""
    for h in history:
        gap_str = f"{h['gap_pct']:.3f}" if h["gap_pct"] is not None else "—"
        html += (f"<tr><td>#{h['incumbent']}</td>"
                 f"<td>{h['elapsed_s']:.2f}</td>"
                 f"<td>${h['obj']:,.0f}</td>"
                 f"<td>${h['bound']:,.0f}</td>"
                 f"<td>{gap_str}</td></tr>\n")
    html += """
      </tbody>
    </table>
  </div>
</body>
</html>"""
    output_html.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CP-SAT anytime-curve recorder")
    parser.add_argument("--time", type=float, default=600.0,
                        help="Time limit in seconds (default: 600 = 10 min)")
    parser.add_argument("--seed", type=int, default=900046,
                        help="UPS realization seed (default 900046 — RL-HH's best seed)")
    parser.add_argument("--lambda-mult", type=float, default=1.0)
    parser.add_argument("--mu-mult", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "cpsat_anytime.log"
    history_json = out_dir / "cpsat_anytime_history.json"
    html_path = out_dir / "cpsat_anytime_curves.html"

    _setup_logging(log_path)
    cp_solver.set_verbose(True)  # log_search_progress = True (CP-SAT internal logging)

    # Generate UPS events for the chosen seed at the chosen cell
    data = load_data()
    params = data.to_env_params()
    ups_lambda = float(data.ups_lambda) * args.lambda_mult
    ups_mu = float(data.ups_mu) * args.mu_mult
    ups_events = generate_ups_events(
        ups_lambda, ups_mu, args.seed,
        int(params["SL"]), list(params["roasters"]),
    )
    print(f"Generated {len(ups_events)} UPS events for seed={args.seed} "
          f"(lambda={ups_lambda}, mu={ups_mu}).")

    print(f"\n=== CP-SAT solve | budget = {args.time:.0f}s ({args.time/60:.1f} min) "
          f"| seed = {args.seed} | workers = {args.num_workers} ===\n")

    t0 = time.perf_counter()
    result = run_pure_cpsat(
        time_limit_sec=int(args.time),
        ups_events=ups_events,
        num_workers=args.num_workers,
    )
    wall = time.perf_counter() - t0

    print(f"\n=== Solve finished in {wall:.1f}s ===")
    print(f"Status: {result.get('status')}")
    print(f"Final obj: ${result.get('obj_value', 0):,.0f}")
    print(f"Final bound: ${result.get('best_bound', 0):,.0f}")
    print(f"Final gap: {result.get('gap_pct')}%")
    print(f"Incumbents: {result.get('num_incumbents', 0)}")

    history = result.get("solution_history", [])
    history_payload = {
        "time_budget_sec": args.time,
        "seed": args.seed,
        "lambda_used": ups_lambda,
        "mu_used": ups_mu,
        "n_ups_events": len(ups_events),
        "num_workers": args.num_workers,
        "wall_sec": round(wall, 3),
        "final_status": result.get("status"),
        "final_obj": result.get("obj_value"),
        "final_bound": result.get("best_bound"),
        "final_gap_pct": result.get("gap_pct"),
        "num_incumbents": result.get("num_incumbents"),
        "history": history,
    }
    history_json.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    print(f"\nHistory JSON: {history_json}")
    print(f"Log file:     {log_path}")

    build_anytime_html(
        history=history,
        time_limit=args.time,
        seed=args.seed,
        n_ups=len(ups_events),
        final_obj=float(result.get("obj_value", 0.0)),
        final_bound=float(result.get("best_bound", 0.0)),
        final_gap_pct=float(result.get("gap_pct") or 0.0),
        status=str(result.get("status", "?")),
        output_html=html_path,
    )
    print(f"HTML report:  {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
