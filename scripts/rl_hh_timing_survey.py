"""RL-HH 100-seed timing survey.

Runs RL-HH inference on 100 seeds at the headline cell (lambda_mult=mu_mult=1.0)
with per-seed wall-time instrumentation. Records detailed timing breakdown,
prints summary stats, and emits a self-contained HTML chart.

Output:
    output/rl_hh_timing_survey.json   — raw per-seed data
    output/rl_hh_timing_survey.html   — Plotly chart with histogram + cumulative
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data
from rl_hh.meta_agent import DuelingDDQNAgent
from rl_hh.rl_hh_strategy import RLHHStrategy


SEEDS = list(range(900000, 900100))   # 100 paired seeds, identical to Block B
CHECKPOINT = _ROOT / "rl_hh" / "outputs" / "rlhh_cycle3_best.pt"
LAMBDA_MULT = 1.0
MU_MULT = 1.0


def main() -> int:
    print("=" * 72)
    print(" RL-HH 100-seed timing survey")
    print(f" checkpoint: {CHECKPOINT.relative_to(_ROOT)}")
    print(f" cell: lambda_mult={LAMBDA_MULT}  mu_mult={MU_MULT}")
    print(f" seeds: {SEEDS[0]}..{SEEDS[-1]} (n={len(SEEDS)})")
    print("=" * 72)

    # ---- one-time setup (excluded from per-seed timing) ----
    t_setup_start = time.perf_counter()
    data = load_data()
    params = data.to_env_params()
    ups_lambda = float(data.ups_lambda) * LAMBDA_MULT
    ups_mu = float(data.ups_mu) * MU_MULT
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    engine = SimulationEngine(params)

    agent = DuelingDDQNAgent()
    agent.load_checkpoint(str(CHECKPOINT))
    agent.epsilon = 0.0
    setup_sec = time.perf_counter() - t_setup_start
    print(f"\nSetup (one-time): {setup_sec*1000:.0f} ms (data + engine + agent load)\n")

    # ---- per-seed timed loop ----
    rows: list[dict] = []
    for seed in SEEDS:
        # phase 1: UPS realization
        t0 = time.perf_counter()
        ups = generate_ups_events(ups_lambda, ups_mu, seed, SL, roasters)
        t_ups = time.perf_counter() - t0

        # phase 2: strategy construction (fresh per seed — cleared tool_counts)
        t0 = time.perf_counter()
        strategy = RLHHStrategy(agent, data, training=False)
        t_strat = time.perf_counter() - t0

        # phase 3: simulation roll-out (the 480-min loop incl. all agent fires)
        t0 = time.perf_counter()
        kpi, state = engine.run(strategy, ups)
        t_run = time.perf_counter() - t0

        # phase 4: kpi extraction
        t0 = time.perf_counter()
        k = kpi.to_dict()
        net = float(kpi.net_profit())
        t_kpi = time.perf_counter() - t0

        total = t_ups + t_strat + t_run + t_kpi
        rows.append({
            "seed": seed,
            "n_ups_events": len(ups),
            "net_profit": net,
            "psc_count": int(k["psc_count"]),
            "ndg_count": int(k.get("ndg_count", 0)),
            "busta_count": int(k.get("busta_count", 0)),
            "ups_gen_ms": round(t_ups * 1000, 3),
            "strategy_init_ms": round(t_strat * 1000, 3),
            "engine_run_ms": round(t_run * 1000, 3),
            "kpi_extract_ms": round(t_kpi * 1000, 3),
            "total_ms": round(total * 1000, 3),
        })
        if (len(rows)) % 10 == 0:
            print(f"  seed {seed}: {total*1000:7.1f} ms  "
                  f"(run={t_run*1000:6.1f}ms)  "
                  f"profit=${net:>10,.0f}  ups={len(ups)}")

    # ---- summary stats ----
    totals = [r["total_ms"] for r in rows]
    runs = [r["engine_run_ms"] for r in rows]
    profits = [r["net_profit"] for r in rows]
    ups_counts = [r["n_ups_events"] for r in rows]

    def stats(label: str, vals: list[float], unit: str = "ms") -> dict:
        return {
            "label": label,
            "unit": unit,
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "p5": statistics.quantiles(vals, n=20)[0] if len(vals) >= 20 else min(vals),
            "p95": statistics.quantiles(vals, n=20)[18] if len(vals) >= 20 else max(vals),
        }

    summary = {
        "n_seeds": len(rows),
        "setup_ms": round(setup_sec * 1000, 1),
        "total_wall_sec": round(sum(totals) / 1000.0, 2),
        "total_wall_with_setup_sec": round((setup_sec + sum(totals) / 1000.0), 2),
        "total_ms": stats("total per seed", totals),
        "engine_run_ms": stats("engine_run only", runs),
        "ups_gen_ms": stats("ups generation", [r["ups_gen_ms"] for r in rows]),
        "strategy_init_ms": stats("strategy init", [r["strategy_init_ms"] for r in rows]),
        "kpi_extract_ms": stats("kpi extract", [r["kpi_extract_ms"] for r in rows]),
        "net_profit": stats("net profit", profits, unit="$"),
        "n_ups_events": stats("n_ups_events", ups_counts, unit="events"),
    }

    print()
    print("=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    print(f"  setup (one-time):           {summary['setup_ms']:>8.1f} ms")
    print(f"  total wall (eval only):     {summary['total_wall_sec']:>8.2f} s  "
          f"({summary['total_wall_sec']*1000/len(rows):.1f} ms/seed avg)")
    print(f"  total wall (incl. setup):   {summary['total_wall_with_setup_sec']:>8.2f} s")
    print()
    print(f"  per-seed total:   mean={summary['total_ms']['mean']:.1f}ms  "
          f"median={summary['total_ms']['median']:.1f}ms  "
          f"std={summary['total_ms']['stdev']:.1f}ms  "
          f"[{summary['total_ms']['min']:.1f} .. {summary['total_ms']['max']:.1f}]")
    print(f"  engine_run only:  mean={summary['engine_run_ms']['mean']:.1f}ms  "
          f"median={summary['engine_run_ms']['median']:.1f}ms  "
          f"[{summary['engine_run_ms']['min']:.1f} .. {summary['engine_run_ms']['max']:.1f}]")
    print(f"  net_profit:       mean=${summary['net_profit']['mean']:,.0f}  "
          f"median=${summary['net_profit']['median']:,.0f}")
    print(f"  n_ups_events:     mean={summary['n_ups_events']['mean']:.1f}  "
          f"min={summary['n_ups_events']['min']:.0f}  "
          f"max={summary['n_ups_events']['max']:.0f}")

    # ---- write JSON ----
    out_dir = _ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rl_hh_timing_survey.json"
    json_path.write_text(json.dumps({"summary": summary, "per_seed": rows}, indent=2),
                         encoding="utf-8")
    print(f"\n  JSON: {json_path}")

    # ---- HTML chart ----
    try:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
    except ImportError:
        print("  plotly not installed — skipping HTML chart")
        return 0

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Per-seed total wall time (n={len(rows)})",
            "Phase breakdown (mean, ms)",
            "Wall time vs UPS event count (correlation)",
            "Cumulative wall time across the 100-seed sweep",
        ),
        horizontal_spacing=0.12, vertical_spacing=0.18,
    )

    # (1,1) histogram of total_ms
    fig.add_trace(
        go.Histogram(x=totals, nbinsx=25, name="total ms",
                     hovertemplate="%{x:.1f} ms<br>%{y} seeds<extra></extra>"),
        row=1, col=1,
    )
    fig.add_vline(x=summary["total_ms"]["mean"], line_dash="dash", line_color="red",
                  annotation_text=f"mean={summary['total_ms']['mean']:.1f}ms",
                  row=1, col=1)

    # (1,2) phase breakdown bar
    phases = ["ups_gen", "strategy_init", "engine_run", "kpi_extract"]
    means = [
        summary["ups_gen_ms"]["mean"],
        summary["strategy_init_ms"]["mean"],
        summary["engine_run_ms"]["mean"],
        summary["kpi_extract_ms"]["mean"],
    ]
    fig.add_trace(
        go.Bar(x=phases, y=means, text=[f"{v:.2f}" for v in means],
               textposition="outside", name="mean ms",
               hovertemplate="%{x}: %{y:.2f} ms<extra></extra>"),
        row=1, col=2,
    )

    # (2,1) scatter of total_ms vs n_ups_events
    fig.add_trace(
        go.Scatter(
            x=ups_counts, y=totals, mode="markers",
            marker=dict(size=6, opacity=0.6),
            name="seed",
            hovertemplate=(
                "seed=%{customdata[0]}<br>"
                "ups_events=%{x}<br>"
                "total=%{y:.1f}ms<br>"
                "profit=$%{customdata[1]:,.0f}<extra></extra>"
            ),
            customdata=[[r["seed"], r["net_profit"]] for r in rows],
        ),
        row=2, col=1,
    )

    # (2,2) cumulative wall time
    cum = []
    s = 0.0
    for v in totals:
        s += v
        cum.append(s / 1000.0)
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(rows) + 1)), y=cum, mode="lines",
            name="cumulative s",
            hovertemplate="seed #%{x}<br>cum=%{y:.2f}s<extra></extra>",
        ),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="ms per seed", row=1, col=1)
    fig.update_yaxes(title_text="seeds", row=1, col=1)
    fig.update_xaxes(title_text="phase", row=1, col=2)
    fig.update_yaxes(title_text="mean ms", row=1, col=2)
    fig.update_xaxes(title_text="n_ups_events", row=2, col=1)
    fig.update_yaxes(title_text="total ms", row=2, col=1)
    fig.update_xaxes(title_text="seed index (1..100)", row=2, col=2)
    fig.update_yaxes(title_text="cumulative s", row=2, col=2)

    fig.update_layout(
        title=dict(
            text=(f"RL-HH 100-Seed Timing Survey<br>"
                  f"<sub>checkpoint=rlhh_cycle3_best.pt | "
                  f"cell λ_mult={LAMBDA_MULT}, μ_mult={MU_MULT} | "
                  f"total wall (eval only) = {summary['total_wall_sec']:.2f}s | "
                  f"mean per-seed = {summary['total_ms']['mean']:.1f}ms</sub>"),
            x=0.5, xanchor="center",
        ),
        showlegend=False,
        height=720, width=1200,
        template="plotly_white",
        margin=dict(t=120, l=60, r=40, b=60),
    )

    html_path = out_dir / "rl_hh_timing_survey.html"
    html_path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True),
                         encoding="utf-8")
    print(f"  HTML: {html_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
