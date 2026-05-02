"""RL-HH consistency check at seed=69, lambda=5, mu=20 (matches CP-SAT 6-hour reference).

CP-SAT reference: results/method_comparison/20260423_163357/seed_69/cpsat_result.json
  net_profit = $443,400, FEASIBLE, gap = 30.06%, solve = 28,836 s (~8 h),
  5 UPS events applied.

This script runs RL-HH 100 times on the IDENTICAL instance:
  - same seed (69)
  - same lambda=5.0, mu=20.0
  - same UPS realization (deterministic given seed)
  - greedy policy (epsilon=0)

Expected: all 100 runs return the same net_profit (proves RL-HH is deterministic
under realistic UPS load, not just at lambda=0). Plus optimality gap vs the
8-hour CP-SAT reference.

Output: output/rl_hh_consistency_seed69_ups.json + ..._report.html
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


CHECKPOINT      = _ROOT / "rl_hh" / "outputs" / "rlhh_cycle3_best.pt"
CPSAT_FILE      = _ROOT / "results" / "method_comparison" / "20260423_163357" / "seed_69" / "cpsat_result.json"
SEED            = 69
LAMBDA          = 5.0
MU              = 20.0
N_RUNS          = 100


def main() -> int:
    cpsat_dat = json.load(CPSAT_FILE.open())
    CPSAT_PROFIT = float(cpsat_dat["kpi"]["net_profit"])
    CPSAT_SOLVE_S = float(cpsat_dat["metadata"]["solve_time_sec"])
    CPSAT_GAP_PCT = 30.0631
    CPSAT_STATUS  = cpsat_dat["metadata"]["status"]
    n_ups_cpsat   = len(cpsat_dat.get("ups_events", []))

    print("=" * 76)
    print(" RL-HH Consistency Check — seed=69, lambda=5, mu=20  (matches CP-SAT 8h)")
    print(f" CP-SAT reference: ${CPSAT_PROFIT:,.0f}  status={CPSAT_STATUS}  "
          f"gap={CPSAT_GAP_PCT:.2f}%  solve={CPSAT_SOLVE_S/3600:.2f}h")
    print(f"                  {n_ups_cpsat} UPS events (deterministic given seed)")
    print("=" * 76)

    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    engine = SimulationEngine(params)

    agent = DuelingDDQNAgent()
    agent.load_checkpoint(str(CHECKPOINT))
    agent.epsilon = 0.0

    ups = generate_ups_events(LAMBDA, MU, SEED, SL, roasters)
    print(f"\nGenerated {len(ups)} UPS events for seed={SEED}, lambda={LAMBDA}, mu={MU}:")
    for i, ev in enumerate(ups):
        print(f"   ev{i+1}: t={ev.t}  roaster={ev.roaster_id}  duration={ev.duration}min")
    if n_ups_cpsat != len(ups):
        print(f"   WARNING: CP-SAT had {n_ups_cpsat} UPS events; we have {len(ups)}")

    print(f"\n[Run] {N_RUNS}x RL-HH on this exact instance (greedy, epsilon=0)")
    rows: list[dict] = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        strategy = RLHHStrategy(agent, data, training=False)
        kpi, _ = engine.run(strategy, ups)
        wall = time.perf_counter() - t0
        k = kpi.to_dict()
        rows.append({
            "run_idx": i,
            "net_profit": float(kpi.net_profit()),
            "revenue": float(k["total_revenue"]),
            "tard_cost": float(k["tard_cost"]),
            "setup_cost": float(k["setup_cost"]),
            "stockout_cost": float(k["stockout_cost"]),
            "idle_cost": float(k["idle_cost"]),
            "psc_count": int(k["psc_count"]),
            "ndg_count": int(k.get("ndg_count", 0)),
            "busta_count": int(k.get("busta_count", 0)),
            "setup_events": int(k["setup_events"]),
            "restock_count": int(k["restock_count"]),
            "wall_ms": round(wall * 1000, 2),
        })
        if (i+1) % 25 == 0:
            print(f"   completed {i+1}/{N_RUNS}  last_profit=${rows[-1]['net_profit']:,.0f}  "
                  f"wall={rows[-1]['wall_ms']:.1f}ms")

    profits = [r["net_profit"] for r in rows]
    walls = [r["wall_ms"] for r in rows]
    unique = sorted(set(profits))

    print()
    print("=" * 76)
    print(" SUMMARY")
    print("=" * 76)
    print(f"   all {N_RUNS} runs identical:    "
          f"{'YES' if len(unique)==1 else f'NO ({len(unique)} unique values)'}")
    print(f"   profit:  mean=${statistics.mean(profits):,.2f}  "
          f"min=${min(profits):,.2f}  max=${max(profits):,.2f}  "
          f"std=${statistics.stdev(profits) if len(profits)>1 else 0.0:,.2f}")
    print(f"   wall:    mean={statistics.mean(walls):.1f}ms  "
          f"min={min(walls):.1f}ms  max={max(walls):.1f}ms  "
          f"total={sum(walls)/1000:.2f}s")

    rl_hh_value = statistics.mean(profits)
    gap_pct = (CPSAT_PROFIT - rl_hh_value) / CPSAT_PROFIT * 100.0
    speedup = CPSAT_SOLVE_S * 1000.0 / statistics.mean(walls)
    print()
    print("=" * 76)
    print(" Headline comparison")
    print("=" * 76)
    print(f"   CP-SAT (seed=69, lambda=5, mu=20, 8h solve):  ${CPSAT_PROFIT:>10,.0f}")
    print(f"   RL-HH  (same instance, {N_RUNS} runs ~0.28s each):  ${rl_hh_value:>10,.0f}")
    print(f"   Gap:                                          {gap_pct:+>10.2f}%")
    print(f"   Speed-up:                                     ~{speedup:>10,.0f}x")

    out_dir = _ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rl_hh_consistency_seed69_ups.json"
    json_path.write_text(json.dumps({
        "cpsat_reference": {
            "net_profit": CPSAT_PROFIT,
            "seed": SEED, "lambda": LAMBDA, "mu": MU,
            "solve_sec": CPSAT_SOLVE_S, "status": CPSAT_STATUS, "gap_pct": CPSAT_GAP_PCT,
            "n_ups_events": n_ups_cpsat,
            "source_file": str(CPSAT_FILE.relative_to(_ROOT)),
        },
        "rl_hh_summary": {
            "n_runs": N_RUNS, "all_identical": len(unique) == 1,
            "n_unique": len(unique), "profit_mean": rl_hh_value,
            "profit_min": min(profits), "profit_max": max(profits),
            "profit_stdev": statistics.stdev(profits) if len(profits) > 1 else 0.0,
            "wall_ms_mean": statistics.mean(walls),
            "wall_ms_total": sum(walls),
            "gap_pct_vs_cpsat": gap_pct, "speedup": speedup,
        },
        "ups_events": [{"t": e.t, "roaster_id": e.roaster_id, "duration": e.duration} for e in ups],
        "runs": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n   JSON: {json_path}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("   plotly not installed — skipping HTML")
        return 0

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Net profit per run (n={N_RUNS}, seed=69, λ=5, μ=20)",
            "Wall-time per run (ms)",
            "RL-HH vs CP-SAT (same instance)",
            "Gap to CP-SAT reference (%)",
        ),
        horizontal_spacing=0.13, vertical_spacing=0.18,
    )
    fig.add_trace(go.Scatter(x=list(range(1, N_RUNS+1)), y=profits, mode="markers",
                             name="RL-HH profit"), row=1, col=1)
    fig.add_hline(y=CPSAT_PROFIT, line_dash="dash", line_color="red",
                  annotation_text=f"CP-SAT ${CPSAT_PROFIT:,.0f}",
                  annotation_position="top right", row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1, N_RUNS+1)), y=walls, mode="lines+markers",
                             name="wall ms"), row=1, col=2)
    fig.add_trace(go.Bar(
        x=["CP-SAT (8h, FEASIBLE)", "RL-HH (mean of 100, 0.28s each)"],
        y=[CPSAT_PROFIT, rl_hh_value],
        text=[f"${CPSAT_PROFIT:,.0f}", f"${rl_hh_value:,.0f}"],
        textposition="outside",
        marker_color=["#4c8bf5", "#5cb85c"],
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=["RL-HH gap (%)"],
        y=[gap_pct],
        text=[f"{gap_pct:+.2f}%"],
        textposition="outside",
        marker_color=["#f0ad4e" if gap_pct > 0 else "#5cb85c"],
    ), row=2, col=2)

    fig.update_xaxes(title_text="run index", row=1, col=1)
    fig.update_yaxes(title_text="net profit ($)", row=1, col=1)
    fig.update_xaxes(title_text="run index", row=1, col=2)
    fig.update_yaxes(title_text="wall ms", row=1, col=2)
    fig.update_yaxes(title_text="$", row=2, col=1)
    fig.update_yaxes(title_text="% gap", row=2, col=2)

    consistency_msg = (
        f"<b>Consistency:</b> {'✅ ALL 100 RUNS IDENTICAL' if len(unique)==1 else '⚠️ ' + str(len(unique)) + ' unique'} "
        f"(std=${statistics.stdev(profits) if len(profits)>1 else 0:,.2f}) &nbsp;|&nbsp; "
        f"<b>Gap vs CP-SAT (8h):</b> {gap_pct:+.2f}% &nbsp;|&nbsp; "
        f"<b>Speedup:</b> ~{speedup:,.0f}× faster"
    )
    fig.update_layout(
        title=dict(
            text=("RL-HH Consistency on seed=69 with UPS (λ=5, μ=20)<br>"
                  f"<sub>{consistency_msg}</sub>"),
            x=0.5, xanchor="center",
        ),
        showlegend=False,
        height=800, width=1200,
        template="plotly_white",
        margin=dict(t=120, l=60, r=40, b=60),
    )
    html_path = out_dir / "rl_hh_consistency_seed69_ups.html"
    html_path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True),
                         encoding="utf-8")
    print(f"   HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
