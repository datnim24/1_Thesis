"""RL-HH consistency check at no-UPS (lambda=0).

Purpose: prove that RL-HH produces the same schedule on repeated runs of the
same input. Two tests:

  Test A — "pure determinism":  100 runs with the SAME seed (seed=69, lambda=0).
           All 100 should be identical to byte-perfect; any variation reveals
           hidden RNG inside the strategy or tools.

  Test B — "seed-invariance at no-UPS":  100 runs with DIFFERENT seeds
           (900000..900099) at lambda=0. Since lambda=0 means UPS event list is
           empty regardless of seed, the seed argument is functionally inert —
           all 100 should still be identical.

Comparison: CP-SAT no-UPS reference is $481,200 on seed=69 (8h solve, FEASIBLE,
gap=26.4%).  Reports per-run RL-HH profit, optimality gap vs CP-SAT, and
distribution stats.

Output: output/rl_hh_consistency_no_ups.json + ..._report.html
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


CHECKPOINT = _ROOT / "rl_hh" / "outputs" / "rlhh_cycle3_best.pt"
CPSAT_REFERENCE = 481200.0   # from results/method_comparison/20260424_084805/seed_69/cpsat_result.json
CPSAT_SEED = 69
CPSAT_SOLVE_HOURS = 8.0


def run_one(engine, strategy_factory, ups, label: str) -> dict:
    t0 = time.perf_counter()
    strategy = strategy_factory()
    kpi, state = engine.run(strategy, ups)
    wall = time.perf_counter() - t0
    k = kpi.to_dict()
    return {
        "label": label,
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
    }


def summarize(rows: list[dict], name: str) -> dict:
    profits = [r["net_profit"] for r in rows]
    walls = [r["wall_ms"] for r in rows]
    unique_profits = sorted(set(profits))
    return {
        "name": name,
        "n_runs": len(rows),
        "all_identical": len(unique_profits) == 1,
        "n_unique_profits": len(unique_profits),
        "unique_profits": unique_profits[:5] + (["..."] if len(unique_profits) > 5 else []),
        "profit_mean": statistics.mean(profits),
        "profit_min": min(profits),
        "profit_max": max(profits),
        "profit_stdev": statistics.stdev(profits) if len(profits) > 1 else 0.0,
        "wall_ms_mean": statistics.mean(walls),
        "wall_ms_min": min(walls),
        "wall_ms_max": max(walls),
        "wall_ms_total": sum(walls),
    }


def main() -> int:
    print("=" * 76)
    print(" RL-HH Consistency Check — no-UPS (lambda=0)")
    print(f" CP-SAT reference: ${CPSAT_REFERENCE:,.0f}  (seed={CPSAT_SEED}, "
          f"solve={CPSAT_SOLVE_HOURS}h, FEASIBLE, gap=26.4%)")
    print("=" * 76)

    # --- one-time setup ---
    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    engine = SimulationEngine(params)

    agent = DuelingDDQNAgent()
    agent.load_checkpoint(str(CHECKPOINT))
    agent.epsilon = 0.0

    def make_strategy():
        return RLHHStrategy(agent, data, training=False)

    # At lambda=0 the UPS list is always empty regardless of seed,
    # so the input is identical for every run.
    ups_empty_a = generate_ups_events(0.0, 0.0, CPSAT_SEED, SL, roasters)
    assert len(ups_empty_a) == 0, "lambda=0 should yield empty UPS list"

    # --- Test A: same seed, 100 repeats ---
    print(f"\n[Test A] 100 runs with FIXED seed={CPSAT_SEED}, lambda=0  "
          "(pure determinism check)")
    rows_a = []
    for i in range(100):
        r = run_one(engine, make_strategy, ups_empty_a, f"A_run{i:03d}")
        rows_a.append(r)
        if (i+1) % 25 == 0:
            print(f"   completed {i+1}/100  last_profit=${r['net_profit']:,.0f}  "
                  f"wall={r['wall_ms']:.1f}ms")
    sum_a = summarize(rows_a, "Test A: same seed, 100 repeats")

    # --- Test B: 100 different seeds at lambda=0 ---
    print(f"\n[Test B] 100 runs with seeds 900000..900099, lambda=0  "
          "(seed-invariance at no-UPS)")
    rows_b = []
    for i, seed in enumerate(range(900000, 900100)):
        ups = generate_ups_events(0.0, 0.0, seed, SL, roasters)
        assert len(ups) == 0
        r = run_one(engine, make_strategy, ups, f"B_seed{seed}")
        rows_b.append(r)
        if (i+1) % 25 == 0:
            print(f"   completed {i+1}/100  last_profit=${r['net_profit']:,.0f}  "
                  f"wall={r['wall_ms']:.1f}ms")
    sum_b = summarize(rows_b, "Test B: seeds 900000..900099")

    # --- Print summary ---
    def report(s: dict) -> None:
        print()
        print(f"  {s['name']}")
        print(f"  ---")
        print(f"  all 100 results identical:  "
              f"{'YES' if s['all_identical'] else 'NO ('+str(s['n_unique_profits'])+' distinct)'}")
        print(f"  profit:  mean=${s['profit_mean']:,.2f}  "
              f"min=${s['profit_min']:,.2f}  max=${s['profit_max']:,.2f}  "
              f"std=${s['profit_stdev']:,.2f}")
        print(f"  wall:    mean={s['wall_ms_mean']:.1f}ms  "
              f"min={s['wall_ms_min']:.1f}ms  max={s['wall_ms_max']:.1f}ms  "
              f"total={s['wall_ms_total']/1000:.2f}s")
        gap = (CPSAT_REFERENCE - s['profit_mean']) / CPSAT_REFERENCE * 100.0
        print(f"  optimality gap vs CP-SAT (${CPSAT_REFERENCE:,.0f}): {gap:+.2f}%")

    print("\n" + "=" * 76 + "\n SUMMARY\n" + "=" * 76)
    report(sum_a)
    report(sum_b)

    rl_hh_value = sum_a["profit_mean"]
    gap_pct = (CPSAT_REFERENCE - rl_hh_value) / CPSAT_REFERENCE * 100.0
    print()
    print("=" * 76)
    print(" Headline comparison")
    print("=" * 76)
    print(f"   CP-SAT (no-UPS, seed=69, 8h solve):       ${CPSAT_REFERENCE:>10,.0f}")
    print(f"   RL-HH  (no-UPS, 100 runs, ~0.24s each):   ${rl_hh_value:>10,.0f}")
    print(f"   Gap:                                     {gap_pct:+>10.2f}%")
    print(f"   Speed-up:                                 ~{CPSAT_SOLVE_HOURS*3600/sum_a['wall_ms_mean']*1000:>10,.0f}x")

    # --- Save JSON ---
    out_dir = _ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rl_hh_consistency_no_ups.json"
    json_path.write_text(json.dumps({
        "cpsat_reference": {
            "net_profit": CPSAT_REFERENCE,
            "seed": CPSAT_SEED,
            "solve_hours": CPSAT_SOLVE_HOURS,
            "status": "FEASIBLE",
            "gap_pct": 26.422,
            "source_file": "results/method_comparison/20260424_084805/seed_69/cpsat_result.json",
        },
        "test_A_same_seed":  {"summary": sum_a, "runs": rows_a},
        "test_B_diff_seeds": {"summary": sum_b, "runs": rows_b},
    }, indent=2), encoding="utf-8")
    print(f"\n   JSON: {json_path}")

    # --- HTML report ---
    try:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
    except ImportError:
        print("   plotly not installed — skipping HTML")
        return 0

    profits_a = [r["net_profit"] for r in rows_a]
    profits_b = [r["net_profit"] for r in rows_b]
    walls_a = [r["wall_ms"] for r in rows_a]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Test A: Net profit per run (same seed=69, n=100)",
            "Test B: Net profit per run (seeds 900000..099, n=100)",
            "Test A: Wall-time per run (ms)",
            "Comparison: CP-SAT vs RL-HH at no-UPS",
        ),
        horizontal_spacing=0.12, vertical_spacing=0.18,
    )
    fig.add_trace(go.Scatter(x=list(range(1, 101)), y=profits_a, mode="markers",
                             name="Test A profit"), row=1, col=1)
    fig.add_hline(y=CPSAT_REFERENCE, line_dash="dash", line_color="red",
                  annotation_text=f"CP-SAT ${CPSAT_REFERENCE:,.0f}",
                  annotation_position="top right", row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(900000, 900100)), y=profits_b, mode="markers",
                             name="Test B profit"), row=1, col=2)
    fig.add_hline(y=CPSAT_REFERENCE, line_dash="dash", line_color="red",
                  annotation_text=f"CP-SAT ${CPSAT_REFERENCE:,.0f}",
                  annotation_position="top right", row=1, col=2)
    fig.add_trace(go.Scatter(x=list(range(1, 101)), y=walls_a, mode="lines+markers",
                             name="wall ms"), row=2, col=1)
    fig.add_trace(go.Bar(
        x=["CP-SAT (no-UPS)", "RL-HH (no-UPS, mean)"],
        y=[CPSAT_REFERENCE, rl_hh_value],
        text=[f"${CPSAT_REFERENCE:,.0f}", f"${rl_hh_value:,.0f}"],
        textposition="outside",
        marker_color=["#4c8bf5", "#5cb85c"],
    ), row=2, col=2)

    fig.update_xaxes(title_text="run index", row=1, col=1)
    fig.update_yaxes(title_text="net profit ($)", row=1, col=1)
    fig.update_xaxes(title_text="seed", row=1, col=2)
    fig.update_yaxes(title_text="net profit ($)", row=1, col=2)
    fig.update_xaxes(title_text="run index", row=2, col=1)
    fig.update_yaxes(title_text="wall ms", row=2, col=1)
    fig.update_yaxes(title_text="$", row=2, col=2)

    consistency_msg = (
        f"<b>Test A:</b> {'✅ DETERMINISTIC' if sum_a['all_identical'] else '⚠️ NOT identical'} "
        f"({sum_a['n_unique_profits']} unique value{'s' if sum_a['n_unique_profits']>1 else ''})  "
        f"&nbsp;|&nbsp; "
        f"<b>Test B:</b> {'✅ SEED-INVARIANT' if sum_b['all_identical'] else '⚠️ varies with seed'} "
        f"({sum_b['n_unique_profits']} unique value{'s' if sum_b['n_unique_profits']>1 else ''})  "
        f"&nbsp;|&nbsp; "
        f"<b>Optimality gap:</b> {gap_pct:+.2f}%"
    )
    fig.update_layout(
        title=dict(
            text=("RL-HH Consistency at No-UPS (lambda=0)<br>"
                  f"<sub>{consistency_msg}</sub>"),
            x=0.5, xanchor="center",
        ),
        showlegend=False,
        height=800, width=1200,
        template="plotly_white",
        margin=dict(t=120, l=60, r=40, b=60),
    )
    html_path = out_dir / "rl_hh_consistency_no_ups.html"
    html_path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True),
                         encoding="utf-8")
    print(f"   HTML: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
