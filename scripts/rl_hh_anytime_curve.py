"""Time RL-HH inference end-to-end on a single seed and render the schedule
via plot_result._build_html. Companion to scripts/cpsat_anytime_curve.py.

Outputs to ``output/`` so the two anytime artefacts sit side-by-side for the
V4Evaluation.md §10.2 fair-compute comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RL-HH timing + schedule rendering")
    parser.add_argument("--checkpoint", default="rl_hh/outputs/rlhh_cycle3_best.pt")
    parser.add_argument("--seed", type=int, default=900046)
    parser.add_argument("--lambda-mult", type=float, default=1.0)
    parser.add_argument("--mu-mult", type=float, default=1.0)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0 — imports
    t0 = time.perf_counter()
    from env.simulation_engine import SimulationEngine
    from env.ups_generator import generate_ups_events
    from PPOmask.Engine.data_loader import load_data
    from result_schema import create_result
    from test_rl_hh.meta_agent import DuelingDDQNAgent
    from test_rl_hh.rl_hh_strategy import RLHHStrategy
    from test_rl_hh.export_result import (
        _batch_to_schedule_entry, _cancelled_to_entry, _restock_to_entry,
        _ups_to_entry, _normalize_gc_dict,
    )
    from plot_result import _build_html
    t_imports = time.perf_counter() - t0
    print(f"  [phase 0] Imports completed in {t_imports*1000:.1f} ms")

    # Phase 1 — load Input_data
    t1 = time.perf_counter()
    data = load_data()
    params = data.to_env_params()
    t_load_data = time.perf_counter() - t1
    print(f"  [phase 1] Input_data loaded in {t_load_data*1000:.1f} ms")

    # Phase 2 — engine
    t2 = time.perf_counter()
    engine = SimulationEngine(params)
    t_engine = time.perf_counter() - t2
    print(f"  [phase 2] SimulationEngine built in {t_engine*1000:.1f} ms")

    # Phase 3 — checkpoint
    t3 = time.perf_counter()
    agent = DuelingDDQNAgent()
    agent.load_checkpoint(args.checkpoint)
    agent.epsilon = 0.0
    t_ckpt = time.perf_counter() - t3
    print(f"  [phase 3] Checkpoint loaded ({args.checkpoint}) in {t_ckpt*1000:.1f} ms")

    # Phase 4 — UPS realization
    t4 = time.perf_counter()
    ups_lambda = float(data.ups_lambda) * args.lambda_mult
    ups_mu = float(data.ups_mu) * args.mu_mult
    ups = generate_ups_events(
        ups_lambda, ups_mu, args.seed,
        int(params["SL"]), list(params["roasters"]),
    )
    t_ups = time.perf_counter() - t4
    print(f"  [phase 4] UPS realization for seed={args.seed}: {len(ups)} events ({t_ups*1000:.1f} ms)")

    # Phase 5 — strategy bind
    t5 = time.perf_counter()
    strategy = RLHHStrategy(agent, data, training=False)
    t_strat = time.perf_counter() - t5
    print(f"  [phase 5] RLHHStrategy bound in {t_strat*1000:.1f} ms")

    # Phase 6 — full episode
    print()
    print("  ==== Running full 480-slot episode ====")
    t6 = time.perf_counter()
    kpi, state = engine.run(strategy, ups)
    t_run = time.perf_counter() - t6
    total_decisions = sum(strategy.tool_counts.values())
    ms_per_decision = (t_run / total_decisions) * 1000.0 if total_decisions else 0.0
    kpi_dict = kpi.to_dict()
    print(f"  [phase 6] Episode finished in {t_run*1000:.1f} ms ({t_run:.3f}s)")
    print(f"              Tool decisions made: {total_decisions}  -> {ms_per_decision:.3f} ms/decision")
    print(f"              Net profit: ${float(kpi.net_profit()):,.0f}")
    print(f"              PSC: {kpi_dict['psc_count']} | NDG: {kpi_dict['ndg_count']} | BUSTA: {kpi_dict['busta_count']}")
    print(f"              Tardiness cost: ${float(kpi_dict['tard_cost']):,.0f}")
    print(f"              Setup events: {kpi_dict['setup_events']}")
    print(f"              Idle min: {kpi_dict['idle_min']}")
    print(f"              Restocks: {kpi_dict['restock_count']}")

    # Phase 7 — result dict
    t7 = time.perf_counter()
    schedule = [_batch_to_schedule_entry(b, params) for b in state.completed_batches]
    cancelled = [_cancelled_to_entry(b, params) for b in state.cancelled_batches]
    restocks = [_restock_to_entry(r) for r in state.completed_restocks]
    ups_entries = [_ups_to_entry(e) for e in state.ups_events_fired]
    last_sku = dict(params.get("roaster_initial_sku", {}))
    for entry in sorted(schedule, key=lambda e: (e["roaster"], e["start"])):
        rid = entry["roaster"]
        if entry["sku"] != last_sku.get(rid, "PSC"):
            entry["setup"] = "Yes"
        last_sku[rid] = entry["sku"]

    # Tool distribution string for notes
    tool_names = ["PSC_THROUGHPUT", "GC_RESTOCK", "MTO_DEADLINE", "SETUP_AVOID", "WAIT"]
    tool_dist = {
        tool_names[tid]: f"{cnt}/{total_decisions} ({cnt / max(1, total_decisions) * 100:.1f}%)"
        for tid, cnt in sorted(strategy.tool_counts.items())
        if tid < len(tool_names)
    }
    notes = (
        f"Tool decisions: {total_decisions} | inference: {t_run*1000:.1f}ms total, "
        f"{ms_per_decision:.3f}ms/decision | tool dist: {json.dumps(tool_dist)}"
    )

    result = create_result(
        metadata={
            "solver_engine": "simulation",
            "solver_name": "RL-HH (Dueling DDQN)",
            "status": "Completed",
            "solve_time_sec": t_run,
            "total_compute_ms": t_run * 1000.0,
            "num_resolves": 0,
            "allow_r3_flex": params.get("allow_r3_flex", True),
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "input_dir": str(data.input_dir),
            "notes": notes,
            "model_path": args.checkpoint,
        },
        experiment={"lambda_rate": ups_lambda, "mu_mean": ups_mu,
                    "seed": args.seed, "scenario_label": "ups"},
        kpi={
            "net_profit": kpi_dict["net_profit"],
            "total_revenue": kpi_dict["total_revenue"],
            "total_costs": kpi_dict["total_costs"],
            "psc_count": kpi_dict["psc_count"],
            "ndg_count": kpi_dict["ndg_count"],
            "busta_count": kpi_dict["busta_count"],
            "revenue_psc": kpi_dict["revenue_psc"],
            "revenue_ndg": kpi_dict["revenue_ndg"],
            "revenue_busta": kpi_dict["revenue_busta"],
            "tardiness_min": kpi_dict["tardiness_min"],
            "tard_cost": kpi_dict["tard_cost"],
            "setup_events": kpi_dict["setup_events"],
            "setup_cost": kpi_dict["setup_cost"],
            "stockout_events": kpi_dict["stockout_events"],
            "stockout_duration": kpi_dict["stockout_duration"],
            "stockout_cost": kpi_dict["stockout_cost"],
            "idle_min": kpi_dict["idle_min"],
            "idle_cost": kpi_dict["idle_cost"],
            "over_min": kpi_dict["over_min"],
            "over_cost": kpi_dict["over_cost"],
            "restock_count": kpi_dict["restock_count"],
            "gc_init": _normalize_gc_dict(params.get("gc_init", {})),
            "gc_final": _normalize_gc_dict(dict(state.gc_stock)),
        },
        schedule=schedule, cancelled_batches=cancelled,
        ups_events=ups_entries, restocks=restocks,
        parameters={
            "SL": params["SL"], "sigma": params["sigma"], "DC": params["DC"],
            "max_rc": params["max_rc"], "safety_stock": params["safety_stock"],
            "rc_init": params["rc_init"],
            "restock_duration": params["restock_duration"],
            "restock_qty": params["restock_qty"],
            "roast_time_by_sku": params["roast_time_by_sku"],
            "consume_events": {k: list(v) for k, v in params["consume_events"].items()},
            "gc_capacity": _normalize_gc_dict(params["gc_capacity"]),
            "gc_init": _normalize_gc_dict(params["gc_init"]),
            "feasible_gc_pairs": [f"{k[0]}_{k[1]}" for k in params["gc_capacity"].keys()],
            "sku_revenue": {"PSC": params["rev_psc"],
                            "NDG": params["rev_ndg"],
                            "BUSTA": params["rev_busta"]},
            "c_tard": params["c_tard"], "c_stock": params["c_stock"],
            "c_idle": params["c_idle"], "c_over": params["c_over"],
            "c_setup": params["c_setup"],
        },
    )
    t_result = time.perf_counter() - t7
    print(f"  [phase 7] result dict built in {t_result*1000:.1f} ms")

    # Phase 8 — render HTML
    t8 = time.perf_counter()
    html = _build_html(result, None, offline=True)
    t_render = time.perf_counter() - t8
    print(f"  [phase 8] plot_result._build_html rendered in {t_render*1000:.1f} ms")

    # Phase 9 — write
    t9 = time.perf_counter()
    json_path = out_dir / f"rl_hh_seed{args.seed}_result.json"
    html_path = out_dir / f"rl_hh_seed{args.seed}_schedule.html"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    t_write = time.perf_counter() - t9
    print(f"  [phase 9] JSON + HTML written in {t_write*1000:.1f} ms")

    total_wall = time.perf_counter() - t0
    print()
    print("=" * 70)
    print(f"  RL-HH evaluation timing summary (seed={args.seed})")
    print("=" * 70)
    phases = [
        ("Imports", t_imports * 1000),
        ("Input_data load", t_load_data * 1000),
        ("Engine build", t_engine * 1000),
        ("Checkpoint load", t_ckpt * 1000),
        ("UPS gen", t_ups * 1000),
        ("Strategy bind", t_strat * 1000),
        ("** Episode (inference) **", t_run * 1000),
        ("Result dict build", t_result * 1000),
        ("plot_result render", t_render * 1000),
        ("Disk write", t_write * 1000),
    ]
    for name, ms in phases:
        bar = "#" * max(1, int(ms / 30.0)) if ms > 0.5 else ""
        print(f"  {name:<28}  {ms:>9.1f} ms  {bar}")
    print("-" * 70)
    print(f"  {'Total wall (incl. imports)':<28}  {total_wall*1000:>9.1f} ms ({total_wall:.2f}s)")
    print(f"  {'Per-decision time':<28}  {ms_per_decision:>9.3f} ms / tool selection")
    print("=" * 70)

    # Save timing JSON for the report
    timing_path = out_dir / f"rl_hh_seed{args.seed}_timing.json"
    timing_path.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "ups_lambda": ups_lambda,
        "ups_mu": ups_mu,
        "n_ups_events": len(ups),
        "phases_ms": {name: ms for name, ms in phases},
        "total_wall_ms": total_wall * 1000,
        "total_decisions": total_decisions,
        "ms_per_decision": ms_per_decision,
        "kpi": {
            "net_profit": float(kpi.net_profit()),
            "psc_count": kpi_dict["psc_count"],
            "ndg_count": kpi_dict["ndg_count"],
            "busta_count": kpi_dict["busta_count"],
            "tard_cost": float(kpi_dict["tard_cost"]),
            "setup_events": kpi_dict["setup_events"],
            "idle_min": kpi_dict["idle_min"],
            "restock_count": kpi_dict["restock_count"],
        },
        "tool_distribution": tool_dist,
    }, indent=2), encoding="utf-8")

    print(f"\n  JSON:   {json_path.resolve()}")
    print(f"  HTML:   {html_path.resolve()}")
    print(f"  Timing: {timing_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
