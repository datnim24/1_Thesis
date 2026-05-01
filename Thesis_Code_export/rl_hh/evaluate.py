"""RL-HH evaluation: single-seed schema-compatible export and aggregate stats.

Loads a trained Dueling DDQN checkpoint and runs greedy episodes via
``SimulationEngine.run()`` (engine-based mode — comparable to baselines).
Writes ``result.json`` (universal schema) + ``report.html`` to
``Results/<YYYYMMDD_HHMMSS>_Eval_RLHH_<RunName>/``.

Usage:
    python -m rl_hh.evaluate --checkpoint Results/<train_dir>/rlhh_best.pt \
        --seed 42 --name SmokeEval
    python -m rl_hh.evaluate --checkpoint <path> --aggregate --n-episodes 100 \
        --base-seed 900000 --name 100Seed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from env.data_loader import load_data
from evaluation.result_schema import create_result, make_run_dir

# Pull DDQNAgent + RLHHStrategy + ToolKit + configs from rl_hh.train
from rl_hh.train import C, DuelingDDQNAgent, RLHHStrategy, ToolKit, WAIT_ACTION_ID


# ============================================================================
# Single-seed export (universal schema)
# ============================================================================

def _batch_to_schedule_entry(batch, refs: dict) -> dict[str, Any]:
    roaster = batch.roaster
    sku = batch.sku
    pipeline = refs.get("R_pipe", {}).get(roaster, "L1")
    dc = refs.get("DC", 3)
    return {
        "batch_id": str(batch.batch_id),
        "job_id": batch.batch_id[0] if batch.is_mto else None,
        "sku": sku,
        "roaster": roaster,
        "start": int(batch.start),
        "end": int(batch.end),
        "pipeline": pipeline,
        "pipeline_start": int(batch.start),
        "pipeline_end": int(batch.start) + dc,
        "output_line": batch.output_line,
        "is_mto": batch.is_mto,
        "setup": "No",
        "status": "completed",
    }


def _cancelled_to_entry(batch, refs: dict) -> dict[str, Any]:
    e = _batch_to_schedule_entry(batch, refs)
    e["status"] = "cancelled"
    return e


def _restock_to_entry(rst) -> dict[str, Any]:
    return {
        "line_id": rst.line_id, "sku": rst.sku,
        "start": int(rst.start), "end": int(rst.end), "qty": int(rst.qty),
    }


def _ups_to_entry(ev) -> dict[str, Any]:
    return {"t": int(ev.t), "roaster_id": ev.roaster_id, "duration": int(ev.duration)}


def _normalize_gc(d: dict) -> dict[str, Any]:
    return {
        f"{k[0]}_{k[1]}" if isinstance(k, tuple) else str(k): v
        for k, v in d.items()
    }


def run_and_export(
    checkpoint_path: str | Path,
    seed: int = 42,
    ups_lambda: float | None = None,
    ups_mu: float | None = None,
    input_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one greedy episode, return universal-schema result dict."""
    data = load_data(input_dir)
    params = data.to_env_params()
    engine = SimulationEngine(params)
    shift_length = int(params["SL"])
    roasters = list(params["roasters"])

    if ups_lambda is None:
        ups_lambda = data.ups_lambda
    if ups_mu is None:
        ups_mu = data.ups_mu

    agent = DuelingDDQNAgent()
    agent.load_checkpoint(checkpoint_path)
    agent.epsilon = 0.0

    strategy = RLHHStrategy(agent, data, training=False)
    ups = generate_ups_events(ups_lambda, ups_mu, seed, shift_length, roasters)

    t0 = time.perf_counter()
    kpi, state = engine.run(strategy, ups)
    elapsed = time.perf_counter() - t0

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

    tool_names = ["PSC_THROUGHPUT", "GC_RESTOCK", "MTO_DEADLINE", "SETUP_AVOID", "WAIT"]
    total_tools = sum(strategy.tool_counts.values()) or 1
    tool_dist = {
        tool_names[tid]: f"{cnt}/{total_tools} ({cnt/total_tools*100:.1f}%)"
        for tid, cnt in sorted(strategy.tool_counts.items())
        if tid < len(tool_names)
    }

    kpi_dict = kpi.to_dict()
    ckpt = Path(checkpoint_path)
    train_started = datetime.fromtimestamp(ckpt.stat().st_mtime).isoformat(timespec="seconds") if ckpt.exists() else None

    return create_result(
        metadata={
            "solver_engine": "simulation",
            "solver_name": "RL-HH (Dueling DDQN)",
            "status": "Completed",
            "solve_time_sec": elapsed,
            "total_compute_ms": elapsed * 1000.0,
            "num_resolves": 0,
            "allow_r3_flex": params.get("allow_r3_flex", True),
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "input_dir": str(data.input_dir),
            "notes": f"Tool selection: {json.dumps(tool_dist)}",
            "model_path": str(ckpt),
            "training_started_at": train_started,
        },
        experiment={
            "lambda_rate": ups_lambda,
            "mu_mean": ups_mu,
            "seed": seed,
            "scenario_label": "ups" if ups else "no_ups",
        },
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
            "gc_init": _normalize_gc(params.get("gc_init", {})),
            "gc_final": _normalize_gc(dict(state.gc_stock)),
        },
        schedule=schedule,
        cancelled_batches=cancelled,
        ups_events=ups_entries,
        restocks=restocks,
        parameters={
            "SL": params["SL"],
            "sigma": params["sigma"],
            "DC": params["DC"],
            "max_rc": params["max_rc"],
            "safety_stock": params["safety_stock"],
            "rc_init": params["rc_init"],
            "restock_duration": params["restock_duration"],
            "restock_qty": params["restock_qty"],
            "roast_time_by_sku": params["roast_time_by_sku"],
            "consume_events": {k: list(v) for k, v in params["consume_events"].items()},
            "gc_capacity": _normalize_gc(params["gc_capacity"]),
            "gc_init": _normalize_gc(params["gc_init"]),
            "feasible_gc_pairs": [f"{k[0]}_{k[1]}" for k in params["gc_capacity"].keys()],
            "sku_revenue": {"PSC": params["rev_psc"], "NDG": params["rev_ndg"], "BUSTA": params["rev_busta"]},
            "c_tard": params["c_tard"],
            "c_stock": params["c_stock"],
            "c_idle": params["c_idle"],
            "c_over": params["c_over"],
            "c_setup": params["c_setup"],
        },
    )


# ============================================================================
# Aggregate evaluation (n_episodes via SimulationEngine + RLHHStrategy)
# ============================================================================

def evaluate_engine(
    agent: "DuelingDDQNAgent",
    data,
    n_episodes: int = 100,
    ups_lambda: float | None = None,
    ups_mu: float | None = None,
    base_seed: int = 900_000,
) -> dict:
    if ups_lambda is None:
        ups_lambda = data.ups_lambda
    if ups_mu is None:
        ups_mu = data.ups_mu
    params = data.to_env_params() if hasattr(data, "to_env_params") else data
    engine = SimulationEngine(params)
    shift_length = int(params["SL"])
    roasters = list(params["roasters"])

    profits: list[float] = []
    kpis: list[dict] = []
    tool_counts_total: dict[int, int] = defaultdict(int)

    for ep in range(n_episodes):
        strategy = RLHHStrategy(agent, data, training=False)
        ups = generate_ups_events(ups_lambda, ups_mu, base_seed + ep, shift_length, roasters)
        kpi, state = engine.run(strategy, ups)

        profits.append(float(kpi.net_profit()))
        kpis.append(kpi.to_dict())
        for tid, cnt in strategy.tool_counts.items():
            tool_counts_total[tid] += cnt

    total_tool_selections = max(1, sum(tool_counts_total.values()))
    tool_dist = {
        C.TOOL_NAMES[tid]: round(cnt / total_tool_selections, 4)
        for tid, cnt in sorted(tool_counts_total.items())
        if tid < len(C.TOOL_NAMES)
    }
    return {
        "method": "RL-HH (Dueling DDQN)",
        "n_episodes": n_episodes,
        "profit_mean": round(float(np.mean(profits)), 2),
        "profit_std": round(float(np.std(profits)), 2),
        "profit_min": round(float(np.min(profits)), 2),
        "profit_max": round(float(np.max(profits)), 2),
        "psc_count_mean": round(float(np.mean([k["psc_count"] for k in kpis])), 1),
        "ndg_count_mean": round(float(np.mean([k.get("ndg_count", 0) for k in kpis])), 1),
        "busta_count_mean": round(float(np.mean([k.get("busta_count", 0) for k in kpis])), 1),
        "stockout_cost_mean": round(float(np.mean([k["stockout_cost"] for k in kpis])), 2),
        "setup_cost_mean": round(float(np.mean([k["setup_cost"] for k in kpis])), 2),
        "tard_cost_mean": round(float(np.mean([k["tard_cost"] for k in kpis])), 2),
        "idle_cost_mean": round(float(np.mean([k["idle_cost"] for k in kpis])), 2),
        "restock_count_mean": round(float(np.mean([k["restock_count"] for k in kpis])), 1),
        "tool_distribution": tool_dist,
        "profits_per_seed": profits,
    }


# ============================================================================
# CLI
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained RL-HH (engine mode).")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--name", required=True, help="Run name (output folder suffix)")
    parser.add_argument("--seed", type=int, default=42, help="Single-seed evaluation seed")
    parser.add_argument("--ups-lambda", type=float, default=None)
    parser.add_argument("--ups-mu", type=float, default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--aggregate", action="store_true",
                        help="Run aggregated multi-seed evaluation instead of single-seed export.")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=900_000)
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report.")
    args = parser.parse_args(argv)

    out_dir = make_run_dir("Eval_RLHH", args.name)
    print(f"[rl_hh.evaluate] Output -> {out_dir}")

    if args.aggregate:
        data = load_data(args.input_dir)
        agent = DuelingDDQNAgent()
        agent.load_checkpoint(args.checkpoint)
        agent.epsilon = 0.0
        results = evaluate_engine(
            agent, data, args.n_episodes,
            args.ups_lambda, args.ups_mu, args.base_seed,
        )
        out_path = out_dir / "aggregate.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  Mean profit: ${results['profit_mean']:,.0f} +/- {results['profit_std']:,.0f}")
        print(f"  Aggregate JSON: {out_path}")
        return 0

    result = run_and_export(
        args.checkpoint, args.seed, args.ups_lambda, args.ups_mu, args.input_dir,
    )
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  result.json: {json_path}")

    if not args.no_report:
        try:
            from evaluation.plot_result import _build_html
            html = _build_html(result, compare=None, offline=True)
            (out_dir / "report.html").write_text(html, encoding="utf-8")
            print(f"  report.html: {out_dir / 'report.html'}")
        except Exception as exc:
            print(f"  Report generation failed: {exc}", file=sys.stderr)

    print(f"  Net profit: ${result['kpi'].get('net_profit', 0):,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
