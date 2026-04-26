"""Export an RL-HH evaluation run as a plot_result.py-compatible JSON.

Runs a single greedy episode via SimulationEngine.run() and converts the
full KPI + state into the universal result schema format.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data
from result_schema import create_result

from .meta_agent import DuelingDDQNAgent
from .rl_hh_strategy import RLHHStrategy


def _batch_to_schedule_entry(batch, refs: dict) -> dict[str, Any]:
    """Convert a BatchRecord to a schedule entry dict."""
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
    entry = _batch_to_schedule_entry(batch, refs)
    entry["status"] = "cancelled"
    return entry


def _restock_to_entry(rst) -> dict[str, Any]:
    return {
        "line_id": rst.line_id,
        "sku": rst.sku,
        "start": int(rst.start),
        "end": int(rst.end),
        "qty": int(rst.qty),
    }


def _ups_to_entry(ev) -> dict[str, Any]:
    return {
        "t": int(ev.t),
        "roaster_id": ev.roaster_id,
        "duration": int(ev.duration),
    }


def _normalize_gc_dict(d: dict) -> dict[str, Any]:
    """Convert tuple-keyed GC dicts to string-keyed for JSON."""
    return {
        f"{k[0]}_{k[1]}" if isinstance(k, tuple) else str(k): v
        for k, v in d.items()
    }


def run_and_export(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    ups_lambda: float | None = None,
    ups_mu: float | None = None,
    seed: int = 42,
    input_dir: str | Path | None = None,
    training_started_at: str | None = None,
) -> dict[str, Any]:
    """Run one greedy episode and export as universal result JSON.

    UPS parameters default to whatever the input data specifies.
    Pass explicit values to override.
    """

    data = load_data(input_dir)
    params = data.to_env_params()
    engine = SimulationEngine(params)
    shift_length = int(params["SL"])
    roasters = list(params["roasters"])

    # Respect input data UPS settings unless explicitly overridden
    if ups_lambda is None:
        ups_lambda = data.ups_lambda
    if ups_mu is None:
        ups_mu = data.ups_mu

    # Load agent
    agent = DuelingDDQNAgent()
    agent.load_checkpoint(checkpoint_path)
    agent.epsilon = 0.0

    strategy = RLHHStrategy(agent, data, training=False)
    ups = generate_ups_events(ups_lambda, ups_mu, seed, shift_length, roasters)

    t0 = time.perf_counter()
    kpi, state = engine.run(strategy, ups)
    elapsed = time.perf_counter() - t0

    # Build schedule entries
    schedule = [_batch_to_schedule_entry(b, params) for b in state.completed_batches]
    cancelled = [_cancelled_to_entry(b, params) for b in state.cancelled_batches]
    restocks = [_restock_to_entry(r) for r in state.completed_restocks]
    ups_entries = [_ups_to_entry(e) for e in state.ups_events_fired]

    # Infer setup from schedule
    last_sku = dict(params.get("roaster_initial_sku", {}))
    for entry in sorted(schedule, key=lambda e: (e["roaster"], e["start"])):
        rid = entry["roaster"]
        if entry["sku"] != last_sku.get(rid, "PSC"):
            entry["setup"] = "Yes"
        last_sku[rid] = entry["sku"]

    # Tool distribution for notes
    tool_names = ["PSC_THROUGHPUT", "GC_RESTOCK", "MTO_DEADLINE", "SETUP_AVOID", "WAIT"]
    total_tools = sum(strategy.tool_counts.values()) or 1
    tool_dist = {
        tool_names[tid]: f"{cnt}/{total_tools} ({cnt/total_tools*100:.1f}%)"
        for tid, cnt in sorted(strategy.tool_counts.items())
        if tid < len(tool_names)
    }

    kpi_dict = kpi.to_dict()

    _ckpt_path = Path(checkpoint_path)
    _train_started = training_started_at
    if _train_started is None and _ckpt_path.exists():
        _train_started = datetime.fromtimestamp(_ckpt_path.stat().st_mtime).isoformat(timespec="seconds")

    result = create_result(
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
            "model_path": str(_ckpt_path),
            "training_started_at": _train_started,
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
            "gc_init": _normalize_gc_dict(params.get("gc_init", {})),
            "gc_final": _normalize_gc_dict(dict(state.gc_stock)),
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
            "gc_capacity": _normalize_gc_dict(params["gc_capacity"]),
            "gc_init": _normalize_gc_dict(params["gc_init"]),
            "feasible_gc_pairs": [f"{k[0]}_{k[1]}" for k in params["gc_capacity"].keys()],
            "sku_revenue": {"PSC": params["rev_psc"], "NDG": params["rev_ndg"], "BUSTA": params["rev_busta"]},
            "c_tard": params["c_tard"],
            "c_stock": params["c_stock"],
            "c_idle": params["c_idle"],
            "c_over": params["c_over"],
            "c_setup": params["c_setup"],
        },
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {out}")

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export RL-HH result for plot_result.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="test_rl_hh/outputs/rlhh_result.json")
    parser.add_argument("--ups-lambda", type=float, default=None, help="Override UPS lambda (default: from input data)")
    parser.add_argument("--ups-mu", type=float, default=None, help="Override UPS mu (default: from input data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dir", default=None)
    args = parser.parse_args()
    run_and_export(args.checkpoint, args.output, args.ups_lambda, args.ups_mu, args.seed, args.input_dir)


if __name__ == "__main__":
    main()
