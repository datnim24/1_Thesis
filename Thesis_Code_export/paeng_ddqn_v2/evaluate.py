"""Paeng DDQN v2 evaluation: single-seed schema-export and multi-seed aggregate.

Loads a trained PaengAgentV2 checkpoint and runs greedy episodes via
``SimulationEngine.run()``. Writes ``result.json`` (universal schema) +
``report.html`` (single-seed) or ``aggregate.json`` (multi-seed) to
``Results/<YYYYMMDD_HHMMSS>_Eval_PaengDDQNv2_<RunName>/``.

Usage:
    python -m paeng_ddqn_v2.evaluate --checkpoint <path> --seed 42 --name SmokeEval
    python -m paeng_ddqn_v2.evaluate --checkpoint <path> --aggregate \\
        --n-seeds 100 --base-seed 900000 --name 100Seed
"""

from __future__ import annotations

import argparse
import json
import sys
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

# Pull agent + strategy from the consolidated train module
from paeng_ddqn_v2.train import PaengAgentV2, PaengStrategyV2


def _json_safe(obj):
    """Coerce tuple keys + non-JSON-native values for json.dump."""
    if isinstance(obj, dict):
        return {
            ("_".join(str(p) for p in k) if isinstance(k, tuple)
             else str(k) if not isinstance(k, (str, int, float, bool)) and k is not None
             else k): _json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        try:
            return sorted(obj)
        except TypeError:
            return [_json_safe(x) for x in obj]
    return obj


def _build_metadata(checkpoint_path: Path) -> dict:
    metadata = {
        "solver_engine": "paeng_ddqn_v2",
        "solver_name": "Paeng DDQN v2 (3,35 state, faithful to 2021 paper + domain extension)",
        "status": "Completed",
        "input_dir": "Input_data",
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_path": str(checkpoint_path),
    }
    summary_path = Path(checkpoint_path).parent / "training_summary.json"
    if summary_path.exists():
        try:
            tsum = json.loads(summary_path.read_text(encoding="utf-8"))
            metadata["notes"] = (
                f"Paeng DDQN v2: {tsum.get('episodes', '?')} episodes, "
                f"best train profit ${tsum.get('best_profit', '?')}, "
                f"final epsilon {tsum.get('final_epsilon', '?')}"
            )
            metadata["training_run_dir"] = str(Path(checkpoint_path).parent)
        except Exception:
            pass
    return metadata


def _batch_to_entry(batch, params: dict, status: str = "completed",
                    cancel_time: int | None = None) -> dict[str, Any]:
    roaster = batch.roaster
    sku = batch.sku
    pipeline = params.get("R_pipe", {}).get(roaster, "L1")
    dc = params.get("DC", 3)
    entry = {
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
        "status": status,
    }
    if cancel_time is not None:
        entry["cancel_time"] = int(cancel_time)
    return entry


def _match_cancel_time(batch, ups_events, used: set[tuple[int, str, int]]) -> int | None:
    """Find the UPS event that cancelled this batch (best-effort)."""
    for ev in ups_events:
        key = (int(ev.t), ev.roaster_id, int(ev.duration))
        if key in used:
            continue
        if ev.roaster_id == batch.roaster and int(batch.start) <= int(ev.t) < int(batch.end):
            used.add(key)
            return int(ev.t)
    return None


def evaluate_one_seed(
    checkpoint_path: Path,
    seed: int,
    lambda_mult: float = 1.0,
    mu_mult: float = 1.0,
) -> tuple[dict, dict, object]:
    """Run a single-seed greedy episode; return (universal-schema result, kpi_dict, state)."""
    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    base_lam = float(data.ups_lambda)
    base_mu = float(data.ups_mu)
    lam = base_lam * lambda_mult
    mu = base_mu * mu_mult

    agent = PaengAgentV2.from_checkpoint(checkpoint_path)
    agent.epsilon = 0.0  # greedy
    strategy = PaengStrategyV2(agent, data, training=False, params=params)

    engine = SimulationEngine(params)
    ups_events = generate_ups_events(lam, mu, seed, SL, roasters)
    kpi, state = engine.run(strategy, ups_events)

    used: set[tuple[int, str, int]] = set()
    metadata = _build_metadata(Path(checkpoint_path))
    metadata["lambda_mult"] = round(lambda_mult, 4)
    metadata["mu_mult"] = round(mu_mult, 4)
    metadata["ups_lambda_used"] = round(lam, 4)
    metadata["ups_mu_used"] = round(mu, 4)
    metadata["notes"] = (
        (metadata.get("notes", "") + "  action_dist=" + json.dumps(dict(strategy.action_counts)))
    ).strip()

    result = create_result(
        metadata=metadata,
        experiment={
            "lambda_rate": lam, "mu_mean": mu, "seed": seed,
            "scenario_label": "paeng_v2_single",
        },
        kpi=kpi.to_dict(),
        schedule=[_batch_to_entry(b, params) for b in state.completed_batches],
        cancelled_batches=[
            _batch_to_entry(b, params, "cancelled", _match_cancel_time(b, ups_events, used))
            for b in state.cancelled_batches
        ],
        ups_events=[
            {"t": int(e.t), "roaster_id": e.roaster_id, "duration": int(e.duration)}
            for e in ups_events
        ],
        parameters=params,
        restocks=[
            {"line_id": rst.line_id, "sku": rst.sku,
             "start": int(rst.start), "end": int(rst.end), "qty": int(rst.qty)}
            for rst in state.completed_restocks
        ],
    )
    return result, kpi.to_dict(), state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained Paeng DDQN v2 checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to paeng_v2_best.pt")
    parser.add_argument("--name", required=True, help="Run name (output folder suffix)")
    parser.add_argument("--seed", type=int, default=42, help="Single-seed evaluation seed")
    parser.add_argument("--aggregate", action="store_true",
                        help="Run multi-seed aggregate instead of single-seed export.")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=900_000)
    parser.add_argument("--lambda-mult", type=float, default=1.0)
    parser.add_argument("--mu-mult", type=float, default=1.0)
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report.")
    args = parser.parse_args(argv)

    out_dir = make_run_dir("Eval_PaengDDQNv2", args.name)
    print(f"[paeng_v2.evaluate] Output -> {out_dir}")

    if args.aggregate:
        profits: list[float] = []
        per_seed: list[dict] = []
        for i in range(args.n_seeds):
            seed = args.base_seed + i
            _, kpi_dict, _ = evaluate_one_seed(
                Path(args.checkpoint), seed, args.lambda_mult, args.mu_mult,
            )
            p = float(kpi_dict["net_profit"])
            profits.append(p)
            per_seed.append({"seed": seed, "net_profit": p, **kpi_dict})
            if (i + 1) % 20 == 0:
                print(f"  [{i + 1}/{args.n_seeds}] seeds done")

        arr = np.array(profits)
        aggregate = {
            "package": "paeng_ddqn_v2",
            "checkpoint": str(args.checkpoint),
            "n_seeds": args.n_seeds,
            "base_seed": args.base_seed,
            "lambda_mult": args.lambda_mult,
            "mu_mult": args.mu_mult,
            "profit_mean": float(arr.mean()),
            "profit_std": float(arr.std()),
            "profit_median": float(np.median(arr)),
            "profit_p25": float(np.percentile(arr, 25)),
            "profit_p75": float(np.percentile(arr, 75)),
            "profit_min": float(arr.min()),
            "profit_max": float(arr.max()),
            "per_seed": per_seed,
        }
        out_path = out_dir / "aggregate.json"
        out_path.write_text(json.dumps(_json_safe(aggregate), indent=2), encoding="utf-8")
        print(f"  Mean ${aggregate['profit_mean']:,.0f} +/- ${aggregate['profit_std']:,.0f}")
        print(f"  Median ${aggregate['profit_median']:,.0f}, [min ${aggregate['profit_min']:,.0f}, max ${aggregate['profit_max']:,.0f}]")
        print(f"  Saved: {out_path}")
        return 0

    # Single-seed mode
    result, kpi_dict, _state = evaluate_one_seed(
        Path(args.checkpoint), args.seed, args.lambda_mult, args.mu_mult,
    )
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    print(f"  result.json: {json_path}")

    if not args.no_report:
        try:
            from evaluation.plot_result import _build_html
            html = _build_html(result, None, offline=True)
            (out_dir / "report.html").write_text(html, encoding="utf-8")
            print(f"  report.html: {out_dir / 'report.html'}")
        except Exception as exc:
            print(f"  Report generation failed: {exc}", file=sys.stderr)

    print(f"  Net profit: ${kpi_dict.get('net_profit', 0):,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
