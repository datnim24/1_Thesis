"""Q-learning evaluation: single-seed runs and report rendering.

Combines q_learning_run + export_for_plot_result. Loads a trained Q-table
and runs a single-seed simulation, then writes ``result.json`` + ``report.html``
inside ``Results/<YYYYMMDD_HHMMSS>_Eval_QLearning_<RunName>/``.

Usage:
    python -m q_learning.evaluate --q-table <path> --seed 42 --name SmokeEval
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.data_bridge import get_sim_params
from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from evaluation.plot_result import _build_html
from evaluation.result_schema import create_result, make_run_dir

from q_learning.train import QStrategy, load_q_table

# ============================================================================
# SINGLE-SEED EVALUATION
# ============================================================================


_QL_DIR = Path(__file__).resolve().parent
_RESULTS_ROOT = _PROJECT_ROOT / "Results"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_latest_result() -> tuple[Path | None, Path | None, dict | None]:
    """Return (q_table_path, log_path, meta) for the most recent QLearning run."""
    if _RESULTS_ROOT.is_dir():
        folders = sorted(
            [d for d in _RESULTS_ROOT.iterdir() if d.is_dir() and "_QLearning_" in d.name],
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
# Result-dict assembly (schema-compatible with plot_result._build_html)
# ---------------------------------------------------------------------------

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


def _build_metadata(q_path: Path, meta: dict | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "solver_engine": "simulation",
        "solver_name": "Q-Learning",
        "status": "Completed",
        "input_dir": "Input_data",
        "timestamp": datetime.now().isoformat(),
        "model_path": str(q_path),
        "training_run_dir": str(q_path.parent),
    }
    if meta:
        if meta.get("timestamp"):
            metadata["training_started_at"] = meta["timestamp"]
        note_bits: list[str] = []
        for key, label in (
            ("episodes", "episodes"),
            ("alpha", "alpha"),
            ("gamma", "gamma"),
            ("final_avg_profit_1000", "final_avg_1000"),
            ("q_table_entries", "q_entries"),
        ):
            if meta.get(key) is not None:
                note_bits.append(f"{label}={meta[key]}")
        if note_bits:
            metadata["notes"] = "Q-Learning training: " + ", ".join(note_bits)
    return metadata


def _assemble_result(kpi, state, ups_events, params, lam, mu, seed, q_path, meta) -> dict[str, Any]:
    used_events: set[tuple[int, str, int]] = set()
    return create_result(
        metadata=_build_metadata(q_path, meta),
        experiment={"lambda_rate": lam, "mu_mean": mu, "seed": seed, "scenario_label": "ql_single"},
        kpi=kpi.to_dict(),
        schedule=[_batch_to_result_entry(b, params) for b in state.completed_batches],
        cancelled_batches=[
            _batch_to_result_entry(b, params, "cancelled", _match_cancel_time(b, ups_events, used_events))
            for b in state.cancelled_batches
        ],
        ups_events=[{"t": int(e.t), "roaster_id": e.roaster_id, "duration": int(e.duration)} for e in ups_events],
        parameters=params,
        restocks=[
            {"line_id": rst.line_id, "sku": rst.sku, "start": int(rst.start), "end": int(rst.end), "qty": int(rst.qty)}
            for rst in state.completed_restocks
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-episode Q-learning evaluation — renders via plot_result._build_html.",
    )
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Path to Q-table .pkl (default: latest under ql_results/).")
    parser.add_argument("--seed", type=int, default=42, help="UPS realization seed (default: 42).")
    parser.add_argument("--ups-lambda", type=float, default=None,
                        help="Override UPS lambda (default: Input_data value).")
    parser.add_argument("--ups-mu", type=float, default=None,
                        help="Override UPS mu (default: Input_data value).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: <run_dir>/<run_dir>.html).")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open in browser.")
    args = parser.parse_args(argv)

    params = get_sim_params()
    q_path, log_path, meta, result_dir = _resolve_file_arg(args.file)
    if q_path is None:
        return 1

    q_table = load_q_table(str(q_path))
    print(f"Loaded Q-table: {q_path}")
    print(f"  {len(q_table):,} entries")
    if meta:
        print(f"  Training: {meta.get('episodes', '-')} ep, alpha={meta.get('alpha', '-')}, "
              f"gamma={meta.get('gamma', '-')}, name={meta.get('name', '-')}")

    lam = float(args.ups_lambda) if args.ups_lambda is not None else float(params.get("ups_lambda", 0.0))
    mu = float(args.ups_mu) if args.ups_mu is not None else float(params.get("ups_mu", 0.0))
    ups_events = generate_ups_events(lam, mu, args.seed)

    print(f"\nRunning single episode: seed={args.seed}, lambda={lam}, mu={mu}")
    engine = SimulationEngine(params)
    strategy = QStrategy(params, q_table=q_table)
    kpi, state = engine.run(strategy, ups_events)
    net_profit = float(kpi.net_profit())
    print(f"  Net profit: ${net_profit:,.0f}")
    print(f"  Completed batches: {len(state.completed_batches)}  |  "
          f"Cancelled: {len(state.cancelled_batches)}  |  Restocks: {len(state.completed_restocks)}")

    result = _assemble_result(kpi, state, ups_events, params, lam, mu, args.seed, q_path, meta)

    training_info: dict[str, Any] | None = None
    if log_path and Path(log_path).exists():
        training_info = {
            "log_path": str(log_path),
            "epsilon_start": (meta or {}).get("epsilon_start"),
            "epsilon_end": (meta or {}).get("epsilon_end"),
        }

    html = _build_html(result, None, offline=True, training_info=training_info)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = _PROJECT_ROOT / out_path
    else:
        report_name = result_dir.name if result_dir != _QL_DIR else "q_learning_report"
        out_path = result_dir / f"{report_name}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nInteractive HTML report: {out_path}")

    # Also persist the universal-schema result.json next to the report.
    result_json_path = result_dir / "result.json" if result_dir != _QL_DIR else out_path.with_suffix(".json")
    result_json_path.write_text(
        json.dumps(_json_safe(result), indent=2),
        encoding="utf-8",
    )
    print(f"Schema-compatible JSON: {result_json_path}")
    if not args.no_open:
        try:
            webbrowser.open(out_path.resolve().as_uri())
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ============================================================================
# REPORT EXPORT (legacy export_for_plot_result entry point)
# ============================================================================


def _json_safe(obj):
    """Recursively coerce non-JSON-native types (sets, tuples, tuple-keyed
    dicts) so result_schema dicts serialize cleanly. Same pattern as
    ``plot_result._json_safe``.
    """
    if isinstance(obj, dict):
        return {
            ("_".join(str(p) for p in k) if isinstance(k, tuple) else str(k) if not isinstance(k, (str, int, float, bool)) and k is not None else k): _json_safe(v)
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


def main_export(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Q-learning single-seed evaluation exported as universal-schema JSON.",
    )
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to Q-table .pkl (default: latest under ql_results/).")
    parser.add_argument("--seed", type=int, default=42, help="UPS realization seed.")
    parser.add_argument("--ups-lambda", type=float, default=None,
                        help="Override UPS lambda (default: Input_data value).")
    parser.add_argument("--ups-mu", type=float, default=None,
                        help="Override UPS mu (default: Input_data value).")
    parser.add_argument("--lambda-mult", type=float, default=1.0,
                        help="Multiplier on Input_data UPS lambda (overrides --ups-lambda if both set).")
    parser.add_argument("--mu-mult", type=float, default=1.0,
                        help="Multiplier on Input_data UPS mu.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path (will also write a sibling _report.html if --html).")
    parser.add_argument("--html", action="store_true",
                        help="Also generate a plot_result HTML next to the JSON.")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open the HTML in browser.")
    args = parser.parse_args(argv)

    params = get_sim_params()
    q_path, log_path, meta, result_dir = _resolve_file_arg(args.checkpoint)
    if q_path is None:
        return 1

    q_table = load_q_table(str(q_path))
    print(f"Loaded Q-table: {q_path} ({len(q_table):,} entries)")

    base_lam = float(params.get("ups_lambda", 0.0))
    base_mu = float(params.get("ups_mu", 0.0))
    lam = float(args.ups_lambda) if args.ups_lambda is not None else base_lam * args.lambda_mult
    mu = float(args.ups_mu) if args.ups_mu is not None else base_mu * args.mu_mult

    print(f"Running seed={args.seed}, lambda={lam:.2f} (x{args.lambda_mult}), mu={mu:.2f} (x{args.mu_mult})")
    ups_events = generate_ups_events(lam, mu, args.seed)
    engine = SimulationEngine(params)
    strategy = QStrategy(params, q_table=q_table)
    kpi, state = engine.run(strategy, ups_events)
    net_profit = float(kpi.net_profit())
    print(f"  Net profit: ${net_profit:,.0f}")

    result: dict[str, Any] = _assemble_result(
        kpi, state, ups_events, params, lam, mu, args.seed, q_path, meta,
    )
    # Tag the multipliers used so the JSON is Block-B-cell-aware
    result["metadata"]["lambda_mult"] = round(args.lambda_mult, 4)
    result["metadata"]["mu_mult"] = round(args.mu_mult, 4)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)
    print(f"  JSON saved to: {out_path}")

    if args.html:
        training_info: dict[str, Any] | None = None
        if log_path and Path(log_path).exists():
            training_info = {
                "log_path": str(log_path),
                "epsilon_start": (meta or {}).get("epsilon_start"),
                "epsilon_end": (meta or {}).get("epsilon_end"),
            }
        html = _build_html(result, None, offline=True, training_info=training_info)
        html_path = out_path.with_suffix("").with_name(out_path.stem + "_report.html")
        html_path.write_text(html, encoding="utf-8")
        print(f"  HTML report: {html_path}")
        if not args.no_open:
            try:
                webbrowser.open(html_path.resolve().as_uri())
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
