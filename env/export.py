"""Export simulation results, schedules, UPS events, and traces."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _resolve_output_dir(output_dir: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    path = Path(output_dir)
    if not path.is_absolute():
        path = base_dir / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _batch_to_dict(batch) -> dict[str, Any]:
    return {
        "batch_id": str(batch.batch_id),
        "sku": batch.sku,
        "roaster": batch.roaster,
        "start": int(batch.start),
        "end": int(batch.end),
        "output_line": batch.output_line,
        "is_mto": bool(batch.is_mto),
    }


def _cancelled_batch_to_dict(batch, ups_events: list[Any], used_events: set[tuple[int, str, int]]) -> dict[str, Any]:
    payload = _batch_to_dict(batch)
    cancel_time = None
    for event in sorted(ups_events, key=lambda e: (int(e.t), e.roaster_id, int(e.duration))):
        event_key = (int(event.t), event.roaster_id, int(event.duration))
        if event_key in used_events:
            continue
        if event.roaster_id != batch.roaster:
            continue
        if int(batch.start) <= int(event.t) < int(batch.end):
            cancel_time = int(event.t)
            used_events.add(event_key)
            break
    payload["status"] = "cancelled"
    if cancel_time is not None:
        payload["cancel_time"] = cancel_time
    return payload


def _restock_to_dict(rst) -> dict[str, Any]:
    return {
        "line_id": rst.line_id,
        "sku": rst.sku,
        "start": int(rst.start),
        "end": int(rst.end),
        "qty": int(rst.qty),
    }


def export_run(
    kpi: dict,
    state,
    params: dict,
    ups_events: list,
    output_dir: str = "export_output",
    run_id: str | None = None,
) -> dict:
    """Export summary JSON, schedule JSON, trace CSV, UPS JSON, cancelled JSON, and restock JSON."""

    out_dir = _resolve_output_dir(output_dir)
    run_name = run_id or "run"

    result_path = out_dir / f"{run_name}_result.json"
    schedule_path = out_dir / f"{run_name}_schedule.json"
    trace_path = out_dir / f"{run_name}_trace.csv"
    ups_path = out_dir / f"{run_name}_ups.json"
    cancelled_path = out_dir / f"{run_name}_cancelled.json"
    restock_path = out_dir / f"{run_name}_restocks.json"

    result_payload = dict(kpi)
    result_payload.setdefault("allow_r3_flex", bool(params.get("allow_r3_flex", False)))
    result_payload.setdefault("solver_engine", "simulation")

    gc_init = {f"{k[0]}_{k[1]}": v for k, v in params.get("gc_init", {}).items()}
    gc_final = {f"{k[0]}_{k[1]}": v for k, v in state.gc_stock.items()}
    result_payload["gc_init"] = gc_init
    result_payload["gc_final"] = gc_final
    gc_capacity = {f"{k[0]}_{k[1]}": v for k, v in params.get("gc_capacity", {}).items()}
    result_payload["gc_capacity"] = gc_capacity
    result_payload["restock_duration"] = params.get("restock_duration", 15)
    result_payload["restock_qty"] = params.get("restock_qty", 5)
    if state.active_restock is not None:
        result_payload["active_restock_at_end"] = _restock_to_dict(state.active_restock)

    schedule_payload = [_batch_to_dict(b) for b in state.completed_batches]
    used_cancel_events: set[tuple[int, str, int]] = set()
    cancelled_payload = [
        _cancelled_batch_to_dict(b, ups_events, used_cancel_events)
        for b in state.cancelled_batches
    ]
    restock_payload = [_restock_to_dict(r) for r in state.completed_restocks]
    ups_payload = [
        {"t": int(e.t), "roaster_id": e.roaster_id, "duration": int(e.duration)}
        for e in ups_events
    ]

    result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    schedule_path.write_text(json.dumps(schedule_payload, indent=2), encoding="utf-8")
    ups_path.write_text(json.dumps(ups_payload, indent=2), encoding="utf-8")
    cancelled_path.write_text(json.dumps(cancelled_payload, indent=2), encoding="utf-8")
    restock_path.write_text(json.dumps(restock_payload, indent=2), encoding="utf-8")

    fieldnames = ["t"]
    fieldnames.extend(f"status_{r}" for r in params["roasters"])
    fieldnames.extend(f"remaining_{r}" for r in params["roasters"])
    fieldnames.extend(f"needs_decision_{r}" for r in params["roasters"])
    fieldnames.extend(f"rc_{l}" for l in params["lines"])
    fieldnames.extend(f"pipeline_{l}" for l in params["lines"])
    fieldnames.extend(f"pipeline_mode_{l}" for l in params["lines"])
    for pair in params.get("feasible_gc_pairs", []):
        fieldnames.append(f"gc_{pair[0]}_{pair[1]}")
    fieldnames.append("restock_busy")

    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in state.trace:
            row = {"t": snapshot["t"]}
            for rid in params["roasters"]:
                row[f"status_{rid}"] = snapshot["status"][rid]
                row[f"remaining_{rid}"] = snapshot["remaining"][rid]
                row[f"needs_decision_{rid}"] = int(snapshot["needs_decision"][rid])
            for lid in params["lines"]:
                row[f"rc_{lid}"] = snapshot["rc_stock"][lid]
                row[f"pipeline_{lid}"] = snapshot["pipeline_busy"][lid]
                row[f"pipeline_mode_{lid}"] = snapshot.get("pipeline_mode", {}).get(lid, "")
            for pair in params.get("feasible_gc_pairs", []):
                key = f"{pair[0]}_{pair[1]}"
                row[f"gc_{key}"] = snapshot.get("gc_stock", {}).get(key, "")
            row["restock_busy"] = snapshot.get("restock_busy", 0)
            writer.writerow(row)

    return {
        "result_json": str(result_path),
        "schedule_json": str(schedule_path),
        "trace_csv": str(trace_path),
        "ups_json": str(ups_path),
        "cancelled_json": str(cancelled_path),
        "restocks_json": str(restock_path),
    }


def export_schedule_to_gantt(state, output_path: str):
    """Export completed batches as a flat CSV for Gantt tools."""
    path = Path(output_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["batch_id", "sku", "roaster", "start", "end", "output_line", "is_mto"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch in state.completed_batches:
            writer.writerow(_batch_to_dict(batch))


def load_run_for_test(result_json_path: str) -> dict:
    """Load a previously exported summary JSON."""
    with open(result_json_path, encoding="utf-8") as handle:
        return json.load(handle)
