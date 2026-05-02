"""Data loader for CP_SAT_v2.

Reads the current root ``Input_data/`` schema directly and prepares the
deterministic, no-UPS CP-SAT inputs for the RC + GC inventory model.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any


logger = logging.getLogger("cpsat_data")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "roasters.csv",
    "skus.csv",
    "jobs.csv",
    "shift_parameters.csv",
    "planned_downtime.csv",
    "solver_config.csv",
)

REQUIRED_SHIFT_KEYS = (
    "shift_length_min",
    "max_rc_batches_per_line",
    "safety_stock_batches",
    "setup_time_diff_sku_min",
    "initial_rc_l1",
    "initial_rc_l2",
    "psc_consume_rate_l1_min_per_batch",
    "psc_consume_rate_l2_min_per_batch",
    "tardiness_cost_per_min",
    "idle_cost_per_min_per_roaster",
    "overflow_idle_cost_per_min_per_roaster",
    "setup_cost_per_event",
    "stockout_cost_per_event_per_line",
    "mto_skip_penalty_per_batch",
    "gc_capacity_l1_psc",
    "gc_capacity_l1_ndg",
    "gc_capacity_l1_busta",
    "gc_capacity_l2_psc",
    "gc_initial_l1_psc",
    "gc_initial_l1_ndg",
    "gc_initial_l1_busta",
    "gc_initial_l2_psc",
    "restock_duration_min",
    "restock_qty_batches",
)

REQUIRED_SOLVER_KEYS = (
    "time_limit_sec",
    "mip_gap_target",
    "allow_r3_flexible_output",
)

SKU_FLAG_COLUMNS = (
    ("PSC", "can_process_psc"),
    ("NDG", "can_process_ndg"),
    ("BUSTA", "can_process_busta"),
)

OUTPUT_FLAG_COLUMNS = (
    ("L1", "can_output_l1"),
    ("L2", "can_output_l2"),
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Missing required CSV file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_parameter_map(path: Path, required_keys: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in _read_csv_rows(path):
        key = (row.get("parameter") or "").strip()
        raw_value = row.get("value")
        if key:
            values[key] = "" if raw_value is None else raw_value.strip()
    for key in required_keys:
        if key not in values:
            raise ValueError(f"Missing required parameter key '{key}' in {path.name}")
    return values


def _resolve_input_dir(input_dir: str | Path) -> Path:
    path = Path(input_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_int(raw_value: Any, label: str) -> int:
    try:
        return int(float(raw_value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {label}: {raw_value!r}") from exc


def _parse_float(raw_value: Any, label: str) -> float:
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {label}: {raw_value!r}") from exc


def _parse_bool(raw_value: Any, label: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    text = str(raw_value).strip()
    if text in {"0", "1"}:
        return text == "1"
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    raise ValueError(f"Invalid boolean flag for {label}: {raw_value!r}")


def _safe_bool(raw_value: Any, label: str, default: bool = False) -> bool:
    if raw_value in (None, ""):
        return default
    return _parse_bool(raw_value, label)


def _expand_downtime_slots(start_min: int, end_min: int, convention: str) -> set[int]:
    if convention.strip().lower() == "inclusive":
        return set(range(start_min, end_min + 1))
    return set(range(start_min, end_min))


def load(input_dir: str = "Input_data", overrides: dict | None = None) -> dict[str, Any]:
    input_path = _resolve_input_dir(input_dir)

    for file_name in REQUIRED_FILES:
        file_path = input_path / file_name
        if not file_path.exists():
            raise ValueError(f"Missing required CSV file: {file_path}")

    roaster_rows = _read_csv_rows(input_path / "roasters.csv")
    sku_rows = _read_csv_rows(input_path / "skus.csv")
    job_rows = _read_csv_rows(input_path / "jobs.csv")
    shift_params = _read_parameter_map(input_path / "shift_parameters.csv", REQUIRED_SHIFT_KEYS)
    downtime_rows = _read_csv_rows(input_path / "planned_downtime.csv")
    solver_params = _read_parameter_map(input_path / "solver_config.csv", REQUIRED_SOLVER_KEYS)

    active_roaster_rows = [
        row for row in roaster_rows if _safe_bool(row.get("is_active"), "is_active", default=False)
    ]
    if not active_roaster_rows:
        raise ValueError("No active roasters found in roasters.csv")

    roasters = [row["roaster_id"].strip() for row in active_roaster_rows]
    lines = sorted({row["line_id"].strip() for row in active_roaster_rows})

    roaster_line: dict[str, str] = {}
    roaster_pipeline: dict[str, str] = {}
    roaster_can_output: dict[str, list[str]] = {}
    roaster_eligible_skus: dict[str, list[str]] = {}
    roaster_initial_sku: dict[str, str] = {}
    consume_time_by_roaster: dict[str, int] = {}

    for row in active_roaster_rows:
        roaster_id = row["roaster_id"].strip()
        roaster_line[roaster_id] = row["line_id"].strip()
        roaster_pipeline[roaster_id] = row["pipeline_line"].strip()
        roaster_can_output[roaster_id] = [
            line_id
            for line_id, column_name in OUTPUT_FLAG_COLUMNS
            if _safe_bool(row.get(column_name), f"{roaster_id}.{column_name}", default=False)
        ]
        roaster_eligible_skus[roaster_id] = [
            sku
            for sku, column_name in SKU_FLAG_COLUMNS
            if _safe_bool(row.get(column_name), f"{roaster_id}.{column_name}", default=False)
        ]
        roaster_initial_sku[roaster_id] = (row.get("initial_last_sku") or "PSC").strip()
        consume_time_by_roaster[roaster_id] = _parse_int(
            row.get("consume_time_min", 3),
            f"{roaster_id}.consume_time_min",
        )

    skus = [row["sku"].strip() for row in sku_rows]
    sku_revenue = {
        row["sku"].strip(): _parse_float(
            row["revenue_per_batch_usd"],
            f"{row['sku']}.revenue_per_batch_usd",
        )
        for row in sku_rows
    }
    sku_credits_rc = {
        row["sku"].strip(): _safe_bool(
            row.get("credits_rc_stock"),
            f"{row['sku']}.credits_rc_stock",
            default=False,
        )
        for row in sku_rows
    }
    roast_time_by_sku = {
        row["sku"].strip(): _parse_int(row["roast_time_min"], f"{row['sku']}.roast_time_min")
        for row in sku_rows
    }
    sku_eligible_roasters = {
        sku: [roaster_id for roaster_id in roasters if sku in roaster_eligible_skus[roaster_id]]
        for sku in skus
    }

    jobs = [row["job_id"].strip() for row in job_rows]
    job_sku = {row["job_id"].strip(): row["sku"].strip() for row in job_rows}
    job_batches = {
        row["job_id"].strip(): _parse_int(row["required_batches"], f"{row['job_id']}.required_batches")
        for row in job_rows
    }
    job_due = {
        row["job_id"].strip(): _parse_int(row["due_time_min"], f"{row['job_id']}.due_time_min")
        for row in job_rows
    }
    job_release = {
        row["job_id"].strip(): _parse_int(row.get("release_time_min", 0), f"{row['job_id']}.release_time_min")
        for row in job_rows
    }

    shift_length = _parse_int(shift_params["shift_length_min"], "shift_length_min")
    consume_time = _parse_int(
        next(iter(consume_time_by_roaster.values())),
        "consume_time_min",
    )
    setup_time = _parse_int(shift_params["setup_time_diff_sku_min"], "setup_time_diff_sku_min")
    max_rc = _parse_int(shift_params["max_rc_batches_per_line"], "max_rc_batches_per_line")
    safety_stock = _parse_int(shift_params["safety_stock_batches"], "safety_stock_batches")
    psc_pool_per_roaster = math.floor(shift_length / roast_time_by_sku["PSC"])
    rc_init = {
        "L1": _parse_int(shift_params["initial_rc_l1"], "initial_rc_l1"),
        "L2": _parse_int(shift_params["initial_rc_l2"], "initial_rc_l2"),
    }
    consume_rate = {
        "L1": _parse_float(
            shift_params["psc_consume_rate_l1_min_per_batch"],
            "psc_consume_rate_l1_min_per_batch",
        ),
        "L2": _parse_float(
            shift_params["psc_consume_rate_l2_min_per_batch"],
            "psc_consume_rate_l2_min_per_batch",
        ),
    }

    gc_capacity = {
        ("L1", "PSC"): _parse_int(shift_params["gc_capacity_l1_psc"], "gc_capacity_l1_psc"),
        ("L1", "NDG"): _parse_int(shift_params["gc_capacity_l1_ndg"], "gc_capacity_l1_ndg"),
        ("L1", "BUSTA"): _parse_int(shift_params["gc_capacity_l1_busta"], "gc_capacity_l1_busta"),
        ("L2", "PSC"): _parse_int(shift_params["gc_capacity_l2_psc"], "gc_capacity_l2_psc"),
    }
    gc_init = {
        ("L1", "PSC"): _parse_int(shift_params["gc_initial_l1_psc"], "gc_initial_l1_psc"),
        ("L1", "NDG"): _parse_int(shift_params["gc_initial_l1_ndg"], "gc_initial_l1_ndg"),
        ("L1", "BUSTA"): _parse_int(shift_params["gc_initial_l1_busta"], "gc_initial_l1_busta"),
        ("L2", "PSC"): _parse_int(shift_params["gc_initial_l2_psc"], "gc_initial_l2_psc"),
    }
    feasible_gc_pairs = list(gc_capacity.keys())
    restock_duration = _parse_int(shift_params["restock_duration_min"], "restock_duration_min")
    restock_qty = _parse_int(shift_params["restock_qty_batches"], "restock_qty_batches")

    cost_tardiness = _parse_float(shift_params["tardiness_cost_per_min"], "tardiness_cost_per_min")
    cost_idle = _parse_float(
        shift_params["idle_cost_per_min_per_roaster"],
        "idle_cost_per_min_per_roaster",
    )
    cost_overflow = _parse_float(
        shift_params["overflow_idle_cost_per_min_per_roaster"],
        "overflow_idle_cost_per_min_per_roaster",
    )
    cost_setup = _parse_float(shift_params["setup_cost_per_event"], "setup_cost_per_event")
    cost_stockout = _parse_float(
        shift_params["stockout_cost_per_event_per_line"],
        "stockout_cost_per_event_per_line",
    )
    cost_skip_mto = _parse_float(
        shift_params["mto_skip_penalty_per_batch"],
        "mto_skip_penalty_per_batch",
    )

    time_limit = _parse_int(solver_params["time_limit_sec"], "time_limit_sec")
    mip_gap = _parse_float(solver_params["mip_gap_target"], "mip_gap_target")
    allow_r3_flex = _parse_bool(
        solver_params["allow_r3_flexible_output"],
        "allow_r3_flexible_output",
    )

    if overrides:
        if "time_limit" in overrides:
            time_limit = _parse_int(overrides["time_limit"], "override.time_limit")
        if "mip_gap" in overrides:
            mip_gap = _parse_float(overrides["mip_gap"], "override.mip_gap")
        if "allow_r3_flex" in overrides:
            allow_r3_flex = _parse_bool(overrides["allow_r3_flex"], "override.allow_r3_flex")

    mto_batches = [
        (job_id, batch_index)
        for job_id in jobs
        for batch_index in range(job_batches[job_id])
    ]
    psc_pool = [
        (roaster_id, slot_index)
        for roaster_id in roasters
        for slot_index in range(psc_pool_per_roaster)
    ]
    all_batches = mto_batches + psc_pool

    batch_sku = {batch_id: job_sku[batch_id[0]] for batch_id in mto_batches}
    batch_sku.update({batch_id: "PSC" for batch_id in psc_pool})

    batch_is_mto = {batch_id: True for batch_id in mto_batches}
    batch_is_mto.update({batch_id: False for batch_id in psc_pool})

    batch_eligible_roasters = {
        batch_id: list(sku_eligible_roasters[batch_sku[batch_id]])
        for batch_id in mto_batches
    }
    batch_eligible_roasters.update({batch_id: list(sku_eligible_roasters["PSC"]) for batch_id in psc_pool})

    sched_eligible_roasters = {
        batch_id: [batch_id[0]] if batch_id in psc_pool else list(batch_eligible_roasters[batch_id])
        for batch_id in all_batches
    }

    consumption_events: dict[str, list[int]] = {}
    for line_id in lines:
        rate = consume_rate[line_id]
        event_count = math.floor(shift_length / rate)
        consumption_events[line_id] = [
            math.floor(event_index * rate)
            for event_index in range(1, event_count + 1)
            if math.floor(event_index * rate) < shift_length
        ]

    downtime_slots = {roaster_id: set() for roaster_id in roasters}
    for row in downtime_rows:
        roaster_id = (row.get("roaster_id") or "").strip()
        if roaster_id not in downtime_slots:
            continue
        start_min = _parse_int(row.get("start_min"), f"{roaster_id}.start_min")
        end_min = _parse_int(row.get("end_min"), f"{roaster_id}.end_min")
        convention = (row.get("end_min_convention") or "").strip()
        downtime_slots[roaster_id].update(_expand_downtime_slots(start_min, end_min, convention))

    pair_to_roasters = {
        pair: [
            roaster_id
            for roaster_id in roasters
            if roaster_pipeline[roaster_id] == pair[0] and pair[1] in roaster_eligible_skus[roaster_id]
        ]
        for pair in feasible_gc_pairs
    }

    return {
        "shift_length": shift_length,
        "consume_time": consume_time,
        "setup_time": setup_time,
        "max_rc": max_rc,
        "safety_stock": safety_stock,
        "psc_pool_per_roaster": psc_pool_per_roaster,
        "rc_init": rc_init,
        "consume_rate": consume_rate,
        "cost_tardiness": cost_tardiness,
        "cost_idle": cost_idle,
        "cost_overflow": cost_overflow,
        "cost_setup": cost_setup,
        "cost_stockout": cost_stockout,
        "cost_skip_mto": cost_skip_mto,
        "time_limit": time_limit,
        "mip_gap": mip_gap,
        "allow_r3_flex": allow_r3_flex,
        "lines": lines,
        "roasters": roasters,
        "roaster_line": roaster_line,
        "roaster_pipeline": roaster_pipeline,
        "roaster_can_output": roaster_can_output,
        "roaster_eligible_skus": roaster_eligible_skus,
        "roaster_initial_sku": roaster_initial_sku,
        "skus": skus,
        "sku_revenue": sku_revenue,
        "sku_credits_rc": sku_credits_rc,
        "roast_time_by_sku": roast_time_by_sku,
        "sku_eligible_roasters": sku_eligible_roasters,
        "jobs": jobs,
        "job_sku": job_sku,
        "job_batches": job_batches,
        "job_due": job_due,
        "job_release": job_release,
        "mto_batches": mto_batches,
        "psc_pool": psc_pool,
        "all_batches": all_batches,
        "batch_sku": batch_sku,
        "batch_is_mto": batch_is_mto,
        "batch_eligible_roasters": batch_eligible_roasters,
        "sched_eligible_roasters": sched_eligible_roasters,
        "consumption_events": consumption_events,
        "downtime_slots": downtime_slots,
        "gc_capacity": gc_capacity,
        "gc_init": gc_init,
        "feasible_gc_pairs": feasible_gc_pairs,
        "restock_duration": restock_duration,
        "restock_qty": restock_qty,
        "pair_to_roasters": pair_to_roasters,
        "MS_by_sku": {
            sku: shift_length - duration for sku, duration in roast_time_by_sku.items()
        },
        "input_dir": str(input_path),
    }
