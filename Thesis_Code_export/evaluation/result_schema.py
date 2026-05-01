"""Universal result schema and legacy converters for thesis experiments.

This module is intentionally standalone. It reads plain JSON and the root
``Input_data/`` CSV files only, so every solver and post-processing tool can
share one canonical result format without importing solver-specific code.
"""

from __future__ import annotations

import ast
import copy
import csv
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = "Input_data"
DEFAULT_RESULTS_DIR = "Results"
DEFAULT_LINES = ("L1", "L2")
DEFAULT_ROASTERS = ("R1", "R2", "R3", "R4", "R5")
DEFAULT_JOBS = ("J1", "J2")
DEFAULT_SKU_REVENUE = {"PSC": 4000.0, "NDG": 7000.0, "BUSTA": 7000.0}


def make_run_dir(method: str, run_name: str, root: Path | None = None) -> Path:
    """Create a uniformly-named output directory under ``Results/``.

    Naming: ``Results/<YYYYMMDD_HHMMSS>_<Method>_<RunName>/``. Method labels
    follow the project vocabulary: ``CPSAT``, ``Dispatch``, ``QLearning``,
    ``RLHH``, ``PaengDDQNv2``, ``MasterEval``, ``100SeedEval``, ``BlockB``,
    ``Eval_<MethodName>``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_method = "".join(c if c.isalnum() or c in "-_" else "_" for c in method)
    safe_run = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
    base = Path(root) if root is not None else (ROOT_DIR / DEFAULT_RESULTS_DIR)
    out = base / f"{ts}_{safe_method}_{safe_run}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    return int(float(str(value).strip()))


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return _as_int(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(str(value).strip())


def _resolve_input_dir(input_dir: str | Path | None = None) -> Path:
    if input_dir is None:
        return ROOT_DIR / DEFAULT_INPUT_DIR
    path = Path(input_dir)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=8)
def load_reference_data(input_dir: str | Path | None = None) -> dict[str, Any]:
    """Load structural reference data from the root ``Input_data/`` folder."""

    input_path = _resolve_input_dir(input_dir)
    roaster_rows = _read_csv_rows(input_path / "roasters.csv")
    job_rows = _read_csv_rows(input_path / "jobs.csv")
    downtime_rows = _read_csv_rows(input_path / "planned_downtime.csv")
    sku_rows = _read_csv_rows(input_path / "skus.csv")

    roasters: list[str] = []
    roaster_line: dict[str, str] = {}
    roaster_pipeline: dict[str, str] = {}
    roaster_can_output: dict[str, list[str]] = {}
    roaster_eligible_skus: dict[str, list[str]] = {}
    roaster_initial_sku: dict[str, str] = {}

    for row in roaster_rows:
        if not _as_bool(row.get("is_active"), True):
            continue
        roaster = row["roaster_id"].strip()
        roasters.append(roaster)
        roaster_line[roaster] = row["line_id"].strip()
        roaster_pipeline[roaster] = row["pipeline_line"].strip()
        eligible: list[str] = []
        if _as_bool(row.get("can_process_psc")):
            eligible.append("PSC")
        if _as_bool(row.get("can_process_ndg")):
            eligible.append("NDG")
        if _as_bool(row.get("can_process_busta")):
            eligible.append("BUSTA")
        roaster_eligible_skus[roaster] = eligible
        outputs: list[str] = []
        if _as_bool(row.get("can_output_l1")):
            outputs.append("L1")
        if _as_bool(row.get("can_output_l2")):
            outputs.append("L2")
        roaster_can_output[roaster] = outputs
        roaster_initial_sku[roaster] = (row.get("initial_last_sku") or "PSC").strip()

    job_batches: dict[str, int] = {}
    job_due: dict[str, int] = {}
    job_sku: dict[str, str] = {}
    for row in job_rows:
        job_id = row["job_id"].strip()
        job_batches[job_id] = _as_int(row.get("required_batches"))
        job_due[job_id] = _as_int(row.get("due_time_min"))
        job_sku[job_id] = (row.get("sku") or "").strip()

    downtime_slots: dict[str, set[int]] = {roaster: set() for roaster in roasters}
    for row in downtime_rows:
        roaster = row["roaster_id"].strip()
        start = _as_int(row.get("start_min"))
        end = _as_int(row.get("end_min"))
        if (row.get("end_min_convention") or "").strip().lower() == "inclusive":
            end += 1
        downtime_slots.setdefault(roaster, set()).update(range(start, end))

    sku_revenue: dict[str, float] = {}
    sku_is_mto: dict[str, bool] = {}
    sku_credits_rc: dict[str, bool] = {}
    roast_time_by_sku: dict[str, int] = {}
    for row in sku_rows:
        sku = row["sku"].strip()
        sku_revenue[sku] = _as_float(row.get("revenue_per_batch_usd"))
        sku_is_mto[sku] = _as_bool(row.get("is_mto"))
        sku_credits_rc[sku] = _as_bool(row.get("credits_rc_stock"))
        rt = row.get("roast_time_min")
        if rt is not None and str(rt).strip():
            roast_time_by_sku[sku] = _as_int(rt, 15)

    return {
        "roasters": tuple(roasters or DEFAULT_ROASTERS),
        "lines": DEFAULT_LINES,
        "roaster_line": roaster_line,
        "roaster_pipeline": roaster_pipeline,
        "roaster_can_output": roaster_can_output,
        "roaster_eligible_skus": roaster_eligible_skus,
        "roaster_initial_sku": roaster_initial_sku,
        "job_batches": job_batches,
        "job_due": job_due,
        "job_sku": job_sku,
        "downtime_slots": downtime_slots,
        "sku_revenue": sku_revenue or DEFAULT_SKU_REVENUE.copy(),
        "sku_is_mto": sku_is_mto,
        "sku_credits_rc": sku_credits_rc,
        "roast_time_by_sku": roast_time_by_sku or {"PSC": 15, "NDG": 17, "BUSTA": 18},
        "sku_rows": sku_rows,
    }


@lru_cache(maxsize=8)
def load_default_parameters(input_dir: str | Path | None = None) -> dict[str, Any]:
    """Load numeric parameters from ``Input_data/`` with clean canonical keys."""

    input_path = _resolve_input_dir(input_dir)
    rows = _read_csv_rows(input_path / "shift_parameters.csv")
    lookup = {row["parameter"].strip(): row.get("value", "").strip() for row in rows}
    refs = load_reference_data(input_dir)

    roast_time_by_sku = copy.deepcopy(refs.get("roast_time_by_sku", {"PSC": 15, "NDG": 17, "BUSTA": 18}))

    gc_capacity: dict[str, int] = {}
    gc_init: dict[str, int] = {}
    feasible_gc_pairs: list[str] = []
    for key, val in lookup.items():
        if key.startswith("gc_capacity_"):
            pair = key[len("gc_capacity_"):].upper()
            gc_capacity[pair] = _as_int(val, 0)
            if pair not in feasible_gc_pairs:
                feasible_gc_pairs.append(pair)
        elif key.startswith("gc_initial_"):
            pair = key[len("gc_initial_"):].upper()
            gc_init[pair] = _as_int(val, 0)
            if pair not in feasible_gc_pairs:
                feasible_gc_pairs.append(pair)

    return {
        "P": _as_int(lookup.get("process_time_min"), 15),
        "DC": _as_int(lookup.get("consume_time_min"), 3),
        "sigma": _as_int(lookup.get("setup_time_diff_sku_min"), 5),
        "SL": _as_int(lookup.get("shift_length_min"), 480),
        "MS": _as_int(lookup.get("max_start_min"), 465) or 465,
        "max_rc": _as_int(lookup.get("max_rc_batches_per_line"), 40),
        "safety_stock": _as_int(lookup.get("safety_stock_batches"), 20),
        "rc_init": {
            "L1": _as_int(lookup.get("initial_rc_l1"), 12),
            "L2": _as_int(lookup.get("initial_rc_l2"), 15),
        },
        "consume_rate": {
            "L1": _as_float(lookup.get("psc_consume_rate_l1_min_per_batch"), 8.1),
            "L2": _as_float(lookup.get("psc_consume_rate_l2_min_per_batch"), 7.8),
        },
        "sku_revenue": copy.deepcopy(refs["sku_revenue"]),
        "roast_time_by_sku": roast_time_by_sku,
        "c_tard": _as_float(lookup.get("tardiness_cost_per_min"), 1000.0),
        "c_stock": _as_float(lookup.get("stockout_cost_per_event_per_line"), 1500.0),
        "c_idle": _as_float(lookup.get("idle_cost_per_min_per_roaster"), 200.0),
        "c_over": _as_float(lookup.get("overflow_idle_cost_per_min_per_roaster"), 50.0),
        "c_setup": _as_float(lookup.get("setup_cost_per_event"), 800.0),
        "gc_capacity": gc_capacity,
        "gc_init": gc_init,
        "restock_duration": _as_int(lookup.get("restock_duration_min"), 15),
        "restock_qty": _as_int(lookup.get("restock_qty_batches"), 5),
        "feasible_gc_pairs": feasible_gc_pairs,
    }


def _deep_merge(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    if not updates:
        return merged
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _derive_job_id(batch_id: Any) -> str | None:
    if batch_id is None:
        return None
    if isinstance(batch_id, (list, tuple)) and batch_id:
        head = str(batch_id[0])
        return head if head.startswith("J") else None
    text = str(batch_id).strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)) and parsed:
        head = str(parsed[0])
        return head if head.startswith("J") else None
    if text.startswith("J"):
        return text.split("_", 1)[0]
    return None


def _normalize_setup(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value in (None, ""):
        return "No"
    text = str(value).strip()
    if text.lower() in {"1", "true", "yes", "y"}:
        return "Yes"
    if text.lower() in {"0", "false", "no", "n"}:
        return "No"
    return text


def _normalize_ups_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "t": _as_int(event.get("t")),
        "roaster_id": str(event.get("roaster_id", "")).strip(),
        "duration": _as_int(event.get("duration"), 0),
    }


def normalize_schedule_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize one batch schedule entry into the canonical schema."""

    refs = load_reference_data()
    params = load_default_parameters()
    roaster = str(entry.get("roaster", "")).strip()
    start = _as_int(entry.get("start"), 0)
    sku = str(entry.get("sku", "PSC")).strip()
    roast_times = params.get("roast_time_by_sku", {"PSC": 15, "NDG": 17, "BUSTA": 18})
    default_duration = roast_times.get(sku, 15)
    end = _as_int(entry.get("end"), start + default_duration)
    pipeline = entry.get("pipeline")
    if isinstance(pipeline, str) and "[" in pipeline:
        pipeline = pipeline.split("[", 1)[0]
    if pipeline in (None, ""):
        pipeline = refs["roaster_pipeline"].get(roaster)

    output_line = entry.get("output_line")
    if output_line in ("", "None"):
        output_line = None
    if output_line is None:
        outputs = refs["roaster_can_output"].get(roaster, [])
        if len(outputs) == 1:
            output_line = outputs[0]

    batch_id = str(entry.get("batch_id", ""))
    job_id = entry.get("job_id")
    if job_id in ("", None):
        job_id = _derive_job_id(batch_id)

    is_mto = entry.get("is_mto")
    if is_mto is None:
        is_mto = bool(job_id) or str(entry.get("sku", "")).strip().upper() in {"NDG", "BUSTA"}

    pipeline_start = _as_int(entry.get("pipeline_start"), start)
    pipeline_end = _as_int(entry.get("pipeline_end"), start + params["DC"])
    cancel_time = _optional_int(entry.get("cancel_time"))

    return {
        "batch_id": batch_id,
        "job_id": str(job_id) if job_id not in (None, "") else None,
        "sku": str(entry.get("sku", "")).strip().upper(),
        "roaster": roaster,
        "start": start,
        "end": end,
        "pipeline": str(pipeline) if pipeline not in (None, "") else None,
        "pipeline_start": pipeline_start,
        "pipeline_end": pipeline_end,
        "output_line": str(output_line) if output_line not in (None, "") else None,
        "is_mto": bool(is_mto),
        "setup": _normalize_setup(entry.get("setup")),
        "status": str(entry.get("status", "completed")).strip().lower() or "completed",
        "cancel_time": cancel_time,
    }


def _default_metadata() -> dict[str, Any]:
    return {
        "solver_engine": "unknown",
        "solver_name": "",
        "status": "Unknown",
        "solve_time_sec": 0.0,
        "total_compute_ms": 0.0,
        "num_resolves": 0,
        "obj_value": None,
        "best_bound": None,
        "gap_pct": None,
        "allow_r3_flex": True,
        "timestamp": _now_iso(),
        "input_dir": DEFAULT_INPUT_DIR,
        "notes": "",
    }


def _default_experiment() -> dict[str, Any]:
    return {
        "lambda_rate": 0,
        "mu_mean": 0,
        "seed": None,
        "replication": None,
        "scenario_label": "no_ups",
    }


def _default_kpi() -> dict[str, Any]:
    return {
        "net_profit": 0.0,
        "total_revenue": 0.0,
        "total_costs": 0.0,
        "psc_count": 0,
        "ndg_count": 0,
        "busta_count": 0,
        "total_batches": 0,
        "revenue_psc": 0.0,
        "revenue_ndg": 0.0,
        "revenue_busta": 0.0,
        "tardiness_min": {job: 0.0 for job in DEFAULT_JOBS},
        "tard_cost": 0.0,
        "setup_events": 0,
        "setup_cost": 0.0,
        "stockout_events": {line: 0 for line in DEFAULT_LINES},
        "stockout_duration": {line: 0 for line in DEFAULT_LINES},
        "stockout_cost": 0.0,
        "idle_min": 0.0,
        "idle_min_per_roaster": {roaster: 0.0 for roaster in DEFAULT_ROASTERS},
        "idle_cost": 0.0,
        "over_min": 0.0,
        "over_min_per_roaster": {roaster: 0.0 for roaster in DEFAULT_ROASTERS},
        "over_cost": 0.0,
        "restock_count": 0,
    }


def _canonical_parameters(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base = load_default_parameters()
    merged = _deep_merge(base, overrides)

    def _normalize_pair_map(raw_map: dict[Any, Any] | None) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for raw_key, value in (raw_map or {}).items():
            if isinstance(raw_key, tuple) and len(raw_key) == 2:
                key = f"{raw_key[0]}_{raw_key[1]}"
            else:
                key = str(raw_key)
            normalized[key] = value
        return normalized

    def _normalize_pair_list(raw_pairs: list[Any] | None) -> list[str]:
        normalized: list[str] = []
        for raw_pair in raw_pairs or []:
            if isinstance(raw_pair, tuple) and len(raw_pair) == 2:
                pair = f"{raw_pair[0]}_{raw_pair[1]}"
            else:
                pair = str(raw_pair)
            if pair not in normalized:
                normalized.append(pair)
        return normalized

    merged["P"] = _as_int(merged.get("P"), base["P"])
    merged["DC"] = _as_int(merged.get("DC"), base["DC"])
    merged["sigma"] = _as_int(merged.get("sigma"), base["sigma"])
    merged["SL"] = _as_int(merged.get("SL"), base["SL"])
    merged["MS"] = _as_int(merged.get("MS"), base["MS"])
    merged["max_rc"] = _as_int(merged.get("max_rc"), base["max_rc"])
    merged["safety_stock"] = _as_int(merged.get("safety_stock"), base["safety_stock"])
    merged["rc_init"] = {
        line: _as_int((merged.get("rc_init") or {}).get(line), base["rc_init"][line])
        for line in DEFAULT_LINES
    }
    merged["consume_rate"] = {
        line: _as_float((merged.get("consume_rate") or {}).get(line), base["consume_rate"][line])
        for line in DEFAULT_LINES
    }
    merged["sku_revenue"] = {
        sku: _as_float((merged.get("sku_revenue") or {}).get(sku), base["sku_revenue"].get(sku, 0.0))
        for sku in sorted(set(base["sku_revenue"]) | set((merged.get("sku_revenue") or {}).keys()))
    }
    merged["c_tard"] = _as_float(merged.get("c_tard"), base["c_tard"])
    merged["c_stock"] = _as_float(merged.get("c_stock"), base["c_stock"])
    merged["c_idle"] = _as_float(merged.get("c_idle"), base["c_idle"])
    merged["c_over"] = _as_float(merged.get("c_over"), base["c_over"])
    merged["c_setup"] = _as_float(merged.get("c_setup"), base.get("c_setup", 800.0))
    merged["roast_time_by_sku"] = merged.get("roast_time_by_sku", base.get("roast_time_by_sku", {"PSC": 15, "NDG": 17, "BUSTA": 18}))
    merged["gc_capacity"] = _normalize_pair_map(merged.get("gc_capacity", base.get("gc_capacity", {})))
    merged["gc_init"] = _normalize_pair_map(merged.get("gc_init", base.get("gc_init", {})))
    merged["restock_duration"] = _as_int(merged.get("restock_duration", base.get("restock_duration", 15)))
    merged["restock_qty"] = _as_int(merged.get("restock_qty", base.get("restock_qty", 5)))
    merged["feasible_gc_pairs"] = _normalize_pair_list(
        merged.get("feasible_gc_pairs", base.get("feasible_gc_pairs", []))
    )
    return merged


def create_result(
    metadata: dict[str, Any] | None = None,
    experiment: dict[str, Any] | None = None,
    kpi: dict[str, Any] | None = None,
    schedule: list[dict[str, Any]] | None = None,
    cancelled_batches: list[dict[str, Any]] | None = None,
    ups_events: list[dict[str, Any]] | None = None,
    parameters: dict[str, Any] | None = None,
    restocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a canonical result dict, filling defaults for missing fields."""

    result_metadata = _deep_merge(_default_metadata(), metadata)
    result_experiment = _deep_merge(_default_experiment(), experiment)
    result_kpi = _deep_merge(_default_kpi(), kpi)
    result_parameters = _canonical_parameters(parameters)
    normalized_schedule = [normalize_schedule_entry(entry) for entry in (schedule or [])]
    normalized_cancelled = []
    for entry in cancelled_batches or []:
        item = normalize_schedule_entry(entry)
        item["status"] = "cancelled"
        normalized_cancelled.append(item)
    normalized_ups = [_normalize_ups_event(event) for event in (ups_events or [])]
    normalized_restocks = reconstruct_restock_timeline(restocks) if restocks else []

    if not result_experiment.get("scenario_label"):
        result_experiment["scenario_label"] = "ups" if normalized_ups else "no_ups"
    elif normalized_ups and result_experiment["scenario_label"] == "no_ups":
        result_experiment["scenario_label"] = "ups"

    result_kpi["psc_count"] = _as_int(result_kpi.get("psc_count"))
    result_kpi["ndg_count"] = _as_int(result_kpi.get("ndg_count"))
    result_kpi["busta_count"] = _as_int(result_kpi.get("busta_count"))
    if not result_kpi.get("total_batches"):
        result_kpi["total_batches"] = (
            result_kpi["psc_count"] + result_kpi["ndg_count"] + result_kpi["busta_count"]
        )
    else:
        result_kpi["total_batches"] = _as_int(result_kpi.get("total_batches"))

    revenue_defaults = result_parameters["sku_revenue"]
    result_kpi["revenue_psc"] = _as_float(
        result_kpi.get("revenue_psc"),
        result_kpi["psc_count"] * revenue_defaults.get("PSC", 0.0),
    )
    result_kpi["revenue_ndg"] = _as_float(
        result_kpi.get("revenue_ndg"),
        result_kpi["ndg_count"] * revenue_defaults.get("NDG", 0.0),
    )
    result_kpi["revenue_busta"] = _as_float(
        result_kpi.get("revenue_busta"),
        result_kpi["busta_count"] * revenue_defaults.get("BUSTA", 0.0),
    )

    for field in (
        "net_profit",
        "total_revenue",
        "total_costs",
        "tard_cost",
        "setup_cost",
        "stockout_cost",
        "idle_min",
        "idle_cost",
        "over_min",
        "over_cost",
    ):
        result_kpi[field] = _as_float(result_kpi.get(field))
    result_kpi["setup_events"] = _as_int(result_kpi.get("setup_events"), 0)
    result_kpi["restock_count"] = _as_int(result_kpi.get("restock_count"), 0)
    if "gc_init" in result_kpi:
        result_kpi["gc_init"] = result_kpi["gc_init"]
    if "gc_final" in result_kpi:
        result_kpi["gc_final"] = result_kpi["gc_final"]
    for nested in ("tardiness_min", "stockout_events", "stockout_duration", "idle_min_per_roaster", "over_min_per_roaster"):
        result_kpi[nested] = {
            str(key): _as_float(value) if nested.endswith("_min") or nested.endswith("_roaster") else _as_int(value)
            for key, value in result_kpi.get(nested, {}).items()
        }
    # Correct integer-only nested KPI fields.
    result_kpi["stockout_events"] = {
        line: _as_int(result_kpi["stockout_events"].get(line), 0) for line in DEFAULT_LINES
    }
    result_kpi["stockout_duration"] = {
        line: _as_int(result_kpi["stockout_duration"].get(line), 0) for line in DEFAULT_LINES
    }
    result_kpi["tardiness_min"] = {
        job: _as_float(result_kpi["tardiness_min"].get(job), 0.0) for job in DEFAULT_JOBS
    }
    result_kpi["idle_min_per_roaster"] = {
        roaster: _as_float(result_kpi["idle_min_per_roaster"].get(roaster), 0.0)
        for roaster in DEFAULT_ROASTERS
    }
    result_kpi["over_min_per_roaster"] = {
        roaster: _as_float(result_kpi["over_min_per_roaster"].get(roaster), 0.0)
        for roaster in DEFAULT_ROASTERS
    }

    return {
        "metadata": result_metadata,
        "experiment": result_experiment,
        "kpi": result_kpi,
        "schedule": normalized_schedule,
        "cancelled_batches": normalized_cancelled,
        "ups_events": normalized_ups,
        "restocks": normalized_restocks,
        "parameters": result_parameters,
    }


def _infer_source(old_dict: dict[str, Any], schedule: list[dict[str, Any]] | None) -> str:
    if {"metadata", "experiment", "kpi", "schedule", "parameters"}.issubset(old_dict):
        return "universal"
    if "schedule" not in old_dict and schedule is None:
        return "simulation"
    solver_engine = str(old_dict.get("solver_engine", "")).lower()
    solver_name = str(old_dict.get("solver_name", "")).lower()
    if any(tag in solver_engine for tag in ("cp", "sat")) or any(tag in solver_name for tag in ("cp", "sat")):
        return "cpsat"
    if "solution_history" in old_dict or "num_incumbents" in old_dict:
        return "cpsat"
    if "inventory_option" in old_dict:
        return "milp"
    return "milp"


def convert_legacy_result(
    old_dict: dict[str, Any],
    source: str = "auto",
    schedule: list[dict[str, Any]] | None = None,
    cancelled: list[dict[str, Any]] | None = None,
    ups: list[dict[str, Any]] | None = None,
    experiment: dict[str, Any] | None = None,
    restocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert a legacy result dict into the universal schema."""

    if source == "auto":
        source = _infer_source(old_dict, schedule)
    if source == "universal":
        return create_result(**old_dict)

    solve_time = _as_float(old_dict.get("solve_time_sec", old_dict.get("solve_time", 0.0)))
    allow_r3_flex = _as_bool(old_dict.get("allow_r3_flex"), True)
    schedule_data = schedule if schedule is not None else old_dict.get("schedule", [])
    cancelled_data = cancelled if cancelled is not None else old_dict.get("cancelled_batches", [])
    ups_data = ups if ups is not None else old_dict.get("ups_events", [])

    metadata = {
        "solver_engine": old_dict.get("solver_engine"),
        "solver_name": old_dict.get("solver_name"),
        "status": old_dict.get("status", "Completed" if source == "simulation" else "Unknown"),
        "solve_time_sec": solve_time,
        "total_compute_ms": old_dict.get("total_compute_ms", solve_time * 1000.0),
        "num_resolves": old_dict.get("num_resolves", 0),
        "obj_value": old_dict.get("obj_value"),
        "best_bound": old_dict.get("best_bound", old_dict.get("lp_bound")),
        "gap_pct": old_dict.get("gap_pct"),
        "allow_r3_flex": allow_r3_flex,
        "notes": "",
    }
    if source == "simulation":
        metadata["solver_engine"] = metadata["solver_engine"] or "simulation"
        metadata["solver_name"] = metadata["solver_name"] or "Simulation"
    elif source == "cpsat":
        metadata["solver_engine"] = metadata["solver_engine"] or "cpsat"
        metadata["solver_name"] = metadata["solver_name"] or "CP-SAT"
    else:
        inventory_option = old_dict.get("inventory_option")
        metadata["solver_engine"] = metadata["solver_engine"] or "milp"
        metadata["solver_name"] = metadata["solver_name"] or "MILP"
        if inventory_option not in (None, ""):
            metadata["notes"] = f"inventory_option={inventory_option}"

    kpi = {
        "net_profit": old_dict.get("net_profit", 0.0),
        "total_revenue": old_dict.get("total_revenue", 0.0),
        "total_costs": old_dict.get("total_costs", 0.0),
        "psc_count": old_dict.get("psc_count", 0),
        "ndg_count": old_dict.get("ndg_count", 0),
        "busta_count": old_dict.get("busta_count", 0),
        "total_batches": old_dict.get("total_batches", 0),
        "revenue_psc": old_dict.get("revenue_psc", 0.0),
        "revenue_ndg": old_dict.get("revenue_ndg", 0.0),
        "revenue_busta": old_dict.get("revenue_busta", 0.0),
        "tardiness_min": old_dict.get("tardiness_min", {}),
        "tard_cost": old_dict.get("tard_cost", 0.0),
        "setup_events": old_dict.get("setup_events", 0),
        "setup_cost": old_dict.get("setup_cost", 0.0),
        "stockout_events": old_dict.get("stockout_events", {}),
        "stockout_duration": old_dict.get("stockout_duration", {}),
        "stockout_cost": old_dict.get("stockout_cost", 0.0),
        "idle_min": old_dict.get("idle_min", 0.0),
        "idle_min_per_roaster": old_dict.get("idle_min_per_roaster", {}),
        "idle_cost": old_dict.get("idle_cost", 0.0),
        "over_min": old_dict.get("over_min", 0.0),
        "over_min_per_roaster": old_dict.get("over_min_per_roaster", {}),
        "over_cost": old_dict.get("over_cost", 0.0),
        "restock_count": old_dict.get("restock_count", 0),
        "gc_final": old_dict.get("gc_final"),
    }

    parameter_overrides = {
        "sku_revenue": old_dict.get("sku_revenue"),
        "c_tard": old_dict.get("cost_tardiness"),
        "c_stock": old_dict.get("cost_stockout"),
        "c_idle": old_dict.get("cost_idle"),
        "c_over": old_dict.get("cost_overflow"),
        "gc_capacity": old_dict.get("gc_capacity"),
        "gc_init": old_dict.get("gc_init"),
        "restock_duration": old_dict.get("restock_duration"),
        "restock_qty": old_dict.get("restock_qty"),
    }
    parameter_overrides = {key: value for key, value in parameter_overrides.items() if value is not None}

    experiment_data = experiment or {
        "lambda_rate": old_dict.get("lambda_rate", 0),
        "mu_mean": old_dict.get("mu_mean", 0),
        "seed": old_dict.get("seed"),
        "replication": old_dict.get("replication"),
        "scenario_label": old_dict.get("scenario_label", "ups" if ups_data else "no_ups"),
    }

    restocks_data = restocks if restocks is not None else old_dict.get("restocks", [])

    return create_result(
        metadata=metadata,
        experiment=experiment_data,
        kpi=kpi,
        schedule=schedule_data,
        cancelled_batches=cancelled_data,
        ups_events=ups_data,
        parameters=parameter_overrides or None,
        restocks=restocks_data or None,
    )


def _consume_events(parameters: dict[str, Any]) -> dict[str, list[int]]:
    shift_length = _as_int(parameters.get("SL"), 480)
    events: dict[str, list[int]] = {}
    for line in DEFAULT_LINES:
        rate = _as_float((parameters.get("consume_rate") or {}).get(line), 0.0)
        if rate <= 0:
            events[line] = []
            continue
        count = int(math.floor(shift_length / rate))
        seen: list[int] = []
        for index in range(1, count + 1):
            slot = int(math.floor(index * rate))
            if slot < shift_length:
                seen.append(slot)
        events[line] = seen
    return events


def reconstruct_rc_trajectory(schedule: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct per-slot RC stock for both lines from the schedule."""

    params = _canonical_parameters(parameters)
    shift_length = params["SL"]
    trajectories: dict[str, list[int]] = {line: [] for line in DEFAULT_LINES}
    completion_map: dict[str, Counter[int]] = {line: Counter() for line in DEFAULT_LINES}
    consumption_events = _consume_events(params)

    for raw_entry in schedule:
        entry = normalize_schedule_entry(raw_entry)
        if entry["status"] != "completed":
            continue
        if entry["sku"] != "PSC":
            continue
        line = entry.get("output_line")
        if line not in DEFAULT_LINES:
            continue
        completion_map[line][entry["end"]] += 1

    for line in DEFAULT_LINES:
        rc = params["rc_init"][line]
        for slot in range(shift_length):
            rc += completion_map[line].get(slot, 0)
            if slot in consumption_events[line]:
                rc -= 1
            trajectories[line].append(rc)

    return {
        "L1": trajectories["L1"],
        "L2": trajectories["L2"],
        "L1_events": {
            "completions": sorted(completion_map["L1"].elements()),
            "consumptions": consumption_events["L1"],
        },
        "L2_events": {
            "completions": sorted(completion_map["L2"].elements()),
            "consumptions": consumption_events["L2"],
        },
    }


def reconstruct_gc_trajectory(
    schedule: list[dict[str, Any]],
    restocks: list[dict[str, Any]] | None,
    parameters: dict[str, Any],
    cancelled_batches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reconstruct per-slot GC stock for all feasible silo pairs.

    Returns dict with keys like "L1_PSC", "L1_NDG", "L1_BUSTA", "L2_PSC",
    each mapping to a list of 480 ints.
    Also returns "events" with completed/cancelled batch starts and restock timing.
    """
    params = _canonical_parameters(parameters)
    shift_length = params["SL"]
    gc_cap = params.get("gc_capacity", {"L1_PSC": 40, "L1_NDG": 10, "L1_BUSTA": 10, "L2_PSC": 40})
    gc_init_vals = params.get("gc_init", {"L1_PSC": 20, "L1_NDG": 5, "L1_BUSTA": 5, "L2_PSC": 20})
    feasible = list(gc_cap.keys())
    roast_times = params.get("roast_time_by_sku", {"PSC": 15, "NDG": 17, "BUSTA": 18})

    refs = load_reference_data()
    roaster_pipeline = refs.get("roaster_pipeline", {
        "R1": "L1", "R2": "L1", "R3": "L2", "R4": "L2", "R5": "L2"
    })

    def _collect_batch_starts(entries: list[dict[str, Any]] | None) -> dict[str, list[int]]:
        starts: dict[str, list[int]] = {pair: [] for pair in feasible}
        for raw_entry in entries or []:
            entry = normalize_schedule_entry(raw_entry)
            roaster = entry["roaster"]
            sku = entry["sku"]
            pipe_line = roaster_pipeline.get(roaster, "L1")
            pair_key = f"{pipe_line}_{sku}"
            if pair_key in starts:
                starts[pair_key].append(entry["start"])
        return starts

    completed_batch_starts = _collect_batch_starts(schedule)
    cancelled_batch_starts = _collect_batch_starts(cancelled_batches)
    batch_starts = {
        pair_key: sorted(completed_batch_starts[pair_key] + cancelled_batch_starts[pair_key])
        for pair_key in feasible
    }

    restock_starts: dict[str, list[int]] = {p: [] for p in feasible}
    restock_completions: dict[str, list[int]] = {p: [] for p in feasible}
    restock_timeline = reconstruct_restock_timeline(restocks)
    for rst in restock_timeline:
        pair_key = rst["pair"]
        if pair_key in restock_completions:
            restock_starts[pair_key].append(rst["start"])
            restock_completions[pair_key].append(rst["end"])

    trajectories: dict[str, list[int]] = {}
    for pair_key in feasible:
        gc = gc_init_vals.get(pair_key, 0)
        starts_set = Counter(batch_starts[pair_key])
        completions_set = Counter(restock_completions[pair_key])
        traj = []
        for slot in range(shift_length):
            gc += completions_set.get(slot, 0) * _as_int(params.get("restock_qty", 5))
            gc -= starts_set.get(slot, 0)
            traj.append(gc)
        trajectories[pair_key] = traj

    result = dict(trajectories)
    result["events"] = {
        pair_key: {
            "batch_starts": sorted(batch_starts[pair_key]),
            "completed_batch_starts": sorted(completed_batch_starts[pair_key]),
            "cancelled_batch_starts": sorted(cancelled_batch_starts[pair_key]),
            "restock_starts": sorted(restock_starts[pair_key]),
            "restock_completions": sorted(restock_completions[pair_key]),
        }
        for pair_key in feasible
    }
    return result


def reconstruct_restock_timeline(restocks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize restock records into a clean timeline."""
    if not restocks:
        return []
    timeline = []
    for rst in restocks:
        timeline.append({
            "line_id": str(rst.get("line_id", "")),
            "sku": str(rst.get("sku", "")),
            "start": _as_int(rst.get("start"), 0),
            "end": _as_int(rst.get("end"), 0),
            "qty": _as_int(rst.get("qty"), 5),
            "pair": f"{rst.get('line_id', '')}_{rst.get('sku', '')}",
        })
    return sorted(timeline, key=lambda r: r["start"])


def validate_result(result: dict[str, Any]) -> list[str]:
    """Return schema and consistency validation errors for a result dict."""

    errors: list[str] = []
    required_sections = ("metadata", "experiment", "kpi", "schedule", "cancelled_batches", "ups_events", "parameters")
    for section in required_sections:
        if section not in result:
            errors.append(f"Missing section: {section}")

    if errors:
        return errors

    metadata = result["metadata"]
    experiment = result["experiment"]
    kpi = result["kpi"]
    parameters = result["parameters"]

    if not isinstance(result["schedule"], list):
        errors.append("Schedule must be a list.")
    if not isinstance(result["cancelled_batches"], list):
        errors.append("cancelled_batches must be a list.")
    if not isinstance(result["ups_events"], list):
        errors.append("ups_events must be a list.")

    for section_name, section in (("metadata", metadata), ("experiment", experiment), ("kpi", kpi), ("parameters", parameters)):
        if not isinstance(section, dict):
            errors.append(f"{section_name} must be a dict.")

    if errors:
        return errors

    try:
        net_profit = _as_float(kpi.get("net_profit"))
        total_revenue = _as_float(kpi.get("total_revenue"))
        total_costs = _as_float(kpi.get("total_costs"))
        if abs(net_profit - (total_revenue - total_costs)) > 0.1:
            errors.append(
                f"Accounting mismatch: net_profit={net_profit} vs total_revenue-total_costs={total_revenue - total_costs}"
            )
    except (TypeError, ValueError):
        errors.append("Accounting values must be numeric.")

    total_batches = _as_int(kpi.get("total_batches"))
    counted_batches = _as_int(kpi.get("psc_count")) + _as_int(kpi.get("ndg_count")) + _as_int(kpi.get("busta_count"))
    if total_batches != counted_batches:
        errors.append(f"Batch count mismatch: total_batches={total_batches} vs counts sum={counted_batches}")

    try:
        _canonical_parameters(parameters)
    except Exception as exc:  # pragma: no cover - defensive validation
        errors.append(f"Invalid parameters section: {exc}")

    for index, raw_entry in enumerate(result["schedule"]):
        entry = normalize_schedule_entry(raw_entry)
        required = ("batch_id", "sku", "roaster", "start", "end", "pipeline", "pipeline_start", "pipeline_end", "status")
        for field in required:
            if entry.get(field) in ("", None) and field not in {"output_line", "job_id"}:
                errors.append(f"Schedule entry {index} missing field: {field}")
        if entry["end"] < entry["start"]:
            errors.append(f"Schedule entry {index} has end < start.")

    for index, event in enumerate(result["ups_events"]):
        if not isinstance(event, dict):
            errors.append(f"UPS event {index} is not a dict.")
            continue
        if "t" not in event or "roaster_id" not in event or "duration" not in event:
            errors.append(f"UPS event {index} missing required fields.")

    if experiment.get("scenario_label") == "no_ups" and result["ups_events"]:
        errors.append("Experiment says no_ups but ups_events is non-empty.")

    return errors


def save_result(result: dict[str, Any], path: str | Path) -> Path:
    """Validate then write a universal-format JSON file."""

    errors = validate_result(result)
    if errors:
        raise ValueError("Result validation failed:\n- " + "\n- ".join(errors))

    target = Path(path)
    if not target.is_absolute():
        target = ROOT_DIR / target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=False)
    return target


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a result JSON and warn if validation detects issues."""

    source = Path(path)
    if not source.is_absolute():
        source = ROOT_DIR / source
    with source.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    errors = validate_result(result)
    for error in errors:
        logger.warning("Result validation warning for %s: %s", source, error)
    return result
