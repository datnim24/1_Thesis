"""Load all simulation parameters directly from Input_data/ CSVs.

This is the SINGLE entry point for environment configuration.
No dependency on MILP_Test_v3 or any solver-specific module.
"""

from __future__ import annotations

import csv
import math
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("data_bridge")

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INPUT_DIR = _ROOT / "Input_data"

_REQUIRED_SHIFT_KEYS = (
    "shift_length_min",
    "max_rc_batches_per_line",
    "safety_stock_batches",
    "setup_time_diff_sku_min",
    "initial_rc_l1",
    "initial_rc_l2",
    "psc_consume_rate_l1_min_per_batch",
    "psc_consume_rate_l2_min_per_batch",
    "stockout_cost_per_event_per_line",
    "tardiness_cost_per_min",
    "idle_cost_per_min_per_roaster",
    "overflow_idle_cost_per_min_per_roaster",
    "setup_cost_per_event",
)

_SKU_FLAG_COLUMNS = (
    ("PSC", "can_process_psc"),
    ("NDG", "can_process_ndg"),
    ("BUSTA", "can_process_busta"),
)

_OUTPUT_FLAG_COLUMNS = (
    ("L1", "can_output_l1"),
    ("L2", "can_output_l2"),
)


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise ValueError(f"Missing required CSV file: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_param_map(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in _read_csv_rows(path):
        key = (row.get("parameter") or "").strip()
        val = row.get("value")
        if key:
            values[key] = "" if val is None else val.strip()
    return values


def _int(raw, label: str) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {label}: {raw!r}") from exc


def _float(raw, label: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {label}: {raw!r}") from exc


def _bool(raw, label: str) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip()
    if text in {"0", "1"}:
        return text == "1"
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    raise ValueError(f"Invalid boolean for {label}: {raw!r}")


def _resolve_input_dir(input_dir: str | Path | None) -> Path:
    if input_dir is None:
        return _DEFAULT_INPUT_DIR
    p = Path(input_dir)
    return p if p.is_absolute() else _ROOT / p


def get_sim_params(input_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a flat parameter dict loaded entirely from Input_data/ CSVs."""

    inp = _resolve_input_dir(input_dir)

    # --- shift_parameters.csv ---
    sp = _read_param_map(inp / "shift_parameters.csv")
    for k in _REQUIRED_SHIFT_KEYS:
        if k not in sp:
            raise ValueError(f"Missing required key '{k}' in shift_parameters.csv")

    shift_length = _int(sp["shift_length_min"], "shift_length_min")
    setup_time = _int(sp["setup_time_diff_sku_min"], "setup_time_diff_sku_min")
    max_rc = _int(sp["max_rc_batches_per_line"], "max_rc_batches_per_line")
    safety_stock = _int(sp["safety_stock_batches"], "safety_stock_batches")

    rc_init = {
        "L1": _int(sp["initial_rc_l1"], "initial_rc_l1"),
        "L2": _int(sp["initial_rc_l2"], "initial_rc_l2"),
    }
    consume_rate = {
        "L1": _float(sp["psc_consume_rate_l1_min_per_batch"], "psc_consume_rate_l1"),
        "L2": _float(sp["psc_consume_rate_l2_min_per_batch"], "psc_consume_rate_l2"),
    }

    c_stock = _float(sp["stockout_cost_per_event_per_line"], "stockout_cost")
    c_tard = _float(sp["tardiness_cost_per_min"], "tardiness_cost")
    c_idle = _float(sp["idle_cost_per_min_per_roaster"], "idle_cost")
    c_over = _float(sp["overflow_idle_cost_per_min_per_roaster"], "overflow_cost")
    c_setup = _float(sp["setup_cost_per_event"], "setup_cost")
    c_skip_mto = _float(sp.get("mto_skip_penalty_per_batch", "50000"), "mto_skip_penalty")

    ups_lambda = _float(sp["ups_lambda"], "ups_lambda") if "ups_lambda" in sp else 2.0
    ups_mu = _float(sp["ups_mu"], "ups_mu") if "ups_mu" in sp else 20.0

    # GC silo parameters
    gc_capacity: dict[tuple[str, str], int] = {}
    gc_init: dict[tuple[str, str], int] = {}
    _GC_CAP_KEYS = {
        "gc_capacity_l1_psc": ("L1", "PSC"),
        "gc_capacity_l1_ndg": ("L1", "NDG"),
        "gc_capacity_l1_busta": ("L1", "BUSTA"),
        "gc_capacity_l2_psc": ("L2", "PSC"),
    }
    _GC_INIT_KEYS = {
        "gc_initial_l1_psc": ("L1", "PSC"),
        "gc_initial_l1_ndg": ("L1", "NDG"),
        "gc_initial_l1_busta": ("L1", "BUSTA"),
        "gc_initial_l2_psc": ("L2", "PSC"),
    }
    for csv_key, pair in _GC_CAP_KEYS.items():
        gc_capacity[pair] = _int(sp.get(csv_key, "0"), csv_key)
    for csv_key, pair in _GC_INIT_KEYS.items():
        gc_init[pair] = _int(sp.get(csv_key, "0"), csv_key)

    feasible_gc_pairs = list(gc_capacity.keys())

    restock_duration = _int(sp.get("restock_duration_min", "15"), "restock_duration_min")
    restock_qty = _int(sp.get("restock_qty_batches", "5"), "restock_qty_batches")

    # --- solver_config.csv (only for allow_r3_flex) ---
    sc = _read_param_map(inp / "solver_config.csv")
    allow_r3_flex = _bool(sc.get("allow_r3_flexible_output", "1"), "allow_r3_flex")

    # --- roasters.csv ---
    roaster_rows = _read_csv_rows(inp / "roasters.csv")
    active_rows = [r for r in roaster_rows if _bool(r.get("is_active", "0"), "is_active")]
    if not active_rows:
        raise ValueError("No active roasters found in roasters.csv")

    roasters: list[str] = []
    lines_set: set[str] = set()
    R_line: dict[str, str] = {}
    R_pipe: dict[str, str] = {}
    R_out: dict[str, list[str]] = {}
    R_elig_skus: dict[str, list[str]] = {}
    roaster_initial_sku: dict[str, str] = {}
    consume_time_by_roaster: dict[str, int] = {}

    for row in active_rows:
        rid = row["roaster_id"].strip()
        roasters.append(rid)
        lid = row["line_id"].strip()
        lines_set.add(lid)
        R_line[rid] = lid
        R_pipe[rid] = row["pipeline_line"].strip()
        R_out[rid] = [
            line_id for line_id, col in _OUTPUT_FLAG_COLUMNS
            if _bool(row.get(col, "0"), f"{rid}.{col}")
        ]
        R_elig_skus[rid] = [
            sku for sku, col in _SKU_FLAG_COLUMNS
            if _bool(row.get(col, "0"), f"{rid}.{col}")
        ]
        roaster_initial_sku[rid] = (row.get("initial_last_sku") or "PSC").strip()
        consume_time_by_roaster[rid] = _int(row.get("consume_time_min", "3"), f"{rid}.consume_time_min")

    lines = sorted(lines_set)
    DC = consume_time_by_roaster.get(roasters[0], 3)

    # --- skus.csv ---
    sku_rows = _read_csv_rows(inp / "skus.csv")
    skus = [r["sku"].strip() for r in sku_rows]
    sku_revenue: dict[str, float] = {}
    sku_credits_rc: dict[str, bool] = {}
    roast_time_by_sku: dict[str, int] = {}

    for row in sku_rows:
        s = row["sku"].strip()
        sku_revenue[s] = _float(row["revenue_per_batch_usd"], f"{s}.revenue")
        sku_credits_rc[s] = _bool(row.get("credits_rc_stock", "0"), f"{s}.credits_rc")
        roast_time_by_sku[s] = _int(row["roast_time_min"], f"{s}.roast_time_min")
    psc_pool_per_roaster = math.floor(shift_length / roast_time_by_sku["PSC"])

    # --- jobs.csv ---
    job_rows = _read_csv_rows(inp / "jobs.csv")
    jobs = [r["job_id"].strip() for r in job_rows]
    job_sku: dict[str, str] = {}
    job_batches: dict[str, int] = {}
    job_due: dict[str, int] = {}

    for row in job_rows:
        jid = row["job_id"].strip()
        job_sku[jid] = row["sku"].strip()
        job_batches[jid] = _int(row["required_batches"], f"{jid}.required_batches")
        job_due[jid] = _int(row["due_time_min"], f"{jid}.due_time_min")

    # --- planned_downtime.csv ---
    dt_rows = _read_csv_rows(inp / "planned_downtime.csv")
    downtime_slots: dict[str, set[int]] = {rid: set() for rid in roasters}
    for row in dt_rows:
        rid = (row.get("roaster_id") or "").strip()
        if rid not in downtime_slots:
            continue
        s = _int(row["start_min"], f"{rid}.start_min")
        e = _int(row["end_min"], f"{rid}.end_min")
        downtime_slots[rid].update(range(s, e + 1))

    # --- derived: batch pools ---
    mto_batches = [
        (jid, idx)
        for jid in jobs
        for idx in range(job_batches[jid])
    ]
    psc_pool = [
        (rid, idx)
        for rid in roasters
        for idx in range(psc_pool_per_roaster)
    ]
    all_batches = mto_batches + psc_pool

    batch_sku_map = {bid: job_sku[bid[0]] for bid in mto_batches}
    batch_sku_map.update({bid: "PSC" for bid in psc_pool})

    batch_is_mto = {bid: True for bid in mto_batches}
    batch_is_mto.update({bid: False for bid in psc_pool})

    # --- derived: consumption events ---
    consume_events: dict[str, list[int]] = {}
    for lid in lines:
        rate = consume_rate[lid]
        n = math.floor(shift_length / rate)
        consume_events[lid] = [math.floor(i * rate) for i in range(1, n + 1)]

    # --- derived: PSC roast time for pool sizing ---
    psc_roast = roast_time_by_sku.get("PSC", 15)

    return {
        "roasters": roasters,
        "lines": lines,
        "R_line": R_line,
        "R_pipe": R_pipe,
        "R_out": R_out,
        "R_elig_skus": R_elig_skus,
        "roaster_initial_sku": roaster_initial_sku,
        "allow_r3_flex": allow_r3_flex,
        "roast_time_by_sku": roast_time_by_sku,
        "DC": DC,
        "sigma": setup_time,
        "SL": shift_length,
        "max_rc": max_rc,
        "safety_stock": safety_stock,
        "rc_init": rc_init,
        "consume_events": consume_events,
        "downtime_slots": downtime_slots,
        "jobs": jobs,
        "job_batches": job_batches,
        "job_sku": job_sku,
        "mto_batches": mto_batches,
        "batch_sku": batch_sku_map,
        "batch_is_mto": batch_is_mto,
        "job_due": job_due,
        "psc_pool": psc_pool,
        "all_batches": all_batches,
        "sku_credits_rc": sku_credits_rc,
        "sku_revenue": sku_revenue,
        "rev_psc": sku_revenue.get("PSC", 4000.0),
        "rev_ndg": sku_revenue.get("NDG", 7000.0),
        "rev_busta": sku_revenue.get("BUSTA", 7000.0),
        "c_tard": c_tard,
        "c_stock": c_stock,
        "c_idle": c_idle,
        "c_over": c_over,
        "c_setup": c_setup,
        "c_skip_mto": c_skip_mto,
        "ups_lambda": ups_lambda,
        "ups_mu": ups_mu,
        "gc_capacity": gc_capacity,
        "gc_init": gc_init,
        "feasible_gc_pairs": feasible_gc_pairs,
        "restock_duration": restock_duration,
        "restock_qty": restock_qty,
    }
