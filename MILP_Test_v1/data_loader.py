"""
data_loader.py — CSV parsing and validation for MILP scheduling solver.

Reads all input CSVs from the Input_data_sample folder and builds a
ShiftData object that the MILP solver and GUI consume.
"""

import os
import math
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class SKU:
    name: str
    is_mto: bool
    credits_rc: bool
    revenue: float
    default_due_time: Optional[int]
    batch_size_kg: float


@dataclass
class Roaster:
    roaster_id: str
    line_id: str
    pipeline_line: str
    eligible_skus: Set[str]
    can_output_l1: bool
    can_output_l2: bool
    process_time: int
    consume_time: int
    initial_last_sku: str
    is_active: bool


@dataclass
class Job:
    job_id: str
    sku: str
    required_batches: int
    due_time: int
    priority: int
    release_time: int


@dataclass
class PlannedDowntime:
    roaster_id: str
    start_min: int
    end_min: int
    reason: str


@dataclass
class Disruption:
    time_min: int
    roaster_id: str
    event_type: str
    duration_min: int
    note: str


@dataclass
class ShiftData:
    """Complete shift input data — single object passed to solver and GUI."""
    # Timing
    shift_length: int
    time_step: int

    # RC parameters
    max_rc_per_line: int
    safety_stock: int
    setup_time: int

    # Cost parameters
    stockout_cost: float
    tardiness_cost: float
    idle_cost: float
    overflow_idle_cost: float

    # Initial RC stock
    initial_rc_l1: int
    initial_rc_l2: int

    # PSC consumption rates (minutes per batch)
    consume_rate_l1: float
    consume_rate_l2: float

    # Derived consumption schedules
    consumption_schedule_l1: List[int] = field(default_factory=list)
    consumption_schedule_l2: List[int] = field(default_factory=list)

    # Entities
    skus: Dict[str, SKU] = field(default_factory=dict)
    roasters: Dict[str, Roaster] = field(default_factory=dict)
    jobs: List[Job] = field(default_factory=list)
    planned_downtime: List[PlannedDowntime] = field(default_factory=list)
    disruptions: List[Disruption] = field(default_factory=list)

    # Solver config
    solver_config: Dict[str, str] = field(default_factory=dict)

    # Derived sets
    lines: List[str] = field(default_factory=lambda: ["L1", "L2"])
    roaster_ids_by_line: Dict[str, List[str]] = field(default_factory=dict)
    roaster_ids_by_pipeline: Dict[str, List[str]] = field(default_factory=dict)
    eligible_roasters_by_sku: Dict[str, List[str]] = field(default_factory=dict)
    downtime_slots: Dict[str, Set[int]] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────

def _validate_columns(filepath: str, rows: List[dict], required: List[str]):
    """Raise ValueError if any required column is missing."""
    if not rows:
        raise ValueError(f"Empty CSV: {filepath}")
    actual = set(rows[0].keys())
    missing = set(required) - actual
    if missing:
        raise ValueError(
            f"Missing columns in {os.path.basename(filepath)}: {missing}. "
            f"Found: {actual}"
        )


def _read_csv(filepath: str) -> List[dict]:
    """Read CSV file and return list of dicts."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ──────────────────────────────────────────────
# Individual loaders
# ──────────────────────────────────────────────

def _load_skus(input_dir: str) -> Dict[str, SKU]:
    fp = os.path.join(input_dir, "skus.csv")
    rows = _read_csv(fp)
    _validate_columns(fp, rows, [
        "sku", "is_mto", "credits_rc_stock",
        "revenue_per_batch_usd", "batch_size_kg"
    ])
    skus = {}
    for r in rows:
        due = int(r["default_due_time_min"]) if r.get("default_due_time_min") else None
        skus[r["sku"]] = SKU(
            name=r["sku"],
            is_mto=bool(int(r["is_mto"])),
            credits_rc=bool(int(r["credits_rc_stock"])),
            revenue=float(r["revenue_per_batch_usd"]),
            default_due_time=due,
            batch_size_kg=float(r["batch_size_kg"]),
        )
    return skus


def _load_roasters(input_dir: str) -> Dict[str, Roaster]:
    fp = os.path.join(input_dir, "roasters.csv")
    rows = _read_csv(fp)
    _validate_columns(fp, rows, [
        "roaster_id", "line_id", "pipeline_line",
        "can_process_psc", "can_process_ndg", "can_process_busta",
        "can_output_l1", "can_output_l2",
        "process_time_min", "consume_time_min",
        "initial_last_sku", "is_active"
    ])
    roasters = {}
    for r in rows:
        eligible = set()
        if int(r["can_process_psc"]):
            eligible.add("PSC")
        if int(r["can_process_ndg"]):
            eligible.add("NDG")
        if int(r["can_process_busta"]):
            eligible.add("BUSTA")
        roasters[r["roaster_id"]] = Roaster(
            roaster_id=r["roaster_id"],
            line_id=r["line_id"],
            pipeline_line=r["pipeline_line"],
            eligible_skus=eligible,
            can_output_l1=bool(int(r["can_output_l1"])),
            can_output_l2=bool(int(r["can_output_l2"])),
            process_time=int(r["process_time_min"]),
            consume_time=int(r["consume_time_min"]),
            initial_last_sku=r["initial_last_sku"],
            is_active=bool(int(r["is_active"])),
        )
    return roasters


def _load_jobs(input_dir: str) -> List[Job]:
    fp = os.path.join(input_dir, "jobs.csv")
    rows = _read_csv(fp)
    _validate_columns(fp, rows, [
        "job_id", "sku", "required_batches",
        "due_time_min", "priority", "release_time_min"
    ])
    return [
        Job(
            job_id=r["job_id"],
            sku=r["sku"],
            required_batches=int(r["required_batches"]),
            due_time=int(r["due_time_min"]),
            priority=int(r["priority"]),
            release_time=int(r["release_time_min"]),
        )
        for r in rows
    ]


def _load_shift_params(input_dir: str) -> dict:
    fp = os.path.join(input_dir, "shift_parameters.csv")
    rows = _read_csv(fp)
    _validate_columns(fp, rows, [
        "shift_length_min", "max_rc_batches_per_line",
        "safety_stock_batches", "setup_time_diff_sku_min",
        "stockout_cost_per_min_per_line", "tardiness_cost_per_min",
        "idle_cost_per_min_per_roaster", "overflow_idle_cost_per_min_per_roaster",
        "initial_rc_l1", "initial_rc_l2",
        "psc_consume_rate_l1_min_per_batch", "psc_consume_rate_l2_min_per_batch"
    ])
    return rows[0]  # single row


def _load_planned_downtime(input_dir: str) -> List[PlannedDowntime]:
    fp = os.path.join(input_dir, "planned_downtime.csv")
    rows = _read_csv(fp)
    _validate_columns(fp, rows, ["roaster_id", "start_min", "end_min", "reason"])
    return [
        PlannedDowntime(
            roaster_id=r["roaster_id"],
            start_min=int(r["start_min"]),
            end_min=int(r["end_min"]),
            reason=r["reason"],
        )
        for r in rows
    ]


def _load_disruptions(input_dir: str) -> List[Disruption]:
    fp = os.path.join(input_dir, "manual_disruptions_template.csv")
    if not os.path.exists(fp):
        return []
    rows = _read_csv(fp)
    _validate_columns(fp, rows, [
        "time_min", "roaster_id", "event_type", "duration_min"
    ])
    return [
        Disruption(
            time_min=int(r["time_min"]),
            roaster_id=r["roaster_id"],
            event_type=r["event_type"],
            duration_min=int(r["duration_min"]),
            note=r.get("note", ""),
        )
        for r in rows
    ]


def _load_solver_config(input_dir: str) -> Dict[str, str]:
    fp = os.path.join(input_dir, "solver_config.csv")
    rows = _read_csv(fp)
    _validate_columns(fp, rows, ["parameter", "value"])
    return {r["parameter"]: r["value"] for r in rows}


# ──────────────────────────────────────────────
# Derived computations
# ──────────────────────────────────────────────

def _compute_consumption_schedule(shift_length: int, rate: float) -> List[int]:
    """
    Compute PSC consumption event times.
    E_l = {floor(i * rho_l) | i = 1, 2, ..., floor(shift_length / rho_l)}
    """
    n_events = int(math.floor(shift_length / rate))
    return [int(math.floor(i * rate)) for i in range(1, n_events + 1)]


def _compute_downtime_slots(
    planned: List[PlannedDowntime],
    roaster_ids: List[str],
) -> Dict[str, Set[int]]:
    """Build a dict of roaster_id → set of unavailable time slots."""
    slots = {rid: set() for rid in roaster_ids}
    for dt in planned:
        if dt.roaster_id in slots:
            for t in range(dt.start_min, dt.end_min):
                slots[dt.roaster_id].add(t)
    return slots


# ──────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────

def load_all_data(input_dir: str) -> ShiftData:
    """
    Load all CSV files from input_dir and build a ShiftData object.
    Validates columns before constructing objects.
    """
    # Load raw data
    skus = _load_skus(input_dir)
    roasters = _load_roasters(input_dir)
    jobs = _load_jobs(input_dir)
    sp = _load_shift_params(input_dir)
    planned_dt = _load_planned_downtime(input_dir)
    disruptions = _load_disruptions(input_dir)
    solver_config = _load_solver_config(input_dir)

    shift_length = int(sp["shift_length_min"])
    consume_rate_l1 = float(sp["psc_consume_rate_l1_min_per_batch"])
    consume_rate_l2 = float(sp["psc_consume_rate_l2_min_per_batch"])

    # Derived: consumption schedules
    cons_l1 = _compute_consumption_schedule(shift_length, consume_rate_l1)
    cons_l2 = _compute_consumption_schedule(shift_length, consume_rate_l2)

    # Derived: roasters by line and pipeline
    roaster_ids_by_line: Dict[str, List[str]] = {"L1": [], "L2": []}
    roaster_ids_by_pipeline: Dict[str, List[str]] = {"L1": [], "L2": []}

    for rid, r in roasters.items():
        if r.is_active:
            roaster_ids_by_line.setdefault(r.line_id, []).append(rid)
            roaster_ids_by_pipeline.setdefault(r.pipeline_line, []).append(rid)

    # Derived: eligible roasters per SKU
    eligible_by_sku: Dict[str, List[str]] = {}
    for sku_name in skus:
        eligible_by_sku[sku_name] = [
            rid for rid, r in roasters.items()
            if r.is_active and sku_name in r.eligible_skus
        ]

    # Derived: downtime slots
    all_rids = [rid for rid in roasters]
    dt_slots = _compute_downtime_slots(planned_dt, all_rids)

    data = ShiftData(
        shift_length=shift_length,
        time_step=int(sp["time_step_min"]),
        max_rc_per_line=int(sp["max_rc_batches_per_line"]),
        safety_stock=int(sp["safety_stock_batches"]),
        setup_time=int(sp["setup_time_diff_sku_min"]),
        stockout_cost=float(sp["stockout_cost_per_min_per_line"]),
        tardiness_cost=float(sp["tardiness_cost_per_min"]),
        idle_cost=float(sp["idle_cost_per_min_per_roaster"]),
        overflow_idle_cost=float(sp["overflow_idle_cost_per_min_per_roaster"]),
        initial_rc_l1=int(sp["initial_rc_l1"]),
        initial_rc_l2=int(sp["initial_rc_l2"]),
        consume_rate_l1=consume_rate_l1,
        consume_rate_l2=consume_rate_l2,
        consumption_schedule_l1=cons_l1,
        consumption_schedule_l2=cons_l2,
        skus=skus,
        roasters=roasters,
        jobs=jobs,
        planned_downtime=planned_dt,
        disruptions=disruptions,
        solver_config=solver_config,
        roaster_ids_by_line=roaster_ids_by_line,
        roaster_ids_by_pipeline=roaster_ids_by_pipeline,
        eligible_roasters_by_sku=eligible_by_sku,
        downtime_slots=dt_slots,
    )
    return data


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "../Input_data_sample"
    d = load_all_data(input_dir)

    print(f"Shift length:  {d.shift_length} min")
    print(f"Roasters:      {len(d.roasters)} ({', '.join(d.roasters.keys())})")
    print(f"SKUs:          {len(d.skus)} ({', '.join(d.skus.keys())})")
    print(f"MTO Jobs:      {len(d.jobs)}")
    for j in d.jobs:
        print(f"  {j.job_id}: {j.sku} × {j.required_batches}, due={j.due_time}")
    print(f"Planned DT:    {len(d.planned_downtime)}")
    print(f"Consume L1:    {len(d.consumption_schedule_l1)} events (rate={d.consume_rate_l1} min/batch)")
    print(f"Consume L2:    {len(d.consumption_schedule_l2)} events (rate={d.consume_rate_l2} min/batch)")
    print(f"Initial RC:    L1={d.initial_rc_l1}, L2={d.initial_rc_l2}")
    print(f"Max RC:        {d.max_rc_per_line} batches/line")
    print(f"Safety stock:  {d.safety_stock}")
    print(f"Setup time:    {d.setup_time} min")
    print(f"Costs:         stockout=${d.stockout_cost}/min, tard=${d.tardiness_cost}/min")
    print(f"               idle=${d.idle_cost}/min, overflow=${d.overflow_idle_cost}/min")

    for sku_name, rids in d.eligible_roasters_by_sku.items():
        print(f"SKU {sku_name}: eligible on {rids}")

    for rid, slots in d.downtime_slots.items():
        if slots:
            print(f"Downtime {rid}: {min(slots)}-{max(slots)} ({len(slots)} slots)")
