"""Stage 1 data loader for the deterministic MILP benchmark."""

from __future__ import annotations

import csv
import logging
import math
import sys
from pathlib import Path


logger = logging.getLogger("data")


REQUIRED_FILES = (
    "roasters.csv",
    "skus.csv",
    "jobs.csv",
    "shift_parameters.csv",
    "planned_downtime.csv",
    "manual_disruptions_template.csv",
    "solver_config.csv",
)

REQUIRED_SHIFT_KEYS = (
    "shift_length_min",
    "max_rc_batches_per_line",
    "safety_stock_batches",
    "setup_time_diff_sku_min",
    "psc_pool_size_per_roaster",
    "initial_rc_l1",
    "initial_rc_l2",
    "psc_consume_rate_l1_min_per_batch",
    "psc_consume_rate_l2_min_per_batch",
    "stockout_cost_per_event_per_line",
    "tardiness_cost_per_min",
    "idle_cost_per_min_per_roaster",
    "overflow_idle_cost_per_min_per_roaster",
)

REQUIRED_SOLVER_KEYS = (
    "solver_name",
    "time_limit_sec",
    "mip_gap_target",
    "allow_r3_flexible_output",
    "enable_disruptions",
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


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        message = f"Missing required CSV file: {path}"
        logger.error(message)
        raise ValueError(message)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    logger.debug("Read %d rows from %s", len(rows), path.name)
    return rows


def _read_parameter_map(path: Path, required_keys: tuple[str, ...]) -> dict[str, str]:
    rows = _read_csv_rows(path)
    values: dict[str, str] = {}
    for row in rows:
        parameter = (row.get("parameter") or "").strip()
        value = row.get("value")
        if parameter:
            values[parameter] = "" if value is None else value.strip()

    for key in required_keys:
        if key not in values:
            message = f"Missing required parameter key '{key}' in {path.name}"
            logger.error(message)
            raise ValueError(message)

    return values


def _parse_int(raw_value, label: str) -> int:
    try:
        return int(float(raw_value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {label}: {raw_value!r}") from exc


def _parse_float(raw_value, label: str) -> float:
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {label}: {raw_value!r}") from exc


def _parse_bool_flag(raw_value, label: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(int(raw_value) == 1)
    text = str(raw_value).strip()
    if text in {"0", "1"}:
        return bool(int(text) == 1)
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    raise ValueError(f"Invalid boolean flag for {label}: {raw_value!r}")


def _is_truthy_flag(raw_value) -> bool:
    try:
        return _parse_bool_flag(raw_value, "flag")
    except ValueError:
        return False


def _is_blank_row(row: dict) -> bool:
    return all((value or "").strip() == "" for value in row.values())


def _format_number(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def load(
    input_dir: str = "Input_data",
    overrides: dict = None,
) -> dict:
    input_path = Path(input_dir)

    for file_name in REQUIRED_FILES:
        file_path = input_path / file_name
        if not file_path.exists():
            message = f"Missing required CSV file: {file_path}"
            logger.error(message)
            raise ValueError(message)

    roaster_rows = _read_csv_rows(input_path / "roasters.csv")
    sku_rows = _read_csv_rows(input_path / "skus.csv")
    job_rows = _read_csv_rows(input_path / "jobs.csv")
    shift_params = _read_parameter_map(
        input_path / "shift_parameters.csv",
        REQUIRED_SHIFT_KEYS,
    )
    downtime_rows = _read_csv_rows(input_path / "planned_downtime.csv")
    disruption_rows = _read_csv_rows(input_path / "manual_disruptions_template.csv")
    solver_config = _read_parameter_map(
        input_path / "solver_config.csv",
        REQUIRED_SOLVER_KEYS,
    )

    active_roaster_rows = [
        row for row in roaster_rows if _parse_bool_flag(row.get("is_active", "0"), "is_active")
    ]
    roasters = [row["roaster_id"].strip() for row in active_roaster_rows]
    lines = sorted({row["line_id"].strip() for row in active_roaster_rows})

    roaster_line: dict[str, str] = {}
    roaster_pipeline: dict[str, str] = {}
    roaster_can_output: dict[str, list[str]] = {}
    roaster_eligible_skus: dict[str, list[str]] = {}
    roaster_initial_sku: dict[str, str] = {}

    process_time = None
    consume_time = None
    for row in active_roaster_rows:
        roaster_id = row["roaster_id"].strip()
        roaster_line[roaster_id] = row["line_id"].strip()
        roaster_pipeline[roaster_id] = row["pipeline_line"].strip()
        roaster_can_output[roaster_id] = [
            line_id
            for line_id, column_name in OUTPUT_FLAG_COLUMNS
            if _parse_bool_flag(row.get(column_name, "0"), f"{roaster_id}.{column_name}")
        ]
        roaster_eligible_skus[roaster_id] = [
            sku
            for sku, column_name in SKU_FLAG_COLUMNS
            if _parse_bool_flag(row.get(column_name, "0"), f"{roaster_id}.{column_name}")
        ]
        roaster_initial_sku[roaster_id] = (row.get("initial_last_sku") or "").strip()

        row_process_time = _parse_int(row.get("process_time_min"), f"{roaster_id}.process_time_min")
        row_consume_time = _parse_int(row.get("consume_time_min"), f"{roaster_id}.consume_time_min")
        if row_process_time != 15 or row_consume_time != 3:
            logger.warning(
                "Roaster %s has non-standard timing: process=%s, consume=%s",
                roaster_id,
                row_process_time,
                row_consume_time,
            )
        if process_time is None:
            process_time = row_process_time
        if consume_time is None:
            consume_time = row_consume_time

    if process_time is None or consume_time is None:
        raise ValueError("No active roasters found in roasters.csv")

    skus = [row["sku"].strip() for row in sku_rows]
    sku_revenue = {
        row["sku"].strip(): _parse_float(
            row.get("revenue_per_batch_usd"),
            f"{row['sku']}.revenue_per_batch_usd",
        )
        for row in sku_rows
    }
    sku_credits_rc = {
        row["sku"].strip(): _parse_bool_flag(
            row.get("credits_rc_stock", "0"),
            f"{row['sku']}.credits_rc_stock",
        )
        for row in sku_rows
    }
    sku_eligible_roasters = {
        sku: [
            roaster_id
            for roaster_id in roasters
            if sku in roaster_eligible_skus[roaster_id]
        ]
        for sku in skus
    }

    jobs = [row["job_id"].strip() for row in job_rows]
    job_sku = {row["job_id"].strip(): row["sku"].strip() for row in job_rows}
    job_batches = {
        row["job_id"].strip(): _parse_int(row.get("required_batches"), f"{row['job_id']}.required_batches")
        for row in job_rows
    }
    job_due = {
        row["job_id"].strip(): _parse_int(row.get("due_time_min"), f"{row['job_id']}.due_time_min")
        for row in job_rows
    }
    job_release = {
        row["job_id"].strip(): _parse_int(
            row.get("release_time_min"),
            f"{row['job_id']}.release_time_min",
        )
        for row in job_rows
    }

    shift_length = _parse_int(shift_params["shift_length_min"], "shift_length_min")
    setup_time = _parse_int(shift_params["setup_time_diff_sku_min"], "setup_time_diff_sku_min")
    max_rc = _parse_int(shift_params["max_rc_batches_per_line"], "max_rc_batches_per_line")
    safety_stock = _parse_int(shift_params["safety_stock_batches"], "safety_stock_batches")
    psc_pool_per_roaster = _parse_int(
        shift_params["psc_pool_size_per_roaster"],
        "psc_pool_size_per_roaster",
    )
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
    for line_id, rate in consume_rate.items():
        if rate <= 0:
            message = f"Consumption rate must be positive for {line_id}: {rate}"
            logger.error(message)
            raise ValueError(message)

    cost_stockout = _parse_float(
        shift_params["stockout_cost_per_event_per_line"],
        "stockout_cost_per_event_per_line",
    )
    cost_tardiness = _parse_float(
        shift_params["tardiness_cost_per_min"],
        "tardiness_cost_per_min",
    )
    cost_idle = _parse_float(
        shift_params["idle_cost_per_min_per_roaster"],
        "idle_cost_per_min_per_roaster",
    )
    cost_overflow = _parse_float(
        shift_params["overflow_idle_cost_per_min_per_roaster"],
        "overflow_idle_cost_per_min_per_roaster",
    )

    solver_name = solver_config["solver_name"].strip()
    time_limit = _parse_int(solver_config["time_limit_sec"], "time_limit_sec")
    mip_gap = _parse_float(solver_config["mip_gap_target"], "mip_gap_target")
    allow_r3_flex = _parse_bool_flag(
        solver_config["allow_r3_flexible_output"],
        "allow_r3_flexible_output",
    )
    enable_disruptions = _parse_bool_flag(
        solver_config["enable_disruptions"],
        "enable_disruptions",
    )

    if overrides:
        if "solver_name" in overrides:
            logger.debug(
                "Override applied: solver_name: %s -> %s",
                solver_name,
                overrides["solver_name"],
            )
            solver_name = str(overrides["solver_name"])
        if "time_limit" in overrides:
            logger.debug(
                "Override applied: time_limit: %s -> %s",
                time_limit,
                overrides["time_limit"],
            )
            time_limit = _parse_int(overrides["time_limit"], "override.time_limit")
        if "mip_gap" in overrides:
            logger.debug(
                "Override applied: mip_gap: %s -> %s",
                mip_gap,
                overrides["mip_gap"],
            )
            mip_gap = _parse_float(overrides["mip_gap"], "override.mip_gap")
        if "allow_r3_flex" in overrides:
            logger.debug(
                "Override applied: allow_r3_flex: %s -> %s",
                allow_r3_flex,
                overrides["allow_r3_flex"],
            )
            allow_r3_flex = _parse_bool_flag(
                overrides["allow_r3_flex"],
                "override.allow_r3_flex",
            )
        if "enable_disruptions" in overrides:
            logger.debug(
                "Override applied: enable_disruptions: %s -> %s",
                enable_disruptions,
                overrides["enable_disruptions"],
            )
            enable_disruptions = _parse_bool_flag(
                overrides["enable_disruptions"],
                "override.enable_disruptions",
            )

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

    consumption_events: dict[str, list[int]] = {}
    for line_id in lines:
        rate = consume_rate[line_id]
        event_count = math.floor(shift_length / rate)
        events = [math.floor(index * rate) for index in range(1, event_count + 1)]
        consumption_events[line_id] = events
        logger.debug("Computed %d consumption events for %s", len(events), line_id)

    downtime_slots = {roaster_id: set() for roaster_id in roasters}
    for row in downtime_rows:
        roaster_id = (row.get("roaster_id") or "").strip()
        if roaster_id not in downtime_slots:
            continue
        start_min = _parse_int(row.get("start_min"), f"{roaster_id}.start_min")
        end_min = _parse_int(row.get("end_min"), f"{roaster_id}.end_min")
        downtime_slots[roaster_id].update(range(start_min, end_min + 1))

    for roaster_id in roasters:
        logger.debug(
            "Expanded %d downtime slots for %s",
            len(downtime_slots[roaster_id]),
            roaster_id,
        )

    filtered_disruption_rows = [row for row in disruption_rows if not _is_blank_row(row)]
    disruption_events = []
    if enable_disruptions and filtered_disruption_rows:
        for row in filtered_disruption_rows:
            disruption_events.append(
                {
                    "time_min": _parse_int(row.get("time_min"), "disruption.time_min"),
                    "roaster_id": (row.get("roaster_id") or "").strip(),
                    "event_type": (row.get("event_type") or "").strip(),
                    "duration_min": _parse_int(
                        row.get("duration_min"),
                        "disruption.duration_min",
                    ),
                }
            )
    elif filtered_disruption_rows and not enable_disruptions:
        logger.warning(
            "Disruption file has %d data rows but enable_disruptions=False — ignoring. "
            "Set --disruptions 1 to enable.",
            len(filtered_disruption_rows),
        )

    # Tuple batch identifiers keep MTO and PSC batches in one namespace without
    # adding string parsing rules that downstream model code would depend on.
    batch_sku = {batch_id: job_sku[batch_id[0]] for batch_id in mto_batches}
    batch_sku.update({batch_id: "PSC" for batch_id in psc_pool})

    batch_is_mto = {batch_id: True for batch_id in mto_batches}
    batch_is_mto.update({batch_id: False for batch_id in psc_pool})

    batch_eligible_roasters = {
        batch_id: list(sku_eligible_roasters[batch_sku[batch_id]])
        for batch_id in mto_batches
    }
    batch_eligible_roasters.update(
        {batch_id: list(roasters) for batch_id in psc_pool}
    )

    pipeline_batches = {
        pipeline_line: [
            batch_id
            for batch_id in all_batches
            if any(
                roaster_pipeline[roaster_id] == pipeline_line
                for roaster_id in batch_eligible_roasters[batch_id]
            )
        ]
        for pipeline_line in sorted(set(roaster_pipeline.values()))
    }

    data = {
        "shift_length": shift_length,
        "process_time": process_time,
        "consume_time": consume_time,
        "setup_time": setup_time,
        "max_rc": max_rc,
        "safety_stock": safety_stock,
        "psc_pool_per_roaster": psc_pool_per_roaster,
        "rc_init": rc_init,
        "consume_rate": consume_rate,
        "cost_stockout": cost_stockout,
        "cost_tardiness": cost_tardiness,
        "cost_idle": cost_idle,
        "cost_overflow": cost_overflow,
        "solver_name": solver_name,
        "time_limit": time_limit,
        "mip_gap": mip_gap,
        "allow_r3_flex": allow_r3_flex,
        "enable_disruptions": enable_disruptions,
        "max_start": shift_length - process_time,
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
        "sku_eligible_roasters": sku_eligible_roasters,
        "jobs": jobs,
        "job_sku": job_sku,
        "job_batches": job_batches,
        "job_due": job_due,
        "job_release": job_release,
        "mto_batches": mto_batches,
        "psc_pool": psc_pool,
        "all_batches": all_batches,
        "consumption_events": consumption_events,
        "downtime_slots": downtime_slots,
        "disruption_events": disruption_events,
        "batch_sku": batch_sku,
        "batch_eligible_roasters": batch_eligible_roasters,
        "batch_is_mto": batch_is_mto,
        "pipeline_batches": pipeline_batches,
    }

    logger.info("Data loaded from %s", input_dir)
    logger.info(
        "Final counts: roasters=%d, jobs=%d, MTO batches=%d, PSC pool=%d, all batches=%d",
        len(roasters),
        len(jobs),
        len(mto_batches),
        len(psc_pool),
        len(all_batches),
    )

    return data


def _print_validation_report(input_dir: str, data: dict, overrides: dict | None) -> None:
    print("=== DATA VALIDATION REPORT ===")
    print(f"Input directory       : {input_dir}")
    print(f"Roasters loaded       : {len(data['roasters'])} active (expected 5)")
    print(
        f"  R1: {data['roaster_line']['R1']}/{data['roaster_pipeline']['R1']}-pipeline, "
        f"{'+'.join(data['roaster_eligible_skus']['R1'])}"
    )
    print(
        f"  R2: {data['roaster_line']['R2']}/{data['roaster_pipeline']['R2']}-pipeline, "
        f"{'+'.join(data['roaster_eligible_skus']['R2'])}"
    )
    print(
        f"  R3: {data['roaster_line']['R3']}/{data['roaster_pipeline']['R3']}-pipeline, "
        f"PSC — flex output: {data['allow_r3_flex']}"
    )
    print(
        f"  R4: {data['roaster_line']['R4']}/{data['roaster_pipeline']['R4']}-pipeline, "
        f"{'+'.join(data['roaster_eligible_skus']['R4'])}"
    )
    print(
        f"  R5: {data['roaster_line']['R5']}/{data['roaster_pipeline']['R5']}-pipeline, "
        f"{'+'.join(data['roaster_eligible_skus']['R5'])}"
    )
    print(f"SKUs loaded           : {data['skus']}")
    print(
        "Revenue check         : "
        f"PSC=${_format_number(data['sku_revenue']['PSC'])}, "
        f"NDG=${_format_number(data['sku_revenue']['NDG'])}, "
        f"BUSTA=${_format_number(data['sku_revenue']['BUSTA'])}"
    )
    total_mto_batches = sum(data["job_batches"].values())
    print(f"MTO jobs              : {len(data['jobs'])} jobs → {total_mto_batches} total batches")
    for job_id in data["jobs"]:
        print(
            f"  {job_id}: {data['job_sku'][job_id]} × {data['job_batches'][job_id]}, "
            f"due={data['job_due'][job_id]}"
        )
    print(f"PSC pool              : {len(data['psc_pool'])} entries (expected 160)")
    print(f"All batches           : {len(data['all_batches'])} (expected 164)")
    print(
        "Consumption events    : "
        f"L1={len(data['consumption_events']['L1'])} (expected ~94), "
        f"L2={len(data['consumption_events']['L2'])} (expected ~100)"
    )
    print(f"  L1 first 5 events: {data['consumption_events']['L1'][:5]}")
    print(f"  L2 first 5 events: {data['consumption_events']['L2'][:5]}")
    r3_slots = sorted(data["downtime_slots"]["R3"])
    print(
        "Downtime R3           : "
        f"{len(r3_slots)} slots (expected 30) — first={r3_slots[0]}, last={r3_slots[-1]}"
    )
    print(
        f"Disruptions           : {len(data['disruption_events'])} events loaded (expected 0 in baseline)"
    )
    if overrides:
        applied = sorted(overrides.keys())
        print(f"Overrides applied     : {applied}")
    else:
        print("Overrides applied     : none")
    print(f"R3 flexible output    : {data['allow_r3_flex']}")
    print(f"Max batch start       : {data['max_start']} (expected 465)")
    print(
        "Solver config         : "
        f"{data['solver_name']}, time_limit={data['time_limit']}s, gap={data['mip_gap'] * 100:g}%"
    )
    print("=== END VALIDATION ===")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    default_input_dir = "Input_data"
    validation_overrides = None
    validation_data = load(default_input_dir, validation_overrides)
    _print_validation_report(default_input_dir, validation_data, validation_overrides)
