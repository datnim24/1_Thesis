"""Interactive Plotly dashboard for thesis result JSON files and existing HTML reports."""

from __future__ import annotations

import argparse
import json
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.result_schema import (
    convert_legacy_result,
    create_result,
    load_reference_data,
    normalize_schedule_entry,
    reconstruct_gc_trajectory,
    reconstruct_rc_trajectory,
    reconstruct_restock_timeline,
)

SKU_COLORS = {
    "PSC": "#2196F3",
    "NDG": "#FF9800",
    "BUSTA": "#4CAF50",
}
SETUP_COLOR = "#FFC107"
DOWN_COLOR = "#F44336"
IDLE_COLOR = "#E0E0E0"
PLANNED_DOWN_COLOR = "#9E9E9E"
L1_COLOR = "#1565C0"
L2_COLOR = "#C62828"
SAFETY_COLOR = "#FF5722"
RESTOCK_COLOR = "#7B1FA2"
GC_PAIR_COLORS = {
    "L1_PSC": "#2196F3",
    "L1_NDG": "#FF9800",
    "L1_BUSTA": "#4CAF50",
    "L2_PSC": "#1565C0",
}

ROOT_DIR = Path(__file__).resolve().parents[1]
_RESULT_SEARCH_ROOTS = (
    ROOT_DIR / "Results",
)
_SIDECAR_SUFFIXES = ("_schedule.json", "_cancelled.json", "_ups.json", "_restocks.json", "_trace.json")
_IGNORE_FILES = {"meta.json", "settings.json"}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sidecar_data(result_path: Path):
    stem = result_path.stem
    prefix = stem[:-7] if stem.endswith("_result") else stem
    schedule_path = result_path.with_name(f"{prefix}_schedule.json")
    cancelled_path = result_path.with_name(f"{prefix}_cancelled.json")
    ups_path = result_path.with_name(f"{prefix}_ups.json")
    restocks_path = result_path.with_name(f"{prefix}_restocks.json")
    schedule = _load_json(schedule_path) if schedule_path.exists() else None
    cancelled = _load_json(cancelled_path) if cancelled_path.exists() else None
    ups = _load_json(ups_path) if ups_path.exists() else None
    restocks = _load_json(restocks_path) if restocks_path.exists() else None
    return schedule, cancelled, ups, restocks


def _load_any_result(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    raw = _load_json(result_path)
    if _is_reactive_result(raw):
        return _convert_reactive_to_universal(raw)
    if {"metadata", "experiment", "kpi", "schedule", "parameters"}.issubset(raw):
        return raw
    schedule, cancelled, ups, restocks = _sidecar_data(result_path)
    result = convert_legacy_result(raw, source="auto", schedule=schedule, cancelled=cancelled, ups=ups, restocks=restocks)
    return result


def _is_reactive_result(raw: dict) -> bool:
    """Check if a dict is a Reactive CP-SAT oracle result."""
    return isinstance(raw, dict) and "oracle_results" in raw and "baseline_kpi" in raw


def _looks_like_result_file(path: Path) -> bool:
    name = path.name.lower()
    if name in _IGNORE_FILES or name.endswith(_SIDECAR_SUFFIXES):
        return False
    if name.endswith("_result.json"):
        return True
    try:
        raw = _load_json(path)
    except Exception:
        return False
    if not isinstance(raw, dict):
        return False
    if _is_reactive_result(raw):
        return True
    if {"metadata", "experiment", "kpi", "schedule", "parameters"}.issubset(raw):
        return True
    if "schedule" in raw and "net_profit" in raw:
        return True
    if {"net_profit", "total_revenue", "total_costs"} & set(raw):
        if any(key in raw for key in ("solver_name", "solver_engine", "inventory_option", "obj_value", "schedule")):
            return True
    return False


def _looks_like_dashboard_file(path: Path) -> bool:
    if path.suffix.lower() != ".html":
        return False
    if path.parent.name.lower() == "plots":
        return False
    if (path.parent / "meta.json").exists():
        return True
    return path.parent.name == path.stem


def _find_latest_input_file() -> Path | None:
    candidates: list[Path] = []
    for root in _RESULT_SEARCH_ROOTS:
        if not root.exists():
            continue
        candidates.extend(path for path in root.rglob("*.json") if path.is_file())
        candidates.extend(path for path in root.rglob("*.html") if path.is_file())
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        if _looks_like_result_file(path) or _looks_like_dashboard_file(path):
            return path
    return None


def _open_in_browser(path: Path) -> None:
    try:
        webbrowser.open(path.resolve().as_uri())
    except Exception:
        pass


def _title_label(result: dict[str, Any]) -> str:
    meta = result["metadata"]
    return f"{meta.get('solver_engine', 'unknown')} ({meta.get('solver_name', '')})".strip()


def _infer_setup_intervals(schedule: list[dict[str, Any]], refs: dict[str, Any], sigma: int) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    by_roaster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in schedule:
        if entry["status"] == "completed":
            by_roaster[entry["roaster"]].append(entry)
    for roaster, batches in by_roaster.items():
        batches.sort(key=lambda item: (item["start"], item["end"]))
        prev_sku = refs["roaster_initial_sku"].get(roaster, "PSC")
        for entry in batches:
            if entry["sku"] != prev_sku and entry["start"] >= sigma:
                intervals.append(
                    {
                        "roaster": roaster,
                        "start": entry["start"] - sigma,
                        "end": entry["start"],
                        "sku": entry["sku"],
                        "batch_id": entry["batch_id"],
                    }
                )
            prev_sku = entry["sku"]
    return intervals


def _interval_slots(intervals: list[tuple[int, int]], shift_length: int) -> set[int]:
    slots: set[int] = set()
    for start, end in intervals:
        slots.update(range(max(0, start), min(shift_length, end)))
    return slots


def _planned_downtime_intervals(refs: dict[str, Any], shift_length: int) -> dict[str, list[tuple[int, int]]]:
    output: dict[str, list[tuple[int, int]]] = {}
    for roaster, slots in refs["downtime_slots"].items():
        if not slots:
            output[roaster] = []
            continue
        ordered = sorted(slot for slot in slots if 0 <= slot < shift_length)
        intervals: list[tuple[int, int]] = []
        start = prev = ordered[0]
        for slot in ordered[1:]:
            if slot == prev + 1:
                prev = slot
            else:
                intervals.append((start, prev + 1))
                start = prev = slot
        intervals.append((start, prev + 1))
        output[roaster] = intervals
    return output


def _compute_utilization(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    refs = load_reference_data(result["metadata"].get("input_dir"))
    params = result["parameters"]
    schedule = [normalize_schedule_entry(entry) for entry in result["schedule"]]
    shift_length = int(params["SL"])
    sigma = int(params["sigma"])
    setup_intervals = _infer_setup_intervals(schedule, refs, sigma)
    planned = _planned_downtime_intervals(refs, shift_length)

    running_slots = {roaster: set() for roaster in refs["roasters"]}
    setup_slots = {roaster: set() for roaster in refs["roasters"]}
    down_slots = {roaster: set() for roaster in refs["roasters"]}
    for entry in schedule:
        if entry["status"] == "completed":
            running_slots[entry["roaster"]].update(range(entry["start"], entry["end"]))
    for setup in setup_intervals:
        setup_slots[setup["roaster"]].update(range(setup["start"], setup["end"]))
    for roaster, intervals in planned.items():
        down_slots[roaster].update(_interval_slots(intervals, shift_length))
    for event in result.get("ups_events", []):
        roaster = event["roaster_id"]
        down_slots.setdefault(roaster, set()).update(range(event["t"], min(shift_length, event["t"] + event["duration"])))

    utilization: dict[str, dict[str, float]] = {}
    for roaster in refs["roasters"]:
        running = running_slots[roaster]
        setup = setup_slots[roaster] - running
        down = down_slots[roaster] - running - setup
        idle = max(0, shift_length - len(running | setup | down))
        utilization[roaster] = {
            "Running": float(len(running)),
            "Setup": float(len(setup)),
            "Down": float(len(down)),
            "Idle": float(idle),
        }
    return utilization


def _pipeline_occupancy(result: dict[str, Any]) -> dict[str, list[int]]:
    params = result["parameters"]
    shift_length = int(params["SL"])
    occupancy = {"L1": [0] * shift_length, "L2": [0] * shift_length}
    for raw_entry in result["schedule"]:
        entry = normalize_schedule_entry(raw_entry)
        if entry["status"] != "completed":
            continue
        pipeline = entry["pipeline"]
        if pipeline not in occupancy:
            continue
        for slot in range(max(0, entry["pipeline_start"]), min(shift_length, entry["pipeline_end"])):
            occupancy[pipeline][slot] = 1
    return occupancy


def _build_kpi_header(result: dict[str, Any], compare: dict[str, Any] | None = None) -> go.Figure:
    meta = result["metadata"]
    kpi = result["kpi"]
    left = (
        f"Solver: {meta.get('solver_engine')} ({meta.get('solver_name')})<br>"
        f"Status: {meta.get('status')}<br>"
        f"Solve time: {meta.get('solve_time_sec', 0):,.2f}s<br>"
        f"Net Profit: ${kpi.get('net_profit', 0):,.0f}"
    )
    center = (
        f"PSC: {kpi.get('psc_count', 0)} batches (${kpi.get('revenue_psc', 0):,.0f})<br>"
        f"NDG: {kpi.get('ndg_count', 0)} batches (${kpi.get('revenue_ndg', 0):,.0f})<br>"
        f"BUSTA: {kpi.get('busta_count', 0)} batches (${kpi.get('revenue_busta', 0):,.0f})<br>"
        f"Total: {kpi.get('total_batches', 0)} batches"
    )
    right = (
        f"Tardiness: J1={kpi.get('tardiness_min', {}).get('J1', 0)}min, "
        f"J2={kpi.get('tardiness_min', {}).get('J2', 0)}min (${kpi.get('tard_cost', 0):,.0f})<br>"
        f"Stockout: L1={kpi.get('stockout_events', {}).get('L1', 0)}, "
        f"L2={kpi.get('stockout_events', {}).get('L2', 0)} (${kpi.get('stockout_cost', 0):,.0f})<br>"
        f"Idle: {kpi.get('idle_min', 0)}min (${kpi.get('idle_cost', 0):,.0f})<br>"
        f"Over-idle: {kpi.get('over_min', 0)}min (${kpi.get('over_cost', 0):,.0f})<br>"
        f"Restocks: {kpi.get('restock_count', 0)}"
    )
    headers = ["Overview", "Production", "Penalties"]
    values = [left, center, right]
    if compare is not None:
        ckpi = compare["kpi"]
        headers.append("Comparison")
        delta = kpi.get("net_profit", 0) - ckpi.get("net_profit", 0)
        values.append(
            f"A: {_title_label(result)}<br>"
            f"B: {_title_label(compare)}<br>"
            f"Net delta: ${delta:,.0f}<br>"
            f"Batch delta: {kpi.get('total_batches', 0) - ckpi.get('total_batches', 0)}"
        )
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers, fill_color="#F5F5F5", align="left", font=dict(color="#212121", size=13)),
                cells=dict(values=values, fill_color="white", align="left", height=52, font=dict(color="#212121", size=12)),
            )
        ]
    )
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=170)
    return fig


def _add_interval_bar(
    fig: go.Figure,
    row_label: str,
    start: int,
    end: int,
    color: str,
    name: str,
    hover: str,
    opacity: float = 1.0,
    line_width: float = 1.0,
    pattern: str | None = None,
    showlegend: bool = False,
):
    marker = dict(color=color, opacity=opacity, line=dict(color="#424242", width=line_width))
    if pattern:
        marker["pattern"] = dict(shape=pattern)
    fig.add_trace(
        go.Bar(
            x=[end - start],
            y=[row_label],
            base=[start],
            orientation="h",
            name=name,
            hovertemplate=hover + "<extra></extra>",
            marker=marker,
            showlegend=showlegend,
        )
    )


def _build_gantt(result: dict[str, Any], title_suffix: str = "") -> go.Figure:
    refs = load_reference_data(result["metadata"].get("input_dir"))
    params = result["parameters"]
    sigma = int(params["sigma"])
    schedule = [normalize_schedule_entry(entry) for entry in result["schedule"]]
    setup_intervals = _infer_setup_intervals(schedule, refs, sigma)
    planned = _planned_downtime_intervals(refs, int(params["SL"]))

    fig = go.Figure()
    row_order = ["Pipeline L2", "Pipeline L1", "R5", "R4", "R3", "R2", "R1"]

    legend_flags = {
        "PSC": False,
        "NDG": False,
        "BUSTA": False,
        "Setup": False,
        "Planned Down": False,
        "UPS Down": False,
        "Pipe L1": False,
        "Pipe L2": False,
    }

    for roaster, intervals in planned.items():
        for start, end in intervals:
            _add_interval_bar(
                fig,
                roaster,
                start,
                end,
                PLANNED_DOWN_COLOR,
                "Planned Down",
                f"Planned downtime<br>{roaster}: {start}–{end}",
                opacity=0.35,
                pattern="/",
                showlegend=not legend_flags["Planned Down"],
            )
            legend_flags["Planned Down"] = True

    for event in result.get("ups_events", []):
        _add_interval_bar(
            fig,
            event["roaster_id"],
            int(event["t"]),
            int(event["t"]) + int(event["duration"]),
            DOWN_COLOR,
            "UPS Down",
            f"UPS downtime<br>{event['roaster_id']}: {event['t']}–{event['t'] + event['duration']}",
            opacity=0.45,
            showlegend=not legend_flags["UPS Down"],
        )
        legend_flags["UPS Down"] = True

    for setup in setup_intervals:
        _add_interval_bar(
            fig,
            setup["roaster"],
            setup["start"],
            setup["end"],
            SETUP_COLOR,
            "Setup",
            f"Setup<br>{setup['roaster']}: {setup['start']}–{setup['end']}<br>Target SKU: {setup['sku']}<br>Batch: {setup['batch_id']}",
            showlegend=not legend_flags["Setup"],
        )
        legend_flags["Setup"] = True

    for entry in sorted(schedule, key=lambda item: (item["roaster"], item["start"], item["end"])):
        if entry["status"] != "completed":
            continue
        color = SKU_COLORS.get(entry["sku"], "#607D8B")
        hover = (
            f"Batch: {entry['batch_id']}<br>"
            f"SKU: {entry['sku']}<br>"
            f"Time: {entry['start']}–{entry['end']} ({entry['end'] - entry['start']} min)<br>"
            f"Pipeline: {entry['pipeline']} [{entry['pipeline_start']}..{entry['pipeline_end'] - 1}]<br>"
            f"Output: {entry['output_line']}<br>"
            f"Setup: {entry['setup']}"
        )
        _add_interval_bar(
            fig,
            entry["roaster"],
            entry["start"],
            entry["end"],
            color,
            entry["sku"],
            hover,
            line_width=2.5 if entry["is_mto"] else 1.0,
            showlegend=not legend_flags[entry["sku"]],
        )
        legend_flags[entry["sku"]] = True

        pipe_name = "Pipe L1" if entry["pipeline"] == "L1" else "Pipe L2"
        pipe_color = L1_COLOR if entry["pipeline"] == "L1" else L2_COLOR
        _add_interval_bar(
            fig,
            f"Pipeline {entry['pipeline']}",
            entry["pipeline_start"],
            entry["pipeline_end"],
            pipe_color,
            pipe_name,
            f"Pipeline {entry['pipeline']}<br>{entry['batch_id']}: {entry['pipeline_start']}–{entry['pipeline_end']}",
            opacity=0.55,
            showlegend=not legend_flags[pipe_name],
        )
        legend_flags[pipe_name] = True

    restock_shown = False
    for rst in reconstruct_restock_timeline(result.get("restocks")):
        _add_interval_bar(
            fig,
            f"Pipeline {rst['line_id']}",
            rst["start"],
            rst["end"],
            RESTOCK_COLOR,
            "Restock",
            f"Restock<br>{rst['line_id']} {rst['sku']}<br>"
            f"{rst['start']}–{rst['end']} (+{rst['qty']} GC at completion)",
            opacity=0.75,
            pattern="x",
            showlegend=not restock_shown,
        )
        restock_shown = True

    fig.update_layout(
        template="plotly_white",
        title=f"Roasting Schedule{title_suffix}",
        barmode="overlay",
        height=480,
        margin=dict(l=70, r=30, t=60, b=40),
        xaxis=dict(title="Time Slot", range=[0, params["SL"]], showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="", categoryorder="array", categoryarray=row_order),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def _build_rc_plot(result: dict[str, Any], compare: dict[str, Any] | None = None) -> go.Figure:
    params = result["parameters"]
    traj = reconstruct_rc_trajectory(result["schedule"], params)
    slots = list(range(params["SL"]))
    fig = go.Figure()
    min_stock = min(min(traj["L1"]), min(traj["L2"]))

    fig.add_trace(go.Scatter(x=slots, y=traj["L1"], mode="lines", name=f"L1 {_title_label(result)}", line=dict(color=L1_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=slots, y=traj["L2"], mode="lines", name=f"L2 {_title_label(result)}", line=dict(color=L2_COLOR, width=2)))
    if compare is not None:
        ctraj = reconstruct_rc_trajectory(compare["schedule"], compare["parameters"])
        fig.add_trace(
            go.Scatter(x=slots, y=ctraj["L1"], mode="lines", name=f"L1 {_title_label(compare)}", line=dict(color=L1_COLOR, width=2, dash="dash"))
        )
        fig.add_trace(
            go.Scatter(x=slots, y=ctraj["L2"], mode="lines", name=f"L2 {_title_label(compare)}", line=dict(color=L2_COLOR, width=2, dash="dash"))
        )

    fig.add_hline(y=params["safety_stock"], line_dash="dash", line_color=SAFETY_COLOR, annotation_text="Safety Threshold")
    fig.add_hline(y=params["max_rc"], line_dash="dash", line_color="#616161", annotation_text="Buffer Capacity")
    fig.add_hrect(y0=min_stock - 2, y1=params["safety_stock"], fillcolor="#FFEBEE", opacity=0.3, line_width=0)

    completion_y = params["max_rc"] + 1
    for line_id, color in (("L1", L1_COLOR), ("L2", L2_COLOR)):
        fig.add_trace(
            go.Scatter(
                x=traj[f"{line_id}_events"]["completions"],
                y=[completion_y] * len(traj[f"{line_id}_events"]["completions"]),
                mode="markers",
                name=f"{line_id} completions",
                marker=dict(symbol="circle", size=7, color=color),
                hovertemplate=f"{line_id} PSC completion at t=%{{x}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=traj[f"{line_id}_events"]["consumptions"],
                y=[min_stock - 1] * len(traj[f"{line_id}_events"]["consumptions"]),
                mode="markers",
                name=f"{line_id} draws",
                marker=dict(symbol="triangle-down", size=7, color=color),
                hovertemplate=f"{line_id} consumption at t=%{{x}}<extra></extra>",
            )
        )

    for event in result.get("ups_events", []):
        fig.add_vline(x=event["t"], line_color=DOWN_COLOR, line_dash="dot", opacity=0.7)

    fig.update_layout(
        template="plotly_white",
        title="RC Stock Trajectory",
        height=340,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(title="Time Slot", range=[0, params["SL"]]),
        yaxis=dict(title="RC Stock (batches)", range=[min_stock - 2, params["max_rc"] + 2]),
        hovermode="x unified",
    )
    return fig


def _build_waterfall(result: dict[str, Any], compare: dict[str, Any] | None = None) -> go.Figure:
    def _measure_data(item: dict[str, Any]) -> tuple[list[str], list[str], list[float]]:
        kpi = item["kpi"]
        x = ["Revenue", "Tardiness", "Stockout", "Idle", "Overflow", "Net Profit"]
        measure = ["absolute", "relative", "relative", "relative", "relative", "total"]
        y = [
            float(kpi["total_revenue"]),
            -float(kpi["tard_cost"]),
            -float(kpi["stockout_cost"]),
            -float(kpi["idle_cost"]),
            -float(kpi["over_cost"]),
            float(kpi["net_profit"]),
        ]
        return x, measure, y

    fig = go.Figure()
    x, measure, y = _measure_data(result)
    fig.add_trace(
        go.Waterfall(
            name=_title_label(result),
            x=x,
            measure=measure,
            y=y,
            connector={"line": {"color": "#9E9E9E"}},
            increasing={"marker": {"color": "#4CAF50"}},
            decreasing={"marker": {"color": "#F44336"}},
            totals={"marker": {"color": "#2196F3"}},
        )
    )
    if compare is not None:
        cx, cmeasure, cy = _measure_data(compare)
        fig.add_trace(
            go.Waterfall(
                name=_title_label(compare),
                x=cx,
                measure=cmeasure,
                y=cy,
                connector={"line": {"color": "#BDBDBD"}},
                increasing={"marker": {"color": "#81C784"}},
                decreasing={"marker": {"color": "#EF9A9A"}},
                totals={"marker": {"color": "#64B5F6"}},
            )
        )
    fig.update_layout(template="plotly_white", title="Profit Decomposition", height=290, waterfallgroupgap=0.35)
    return fig


def _build_r3_routing(result: dict[str, Any], compare: dict[str, Any] | None = None) -> go.Figure | None:
    if not result["metadata"].get("allow_r3_flex", True):
        return None
    schedule = [normalize_schedule_entry(entry) for entry in result["schedule"]]
    r3_batches = [entry for entry in schedule if entry["roaster"] == "R3" and entry["sku"] == "PSC" and entry["status"] == "completed"]
    if not r3_batches and compare is None:
        return None
    fig = go.Figure()
    if r3_batches:
        fig.add_trace(
            go.Scatter(
                x=[entry["start"] for entry in r3_batches],
                y=[entry["output_line"] for entry in r3_batches],
                mode="markers",
                marker=dict(size=10, color=[L1_COLOR if entry["output_line"] == "L1" else L2_COLOR for entry in r3_batches]),
                name=_title_label(result),
                text=[entry["batch_id"] for entry in r3_batches],
                hovertemplate="Batch %{text}<br>Start %{x}<br>Route %{y}<extra></extra>",
            )
        )
    if compare is not None:
        cschedule = [normalize_schedule_entry(entry) for entry in compare["schedule"]]
        cr3 = [entry for entry in cschedule if entry["roaster"] == "R3" and entry["sku"] == "PSC" and entry["status"] == "completed"]
        if cr3:
            fig.add_trace(
                go.Scatter(
                    x=[entry["start"] for entry in cr3],
                    y=[entry["output_line"] for entry in cr3],
                    mode="markers",
                    marker=dict(size=9, symbol="diamond", color=[L1_COLOR if entry["output_line"] == "L1" else L2_COLOR for entry in cr3]),
                    name=_title_label(compare),
                    text=[entry["batch_id"] for entry in cr3],
                    hovertemplate="Batch %{text}<br>Start %{x}<br>Route %{y}<extra></extra>",
                )
            )
    counts = Counter(entry["output_line"] for entry in r3_batches)
    title = f"R3 Routing Analysis — R3: {counts.get('L1', 0)} batches → L1, {counts.get('L2', 0)} batches → L2"
    fig.update_layout(template="plotly_white", title=title, height=230, xaxis_title="Batch Start Time", yaxis_title="", yaxis=dict(categoryorder="array", categoryarray=["L2", "L1"]))
    return fig


def _build_utilization(result: dict[str, Any], compare: dict[str, Any] | None = None) -> go.Figure:
    util = _compute_utilization(result)
    roasters = list(util.keys())
    fig = go.Figure()
    colors = {"Running": SKU_COLORS["PSC"], "Setup": SETUP_COLOR, "Down": DOWN_COLOR, "Idle": IDLE_COLOR}
    for label in ("Running", "Setup", "Down", "Idle"):
        fig.add_trace(
            go.Bar(
                y=roasters,
                x=[util[roaster][label] for roaster in roasters],
                name=f"{label} {_title_label(result)}" if compare else label,
                orientation="h",
                marker_color=colors[label],
            )
        )
    if compare is not None:
        cutil = _compute_utilization(compare)
        for label in ("Running", "Setup", "Down", "Idle"):
            fig.add_trace(
                go.Bar(
                    y=roasters,
                    x=[cutil[roaster][label] for roaster in roasters],
                    name=f"{label} {_title_label(compare)}",
                    orientation="h",
                    marker_color=colors[label],
                    opacity=0.45,
                )
            )
    fig.update_layout(template="plotly_white", title="Per-Roaster Utilization", barmode="stack", height=260, xaxis_title="Minutes", yaxis=dict(categoryorder="array", categoryarray=list(reversed(roasters))))
    return fig


def _build_pipeline_timeline(result: dict[str, Any], title_suffix: str = "") -> go.Figure:
    occ = _pipeline_occupancy(result)
    utilization = {line: sum(occ[line]) / len(occ[line]) * 100.0 for line in ("L1", "L2")}
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=[occ["L2"], occ["L1"]],
                x=list(range(len(occ["L1"]))),
                y=["Pipeline L2", "Pipeline L1"],
                colorscale=[[0.0, "#F5F5F5"], [1.0, "#455A64"]],
                showscale=False,
                hovertemplate="%{y}<br>t=%{x}<br>%{z:.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Pipeline Utilization Timeline{title_suffix} — L1 {utilization['L1']:.1f}%, L2 {utilization['L2']:.1f}%",
        height=180,
        margin=dict(l=60, r=20, t=50, b=35),
        xaxis_title="Time Slot",
    )
    return fig


def _build_gc_plot(result: dict[str, Any], line_id: str, title_suffix: str = "") -> go.Figure | None:
    """Build a GC silo inventory plot for a given line."""
    params = result["parameters"]
    gc_cap = params.get("gc_capacity", {})
    gc_init_vals = params.get("gc_init", {})
    if not gc_cap and not gc_init_vals:
        return None

    restocks = result.get("restocks", [])
    gc_traj = reconstruct_gc_trajectory(
        result["schedule"],
        restocks,
        params,
        result.get("cancelled_batches"),
    )
    restock_timeline = reconstruct_restock_timeline(restocks)

    pairs_for_line = [k for k in gc_traj if k != "events" and k.startswith(f"{line_id}_")]
    if not pairs_for_line:
        return None

    shift_length = int(params["SL"])
    slots = list(range(shift_length))
    fig = go.Figure()

    for pair_key in sorted(pairs_for_line):
        sku = pair_key.split("_", 1)[1]
        color = GC_PAIR_COLORS.get(pair_key, "#607D8B")
        fig.add_trace(go.Scatter(
            x=slots, y=gc_traj[pair_key], mode="lines",
            name=f"{sku}",
            line=dict(color=color, width=2),
            hovertemplate=f"{pair_key}<br>t=%{{x}}<br>stock=%{{y}}<extra></extra>",
        ))
        cap_val = gc_cap.get(pair_key, 0)
        if cap_val > 0:
            fig.add_hline(
                y=cap_val, line_dash="dot", line_color=color,
                annotation_text=f"{sku} cap={cap_val}",
                opacity=0.4,
            )

    events = gc_traj.get("events", {})
    for pair_key in sorted(pairs_for_line):
        sku = pair_key.split("_", 1)[1]
        color = GC_PAIR_COLORS.get(pair_key, "#607D8B")
        pair_events = events.get(pair_key, {})
        cancelled_starts = pair_events.get("cancelled_batch_starts", [])
        rst_completions = pair_events.get("restock_completions", [])
        if cancelled_starts:
            fig.add_trace(go.Scatter(
                x=cancelled_starts,
                y=[gc_traj[pair_key][min(t, shift_length - 1)] for t in cancelled_starts],
                mode="markers",
                name=f"{sku} cancelled start",
                marker=dict(symbol="x", size=9, color="#B71C1C", line=dict(width=2)),
                hovertemplate=f"Cancelled batch still consumed GC: {pair_key} at t=%{{x}}<extra></extra>",
            ))
        if rst_completions:
            fig.add_trace(go.Scatter(
                x=rst_completions,
                y=[gc_traj[pair_key][min(t, shift_length - 1)] for t in rst_completions],
                mode="markers",
                name=f"{sku} restock",
                marker=dict(symbol="triangle-up", size=10, color=RESTOCK_COLOR),
                hovertemplate=f"Restock +5 {pair_key} at t=%{{x}}<extra></extra>",
            ))

    restock_shown = False
    for rst in restock_timeline:
        if rst["line_id"] != line_id:
            continue
        color = GC_PAIR_COLORS.get(rst["pair"], RESTOCK_COLOR)
        fig.add_vrect(
            x0=rst["start"],
            x1=rst["end"],
            fillcolor=color,
            opacity=0.08,
            line_width=0,
            annotation_text=None,
            layer="below",
        )
        if not restock_shown:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Restock window",
                marker=dict(symbol="square", size=10, color=RESTOCK_COLOR),
                hoverinfo="skip",
            ))
            restock_shown = True

    fig.add_hline(y=0, line_color="#F44336", line_dash="dash", line_width=1,
                  annotation_text="Empty", opacity=0.5)

    fig.update_layout(
        template="plotly_white",
        title=f"GC Silo Inventory — {line_id}{title_suffix}",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(title="Time Slot", range=[0, shift_length]),
        yaxis=dict(title="GC Stock (batch-units)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def _build_restock_timeline(result: dict[str, Any]) -> go.Figure | None:
    """Build a timeline showing restock operations."""
    restocks = reconstruct_restock_timeline(result.get("restocks"))
    if not restocks:
        return None

    params = result["parameters"]
    shift_length = int(params["SL"])
    fig = go.Figure()

    for rst in restocks:
        pair_key = rst["pair"]
        color = GC_PAIR_COLORS.get(pair_key, RESTOCK_COLOR)
        fig.add_trace(go.Bar(
            x=[rst["end"] - rst["start"]],
            y=[f"{rst['line_id']} {rst['sku']}"],
            base=[rst["start"]],
            orientation="h",
            name=pair_key,
            marker=dict(color=color, opacity=0.7, pattern=dict(shape="x"),
                        line=dict(color="#424242", width=1)),
            hovertemplate=(
                f"Restock {pair_key}<br>"
                f"{rst['start']}–{rst['end']} (+{rst['qty']} GC)"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        fig.add_annotation(
            x=rst["end"], y=f"{rst['line_id']} {rst['sku']}",
            text=f"+{rst['qty']}", showarrow=False,
            font=dict(size=10, color=RESTOCK_COLOR),
            xshift=12,
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Restock Operations ({len(restocks)} total)",
        barmode="overlay",
        height=max(150, 60 + 40 * len(set(r["pair"] for r in restocks))),
        margin=dict(l=80, r=30, t=50, b=35),
        xaxis=dict(title="Time Slot", range=[0, shift_length]),
        yaxis=dict(title=""),
    )
    return fig


def _load_training_profits(training_log_path: Path) -> list[float] | None:
    """Return per-episode profits as a list, or None if unreadable.

    Supports two on-disk formats:
    - dict with ``episode_profits`` key (PPO-style training log)
    - list/tuple of per-episode profit floats (QL-style training log)
    """
    import pickle
    if not training_log_path.exists():
        return None
    try:
        with open(training_log_path, "rb") as f:
            log_data = pickle.load(f)
    except Exception:
        return None
    if isinstance(log_data, dict):
        profits = log_data.get("episode_profits", [])
    elif isinstance(log_data, (list, tuple)):
        profits = list(log_data)
    else:
        return None
    if not profits or len(profits) < 2:
        return None
    try:
        return [float(p) for p in profits]
    except (TypeError, ValueError):
        return None


def _downsample_pairs(xs: list, ys: list, max_points: int = 4000) -> tuple[list, list]:
    if len(xs) <= max_points:
        return xs, ys
    step = max(1, len(xs) // max_points)
    return xs[::step], ys[::step]


def _build_training_curve(training_log_path: Path) -> go.Figure | None:
    """Training curve (raw episode profits + rolling average)."""
    profits = _load_training_profits(training_log_path)
    if profits is None:
        return None

    n = len(profits)
    episodes = list(range(1, n + 1))
    window = max(1, min(500, n // 20))
    smoothed: list[float] = []
    running = 0.0
    q: list[float] = []
    for p in profits:
        q.append(p)
        running += p
        if len(q) > window:
            running -= q.pop(0)
        smoothed.append(running / len(q))

    raw_x, raw_y = _downsample_pairs(episodes, profits)
    smooth_x, smooth_y = _downsample_pairs(episodes, smoothed)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=raw_x, y=raw_y, mode="markers", name="Episode Profit",
        marker=dict(size=2, color="rgba(33, 150, 243, 0.25)"),
    ))
    fig.add_trace(go.Scatter(
        x=smooth_x, y=smooth_y, mode="lines", name=f"Smoothed ({window}-ep avg)",
        line=dict(color="#0D47A1", width=2.5),
    ))
    fig.update_layout(
        template="plotly_white",
        title=f"Training Curve ({n:,} episodes)",
        xaxis_title="Episode",
        yaxis_title="Net Profit ($)",
        height=380,
        margin=dict(l=60, r=20, t=55, b=50),
        hovermode="x unified",
    )
    return fig


def _build_convergence_plot(
    training_log_path: Path,
    epsilon_start: float | None = None,
    epsilon_end: float | None = None,
    epsilon_decay_fraction: float = 0.7,
) -> go.Figure | None:
    """Two-panel convergence plot: epsilon decay (if params given) + smoothed profit."""
    profits = _load_training_profits(training_log_path)
    if profits is None:
        return None
    from plotly.subplots import make_subplots

    n = len(profits)
    window = max(1, min(500, n // 20))
    smoothed: list[float] = []
    running = 0.0
    q: list[float] = []
    for p in profits:
        q.append(p)
        running += p
        if len(q) > window:
            running -= q.pop(0)
        smoothed.append(running / len(q))

    has_eps = epsilon_start is not None and epsilon_end is not None
    if has_eps:
        decay_span = max(1.0, epsilon_decay_fraction * n)
        eps_curve = [
            max(epsilon_end, epsilon_start - min(1.0, i / decay_span) * (epsilon_start - epsilon_end))
            for i in range(n)
        ]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Exploration Decay (epsilon)", f"Profit Convergence ({window}-ep rolling avg)"))
        eps_x, eps_y = _downsample_pairs(list(range(1, n + 1)), eps_curve)
        fig.add_trace(go.Scatter(x=eps_x, y=eps_y, mode="lines", line=dict(color="#C62828", width=2), name="Epsilon"), row=1, col=1)
        prof_x, prof_y = _downsample_pairs(list(range(1, n + 1)), smoothed)
        fig.add_trace(go.Scatter(x=prof_x, y=prof_y, mode="lines", line=dict(color="#1565C0", width=2), name="Smoothed profit"), row=1, col=2)
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_yaxes(title_text="Epsilon", row=1, col=1)
        fig.update_yaxes(title_text="Smoothed Profit ($)", row=1, col=2)
        fig.update_layout(template="plotly_white", height=340, margin=dict(l=60, r=20, t=60, b=40), showlegend=False)
        return fig

    prof_x, prof_y = _downsample_pairs(list(range(1, n + 1)), smoothed)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prof_x, y=prof_y, mode="lines", line=dict(color="#1565C0", width=2), name="Smoothed profit"))
    fig.update_layout(
        template="plotly_white",
        title=f"Profit Convergence ({window}-ep rolling avg)",
        height=320,
        margin=dict(l=60, r=20, t=55, b=40),
        xaxis_title="Episode",
        yaxis_title="Smoothed Profit ($)",
    )
    return fig


FIXED_COLOR = "#90CAF9"


def _convert_reactive_to_universal(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a Reactive CP-SAT reactive_result.json into the universal schema.

    Uses the baseline KPI and constructs a combined schedule from the baseline
    simulation (the actual run), plus oracle data stored under 'reactive_oracle'.
    """
    bk = raw.get("baseline_kpi", {})

    metadata = {
        "solver_engine": "reactive_cpsat",
        "solver_name": "Reactive CP-SAT Oracle",
        "status": "Completed",
        "solve_time_sec": raw.get("aggregate", {}).get("mean_solve_time", 0.0),
        "allow_r3_flex": raw.get("r3_flex", False),
        "notes": f"Seed={raw.get('seed')}, lambda={raw.get('lambda_rate')}, mu={raw.get('mu_mean')}",
    }

    kpi = {
        "net_profit": bk.get("net_profit", 0.0),
        "total_revenue": bk.get("total_revenue", 0.0),
        "total_costs": bk.get("total_costs", 0.0),
        "psc_count": bk.get("psc_count", 0),
        "ndg_count": bk.get("ndg_count", 0),
        "busta_count": bk.get("busta_count", 0),
        "total_batches": bk.get("psc_count", 0) + bk.get("ndg_count", 0) + bk.get("busta_count", 0),
        "tardiness_min": bk.get("tardiness_min", {}),
        "tard_cost": bk.get("tard_cost", 0.0),
        "setup_events": bk.get("setup_events", 0),
        "setup_cost": bk.get("setup_cost", 0.0),
        "stockout_events": bk.get("stockout_events", {}),
        "stockout_cost": bk.get("stockout_cost", 0.0),
        "idle_min": bk.get("idle_min", 0.0),
        "idle_cost": bk.get("idle_cost", 0.0),
        "over_min": bk.get("over_min", 0.0),
        "over_cost": bk.get("over_cost", 0.0),
        "restock_count": bk.get("restock_count", 0),
    }

    experiment = {
        "lambda_rate": raw.get("lambda_rate", 0),
        "mu_mean": raw.get("mu_mean", 0),
        "seed": raw.get("seed"),
        "scenario_label": "reactive_oracle",
    }

    result = create_result(
        metadata=metadata,
        experiment=experiment,
        kpi=kpi,
        schedule=raw.get("baseline_schedule", []),
        cancelled_batches=raw.get("baseline_cancelled", []),
        ups_events=raw.get("baseline_ups", []),
        restocks=raw.get("baseline_restocks", []),
    )

    # Attach oracle data for the reactive-specific panels
    result["reactive_oracle"] = raw.get("oracle_results", [])
    result["reactive_aggregate"] = raw.get("aggregate", {})
    result["reactive_run_name"] = raw.get("run_name", "")
    result["reactive_time_limit"] = raw.get("time_limit_sec", 0)
    return result


def _build_reactive_summary(result: dict[str, Any]) -> go.Figure:
    """Build a table summarising each UPS oracle solve."""
    oracle_results = result.get("reactive_oracle", [])
    baseline_profit = result["kpi"].get("net_profit", 0)
    agg = result.get("reactive_aggregate", {})

    headers = ["UPS #", "t0", "Roaster", "Duration", "Baseline", "Oracle*", "Diff",
               "Status", "Solve (s)", "Gap %", "Batches"]
    rows: list[list[str]] = [[] for _ in headers]
    for o in oracle_results:
        op = o.get("oracle_profit")
        diff = f"${op - baseline_profit:+,.0f}" if op is not None else "N/A"
        sched = [e for e in o.get("schedule", []) if e.get("status") != "fixed_in_progress"]
        rows[0].append(str(o.get("ups_index", "")))
        rows[1].append(str(o["ups_t"]))
        rows[2].append(str(o["ups_roaster"]))
        rows[3].append(f"{o['ups_duration']} min")
        rows[4].append(f"${baseline_profit:,.0f}")
        rows[5].append(f"${op:,.0f}" if op is not None else "N/A")
        rows[6].append(diff)
        rows[7].append(o["oracle_status"])
        rows[8].append(f"{o['solve_time']:.1f}")
        rows[9].append(f"{o.get('gap_pct', 'N/A')}")
        rows[10].append(str(len(sched)))

    fig = go.Figure(data=[go.Table(
        header=dict(values=headers, fill_color="#1565C0", font=dict(color="white", size=12), align="center"),
        cells=dict(values=rows, fill_color="white", align="center", height=28, font=dict(size=11)),
    )])
    mean_p = agg.get("mean_oracle_profit")
    mean_str = f"${mean_p:,.0f}" if mean_p is not None else "N/A"
    title = (
        f"Reactive Oracle Summary — {len(oracle_results)} UPS events | "
        f"Mean Oracle: {mean_str} | "
        f"Feasible: {agg.get('feasible_solves', 0)}/{len(oracle_results)} | "
        f"Optimal: {agg.get('optimal_solves', 0)}/{len(oracle_results)}"
    )
    fig.update_layout(template="plotly_white", title=title, height=max(180, 100 + 30 * len(oracle_results)),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _build_reactive_gantt(oracle_result: dict[str, Any], shift_length: int = 480) -> go.Figure:
    """Build a Gantt chart for a single oracle solve's schedule."""
    t0 = oracle_result["ups_t"]
    ups_roaster = oracle_result["ups_roaster"]
    ups_duration = oracle_result["ups_duration"]
    schedule = oracle_result.get("schedule", [])
    restocks = oracle_result.get("restocks", [])

    fig = go.Figure()

    roasters = sorted({e["roaster"] for e in schedule}) or [ups_roaster]
    row_order = list(reversed(roasters))

    legend_flags: dict[str, bool] = {}

    # UPS downtime block
    _add_interval_bar(
        fig, ups_roaster, t0, min(t0 + ups_duration, shift_length),
        DOWN_COLOR, "UPS Down",
        f"UPS downtime<br>{ups_roaster}: {t0}–{t0 + ups_duration}",
        opacity=0.45, showlegend=True,
    )

    for entry in sorted(schedule, key=lambda e: (e["roaster"], e["start"])):
        is_fixed = entry.get("status") == "fixed_in_progress"
        sku = entry["sku"]
        color = FIXED_COLOR if is_fixed else SKU_COLORS.get(sku, "#607D8B")
        label = f"{sku} (fixed)" if is_fixed else sku
        hover = (
            f"{'Fixed in-progress' if is_fixed else 'Oracle scheduled'}<br>"
            f"SKU: {sku}<br>"
            f"Roaster: {entry['roaster']}<br>"
            f"Time: {entry['start']}–{entry['end']}"
        )
        show = label not in legend_flags
        _add_interval_bar(
            fig, entry["roaster"], entry["start"], entry["end"],
            color, label, hover,
            opacity=0.7 if is_fixed else 1.0,
            line_width=1.0,
            showlegend=show,
        )
        legend_flags[label] = True

    title = (
        f"Oracle Schedule — UPS #{oracle_result.get('ups_index', '?')} "
        f"(t={t0}, {ups_roaster} down {ups_duration}min) — "
        f"Profit: ${oracle_result.get('oracle_profit', 0):,.0f} [{oracle_result['oracle_status']}]"
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        barmode="overlay",
        height=max(200, 60 + 50 * len(roasters)),
        margin=dict(l=70, r=30, t=60, b=40),
        xaxis=dict(title="Time Slot", range=[t0, shift_length], showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="", categoryorder="array", categoryarray=row_order),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
    )
    return fig


def _build_reactive_html(result: dict[str, Any], offline: bool) -> str:
    """Build the full HTML dashboard for a reactive oracle result."""
    figures: list[go.Figure] = []
    figures.append(_build_kpi_header(result))

    # Full-shift baseline Gantt (if schedule data exists)
    if result.get("schedule"):
        figures.append(_build_gantt(result, title_suffix=" — Baseline (Full Shift)"))

    # Full-shift RC stock trajectory
    if result.get("schedule"):
        figures.append(_build_rc_plot(result))

    # GC silo plots
    for line_id in ("L1", "L2"):
        gc_fig = _build_gc_plot(result, line_id, title_suffix=" — Baseline")
        if gc_fig is not None:
            figures.append(gc_fig)

    # Restock timeline
    restock_fig = _build_restock_timeline(result)
    if restock_fig is not None:
        figures.append(restock_fig)

    # Utilization
    if result.get("schedule"):
        figures.append(_build_utilization(result))

    # Reactive oracle summary table + per-UPS Gantts
    figures.append(_build_reactive_summary(result))

    oracle_results = result.get("reactive_oracle", [])
    sl = int(result["parameters"].get("SL", 480))
    for o in oracle_results:
        if o.get("schedule"):
            figures.append(_build_reactive_gantt(o, shift_length=sl))

    figures.append(_build_waterfall(result))
    figures.append(_build_parameters_table(result))

    include_mode: str | bool = True if offline else "cdn"
    divs: list[str] = []
    for index, fig in enumerate(figures):
        divs.append(_figure_to_html(fig, include_plotlyjs=include_mode if index == 0 else False))

    run_name = result.get("reactive_run_name", "Reactive CP-SAT Oracle")
    experiment = result.get("experiment", {})
    seed = experiment.get("seed", "unknown")
    lam = experiment.get("lambda_rate", "?")
    mu = experiment.get("mu_mean", "?")
    title = f"Reactive CP-SAT Oracle — {run_name} — Seed {seed}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{
      background: #ffffff;
      color: #212121;
      font-family: "Segoe UI", Arial, sans-serif;
      margin: 0;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 28px;
    }}
    p.subtitle {{
      margin: 0 0 20px 0;
      color: #616161;
    }}
    .seed-banner {{
      display: inline-block;
      background: #1565C0;
      color: white;
      font-weight: 600;
      font-size: 15px;
      padding: 6px 14px;
      border-radius: 6px;
      margin: 0 0 16px 0;
    }}
    .section {{
      margin-bottom: 18px;
      border: 1px solid #EEEEEE;
      border-radius: 10px;
      padding: 10px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="seed-banner">Eval Seed: {seed} &nbsp;|&nbsp; UPS λ = {lam} &nbsp;|&nbsp; μ = {mu} min</div>
  <p class="subtitle">Offline oracle re-optimisation at each UPS event — interactive dashboard.</p>
  {''.join(f'<div class="section">{div}</div>' for div in divs)}
</body>
</html>"""


def _build_parameters_table(result: dict[str, Any]) -> go.Figure:
    """Build a table showing the shift parameters used to produce this result."""
    params = result.get("parameters", {})
    experiment = result.get("experiment", {})
    metadata = result.get("metadata", {})
    refs = load_reference_data(metadata.get("input_dir"))

    rows_param: list[str] = []
    rows_value: list[str] = []

    def _add(name: str, value: str) -> None:
        rows_param.append(name)
        rows_value.append(value)

    # --- Model provenance (top of table) ---
    _add("Solver", f"{metadata.get('solver_name', '?')} ({metadata.get('solver_engine', '?')})")
    model_path = metadata.get("model_path")
    if model_path:
        _add("Model (exact path)", str(model_path))
    train_started = metadata.get("training_started_at")
    if train_started:
        _add("Training Started", str(train_started))
    train_run_dir = metadata.get("training_run_dir")
    if train_run_dir:
        _add("Training Run Folder", str(train_run_dir))
    if metadata.get("timestamp"):
        _add("Result Generated", str(metadata.get("timestamp")))

    _add("Shift Length", f"{params.get('SL', '?')} min")
    _add("Setup Time", f"{params.get('sigma', '?')} min")
    _add("Consume Duration", f"{params.get('DC', '?')} min")
    _add("Max RC (buffer)", f"{params.get('max_rc', '?')} batches")
    _add("Safety Stock", f"{params.get('safety_stock', '?')} batches")

    rc_init = params.get("rc_init", {})
    _add("RC Init L1", f"{rc_init.get('L1', '?')} batches")
    _add("RC Init L2", f"{rc_init.get('L2', '?')} batches")

    _add("UPS Lambda (Input_data)", str(experiment.get("lambda_rate", params.get("ups_lambda", "?"))))
    _add("UPS Mu (Input_data)", f"{experiment.get('mu_mean', params.get('ups_mu', '?'))} min")
    _add("Restock Duration", f"{params.get('restock_duration', '?')} min")
    _add("Restock Qty", f"{params.get('restock_qty', '?')} batches")
    _add("Roasters", ", ".join(refs.get("roasters", ())))
    _add("Lines", ", ".join(refs.get("lines", ())))

    gc_cap = params.get("gc_capacity", {})
    gc_init = params.get("gc_init", {})
    for pair_key in sorted(set(list(gc_cap.keys()) + list(gc_init.keys()))):
        init_v = gc_init.get(pair_key, "?")
        cap_v = gc_cap.get(pair_key, "?")
        _add(f"GC {pair_key}", f"init={init_v}, cap={cap_v}")

    for jid in sorted(refs.get("job_batches", {}).keys()):
        sku = refs.get("job_sku", {}).get(jid, "?")
        count = refs.get("job_batches", {}).get(jid, "?")
        due = refs.get("job_due", {}).get(jid, "?")
        _add(f"MTO Job {jid}", f"{sku} x{count}, due={due}")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Parameter</b>", "<b>Value</b>"],
            fill_color="#37474F",
            font=dict(color="white", size=13),
            align=["left", "right"],
        ),
        cells=dict(
            values=[rows_param, rows_value],
            fill_color=[["white", "#F5F5F5"] * ((len(rows_param) + 1) // 2)],
            align=["left", "right"],
            height=28,
            font=dict(size=12),
        ),
    )])
    fig.update_layout(
        template="plotly_white",
        title="Shift Parameters (Input Data)",
        height=max(250, 60 + 30 * len(rows_param)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _figure_to_html(fig: go.Figure, include_plotlyjs: str | bool) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=include_plotlyjs, config={"displaylogo": False, "responsive": True})


def _build_html(
    result: dict[str, Any],
    compare: dict[str, Any] | None,
    offline: bool,
    training_info: dict[str, Any] | None = None,
) -> str:
    """Render a schedule-analysis dashboard.

    ``training_info`` is an optional dict for learned-policy results:
        {"log_path": str | Path, "epsilon_start": float, "epsilon_end": float}
    When provided and the log exists, a training curve (and convergence
    panel if epsilon params are given) is appended after the parameters table.
    """

    if "reactive_oracle" in result:
        return _build_reactive_html(result, offline)

    figures: list[go.Figure] = []
    figures.append(_build_kpi_header(result, compare))
    figures.append(_build_gantt(result, title_suffix=" — Result A" if compare else ""))
    if compare:
        figures.append(_build_gantt(compare, title_suffix=" — Result B"))
    figures.append(_build_rc_plot(result, compare))

    gc_l1 = _build_gc_plot(result, "L1")
    if gc_l1 is not None:
        figures.append(gc_l1)
    gc_l2 = _build_gc_plot(result, "L2")
    if gc_l2 is not None:
        figures.append(gc_l2)

    restock_fig = _build_restock_timeline(result)
    if restock_fig is not None:
        figures.append(restock_fig)

    figures.append(_build_waterfall(result, compare))
    routing = _build_r3_routing(result, compare)
    if routing is not None:
        figures.append(routing)
    figures.append(_build_utilization(result, compare))
    figures.append(_build_pipeline_timeline(result, title_suffix=" — Result A" if compare else ""))
    if compare:
        figures.append(_build_pipeline_timeline(compare, title_suffix=" — Result B"))
    figures.append(_build_parameters_table(result))

    if training_info:
        log_path_raw = training_info.get("log_path")
        if log_path_raw:
            log_path = Path(log_path_raw)
            curve_fig = _build_training_curve(log_path)
            if curve_fig is not None:
                figures.append(curve_fig)
            conv_fig = _build_convergence_plot(
                log_path,
                epsilon_start=training_info.get("epsilon_start"),
                epsilon_end=training_info.get("epsilon_end"),
            )
            if conv_fig is not None:
                figures.append(conv_fig)

    include_mode: str | bool = True if offline else "cdn"
    divs: list[str] = []
    for index, fig in enumerate(figures):
        divs.append(_figure_to_html(fig, include_plotlyjs=include_mode if index == 0 else False))
    experiment = result.get("experiment", {})
    seed = experiment.get("seed", "unknown")
    lam = experiment.get("lambda_rate", "?")
    mu = experiment.get("mu_mean", "?")
    title = f"Schedule Analysis — {result['metadata'].get('solver_engine', 'unknown')} — Seed {seed}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{
      background: #ffffff;
      color: #212121;
      font-family: "Segoe UI", Arial, sans-serif;
      margin: 0;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 28px;
    }}
    p.subtitle {{
      margin: 0 0 20px 0;
      color: #616161;
    }}
    .seed-banner {{
      display: inline-block;
      background: #1565C0;
      color: white;
      font-weight: 600;
      font-size: 15px;
      padding: 6px 14px;
      border-radius: 6px;
      margin: 0 0 16px 0;
    }}
    .section {{
      margin-bottom: 18px;
      border: 1px solid #EEEEEE;
      border-radius: 10px;
      padding: 10px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="seed-banner">Eval Seed: {seed} &nbsp;|&nbsp; UPS λ = {lam} &nbsp;|&nbsp; μ = {mu} min</div>
  <p class="subtitle">Interactive schedule analysis dashboard generated from the universal result schema.</p>
  {''.join(f'<div class="section">{div}</div>' for div in divs)}
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot a thesis result JSON as an interactive Plotly dashboard, or open an existing report HTML.")
    parser.add_argument("result_json", nargs="?", help="Primary result JSON path, or an existing report HTML path. Defaults to the latest detected result artifact.")
    parser.add_argument("--output", help="Output HTML path.")
    parser.add_argument("--compare", help="Optional second result JSON for comparison mode.")
    parser.add_argument("--offline", action="store_true", help="Embed plotly.js directly instead of using the CDN.")
    args = parser.parse_args(argv)

    result_json = args.result_json
    if not result_json:
        latest = _find_latest_input_file()
        if latest is None:
            parser.error("result_json was not provided and no result JSON or report HTML could be found automatically.")
        result_json = str(latest)
        print(f"Using latest result artifact: {latest}")

    input_path = Path(result_json)
    if input_path.suffix.lower() == ".html" and input_path.exists():
        if args.compare:
            parser.error("--compare is only supported when the primary input is a result JSON file.")
        if args.output:
            parser.error("--output is not supported when opening an existing HTML report.")
        print(f"Opening existing HTML report: {input_path.resolve()}")
        _open_in_browser(input_path)
        return 0

    result = _load_any_result(result_json)
    compare = _load_any_result(args.compare) if args.compare else None
    output = Path(args.output) if args.output else Path("results") / f"{Path(result_json).stem}_plot.html"
    if not output.is_absolute():
        output = ROOT_DIR / output

    html = _build_html(result, compare, args.offline)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Saved dashboard to {output}")
    _open_in_browser(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
