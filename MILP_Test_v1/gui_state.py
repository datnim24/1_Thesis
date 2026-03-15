"""
gui_state.py — State container passed to the Flask frontend.

Holds solve results and provides methods to produce
Plotly-ready JSON for Gantt chart, RC stock chart, and solver summary.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class BatchResult:
    """Single batch assignment produced by the solver."""
    batch_id: str
    job_id: Optional[str]      # None for PSC pool batches
    sku: str
    roaster_id: str
    start_time: int
    end_time: int              # start_time + process_time
    consume_start: int         # = start_time
    consume_end: int           # = start_time + consume_time
    output_line: str           # L1 or L2
    setup_before: int          # 0 or setup_time
    revenue: float


@dataclass
class SolveResult:
    """Complete result from one MILP solve."""
    # Solver metrics
    status: str = "not_solved"          # optimal, feasible, infeasible, error
    solve_time_sec: float = 0.0
    mip_gap: float = 0.0
    objective_value: float = 0.0
    best_bound: float = 0.0
    num_variables: int = 0
    num_constraints: int = 0
    solver_log: str = ""

    # Schedule
    batches: List[BatchResult] = field(default_factory=list)

    # RC trajectories: {line_id: [(time, stock_level), ...]}
    rc_trajectory: Dict[str, List[tuple]] = field(default_factory=dict)

    # KPIs
    total_revenue: float = 0.0
    psc_batches_completed: int = 0
    mto_batches_completed: int = 0
    total_tardiness_min: float = 0.0
    tardiness_cost: float = 0.0
    stockout_minutes: int = 0
    stockout_cost: float = 0.0
    idle_minutes: int = 0
    idle_cost: float = 0.0
    overflow_idle_minutes: int = 0
    overflow_idle_cost: float = 0.0
    net_profit: float = 0.0

    # Per-roaster utilization
    utilization: Dict[str, float] = field(default_factory=dict)

    # Per-job tardiness
    job_tardiness: Dict[str, float] = field(default_factory=dict)

    # Disruptions applied
    disruptions_applied: List[Dict] = field(default_factory=list)

    # Solve mode
    solve_mode: str = "deterministic"   # deterministic or reactive


def build_gantt_data(result: SolveResult, roaster_order: List[str]) -> Dict[str, Any]:
    """
    Build Plotly-ready data for the Gantt chart.
    Returns dict with 'data' (list of traces) and 'layout'.
    """
    # Color mapping by SKU
    sku_colors = {
        "PSC":   "#00d4aa",   # teal
        "NDG":   "#a855f7",   # purple
        "BUSTA": "#f97316",   # orange
    }
    sku_colors_setup = {
        "PSC":   "#005544",
        "NDG":   "#4a2080",
        "BUSTA": "#7a3a0b",
    }

    traces = []

    # Setup bars
    for b in result.batches:
        if b.setup_before > 0:
            setup_start = b.start_time - b.setup_before
            traces.append({
                "type": "bar",
                "y": [b.roaster_id],
                "x": [b.setup_before],
                "base": [setup_start],
                "orientation": "h",
                "marker": {"color": sku_colors_setup.get(b.sku, "#333")},
                "name": f"Setup",
                "hovertemplate": (
                    f"<b>SETUP</b><br>"
                    f"Roaster: {b.roaster_id}<br>"
                    f"Before: {b.sku}<br>"
                    f"Time: {setup_start}–{b.start_time}<br>"
                    "<extra></extra>"
                ),
                "showlegend": False,
            })

    # Batch bars
    for b in result.batches:
        traces.append({
            "type": "bar",
            "y": [b.roaster_id],
            "x": [b.end_time - b.start_time],
            "base": [b.start_time],
            "orientation": "h",
            "marker": {"color": sku_colors.get(b.sku, "#888")},
            "name": b.sku,
            "hovertemplate": (
                f"<b>{b.batch_id}</b> ({b.sku})<br>"
                f"Roaster: {b.roaster_id}<br>"
                f"Time: {b.start_time}–{b.end_time}<br>"
                f"Output: {b.output_line}<br>"
                f"Revenue: ${b.revenue:,.0f}<br>"
                "<extra></extra>"
            ),
            "showlegend": False,
        })

    # Disruption overlays
    for d in result.disruptions_applied:
        traces.append({
            "type": "bar",
            "y": [d["roaster_id"]],
            "x": [d["duration_min"]],
            "base": [d["time_min"]],
            "orientation": "h",
            "marker": {
                "color": "rgba(239, 68, 68, 0.5)",
                "line": {"color": "#ef4444", "width": 2},
            },
            "name": "Disruption",
            "hovertemplate": (
                f"<b>DISRUPTION</b><br>"
                f"Roaster: {d['roaster_id']}<br>"
                f"Time: {d['time_min']}–{d['time_min']+d['duration_min']}<br>"
                f"Duration: {d['duration_min']} min<br>"
                "<extra></extra>"
            ),
            "showlegend": False,
        })

    # Planned downtime overlays
    # (these are passed via result.disruptions_applied with event_type='planned')

    layout = {
        "barmode": "overlay",
        "yaxis": {
            "categoryorder": "array",
            "categoryarray": list(reversed(roaster_order)),
            "title": "",
        },
        "xaxis": {
            "title": "Time (minutes)",
            "range": [0, 480],
            "dtick": 30,
        },
        "height": 320,
        "margin": {"l": 60, "r": 20, "t": 30, "b": 40},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15,17,23,0.5)",
        "font": {"color": "#e0e0e0"},
    }

    return {"data": traces, "layout": layout}


def build_rc_chart_data(result: SolveResult, max_rc: int, safety: int) -> Dict[str, Any]:
    """
    Build Plotly-ready data for the RC stock chart.
    """
    traces = []

    line_colors = {"L1": "#38bdf8", "L2": "#fb923c"}

    for line_id, trajectory in result.rc_trajectory.items():
        times = [pt[0] for pt in trajectory]
        levels = [pt[1] for pt in trajectory]
        traces.append({
            "type": "scatter",
            "x": times,
            "y": levels,
            "mode": "lines",
            "name": f"RC {line_id}",
            "line": {"color": line_colors.get(line_id, "#888"), "width": 2},
        })

    # Safety stock threshold line
    traces.append({
        "type": "scatter",
        "x": [0, 480],
        "y": [safety, safety],
        "mode": "lines",
        "name": "Safety Stock",
        "line": {"color": "#facc15", "width": 1, "dash": "dash"},
    })

    # Max buffer line
    traces.append({
        "type": "scatter",
        "x": [0, 480],
        "y": [max_rc, max_rc],
        "mode": "lines",
        "name": "Max Buffer",
        "line": {"color": "#ef4444", "width": 1, "dash": "dash"},
    })

    # Zero line
    traces.append({
        "type": "scatter",
        "x": [0, 480],
        "y": [0, 0],
        "mode": "lines",
        "name": "Stockout",
        "line": {"color": "#ef4444", "width": 2, "dash": "dot"},
    })

    layout = {
        "xaxis": {"title": "Time (minutes)", "range": [0, 480], "dtick": 30},
        "yaxis": {"title": "RC Stock (batches)", "range": [-2, max_rc + 5]},
        "height": 280,
        "margin": {"l": 60, "r": 20, "t": 30, "b": 40},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15,17,23,0.5)",
        "font": {"color": "#e0e0e0"},
        "legend": {"x": 0.01, "y": 0.99, "bgcolor": "rgba(0,0,0,0.3)"},
    }

    return {"data": traces, "layout": layout}


def build_solver_summary(result: SolveResult) -> Dict[str, Any]:
    """Build a summary dict for the solver metrics panel."""
    return {
        "status": result.status,
        "solve_time": f"{result.solve_time_sec:.2f}s",
        "mip_gap": f"{result.mip_gap*100:.2f}%",
        "objective": f"${result.objective_value:,.0f}",
        "best_bound": f"${result.best_bound:,.0f}",
        "num_variables": result.num_variables,
        "num_constraints": result.num_constraints,
        "total_revenue": f"${result.total_revenue:,.0f}",
        "net_profit": f"${result.net_profit:,.0f}",
        "psc_batches": result.psc_batches_completed,
        "mto_batches": result.mto_batches_completed,
        "tardiness_min": f"{result.total_tardiness_min:.0f}",
        "tardiness_cost": f"${result.tardiness_cost:,.0f}",
        "stockout_min": result.stockout_minutes,
        "stockout_cost": f"${result.stockout_cost:,.0f}",
        "idle_min": result.idle_minutes,
        "idle_cost": f"${result.idle_cost:,.0f}",
        "overflow_idle_min": result.overflow_idle_minutes,
        "overflow_idle_cost": f"${result.overflow_idle_cost:,.0f}",
        "solve_mode": result.solve_mode,
    }
