"""
app.py — Flask application for the MILP scheduling solver GUI.

Routes:
  GET  /             — serve main page
  POST /api/load     — load CSV data
  POST /api/solve    — run base MILP solve
  POST /api/disrupt  — insert a disruption
  POST /api/resolve  — re-solve from disruption time
  GET  /api/export   — export schedule as CSV
  GET  /api/state    — get current state
"""

import os
import sys
import json
import csv
import io
from flask import Flask, render_template, request, jsonify, send_file

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all_data, ShiftData
from gui_state import (
    SolveResult, build_gantt_data, build_rc_chart_data, build_solver_summary
)
from MILP import solve_milp

app = Flask(__name__)

# ──────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────

_state = {
    "data": None,            # ShiftData
    "result": None,           # SolveResult
    "disruptions": [],        # list of {time_min, roaster_id, duration_min, ...}
    "data_loaded": False,
    "solved": False,
    "input_dir": None,
}


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load", methods=["POST"])
def api_load():
    """Load CSV data from input directory."""
    try:
        # Accept custom input_dir from request, or use default
        body = request.get_json(silent=True) or {}
        input_dir = body.get("input_dir", None)

        if not input_dir:
            # Default: look for Input_data_sample relative to project
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            input_dir = os.path.join(project_root, "Input_data_sample")

        if not os.path.isdir(input_dir):
            return jsonify({"success": False, "error": f"Directory not found: {input_dir}"}), 400

        data = load_all_data(input_dir)
        _state["data"] = data
        _state["input_dir"] = input_dir
        _state["data_loaded"] = True
        _state["disruptions"] = []
        _state["solved"] = False
        _state["result"] = None

        summary = {
            "roasters": len(data.roasters),
            "roaster_ids": list(data.roasters.keys()),
            "skus": list(data.skus.keys()),
            "jobs": [
                {
                    "job_id": j.job_id,
                    "sku": j.sku,
                    "batches": j.required_batches,
                    "due_time": j.due_time,
                }
                for j in data.jobs
            ],
            "shift_length": data.shift_length,
            "initial_rc": {"L1": data.initial_rc_l1, "L2": data.initial_rc_l2},
            "max_rc": data.max_rc_per_line,
            "safety_stock": data.safety_stock,
            "consume_events_l1": len(data.consumption_schedule_l1),
            "consume_events_l2": len(data.consumption_schedule_l2),
            "planned_downtime": [
                {
                    "roaster_id": d.roaster_id,
                    "start": d.start_min,
                    "end": d.end_min,
                    "reason": d.reason,
                }
                for d in data.planned_downtime
            ],
        }

        return jsonify({"success": True, "summary": summary})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/solve", methods=["POST"])
def api_solve():
    """Run the base deterministic MILP solve."""
    if not _state["data_loaded"]:
        return jsonify({"success": False, "error": "Data not loaded. Click Load Data first."}), 400

    try:
        data = _state["data"]
        _state["disruptions"] = []

        result = solve_milp(data)
        _state["result"] = result
        _state["solved"] = True

        return jsonify({
            "success": True,
            "summary": build_solver_summary(result),
            "gantt": build_gantt_data(result, list(data.roasters.keys())),
            "rc_chart": build_rc_chart_data(result, data.max_rc_per_line, data.safety_stock),
            "utilization": result.utilization,
            "job_tardiness": result.job_tardiness,
            "batch_count": len(result.batches),
            "schedule": [
                {
                    "batch_id": b.batch_id,
                    "sku": b.sku,
                    "roaster": b.roaster_id,
                    "start": b.start_time,
                    "end": b.end_time,
                    "output_line": b.output_line,
                    "setup": b.setup_before,
                    "revenue": b.revenue,
                }
                for b in result.batches
            ],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/disrupt", methods=["POST"])
def api_disrupt():
    """Insert a disruption event."""
    body = request.get_json()
    if not body:
        return jsonify({"success": False, "error": "No JSON body"}), 400

    required = ["time_min", "roaster_id", "duration_min"]
    for key in required:
        if key not in body:
            return jsonify({"success": False, "error": f"Missing field: {key}"}), 400

    disruption = {
        "time_min": int(body["time_min"]),
        "roaster_id": body["roaster_id"],
        "event_type": body.get("event_type", "breakdown"),
        "duration_min": int(body["duration_min"]),
        "note": body.get("note", "manual"),
    }

    # Validate roaster exists
    if _state["data"] and disruption["roaster_id"] not in _state["data"].roasters:
        return jsonify({"success": False, "error": f"Unknown roaster: {disruption['roaster_id']}"}), 400

    _state["disruptions"].append(disruption)

    return jsonify({
        "success": True,
        "disruptions": _state["disruptions"],
        "count": len(_state["disruptions"]),
    })


@app.route("/api/resolve", methods=["POST"])
def api_resolve():
    """Re-solve from the earliest disruption time."""
    if not _state["data_loaded"]:
        return jsonify({"success": False, "error": "Data not loaded."}), 400
    if not _state["disruptions"]:
        return jsonify({"success": False, "error": "No disruptions to re-solve from."}), 400

    try:
        data = _state["data"]
        disruptions = _state["disruptions"]

        # Find earliest disruption time
        t_start = min(d["time_min"] for d in disruptions)

        # Get current RC state at disruption time from previous solve
        rc_at_t = {"L1": data.initial_rc_l1, "L2": data.initial_rc_l2}
        if _state["result"]:
            prev = _state["result"]
            for line in ["L1", "L2"]:
                traj = prev.rc_trajectory.get(line, [])
                for t, lvl in traj:
                    if t <= t_start:
                        rc_at_t[line] = lvl
                    else:
                        break

        result = solve_milp(
            data,
            t_start=t_start,
            initial_rc=rc_at_t,
            disruptions=disruptions,
            reactive_mode=True,
        )
        _state["result"] = result
        _state["solved"] = True

        return jsonify({
            "success": True,
            "summary": build_solver_summary(result),
            "gantt": build_gantt_data(result, list(data.roasters.keys())),
            "rc_chart": build_rc_chart_data(result, data.max_rc_per_line, data.safety_stock),
            "utilization": result.utilization,
            "job_tardiness": result.job_tardiness,
            "batch_count": len(result.batches),
            "disruptions": disruptions,
            "schedule": [
                {
                    "batch_id": b.batch_id,
                    "sku": b.sku,
                    "roaster": b.roaster_id,
                    "start": b.start_time,
                    "end": b.end_time,
                    "output_line": b.output_line,
                    "setup": b.setup_before,
                    "revenue": b.revenue,
                }
                for b in result.batches
            ],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/export", methods=["GET"])
def api_export():
    """Export the current schedule as CSV."""
    if not _state["solved"] or not _state["result"]:
        return jsonify({"success": False, "error": "No solution to export."}), 400

    result = _state["result"]

    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "batch_id", "sku", "roaster_id", "start_time", "end_time",
        "consume_start", "consume_end", "output_line",
        "setup_before", "revenue", "job_id"
    ])
    for b in result.batches:
        writer.writerow([
            b.batch_id, b.sku, b.roaster_id, b.start_time, b.end_time,
            b.consume_start, b.consume_end, b.output_line,
            b.setup_before, b.revenue, b.job_id or ""
        ])

    # Save to results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "schedule_export.csv")
    with open(csv_path, "w", newline="") as f:
        f.write(output.getvalue())

    # Return as download
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="schedule_export.csv",
    )


@app.route("/api/state", methods=["GET"])
def api_state():
    """Get current application state."""
    return jsonify({
        "data_loaded": _state["data_loaded"],
        "solved": _state["solved"],
        "disruptions": _state["disruptions"],
        "roaster_ids": list(_state["data"].roasters.keys()) if _state["data"] else [],
    })


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MILP Roasting Scheduler — Flask GUI")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
