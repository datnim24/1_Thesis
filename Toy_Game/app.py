"""
Flask application for the Coffee Roasting Simulation.
Provides REST API for simulation control and serves the dashboard UI.
"""

from flask import Flask, render_template, jsonify, request, Response
from simulation.engine import SimulationEngine
import csv
import io
import json

app = Flask(__name__)
engine = None  # Global simulation engine instance


def get_engine():
    global engine
    if engine is None:
        engine = SimulationEngine()
    return engine


# --- Page Routes ---

@app.route("/")
def index():
    return render_template("index.html")


# --- API Routes ---

@app.route("/api/start", methods=["POST"])
def api_start():
    """Initialize or reset the simulation."""
    global engine
    data = request.get_json() or {}

    config = {
        "rc_init_L1": data.get("rc_init_L1", 20),
        "rc_init_L2": data.get("rc_init_L2", 20),
        "rc_max_L1": data.get("rc_max_L1", 40),
        "rc_max_L2": data.get("rc_max_L2", 40),
        "consumption_rate_L1": data.get("consumption_rate_L1", 5.1),
        "consumption_rate_L2": data.get("consumption_rate_L2", 4.8),
        "disruption_lambda": data.get("disruption_lambda", 0),
        "disruption_min": data.get("disruption_min", 10),
        "disruption_max": data.get("disruption_max", 30),
        "mto_ndg": data.get("mto_ndg", 3),
        "mto_busta": data.get("mto_busta", 1),
        "seed": data.get("seed", None),
    }

    engine = SimulationEngine(config)
    return jsonify({"success": True, "message": "Simulation initialized"})


@app.route("/api/state", methods=["GET"])
def api_state():
    """Get current simulation state."""
    eng = get_engine()
    return jsonify(eng.get_state())


@app.route("/api/step", methods=["POST"])
def api_step():
    """Advance simulation by N steps."""
    eng = get_engine()
    data = request.get_json() or {}
    n = data.get("steps", 1)
    n = min(n, 50)  # Cap at 50 steps per request
    results = eng.step(n)
    return jsonify({
        "success": True,
        "steps_taken": len(results),
        "state": eng.get_state(),
    })


@app.route("/api/schedule", methods=["POST"])
def api_schedule():
    """Schedule a batch on a roaster."""
    eng = get_engine()
    data = request.get_json() or {}
    roaster_id = data.get("roaster")
    sku = data.get("sku")
    target_line = data.get("target_line")

    if not roaster_id or not sku:
        return jsonify({"error": "Missing roaster or sku"}), 400

    result = eng.schedule_batch(roaster_id, sku, target_line)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/disrupt", methods=["POST"])
def api_disrupt():
    """Trigger a manual disruption."""
    eng = get_engine()
    data = request.get_json() or {}
    roaster_id = data.get("roaster")
    duration = data.get("duration", 20)

    if not roaster_id:
        return jsonify({"error": "Missing roaster"}), 400

    result = eng.trigger_disruption(roaster_id, duration)
    return jsonify(result)


@app.route("/api/auto", methods=["POST"])
def api_auto():
    """Toggle auto mode."""
    eng = get_engine()
    data = request.get_json() or {}
    eng.auto_mode = data.get("enabled", not eng.auto_mode)
    return jsonify({"auto_mode": eng.auto_mode})


@app.route("/api/speed", methods=["POST"])
def api_speed():
    """Set simulation speed."""
    eng = get_engine()
    data = request.get_json() or {}
    eng.speed = data.get("speed", 1)
    return jsonify({"speed": eng.speed})


@app.route("/api/consumption_rate", methods=["POST"])
def api_consumption_rate():
    """Update PSC consumption rate."""
    eng = get_engine()
    data = request.get_json() or {}
    line = data.get("line", "L1")
    rate = data.get("rate", 5.0)
    rate = max(1.0, min(20.0, rate))

    if line in eng.silos:
        eng.silos[line].update_consumption_schedule(rate)
    return jsonify({"success": True, "line": line, "rate": rate})


@app.route("/api/report", methods=["GET"])
def api_report():
    """Generate end-of-shift report."""
    eng = get_engine()
    report = eng.generate_report()
    return jsonify(report)


@app.route("/api/report/csv", methods=["GET"])
def api_report_csv():
    """Download report as CSV."""
    eng = get_engine()
    report = eng.generate_report()

    output = io.StringIO()
    writer = csv.writer(output)

    # KPI Summary
    writer.writerow(["KPI", "Value"])
    writer.writerow(["Shift Duration (min)", report["shift_duration"]])
    writer.writerow(["Total PSC Batches", report["total_psc_batches"]])
    writer.writerow(["PSC L1", report["psc_L1"]])
    writer.writerow(["PSC L2", report["psc_L2"]])
    writer.writerow(["MTO NDG Completed", f"{report['mto_ndg_completed']}/{report['mto_ndg_total']}"])
    writer.writerow(["MTO Busta Completed", f"{report['mto_busta_completed']}/{report['mto_busta_total']}"])
    writer.writerow(["MTO Tardiness (min)", report["mto_tardiness"]])
    writer.writerow(["Stockout Events", report["stockout_events"]])
    writer.writerow(["Stockout Duration (min)", report["stockout_duration"]])
    writer.writerow(["Avg Utilization (%)", report["avg_utilization"]])
    writer.writerow([])

    # Roaster details
    writer.writerow(["Roaster", "Utilization (%)", "Batches", "Roast Time", "Setup Time", "Down Time", "Idle Time"])
    for rid, data in report["roaster_utilization"].items():
        writer.writerow([
            rid, data["utilization"], data["batches"],
            data["roast_time"], data["setup_time"],
            data["down_time"], data["idle_time"],
        ])

    content = output.getvalue()
    return Response(
        content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=shift_report.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
