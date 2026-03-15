/* ==========================================================
   Coffee Roasting Simulation — Frontend Application
   Handles polling, UI rendering, drag-and-drop, charts
   ========================================================== */

// --- State ---
let simState = null;
let playing = false;
let playInterval = null;
let speed = 1;
let charts = {};

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    initDragDrop();
    initSimulation();
});

async function initSimulation() {
    // Read MTO config from inputs if they exist
    const mtoNdg = parseInt(document.getElementById('input-mto-ndg')?.value) || 3;
    const mtoBusta = parseInt(document.getElementById('input-mto-busta')?.value) || 1;

    // Read current consumption rate slider values so Reset preserves them
    const rateL1 = parseFloat(document.getElementById('rate-slider-L1')?.value) || 5.1;
    const rateL2 = parseFloat(document.getElementById('rate-slider-L2')?.value) || 4.8;

    const res = await apiPost("/api/start", {
        mto_ndg: mtoNdg,
        mto_busta: mtoBusta,
        consumption_rate_L1: rateL1,
        consumption_rate_L2: rateL2,
    });
    if (res.success) {
        playing = false;
        clearTimeout(playInterval);
        updatePlayButton();

        // Reset charts data
        if (charts.rcStock) {
            charts.rcStock.data.labels = [];
            charts.rcStock.data.datasets[0].data = [];
            charts.rcStock.data.datasets[1].data = [];
            charts.rcStock.update('none');
        }
        if (charts.utilization) {
            charts.utilization.data.datasets[0].data = [0,0,0,0,0];
            charts.utilization.update('none');
        }
        if (charts.batches) {
            charts.batches.data.datasets[0].data = [0,0,0,0,0];
            charts.batches.update('none');
        }

        await fetchState();
        showToast("Simulation initialized", "success");
    }
}

// --- API Helpers ---
async function apiGet(url) {
    const res = await fetch(url);
    return res.json();
}

async function apiPost(url, data) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    return res.json();
}

// --- State Fetching ---
async function fetchState() {
    simState = await apiGet("/api/state");
    updateUI(simState);
}

// --- Play / Pause ---
function togglePlay() {
    if (simState && simState.finished) {
        showToast("Shift complete! Generate a report.", "");
        return;
    }
    playing = !playing;
    updatePlayButton();

    if (playing) {
        runSimLoop();
    } else {
        clearInterval(playInterval);
    }
}

function updatePlayButton() {
    const btn = document.getElementById("btn-play");
    if (playing) {
        btn.textContent = "⏸ Pause";
        btn.classList.add("playing");
    } else {
        btn.textContent = "▶ Play";
        btn.classList.remove("playing");
    }
}

async function runSimLoop() {
    if (!playing) return;

    // For slow speeds (<1), always step 1 but increase delay
    // For fast speeds (>=1), step more per tick with shorter delay
    const stepsPerTick = speed >= 1 ? Math.min(Math.floor(speed), 50) : 1;
    const res = await apiPost("/api/step", { steps: stepsPerTick });
    if (res.state) {
        simState = res.state;
        updateUI(simState);
    }

    if (simState && simState.finished) {
        playing = false;
        updatePlayButton();
        showToast("🏁 Shift complete!", "success");
        return;
    }

    // Delay mapping: slower speed = longer delay between ticks
    let delay;
    if (speed <= 0.1) delay = 2000;
    else if (speed <= 0.2) delay = 1000;
    else if (speed <= 0.5) delay = 500;
    else if (speed <= 1) delay = 200;
    else if (speed <= 5) delay = 100;
    else if (speed <= 10) delay = 60;
    else delay = 30;

    playInterval = setTimeout(runSimLoop, delay);
}

async function stepOnce() {
    const res = await apiPost("/api/step", { steps: 1 });
    if (res.state) {
        simState = res.state;
        updateUI(simState);
    }
}

function setSpeed(s) {
    speed = s;
    document.querySelectorAll(".btn-speed").forEach(b => {
        b.classList.toggle("active", parseFloat(b.dataset.speed) === s);
    });
    apiPost("/api/speed", { speed: s });
}

async function toggleAuto() {
    const res = await apiPost("/api/auto", {});
    const btn = document.getElementById("btn-auto");
    btn.classList.toggle("active", res.auto_mode);
    showToast(res.auto_mode ? "Auto Mode ON" : "Auto Mode OFF", "success");
}

// --- UI Update ---
function updateUI(state) {
    if (!state) return;

    updateClock(state.time);
    updateRoasters(state.roasters);
    updateSilos(state.silos);
    updatePipelines(state.pipelines);
    updateMTO(state.mto_jobs);
    updateGantt(state.gantt, state.time);
    updateEventLog(state.recent_events);
    updateDecisionLog(state.decision_log);
    updateCharts(state);
}

function updateClock(time) {
    const hours = Math.floor(time / 60);
    const mins = time % 60;
    document.getElementById("clock-value").textContent =
        `${String(hours).padStart(2, "0")}:${String(mins).padStart(2, "0")}`;
    document.getElementById("clock-minutes").textContent = `${time} / 480 min`;
    document.getElementById("shift-progress").style.width = `${(time / 480) * 100}%`;
}

function updateRoasters(roasters) {
    for (const [rid, r] of Object.entries(roasters)) {
        const card = document.getElementById(`roaster-${rid}`);
        if (!card) continue;

        // Remove old state classes
        card.className = `roaster-card state-${r.state}`;

        // Timer bar width
        let timerPct = 0;
        if (r.state === "Roasting") timerPct = (r.remaining_time / 15) * 100;
        else if (r.state === "Setup") timerPct = (r.remaining_time / 5) * 100;
        else if (r.state === "Down") timerPct = Math.min(100, (r.remaining_time / 30) * 100);

        card.innerHTML = `
            <div class="roaster-id">${rid}</div>
            <span class="roaster-state st-${r.state}">${r.state}</span>
            <div class="roaster-info">
                ${r.current_sku ? `<div><strong>SKU:</strong> ${r.current_sku}</div>` : ""}
                ${r.remaining_time > 0 ? `<div><strong>Time:</strong> ${r.remaining_time} min</div>` : ""}
                ${r.pending_sku ? `<div><strong>Next:</strong> ${r.pending_sku}</div>` : ""}
                <div><strong>Done:</strong> ${r.batches_completed} batches</div>
                <div><strong>Util:</strong> ${r.utilization}%</div>
            </div>
            <div class="roaster-timer" style="width:${timerPct}%;
                background:${r.state === 'Down' ? 'var(--danger)' : r.state === 'Setup' ? 'var(--setup-color)' : 'var(--success)'}">
            </div>
        `;
    }
}

function updateSilos(silos) {
    for (const [lid, s] of Object.entries(silos)) {
        const fill = document.getElementById(`silo-fill-${lid}`);
        const level = document.getElementById(`silo-level-${lid}`);
        const rate = document.getElementById(`silo-rate-${lid}`);

        if (fill) {
            const pct = Math.max(0, Math.min(100, s.fill_percent));
            fill.style.height = `${pct}%`;
            fill.className = `silo-fill status-${s.status}`;
        }
        if (level) level.textContent = s.stock;
        if (rate) rate.textContent = `ρ = ${s.consumption_rate}`;
    }
}

function updatePipelines(pipelines) {
    for (const [lid, p] of Object.entries(pipelines)) {
        const el = document.getElementById(`pipeline-${lid}`);
        if (!el) continue;
        const statusEl = el.querySelector(".pipe-status");
        if (p.busy > 0) {
            statusEl.textContent = `Busy (${p.roaster}, ${p.busy}m)`;
            statusEl.classList.add("busy");
        } else {
            statusEl.textContent = "Free";
            statusEl.classList.remove("busy");
        }
    }
}

function updateMTO(mto) {
    const ndg = mto.NDG;
    const busta = mto.Busta;
    document.getElementById("mto-ndg").textContent =
        `${ndg.completed}/${ndg.completed + ndg.remaining}`;
    document.getElementById("mto-busta").textContent =
        `${busta.completed}/${busta.completed + busta.remaining}`;
}

// --- Gantt Chart ---
function updateGantt(gantt, currentTime) {
    const container = document.getElementById("gantt-chart");
    if (!container || !gantt) return;

    const maxTime = 480;
    const pxPerMin = Math.max(1.2, (container.clientWidth - 50) / maxTime);

    let html = "";
    for (const rid of ["R1", "R2", "R3", "R4", "R5"]) {
        const bars = gantt[rid] || [];
        let barsHtml = "";

        for (const bar of bars) {
            const left = bar.start * pxPerMin;
            const width = Math.max(2, (bar.end - bar.start) * pxPerMin);
            const skuClass = bar.sku ? `sku-${bar.sku}` : "";
            const cancelClass = bar.cancelled ? "cancelled" : "";
            const label = bar.type === "down" ? "⚡" : (bar.sku || "");

            barsHtml += `<div class="gantt-bar type-${bar.type} ${skuClass} ${cancelClass}"
                style="left:${left}px; width:${width}px"
                title="${bar.type}: ${bar.sku || 'down'} [${bar.start}-${bar.end}]">${label}</div>`;
        }

        // Current time marker
        const markerLeft = currentTime * pxPerMin;

        html += `<div class="gantt-row">
            <span class="gantt-label">${rid}</span>
            <div class="gantt-track">
                ${barsHtml}
                <div class="gantt-time-marker" style="left:${markerLeft}px"></div>
            </div>
        </div>`;
    }

    container.innerHTML = html;
}

// --- Event Log ---
function updateEventLog(events) {
    const container = document.getElementById("event-log");
    if (!container || !events) return;

    let html = "";
    for (const ev of events.slice(-15).reverse()) {
        const timeStr = `[${String(Math.floor(ev.time / 60)).padStart(2, "0")}:${String(ev.time % 60).padStart(2, "0")}]`;
        let msg = "";

        switch (ev.type) {
            case "batch_complete":
                msg = `${ev.roaster}: ${ev.sku} batch completed`;
                break;
            case "setup_complete":
                msg = `${ev.roaster}: Setup complete (ready for ${ev.pending_sku || "?"})`;
                break;
            case "repair_complete":
                msg = `${ev.roaster}: Repair complete — back online`;
                break;
            case "disruption":
                msg = `${ev.roaster}: BREAKDOWN (${ev.duration} min, ${ev.source})`;
                break;
            case "stockout":
                msg = `${ev.line}: STOCKOUT (stock=${ev.stock})`;
                break;
            default:
                msg = JSON.stringify(ev);
        }

        html += `<div class="log-entry">
            <span class="log-time">${timeStr}</span>
            <span class="log-type type-${ev.type}">${ev.type.replace("_", " ")}</span>
            ${msg}
        </div>`;
    }

    container.innerHTML = html;
}

// --- AI Decision Log ---
function updateDecisionLog(decisions) {
    const container = document.getElementById("decision-log");
    if (!container || !decisions) return;

    let html = "";
    for (const d of decisions.slice(-25).reverse()) {
        const timeStr = `[${String(Math.floor(d.time / 60)).padStart(2, "0")}:${String(d.time % 60).padStart(2, "0")}]`;

        let actionClass = "act-schedule";
        if (d.action === "WAIT") actionClass = "act-wait";
        else if (d.action.startsWith("FAILED")) actionClass = "act-failed";

        html += `<div class="decision-entry">
            <span class="decision-time">${timeStr}</span>
            <span class="decision-roaster r-${d.roaster}">${d.roaster}</span>
            <span class="decision-action ${actionClass}">${d.action}</span>
            <span class="decision-reason">${d.reason}</span>
        </div>`;
    }

    container.innerHTML = html;
}

// --- Drag and Drop ---
function initDragDrop() {
    document.querySelectorAll(".sku-badge").forEach(badge => {
        badge.addEventListener("dragstart", (e) => {
            e.dataTransfer.setData("text/plain", badge.dataset.sku);
            badge.classList.add("dragging");
        });
        badge.addEventListener("dragend", () => {
            badge.classList.remove("dragging");
            document.querySelectorAll(".roaster-card").forEach(c => c.classList.remove("drag-over"));
        });
    });

    document.querySelectorAll(".roaster-card").forEach(card => {
        card.addEventListener("dragover", (e) => {
            e.preventDefault();
            card.classList.add("drag-over");
        });
        card.addEventListener("dragleave", () => {
            card.classList.remove("drag-over");
        });
        card.addEventListener("drop", async (e) => {
            e.preventDefault();
            card.classList.remove("drag-over");
            const sku = e.dataTransfer.getData("text/plain");
            const roasterId = card.dataset.roaster;
            if (sku && roasterId) {
                await scheduleBatch(roasterId, sku);
            }
        });
    });
}

async function scheduleBatch(roasterId, sku) {
    const res = await apiPost("/api/schedule", { roaster: roasterId, sku });
    if (res.error) {
        showToast(res.error, "error");
    } else {
        showToast(res.message, "success");
        await fetchState();
    }
}

// --- Disruptions ---
async function triggerDisruption(roasterId) {
    const res = await apiPost("/api/disrupt", { roaster: roasterId, duration: 20 });
    if (res.error) {
        showToast(res.error, "error");
    } else {
        showToast(`⚡ ${roasterId} breakdown!`, "error");
        await fetchState();
    }
}

// --- Consumption Rate ---
async function updateConsumptionRate(line, value) {
    const val = parseFloat(value);
    document.getElementById(`rate-val-${line}`).textContent = val.toFixed(1);
    await apiPost("/api/consumption_rate", { line, rate: val });
}

// --- Charts ---
function initCharts() {
    const defaultOpts = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
            legend: {
                labels: { color: "#9ca3af", font: { size: 10 } }
            }
        },
        scales: {
            x: {
                ticks: { color: "#6b7280", font: { size: 9 } },
                grid: { color: "rgba(255,255,255,0.04)" }
            },
            y: {
                ticks: { color: "#6b7280", font: { size: 9 } },
                grid: { color: "rgba(255,255,255,0.04)" }
            }
        }
    };

    // RC Stock chart
    charts.rcStock = new Chart(document.getElementById("chart-rc-stock"), {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Line 1",
                    data: [],
                    borderColor: "#6366f1",
                    backgroundColor: "rgba(99,102,241,0.1)",
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: "Line 2",
                    data: [],
                    borderColor: "#ec4899",
                    backgroundColor: "rgba(236,72,153,0.1)",
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                },
            ]
        },
        options: {
            ...defaultOpts,
            scales: {
                ...defaultOpts.scales,
                y: { ...defaultOpts.scales.y, min: -5 }
            }
        }
    });

    // Utilization chart
    charts.utilization = new Chart(document.getElementById("chart-utilization"), {
        type: "bar",
        data: {
            labels: ["R1", "R2", "R3", "R4", "R5"],
            datasets: [{
                label: "Utilization %",
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    "rgba(99,102,241,0.7)",
                    "rgba(168,85,247,0.7)",
                    "rgba(236,72,153,0.7)",
                    "rgba(245,158,11,0.7)",
                    "rgba(16,185,129,0.7)",
                ],
                borderRadius: 4,
            }]
        },
        options: {
            ...defaultOpts,
            scales: {
                ...defaultOpts.scales,
                y: { ...defaultOpts.scales.y, min: 0, max: 100 }
            }
        }
    });

    // Completed batches chart
    charts.batches = new Chart(document.getElementById("chart-batches"), {
        type: "bar",
        data: {
            labels: ["R1", "R2", "R3", "R4", "R5"],
            datasets: [{
                label: "Batches",
                data: [0, 0, 0, 0, 0],
                backgroundColor: "rgba(16,185,129,0.7)",
                borderRadius: 4,
            }]
        },
        options: defaultOpts,
    });
}

function updateCharts(state) {
    if (!charts.rcStock) return;

    // RC Stock
    const rcL1 = state.rc_history.L1;
    const rcL2 = state.rc_history.L2;
    if (rcL1 && rcL2) {
        charts.rcStock.data.labels = rcL1.map(p => p.t);
        charts.rcStock.data.datasets[0].data = rcL1.map(p => p.stock);
        charts.rcStock.data.datasets[1].data = rcL2.map(p => p.stock);
        charts.rcStock.update("none");
    }

    // Utilization
    const rids = ["R1", "R2", "R3", "R4", "R5"];
    charts.utilization.data.datasets[0].data = rids.map(
        rid => state.roasters[rid]?.utilization || 0
    );
    charts.utilization.update("none");

    // Batches
    charts.batches.data.datasets[0].data = rids.map(
        rid => state.roasters[rid]?.batches_completed || 0
    );
    charts.batches.update("none");
}

// --- Report ---
async function generateReport() {
    const report = await apiGet("/api/report");

    const body = document.getElementById("report-body");

    // KPI Cards
    let html = `<div class="report-kpis">
        <div class="kpi-card">
            <div class="kpi-value success">${report.total_psc_batches}</div>
            <div class="kpi-label">PSC Batches</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">${report.psc_L1} / ${report.psc_L2}</div>
            <div class="kpi-label">L1 / L2 Split</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value ${report.mto_tardiness > 0 ? 'danger' : 'success'}">
                ${report.mto_tardiness} min</div>
            <div class="kpi-label">MTO Tardiness</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value ${report.stockout_events > 0 ? 'danger' : 'success'}">
                ${report.stockout_events}</div>
            <div class="kpi-label">Stockout Events</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value ${report.stockout_duration > 0 ? 'warning' : 'success'}">
                ${report.stockout_duration} min</div>
            <div class="kpi-label">Stockout Duration</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">${report.avg_utilization}%</div>
            <div class="kpi-label">Avg Utilization</div>
        </div>
    </div>`;

    // MTO Summary
    html += `<h3 style="margin:16px 0 8px; color: var(--text-secondary); font-size: 0.85rem;">MTO Completion</h3>
    <div class="report-kpis">
        <div class="kpi-card">
            <div class="kpi-value">${report.mto_ndg_completed}/${report.mto_ndg_total}</div>
            <div class="kpi-label">NDG Complete</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">${report.mto_busta_completed}/${report.mto_busta_total}</div>
            <div class="kpi-label">Busta Complete</div>
        </div>
    </div>`;

    // Utilization bars
    html += `<h3 style="margin:16px 0 8px; color: var(--text-secondary); font-size: 0.85rem;">Roaster Utilization</h3>`;
    for (const [rid, data] of Object.entries(report.roaster_utilization)) {
        html += `<div class="util-bar-container">
            <div class="util-label">
                <span>${rid} — ${data.batches} batches</span>
                <span>${data.utilization}%</span>
            </div>
            <div class="util-bar-bg">
                <div class="util-bar-fill" style="width:${data.utilization}%"></div>
            </div>
        </div>`;
    }

    // Disruption log
    if (report.disruption_log && report.disruption_log.length > 0) {
        html += `<h3 style="margin:16px 0 8px; color: var(--text-secondary); font-size: 0.85rem;">Disruption Log</h3>`;
        html += `<table style="width:100%; font-size:0.75rem; color: var(--text-secondary); border-collapse: collapse;">
            <tr style="border-bottom:1px solid var(--border-color)">
                <th style="text-align:left; padding:4px;">Time</th>
                <th style="text-align:left; padding:4px;">Roaster</th>
                <th style="text-align:left; padding:4px;">Duration</th>
                <th style="text-align:left; padding:4px;">Type</th>
            </tr>`;
        for (const d of report.disruption_log) {
            html += `<tr style="border-bottom:1px solid rgba(255,255,255,0.03)">
                <td style="padding:4px">${d.time} min</td>
                <td style="padding:4px">${d.roaster}</td>
                <td style="padding:4px">${d.duration} min</td>
                <td style="padding:4px">${d.type}</td>
            </tr>`;
        }
        html += `</table>`;
    }

    body.innerHTML = html;
    document.getElementById("report-modal").style.display = "flex";
}

function closeReport() {
    document.getElementById("report-modal").style.display = "none";
}

function downloadCSV() {
    window.open("/api/report/csv", "_blank");
}

// --- Toast ---
function showToast(message, type = "") {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.className = `toast show ${type}`;
    setTimeout(() => {
        toast.classList.remove("show");
    }, 3000);
}
