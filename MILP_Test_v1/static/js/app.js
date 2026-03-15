/* ═══════════════════════════════════════════════
   app.js — Frontend logic for MILP Scheduler GUI
   ═══════════════════════════════════════════════ */

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

function setStatus(text, loading = false) {
    document.getElementById('status-text').textContent = text;
    const spinner = document.getElementById('status-spinner');
    spinner.classList.toggle('hidden', !loading);
}

function showSection(id) {
    const el = document.getElementById(id);
    if (el) {
        el.classList.remove('hidden');
        el.classList.add('fade-in');
    }
}

function enableBtn(id, enabled = true) {
    const btn = document.getElementById(id);
    if (btn) btn.disabled = !enabled;
}

async function apiCall(url, method = 'POST', body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body && method !== 'GET') {
        opts.body = JSON.stringify(body);
    }
    const resp = await fetch(url, opts);
    return resp.json();
}


// ──────────────────────────────────────────────
// Load Data
// ──────────────────────────────────────────────

async function loadData() {
    setStatus('Loading CSV data...', true);
    enableBtn('btn-load', false);

    try {
        const res = await apiCall('/api/load');

        if (!res.success) {
            setStatus('❌ Load failed: ' + res.error);
            enableBtn('btn-load', true);
            return;
        }

        const s = res.summary;

        // Show data summary
        const summaryHtml = `
            <div class="summary-item">
                <h4>Shift</h4>
                <p>${s.shift_length} minutes</p>
            </div>
            <div class="summary-item">
                <h4>Roasters</h4>
                <p>${s.roaster_ids.join(', ')}</p>
            </div>
            <div class="summary-item">
                <h4>SKUs</h4>
                <p>${s.skus.join(', ')}</p>
            </div>
            <div class="summary-item">
                <h4>MTO Jobs</h4>
                <p>${s.jobs.map(j => `${j.job_id}: ${j.sku}×${j.batches}`).join(', ')}</p>
            </div>
            <div class="summary-item">
                <h4>Initial RC</h4>
                <p>L1=${s.initial_rc.L1}, L2=${s.initial_rc.L2} (max ${s.max_rc})</p>
            </div>
            <div class="summary-item">
                <h4>Consumption Events</h4>
                <p>L1: ${s.consume_events_l1}, L2: ${s.consume_events_l2}</p>
            </div>
            <div class="summary-item">
                <h4>Planned Downtime</h4>
                <p>${s.planned_downtime.length > 0
                    ? s.planned_downtime.map(d => `${d.roaster_id} [${d.start}-${d.end}]`).join(', ')
                    : 'None'}</p>
            </div>
        `;

        document.getElementById('data-summary-content').innerHTML = summaryHtml;
        showSection('data-summary');

        // Populate roaster dropdown
        const sel = document.getElementById('disrupt-roaster');
        sel.innerHTML = '';
        s.roaster_ids.forEach(rid => {
            const opt = document.createElement('option');
            opt.value = rid;
            opt.textContent = rid;
            sel.appendChild(opt);
        });

        enableBtn('btn-load', true);
        enableBtn('btn-solve', true);
        setStatus('✅ Data loaded — ' + s.roasters + ' roasters, ' + s.jobs.length + ' MTO jobs');

    } catch (e) {
        setStatus('❌ Error: ' + e.message);
        enableBtn('btn-load', true);
    }
}


// ──────────────────────────────────────────────
// Solve MILP
// ──────────────────────────────────────────────

async function solve() {
    setStatus('🔧 Solving MILP — this may take a moment...', true);
    enableBtn('btn-solve', false);

    try {
        const res = await apiCall('/api/solve');

        if (!res.success) {
            setStatus('❌ Solve failed: ' + res.error);
            enableBtn('btn-solve', true);
            return;
        }

        if (res.summary && res.summary.status === 'infeasible') {
            document.getElementById('log-content').textContent = "The MILP solver could not find a feasible schedule. This could be due to tight constraints, or limits exceeded (if using Community Edition).";
            document.getElementById('log-modal').classList.remove('hidden');
            setStatus('❌ Model Infeasible — 0 batches scheduled');
            updateUI(res);
        } else {
            updateUI(res);
            setStatus('✅ Solved — ' + res.batch_count + ' batches scheduled');
        }

        enableBtn('btn-solve', true);
        enableBtn('btn-export', true);

    } catch (e) {
        setStatus('❌ Error: ' + e.message);
        enableBtn('btn-solve', true);
    }
}


// ──────────────────────────────────────────────
// Add Disruption
// ──────────────────────────────────────────────

async function addDisruption() {
    const time_min = parseInt(document.getElementById('disrupt-time').value);
    const roaster_id = document.getElementById('disrupt-roaster').value;
    const duration_min = parseInt(document.getElementById('disrupt-duration').value);

    if (!roaster_id) {
        setStatus('⚠️ Select a roaster first');
        return;
    }

    setStatus('Adding disruption...', true);

    try {
        const res = await apiCall('/api/disrupt', 'POST', {
            time_min,
            roaster_id,
            duration_min,
        });

        if (!res.success) {
            setStatus('❌ ' + res.error);
            return;
        }

        updateDisruptionList(res.disruptions);
        setStatus(`✅ Disruption added: ${roaster_id} at t=${time_min} for ${duration_min} min`);

    } catch (e) {
        setStatus('❌ Error: ' + e.message);
    }
}


// ──────────────────────────────────────────────
// Re-Solve
// ──────────────────────────────────────────────

async function reResolve() {
    setStatus('🔄 Re-solving with disruptions...', true);
    enableBtn('btn-solve', false);

    try {
        const res = await apiCall('/api/resolve');

        if (!res.success) {
            setStatus('❌ Re-solve failed: ' + res.error);
            enableBtn('btn-solve', true);
            return;
        }

        if (res.summary && res.summary.status === 'infeasible') {
            document.getElementById('log-content').textContent = "The MILP solver could not find a feasible schedule in reactive mode. Trying relaxing constraints or removing disruptions.";
            document.getElementById('log-modal').classList.remove('hidden');
            setStatus('❌ Model Infeasible — 0 batches (reactive mode)');
            updateUI(res);
        } else {
            updateUI(res);
            setStatus('✅ Re-solved — ' + res.batch_count + ' batches (reactive mode)');
        }

        enableBtn('btn-solve', true);
        enableBtn('btn-export', true);

    } catch (e) {
        setStatus('❌ Error: ' + e.message);
        enableBtn('btn-solve', true);
    }
}


// ──────────────────────────────────────────────
// Export CSV
// ──────────────────────────────────────────────

function exportCSV() {
    window.location.href = '/api/export';
    setStatus('📥 Schedule exported to CSV');
}


// ──────────────────────────────────────────────
// UI Update
// ──────────────────────────────────────────────

function updateUI(res) {
    const s = res.summary;

    // Metrics
    showSection('metrics-section');
    document.getElementById('mv-status').textContent = s.status;
    document.getElementById('mv-objective').textContent = s.objective;
    document.getElementById('mv-profit').textContent = s.net_profit;
    document.getElementById('mv-gap').textContent = s.mip_gap;
    document.getElementById('mv-time').textContent = s.solve_time;
    document.getElementById('mv-batches').textContent = `${s.psc_batches} PSC + ${s.mto_batches} MTO`;

    // Color status card based on result
    const statusCard = document.getElementById('mc-status');
    statusCard.style.borderColor = s.status.includes('optimal') ? '#22c55e' : '#f59e0b';

    // Gantt chart
    showSection('charts-section');
    if (res.gantt && res.gantt.data) {
        Plotly.newPlot('gantt-chart', res.gantt.data, res.gantt.layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // RC chart
    if (res.rc_chart && res.rc_chart.data) {
        Plotly.newPlot('rc-chart', res.rc_chart.data, res.rc_chart.layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // Bottom section
    showSection('bottom-section');

    // KPIs
    document.getElementById('kpi-revenue').textContent = s.total_revenue;
    document.getElementById('kpi-psc').textContent = s.psc_batches;
    document.getElementById('kpi-mto').textContent = s.mto_batches;
    document.getElementById('kpi-tard').textContent = s.tardiness_min + ' min';
    document.getElementById('kpi-tard-cost').textContent = s.tardiness_cost;
    document.getElementById('kpi-stockout').textContent = s.stockout_min + ' min (' + s.stockout_cost + ')';
    document.getElementById('kpi-idle').textContent = s.idle_min + ' min (' + s.idle_cost + ')';
    document.getElementById('kpi-overflow').textContent = s.overflow_idle_min + ' min (' + s.overflow_idle_cost + ')';

    // MTO details
    if (res.job_tardiness) {
        let html = '<table class="detail-table"><tbody>';
        for (const [jobId, tard] of Object.entries(res.job_tardiness)) {
            const tardClass = tard > 0 ? 'style="color: var(--accent-red)"' : '';
            html += `<tr><td>${jobId}</td><td ${tardClass}>${tard.toFixed(0)} min late</td></tr>`;
        }
        html += '</tbody></table>';
        document.getElementById('mto-details').innerHTML = html;
    }

    // Utilization
    if (res.utilization) {
        let html = '';
        for (const [rid, util] of Object.entries(res.utilization)) {
            const pct = (util * 100).toFixed(1);
            const color = util > 0.8 ? 'var(--accent-green)' :
                          util > 0.5 ? 'var(--accent-yellow)' : 'var(--accent-red)';
            html += `
                <div class="util-bar-container">
                    <span class="util-label">${rid}</span>
                    <div class="util-bar-track">
                        <div class="util-bar-fill" style="width:${pct}%; background:${color}"></div>
                    </div>
                    <span class="util-value">${pct}%</span>
                </div>
            `;
        }
        document.getElementById('util-details').innerHTML = html;
    }

    // Schedule table
    if (res.schedule) {
        let html = `
            <table class="schedule-table">
                <thead>
                    <tr>
                        <th>Batch</th>
                        <th>SKU</th>
                        <th>Roaster</th>
                        <th>Start</th>
                        <th>End</th>
                        <th>Out</th>
                        <th>Setup</th>
                        <th>Revenue</th>
                    </tr>
                </thead>
                <tbody>
        `;
        for (const b of res.schedule) {
            html += `
                <tr>
                    <td>${b.batch_id}</td>
                    <td><span class="sku-badge sku-${b.sku}">${b.sku}</span></td>
                    <td>${b.roaster}</td>
                    <td>${b.start}</td>
                    <td>${b.end}</td>
                    <td>${b.output_line}</td>
                    <td>${b.setup > 0 ? b.setup + 'm' : '—'}</td>
                    <td>$${b.revenue.toLocaleString()}</td>
                </tr>
            `;
        }
        html += '</tbody></table>';
        document.getElementById('schedule-table-container').innerHTML = html;
    }

    // Disruptions list
    if (res.disruptions) {
        updateDisruptionList(res.disruptions);
    }
}


// ──────────────────────────────────────────────
// Disruption List
// ──────────────────────────────────────────────

function updateDisruptionList(disruptions) {
    const container = document.getElementById('disruption-list');
    if (!disruptions || disruptions.length === 0) {
        container.innerHTML = '<p class="empty-state">No disruptions added</p>';
        return;
    }

    let html = '';
    disruptions.forEach((d, i) => {
        html += `
            <div class="disruption-item">
                <span class="d-info">#${i + 1} — ${d.roaster_id}</span>
                <span class="d-detail">t=${d.time_min}, ${d.duration_min} min</span>
            </div>
        `;
    });
    container.innerHTML = html;
}


// ──────────────────────────────────────────────
// Tabs
// ──────────────────────────────────────────────

function switchTab(tabId) {
    // Deactivate all tabs
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    // Activate selected
    document.getElementById(tabId).classList.add('active');

    // Find the button that matches
    document.querySelectorAll('.tab').forEach(t => {
        if (t.getAttribute('onclick').includes(tabId)) {
            t.classList.add('active');
        }
    });
}


// ──────────────────────────────────────────────
// Modal
// ──────────────────────────────────────────────

function showLog(logText) {
    document.getElementById('log-content').textContent = logText;
    document.getElementById('log-modal').classList.remove('hidden');
}

function closeModal() {
    document.getElementById('log-modal').classList.add('hidden');
}

// Close modal on outside click
document.addEventListener('click', (e) => {
    const modal = document.getElementById('log-modal');
    if (e.target === modal) {
        closeModal();
    }
});


// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    setStatus('Ready — Click "Load Data" to begin');
});
