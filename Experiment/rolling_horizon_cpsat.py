"""
Rolling-Horizon CP-SAT Experiment  (4 x 120-min windows, 5-min budget each)
===========================================================================
Splits the 480-min shift into 4 windows of 120 minutes.
Each window is solved independently by CP-SAT with a 5-minute budget.
State (RC stock, GC stock, last roaster SKU, remaining MTO batches) is
carried forward between windows.

Compare to:
  Full CP-SAT oracle  : $443,400  (pre-knows all UPS, single 8h solve, seed 69)
  RL-HH (100-seed mean): $359,096  (<1 ms decisions, online)
  Dispatch (100-seed)  : $326,554  (<1 ms decisions, online)

Run:
  python Experiment/rolling_horizon_cpsat.py

Outputs -> Experiment/results/rolling/
  window_results.json   per-window breakdown
  summary.json          totals + comparison table
  report.html           self-contained HTML report with chart
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CPSAT_Pure.data import load as cpsat_load_data
from CPSAT_Pure.model import build as cpsat_build_model
from CPSAT_Pure.solver import solve as cpsat_solve
from env.ups_generator import generate_ups_events

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED            = 69
UPS_LAMBDA      = 5.0
UPS_MU          = 20.0
SHIFT_LENGTH    = 480
WINDOW_SIZE     = 120       # 2 hours per window
N_WINDOWS       = 4         # 4 windows = full 480-min shift
BUDGET_SEC      = 5 * 60    # 5 minutes per window
NUM_WORKERS     = 8
INPUT_DIR       = str(_ROOT / "Input_data")
OUT_DIR         = Path(__file__).parent / "results" / "rolling"

REFERENCE_FULL  = 443_400
RL_HH_MEAN      = 359_096
DISPATCH_MEAN   = 326_554
ROASTERS        = ["R1", "R2", "R3", "R4", "R5"]


# ---------------------------------------------------------------------------
# Data dict builder for a single window
# ---------------------------------------------------------------------------

def _build_window_data(base: dict, window_idx: int, state: dict,
                       ups_events: list, all_ups: list) -> dict:
    """Create a 120-min sub-problem data dict for window `window_idx`.

    `state` carries forward from previous window:
        rc_init        {line: int}
        gc_init        {(line, sku): int}
        roaster_last_sku  {roaster: str}
        mto_remaining  {job_id: int}   batches still to be produced
    """
    w_start = window_idx * WINDOW_SIZE
    w_end   = w_start + WINDOW_SIZE

    d = copy.deepcopy(base)

    # ---- horizon --------------------------------------------------------
    d["shift_length"] = WINDOW_SIZE
    d["time_limit"]   = BUDGET_SEC
    d["MS_by_sku"]    = {sku: WINDOW_SIZE - dur for sku, dur in d["roast_time_by_sku"].items()}

    # ---- initial inventory (carry-forward) ------------------------------
    d["rc_init"] = dict(state["rc_init"])
    d["gc_init"] = {k: v for k, v in state["gc_init"].items()}

    # ---- roaster last SKU (carry-forward) -------------------------------
    d["roaster_initial_sku"] = dict(state["roaster_last_sku"])

    # ---- consumption events: filter to this window, shift to [0, 120) --
    d["consumption_events"] = {
        line: [t - w_start for t in events if w_start <= t < w_end]
        for line, events in base["consumption_events"].items()
    }

    # ---- UPS/downtime: filter to this window, shift to [0, 120) --------
    window_ups = [ev for ev in all_ups
                  if ev is not None and w_start <= getattr(ev, "t", -1) < w_end]
    d["downtime_slots"] = {r: set() for r in d["roasters"]}
    # copy planned downtime (already in base, shift)
    for roaster, slots in base["downtime_slots"].items():
        shifted = {s - w_start for s in slots if w_start <= s < w_end}
        d["downtime_slots"][roaster] = shifted
    # inject UPS events
    for ev in window_ups:
        roaster_id = getattr(ev, "roaster_id", None)
        t          = getattr(ev, "t", None)
        duration   = getattr(ev, "duration", None)
        if roaster_id not in d["downtime_slots"]:
            continue
        eff_dur = max(0, int(duration) - 1)
        t_rel   = int(t) - w_start
        for s in range(t_rel, min(WINDOW_SIZE, t_rel + eff_dur)):
            d["downtime_slots"][roaster_id].add(s)

    # ---- MTO jobs: keep only those with remaining batches ---------------
    mto_remaining = state["mto_remaining"]
    active_jobs   = [jid for jid, rem in mto_remaining.items() if rem > 0]

    d["jobs"]        = active_jobs
    d["job_sku"]     = {jid: base["job_sku"][jid] for jid in active_jobs}
    d["job_batches"] = {jid: mto_remaining[jid] for jid in active_jobs}

    # Adjust due dates relative to this window; clamp to 0 minimum
    d["job_due"] = {
        jid: max(0, base["job_due"][jid] - w_start)
        for jid in active_jobs
    }
    d["job_release"] = {
        jid: max(0, base["job_release"][jid] - w_start)
        for jid in active_jobs
    }

    # ---- rebuild batch lists from scratch -------------------------------
    psc_pool_per_roaster = math.floor(WINDOW_SIZE / d["roast_time_by_sku"]["PSC"])
    d["psc_pool_per_roaster"] = psc_pool_per_roaster

    mto_batches = [
        (jid, bidx)
        for jid in active_jobs
        for bidx in range(d["job_batches"][jid])
    ]
    psc_pool = [
        (roaster_id, slot_idx)
        for roaster_id in d["roasters"]
        for slot_idx in range(psc_pool_per_roaster)
    ]
    all_batches = mto_batches + psc_pool

    d["mto_batches"]  = mto_batches
    d["psc_pool"]     = psc_pool
    d["all_batches"]  = all_batches

    d["batch_sku"] = {bid: d["job_sku"][bid[0]] for bid in mto_batches}
    d["batch_sku"].update({bid: "PSC" for bid in psc_pool})

    d["batch_is_mto"] = {bid: True for bid in mto_batches}
    d["batch_is_mto"].update({bid: False for bid in psc_pool})

    d["batch_eligible_roasters"] = {
        bid: list(d["sku_eligible_roasters"][d["batch_sku"][bid]])
        for bid in mto_batches
    }
    d["batch_eligible_roasters"].update({
        bid: list(d["sku_eligible_roasters"]["PSC"])
        for bid in psc_pool
    })

    d["sched_eligible_roasters"] = {
        bid: [bid[0]] if bid in set(psc_pool) else list(d["batch_eligible_roasters"][bid])
        for bid in all_batches
    }

    return d


# ---------------------------------------------------------------------------
# State extraction from a window result
# ---------------------------------------------------------------------------

def _extract_state(result: dict, prev_state: dict, base: dict) -> dict:
    """Build the carry-forward state from a solved window result."""
    schedule = result.get("schedule", [])

    # RC final
    rc_init = dict(result.get("rc_final", prev_state["rc_init"]))

    # GC final: result uses "L1_PSC" keys → convert to ("L1","PSC")
    gc_final_raw = result.get("gc_final", {})
    gc_init = {}
    for pair in base["feasible_gc_pairs"]:
        key_str = f"{pair[0]}_{pair[1]}"
        gc_init[pair] = int(gc_final_raw.get(key_str, prev_state["gc_init"].get(pair, 0)))
    # Clamp GC to capacity (solver may slightly exceed due to rounding)
    for pair in gc_init:
        gc_init[pair] = max(0, min(gc_init[pair], int(base["gc_capacity"][pair])))

    # Last SKU per roaster: scan schedule for latest end time per roaster
    roaster_last_sku = dict(prev_state["roaster_last_sku"])
    by_roaster: dict[str, list] = {r: [] for r in base["roasters"]}
    for entry in schedule:
        by_roaster[entry["roaster"]].append(entry)
    for roaster, entries in by_roaster.items():
        if entries:
            last = max(entries, key=lambda e: e["end"])
            roaster_last_sku[roaster] = last["sku"]

    # MTO remaining: subtract completed batches per job
    mto_remaining = dict(prev_state["mto_remaining"])
    for entry in schedule:
        if entry.get("is_mto") and entry.get("job_id"):
            jid = entry["job_id"]
            if jid in mto_remaining:
                mto_remaining[jid] = max(0, mto_remaining[jid] - 1)

    return {
        "rc_init":          rc_init,
        "gc_init":          gc_init,
        "roaster_last_sku": roaster_last_sku,
        "mto_remaining":    mto_remaining,
    }


# ---------------------------------------------------------------------------
# Rolling horizon solver
# ---------------------------------------------------------------------------

def run_rolling_horizon(ups_events: list) -> tuple[list[dict], dict]:
    base = cpsat_load_data(input_dir=INPUT_DIR, overrides={"time_limit": BUDGET_SEC})

    # Initial state from CSV
    state: dict = {
        "rc_init":          dict(base["rc_init"]),
        "gc_init":          dict(base["gc_init"]),
        "roaster_last_sku": dict(base["roaster_initial_sku"]),
        "mto_remaining":    {jid: base["job_batches"][jid] for jid in base["jobs"]},
    }

    window_results = []
    total_profit    = 0.0
    total_solve_sec = 0.0

    for w in range(N_WINDOWS):
        w_start = w * WINDOW_SIZE
        w_end   = w_start + WINDOW_SIZE
        print(f"\n  Window {w+1}/4  [{w_start}-{w_end} min] ...", end=" ", flush=True)

        # Check: any MTO jobs with remaining batches and past-due in this window?
        # If job due < w_start, already overdue — apply skip penalty immediately
        skip_cost_carried = 0.0
        for jid in list(state["mto_remaining"].keys()):
            abs_due = base["job_due"][jid]
            if abs_due < w_start and state["mto_remaining"][jid] > 0:
                skipped = state["mto_remaining"][jid]
                skip_cost_carried += skipped * float(base["cost_skip_mto"])
                print(f"\n    [!] Job {jid}: {skipped} batches overdue, applying skip penalty ${skip_cost_carried:,.0f}")
                state["mto_remaining"][jid] = 0

        t0 = time.perf_counter()
        try:
            d = _build_window_data(base, w, state, [], ups_events)
            model, cp_vars = cpsat_build_model(d)
            result = cpsat_solve(d, model, cp_vars, num_workers=NUM_WORKERS)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"FAILED: {exc}")
            window_results.append({
                "window": w + 1, "w_start": w_start, "w_end": w_end,
                "status": "FAILED", "net_profit": None,
                "solve_sec": round(elapsed, 2), "skip_cost_carried": skip_cost_carried,
            })
            continue

        elapsed = time.perf_counter() - t0
        if result is None:
            print(f"UNKNOWN (no feasible solution in {elapsed:.1f}s)")
            window_results.append({
                "window": w + 1, "w_start": w_start, "w_end": w_end,
                "status": "UNKNOWN", "net_profit": None,
                "solve_sec": round(elapsed, 2), "skip_cost_carried": skip_cost_carried,
            })
            continue

        window_profit = float(result["net_profit"]) - skip_cost_carried
        total_profit    += window_profit
        total_solve_sec += elapsed

        psc = result.get("psc_count", 0)
        ndg = result.get("ndg_count", 0)
        bst = result.get("busta_count", 0)
        print(f"profit=${window_profit:>10,.0f}  "
              f"P/N/B={psc}/{ndg}/{bst}  "
              f"tard=${result.get('tard_cost',0):,.0f}  "
              f"wall={elapsed:.1f}s  incumbents={result.get('num_incumbents',0)}")

        window_results.append({
            "window":          w + 1,
            "w_start":         w_start,
            "w_end":           w_end,
            "status":          result.get("status", "?"),
            "net_profit":      round(window_profit, 2),
            "revenue":         round(result.get("total_revenue", 0), 2),
            "tard_cost":       round(result.get("tard_cost", 0), 2),
            "idle_cost":       round(result.get("idle_cost", 0), 2),
            "setup_cost":      round(result.get("setup_cost", 0), 2),
            "psc_count":       psc,
            "ndg_count":       ndg,
            "busta_count":     bst,
            "incumbents":      result.get("num_incumbents", 0),
            "gap_pct":         result.get("gap_pct"),
            "solve_sec":       round(elapsed, 2),
            "skip_cost_carried": round(skip_cost_carried, 2),
        })

        state = _extract_state(result, state, base)

    # Any remaining MTO batches at end of shift = skip penalty
    final_skip_cost = 0.0
    for jid, rem in state["mto_remaining"].items():
        if rem > 0:
            penalty = rem * float(base["cost_skip_mto"])
            final_skip_cost += penalty
            print(f"  [end-of-shift] Job {jid}: {rem} batches unfinished, penalty=${penalty:,.0f}")
    total_profit -= final_skip_cost

    summary = {
        "seed":               SEED,
        "ups_lambda":         UPS_LAMBDA,
        "ups_mu":             UPS_MU,
        "n_windows":          N_WINDOWS,
        "window_size_min":    WINDOW_SIZE,
        "budget_per_window_sec": BUDGET_SEC,
        "total_solve_sec":    round(total_solve_sec, 2),
        "total_frozen_min":   round(total_solve_sec / 60, 2),
        "pct_shift_frozen":   round(total_solve_sec / 60 / SHIFT_LENGTH * 100, 2),
        "total_profit":       round(total_profit, 2),
        "final_skip_cost":    round(final_skip_cost, 2),
        "reference_full_cpsat": REFERENCE_FULL,
        "reference_rlhh":     RL_HH_MEAN,
        "reference_dispatch": DISPATCH_MEAN,
        "gap_vs_full_pct":    round((REFERENCE_FULL - total_profit) / REFERENCE_FULL * 100, 2)
                              if total_profit else None,
    }
    return window_results, summary


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def make_chart(window_results: list[dict], summary: dict, out_path: Path) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [chart] matplotlib not available — skipping")
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Rolling-Horizon CP-SAT  |  {N_WINDOWS} x {WINDOW_SIZE}-min windows  |  "
        f"{BUDGET_SEC//60}-min budget each  |  Seed {SEED}",
        fontsize=12, fontweight="bold"
    )

    # Left: per-window profit bar
    valid = [r for r in window_results if r["net_profit"] is not None]
    labels = [f"W{r['window']}\n[{r['w_start']}-{r['w_end']}]" for r in valid]
    profits = [r["net_profit"] for r in valid]
    x = np.arange(len(labels))
    bars = ax1.bar(x, profits, color="#5b9bd5", width=0.5, zorder=2)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Net Profit ($)")
    ax1.set_title("Per-Window Profit")
    ax1.grid(axis="y", alpha=0.3, zorder=1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    for bar, p in zip(bars, profits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f"${p:,.0f}", ha="center", va="bottom", fontsize=9)

    # Right: total comparison bar
    methods   = ["Full\nCP-SAT\n(oracle)", "Rolling\nCP-SAT\n(4x5min)", "RL-HH\n(proposed)", "Dispatch\nheuristic"]
    values    = [REFERENCE_FULL, summary["total_profit"], RL_HH_MEAN, DISPATCH_MEAN]
    colors    = ["#ed7d31", "#e05c2a", "#70ad47", "#ffc000"]
    ax2.bar(methods, values, color=colors, width=0.5, zorder=2)
    ax2.set_ylabel("Net Profit ($)")
    ax2.set_title("Total Shift Profit Comparison")
    ax2.grid(axis="y", alpha=0.3, zorder=1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    for i, (m, v) in enumerate(zip(methods, values)):
        ax2.text(i, v + 1000, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [chart] saved -> {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def make_html(window_results: list[dict], summary: dict,
              chart_path: str, out_path: Path) -> None:
    import base64
    chart_b64 = ""
    if chart_path and Path(chart_path).exists():
        with open(chart_path, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode()

    def money(v):
        return f"${v:,.0f}" if v is not None else "—"

    window_rows = ""
    for r in window_results:
        p = money(r.get("net_profit"))
        window_rows += (
            f"<tr>"
            f"<td>W{r['window']} [{r['w_start']}–{r['w_end']} min]</td>"
            f"<td>{r.get('status','?')}</td>"
            f"<td>{p}</td>"
            f"<td>{money(r.get('revenue'))}</td>"
            f"<td>{money(r.get('tard_cost'))}</td>"
            f"<td>{money(r.get('idle_cost'))}</td>"
            f"<td>{r.get('psc_count','—')}/{r.get('ndg_count','—')}/{r.get('busta_count','—')}</td>"
            f"<td>{r.get('incumbents','—')}</td>"
            f"<td>{r.get('gap_pct','—')}</td>"
            f"<td>{r.get('solve_sec','—')}s</td>"
            f"</tr>"
        )

    img_tag = (
        f'<img src="data:image/png;base64,{chart_b64}" '
        f'style="max-width:100%;border:1px solid #ddd;border-radius:6px">'
        if chart_b64 else "<p><em>Chart not generated.</em></p>"
    )

    gap = f"{summary.get('gap_vs_full_pct','?')}%" if summary.get('gap_vs_full_pct') else "—"

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head><meta charset='utf-8'>
<title>Rolling Horizon CP-SAT Experiment</title>
<style>
  body{{font-family:-apple-system,Segoe UI,sans-serif;margin:32px;color:#222;max-width:1200px}}
  h1{{margin-bottom:4px}} h2{{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:28px}}
  table{{border-collapse:collapse;margin:10px 0;font-size:.91em;width:100%}}
  th,td{{padding:6px 10px;border:1px solid #ccc;text-align:right}}
  th{{background:#efefef;text-align:center}}
  td:first-child{{text-align:left}}
  .note{{background:#fff9e6;border-left:4px solid #ffc000;padding:10px 16px;border-radius:4px;margin:12px 0;font-size:.91em}}
  .bad{{color:#c0392b;font-weight:bold}} .good{{color:#27ae60;font-weight:bold}}
  .highlight{{background:#eef4ff}}
</style>
</head>
<body>
<h1>Rolling-Horizon CP-SAT Experiment</h1>
<p>
  Seed {SEED} &nbsp;|&nbsp; UPS &lambda;={UPS_LAMBDA} &mu;={UPS_MU} &nbsp;|&nbsp;
  {N_WINDOWS} windows &times; {WINDOW_SIZE} min &nbsp;|&nbsp;
  {BUDGET_SEC//60}-min CP-SAT budget per window
</p>

<div class='note'>
  <strong>Design:</strong> The 480-min shift is split into {N_WINDOWS} windows of {WINDOW_SIZE} min each.
  Each window is solved independently by CP-SAT with a {BUDGET_SEC//60}-minute budget.
  RC stock, GC stock, roaster setup state and remaining MTO batches are carried forward between windows.
  UPS events are injected into each window as planned downtime (same oracle assumption as full CP-SAT).
</div>

<h2>Window Results</h2>
<table>
<thead><tr>
  <th>Window</th><th>Status</th><th>Net Profit</th>
  <th>Revenue</th><th>Tardiness</th><th>Idle Cost</th>
  <th>PSC/NDG/BUSTA</th><th>Incumbents</th><th>MIP Gap</th><th>Solve time</th>
</tr></thead>
<tbody>{window_rows}</tbody>
</table>

<h2>Total Shift Comparison</h2>
<table style='width:auto'>
<thead><tr><th>Method</th><th>Net Profit</th><th>Gap vs Oracle</th><th>Decision latency</th><th>UPS handling</th></tr></thead>
<tbody>
<tr><td>Full CP-SAT (oracle, seed 69)</td><td>{money(REFERENCE_FULL)}</td><td>0% (reference)</td><td>~8 h offline solve</td><td>Pre-known (oracle)</td></tr>
<tr class='highlight'><td><strong>Rolling CP-SAT ({N_WINDOWS}&times;{BUDGET_SEC//60}min)</strong></td>
    <td><strong class='bad'>{money(summary["total_profit"])}</strong></td>
    <td><strong class='bad'>{gap}</strong></td>
    <td class='bad'>{summary["total_frozen_min"]:.1f} min frozen ({summary["pct_shift_frozen"]:.1f}% of shift)</td>
    <td>Re-solve per window</td></tr>
<tr><td>RL-HH — proposed (100-seed mean)</td><td class='good'>{money(RL_HH_MEAN)}</td>
    <td>{((REFERENCE_FULL-RL_HH_MEAN)/REFERENCE_FULL*100):.1f}%</td>
    <td class='good'>&lt;1 ms</td><td class='good'>Native (online)</td></tr>
<tr><td>Dispatching heuristic (100-seed mean)</td><td>{money(DISPATCH_MEAN)}</td>
    <td>{((REFERENCE_FULL-DISPATCH_MEAN)/REFERENCE_FULL*100):.1f}%</td>
    <td class='good'>&lt;1 ms</td><td class='good'>Native (online)</td></tr>
</tbody>
</table>

<h2>Chart</h2>
{img_tag}

<h2>Key Takeaways</h2>
<ol>
  <li><strong>Short windows hurt solution quality.</strong>
      A {WINDOW_SIZE}-min horizon cannot see the full MTO due dates or RC dynamics,
      forcing the solver to make locally-greedy decisions that are globally suboptimal.</li>
  <li><strong>Solver latency is non-negligible production time.</strong>
      {N_WINDOWS} windows &times; {BUDGET_SEC//60} min = {summary["total_frozen_min"]:.0f} min frozen
      ({summary["pct_shift_frozen"]:.1f}% of the shift). Every UPS event mid-window
      would require an additional re-solve (+{BUDGET_SEC//60} min each, up to +25 min for &lambda;=5).</li>
  <li><strong>State carryforward introduces coupling loss.</strong>
      RC/GC levels propagate but the solver cannot plan restock operations that span window boundaries.</li>
  <li><strong>RL-HH still competes on profit</strong> with &lt;1 ms per decision and no planning freeze.</li>
</ol>

<p style='color:#888;font-size:.85em;margin-top:32px'>
  Generated by <code>Experiment/rolling_horizon_cpsat.py</code>
</p>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  [html] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Rolling-Horizon CP-SAT  |  {N_WINDOWS} x {WINDOW_SIZE} min  |  "
          f"{BUDGET_SEC//60}-min budget/window")
    print(f"Seed {SEED}  |  UPS lam={UPS_LAMBDA} mu={UPS_MU}")
    print("=" * 70)

    ups_events = list(generate_ups_events(
        UPS_LAMBDA, UPS_MU, seed=SEED,
        shift_length=SHIFT_LENGTH, roasters=ROASTERS,
    ))
    print(f"UPS events generated: {len(ups_events)}")
    for ev in ups_events:
        print(f"  t={ev.t:>4}  roaster={ev.roaster_id}  dur={ev.duration:.1f} min")

    window_results, summary = run_rolling_horizon(ups_events)

    # Save JSONs
    (OUT_DIR / "window_results.json").write_text(
        json.dumps(window_results, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  [json] saved -> {OUT_DIR}")

    chart_path = make_chart(window_results, summary, OUT_DIR / "rolling_chart.png")
    make_html(window_results, summary, chart_path, OUT_DIR / "report.html")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for r in window_results:
        p = f"${r['net_profit']:,.0f}" if r.get("net_profit") is not None else "FAILED/UNKNOWN"
        print(f"  W{r['window']} [{r['w_start']:>3}-{r['w_end']:>3}]:  {p:>14}  "
              f"incumbents={r.get('incumbents','?'):>3}  "
              f"wall={r.get('solve_sec','?'):>6}s")
    print("-" * 70)
    print(f"  Rolling total:  ${summary['total_profit']:>12,.0f}  "
          f"(gap vs oracle: {summary.get('gap_vs_full_pct','?')}%)")
    print(f"  Frozen time:    {summary['total_frozen_min']:.1f} min "
          f"({summary['pct_shift_frozen']:.1f}% of shift)")
    print(f"  Full CP-SAT:    ${REFERENCE_FULL:>12,}  (oracle, seed 69)")
    print(f"  RL-HH:          ${RL_HH_MEAN:>12,}  (100-seed mean, <1ms)")
    print(f"  Dispatching:    ${DISPATCH_MEAN:>12,}  (100-seed mean, <1ms)")
    print("=" * 70)
    print(f"\nReport: {OUT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
