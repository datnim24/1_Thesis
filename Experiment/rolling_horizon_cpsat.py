"""
Rolling-Horizon CP-SAT Experiment  (4 x 120-min windows, 3-min budget each)
===========================================================================
Splits the 480-min shift into 4 windows of 120 minutes.
Each window is solved independently by CP-SAT with a 3-minute budget.
State (RC stock, GC stock, last roaster SKU, remaining MTO batches) is
carried forward between windows for the SOLVER INPUTS only.

IMPORTANT: final profit is computed by replaying the COMBINED global schedule
through the full 480-min accounting (not summed from per-window results).
This gives a single honest number comparable to the full CP-SAT oracle.

Run:
  python Experiment/rolling_horizon_cpsat.py
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

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
WINDOW_SIZE     = 120
N_WINDOWS       = 4
BUDGET_SEC      = 3 * 60       # 3 minutes per window
NUM_WORKERS     = 8
INPUT_DIR       = str(_ROOT / "Input_data")
OUT_DIR         = Path(__file__).parent / "results" / "rolling"

REFERENCE_FULL  = 443_400
RL_HH_MEAN      = 359_096
DISPATCH_MEAN   = 326_554
ROASTERS        = ["R1", "R2", "R3", "R4", "R5"]


# ---------------------------------------------------------------------------
# Global profit replay
# ---------------------------------------------------------------------------

def compute_global_profit(global_schedule: list[dict],
                          base: dict, ups_events: list) -> dict:
    """Replay the full combined schedule over the 480-min timeline.

    All accounting mirrors the full CP-SAT solver:
      revenue, tardiness, setup events, idle cost, overflow cost, stockout cost.
    """
    SL           = SHIFT_LENGTH
    max_rc       = int(base["max_rc"])
    safety_stock = int(base["safety_stock"])
    cost_tard    = float(base["cost_tardiness"])
    cost_idle    = float(base["cost_idle"])
    cost_over    = float(base["cost_overflow"])
    cost_setup   = float(base["cost_setup"])
    cost_stock   = float(base["cost_stockout"])
    cost_skip    = float(base["cost_skip_mto"])
    sku_rev      = base["sku_revenue"]

    # --- Revenue ---
    psc_n   = sum(1 for e in global_schedule if e["sku"] == "PSC")
    ndg_n   = sum(1 for e in global_schedule if e["sku"] == "NDG")
    busta_n = sum(1 for e in global_schedule if e["sku"] == "BUSTA")
    revenue = psc_n * sku_rev["PSC"] + ndg_n * sku_rev["NDG"] + busta_n * sku_rev["BUSTA"]

    # --- Tardiness ---
    mto_completion: dict[str, int] = {}   # job_id -> global time last batch finishes
    for e in global_schedule:
        if e.get("is_mto") and e.get("job_id"):
            jid = e["job_id"]
            mto_completion[jid] = max(mto_completion.get(jid, 0), int(e["end"]))

    tard_min = 0.0
    skip_cost = 0.0
    for jid in base["jobs"]:
        due = int(base["job_due"][jid])
        if jid in mto_completion:
            late = max(0, mto_completion[jid] - due)
            tard_min += late
        else:
            # not scheduled at all → skip penalty
            skip_cost += base["job_batches"][jid] * cost_skip

    tard_cost = tard_min * cost_tard + skip_cost

    # --- Setup events (per roaster, sorted by start) ---
    by_roaster: dict[str, list] = {r: [] for r in base["roasters"]}
    for e in global_schedule:
        by_roaster[e["roaster"]].append(e)
    for r in by_roaster:
        by_roaster[r].sort(key=lambda x: x["start"])

    setup_events = 0
    for roaster, entries in by_roaster.items():
        prev_sku = base["roaster_initial_sku"][roaster]
        for idx, e in enumerate(entries):
            needs_setup = e["sku"] != prev_sku and (idx > 0 or e.get("is_mto"))
            if needs_setup:
                setup_events += 1
            prev_sku = e["sku"]
    setup_cost = setup_events * cost_setup

    # --- RC timeline (full 480 min) ---
    # build completions per line from PSC batches
    rc_completions: dict[str, dict[int, int]] = {"L1": {}, "L2": {}}
    for e in global_schedule:
        if e["sku"] == "PSC" and e.get("output_line"):
            t = int(e["end"])
            if t < SL:
                line = e["output_line"]
                rc_completions[line][t] = rc_completions[line].get(t, 0) + 1

    # full-shift consumption events from base
    consume_events = base["consumption_events"]  # {line: [t, ...]}

    # UPS downtime per roaster (global slots)
    ups_down: dict[str, set] = {r: set() for r in base["roasters"]}
    for ev in ups_events:
        rid = getattr(ev, "roaster_id", None)
        t   = getattr(ev, "t", None)
        dur = getattr(ev, "duration", None)
        if rid and t is not None and dur is not None and rid in ups_down:
            for s in range(int(t), min(SL, int(t) + max(0, int(dur) - 1))):
                ups_down[rid].add(s)
    # also planned downtime from base
    for rid, slots in base["downtime_slots"].items():
        ups_down[rid] |= slots

    rc_level: dict[str, list] = {}
    stockout_events = {"L1": 0, "L2": 0}
    for line in ("L1", "L2"):
        level = int(base["rc_init"][line])
        consume_set = set(consume_events.get(line, []))
        timeline = []
        for t in range(SL):
            if t in rc_completions[line]:
                level += rc_completions[line][t]
            if t in consume_set:
                if level <= 0:
                    stockout_events[line] += 1
                level -= 1
            timeline.append(level)
        rc_level[line] = timeline

    stockout_cost = sum(stockout_events.values()) * cost_stock

    # --- Idle and overflow minutes ---
    # build busy intervals per roaster
    busy: dict[str, list] = {r: [] for r in base["roasters"]}
    for e in global_schedule:
        busy[e["roaster"]].append((int(e["start"]), int(e["end"])))

    idle_min = 0
    over_min = 0
    for roaster in base["roasters"]:
        line = base["roaster_line"][roaster]
        busy_set = set()
        for s, en in busy[roaster]:
            for t in range(s, en):
                busy_set.add(t)
        for t in range(SL):
            if t in ups_down[roaster]:
                continue
            if t in busy_set:
                continue
            # idle: check safety stock
            if rc_level[line][t] < safety_stock:
                idle_min += 1
            # overflow: check max RC
            if roaster == "R3" and base.get("allow_r3_flex"):
                if rc_level["L1"][t] >= max_rc and rc_level["L2"][t] >= max_rc:
                    over_min += 1
            else:
                out_lines = base["roaster_can_output"].get(roaster, [line])
                if rc_level[out_lines[0]][t] >= max_rc:
                    over_min += 1

    idle_cost = idle_min * cost_idle
    over_cost = over_min * cost_over

    total_costs = tard_cost + setup_cost + idle_cost + over_cost + stockout_cost
    net_profit  = revenue - total_costs

    return {
        "net_profit":      round(net_profit, 2),
        "revenue":         round(revenue, 2),
        "tard_cost":       round(tard_cost, 2),
        "tard_min":        round(tard_min, 2),
        "skip_cost":       round(skip_cost, 2),
        "setup_cost":      round(setup_cost, 2),
        "setup_events":    setup_events,
        "idle_cost":       round(idle_cost, 2),
        "idle_min":        idle_min,
        "over_cost":       round(over_cost, 2),
        "over_min":        over_min,
        "stockout_cost":   round(stockout_cost, 2),
        "stockout_events": dict(stockout_events),
        "psc_count":       psc_n,
        "ndg_count":       ndg_n,
        "busta_count":     busta_n,
        "total_batches":   psc_n + ndg_n + busta_n,
    }


# ---------------------------------------------------------------------------
# Data dict builder for one window
# ---------------------------------------------------------------------------

def _build_window_data(base: dict, window_idx: int, state: dict,
                       all_ups: list) -> dict:
    w_start = window_idx * WINDOW_SIZE
    w_end   = w_start + WINDOW_SIZE

    d = copy.deepcopy(base)
    d["shift_length"] = WINDOW_SIZE
    d["time_limit"]   = BUDGET_SEC
    d["MS_by_sku"]    = {sku: WINDOW_SIZE - dur for sku, dur in d["roast_time_by_sku"].items()}

    d["rc_init"]            = dict(state["rc_init"])
    d["gc_init"]            = {k: v for k, v in state["gc_init"].items()}
    d["roaster_initial_sku"] = dict(state["roaster_last_sku"])

    # consumption events: filter + shift
    d["consumption_events"] = {
        line: [t - w_start for t in evs if w_start <= t < w_end]
        for line, evs in base["consumption_events"].items()
    }

    # downtime: shift base planned downtime + UPS events in this window
    d["downtime_slots"] = {r: set() for r in d["roasters"]}
    for roaster, slots in base["downtime_slots"].items():
        d["downtime_slots"][roaster] = {s - w_start for s in slots if w_start <= s < w_end}
    for ev in all_ups:
        rid = getattr(ev, "roaster_id", None)
        t   = getattr(ev, "t", None)
        dur = getattr(ev, "duration", None)
        if rid not in d["downtime_slots"] or t is None:
            continue
        eff = max(0, int(dur) - 1)
        t_rel = int(t) - w_start
        for s in range(t_rel, min(WINDOW_SIZE, t_rel + eff)):
            if s >= 0:
                d["downtime_slots"][rid].add(s)

    # MTO jobs: only those with remaining batches
    mto_rem   = state["mto_remaining"]
    active    = [jid for jid, rem in mto_rem.items() if rem > 0]
    d["jobs"]        = active
    d["job_sku"]     = {jid: base["job_sku"][jid] for jid in active}
    d["job_batches"] = {jid: mto_rem[jid] for jid in active}
    # Keep ABSOLUTE due date awareness (relative to window start)
    # Solver knows the deadline is coming — this is realistic for a rolling horizon
    d["job_due"]     = {jid: max(0, base["job_due"][jid] - w_start) for jid in active}
    d["job_release"] = {jid: max(0, base["job_release"][jid] - w_start) for jid in active}

    # rebuild batch lists
    psc_pool_size = math.floor(WINDOW_SIZE / d["roast_time_by_sku"]["PSC"])
    d["psc_pool_per_roaster"] = psc_pool_size

    mto_batches = [(jid, bi) for jid in active for bi in range(d["job_batches"][jid])]
    psc_pool    = [(r, si) for r in d["roasters"] for si in range(psc_pool_size)]
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
        bid: list(d["sku_eligible_roasters"]["PSC"]) for bid in psc_pool
    })
    d["sched_eligible_roasters"] = {
        bid: [bid[0]] if bid in set(psc_pool) else list(d["batch_eligible_roasters"][bid])
        for bid in all_batches
    }
    return d


def _extract_state(result: dict, prev_state: dict, base: dict) -> dict:
    schedule = result.get("schedule", [])

    rc_init = dict(result.get("rc_final", prev_state["rc_init"]))

    gc_final_raw = result.get("gc_final", {})
    gc_init = {}
    for pair in base["feasible_gc_pairs"]:
        key_str = f"{pair[0]}_{pair[1]}"
        gc_init[pair] = max(0, min(
            int(gc_final_raw.get(key_str, prev_state["gc_init"].get(pair, 0))),
            int(base["gc_capacity"][pair])
        ))

    roaster_last_sku = dict(prev_state["roaster_last_sku"])
    by_roaster: dict[str, list] = {r: [] for r in base["roasters"]}
    for e in schedule:
        by_roaster[e["roaster"]].append(e)
    for roaster, entries in by_roaster.items():
        if entries:
            roaster_last_sku[roaster] = max(entries, key=lambda e: e["end"])["sku"]

    mto_remaining = dict(prev_state["mto_remaining"])
    for e in schedule:
        if e.get("is_mto") and e.get("job_id"):
            jid = e["job_id"]
            if jid in mto_remaining:
                mto_remaining[jid] = max(0, mto_remaining[jid] - 1)

    return {
        "rc_init":          rc_init,
        "gc_init":          gc_init,
        "roaster_last_sku": roaster_last_sku,
        "mto_remaining":    mto_remaining,
    }


# ---------------------------------------------------------------------------
# Rolling horizon runner
# ---------------------------------------------------------------------------

def run_rolling_horizon(ups_events: list) -> tuple[list[dict], dict, dict]:
    base = cpsat_load_data(input_dir=INPUT_DIR, overrides={"time_limit": BUDGET_SEC})

    state: dict = {
        "rc_init":          dict(base["rc_init"]),
        "gc_init":          dict(base["gc_init"]),
        "roaster_last_sku": dict(base["roaster_initial_sku"]),
        "mto_remaining":    {jid: base["job_batches"][jid] for jid in base["jobs"]},
    }

    window_results   = []
    global_schedule  = []   # all batches, time-shifted to absolute global time
    global_restocks  = []

    for w in range(N_WINDOWS):
        w_start = w * WINDOW_SIZE
        w_end   = w_start + WINDOW_SIZE
        print(f"\n  Window {w+1}/4  [{w_start}-{w_end} min]  budget={BUDGET_SEC//60}min ...",
              end=" ", flush=True)

        t0 = time.perf_counter()
        try:
            d      = _build_window_data(base, w, state, ups_events)
            model, cp_vars = cpsat_build_model(d)
            result = cpsat_solve(d, model, cp_vars, num_workers=NUM_WORKERS)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"FAILED: {exc}")
            window_results.append({"window": w+1, "w_start": w_start, "w_end": w_end,
                                   "status": "FAILED", "solve_sec": round(elapsed,2)})
            continue

        elapsed = time.perf_counter() - t0
        if result is None:
            print(f"UNKNOWN (no feasible solution in {elapsed:.1f}s)")
            window_results.append({"window": w+1, "w_start": w_start, "w_end": w_end,
                                   "status": "UNKNOWN", "solve_sec": round(elapsed,2)})
            continue

        psc = result.get("psc_count", 0)
        ndg = result.get("ndg_count", 0)
        bst = result.get("busta_count", 0)
        print(f"P/N/B={psc}/{ndg}/{bst}  incumbents={result.get('num_incumbents',0)}  "
              f"gap={result.get('gap_pct','?')}%  wall={elapsed:.1f}s")

        window_results.append({
            "window":     w + 1,
            "w_start":    w_start,
            "w_end":      w_end,
            "status":     result.get("status", "?"),
            "psc_count":  psc, "ndg_count": ndg, "busta_count": bst,
            "incumbents": result.get("num_incumbents", 0),
            "gap_pct":    result.get("gap_pct"),
            "solve_sec":  round(elapsed, 2),
        })

        # Time-shift schedule entries to global time
        for e in result.get("schedule", []):
            ge = dict(e)
            ge["start"]          += w_start
            ge["end"]            += w_start
            ge["pipeline_start"] += w_start
            ge["pipeline_end"]   += w_start
            ge["window"]          = w + 1
            global_schedule.append(ge)

        for rst in result.get("restocks", []):
            gr = dict(rst)
            gr["start"] += w_start
            gr["end"]   += w_start
            gr["window"]  = w + 1
            global_restocks.append(gr)

        state = _extract_state(result, state, base)

    # Global profit replay over full 480-min timeline
    print("\n  Computing global profit (full-shift replay)...")
    global_kpi = compute_global_profit(global_schedule, base, ups_events)

    summary = {
        "seed": SEED, "ups_lambda": UPS_LAMBDA, "ups_mu": UPS_MU,
        "n_windows": N_WINDOWS, "window_size_min": WINDOW_SIZE,
        "budget_per_window_sec": BUDGET_SEC,
        "total_solve_sec":   round(sum(r.get("solve_sec",0) for r in window_results), 2),
        "total_frozen_min":  round(sum(r.get("solve_sec",0) for r in window_results) / 60, 2),
        "pct_shift_frozen":  round(sum(r.get("solve_sec",0) for r in window_results) / 60 / SHIFT_LENGTH * 100, 2),
        "reference_full_cpsat": REFERENCE_FULL,
        "reference_rlhh":    RL_HH_MEAN,
        "reference_dispatch": DISPATCH_MEAN,
        **global_kpi,
        "gap_vs_full_pct": round((REFERENCE_FULL - global_kpi["net_profit"]) / REFERENCE_FULL * 100, 2)
                           if global_kpi["net_profit"] else None,
    }
    return window_results, summary, global_schedule


# ---------------------------------------------------------------------------
# Gantt table
# ---------------------------------------------------------------------------

def _gantt_html(global_schedule: list[dict]) -> str:
    by_roaster: dict[str, list] = {r: [] for r in ROASTERS}
    for e in global_schedule:
        by_roaster[e["roaster"]].append(e)
    for r in by_roaster:
        by_roaster[r].sort(key=lambda x: x["start"])

    sku_colors = {"PSC": "#5b9bd5", "NDG": "#70ad47", "BUSTA": "#ed7d31"}
    rows = ""
    for roaster in ROASTERS:
        for e in by_roaster[roaster]:
            col = sku_colors.get(e["sku"], "#ccc")
            mto_tag = " <em>(MTO)</em>" if e.get("is_mto") else ""
            rows += (
                f"<tr>"
                f"<td>{roaster}</td>"
                f"<td style='background:{col};color:#fff;font-weight:bold'>{e['sku']}{mto_tag}</td>"
                f"<td>{e['start']}</td>"
                f"<td>{e['end']}</td>"
                f"<td>{e['end']-e['start']}</td>"
                f"<td>W{e.get('window','?')}</td>"
                f"<td>{e.get('output_line') or '—'}</td>"
                f"</tr>"
            )
    return (
        "<table><thead><tr>"
        "<th>Roaster</th><th>SKU</th><th>Start</th><th>End</th>"
        "<th>Duration</th><th>Window</th><th>Output</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def make_chart(window_results: list[dict], summary: dict, out_path: Path) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return ""

    valid  = [r for r in window_results if r.get("psc_count") is not None]
    labels = [f"W{r['window']}\n[{r['w_start']}-{r['w_end']}]" for r in valid]
    psc_c  = [r.get("psc_count",0)   for r in valid]
    ndg_c  = [r.get("ndg_count",0)   for r in valid]
    bst_c  = [r.get("busta_count",0) for r in valid]
    x = np.arange(len(labels))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Rolling-Horizon CP-SAT  |  4 x 120-min  |  {BUDGET_SEC//60}-min budget  |  Seed {SEED}",
        fontsize=12, fontweight="bold"
    )

    ax1.bar(x - w, psc_c, w, label="PSC", color="#5b9bd5")
    ax1.bar(x,     ndg_c, w, label="NDG", color="#70ad47")
    ax1.bar(x + w, bst_c, w, label="BUSTA", color="#ed7d31")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Batches")
    ax1.set_title("Batches Produced per Window")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    methods = ["Full CP-SAT\n(oracle)", f"Rolling\nCP-SAT\n({N_WINDOWS}x{BUDGET_SEC//60}min)", "RL-HH\n(mean)", "Dispatch\n(mean)"]
    values  = [REFERENCE_FULL, summary["net_profit"], RL_HH_MEAN, DISPATCH_MEAN]
    colors  = ["#ed7d31", "#e05c2a", "#70ad47", "#ffc000"]
    ax2.bar(methods, values, color=colors, width=0.5, zorder=2)
    ax2.set_ylabel("Net Profit ($)")
    ax2.set_title("Global Profit Comparison")
    ax2.grid(axis="y", alpha=0.3, zorder=1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    for i, v in enumerate(values):
        ax2.text(i, v + 1000, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  [chart] saved -> {out_path}")
    return str(out_path)


def make_html(window_results, summary, global_schedule, chart_path, out_path):
    import base64
    chart_b64 = ""
    if chart_path and Path(chart_path).exists():
        with open(chart_path, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode()

    def money(v):
        return f"${v:,.0f}" if v is not None else "—"

    win_rows = ""
    for r in window_results:
        win_rows += (
            f"<tr><td>W{r['window']} [{r['w_start']}–{r['w_end']}]</td>"
            f"<td>{r.get('status','?')}</td>"
            f"<td>{r.get('psc_count','—')}/{r.get('ndg_count','—')}/{r.get('busta_count','—')}</td>"
            f"<td>{r.get('incumbents','—')}</td>"
            f"<td>{r.get('gap_pct','—')}</td>"
            f"<td>{r.get('solve_sec','—')}s</td></tr>"
        )

    gap = f"{summary.get('gap_vs_full_pct','?')}%"
    img_tag = (f'<img src="data:image/png;base64,{chart_b64}" style="max-width:100%;border:1px solid #ddd;border-radius:6px">'
               if chart_b64 else "")
    gantt = _gantt_html(global_schedule)

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head><meta charset='utf-8'>
<title>Rolling Horizon CP-SAT</title>
<style>
  body{{font-family:-apple-system,Segoe UI,sans-serif;margin:32px;color:#222;max-width:1300px}}
  h1{{margin-bottom:4px}} h2{{border-bottom:2px solid #ddd;padding-bottom:4px;margin-top:28px}}
  table{{border-collapse:collapse;margin:10px 0;font-size:.9em}}
  th,td{{padding:5px 10px;border:1px solid #ccc;text-align:right}}
  th{{background:#efefef;text-align:center}} td:first-child{{text-align:left}}
  .note{{background:#fff9e6;border-left:4px solid #ffc000;padding:10px 16px;border-radius:4px;margin:12px 0;font-size:.91em}}
  .bad{{color:#c0392b;font-weight:bold}} .good{{color:#27ae60;font-weight:bold}}
  .hl{{background:#eef4ff}}
</style>
</head>
<body>
<h1>Rolling-Horizon CP-SAT  —  4 x 120 min  |  {BUDGET_SEC//60}-min budget/window</h1>
<p>Seed {SEED} | UPS &lambda;={UPS_LAMBDA} &mu;={UPS_MU} | Profit = <strong>full-shift replay</strong> (not summed per-window)</p>

<div class='note'>
  <strong>Accounting:</strong> The per-window schedules are time-shifted to global time and replayed
  over the full 480-min horizon. Revenue, tardiness, setup events, idle/overflow costs and stockout
  are all computed on the COMBINED schedule against the full consume-event timeline.
  This is directly comparable to the full CP-SAT oracle and the simulation-engine results for RL-HH.
</div>

<h2>Global Full-Shift KPIs</h2>
<table style='width:auto'>
<thead><tr><th>KPI</th><th>Rolling CP-SAT ({N_WINDOWS}&times;{BUDGET_SEC//60}min)</th><th>Full CP-SAT oracle</th></tr></thead>
<tbody>
<tr><td>Net Profit</td><td class='bad'>{money(summary["net_profit"])}</td><td>{money(REFERENCE_FULL)}</td></tr>
<tr><td>Gap vs oracle</td><td class='bad'>{gap}</td><td>0%</td></tr>
<tr><td>Revenue</td><td>{money(summary["revenue"])}</td><td>$486,000</td></tr>
<tr><td>Tardiness Cost</td><td>{money(summary["tard_cost"])}</td><td>$0</td></tr>
<tr><td>Setup Cost</td><td>{money(summary["setup_cost"])}</td><td>$4,000</td></tr>
<tr><td>Idle Cost</td><td>{money(summary["idle_cost"])}</td><td>$38,600</td></tr>
<tr><td>Stockout Cost</td><td>{money(summary["stockout_cost"])}</td><td>$0</td></tr>
<tr><td>PSC / NDG / BUSTA batches</td>
    <td>{summary["psc_count"]} / {summary["ndg_count"]} / {summary["busta_count"]}</td>
    <td>104 / 5 / 5</td></tr>
<tr><td>Solver frozen time</td>
    <td class='bad'>{summary["total_frozen_min"]:.1f} min ({summary["pct_shift_frozen"]:.1f}% of shift)</td>
    <td>—</td></tr>
</tbody>
</table>

<h2>Method Comparison</h2>
<table style='width:auto'>
<thead><tr><th>Method</th><th>Net Profit</th><th>Gap vs Oracle</th><th>Decision latency</th><th>UPS knowledge</th></tr></thead>
<tbody>
<tr><td>Full CP-SAT (oracle, seed 69)</td><td>{money(REFERENCE_FULL)}</td><td>0%</td><td>~8h offline</td><td>Pre-known</td></tr>
<tr class='hl'><td><strong>Rolling CP-SAT ({N_WINDOWS}&times;{BUDGET_SEC//60}min)</strong></td>
    <td><strong>{money(summary["net_profit"])}</strong></td>
    <td><strong class='bad'>{gap}</strong></td>
    <td class='bad'>{summary["total_frozen_min"]:.1f} min frozen</td><td>Pre-known</td></tr>
<tr><td>RL-HH (100-seed mean)</td><td class='good'>{money(RL_HH_MEAN)}</td>
    <td>{((REFERENCE_FULL-RL_HH_MEAN)/REFERENCE_FULL*100):.1f}%</td>
    <td class='good'>&lt;1 ms</td><td class='good'>Online / stochastic</td></tr>
<tr><td>Dispatching (100-seed mean)</td><td>{money(DISPATCH_MEAN)}</td>
    <td>{((REFERENCE_FULL-DISPATCH_MEAN)/REFERENCE_FULL*100):.1f}%</td>
    <td class='good'>&lt;1 ms</td><td class='good'>Online / stochastic</td></tr>
</tbody>
</table>

<h2>Per-Window Solve Info</h2>
<table>
<thead><tr><th>Window</th><th>Status</th><th>PSC/NDG/BUSTA</th><th>Incumbents</th><th>MIP Gap</th><th>Solve time</th></tr></thead>
<tbody>{win_rows}</tbody>
</table>

<h2>Chart</h2>
{img_tag}

<h2>Full Schedule (Gantt)</h2>
<p>All {len(global_schedule)} batches across 4 windows, sorted by roaster then start time.</p>
{gantt}

<p style='color:#888;font-size:.85em;margin-top:32px'>Generated by <code>Experiment/rolling_horizon_cpsat.py</code></p>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  [html] saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"Rolling-Horizon CP-SAT | {N_WINDOWS} x {WINDOW_SIZE} min | "
          f"{BUDGET_SEC//60}-min budget/window")
    print(f"Seed {SEED} | UPS lam={UPS_LAMBDA} mu={UPS_MU}")
    print("=" * 70)

    ups_events = list(generate_ups_events(
        UPS_LAMBDA, UPS_MU, seed=SEED,
        shift_length=SHIFT_LENGTH, roasters=ROASTERS,
    ))
    print(f"UPS events: {len(ups_events)}")
    for ev in ups_events:
        print(f"  t={ev.t:>4}  roaster={ev.roaster_id}  dur={ev.duration:.1f} min")

    window_results, summary, global_schedule = run_rolling_horizon(ups_events)

    (OUT_DIR / "window_results.json").write_text(json.dumps(window_results, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "schedule.json").write_text(json.dumps(global_schedule, indent=2), encoding="utf-8")
    print(f"\n  [json] saved -> {OUT_DIR}")

    chart_path = make_chart(window_results, summary, OUT_DIR / "rolling_chart.png")
    make_html(window_results, summary, global_schedule, chart_path, OUT_DIR / "report.html")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for r in window_results:
        pnb = f"{r.get('psc_count','?')}/{r.get('ndg_count','?')}/{r.get('busta_count','?')}"
        print(f"  W{r['window']} [{r['w_start']:>3}-{r['w_end']:>3}]  "
              f"P/N/B={pnb:<8}  incumbents={r.get('incumbents','?'):>3}  "
              f"gap={str(r.get('gap_pct','?')):>8}%  wall={r.get('solve_sec','?')}s")
    print("-" * 70)
    print(f"  GLOBAL net profit : {money(summary['net_profit'])} "
          f"(gap vs oracle: {summary.get('gap_vs_full_pct','?')}%)")
    print(f"    Revenue         : {money(summary['revenue'])}")
    print(f"    Tardiness cost  : {money(summary['tard_cost'])}")
    print(f"    Idle cost       : {money(summary['idle_cost'])}")
    print(f"    Setup cost      : {money(summary['setup_cost'])}")
    print(f"    Stockout cost   : {money(summary['stockout_cost'])}")
    print(f"  Frozen time       : {summary['total_frozen_min']:.1f} min "
          f"({summary['pct_shift_frozen']:.1f}% of shift)")
    print(f"  Full CP-SAT oracle: {money(REFERENCE_FULL)}")
    print(f"  RL-HH (100-seed)  : {money(RL_HH_MEAN)}")
    print(f"  Dispatching       : {money(DISPATCH_MEAN)}")
    print("=" * 70)
    print(f"\nReport: {OUT_DIR / 'report.html'}")


def money(v):
    return f"${v:,.0f}" if v is not None else "—"


if __name__ == "__main__":
    main()
