"""Round-trip smoke test: CPSAT_Pure -> result_schema JSON -> plot_result HTML.

Verifies the active CP-SAT pipeline still works after Reactive_CPSAT was moved
to OLDCODE. Uses a short budget (60s) so the test is quick.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CPSAT_Pure.runner import run_pure_cpsat
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data
from plot_result import _build_html
from result_schema import create_result


def _normalize_gc(d: dict) -> dict:
    return {f"{k[0]}_{k[1]}" if isinstance(k, tuple) else str(k): v for k, v in d.items()}


def main() -> int:
    SEED = 900046
    BUDGET_S = 60
    OUT_DIR = Path("output")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" Round-trip smoke test: CPSAT_Pure -> plot_result")
    print(f" Seed: {SEED}  |  Budget: {BUDGET_S}s  |  UPS visible upfront: yes")
    print("=" * 70)

    # 1. Generate UPS events for this seed
    data = load_data()
    params = data.to_env_params()
    ups = generate_ups_events(
        data.ups_lambda, data.ups_mu, SEED,
        int(params["SL"]), list(params["roasters"]),
    )
    print(f"\n[1/4] UPS realization for seed={SEED}: {len(ups)} events")
    for i, e in enumerate(ups):
        print(f"      ev{i+1}: t={e.t}  roaster={e.roaster_id}  duration={e.duration}min")

    # 2. Solve CPSAT_Pure with UPS pre-merged into downtime
    print(f"\n[2/4] Calling run_pure_cpsat (CPSAT_Pure.runner)…")
    t0 = time.perf_counter()
    cp_result = run_pure_cpsat(
        time_limit_sec=BUDGET_S,
        ups_events=ups,
        num_workers=8,
    )
    wall = time.perf_counter() - t0
    print(f"      Solved in {wall:.1f}s")
    print(f"      Status: {cp_result['status']}")
    print(f"      Net profit: ${cp_result['net_profit']:,.0f}")
    print(f"      Best bound: ${cp_result['best_bound']:,.0f}")
    print(f"      Gap: {cp_result['gap_pct']:.2f}%")
    print(f"      Incumbents found: {cp_result['num_incumbents']}")
    print(f"      Schedule entries: {len(cp_result['schedule'])}")
    print(f"      UPS events applied: {cp_result.get('ups_events_applied', '?')}")
    so_events = cp_result.get("stockout_events", {})
    so_cost = cp_result.get("stockout_cost", 0.0)
    print(f"      Stockouts: {so_events} (${so_cost:,.0f})")
    skipped = cp_result.get("mto_skipped", 0)
    skip_cost = cp_result.get("skip_cost", 0.0)
    print(f"      MTO skipped: {skipped} batches (${skip_cost:,.0f})")

    # 3. Convert CPSAT result to universal result_schema for plot_result
    print(f"\n[3/4] Converting to universal result_schema…")
    schedule = cp_result["schedule"]
    # Re-tag setup field for plot_result
    last_sku = dict(params.get("roaster_initial_sku", {}))
    for entry in sorted(schedule, key=lambda e: (e["roaster"], e["start"])):
        rid = entry["roaster"]
        sku = entry["sku"]
        entry["setup"] = "Yes" if sku != last_sku.get(rid, "PSC") else "No"
        last_sku[rid] = sku

    result = create_result(
        metadata={
            "solver_engine": "cpsat",
            "solver_name": "CP-SAT (Pure v3, perfect-information UPS)",
            "status": cp_result["status"],
            "solve_time_sec": cp_result["solve_time"],
            "total_compute_ms": cp_result["solve_time"] * 1000.0,
            "obj_value": cp_result["obj_value"],
            "best_bound": cp_result["best_bound"],
            "gap_pct": cp_result["gap_pct"],
            "num_incumbents": cp_result["num_incumbents"],
            "allow_r3_flex": cp_result.get("allow_r3_flex", True),
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "input_dir": str(data.input_dir),
            "notes": (
                f"CPSAT_Pure with {len(ups)} UPS events merged as downtime. "
                f"{cp_result['num_incumbents']} incumbents, "
                f"final gap {cp_result['gap_pct']:.2f}%."
            ),
        },
        experiment={
            "lambda_rate": float(data.ups_lambda),
            "mu_mean": float(data.ups_mu),
            "seed": SEED,
            "scenario_label": "ups_perfect_info" if ups else "no_ups",
        },
        kpi={
            "net_profit": cp_result["net_profit"],
            "total_revenue": cp_result["total_revenue"],
            "total_costs": cp_result["total_costs"],
            "psc_count": cp_result["psc_count"],
            "ndg_count": cp_result["ndg_count"],
            "busta_count": cp_result["busta_count"],
            "revenue_psc": cp_result["revenue_psc"],
            "revenue_ndg": cp_result["revenue_ndg"],
            "revenue_busta": cp_result["revenue_busta"],
            "tardiness_min": cp_result.get("tardiness_min", {}),
            "tard_cost": cp_result["tard_cost"],
            "setup_events": cp_result["setup_events"],
            "setup_cost": cp_result["setup_cost"],
            "stockout_events": cp_result.get("stockout_events", {"L1": 0, "L2": 0}),
            "stockout_duration": cp_result.get("stockout_duration", {"L1": 0, "L2": 0}),
            "stockout_cost": cp_result.get("stockout_cost", 0.0),
            "idle_min": cp_result["idle_min"],
            "idle_cost": cp_result["idle_cost"],
            "over_min": cp_result["over_min"],
            "over_cost": cp_result["over_cost"],
            "restock_count": cp_result["restock_count"],
            "gc_init": cp_result.get("gc_init", {}),
            "gc_final": cp_result.get("gc_final", {}),
        },
        schedule=schedule,
        cancelled_batches=[],
        ups_events=[
            {"t": int(e.t), "roaster_id": e.roaster_id, "duration": int(e.duration)}
            for e in ups
        ],
        restocks=cp_result["restocks"],
        parameters={
            "SL": params["SL"], "sigma": params["sigma"], "DC": params["DC"],
            "max_rc": params["max_rc"], "safety_stock": params["safety_stock"],
            "rc_init": params["rc_init"],
            "restock_duration": params["restock_duration"],
            "restock_qty": params["restock_qty"],
            "roast_time_by_sku": params["roast_time_by_sku"],
            "consume_events": {k: list(v) for k, v in params["consume_events"].items()},
            "gc_capacity": _normalize_gc(params["gc_capacity"]),
            "gc_init": _normalize_gc(params["gc_init"]),
            "feasible_gc_pairs": [f"{k[0]}_{k[1]}" for k in params["gc_capacity"].keys()],
            "sku_revenue": {"PSC": params["rev_psc"], "NDG": params["rev_ndg"], "BUSTA": params["rev_busta"]},
            "c_tard": params["c_tard"], "c_stock": params["c_stock"],
            "c_idle": params["c_idle"], "c_over": params["c_over"],
            "c_setup": params["c_setup"],
        },
    )
    json_path = OUT_DIR / f"cpsat_seed{SEED}_budget{BUDGET_S}s_result.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"      JSON: {json_path}")

    # 4. Render via plot_result._build_html
    print(f"\n[4/4] Rendering via plot_result._build_html…")
    t0 = time.perf_counter()
    html = _build_html(result, None, offline=True)
    t_render = time.perf_counter() - t0
    html_path = OUT_DIR / f"cpsat_seed{SEED}_budget{BUDGET_S}s_schedule.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"      HTML: {html_path} ({len(html)/1024:.1f} KB) in {t_render*1000:.1f}ms")

    print()
    print("=" * 70)
    print(" Round-trip OK. CPSAT_Pure -> result_schema -> plot_result verified.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
