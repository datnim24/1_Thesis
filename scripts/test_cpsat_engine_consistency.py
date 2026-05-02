"""V2 internal-accounting audit: CPSAT_Pure soft model self-consistency.

Verifies that the soft cost terms added to CPSAT_Pure are accounted for
identically in (a) the solver objective, (b) the post-solve KPI extraction,
and (c) the cost coefficients loaded from shift_parameters.csv.

Note on scope: this audit does NOT replay the CPSAT schedule through the
SimulationEngine for KPI parity. That is a separate, pre-existing concern —
CPSAT's interval model uses a different setup/GC/restock-station mechanic
than the engine's per-tick state machine, so a CPSAT schedule rarely executes
identically through the engine. The OLDCODE Reactive_CPSAT oracle was
specifically designed to bridge that gap by re-solving at each UPS event.
What this audit *does* verify is that the new soft constraints (per-event
stockout, MTO skip) are wired through end-to-end without arithmetic drift.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CPSAT_Pure.runner import run_pure_cpsat
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data


SEED = 900046
BUDGET_S = 60


def _read_csv_param(path: Path, key: str) -> float:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if (row.get("parameter") or "").strip() == key:
                return float(row["value"])
    raise KeyError(key)


def main() -> int:
    print("=" * 70)
    print(" V2 audit: CPSAT_Pure soft model — internal accounting consistency")
    print(f" Seed: {SEED}  |  Budget: {BUDGET_S}s")
    print("=" * 70)

    csv_path = _ROOT / "Input_data" / "shift_parameters.csv"
    csv_c_stock = _read_csv_param(csv_path, "stockout_cost_per_event_per_line")
    csv_c_skip = _read_csv_param(csv_path, "mto_skip_penalty_per_batch")
    print(f"\n[CSV] c_stock = ${csv_c_stock:,.0f} / event")
    print(f"[CSV] c_skip_mto = ${csv_c_skip:,.0f} / batch")

    data = load_data()
    params = data.to_env_params()
    ups = generate_ups_events(
        data.ups_lambda, data.ups_mu, SEED,
        int(params["SL"]), list(params["roasters"]),
    )
    print(f"\n[UPS] {len(ups)} events for seed={SEED}")

    print(f"\n[Solve] CPSAT_Pure (budget={BUDGET_S}s)...")
    t0 = time.perf_counter()
    r = run_pure_cpsat(time_limit_sec=BUDGET_S, ups_events=ups, num_workers=8)
    print(f"        Solved in {time.perf_counter()-t0:.1f}s | status={r['status']}")

    checks: list[tuple[str, bool, str]] = []

    # A. Cost coefficients flow CSV -> data.py -> result dict.
    cp_c_stock = float(r.get("cost_stockout", -1))
    cp_c_skip = float(r.get("cost_skip_mto", -1))
    checks.append((
        "A1. cost_stockout matches CSV",
        abs(cp_c_stock - csv_c_stock) < 1e-6,
        f"CSV={csv_c_stock} CPSAT={cp_c_stock}",
    ))
    checks.append((
        "A2. cost_skip_mto matches CSV",
        abs(cp_c_skip - csv_c_skip) < 1e-6,
        f"CSV={csv_c_skip} CPSAT={cp_c_skip}",
    ))

    # B. stockout_cost = c_stock * sum(stockout_events).
    so_events = r.get("stockout_events", {})
    so_total = sum(so_events.values()) if so_events else 0
    expected_so_cost = so_total * cp_c_stock
    checks.append((
        "B1. stockout_cost = c_stock * sum(stockout_events)",
        abs(r["stockout_cost"] - expected_so_cost) < 1e-3,
        f"reported=${r['stockout_cost']:,.2f} expected=${expected_so_cost:,.2f} (events={so_total})",
    ))

    # C. skip_cost = c_skip_mto * mto_skipped.
    skipped = int(r.get("mto_skipped", 0))
    expected_skip_cost = skipped * cp_c_skip
    checks.append((
        "C1. skip_cost = c_skip_mto * mto_skipped",
        abs(r["skip_cost"] - expected_skip_cost) < 1e-3,
        f"reported=${r['skip_cost']:,.2f} expected=${expected_skip_cost:,.2f} (skipped={skipped})",
    ))

    # D. tard_cost = tard_cost_pure + skip_cost (engine convention).
    expected_tard = r["tard_cost_pure"] + r["skip_cost"]
    checks.append((
        "D1. tard_cost = tard_cost_pure + skip_cost  (engine convention)",
        abs(r["tard_cost"] - expected_tard) < 1e-3,
        f"reported=${r['tard_cost']:,.2f} expected=${expected_tard:,.2f}",
    ))

    # E. Solver objective matches exact deterministic profit (already asserted
    #    inside solver.py via warning, but re-check explicitly).
    expected_total_costs = (
        r["tard_cost"] + r["setup_cost"] + r["idle_cost"]
        + r["over_cost"] + r["stockout_cost"]
    )
    expected_profit = r["total_revenue"] - expected_total_costs
    checks.append((
        "E1. net_profit = revenue - (tard + setup + idle + over + stockout)",
        abs(r["net_profit"] - expected_profit) < 1e-2,
        f"reported=${r['net_profit']:,.2f} expected=${expected_profit:,.2f}",
    ))
    checks.append((
        "E2. obj_value matches net_profit (solver -> KPI agreement)",
        abs(r["obj_value"] - r["net_profit"]) < 1.0,
        f"obj=${r['obj_value']:,.2f} net=${r['net_profit']:,.2f}",
    ))

    print()
    print(f"{'Check':<60} {'Result':>8}")
    print("-" * 72)
    all_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"{name:<60} {mark:>8}")
        print(f"   {detail}")

    print()
    print("=" * 70)
    print(" PASS  All internal accounting checks pass." if all_ok else " FAIL  See above.")
    print("=" * 70)

    print()
    print(f"Snapshot: net_profit=${r['net_profit']:,.0f} | gap={r['gap_pct']:.2f}%")
    print(f"          stockouts={so_events} (${r['stockout_cost']:,.0f})")
    print(f"          MTO skipped={skipped} (${r['skip_cost']:,.0f}, rolled into tard_cost)")
    print(f"          tard_cost_pure=${r['tard_cost_pure']:,.0f} | "
          f"tard_cost(combined)=${r['tard_cost']:,.0f}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
