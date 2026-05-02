"""Reliable 5-seed comparison: CP-SAT (4 h each) + 4 reactive methods for seeds 65-69.

Steps
-----
  1. Run CP-SAT for seeds 65-68, 4 hours each  → save per-seed result JSON
  2. Copy seed-69 CP-SAT result from existing 8-hour reference run
  3. Run reactive methods (dispatching, ql, rlhh, paeng_v2) for seeds 65-69
  4. Rescue-render the full comparison_report.html combining all results

Output: results/method_comparison/<timestamp>_reliable_seeds65_69/
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─── Config ──────────────────────────────────────────────────────────────────
SEEDS_NEW    = [65, 66, 67, 68]   # CP-SAT will solve fresh for these seeds
SEED_REF     = 69                  # CP-SAT pre-computed from 8h run
CPSAT_4H_SEC = 14_400              # budget per new seed: 4 hours
NUM_WORKERS  = 8

CPSAT_REF_SRC = (
    _ROOT / "results" / "method_comparison"
    / "20260423_163357_Seed69_UPS" / "seed_69" / "cpsat_result.json"
)

STAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = _ROOT / "results" / "method_comparison" / f"{STAMP}_reliable_seeds65_69"

METHOD_LABELS = {
    "cpsat":       "CP-SAT",
    "dispatching": "Dispatching Heuristic",
    "ql":          "Q-Learning",
    "rlhh":        "RL-HH (Dueling DDQN)",
    "paeng_v2":    "Paeng DDQN v2",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print("\n" + "=" * 76)
    print(f"  {msg}")
    print("=" * 76)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _save_plot_html(result: dict, path: Path) -> None:
    try:
        from plot_result import _build_html
        path.write_text(_build_html(result, compare=None, offline=True), encoding="utf-8")
    except Exception as exc:
        import html as _html
        path.write_text(
            f"<html><body><h1>Plot error</h1><pre>{_html.escape(str(exc))}</pre></body></html>",
            encoding="utf-8",
        )


# ─── CP-SAT result builder ────────────────────────────────────────────────────

def _cpsat_to_schema(raw: dict, params: dict, seed: int, ups: list) -> dict:
    """Convert run_pure_cpsat raw output to result_schema format."""
    from result_schema import create_result
    from master_eval import _build_export_params

    kpi = {
        "net_profit":        raw.get("net_profit", 0.0),
        "total_revenue":     raw.get("total_revenue", 0.0),
        "total_costs":       raw.get("total_costs", 0.0),
        "psc_count":         raw.get("psc_count", 0),
        "ndg_count":         raw.get("ndg_count", 0),
        "busta_count":       raw.get("busta_count", 0),
        "total_batches":     raw.get("total_batches", 0),
        "revenue_psc":       raw.get("revenue_psc", 0.0),
        "revenue_ndg":       raw.get("revenue_ndg", 0.0),
        "revenue_busta":     raw.get("revenue_busta", 0.0),
        "tardiness_min":     raw.get("tardiness_min", {}),
        "tard_cost":         raw.get("tard_cost", 0.0),
        "setup_events":      raw.get("setup_events", 0),
        "setup_cost":        raw.get("setup_cost", 0.0),
        "stockout_events":   raw.get("stockout_events", {"L1": 0, "L2": 0}),
        "stockout_duration": raw.get("stockout_duration", {"L1": 0, "L2": 0}),
        "stockout_cost":     raw.get("stockout_cost", 0.0),
        "idle_min":          raw.get("idle_min", 0.0),
        "idle_cost":         raw.get("idle_cost", 0.0),
        "over_min":          raw.get("over_min", 0.0),
        "over_cost":         raw.get("over_cost", 0.0),
        "restock_count":     raw.get("restock_count", 0),
    }
    return create_result(
        metadata={
            "solver_engine": "cpsat",
            "solver_name": "CP-SAT (Pure v3, 4h budget)",
            "status": raw.get("status", "?"),
            "solve_time_sec": float(raw.get("solve_time", 0.0)),
            "total_compute_ms": float(raw.get("solve_time", 0.0)) * 1000.0,
            "obj_value": raw.get("obj_value", 0.0),
            "best_bound": raw.get("best_bound", 0.0),
            "gap_pct": raw.get("gap_pct"),
            "num_incumbents": raw.get("num_incumbents", 0),
            "allow_r3_flex": raw.get("allow_r3_flex", True),
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "input_dir": str(_ROOT / "Input_data"),
            "notes": (
                f"Pure CP-SAT v3. status={raw.get('status','?')}, "
                f"obj=${raw.get('obj_value',0):,.0f}, gap={raw.get('gap_pct')}%, "
                f"incumbents={raw.get('num_incumbents',0)}, "
                f"UPS_applied={raw.get('ups_events_applied',0)}, "
                f"solve={raw.get('solve_time',0):.1f}s ({raw.get('solve_time',0)/3600:.2f}h)"
            ),
        },
        experiment={
            "lambda_rate": float(params.get("ups_lambda", 0)),
            "mu_mean":     float(params.get("ups_mu", 0)),
            "seed": seed,
            "scenario_label": "ups_perfect_info" if ups else "no_ups",
        },
        kpi=kpi,
        schedule=raw.get("schedule", []),
        cancelled_batches=[],
        ups_events=[
            {"t": int(getattr(e, "t", 0)), "roaster_id": getattr(e, "roaster_id", "?"),
             "duration": int(getattr(e, "duration", 0))}
            for e in ups
        ],
        restocks=raw.get("restocks", []),
        parameters=_build_export_params(params),
    )


# ─── Reactive result builder ──────────────────────────────────────────────────

def _sim_to_schema(tag: str, kpi, state, params: dict, seed: int, model_path: str) -> dict:
    from result_schema import create_result
    from master_eval import _build_export_params
    from rl_hh.export_result import _batch_to_schedule_entry, _restock_to_entry, _ups_to_entry

    schedule  = [_batch_to_schedule_entry(b, params) for b in state.completed_batches]
    cancelled = [_batch_to_schedule_entry(b, params) for b in state.cancelled_batches]
    for c in cancelled:
        c["status"] = "cancelled"
    restocks   = [_restock_to_entry(r) for r in state.completed_restocks]
    ups_entries = [_ups_to_entry(e) for e in state.ups_events_fired]

    return create_result(
        metadata={
            "solver_engine": tag,
            "solver_name": METHOD_LABELS.get(tag, tag),
            "status": "Completed",
            "solve_time_sec": 0.0,
            "input_dir": str(_ROOT / "Input_data"),
            "model_path": model_path,
            "notes": f"run_reliable_comparison_65_69.py, seed={seed}.",
        },
        experiment={
            "lambda_rate": float(params.get("ups_lambda", 0)),
            "mu_mean":     float(params.get("ups_mu", 0)),
            "seed": seed,
        },
        kpi=kpi.to_dict(),
        schedule=schedule,
        cancelled_batches=cancelled,
        ups_events=ups_entries,
        restocks=restocks,
        parameters=_build_export_params(params),
    )


# ─── Step 1 + 2: CP-SAT ──────────────────────────────────────────────────────

def step_cpsat(params: dict) -> None:
    from CPSAT_Pure.runner import run_pure_cpsat
    from env.ups_generator import generate_ups_events

    _banner("STEP 1 — CP-SAT for seeds 65-68  (4 h each)")

    for seed in SEEDS_NEW:
        print(f"\n{'─'*70}")
        print(f"  CP-SAT  seed={seed}  budget={CPSAT_4H_SEC}s ({CPSAT_4H_SEC/3600:.1f}h)  "
              f"workers={NUM_WORKERS}")
        print(f"{'─'*70}")

        ups = generate_ups_events(
            params.get("ups_lambda", 0), params.get("ups_mu", 0),
            seed=seed, shift_length=int(params["SL"]),
            roasters=list(params["roasters"]),
        )
        print(f"  UPS realisation: {len(ups)} event(s)")
        for i, ev in enumerate(ups):
            print(f"    ev{i+1}: t={ev.t}  roaster={ev.roaster_id}  duration={ev.duration} min")

        t0 = time.perf_counter()
        raw = run_pure_cpsat(
            time_limit_sec=CPSAT_4H_SEC,
            ups_events=list(ups),
            num_workers=NUM_WORKERS,
        )
        elapsed = time.perf_counter() - t0

        print(f"\n  ─── Result  seed={seed} ───")
        print(f"  Status:      {raw['status']}")
        print(f"  Net profit:  ${raw['net_profit']:>12,.0f}")
        print(f"  Best bound:  ${raw['best_bound']:>12,.0f}")
        print(f"  Gap:         {raw.get('gap_pct')}%")
        print(f"  Incumbents:  {raw['num_incumbents']}")
        print(f"  Solve time:  {elapsed:.1f}s  ({elapsed/3600:.3f}h)")
        print(f"  Schedule:    {len(raw['schedule'])} batch entries")
        print(f"  UPS applied: {raw.get('ups_events_applied', '?')}")
        if raw.get("stockout_cost", 0.0) > 0:
            print(f"  Stockout:    {raw.get('stockout_events')}  ${raw.get('stockout_cost',0):,.0f}")
        if raw.get("mto_skipped", 0):
            print(f"  MTO skipped: {raw['mto_skipped']} batches  ${raw.get('skip_cost',0):,.0f}")

        result = _cpsat_to_schema(raw, params, seed, ups)
        out = OUT_DIR / f"seed_{seed:02d}" / "cpsat_result.json"
        _save_json(result, out)
        print(f"  Saved → {out.relative_to(_ROOT)}")

    _banner("STEP 2 — Copy seed-69 CP-SAT from 8-hour reference")
    if not CPSAT_REF_SRC.exists():
        print(f"  WARNING: reference not found at:\n    {CPSAT_REF_SRC}")
        print("  Seed-69 CP-SAT will be absent from the report.")
        return
    dst = OUT_DIR / "seed_69" / "cpsat_result.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CPSAT_REF_SRC, dst)
    ref = json.loads(dst.read_text(encoding="utf-8"))
    net = ref.get("kpi", {}).get("net_profit", "?")
    st  = ref.get("metadata", {}).get("solve_time_sec", 0)
    print(f"  Source: {CPSAT_REF_SRC.relative_to(_ROOT)}")
    print(f"  Status: {ref.get('metadata',{}).get('status','?')}  "
          f"net_profit=${net:,.0f}  solve={float(st)/3600:.1f}h  (original 8h run)")
    print(f"  Saved → {dst.relative_to(_ROOT)}")


# ─── Step 3: Reactive methods ─────────────────────────────────────────────────

def step_reactive(all_seeds: list[int], params: dict, data) -> None:
    from env.simulation_engine import SimulationEngine
    from env.ups_generator import generate_ups_events
    from scripts.compare_methods import (
        find_latest_ql_qtable, find_latest_rlhh_ckpt, find_latest_paeng_v2_ckpt,
    )
    from dispatch.dispatching_heuristic import DispatchingHeuristic
    from q_learning.q_strategy import QStrategy, load_q_table
    from rl_hh.meta_agent import DuelingDDQNAgent
    from rl_hh.rl_hh_strategy import RLHHStrategy
    from paeng_ddqn_v2.agent_v2 import PaengAgentV2
    from paeng_ddqn_v2.strategy_v2 import PaengStrategyV2

    _banner("STEP 3 — Reactive methods for seeds 65-69")

    ql_path    = find_latest_ql_qtable(_ROOT)
    rlhh_path  = find_latest_rlhh_ckpt(_ROOT)
    paeng_path = find_latest_paeng_v2_ckpt(_ROOT)

    print(f"  QL model:    {ql_path}")
    print(f"  RLHH model:  {rlhh_path}")
    print(f"  Paeng model: {paeng_path}")

    # Load RL agents once (shared across seeds — eval mode, no mutation)
    ql_table = load_q_table(str(ql_path)) if ql_path else None

    rlhh_agent = None
    if rlhh_path:
        rlhh_agent = DuelingDDQNAgent()
        rlhh_agent.load_checkpoint(str(rlhh_path))
        rlhh_agent.epsilon = 0.0

    paeng_agent = None
    if paeng_path:
        paeng_agent = PaengAgentV2.from_checkpoint(str(paeng_path))
        paeng_agent.epsilon = 0.0

    for seed in all_seeds:
        ups = generate_ups_events(
            params.get("ups_lambda", 0), params.get("ups_mu", 0),
            seed=seed, shift_length=int(params["SL"]),
            roasters=list(params["roasters"]),
        )
        ups_list = list(ups)
        seed_dir = OUT_DIR / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  ── Seed {seed}  ({len(ups_list)} UPS events) ──")

        # dispatching
        t0 = time.perf_counter()
        try:
            kpi, state = SimulationEngine(params).run(DispatchingHeuristic(params), ups_list)
            elapsed = time.perf_counter() - t0
            result = _sim_to_schema("dispatching", kpi, state, params, seed, "(rule-based)")
            k = kpi.to_dict()
            print(f"    dispatching   net=${k['net_profit']:>10,.0f}  "
                  f"P/N/B={k['psc_count']}/{k['ndg_count']}/{k['busta_count']}  "
                  f"tard=${k['tard_cost']:>7,.0f}  idle=${k['idle_cost']:>6,.0f}  ({elapsed:.1f}s)")
            _save_json(result, seed_dir / "dispatching_result.json")
            _save_plot_html(result, seed_dir / "dispatching_report.html")
        except Exception as exc:
            print(f"    dispatching   FAILED: {exc}")

        # Q-Learning
        t0 = time.perf_counter()
        try:
            strategy = QStrategy(params, q_table=ql_table)
            kpi, state = SimulationEngine(params).run(strategy, ups_list)
            elapsed = time.perf_counter() - t0
            result = _sim_to_schema("ql", kpi, state, params, seed, str(ql_path))
            k = kpi.to_dict()
            print(f"    ql            net=${k['net_profit']:>10,.0f}  "
                  f"P/N/B={k['psc_count']}/{k['ndg_count']}/{k['busta_count']}  "
                  f"tard=${k['tard_cost']:>7,.0f}  idle=${k['idle_cost']:>6,.0f}  ({elapsed:.1f}s)")
            _save_json(result, seed_dir / "ql_result.json")
            _save_plot_html(result, seed_dir / "ql_report.html")
        except Exception as exc:
            print(f"    ql            FAILED: {exc}")

        # RL-HH
        t0 = time.perf_counter()
        try:
            strategy = RLHHStrategy(rlhh_agent, data, training=False)
            kpi, state = SimulationEngine(params).run(strategy, ups_list)
            elapsed = time.perf_counter() - t0
            result = _sim_to_schema("rlhh", kpi, state, params, seed, str(rlhh_path))
            k = kpi.to_dict()
            print(f"    rlhh          net=${k['net_profit']:>10,.0f}  "
                  f"P/N/B={k['psc_count']}/{k['ndg_count']}/{k['busta_count']}  "
                  f"tard=${k['tard_cost']:>7,.0f}  idle=${k['idle_cost']:>6,.0f}  ({elapsed:.1f}s)")
            _save_json(result, seed_dir / "rlhh_result.json")
            _save_plot_html(result, seed_dir / "rlhh_report.html")
        except Exception as exc:
            print(f"    rlhh          FAILED: {exc}")

        # Paeng DDQN v2
        t0 = time.perf_counter()
        try:
            strategy = PaengStrategyV2(paeng_agent, data, training=False, params=params)
            kpi, state = SimulationEngine(params).run(strategy, ups_list)
            elapsed = time.perf_counter() - t0
            result = _sim_to_schema("paeng_v2", kpi, state, params, seed, str(paeng_path))
            k = kpi.to_dict()
            print(f"    paeng_v2      net=${k['net_profit']:>10,.0f}  "
                  f"P/N/B={k['psc_count']}/{k['ndg_count']}/{k['busta_count']}  "
                  f"tard=${k['tard_cost']:>7,.0f}  idle=${k['idle_cost']:>6,.0f}  ({elapsed:.1f}s)")
            _save_json(result, seed_dir / "paeng_v2_result.json")
            _save_plot_html(result, seed_dir / "paeng_v2_report.html")
        except Exception as exc:
            print(f"    paeng_v2      FAILED: {exc}")


# ─── Step 4: Rescue render ────────────────────────────────────────────────────

def step_render() -> None:
    _banner("STEP 4 — Rescue render  (CP-SAT + reactive → comparison_report.html)")
    from scripts.compare_methods import rescue_render
    rescue_render(OUT_DIR)
    report = OUT_DIR / "comparison_report.html"
    if report.exists():
        print(f"\n  Report: {report.relative_to(_ROOT)}")
    else:
        print("  WARNING: report HTML not created")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    t_total = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _banner(f"Reliable 5-Seed Comparison  —  seeds {sorted(SEEDS_NEW + [SEED_REF])}")
    print(f"  Output dir:    {OUT_DIR.relative_to(_ROOT)}")
    print(f"  CP-SAT budget: {CPSAT_4H_SEC}s per new seed  ({CPSAT_4H_SEC/3600:.1f}h each, "
          f"~{len(SEEDS_NEW)*CPSAT_4H_SEC/3600:.1f}h total CP-SAT time)")
    print(f"  Seed 69:       8h reference  (exists={CPSAT_REF_SRC.exists()})")

    from env.data_bridge import get_sim_params
    from PPOmask.Engine.data_loader import load_data

    params = get_sim_params(str(_ROOT / "Input_data"))
    data   = load_data(str(_ROOT / "Input_data"))
    all_seeds = sorted(set(SEEDS_NEW) | {SEED_REF})

    step_cpsat(params)
    step_reactive(all_seeds, params, data)
    step_render()

    total_h = (time.perf_counter() - t_total) / 3600
    _banner(f"ALL DONE  —  total wall time: {total_h:.2f}h")
    print(f"  Report: {OUT_DIR.relative_to(_ROOT)}/comparison_report.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
