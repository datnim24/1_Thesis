"""Re-evaluate C25 PPO best/final models at seed=42 UPS, overwrite PPO artifacts
in the master_eval folder, print schedule, and update the Master_Evaluation md.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TARGET_DIR = _ROOT / "results" / "20260419_233328_7800x_6h_1"
C25_RUN = _ROOT / "PPOmask" / "outputs" / "20260420_193326_c25_busta_fix"
C25_BEST = C25_RUN / "checkpoints" / "best_training_profit_model.zip"
C25_FINAL = C25_RUN / "checkpoints" / "final_model.zip"
EVAL_SEED = 42


def _save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _generate_html(result, output_path):
    from plot_result import _build_html
    html = _build_html(result, compare=None, offline=True)
    Path(output_path).write_text(html, encoding="utf-8")


def run_one(model_path, tag, eval_ups, params, data):
    from env.simulation_engine import SimulationEngine
    from PPOmask.Engine.ppo_strategy import PPOStrategy
    from sb3_contrib import MaskablePPO
    from result_schema import create_result
    from rl_hh.export_result import (
        _batch_to_schedule_entry, _restock_to_entry, _ups_to_entry, _normalize_gc_dict,
    )
    from master_eval import _build_export_params

    engine = SimulationEngine(params)
    model = MaskablePPO.load(str(model_path))
    strategy = PPOStrategy(data, model, deterministic=True)
    kpi, state = engine.run(strategy, list(eval_ups))

    schedule = [_batch_to_schedule_entry(b, params) for b in state.completed_batches]
    cancelled = [_batch_to_schedule_entry(b, params) for b in state.cancelled_batches]
    for c in cancelled:
        c["status"] = "cancelled"
    restocks = [_restock_to_entry(r) for r in state.completed_restocks]
    ups_entries = [_ups_to_entry(e) for e in state.ups_events_fired]

    result = create_result(
        metadata={
            "solver_engine": "simulation",
            "solver_name": "MaskedPPO (SB3) — C25 fix",
            "status": "Completed",
            "solve_time_sec": 7200.0,
            "input_dir": str(data.input_dir),
            "model_path": str(model_path),
            "training_started_at": "2026-04-20T19:33:26",
            "training_run_dir": str(C25_RUN),
            "notes": "C25: separate-networks + VecNormalize + completion-bonus + C22 tuning. See PPOtrainProgress.md",
        },
        experiment={
            "lambda_rate": float(params.get("ups_lambda", 0)),
            "mu_mean": float(params.get("ups_mu", 0)),
            "seed": EVAL_SEED,
        },
        kpi=kpi.to_dict(),
        schedule=schedule,
        cancelled_batches=cancelled,
        ups_events=ups_entries,
        restocks=restocks,
        parameters=_build_export_params(params),
    )
    return result, schedule, kpi.to_dict()


def format_schedule(schedule):
    """Format schedule as a readable table."""
    lines = []
    lines.append(f"{'#':>3} | {'roaster':<4} | {'start':>5} | {'end':>5} | {'dur':>3} | "
                 f"{'sku':<6} | {'line':<3} | {'setup':<5}")
    lines.append("-" * 75)
    for i, e in enumerate(sorted(schedule, key=lambda x: (x.get("roaster", ""), x.get("start", 0))), 1):
        start = int(e.get("start", 0))
        end = int(e.get("end", 0))
        lines.append(
            f"{i:>3} | {e.get('roaster',''):<4} | {start:>5} | {end:>5} | "
            f"{end-start:>3} | {e.get('sku',''):<6} | {e.get('line',''):<3} | "
            f"{e.get('setup',''):<5}"
        )
    return "\n".join(lines)


def main():
    from env.data_bridge import get_sim_params
    from env.ups_generator import generate_ups_events
    from PPOmask.Engine.data_loader import load_data

    params = get_sim_params()
    data = load_data()
    eval_ups = generate_ups_events(
        params.get("ups_lambda", 0), params.get("ups_mu", 0),
        seed=EVAL_SEED, shift_length=int(params["SL"]),
        roasters=list(params["roasters"]),
    )
    print(f"Eval seed={EVAL_SEED} | UPS events={len(eval_ups)}")
    for ev in eval_ups:
        print(f"  t={ev.t}, roaster={ev.roaster_id}, duration={ev.duration}")

    results = {}
    for tag, mdl in [("best", C25_BEST), ("final", C25_FINAL)]:
        print(f"\n{'='*80}\nRunning C25 {tag.upper()}: {mdl.name}\n{'='*80}")
        r, sched, kpi = run_one(mdl, tag, eval_ups, params, data)
        results[tag] = (r, sched, kpi)
        print(f"  Net Profit:  ${kpi['net_profit']:>12,.0f}")
        print(f"  Revenue:     ${kpi['total_revenue']:>12,.0f}")
        print(f"  PSC/NDG/BUS: {kpi['psc_count']}/{kpi['ndg_count']}/{kpi['busta_count']}")
        print(f"  Tardiness:   ${kpi['tard_cost']:>12,.0f}  (J1={kpi['tardiness_min'].get('J1',0)}, J2={kpi['tardiness_min'].get('J2',0)} min)")
        print(f"  Setup:       ${kpi['setup_cost']:>12,.0f}  ({kpi['setup_events']} events)")
        print(f"  Stockout:    ${kpi['stockout_cost']:>12,.0f}")
        print(f"  Idle:        ${kpi['idle_cost']:>12,.0f}")
        print(f"  Restocks:    {kpi['restock_count']}")

    # Pick winner for the Master_Evaluation report
    winner_tag = "best" if results["best"][2]["net_profit"] >= results["final"][2]["net_profit"] else "final"
    winner_result, winner_sched, winner_kpi = results[winner_tag]

    print(f"\n{'='*80}\nSchedule for C25 {winner_tag.upper()} @ seed={EVAL_SEED} (winner)\n{'='*80}")
    print(format_schedule(winner_sched))
    print(f"\nTotal batches: {len(winner_sched)}")

    # Write to target dir
    print(f"\n{'='*80}\nOverwriting PPO artifacts in {TARGET_DIR.name}\n{'='*80}")
    _save_json(results["best"][0], TARGET_DIR / "ppo_best_result.json")
    _save_json(results["final"][0], TARGET_DIR / "ppo_final_result.json")
    _generate_html(results["best"][0], TARGET_DIR / "ppo_best_report.html")
    _generate_html(results["final"][0], TARGET_DIR / "ppo_final_report.html")
    shutil.copy2(C25_FINAL, TARGET_DIR / "ppo_final_model.zip")
    shutil.copy2(C25_RUN / "meta.json", TARGET_DIR / "ppo_meta.json")
    print(f"  Wrote ppo_best_result.json / ppo_final_result.json (+reports)")
    print(f"  Wrote ppo_final_model.zip, ppo_meta.json")

    return winner_kpi, winner_tag


if __name__ == "__main__":
    main()
