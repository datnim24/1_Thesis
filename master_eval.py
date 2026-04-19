"""Master training & evaluation script for all 4 thesis methods.

Usage:
    python master_eval.py --name baseline_run
    python master_eval.py --name quick_test --time 600
    python master_eval.py --name overnight --time 32400

Methods (run sequentially):
  1. CP-SAT  — deterministic solver (finishes early if gap <0.5%)
  2. Q-Learning — tabular RL training + auto-eval
  3. MaskedPPO — deep RL training + auto-eval (auto-launches TensorBoard)
  4. RL-HH — Dueling DDQN hyper-heuristic + monitored training

Outputs per run (7 HTML reports):
  results/{timestamp}_{name}/
    cpsat_result.json + cpsat_report.html           (1 report)
    ql_best_result.json + ql_best_report.html       (2 reports)
    ql_final_result.json + ql_final_report.html
    ppo_best_result.json + ppo_best_report.html     (2 reports)
    ppo_final_result.json + ppo_final_report.html
    rlhh_best_result.json + rlhh_best_report.html   (2 reports)
    rlhh_final_result.json + rlhh_final_report.html
    Master_Evaluation_{timestamp}.md                 (summary)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_TIME_PER_METHOD = 9 * 3600  # 9 hours


# ============================================================================
# Utilities
# ============================================================================

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _elapsed_str(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _generate_html(result: dict, output_path: Path) -> Path:
    """Generate plot_result.py HTML from a universal-schema result dict."""
    from plot_result import _build_html
    html = _build_html(result, compare=None, offline=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _bar(pct: float, width: int = 40) -> str:
    filled = int(width * pct)
    return f"[{'#' * filled}{'-' * (width - filled)}] {pct*100:5.1f}%"


def _build_export_params(params: dict) -> dict:
    """Build the 'parameters' dict for result schema from sim params — NO hardcoded fallbacks."""
    from rl_hh.export_result import _normalize_gc_dict
    return {
        "SL": params["SL"],
        "sigma": params["sigma"],
        "DC": params["DC"],
        "max_rc": params["max_rc"],
        "safety_stock": params["safety_stock"],
        "rc_init": params["rc_init"],
        "restock_duration": params["restock_duration"],
        "restock_qty": params["restock_qty"],
        "roast_time_by_sku": params["roast_time_by_sku"],
        "consume_events": {k: list(v) for k, v in params["consume_events"].items()},
        "gc_capacity": _normalize_gc_dict(params["gc_capacity"]),
        "gc_init": _normalize_gc_dict(params["gc_init"]),
        "feasible_gc_pairs": [f"{k[0]}_{k[1]}" for k in params["gc_capacity"].keys()],
        "sku_revenue": {"PSC": params["rev_psc"], "NDG": params["rev_ndg"], "BUSTA": params["rev_busta"]},
        "c_tard": params["c_tard"],
        "c_stock": params["c_stock"],
        "c_idle": params["c_idle"],
        "c_over": params["c_over"],
        "c_setup": params["c_setup"],
    }


def _print_kpi(kpi: dict, label: str) -> None:
    print(f"\n  --- {label} ---")
    print(f"  Net Profit:  ${kpi.get('net_profit', 0):>12,.0f}")
    print(f"  Revenue:     ${kpi.get('total_revenue', 0):>12,.0f}")
    print(f"  PSC/NDG/BUS: {kpi.get('psc_count', 0)} / {kpi.get('ndg_count', 0)} / {kpi.get('busta_count', 0)}")
    print(f"  Tardiness:   ${kpi.get('tard_cost', 0):>12,.0f}")
    print(f"  Setup:       ${kpi.get('setup_cost', 0):>12,.0f}  ({kpi.get('setup_events', 0)} events)")
    print(f"  Stockout:    ${kpi.get('stockout_cost', 0):>12,.0f}")
    print(f"  Idle:        ${kpi.get('idle_cost', 0):>12,.0f}")
    print(f"  Restocks:    {kpi.get('restock_count', 0)}")


# ============================================================================
# Method 1: Reactive CP-SAT (handles UPS via offline re-optimisation)
# ============================================================================

def run_cpsat(out_dir: Path, time_budget: int, eval_ups: list = (), eval_seed: int = 42) -> dict | None:
    print(f"\n{'='*72}")
    print(f"  [1/4] PURE CP-SAT  (budget: {_elapsed_str(time_budget)})")
    print(f"{'='*72}")

    t0 = time.perf_counter()
    try:
        import logging as _logging
        for _name in ("cpsat_solver_v2", "cpsat_model_v2", "cpsat_pure_runner"):
            _lg = _logging.getLogger(_name)
            _lg.setLevel(_logging.INFO)
            if not _lg.handlers:
                _h = _logging.StreamHandler()
                _h.setFormatter(_logging.Formatter("  [CPSAT] %(message)s"))
                _lg.addHandler(_h)
                _lg.propagate = False

        from CPSAT_Pure import run_pure_cpsat
        from env.data_bridge import get_sim_params

        sim_params = get_sim_params()

        per_solve = max(60, int(time_budget))
        print(f"  Time limit: {_elapsed_str(per_solve)}")
        print(f"  UPS events (injected as planned downtime): {len(eval_ups)}")

        raw = run_pure_cpsat(
            time_limit_sec=per_solve,
            ups_events=list(eval_ups),
            input_dir="Input_data",
            num_workers=8,
        )
    except Exception as exc:
        print(f"  Pure CP-SAT FAILED: {exc}")
        import traceback; traceback.print_exc()
        return None

    elapsed = time.perf_counter() - t0

    _save_json(raw, out_dir / "cpsat_raw_result.json")

    from result_schema import create_result

    kpi_dict = {
        "net_profit": raw.get("net_profit", 0.0),
        "total_revenue": raw.get("total_revenue", 0.0),
        "total_costs": raw.get("total_costs", 0.0),
        "psc_count": raw.get("psc_count", 0),
        "ndg_count": raw.get("ndg_count", 0),
        "busta_count": raw.get("busta_count", 0),
        "total_batches": raw.get("total_batches", 0),
        "revenue_psc": raw.get("revenue_psc", 0.0),
        "revenue_ndg": raw.get("revenue_ndg", 0.0),
        "revenue_busta": raw.get("revenue_busta", 0.0),
        "tardiness_min": raw.get("tardiness_min", {}),
        "tard_cost": raw.get("tard_cost", 0.0),
        "setup_events": raw.get("setup_events", 0),
        "setup_cost": raw.get("setup_cost", 0.0),
        "stockout_events": {},
        "stockout_duration": {},
        "stockout_cost": 0.0,
        "idle_min": raw.get("idle_min", 0.0),
        "idle_cost": raw.get("idle_cost", 0.0),
        "over_min": raw.get("over_min", 0.0),
        "over_cost": raw.get("over_cost", 0.0),
        "restock_count": raw.get("restock_count", 0),
    }

    gap_pct = raw.get("gap_pct")
    status = raw.get("status", "?")
    notes = (
        f"Pure deterministic CP-SAT (UPS events baked in as planned downtime). "
        f"Status: {status}, "
        f"obj: ${raw.get('obj_value', 0):,.0f}, "
        f"bound: ${raw.get('best_bound', 0):,.0f}, "
        f"gap: {gap_pct}%, "
        f"UPS applied: {raw.get('ups_events_applied', 0)}, "
        f"solve time: {raw.get('solve_time', 0):.1f}s"
    )

    result = create_result(
        metadata={
            "solver_engine": "cpsat",
            "solver_name": "Pure CP-SAT v3 (OR-Tools, UPS-as-downtime)",
            "status": status,
            "solve_time_sec": elapsed,
            "input_dir": str(_ROOT / "Input_data"),
            "notes": notes,
        },
        experiment={
            "lambda_rate": float(sim_params.get("ups_lambda", 0)),
            "mu_mean": float(sim_params.get("ups_mu", 0)),
            "seed": eval_seed,
        },
        kpi=kpi_dict,
        schedule=raw.get("schedule", []),
        cancelled_batches=[],
        ups_events=[
            {"t": getattr(e, "t", None), "roaster_id": getattr(e, "roaster_id", None),
             "duration": getattr(e, "duration", None)}
            for e in (eval_ups or [])
        ],
        restocks=raw.get("restocks", []),
        parameters=_build_export_params(sim_params),
    )

    json_path = out_dir / "cpsat_result.json"
    html_path = out_dir / "cpsat_report.html"
    _save_json(result, json_path)
    _generate_html(result, html_path)

    _print_kpi(result["kpi"], f"Pure CP-SAT ({_elapsed_str(elapsed)})")
    print(f"  Objective: ${raw.get('obj_value', 0):>12,.0f}  |  Bound: ${raw.get('best_bound', 0):>12,.0f}")
    print(f"  Status: {status}  |  Gap: {gap_pct}%")
    print(f"  Report: {html_path}")

    return result


# ============================================================================
# Method 2: Q-Learning
# ============================================================================

def run_qlearning(out_dir: Path, time_budget: int, eval_ups: list = (), eval_seed: int = 42) -> tuple[dict | None, dict | None]:
    print(f"\n{'='*72}")
    print(f"  [2/4] Q-LEARNING  (budget: {_elapsed_str(time_budget)})")
    print(f"{'='*72}")

    t0 = time.perf_counter()
    try:
        from q_learning.q_learning_train import main as q_train_main
        q_train_main([
            "--name", "master",
            "--time", str(time_budget),
            "--checkpoint-interval", "5000",
            "--eval-episodes", "50",
        ])
    except Exception as exc:
        print(f"  Q-Learning training FAILED: {exc}")
        import traceback; traceback.print_exc()
        return None, None

    elapsed = time.perf_counter() - t0
    print(f"  Training took {_elapsed_str(elapsed)}")

    # Find the result folder
    ql_results = _ROOT / "q_learning" / "ql_results"
    if not ql_results.exists():
        print("  No Q-Learning results found")
        return None, None

    folders = sorted(
        [d for d in ql_results.iterdir() if d.is_dir() and "master" in d.name],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not folders:
        print("  No Q-Learning result folder found")
        return None, None

    run_dir = folders[0]
    print(f"  Result folder: {run_dir}")

    # Find Q-table and export best + final results
    from env.data_bridge import get_sim_params
    from env.simulation_engine import SimulationEngine
    from env.ups_generator import generate_ups_events
    from q_learning.q_strategy import QStrategy, load_q_table
    from result_schema import create_result

    params = get_sim_params()
    engine = SimulationEngine(params)

    qtable_path = list(run_dir.glob("q_table_*.pkl"))
    if not qtable_path:
        print("  No Q-table found")
        return None, None

    q_table = load_q_table(str(qtable_path[0]))
    strategy = QStrategy(params, q_table=q_table)

    # Training start from the Q-Learning run folder name: {DD_MM_YYYY_HHMM}_master
    _ql_train_start = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds")
    try:
        _parts = run_dir.name.split("_")
        if len(_parts) >= 4:
            _ql_train_start = datetime.strptime("_".join(_parts[:4]), "%d_%m_%Y_%H%M").isoformat(timespec="seconds")
    except Exception:
        pass
    _ql_model_path = str(qtable_path[0])

    # Use the shared UPS events for fair comparison
    ups = list(eval_ups)
    kpi, state = engine.run(strategy, ups)

    # Build result via the same pattern as RL-HH export
    from rl_hh.export_result import _batch_to_schedule_entry, _restock_to_entry, _ups_to_entry, _normalize_gc_dict

    schedule = [_batch_to_schedule_entry(b, params) for b in state.completed_batches]
    cancelled = [_batch_to_schedule_entry(b, params) for b in state.cancelled_batches]
    for c in cancelled:
        c["status"] = "cancelled"
    restocks = [_restock_to_entry(r) for r in state.completed_restocks]
    ups_entries = [_ups_to_entry(e) for e in state.ups_events_fired]

    kpi_dict = kpi.to_dict()
    ql_result = create_result(
        metadata={
            "solver_engine": "simulation",
            "solver_name": "Q-Learning (Tabular)",
            "status": "Completed",
            "solve_time_sec": elapsed,
            "input_dir": str(_ROOT / "Input_data"),
            "model_path": _ql_model_path,
            "training_started_at": _ql_train_start,
            "training_run_dir": str(run_dir),
        },
        experiment={"lambda_rate": float(params.get("ups_lambda", 0)), "mu_mean": float(params.get("ups_mu", 0)), "seed": eval_seed},
        kpi=kpi_dict,
        schedule=schedule,
        cancelled_batches=cancelled,
        ups_events=ups_entries,
        restocks=restocks,
        parameters=_build_export_params(params),
    )

    # Save as both best and final (Q-learning has no separate best checkpoint)
    _save_json(ql_result, out_dir / "ql_best_result.json")
    _save_json(ql_result, out_dir / "ql_final_result.json")
    _generate_html(ql_result, out_dir / "ql_best_report.html")
    _generate_html(ql_result, out_dir / "ql_final_report.html")

    _print_kpi(ql_result["kpi"], f"Q-Learning ({_elapsed_str(elapsed)})")

    # Copy Q-table and HTML report
    for src in run_dir.glob("*.pkl"):
        shutil.copy2(src, out_dir / src.name)
    for src in run_dir.glob("*.html"):
        shutil.copy2(src, out_dir / f"ql_training_report_{src.name}")

    return ql_result, ql_result


# ============================================================================
# Method 3: MaskedPPO
# ============================================================================

def run_ppo(out_dir: Path, time_budget: int, eval_ups: list = (), eval_seed: int = 42) -> tuple[dict | None, dict | None]:
    print(f"\n{'='*72}")
    print(f"  [3/4] MASKED PPO  (budget: {_elapsed_str(time_budget)})")
    print(f"{'='*72}")

    t0 = time.perf_counter()
    try:
        from PPOmask.train_maskedppo import main as ppo_train_main
        # PPO config: restore C12 winning setup + reward normalization for
        # the new 100k MTO-skip-penalty scale (terminal rewards reach -1M+).
        # Without these, the agent collapses to WAIT within ~200 episodes.
        # --timesteps set very high so --time (wall-clock) is the binding budget.
        run_dir = ppo_train_main([
            "--time", str(time_budget),
            "--timesteps", "100000000",
            "--n-envs", "8",
            "--subproc",
            "--run-name", "master",
            "--eval-episodes", "50",
            "--no-open-report",
            "--progress-print-seconds", "60",
            "--seed", "300",
            "--learning-rate", "4e-4",
            "--rc-maintenance-bonus", "50",
            "--normalize-reward",
            "--n-epochs", "3",
            "--target-kl", "0.03",
            "--ent-coef", "0.02",
        ])
    except Exception as exc:
        print(f"  PPO training FAILED: {exc}")
        import traceback; traceback.print_exc()
        return None, None

    elapsed = time.perf_counter() - t0
    print(f"  Training took {_elapsed_str(elapsed)}")

    if run_dir is None or not Path(run_dir).exists():
        print("  No PPO run directory found")
        return None, None

    run_dir = Path(run_dir)

    # Find final model and export result
    final_model = run_dir / "checkpoints" / "final_model.zip"
    if not final_model.exists():
        print(f"  No final model at {final_model}")
        return None, None

    # Run a greedy evaluation episode to get a plottable result
    from env.data_bridge import get_sim_params
    from env.simulation_engine import SimulationEngine
    from env.ups_generator import generate_ups_events
    from PPOmask.Engine.data_loader import load_data
    from PPOmask.Engine.ppo_strategy import PPOStrategy
    from sb3_contrib import MaskablePPO
    from result_schema import create_result
    from rl_hh.export_result import _batch_to_schedule_entry, _restock_to_entry, _ups_to_entry, _normalize_gc_dict

    data = load_data()
    params = data.to_env_params()
    engine = SimulationEngine(params)

    model = MaskablePPO.load(str(final_model))
    strategy = PPOStrategy(data, model, deterministic=True)

    # Training start from the PPO run folder name: {YYYYMMDD_HHMMSS}_master
    _ppo_train_start = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds")
    try:
        _prefix = "_".join(run_dir.name.split("_")[:2])
        _ppo_train_start = datetime.strptime(_prefix, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
    except Exception:
        pass
    _ppo_model_path = str(final_model)

    # Use the shared UPS events for fair comparison
    ups = list(eval_ups)
    kpi, state = engine.run(strategy, ups)

    schedule = [_batch_to_schedule_entry(b, params) for b in state.completed_batches]
    cancelled = [_batch_to_schedule_entry(b, params) for b in state.cancelled_batches]
    for c in cancelled:
        c["status"] = "cancelled"
    restocks = [_restock_to_entry(r) for r in state.completed_restocks]
    ups_entries = [_ups_to_entry(e) for e in state.ups_events_fired]

    kpi_dict = kpi.to_dict()
    ppo_result = create_result(
        metadata={
            "solver_engine": "simulation",
            "solver_name": "MaskedPPO (SB3)",
            "status": "Completed",
            "solve_time_sec": elapsed,
            "input_dir": str(data.input_dir),
            "model_path": _ppo_model_path,
            "training_started_at": _ppo_train_start,
            "training_run_dir": str(run_dir),
        },
        experiment={"lambda_rate": data.ups_lambda, "mu_mean": data.ups_mu, "seed": eval_seed},
        kpi=kpi_dict,
        schedule=schedule,
        cancelled_batches=cancelled,
        ups_events=ups_entries,
        restocks=restocks,
        parameters=_build_export_params(params),
    )

    _save_json(ppo_result, out_dir / "ppo_best_result.json")
    _save_json(ppo_result, out_dir / "ppo_final_result.json")
    _generate_html(ppo_result, out_dir / "ppo_best_report.html")
    _generate_html(ppo_result, out_dir / "ppo_final_report.html")

    _print_kpi(ppo_result["kpi"], f"MaskedPPO ({_elapsed_str(elapsed)})")

    # Copy model and meta
    shutil.copy2(final_model, out_dir / "ppo_final_model.zip")
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        shutil.copy2(meta_path, out_dir / "ppo_meta.json")

    # Copy training report if it exists
    for html_file in run_dir.glob("*_report.html"):
        shutil.copy2(html_file, out_dir / f"ppo_training_report.html")
        break

    return ppo_result, ppo_result


# ============================================================================
# Method 4: RL-HH (Dueling DDQN)
# ============================================================================

def run_rlhh(out_dir: Path, time_budget: int, eval_ups: list = (), eval_seed: int = 42) -> tuple[dict | None, dict | None]:
    print(f"\n{'='*72}")
    print(f"  [4/4] RL-HH (Dueling DDQN)  (budget: {_elapsed_str(time_budget)})")
    print(f"{'='*72}")

    # Configure RL-HH time budget
    import rl_hh.train_monitored as rlhh_mod
    rlhh_mod.MAX_WALL_HOURS = time_budget / 3600.0

    _rlhh_train_start = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    try:
        rlhh_mod.train_monitored()
    except Exception as exc:
        print(f"  RL-HH training FAILED: {exc}")
        import traceback; traceback.print_exc()
        return None, None

    elapsed = time.perf_counter() - t0

    # Export best and final results
    from rl_hh.export_result import run_and_export

    best_result, final_result = None, None

    best_ckpt = _ROOT / "rl_hh" / "outputs" / "rlhh_overall_best.pt"
    if best_ckpt.exists():
        best_result = run_and_export(
            str(best_ckpt), str(out_dir / "rlhh_best_result.json"),
            seed=eval_seed, training_started_at=_rlhh_train_start,
        )
        _generate_html(best_result, out_dir / "rlhh_best_report.html")
        _print_kpi(best_result["kpi"], f"RL-HH Best ({_elapsed_str(elapsed)})")
        shutil.copy2(best_ckpt, out_dir / "rlhh_best.pt")

    # Find final checkpoint
    rlhh_out = _ROOT / "rl_hh" / "outputs"
    final_ckpts = sorted(rlhh_out.glob("rlhh_cycle*_final.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if final_ckpts:
        final_result = run_and_export(
            str(final_ckpts[0]), str(out_dir / "rlhh_final_result.json"),
            seed=eval_seed, training_started_at=_rlhh_train_start,
        )
        _generate_html(final_result, out_dir / "rlhh_final_report.html")
        if best_result is None:
            _print_kpi(final_result["kpi"], f"RL-HH Final ({_elapsed_str(elapsed)})")
        shutil.copy2(final_ckpts[0], out_dir / "rlhh_final.pt")

    # Copy monitored log
    log_path = rlhh_out / "monitored_log.csv"
    if log_path.exists():
        shutil.copy2(log_path, out_dir / "rlhh_monitored_log.csv")

    return best_result, final_result


# ============================================================================
# Master summary markdown
# ============================================================================

def generate_summary_md(
    out_dir: Path,
    name: str,
    results: dict[str, dict | None],
    wall_times: dict[str, float],
) -> Path:
    ts = _ts()
    md_path = out_dir / f"Master_Evaluation_{ts}.md"

    lines = [
        f"# Master Evaluation: {name}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        f"- Run name: `{name}`",
        f"- Time per method: {_elapsed_str(max(wall_times.values()) if wall_times else 0)}",
        "",
        "## Results Comparison",
        "",
        "| Metric | CP-SAT | Q-Learning | MaskedPPO | RL-HH |",
        "|--------|--------|------------|-----------|-------|",
    ]

    methods = ["cpsat", "ql_best", "ppo_best", "rlhh_best"]
    labels = ["CP-SAT", "Q-Learning", "MaskedPPO", "RL-HH"]

    def _v(method: str, key: str, fmt: str = "${:,.0f}") -> str:
        r = results.get(method)
        if r is None:
            return "N/A"
        kpi = r.get("kpi", {})
        val = kpi.get(key, None)
        if val is None:
            return "N/A"
        return fmt.format(val)

    def _vi(method: str, key: str) -> str:
        r = results.get(method)
        if r is None:
            return "N/A"
        kpi = r.get("kpi", {})
        val = kpi.get(key, 0)
        return str(int(val))

    rows = [
        ("**Net Profit**", "net_profit", "${:,.0f}"),
        ("Revenue", "total_revenue", "${:,.0f}"),
        ("PSC Batches", "psc_count", None),
        ("NDG Batches", "ndg_count", None),
        ("BUSTA Batches", "busta_count", None),
        ("Total Batches", "total_batches", None),
        ("Tardiness Cost", "tard_cost", "${:,.0f}"),
        ("Setup Cost", "setup_cost", "${:,.0f}"),
        ("Stockout Cost", "stockout_cost", "${:,.0f}"),
        ("Idle Cost", "idle_cost", "${:,.0f}"),
        ("Restocks", "restock_count", None),
    ]

    for label, key, fmt in rows:
        cols = []
        for m in methods:
            if fmt:
                cols.append(_v(m, key, fmt))
            else:
                cols.append(_vi(m, key))
        lines.append(f"| {label} | {' | '.join(cols)} |")

    # Wall times
    lines.append("")
    lines.append("## Training Times")
    lines.append("")
    wt_labels = {"cpsat": "CP-SAT", "qlearning": "Q-Learning", "ppo": "MaskedPPO", "rlhh": "RL-HH"}
    for key, label in wt_labels.items():
        if key in wall_times:
            lines.append(f"- **{label}**: {_elapsed_str(wall_times[key])}")

    # Reports
    lines.append("")
    lines.append("## Reports")
    lines.append("")
    for f in sorted(out_dir.glob("*_report.html")):
        lines.append(f"- [{f.name}]({f.name})")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by master_eval.py*")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master training & evaluation for all 4 thesis methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", "-n", required=True, help="Run name (used in output folder)")
    parser.add_argument("--time", "-t", type=int, default=DEFAULT_TIME_PER_METHOD,
                        help=f"Time budget per method in seconds (default: {DEFAULT_TIME_PER_METHOD}s = 9h)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for UPS generation — same across all methods (default: 42)")
    parser.add_argument("--skip", nargs="*", default=[], choices=["cpsat", "ql", "ppo", "rlhh"],
                        help="Skip specific methods")
    args = parser.parse_args()

    ts = _ts()
    out_dir = _ROOT / "results" / f"{ts}_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-generate UPS events ONCE so all methods face identical disruptions
    from env.data_bridge import get_sim_params
    from env.ups_generator import generate_ups_events
    _params = get_sim_params()
    eval_ups = generate_ups_events(
        _params.get("ups_lambda", 0), _params.get("ups_mu", 0),
        seed=args.seed, shift_length=int(_params["SL"]),
        roasters=list(_params["roasters"]),
    )
    print(f"\n{'#'*72}")
    print(f"  MASTER EVALUATION: {args.name}")
    print(f"  Time per method: {_elapsed_str(args.time)}")
    print(f"  Eval seed: {args.seed}  |  UPS events: {len(eval_ups)}")
    for ev in eval_ups:
        print(f"    t={ev.t}, roaster={ev.roaster_id}, duration={ev.duration}")
    print(f"  Output: {out_dir}")
    print(f"{'#'*72}")

    results: dict[str, dict | None] = {}
    wall_times: dict[str, float] = {}
    html_reports: list[Path] = []
    global_t0 = time.perf_counter()

    # 1 & 2. CP-SAT and Q-Learning in parallel (Q-L starts 60s after CP-SAT).
    # Both use minimal CPU individually; running them concurrently cuts wall
    # time without meaningful contention — CP-SAT's solve is C++ (releases the
    # GIL) and Q-Learning is Python-only but lightweight.
    import threading as _threading
    cpsat_wanted = "cpsat" not in args.skip
    ql_wanted = "ql" not in args.skip
    cpsat_box: dict = {}
    ql_box: dict = {}

    def _cpsat_task() -> None:
        _t0 = time.perf_counter()
        cpsat_box["result"] = run_cpsat(out_dir, args.time, eval_ups, args.seed)
        cpsat_box["elapsed"] = time.perf_counter() - _t0

    def _ql_task() -> None:
        _t0 = time.perf_counter()
        best, final = run_qlearning(out_dir, args.time, eval_ups, args.seed)
        ql_box["best"], ql_box["final"] = best, final
        ql_box["elapsed"] = time.perf_counter() - _t0

    threads: list[_threading.Thread] = []
    if cpsat_wanted:
        t = _threading.Thread(target=_cpsat_task, name="cpsat", daemon=False)
        t.start()
        threads.append(t)
        if ql_wanted:
            print(f"\n  [Parallel phase] CP-SAT started; Q-Learning will start in 60s ...")
            time.sleep(60)
    if ql_wanted:
        t = _threading.Thread(target=_ql_task, name="qlearning", daemon=False)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if cpsat_wanted:
        wall_times["cpsat"] = cpsat_box.get("elapsed", 0.0)
        results["cpsat"] = cpsat_box.get("result")
        if (out_dir / "cpsat_report.html").exists():
            html_reports.append(out_dir / "cpsat_report.html")
    if ql_wanted:
        wall_times["qlearning"] = ql_box.get("elapsed", 0.0)
        results["ql_best"] = ql_box.get("best")
        results["ql_final"] = ql_box.get("final")
        for rpt in ["ql_best_report.html", "ql_final_report.html"]:
            if (out_dir / rpt).exists():
                html_reports.append(out_dir / rpt)

    # 3. MaskedPPO
    if "ppo" not in args.skip:
        t0 = time.perf_counter()
        ppo_best, ppo_final = run_ppo(out_dir, args.time, eval_ups, args.seed)
        wall_times["ppo"] = time.perf_counter() - t0
        results["ppo_best"] = ppo_best
        results["ppo_final"] = ppo_final
        for rpt in ["ppo_best_report.html", "ppo_final_report.html"]:
            if (out_dir / rpt).exists():
                html_reports.append(out_dir / rpt)

    # 4. RL-HH
    if "rlhh" not in args.skip:
        t0 = time.perf_counter()
        rlhh_best, rlhh_final = run_rlhh(out_dir, args.time, eval_ups, args.seed)
        wall_times["rlhh"] = time.perf_counter() - t0
        results["rlhh_best"] = rlhh_best
        results["rlhh_final"] = rlhh_final
        for rpt in ["rlhh_best_report.html", "rlhh_final_report.html"]:
            if (out_dir / rpt).exists():
                html_reports.append(out_dir / rpt)

    total_elapsed = time.perf_counter() - global_t0

    # Generate summary
    md_path = generate_summary_md(out_dir, args.name, results, wall_times)

    # Final summary
    print(f"\n{'#'*72}")
    print(f"  MASTER EVALUATION COMPLETE")
    print(f"  Total wall time: {_elapsed_str(total_elapsed)}")
    print(f"  Output folder:   {out_dir}")
    print(f"{'#'*72}")

    print(f"\n  {'Method':<15} {'Net Profit':>15} {'Time':>12}")
    print(f"  {'-'*42}")
    profit_map = {
        "CP-SAT": "cpsat", "Q-Learning": "ql_best",
        "MaskedPPO": "ppo_best", "RL-HH": "rlhh_best",
    }
    for label, key in profit_map.items():
        r = results.get(key)
        profit_str = f"${r['kpi']['net_profit']:,.0f}" if r else "N/A"
        wt_key = {"CP-SAT": "cpsat", "Q-Learning": "qlearning", "MaskedPPO": "ppo", "RL-HH": "rlhh"}[label]
        time_str = _elapsed_str(wall_times.get(wt_key, 0))
        print(f"  {label:<15} {profit_str:>15} {time_str:>12}")

    print(f"\n  Summary: {md_path}")
    print(f"  Reports: {len(html_reports)} HTML files")

    # Open all reports in browser
    for report in html_reports:
        try:
            webbrowser.open(report.resolve().as_uri())
        except Exception:
            pass

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
