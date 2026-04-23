"""Multi-method comparison: CP-SAT vs Q-Learning vs RL-HH vs MaskedPPO.

Runs each method on N shared seeds (same UPS realization per seed) and builds
a single ultra-detailed HTML report with:
  - Aggregate comparison table across all methods
  - Per-seed winner table
  - Hyperparameters for each method (collapsible)
  - All plots (Gantt, RC/GC stock, Restock, Utilization, ...) as collapsible
    iframes per (seed, method)
  - Input parameters table at the bottom

DESIGN GOAL: easy to modify for sensitivity analysis. Edit the CONFIG block at
the top of this file (or pass CLI flags) — no deep module dives needed.

Usage:
    python scripts/compare_methods.py                   # default: N=10
    python scripts/compare_methods.py --n-runs 20
    python scripts/compare_methods.py --seed-start 100 --n-runs 5
    python scripts/compare_methods.py --skip cpsat      # skip slow solver
    python scripts/compare_methods.py --ppo-model <path>
    python scripts/compare_methods.py --ups-lambda 3 --ups-mu 15  # override UPS
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# =============================================================================
# CONFIG — edit here for sensitivity analysis
# =============================================================================

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CONFIG = {
    # Run parameters
    "n_runs": 10,
    "seed_start": 1,
    "random_seeds": False,       # True = sample N seeds via Random(meta_seed); False = sequential start..start+N-1
    "meta_seed": 42,             # reproducibility when random_seeds=True
    "seed_pool_max": 100000,     # sample seeds from range(1, seed_pool_max)
    "input_dir": _ROOT / "Input_data",
    "output_root": _ROOT / "results" / "method_comparison",

    # Method toggles — set False to skip a method
    "methods_enabled": {
        "cpsat": True,
        "ql": True,
        "rlhh": True,
        "ppo": True,
    },

    # Model overrides — None = auto-discover latest-best
    "model_overrides": {
        "ql_qtable": None,      # path to q_table_master.pkl
        "rlhh_ckpt": None,      # path to rlhh_best.pt
        "ppo_model": None,      # path to best_training_profit_model.zip
    },

    # UPS overrides — None = use values from Input_data/shift_parameters.csv
    "ups_overrides": {
        "lambda": None,
        "mu": None,
    },

    # CP-SAT solver config
    "cpsat": {
        "time_limit_sec": 300,   # 5 min/seed — heavy-UPS cases need it
        "num_workers": 8,
        "mip_gap": None,         # None = default from Input_data
    },

    # Report options
    "save_per_seed_html": True,   # saves individual plot HTMLs (needed for iframe embeds)
    "embed_plots_in_main": True,  # adds collapsible iframe sections
    "report_title": "Scheduling Method Comparison Report",
}


# =============================================================================
# ARG PARSING — CLI overrides CONFIG
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CP-SAT / Q-Learning / RL-HH / MaskedPPO")
    p.add_argument("--n-runs", type=int, default=CONFIG["n_runs"],
                   help=f"Number of seeds to evaluate (default {CONFIG['n_runs']})")
    p.add_argument("--seed-start", type=int, default=CONFIG["seed_start"],
                   help="Starting seed for sequential mode (seeds = start..start+n_runs-1)")
    p.add_argument("--random", action="store_true",
                   help="Use random seeds (reproducible via --meta-seed) instead of sequential")
    p.add_argument("--meta-seed", type=int, default=CONFIG["meta_seed"],
                   help=f"Meta-seed for --random reproducibility (default {CONFIG['meta_seed']})")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["cpsat", "ql", "rlhh", "ppo"],
                   help="Methods to skip (space separated)")
    p.add_argument("--ppo-model", type=str, default=None,
                   help="Override path to PPO .zip model")
    p.add_argument("--ql-qtable", type=str, default=None,
                   help="Override path to Q-Learning q_table_master.pkl")
    p.add_argument("--rlhh-ckpt", type=str, default=None,
                   help="Override path to RL-HH rlhh_best.pt")
    p.add_argument("--ups-lambda", type=float, default=None,
                   help="Override UPS Poisson rate (events/shift)")
    p.add_argument("--ups-mu", type=float, default=None,
                   help="Override UPS mean repair duration (min)")
    p.add_argument("--cpsat-time", type=int, default=CONFIG["cpsat"]["time_limit_sec"],
                   help=f"CP-SAT time budget per seed (default {CONFIG['cpsat']['time_limit_sec']}s)")
    p.add_argument("--output", type=str, default=None,
                   help="Custom output directory (default: results/method_comparison/<timestamp>)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip per-seed plot HTML generation (faster, lighter report)")
    p.add_argument("--rescue", type=str, default=None,
                   help="Re-render report from a previous run's saved per-seed JSONs "
                        "(pass the run directory, e.g. results/method_comparison/20260421_235532)")
    return p.parse_args()


# =============================================================================
# MODEL DISCOVERY — auto-find "latest best" for each method
# =============================================================================

def find_latest_ql_qtable(root: Path) -> Path | None:
    """Scan q_learning/ql_results/*/q_table_master.pkl. Prefer the run with the most
    training episodes (epNNNN in the dir name) — long training generalizes better
    than a short run with a high training-profit label."""
    candidates = list((root / "q_learning" / "ql_results").glob("*/q_table_master.pkl"))
    if not candidates:
        return None
    def _episodes(p: Path) -> int:
        for tok in p.parent.name.split("_"):
            if tok.startswith("ep") and tok[2:].isdigit():
                return int(tok[2:])
        return 0
    candidates.sort(key=_episodes, reverse=True)
    return candidates[0]


def find_latest_rlhh_ckpt(root: Path) -> Path | None:
    """Prefer rlhh_overall_best.pt, then rlhh_best.pt (most recent mtime)."""
    out_dir = root / "rl_hh" / "outputs"
    for name in ("rlhh_overall_best.pt", "rlhh_best.pt"):
        p = out_dir / name
        if p.exists():
            return p
    # Fallback: any *.pt sorted by mtime
    candidates = sorted(out_dir.glob("rlhh*best*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_latest_ppo_model(root: Path) -> Path | None:
    """Scan PPOmask/outputs/*/checkpoints/best_training_profit_model.zip — latest mtime."""
    candidates = list((root / "PPOmask" / "outputs").glob("*/checkpoints/best_training_profit_model.zip"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.parent.parent.stat().st_mtime, reverse=True)
    return candidates[0]


# =============================================================================
# HYPERPARAMETER EXTRACTION
# =============================================================================

def get_ppo_hyperparams(ppo_zip_path: Path) -> dict[str, Any]:
    meta_path = ppo_zip_path.parent.parent / "meta.json"
    if not meta_path.exists():
        return {"model_path": str(ppo_zip_path), "note": "meta.json not found"}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ql_hyperparams(qtable_path: Path) -> dict[str, Any]:
    meta_path = qtable_path.parent / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback: parse from directory name (e.g. ep1051788_a01000_g09900_master_profit211312)
    name = qtable_path.parent.name
    info = {"model_path": str(qtable_path), "run_name": name}
    for token in name.split("_"):
        if token.startswith("ep"):
            info["episodes"] = token[2:]
        elif token.startswith("a") and len(token) > 1 and token[1:].isdigit():
            info["alpha_encoded"] = token[1:]
        elif token.startswith("g") and len(token) > 1 and token[1:].isdigit():
            info["gamma_encoded"] = token[1:]
        elif token.startswith("profit"):
            info["trained_profit"] = token[6:]
    return info


def get_rlhh_hyperparams(rlhh_ckpt: Path) -> dict[str, Any]:
    """RL-HH has no meta.json; probe the .pt for embedded config if possible."""
    info: dict[str, Any] = {"model_path": str(rlhh_ckpt)}
    try:
        import torch
        ckpt = torch.load(str(rlhh_ckpt), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for k in ("config", "hyperparams", "episode", "epsilon", "best_profit",
                      "avg_profit", "network", "optimizer_state_dict"):
                if k in ckpt:
                    if k == "optimizer_state_dict":
                        info["has_optimizer_state"] = True
                    elif k == "network":
                        info["has_network"] = True
                    else:
                        info[k] = ckpt[k] if not isinstance(ckpt[k], dict) else {
                            kk: vv for kk, vv in ckpt[k].items()
                            if not hasattr(vv, "shape")  # skip tensors
                        }
    except Exception as exc:
        info["probe_error"] = str(exc)
    return info


def get_cpsat_hyperparams(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "solver": "OR-Tools CP-SAT (Pure v3)",
        "time_limit_sec": cfg.get("time_limit_sec"),
        "num_workers": cfg.get("num_workers"),
        "mip_gap": cfg.get("mip_gap"),
        "description": "Deterministic MILP/CP-SAT with UPS injected as planned downtime.",
    }


# =============================================================================
# METHOD RUNNERS — uniform signature: (seed, ups, params, data, engine, ...) -> result dict
# =============================================================================

def run_cpsat(seed: int, ups: list, params: dict, input_dir: Path, cpsat_cfg: dict) -> dict:
    from CPSAT_Pure import run_pure_cpsat
    raw = run_pure_cpsat(
        time_limit_sec=cpsat_cfg["time_limit_sec"],
        ups_events=list(ups),
        input_dir=str(input_dir),
        num_workers=cpsat_cfg["num_workers"],
        mip_gap=cpsat_cfg.get("mip_gap"),
    )
    return _build_cpsat_result(raw, params, seed, ups, input_dir)


def run_ql(seed: int, ups: list, params: dict, data: Any, engine: Any, qtable_path: Path) -> tuple:
    from q_learning.q_strategy import QStrategy, load_q_table
    strategy = QStrategy(params, q_table=load_q_table(str(qtable_path)))
    return engine.run(strategy, list(ups))


def run_rlhh(seed: int, ups: list, params: dict, data: Any, engine: Any, ckpt_path: Path) -> tuple:
    from rl_hh.meta_agent import DuelingDDQNAgent
    from rl_hh.rl_hh_strategy import RLHHStrategy
    agent = DuelingDDQNAgent()
    agent.load_checkpoint(str(ckpt_path))
    agent.epsilon = 0.0
    strategy = RLHHStrategy(agent, data, training=False)
    return engine.run(strategy, list(ups))


def run_ppo(seed: int, ups: list, params: dict, data: Any, engine: Any, model_path: Path) -> tuple:
    from PPOmask.Engine.ppo_strategy import PPOStrategy
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(model_path))
    strategy = PPOStrategy(data, model, deterministic=True)
    return engine.run(strategy, list(ups))


# =============================================================================
# RESULT BUILDING (uniform schema for plot_result._build_html)
# =============================================================================

def _build_sim_result(solver_name: str, engine_tag: str, kpi, state, params, data,
                      seed: int, model_path: str) -> dict:
    """Convert engine result (KPITracker, SimulationState) to result_schema dict."""
    from result_schema import create_result
    from rl_hh.export_result import _batch_to_schedule_entry, _restock_to_entry, _ups_to_entry
    from master_eval import _build_export_params

    schedule = [_batch_to_schedule_entry(b, params) for b in state.completed_batches]
    cancelled = [_batch_to_schedule_entry(b, params) for b in state.cancelled_batches]
    for c in cancelled:
        c["status"] = "cancelled"
    restocks = [_restock_to_entry(r) for r in state.completed_restocks]
    ups_entries = [_ups_to_entry(e) for e in state.ups_events_fired]

    return create_result(
        metadata={
            "solver_engine": engine_tag,
            "solver_name": solver_name,
            "status": "Completed",
            "solve_time_sec": 0.0,
            "input_dir": str(data.input_dir),
            "model_path": str(model_path),
            "notes": f"compare_methods.py run, seed={seed}.",
        },
        experiment={
            "lambda_rate": float(params.get("ups_lambda", 0)),
            "mu_mean": float(params.get("ups_mu", 0)),
            "seed": seed,
        },
        kpi=kpi.to_dict(),
        schedule=schedule,
        cancelled_batches=cancelled,
        ups_events=ups_entries,
        restocks=restocks,
        parameters=_build_export_params(params),
    )


def _build_cpsat_result(raw: dict, params: dict, seed: int, ups: list, input_dir: Path) -> dict:
    """Convert CP-SAT raw solver output to result_schema dict (mirrors master_eval.run_cpsat)."""
    from result_schema import create_result
    from master_eval import _build_export_params

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
    notes = (f"Pure CP-SAT v3. Status: {raw.get('status', '?')}, "
             f"obj: ${raw.get('obj_value', 0):,.0f}, gap: {raw.get('gap_pct')}%, "
             f"UPS applied: {raw.get('ups_events_applied', 0)}, "
             f"solve: {raw.get('solve_time', 0):.1f}s")
    return create_result(
        metadata={
            "solver_engine": "cpsat",
            "solver_name": "CP-SAT (Pure v3)",
            "status": raw.get("status", "?"),
            "solve_time_sec": float(raw.get("solve_time", 0.0)),
            "input_dir": str(input_dir),
            "model_path": "(deterministic solver — no learned model)",
            "notes": notes,
        },
        experiment={
            "lambda_rate": float(params.get("ups_lambda", 0)),
            "mu_mean": float(params.get("ups_mu", 0)),
            "seed": seed,
        },
        kpi=kpi_dict,
        schedule=raw.get("schedule", []),
        cancelled_batches=[],
        ups_events=[{"t": getattr(e, "t", None), "roaster_id": getattr(e, "roaster_id", None),
                     "duration": getattr(e, "duration", None)} for e in ups],
        restocks=raw.get("restocks", []),
        parameters=_build_export_params(params),
    )


def _summarize(kpi_dict: dict) -> dict:
    """Flat dict of comparable metrics — used for aggregation tables."""
    tm = kpi_dict.get("tardiness_min", {})
    return {
        "net_profit": float(kpi_dict.get("net_profit", 0)),
        "revenue": float(kpi_dict.get("total_revenue", 0)),
        "psc": int(kpi_dict.get("psc_count", 0)),
        "ndg": int(kpi_dict.get("ndg_count", 0)),
        "busta": int(kpi_dict.get("busta_count", 0)),
        "tard_cost": float(kpi_dict.get("tard_cost", 0)),
        "setup_cost": float(kpi_dict.get("setup_cost", 0)),
        "setup_events": int(kpi_dict.get("setup_events", 0)),
        "idle_cost": float(kpi_dict.get("idle_cost", 0)),
        "stockout_cost": float(kpi_dict.get("stockout_cost", 0)),
        "over_cost": float(kpi_dict.get("over_cost", 0)),
        "restocks": int(kpi_dict.get("restock_count", 0)),
        "j1_late": float(tm.get("J1", 0)),
        "j2_late": float(tm.get("J2", 0)),
    }


# =============================================================================
# HTML REPORT RENDERING
# =============================================================================

def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "avg": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def _render_summary_table(per_method_runs: dict[str, list[dict]],
                          method_labels: dict[str, str]) -> str:
    """Build the main comparison table across all methods."""
    metric_rows = [
        ("Net Profit ($)", "net_profit", _fmt_money),
        ("Revenue ($)", "revenue", _fmt_money),
        ("Tardiness Cost ($)", "tard_cost", _fmt_money),
        ("Setup Cost ($)", "setup_cost", _fmt_money),
        ("Idle Cost ($)", "idle_cost", _fmt_money),
        ("Stockout Cost ($)", "stockout_cost", _fmt_money),
        ("Over-capacity Cost ($)", "over_cost", _fmt_money),
        ("PSC Batches", "psc", lambda v: f"{v:.1f}"),
        ("NDG Batches", "ndg", lambda v: f"{v:.1f}"),
        ("BUSTA Batches", "busta", lambda v: f"{v:.1f}"),
        ("Setup Events", "setup_events", lambda v: f"{v:.1f}"),
        ("Restock Operations", "restocks", lambda v: f"{v:.1f}"),
        ("J1 Tardiness (min)", "j1_late", lambda v: f"{v:.1f}"),
        ("J2 Tardiness (min)", "j2_late", lambda v: f"{v:.1f}"),
    ]

    rows_html = []
    for label, field, fmt in metric_rows:
        cells = [f"<td><strong>{html.escape(label)}</strong></td>"]
        for tag in per_method_runs:
            vals = [r[field] for r in per_method_runs[tag]]
            s = _stats(vals)
            cells.append(
                f"<td>{fmt(s['avg'])}<br>"
                f"<span class='muted'>σ={fmt(s['std'])} &middot; "
                f"min={fmt(s['min'])} &middot; max={fmt(s['max'])}</span></td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    headers = "".join(
        f"<th>{html.escape(method_labels[tag])}</th>" for tag in per_method_runs
    )
    return (
        f"<table class='summary'>"
        f"<thead><tr><th>Metric (avg across N runs)</th>{headers}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _index_by_seed(per_method_runs: dict[str, list[dict]]) -> dict[str, dict[int, dict]]:
    """Reshape {method: [run, ...]} → {method: {seed: run, ...}} for seed-safe lookups."""
    return {t: {r["seed"]: r for r in runs} for t, runs in per_method_runs.items()}


def _render_per_seed_table(per_method_runs: dict[str, list[dict]],
                           method_labels: dict[str, str],
                           seeds: list[int]) -> str:
    """Per-seed net profit + winner. Resilient to methods that skipped a seed (e.g. CP-SAT timeout)."""
    by_seed = _index_by_seed(per_method_runs)
    headers = "".join(f"<th>{html.escape(method_labels[t])}</th>" for t in per_method_runs)
    rows = []
    wins = {t: 0 for t in per_method_runs}
    for seed in seeds:
        cells = [f"<td>{seed}</td>"]
        nets = {t: by_seed[t][seed]["net_profit"]
                for t in per_method_runs if seed in by_seed[t]}
        winner = max(nets, key=nets.get) if nets else None
        if winner is not None:
            wins[winner] += 1
        for t in per_method_runs:
            if seed in by_seed[t]:
                css = "winner" if t == winner else ""
                cells.append(f"<td class='{css}'>{_fmt_money(by_seed[t][seed]['net_profit'])}</td>")
            else:
                cells.append("<td class='failed'>—</td>")
        winner_label = html.escape(method_labels[winner]) if winner else "—"
        cells.append(f"<td>{winner_label}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    wins_row = "<tr class='wins-row'><td><strong>Wins</strong></td>"
    for t in per_method_runs:
        wins_row += f"<td><strong>{wins[t]}/{len(seeds)}</strong></td>"
    wins_row += "<td>—</td></tr>"

    coverage_row = "<tr class='coverage-row'><td><strong>Seeds completed</strong></td>"
    for t in per_method_runs:
        coverage_row += f"<td>{len(by_seed[t])}/{len(seeds)}</td>"
    coverage_row += "<td>—</td></tr>"

    return (
        f"<table class='per-seed'>"
        f"<thead><tr><th>Seed</th>{headers}<th>Winner</th></tr></thead>"
        f"<tbody>{''.join(rows)}{coverage_row}{wins_row}</tbody></table>"
    )


def _render_hyperparams(hyperparams: dict[str, dict], method_labels: dict[str, str]) -> str:
    """Collapsible <details> block per method with JSON dump."""
    blocks = []
    for tag, hp in hyperparams.items():
        label = method_labels.get(tag, tag)
        json_text = html.escape(json.dumps(hp, indent=2, default=str))
        blocks.append(
            f"<details class='hp-block'>"
            f"<summary><strong>{html.escape(label)}</strong> hyperparameters</summary>"
            f"<pre>{json_text}</pre></details>"
        )
    return "\n".join(blocks)


def _render_plot_sections(seeds: list[int], per_seed_dir: Path,
                          method_tags: list[str], method_labels: dict[str, str]) -> str:
    """Nested collapsible iframes: Seed > Method > full plot HTML."""
    blocks = []
    for seed in seeds:
        method_blocks = []
        for tag in method_tags:
            html_name = f"seed_{seed:02d}/{tag}_report.html"
            html_path = per_seed_dir / html_name
            if not html_path.exists():
                continue
            method_blocks.append(
                f"<details class='method-plots'>"
                f"<summary>{html.escape(method_labels[tag])}</summary>"
                f"<iframe src='{html.escape(html_name)}' loading='lazy' "
                f"class='plot-frame'></iframe>"
                f"</details>"
            )
        seed_body = "\n".join(method_blocks) or "<p><em>No plots generated for this seed.</em></p>"
        blocks.append(
            f"<details class='seed-plots'>"
            f"<summary>Seed {seed}</summary>"
            f"{seed_body}"
            f"</details>"
        )
    return "\n".join(blocks)


def _read_input_csvs(input_dir: Path) -> dict[str, list[dict]]:
    """Load all CSVs in Input_data for bottom-of-report display."""
    out: dict[str, list[dict]] = {}
    for csv_path in sorted(input_dir.glob("*.csv")):
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                out[csv_path.name] = list(reader)
        except Exception as exc:
            out[csv_path.name] = [{"error": str(exc)}]
    return out


def _render_input_params(input_csvs: dict[str, list[dict]]) -> str:
    blocks = []
    for fname, rows in input_csvs.items():
        if not rows:
            blocks.append(f"<h4>{html.escape(fname)}</h4><p><em>empty</em></p>")
            continue
        headers = list(rows[0].keys())
        thead = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body_rows = []
        for row in rows:
            cells = "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers)
            body_rows.append(f"<tr>{cells}</tr>")
        blocks.append(
            f"<details class='csv-block' open>"
            f"<summary><code>{html.escape(fname)}</code> ({len(rows)} rows)</summary>"
            f"<table class='csv'><thead><tr>{thead}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table></details>"
        )
    return "\n".join(blocks)


def render_report(out_dir: Path, config: dict, metadata: dict,
                  per_method_runs: dict[str, list[dict]], method_labels: dict[str, str],
                  hyperparams: dict[str, dict], input_csvs: dict[str, list[dict]],
                  seeds: list[int]) -> Path:
    """Compose the main comparison_report.html."""
    css = """
      body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #222; }
      h1 { margin-bottom: 4px; }
      h2 { border-bottom: 2px solid #ddd; padding-bottom: 4px; margin-top: 32px; }
      h3 { margin-top: 20px; }
      .meta { background: #f5f5f7; padding: 12px 16px; border-radius: 6px; font-size: 0.92em; }
      .meta code { background: #fff; padding: 1px 5px; border-radius: 3px; }
      table { border-collapse: collapse; margin: 8px 0; font-size: 0.92em; }
      th, td { padding: 6px 10px; border: 1px solid #d0d0d5; vertical-align: top; }
      th { background: #efefef; text-align: left; }
      table.summary td, table.summary th { min-width: 140px; }
      table.per-seed td.winner { background: #d8f4d6; font-weight: bold; }
      table.per-seed td.failed { background: #fdecea; color: #c0392b; text-align: center; }
      table.per-seed tr.wins-row { background: #eef4ff; }
      table.per-seed tr.coverage-row { background: #fff9e6; font-size: 0.86em; }
      table.csv { font-size: 0.85em; }
      table.csv th, table.csv td { padding: 3px 6px; }
      .muted { color: #777; font-size: 0.84em; }
      details { margin: 6px 0; border: 1px solid #ddd; border-radius: 4px; padding: 8px 12px; background: #fafafa; }
      details > summary { cursor: pointer; font-weight: 600; padding: 2px 0; }
      details[open] > summary { margin-bottom: 8px; }
      details.seed-plots { border-color: #b0c4de; background: #f4f8fc; }
      details.method-plots { border-color: #d3d3d3; background: #fff; margin-left: 18px; }
      details.hp-block pre { background: #1e1e1e; color: #dcdcdc; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 0.86em; }
      iframe.plot-frame { width: 100%; height: 1200px; border: 1px solid #ccc; border-radius: 4px; }
      .winner-tag { display: inline-block; background: #4caf50; color: #fff; padding: 1px 6px; border-radius: 3px; font-size: 0.8em; }
    """

    # Metadata header
    meta_lines = [
        f"<strong>Generated:</strong> {metadata['generated_at']}",
        f"<strong>Seeds:</strong> {seeds[0]}..{seeds[-1]} ({len(seeds)} runs)",
        f"<strong>UPS:</strong> λ={metadata['ups_lambda']} events/shift, μ={metadata['ups_mu']} min",
        f"<strong>Shift length:</strong> {metadata['shift_length']} min",
        f"<strong>Input dir:</strong> <code>{html.escape(str(metadata['input_dir']))}</code>",
    ]
    for tag, label in method_labels.items():
        path = metadata["model_paths"].get(tag, "—")
        meta_lines.append(f"<strong>{html.escape(label)} model:</strong> <code>{html.escape(str(path))}</code>")
    meta_lines.append(f"<strong>Total wall time:</strong> {metadata['total_wall_sec']:.1f}s")
    meta_html = "<br>".join(meta_lines)

    summary_table = _render_summary_table(per_method_runs, method_labels)
    per_seed_table = _render_per_seed_table(per_method_runs, method_labels, seeds)
    hp_html = _render_hyperparams(hyperparams, method_labels)
    plots_html = _render_plot_sections(seeds, out_dir, list(per_method_runs.keys()), method_labels)
    inputs_html = _render_input_params(input_csvs)

    doc = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{html.escape(config['report_title'])}</title>
<style>{css}</style>
</head>
<body>
<h1>{html.escape(config['report_title'])}</h1>
<div class='meta'>{meta_html}</div>

<h2>1. Aggregate Comparison</h2>
<p>Averages and dispersion across <strong>{len(seeds)} runs</strong>. Each seed feeds the
<strong>identical UPS realization</strong> to every method.</p>
{summary_table}

<h2>2. Per-Seed Breakdown (Net Profit)</h2>
<p>Winner highlighted in green. Bottom row tallies wins.</p>
{per_seed_table}

<h2>3. Method Hyperparameters</h2>
{hp_html}

<h2>4. Plots Per Seed Per Method</h2>
<p>Click to expand. Each method report contains Roasting Schedule (Gantt), RC Stock,
GC Stock, Restock Operations, Pipeline Utilization, Cost Waterfall and R3 Routing.</p>
{plots_html}

<h2>5. Input Parameters</h2>
{inputs_html}

<p class='muted' style='margin-top:32px'>Generated by <code>scripts/compare_methods.py</code>.
Edit the CONFIG block at the top of the file to adjust seeds, methods, UPS parameters,
or CP-SAT budget for sensitivity analysis.</p>
</body>
</html>
"""
    out_path = out_dir / "comparison_report.html"
    out_path.write_text(doc, encoding="utf-8")
    return out_path


# =============================================================================
# MAIN
# =============================================================================

def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _save_plot_html(result: dict, path: Path) -> None:
    try:
        from plot_result import _build_html
        path.write_text(_build_html(result, compare=None, offline=True), encoding="utf-8")
    except Exception as exc:
        path.write_text(f"<html><body><h1>Plot generation failed</h1><pre>{html.escape(str(exc))}</pre></body></html>",
                        encoding="utf-8")


def rescue_render(run_dir: Path) -> None:
    """Re-render the HTML report from already-saved per-seed JSONs.

    Useful when the main run crashed during rendering but all per-seed
    artifacts were saved (e.g. CP-SAT timeouts triggered index errors).
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    seed_dirs = sorted(run_dir.glob("seed_*"))
    if not seed_dirs:
        raise RuntimeError(f"No seed_* subdirectories in {run_dir}")

    method_labels = {
        "cpsat": "CP-SAT",
        "ql": "Q-Learning",
        "rlhh": "RL-HH (Dueling DDQN)",
        "ppo": "MaskedPPO",
    }

    per_method_runs: dict[str, list[dict]] = {t: [] for t in method_labels}
    seeds: list[int] = []
    first_result_for_inputs: dict | None = None
    ups_lambda = ups_mu = shift_length = None
    model_paths: dict[str, str] = {}

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("_")[-1])
        except ValueError:
            continue
        seeds.append(seed)
        for tag in method_labels:
            jf = sd / f"{tag}_result.json"
            if not jf.exists():
                continue
            try:
                with open(jf, "r", encoding="utf-8") as fh:
                    result = json.load(fh)
            except Exception:
                continue
            kpi = result.get("kpi", {})
            summary = _summarize(kpi)
            summary["seed"] = seed
            per_method_runs[tag].append(summary)
            if first_result_for_inputs is None:
                first_result_for_inputs = result
                ups_lambda = result.get("experiment", {}).get("lambda_rate")
                ups_mu = result.get("experiment", {}).get("mu_mean")
                shift_length = result.get("parameters", {}).get("SL") or result.get("parameters", {}).get("shift_length") or 480
            mp = result.get("metadata", {}).get("model_path", "—")
            if mp and tag not in model_paths:
                model_paths[tag] = mp

    seeds = sorted(set(seeds))
    per_method_runs = {t: r for t, r in per_method_runs.items() if r}

    # Aggregate
    aggregate: dict[str, dict] = {}
    for tag, runs in per_method_runs.items():
        aggregate[tag] = {
            "solver_name": method_labels[tag],
            "n_runs": len(runs),
            "avg_net": statistics.mean(r["net_profit"] for r in runs),
            "std_net": statistics.stdev(r["net_profit"] for r in runs) if len(runs) > 1 else 0.0,
            "min_net": min(r["net_profit"] for r in runs),
            "max_net": max(r["net_profit"] for r in runs),
            "avg_revenue": statistics.mean(r["revenue"] for r in runs),
            "avg_tard": statistics.mean(r["tard_cost"] for r in runs),
            "avg_setup": statistics.mean(r["setup_cost"] for r in runs),
            "avg_idle": statistics.mean(r["idle_cost"] for r in runs),
            "avg_stockout": statistics.mean(r["stockout_cost"] for r in runs),
            "avg_psc": statistics.mean(r["psc"] for r in runs),
            "avg_ndg": statistics.mean(r["ndg"] for r in runs),
            "avg_busta": statistics.mean(r["busta"] for r in runs),
            "avg_restocks": statistics.mean(r["restocks"] for r in runs),
            "per_seed": runs,
        }

    # Try to load an existing summary.json for hyperparams + metadata; else reconstruct
    hyperparams: dict[str, dict] = {}
    prev_summary = run_dir / "summary.json"
    if prev_summary.exists():
        try:
            with open(prev_summary, "r", encoding="utf-8") as fh:
                prev = json.load(fh)
                hyperparams = prev.get("hyperparameters", {})
        except Exception:
            pass

    # Rebuild hyperparams from discovery if missing
    if not hyperparams:
        if "cpsat" in per_method_runs:
            hyperparams["cpsat"] = get_cpsat_hyperparams(CONFIG["cpsat"])
        if "ql" in per_method_runs:
            qp = model_paths.get("ql")
            if qp and Path(qp).exists():
                hyperparams["ql"] = get_ql_hyperparams(Path(qp))
        if "rlhh" in per_method_runs:
            rp = model_paths.get("rlhh")
            if rp and Path(rp).exists():
                hyperparams["rlhh"] = get_rlhh_hyperparams(Path(rp))
        if "ppo" in per_method_runs:
            pp = model_paths.get("ppo")
            if pp and Path(pp).exists():
                hyperparams["ppo"] = get_ppo_hyperparams(Path(pp))

    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S") + " (rescue-render)",
        "ups_lambda": ups_lambda,
        "ups_mu": ups_mu,
        "shift_length": shift_length,
        "input_dir": CONFIG["input_dir"],
        "model_paths": model_paths,
        "total_wall_sec": 0.0,
    }

    input_csvs = _read_input_csvs(CONFIG["input_dir"])
    report_path = render_report(
        out_dir=run_dir,
        config=CONFIG,
        metadata=metadata,
        per_method_runs=per_method_runs,
        method_labels={t: method_labels[t] for t in per_method_runs},
        hyperparams=hyperparams,
        input_csvs=input_csvs,
        seeds=seeds,
    )

    # Update summary.json
    _save_json({
        "metadata": metadata,
        "seeds": seeds,
        "aggregate": aggregate,
        "hyperparameters": hyperparams,
        "rescued": True,
    }, run_dir / "summary.json")

    print(f"[rescue] Re-rendered report: {report_path}")
    print(f"[rescue] Seeds covered: {len(seeds)}  |  Methods: {list(per_method_runs.keys())}")
    for tag, a in aggregate.items():
        print(f"  {a['solver_name']:<24} N={a['n_runs']:>2}/{len(seeds)}  "
              f"avg=${a['avg_net']:>11,.0f}  std=${a['std_net']:>10,.0f}  "
              f"min=${a['min_net']:>11,.0f}  max=${a['max_net']:>11,.0f}")


def main() -> None:
    args = parse_args()

    if args.rescue:
        rescue_render(Path(args.rescue))
        return

    t_wall0 = time.perf_counter()

    # Merge CLI into CONFIG (CLI wins)
    cfg = json.loads(json.dumps(CONFIG, default=str))  # deep copy (strings for paths)
    cfg["input_dir"] = Path(cfg["input_dir"])
    cfg["output_root"] = Path(cfg["output_root"])
    cfg["n_runs"] = args.n_runs
    cfg["seed_start"] = args.seed_start
    cfg["random_seeds"] = args.random
    cfg["meta_seed"] = args.meta_seed
    cfg["cpsat"]["time_limit_sec"] = args.cpsat_time
    for skip in args.skip:
        cfg["methods_enabled"][skip] = False
    if args.ppo_model:
        cfg["model_overrides"]["ppo_model"] = args.ppo_model
    if args.ql_qtable:
        cfg["model_overrides"]["ql_qtable"] = args.ql_qtable
    if args.rlhh_ckpt:
        cfg["model_overrides"]["rlhh_ckpt"] = args.rlhh_ckpt
    if args.ups_lambda is not None:
        cfg["ups_overrides"]["lambda"] = args.ups_lambda
    if args.ups_mu is not None:
        cfg["ups_overrides"]["mu"] = args.ups_mu
    if args.no_plots:
        cfg["save_per_seed_html"] = False

    # Resolve output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = cfg["output_root"] / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model paths
    model_paths: dict[str, Path | None] = {}
    if cfg["methods_enabled"]["ql"]:
        model_paths["ql"] = (Path(cfg["model_overrides"]["ql_qtable"])
                             if cfg["model_overrides"]["ql_qtable"]
                             else find_latest_ql_qtable(_ROOT))
    if cfg["methods_enabled"]["rlhh"]:
        model_paths["rlhh"] = (Path(cfg["model_overrides"]["rlhh_ckpt"])
                               if cfg["model_overrides"]["rlhh_ckpt"]
                               else find_latest_rlhh_ckpt(_ROOT))
    if cfg["methods_enabled"]["ppo"]:
        model_paths["ppo"] = (Path(cfg["model_overrides"]["ppo_model"])
                              if cfg["model_overrides"]["ppo_model"]
                              else find_latest_ppo_model(_ROOT))
    if cfg["methods_enabled"]["cpsat"]:
        model_paths["cpsat"] = None  # solver, no artifact

    # Validate discovered artifacts
    for tag, p in model_paths.items():
        if tag == "cpsat":
            continue
        if p is None or not Path(p).exists():
            print(f"[WARN] {tag} model not found — disabling this method")
            cfg["methods_enabled"][tag] = False

    print(f"[compare_methods] Output dir: {out_dir}")
    for tag, enabled in cfg["methods_enabled"].items():
        status = "ON" if enabled else "OFF"
        mp = model_paths.get(tag, "—")
        print(f"  {tag:<6} [{status}]  {mp}")

    # Load shared sim params / data / engine
    from env.data_bridge import get_sim_params
    from env.simulation_engine import SimulationEngine
    from env.ups_generator import generate_ups_events
    from PPOmask.Engine.data_loader import load_data

    params = get_sim_params(str(cfg["input_dir"]))
    data = load_data(str(cfg["input_dir"]))
    if cfg["ups_overrides"]["lambda"] is not None:
        params["ups_lambda"] = float(cfg["ups_overrides"]["lambda"])
    if cfg["ups_overrides"]["mu"] is not None:
        params["ups_mu"] = float(cfg["ups_overrides"]["mu"])

    method_labels = {
        "cpsat": "CP-SAT",
        "ql": "Q-Learning",
        "rlhh": "RL-HH (Dueling DDQN)",
        "ppo": "MaskedPPO",
    }

    # Enabled methods in preferred display order
    active_tags = [t for t in ("cpsat", "ql", "rlhh", "ppo") if cfg["methods_enabled"][t]]
    per_method_runs: dict[str, list[dict]] = {t: [] for t in active_tags}

    if cfg["random_seeds"]:
        import random as _random
        rng = _random.Random(cfg["meta_seed"])
        seeds = sorted(rng.sample(range(1, cfg["seed_pool_max"]), cfg["n_runs"]))
        print(f"  Random seeds (meta_seed={cfg['meta_seed']}): {seeds}")
    else:
        seeds = list(range(cfg["seed_start"], cfg["seed_start"] + cfg["n_runs"]))
        print(f"  Sequential seeds: {seeds[0]}..{seeds[-1]}")

    print(f"\n{'='*100}")
    print(f"Seeds {seeds[0]}..{seeds[-1]}  |  UPS λ={params['ups_lambda']} μ={params['ups_mu']}  |  Methods: {active_tags}")
    print(f"{'='*100}")

    for seed in seeds:
        ups = generate_ups_events(
            params.get("ups_lambda", 0), params.get("ups_mu", 0),
            seed=seed, shift_length=int(params["SL"]),
            roasters=list(params["roasters"]),
        )
        ups_list = list(ups)
        seed_dir = out_dir / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Seed {seed}  (UPS events: {len(ups_list)})")

        for tag in active_tags:
            t0 = time.perf_counter()
            try:
                if tag == "cpsat":
                    result = run_cpsat(seed, ups_list, params, cfg["input_dir"], cfg["cpsat"])
                    kpi_dict = result["kpi"]
                else:
                    engine = SimulationEngine(params)
                    if tag == "ql":
                        kpi, state = run_ql(seed, ups_list, params, data, engine, model_paths[tag])
                    elif tag == "rlhh":
                        kpi, state = run_rlhh(seed, ups_list, params, data, engine, model_paths[tag])
                    elif tag == "ppo":
                        kpi, state = run_ppo(seed, ups_list, params, data, engine, model_paths[tag])
                    else:
                        raise RuntimeError(f"Unknown method tag: {tag}")
                    result = _build_sim_result(
                        method_labels[tag], tag, kpi, state, params, data, seed,
                        str(model_paths[tag]),
                    )
                    kpi_dict = kpi.to_dict()
            except Exception as exc:
                print(f"    {tag:<6}  FAILED: {exc}")
                import traceback; traceback.print_exc()
                continue

            elapsed = time.perf_counter() - t0
            summary = _summarize(kpi_dict)
            summary["seed"] = seed
            summary["elapsed_sec"] = elapsed
            per_method_runs[tag].append(summary)

            _save_json(result, seed_dir / f"{tag}_result.json")
            if cfg["save_per_seed_html"]:
                _save_plot_html(result, seed_dir / f"{tag}_report.html")

            print(f"    {tag:<6}  net=${summary['net_profit']:>10,.0f}  "
                  f"rev=${summary['revenue']:>8,.0f}  "
                  f"P/N/B={summary['psc']:>3}/{summary['ndg']}/{summary['busta']:<3}  "
                  f"tard=${summary['tard_cost']:>7,.0f}  "
                  f"idle=${summary['idle_cost']:>6,.0f}  "
                  f"({elapsed:.1f}s)")

    # Hyperparameters
    hyperparams: dict[str, dict] = {}
    if cfg["methods_enabled"]["cpsat"]:
        hyperparams["cpsat"] = get_cpsat_hyperparams(cfg["cpsat"])
    if cfg["methods_enabled"]["ql"]:
        hyperparams["ql"] = get_ql_hyperparams(Path(model_paths["ql"]))
    if cfg["methods_enabled"]["rlhh"]:
        hyperparams["rlhh"] = get_rlhh_hyperparams(Path(model_paths["rlhh"]))
    if cfg["methods_enabled"]["ppo"]:
        hyperparams["ppo"] = get_ppo_hyperparams(Path(model_paths["ppo"]))

    # Filter per_method_runs to only methods that completed ≥1 run
    per_method_runs = {t: r for t, r in per_method_runs.items() if r}

    # Aggregate stats for JSON summary
    aggregate: dict[str, dict] = {}
    for tag, runs in per_method_runs.items():
        aggregate[tag] = {
            "solver_name": method_labels[tag],
            "n_runs": len(runs),
            "avg_net": statistics.mean(r["net_profit"] for r in runs),
            "std_net": statistics.stdev(r["net_profit"] for r in runs) if len(runs) > 1 else 0.0,
            "min_net": min(r["net_profit"] for r in runs),
            "max_net": max(r["net_profit"] for r in runs),
            "avg_revenue": statistics.mean(r["revenue"] for r in runs),
            "avg_tard": statistics.mean(r["tard_cost"] for r in runs),
            "avg_setup": statistics.mean(r["setup_cost"] for r in runs),
            "avg_idle": statistics.mean(r["idle_cost"] for r in runs),
            "avg_stockout": statistics.mean(r["stockout_cost"] for r in runs),
            "avg_psc": statistics.mean(r["psc"] for r in runs),
            "avg_ndg": statistics.mean(r["ndg"] for r in runs),
            "avg_busta": statistics.mean(r["busta"] for r in runs),
            "avg_restocks": statistics.mean(r["restocks"] for r in runs),
            "per_seed": runs,
        }

    wall = time.perf_counter() - t_wall0
    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ups_lambda": params["ups_lambda"],
        "ups_mu": params["ups_mu"],
        "shift_length": int(params["SL"]),
        "input_dir": cfg["input_dir"],
        "model_paths": {tag: str(model_paths.get(tag, "—")) for tag in active_tags},
        "total_wall_sec": wall,
    }

    # Save aggregated summary.json
    _save_json({
        "metadata": metadata,
        "config": {k: v for k, v in cfg.items() if k not in ("output_root", "input_dir")},
        "seeds": seeds,
        "aggregate": aggregate,
        "hyperparameters": hyperparams,
    }, out_dir / "summary.json")

    # Read inputs for bottom-of-report
    input_csvs = _read_input_csvs(cfg["input_dir"])

    # Render main HTML
    report_path = render_report(
        out_dir=out_dir,
        config=cfg,
        metadata=metadata,
        per_method_runs=per_method_runs,
        method_labels={t: method_labels[t] for t in per_method_runs},
        hyperparams=hyperparams,
        input_csvs=input_csvs,
        seeds=seeds,
    )

    print(f"\n{'='*100}\nDONE — wall time {wall:.1f}s")
    print(f"Report:  {report_path}")
    print(f"Summary: {out_dir / 'summary.json'}")

    # Console summary table — seed-safe lookup for wins
    by_seed = _index_by_seed(per_method_runs)
    wins = {t: 0 for t in per_method_runs}
    for seed in seeds:
        nets_this_seed = {t: by_seed[t][seed]["net_profit"]
                          for t in per_method_runs if seed in by_seed[t]}
        if nets_this_seed:
            winner = max(nets_this_seed, key=nets_this_seed.get)
            wins[winner] += 1

    print(f"\n{'Method':<24} | {'Avg Net':>12} | {'Std':>10} | {'Min':>12} | {'Max':>12} | {'N':>5} | Wins")
    print("-" * 100)
    for tag, a in aggregate.items():
        print(f"{a['solver_name']:<24} | ${a['avg_net']:>11,.0f} | ${a['std_net']:>9,.0f} | "
              f"${a['min_net']:>11,.0f} | ${a['max_net']:>11,.0f} | "
              f"{a['n_runs']:>2}/{len(seeds):<2} | {wins[tag]}/{len(seeds)}")


if __name__ == "__main__":
    main()
