"""Block B analysis: pivot tables + Wilcoxon signed-rank tests + heat maps.

Reads the 36 JSONs produced by block_b_runner.py and produces:

    pivot_summary.csv           — method × λ_mult × μ_mult → mean / std / median / p25 / p75
    contrasts.json              — three nested Wilcoxon signed-rank contrasts
    heatmap_<metric>.html       — Plotly heat map per metric (net_profit / idle / tard)
    bootstrap_ci.csv            — 1000-resample 95% CI for each method × cell

Usage:
    python block_b_analysis.py --input-dir results/block_b_<ts>/

The contrasts mirror the v4 plan §3.1:
    1. Rules vs Learning      : dispatching vs {q_learning, paeng_ddqn, rl_hh}
    2. Tabular vs Deep        : q_learning vs {paeng_ddqn, rl_hh}
    3. Standard vs Dueling RL : paeng_ddqn vs rl_hh
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

try:
    from scipy import stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("[block-b-analysis] scipy not installed — using numpy fallback (mean-diff + sign test only)")

_ROOT = Path(__file__).resolve().parent

LAMBDA_MULTS = (0.5, 1.0, 2.0)
MU_MULTS = (0.5, 1.0, 2.0)
METHODS = ("dispatching", "q_learning", "paeng_ddqn", "rl_hh")


def _load_cells(input_dir: Path, methods: tuple[str, ...]) -> dict:
    """Return {(method, lm, mm): result_dict} for whatever cells are present."""
    cells: dict = {}
    for method in methods:
        for lm in LAMBDA_MULTS:
            for mm in MU_MULTS:
                p = input_dir / f"{method}_lm{lm}_mm{mm}.json"
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        cells[(method, lm, mm)] = json.load(f)
    return cells


def write_pivot_summary(cells: dict, out_path: Path) -> None:
    rows: list[dict] = []
    for (method, lm, mm), r in cells.items():
        rows.append({
            "method": method, "lambda_mult": lm, "mu_mult": mm,
            "n_seeds": r.get("n_seeds", 0),
            "profit_mean":   r.get("profit_mean", 0),
            "profit_std":    r.get("profit_std", 0),
            "profit_median": r.get("profit_median", 0),
            "profit_p25":    r.get("profit_p25", 0),
            "profit_p75":    r.get("profit_p75", 0),
            "mean_idle_min":   r.get("mean_idle_min", 0),
            "mean_tard_cost":  r.get("mean_tard_cost", 0),
            "mean_setup_events": r.get("mean_setup_events", 0),
            "mean_restock_count": r.get("mean_restock_count", 0),
        })
    rows.sort(key=lambda x: (x["method"], x["lambda_mult"], x["mu_mult"]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"  pivot: {out_path} ({len(rows)} rows)")


def _per_seed_profits(cell: dict) -> np.ndarray:
    return np.array([s["net_profit"] for s in cell.get("per_seed", [])], dtype=np.float64)


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired contrast: Wilcoxon if scipy is available, else sign-test fallback."""
    diff = a - b
    n = int(len(diff))
    if not np.any(diff):
        return {"statistic": 0.0, "pvalue": 1.0, "n_pairs": n, "median_diff": 0.0,
                "mean_diff": 0.0, "test": "trivial"}
    if _HAS_SCIPY:
        res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return {
            "statistic":   float(res.statistic),
            "pvalue":      float(res.pvalue),
            "n_pairs":     n,
            "median_diff": float(np.median(diff)),
            "mean_diff":   float(np.mean(diff)),
            "test":        "wilcoxon_signed_rank",
        }
    # Fallback: sign test (binomial p-value via normal approximation)
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    n_eff = n_pos + n_neg
    if n_eff == 0:
        return {"statistic": 0.0, "pvalue": 1.0, "n_pairs": n, "median_diff": 0.0,
                "mean_diff": 0.0, "test": "sign_test_fallback"}
    # Two-sided sign test under H0: P(diff>0) = 0.5
    z = (n_pos - n_eff / 2) / np.sqrt(n_eff / 4)
    pvalue = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {
        "statistic":   float(n_pos),
        "pvalue":      float(pvalue),
        "n_pairs":     n,
        "n_pos":       n_pos,
        "n_neg":       n_neg,
        "median_diff": float(np.median(diff)),
        "mean_diff":   float(np.mean(diff)),
        "test":        "sign_test_fallback",
    }


def write_contrasts(cells: dict, out_path: Path) -> None:
    """Three nested contrasts per cell."""
    contrasts: dict = {}
    for lm in LAMBDA_MULTS:
        for mm in MU_MULTS:
            cell_results = {m: cells.get((m, lm, mm)) for m in METHODS}
            cell_profits = {m: _per_seed_profits(r) if r else None
                           for m, r in cell_results.items()}
            cell_key = f"lm{lm}_mm{mm}"
            contrasts[cell_key] = {}

            # 1. Rules vs Learning (dispatching vs each learning method)
            d = cell_profits.get("dispatching")
            if d is not None:
                for m in ("q_learning", "paeng_ddqn", "rl_hh"):
                    a = cell_profits.get(m)
                    if a is not None and len(a) == len(d):
                        contrasts[cell_key][f"dispatch_vs_{m}"] = _wilcoxon(a, d)

            # 2. Tabular vs Deep (q_learning vs each deep method)
            q = cell_profits.get("q_learning")
            if q is not None:
                for m in ("paeng_ddqn", "rl_hh"):
                    a = cell_profits.get(m)
                    if a is not None and len(a) == len(q):
                        contrasts[cell_key][f"qlearn_vs_{m}"] = _wilcoxon(a, q)

            # 3. Paeng (standard DDQN) vs RL-HH (Dueling DDQN)
            p = cell_profits.get("paeng_ddqn")
            r = cell_profits.get("rl_hh")
            if p is not None and r is not None and len(p) == len(r):
                contrasts[cell_key]["paeng_vs_rlhh"] = _wilcoxon(r, p)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contrasts, f, indent=2)
    print(f"  contrasts: {out_path}")


def write_heatmap(cells: dict, metric: str, out_path: Path, title: str) -> None:
    """Plot {metric} mean across methods × (lm, mm)."""
    z_data: list[list[float | None]] = []
    cell_labels = [f"l{lm}_m{mm}" for lm in LAMBDA_MULTS for mm in MU_MULTS]
    for method in METHODS:
        row: list[float | None] = []
        for lm in LAMBDA_MULTS:
            for mm in MU_MULTS:
                r = cells.get((method, lm, mm))
                row.append(float(r[metric]) if r and metric in r else None)
        z_data.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data, x=cell_labels, y=list(METHODS),
        hovertemplate="method=%{y}<br>cell=%{x}<br>" + metric + "=%{z:.2f}<extra></extra>",
        colorscale="RdYlGn" if metric == "profit_mean" else "RdYlGn_r",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="(lambda_mult, mu_mult) cell",
        yaxis_title="method",
        template="plotly_white",
        height=400,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  heatmap: {out_path}")


def write_bootstrap_ci(cells: dict, out_path: Path, n_resamples: int = 1000) -> None:
    rows: list[dict] = []
    rng = np.random.default_rng(42)
    for (method, lm, mm), r in cells.items():
        profits = _per_seed_profits(r)
        if len(profits) == 0:
            continue
        means = []
        n = len(profits)
        for _ in range(n_resamples):
            sample = rng.choice(profits, size=n, replace=True)
            means.append(float(sample.mean()))
        means = np.array(means)
        rows.append({
            "method": method, "lambda_mult": lm, "mu_mult": mm,
            "n_seeds": n,
            "profit_mean": round(float(profits.mean()), 2),
            "ci95_low":    round(float(np.percentile(means, 2.5)), 2),
            "ci95_high":   round(float(np.percentile(means, 97.5)), 2),
        })
    rows.sort(key=lambda x: (x["method"], x["lambda_mult"], x["mu_mult"]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"  bootstrap: {out_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Block B analysis.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = _ROOT / in_dir

    print(f"[block-b-analysis] reading from {in_dir}")
    cells = _load_cells(in_dir, METHODS)
    print(f"  {len(cells)} cells loaded")
    if not cells:
        print("[error] no cells found — did you run block_b_runner first?")
        return

    write_pivot_summary(cells, in_dir / "pivot_summary.csv")
    write_contrasts(cells, in_dir / "contrasts.json")
    write_heatmap(cells, "profit_mean",        in_dir / "heatmap_profit.html",   "Mean profit ($) by method x cell")
    write_heatmap(cells, "mean_idle_min",      in_dir / "heatmap_idle.html",     "Mean idle min by method x cell")
    write_heatmap(cells, "mean_tard_cost",     in_dir / "heatmap_tard.html",     "Mean tardiness cost ($) by method x cell")
    write_heatmap(cells, "mean_restock_count", in_dir / "heatmap_restock.html",  "Mean restocks by method x cell")
    write_bootstrap_ci(cells, in_dir / "bootstrap_ci.csv", args.bootstrap_resamples)

    print("\n[block-b-analysis] done.")


if __name__ == "__main__":
    main()
