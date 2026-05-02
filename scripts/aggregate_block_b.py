"""Aggregate Block B per-cell, per-method JSONs into a single ``aggregated.json``.

Glob-driven: discovers methods present from ``results/block_b/lm*_mm*/*.json``
so adding Paeng-DDQN later requires no aggregator change.

Computes per-cell, per-pair, and across-cells statistics:
- mean, std, min, max, median, p25, p75 (per method per cell)
- 95% bootstrap CI on the mean (deterministic, seed=42)
- pairwise paired Wilcoxon signed-rank with Holm-Bonferroni correction
- pairwise win-rate matrix (strict inequality)
- aggregate-across-cells win-rate and Wilcoxon (450 paired obs at 9 cells x 50 seeds)
- KPI decomposition per method
- paired-seed receipt: assert ups_event_hash matches across methods at every (cell, seed)
- three nested contrasts (C1 = Rules vs Learning; C2/C3 await Paeng)

Usage::

    python scripts/aggregate_block_b.py
    python scripts/aggregate_block_b.py --root results/block_b --output results/block_b/aggregated.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._block_b_stats import (  # noqa: E402
    bootstrap_ci,
    holm_adjusted,
    holm_bonferroni,
    significance_marker,
    wilcoxon_signed_rank,
    win_rate_matrix,
)


# Display order (dispatching → tabular → deep methods)
METHOD_ORDER = ("dispatching", "q_learning", "paeng_ddqn", "rl_hh")
KNOWN_METHODS = set(METHOD_ORDER)


def cell_dirname(lam: float, mu: float) -> str:
    return f"lm{lam}_mm{mu}"


def cell_label(lam: float, mu: float) -> str:
    """Human-readable cell label."""
    lvl_lam = {0.5: "low_lam", 1.0: "med_lam", 2.0: "high_lam"}.get(lam, f"lam{lam}")
    lvl_mu = {0.5: "low_mu", 1.0: "med_mu", 2.0: "high_mu"}.get(mu, f"mu{mu}")
    return f"{lvl_lam}_{lvl_mu}"


def load_cell_jsons(root: Path) -> dict[str, dict[str, dict]]:
    """Return {cell_name: {method: result_dict}} for every cell directory found."""
    out: dict[str, dict[str, dict]] = {}
    for cell_dir in sorted(root.glob("lm*_mm*/")):
        if not cell_dir.is_dir():
            continue
        cell_name = cell_dir.name
        out[cell_name] = {}
        for json_path in sorted(cell_dir.glob("*.json")):
            method = json_path.stem
            if method not in KNOWN_METHODS:
                # Allow unknown methods but report; aggregator doesn't crash.
                continue
            try:
                d = json.loads(json_path.read_text(encoding="utf-8"))
                out[cell_name][method] = d
            except Exception as exc:
                print(f"WARN: could not load {json_path}: {exc}", file=sys.stderr)
    return out


def discover_methods_present(cells: dict[str, dict[str, dict]]) -> tuple[list[str], list[str]]:
    """Return (present, missing) following METHOD_ORDER."""
    present_set: set[str] = set()
    for methods_in_cell in cells.values():
        present_set.update(methods_in_cell.keys())
    present = [m for m in METHOD_ORDER if m in present_set]
    missing = [m for m in METHOD_ORDER if m not in present_set]
    return present, missing


def verify_paired_seed(cells: dict[str, dict[str, dict]]) -> tuple[str, list[str]]:
    """Assert that within every cell, all methods produce the same
    ``ups_event_hash`` per seed.

    Returns (status, mismatches). status ∈ {'VERIFIED', 'FAILED'}.
    """
    mismatches: list[str] = []
    for cell_name, methods in cells.items():
        # Build {seed: {method: hash}}
        per_seed_hashes: dict[int, dict[str, str]] = defaultdict(dict)
        for method, d in methods.items():
            for ps in d.get("per_seed", []):
                seed = int(ps["seed"])
                h = ps.get("ups_event_hash", "<missing>")
                per_seed_hashes[seed][method] = h
        for seed, hashes in per_seed_hashes.items():
            unique = set(hashes.values())
            if len(unique) > 1:
                mismatches.append(
                    f"{cell_name} seed={seed}: hashes differ across methods: {hashes}"
                )
            elif "<missing>" in unique:
                mismatches.append(
                    f"{cell_name} seed={seed}: ups_event_hash missing in {sorted(hashes)}"
                )
    return ("VERIFIED" if not mismatches else "FAILED", mismatches)


def per_method_stats(profits: np.ndarray, kpi_means: dict[str, float]) -> dict:
    mean, lo, hi = bootstrap_ci(profits)
    return {
        "n": int(profits.size),
        "mean": round(mean, 2),
        "ci_95": [round(lo, 2), round(hi, 2)],
        "std": round(float(profits.std()), 2),
        "min": round(float(profits.min()), 2),
        "max": round(float(profits.max()), 2),
        "median": round(float(np.median(profits)), 2),
        "p25": round(float(np.percentile(profits, 25)), 2),
        "p75": round(float(np.percentile(profits, 75)), 2),
        "kpi": {k: round(v, 2) for k, v in kpi_means.items()},
    }


def aggregate_cell(cell_data: dict[str, dict]) -> dict:
    """Compute per-cell stats for a single (lam, mu) cell."""
    methods_in_cell = sorted(cell_data.keys(), key=lambda m: METHOD_ORDER.index(m))
    if not methods_in_cell:
        return {"methods": {}, "wilcoxon": {}, "winrate": {}, "n_seeds": 0}

    # Common UPS metadata from any method
    any_d = next(iter(cell_data.values()))
    ups_lambda_used = float(any_d.get("ups_lambda_used", 0.0))
    ups_mu_used = float(any_d.get("ups_mu_used", 0.0))
    lambda_mult = float(any_d.get("lambda_mult", 1.0))
    mu_mult = float(any_d.get("mu_mult", 1.0))

    # Build {method: profit_array (sorted by seed)} so pairs align
    per_seed_by_method: dict[str, dict[int, dict]] = {}
    for m in methods_in_cell:
        ps_list = cell_data[m].get("per_seed", [])
        per_seed_by_method[m] = {int(p["seed"]): p for p in ps_list}

    common_seeds = sorted(set.intersection(*(set(d.keys()) for d in per_seed_by_method.values())))
    n_seeds = len(common_seeds)

    profits: dict[str, np.ndarray] = {
        m: np.array([per_seed_by_method[m][s]["net_profit"] for s in common_seeds], dtype=float)
        for m in methods_in_cell
    }

    # Per-method stats including KPI breakdown (mean across seeds)
    kpi_keys = ["revenue", "tard_cost", "setup_cost", "stockout_cost", "idle_cost",
                "psc_count", "ndg_count", "busta_count", "setup_events",
                "restock_count", "idle_min"]
    methods_out: dict[str, dict] = {}
    for m in methods_in_cell:
        kpi_means: dict[str, float] = {}
        for k in kpi_keys:
            vals = [per_seed_by_method[m][s].get(k, 0) for s in common_seeds]
            kpi_means[k] = float(np.mean(vals))
        methods_out[m] = per_method_stats(profits[m], kpi_means)

    # Pairwise Wilcoxon (paired) + Holm correction
    pairs = [(a, b) for i, a in enumerate(methods_in_cell)
             for b in methods_in_cell[i + 1:]]
    raw_p: list[float] = []
    raw_z: list[float] = []
    statuses: list[str] = []
    for a, b in pairs:
        z, p, status = wilcoxon_signed_rank(profits[a], profits[b])
        raw_z.append(z)
        raw_p.append(p)
        statuses.append(status)

    if pairs:
        # Holm only over the 'ok' tests (skip ties_only / too_few from family)
        ok_idx = [i for i, s in enumerate(statuses) if s == "ok"]
        ok_p = np.array([raw_p[i] for i in ok_idx], dtype=float)
        adj = holm_adjusted(ok_p) if ok_p.size else np.array([])
        rejected = holm_bonferroni(ok_p, alpha=0.05) if ok_p.size else np.array([], dtype=bool)

        # Map back to all pairs
        full_adj = [None] * len(pairs)
        full_rej = [False] * len(pairs)
        for k, idx in enumerate(ok_idx):
            full_adj[idx] = float(adj[k]) if not math.isnan(adj[k]) else None
            full_rej[idx] = bool(rejected[k])

    wilcoxon_out: dict[str, dict] = {}
    for k, (a, b) in enumerate(pairs):
        key = f"{a}_vs_{b}"
        p_holm = full_adj[k]
        wilcoxon_out[key] = {
            "z": round(raw_z[k], 4) if not (isinstance(raw_z[k], float) and math.isnan(raw_z[k])) else None,
            "p": round(raw_p[k], 6),
            "p_holm": round(p_holm, 6) if p_holm is not None else None,
            "rejected_holm": full_rej[k],
            "sig": significance_marker(p_holm if p_holm is not None else raw_p[k]),
            "status": statuses[k],
        }

    # Win-rate matrix
    wm = win_rate_matrix(profits)
    winrate_out: dict[str, dict] = {}
    for a in methods_in_cell:
        winrate_out[a] = {b: (None if math.isnan(wm[a][b]) else round(wm[a][b], 4))
                          for b in methods_in_cell}

    return {
        "lambda_mult": lambda_mult,
        "mu_mult": mu_mult,
        "ups_lambda_used": ups_lambda_used,
        "ups_mu_used": ups_mu_used,
        "n_seeds": n_seeds,
        "methods_present": methods_in_cell,
        "methods": methods_out,
        "wilcoxon": wilcoxon_out,
        "winrate": winrate_out,
    }


def aggregate_across_cells(cells_data: dict[str, dict[str, dict]],
                           cells_agg: dict[str, dict],
                           methods_present: list[str]) -> dict:
    """Pool paired observations across all cells (n_total = sum of per-cell n_seeds)."""
    if not methods_present:
        return {}

    # Pool per-method profits and per-pair paired profits
    pooled: dict[str, list[float]] = {m: [] for m in methods_present}
    pooled_pairs: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for cell_name, methods in cells_data.items():
        seeds_set = set.intersection(*(
            {int(p["seed"]) for p in methods[m]["per_seed"]} for m in methods_present
            if m in methods
        )) if all(m in methods for m in methods_present) else set()
        if not seeds_set:
            continue
        for m in methods_present:
            if m not in methods:
                continue
            for ps in methods[m]["per_seed"]:
                if int(ps["seed"]) in seeds_set:
                    pooled[m].append(ps["net_profit"])
        # Paired pooling per pair
        seeds_sorted = sorted(seeds_set)
        per_seed_lookup = {
            m: {int(p["seed"]): p["net_profit"] for p in methods[m]["per_seed"]}
            for m in methods_present if m in methods
        }
        for i, a in enumerate(methods_present):
            for b in methods_present[i + 1:]:
                if a not in per_seed_lookup or b not in per_seed_lookup:
                    continue
                pooled_pairs.setdefault((a, b), [])
                for s in seeds_sorted:
                    pa = per_seed_lookup[a].get(s)
                    pb = per_seed_lookup[b].get(s)
                    if pa is not None and pb is not None:
                        pooled_pairs[(a, b)].append((pa, pb))

    # Per-method aggregate stats
    method_means = {}
    method_std_across_cells = {}
    for m in methods_present:
        # Mean of per-cell means
        cell_means = [cells_agg[c]["methods"][m]["mean"]
                      for c in cells_agg if m in cells_agg[c]["methods"]]
        if cell_means:
            method_means[m] = round(float(np.mean(cell_means)), 2)
            method_std_across_cells[m] = round(float(np.std(cell_means)), 2)

    # Aggregate Wilcoxon and win-rate over pooled pairs
    wilcoxon_overall: dict[str, dict] = {}
    winrate_overall: dict[str, dict[str, float]] = {m: {} for m in methods_present}
    for (a, b), pairs in pooled_pairs.items():
        if not pairs:
            continue
        x = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)
        z, p_val, status = wilcoxon_signed_rank(x, y)
        n = x.size
        wins_a = int(np.sum(x > y))
        wins_b = int(np.sum(y > x))
        wilcoxon_overall[f"{a}_vs_{b}"] = {
            "n": n,
            "z": round(z, 4) if not (isinstance(z, float) and math.isnan(z)) else None,
            "p": round(p_val, 6),
            "sig": significance_marker(p_val),
            "status": status,
        }
        winrate_overall[a][b] = round(wins_a / n, 4)
        winrate_overall[b][a] = round(wins_b / n, 4)

    return {
        "method_means": method_means,
        "method_std_across_cells": method_std_across_cells,
        "wilcoxon_overall": wilcoxon_overall,
        "winrate_overall": winrate_overall,
    }


def compute_contrasts(cells_agg: dict[str, dict], methods_present: list[str]) -> dict:
    """Three nested contrasts per V4Evaluation.md §8."""
    out = {
        "C1_rules_vs_learning": None,
        "C2_tabular_vs_deep": None,
        "C3_dueling_vs_standard": None,
    }
    learning_methods = [m for m in methods_present if m != "dispatching"]

    # C1: Dispatching vs each learning method, across all cells
    if "dispatching" in methods_present and learning_methods:
        c1: dict[str, dict] = {}
        for lm in learning_methods:
            wins_lm_in_cells = 0
            total_cells = 0
            sig_cells = 0
            ps_min, ps_max = 1.0, 0.0
            for cell_name, cell in cells_agg.items():
                key = f"dispatching_vs_{lm}" if f"dispatching_vs_{lm}" in cell["wilcoxon"] else f"{lm}_vs_dispatching"
                if key not in cell["wilcoxon"]:
                    continue
                total_cells += 1
                w = cell["wilcoxon"][key]
                # learning_method wins if dispatching < lm
                # Win fraction: use winrate
                wr = cell["winrate"].get(lm, {}).get("dispatching")
                if wr is not None and wr > 0.5:
                    wins_lm_in_cells += 1
                if w["sig"] in ("*", "**", "***") and w["status"] == "ok":
                    sig_cells += 1
                ps_min = min(ps_min, w["p"]) if w["status"] == "ok" else ps_min
                ps_max = max(ps_max, w["p"]) if w["status"] == "ok" else ps_max
            c1[lm] = {
                "majority_wins_cells": f"{wins_lm_in_cells}/{total_cells}",
                "significant_cells_holm": f"{sig_cells}/{total_cells}",
                "raw_p_range": [round(ps_min, 6), round(ps_max, 6)],
            }
        out["C1_rules_vs_learning"] = c1

    # C2 — tabular_vs_deep — needs at least one of {paeng_ddqn} present
    has_deep = "paeng_ddqn" in methods_present
    if "q_learning" in methods_present and has_deep:
        cell_results = []
        for cell_name, cell in cells_agg.items():
            key = "q_learning_vs_paeng_ddqn"
            if key not in cell["wilcoxon"]:
                continue
            w = cell["wilcoxon"][key]
            wr = cell["winrate"].get("paeng_ddqn", {}).get("q_learning")
            cell_results.append({
                "cell": cell_name,
                "p_holm": w.get("p_holm"),
                "sig": w["sig"],
                "deep_winrate_vs_tabular": wr,
            })
        out["C2_tabular_vs_deep"] = {
            "comparison": "q_learning vs paeng_ddqn",
            "cells": cell_results,
        }
    else:
        out["C2_tabular_vs_deep"] = None  # awaits Paeng

    # C3 — Standard DDQN vs Dueling RL-HH — needs paeng_ddqn AND rl_hh
    if "paeng_ddqn" in methods_present and "rl_hh" in methods_present:
        cell_results = []
        for cell_name, cell in cells_agg.items():
            key = "paeng_ddqn_vs_rl_hh"
            if key not in cell["wilcoxon"]:
                continue
            w = cell["wilcoxon"][key]
            wr = cell["winrate"].get("rl_hh", {}).get("paeng_ddqn")
            cell_results.append({
                "cell": cell_name,
                "p_holm": w.get("p_holm"),
                "sig": w["sig"],
                "rl_hh_winrate_vs_paeng": wr,
            })
        out["C3_dueling_vs_standard"] = {
            "comparison": "paeng_ddqn vs rl_hh",
            "cells": cell_results,
        }
    else:
        out["C3_dueling_vs_standard"] = None  # awaits Paeng

    return out


def kpi_decomposition(cells_agg: dict[str, dict], methods_present: list[str]) -> dict:
    """Mean per-method KPI averaged across all cells."""
    out: dict[str, dict[str, float]] = {}
    for m in methods_present:
        agg: dict[str, list[float]] = defaultdict(list)
        for cell in cells_agg.values():
            mstats = cell["methods"].get(m)
            if not mstats:
                continue
            for k, v in mstats["kpi"].items():
                agg[k].append(v)
        out[m] = {k: round(float(np.mean(v)), 2) for k, v in agg.items() if v}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block B aggregator")
    parser.add_argument("--root", default="results/block_b",
                        help="Directory containing lm{lam}_mm{mu}/{method}.json files")
    parser.add_argument("--output", default="results/block_b/aggregated.json")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"FATAL: {root} does not exist; run sweep first.", file=sys.stderr)
        return 1

    cells_data = load_cell_jsons(root)
    if not cells_data:
        print(f"FATAL: no cell JSONs found under {root}", file=sys.stderr)
        return 1

    methods_present, methods_missing = discover_methods_present(cells_data)
    paired_status, mismatches = verify_paired_seed(cells_data)

    cells_agg: dict[str, dict] = {}
    for cell_name in sorted(cells_data.keys()):
        cells_agg[cell_name] = aggregate_cell(cells_data[cell_name])

    aggregate = aggregate_across_cells(cells_data, cells_agg, methods_present)
    contrasts = compute_contrasts(cells_agg, methods_present)
    kpi_decomp = kpi_decomposition(cells_agg, methods_present)

    n_seeds = max((c.get("n_seeds", 0) for c in cells_agg.values()), default=0)
    n_cells = len(cells_agg)

    out = {
        "meta": {
            "n_methods_present": len(methods_present),
            "methods_present": methods_present,
            "methods_missing": methods_missing,
            "n_cells": n_cells,
            "n_seeds": n_seeds,
            "paired_seed_receipt": paired_status,
            "paired_seed_mismatches": mismatches[:10],  # truncate
        },
        "cells": cells_agg,
        "aggregate_across_cells": aggregate,
        "contrasts": contrasts,
        "kpi_decomposition": kpi_decomp,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Stdout summary
    print(f"Methods present: {methods_present}")
    print(f"Methods missing: {methods_missing}")
    print(f"Cells: {n_cells} | Seeds per cell: {n_seeds}")
    print(f"Paired-seed receipt: {paired_status}")
    if mismatches:
        for m in mismatches[:5]:
            print(f"  ! {m}")
    print(f"\nWritten: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
