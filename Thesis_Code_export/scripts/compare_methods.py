"""Compare aggregate JSONs from multiple 100-seed evaluations.

Loads N aggregate.json files (output of scripts/eval_100seeds.py), computes
side-by-side statistics, and writes a markdown comparison + a Plotly HTML
boxplot of per-seed profit distributions.

Usage:
    python scripts/compare_methods.py --name MyComparison \\
        Results/<ts1>_100SeedEval_dispatch/aggregate.json \\
        Results/<ts2>_100SeedEval_ql_C5/aggregate.json \\
        Results/<ts3>_100SeedEval_rlhh_C8/aggregate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.result_schema import make_run_dir


def _load_aggregate(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _make_markdown(aggregates: list[tuple[Path, dict]], out_dir: Path) -> Path:
    md_path = out_dir / "comparison.md"
    lines = [
        f"# Method Comparison — {out_dir.name}",
        "",
        "| Method | N seeds | Profit mean | Profit std | Median | p25 | p75 | Restocks | Idle (min) |",
        "|--------|---------|-------------|------------|--------|-----|-----|----------|------------|",
    ]
    for path, agg in aggregates:
        name = agg.get("package", path.stem)
        lines.append(
            f"| {name} | {agg.get('n_seeds', '?')} | ${agg.get('profit_mean', 0):,.0f} | "
            f"${agg.get('profit_std', 0):,.0f} | ${agg.get('profit_median', 0):,.0f} | "
            f"${agg.get('profit_p25', 0):,.0f} | ${agg.get('profit_p75', 0):,.0f} | "
            f"{agg.get('mean_restock_count', 0):.1f} | {agg.get('mean_idle_min', 0):.1f} |"
        )
    lines += ["", "## Source files", ""]
    for path, _agg in aggregates:
        lines.append(f"- `{path}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _make_html_boxplot(aggregates: list[tuple[Path, dict]], out_dir: Path) -> Path | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[compare] plotly not installed; skipping HTML boxplot.", file=sys.stderr)
        return None

    fig = go.Figure()
    for _path, agg in aggregates:
        per_seed = agg.get("per_seed") or []
        if not per_seed:
            continue
        profits = [s["net_profit"] for s in per_seed]
        fig.add_trace(go.Box(
            y=profits,
            name=agg.get("package", "?"),
            boxmean="sd",
        ))
    fig.update_layout(
        title="Per-seed net-profit distribution",
        yaxis_title="Net profit ($)",
        template="plotly_white",
    )
    html_path = out_dir / "comparison_boxplot.html"
    fig.write_html(html_path)
    return html_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare 100-seed aggregate JSONs across methods.")
    parser.add_argument("--name", required=True, help="Comparison run name.")
    parser.add_argument("aggregates", nargs="+", help="Paths to aggregate.json files (one per method).")
    args = parser.parse_args(argv)

    out_dir = make_run_dir("Compare", args.name)
    print(f"[compare] Output -> {out_dir}")

    loaded: list[tuple[Path, dict]] = []
    for raw in args.aggregates:
        p = Path(raw)
        if not p.is_absolute():
            p = (_PROJECT_ROOT / raw).resolve()
        if not p.exists():
            print(f"[compare] WARNING: not found: {p}", file=sys.stderr)
            continue
        loaded.append((p, _load_aggregate(p)))

    if not loaded:
        print("[compare] No aggregate files loaded.", file=sys.stderr)
        return 1

    md_path = _make_markdown(loaded, out_dir)
    html_path = _make_html_boxplot(loaded, out_dir)
    print(f"[compare] Markdown: {md_path}")
    if html_path:
        print(f"[compare] HTML boxplot: {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
