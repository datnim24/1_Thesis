"""Helpers: run 100-seed eval + format markdown diff vs baseline."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(p: str | Path) -> dict:
    with open(p) as f:
        return json.load(f)


def fmt_delta(new: float, old: float, lower_is_better: bool = False) -> str:
    d = new - old
    sign = "+" if d >= 0 else ""
    arrow = ""
    if lower_is_better:
        if d < -1: arrow = " ✓"
        elif d > 1: arrow = " ✗"
    else:
        if d > 1: arrow = " ✓"
        elif d < -1: arrow = " ✗"
    return f"{sign}{d:,.1f}{arrow}"


def cycle_markdown_block(cycle: int, result: dict, baseline: dict, train_summary: dict, changes: str, hypothesis: str, diagnosis: str | None = None, action: str | None = None) -> str:
    """Produce a markdown block for one cycle."""
    p_mean = result["profit_mean"]
    p_std = result["profit_std"]
    p_med = result["profit_median"]
    p_min = result["profit_min"]
    p_max = result["profit_max"]
    vs_base = p_mean - baseline["profit_mean"]
    beat = "BEAT ✓" if p_mean > baseline["profit_mean"] else "LOSE ✗"

    tool_dist = result["tool_distribution"]
    tool_str = ", ".join(f"{k}={v*100:.1f}%" for k, v in tool_dist.items())

    lines = [
        f"## Cycle {cycle} — {hypothesis}",
        "",
        f"**Result vs baseline**: {beat} (mean=${p_mean:,.0f}, Δ=${vs_base:+,.0f} vs ${baseline['profit_mean']:,.0f})",
        "",
        "### Training",
        f"- Episodes: {train_summary['episodes']:,}",
        f"- Wall: {train_summary['wall_sec']:.0f}s ({train_summary['wall_sec']/60:.1f}min)",
        f"- Best training profit: ${train_summary['best_training_profit']:,.0f} (ep {train_summary['best_training_episode']})",
        f"- Final ε: {train_summary['final_epsilon']}",
        f"- Best ckpt: `{train_summary['best_ckpt']}`",
        "",
        "### Code Changes",
        changes,
        "",
        "### Eval (100 seeds, base_seed=900000, λ=5 μ=20)",
        "| Metric | Cycle {c} | Baseline | Δ |".format(c=cycle),
        "|--------|-----------|----------|---|",
        f"| Profit mean | ${p_mean:,.0f} | ${baseline['profit_mean']:,.0f} | **{fmt_delta(p_mean, baseline['profit_mean'])}** |",
        f"| Profit std | ${p_std:,.0f} | ${baseline['profit_std']:,.0f} | {fmt_delta(p_std, baseline['profit_std'], lower_is_better=True)} |",
        f"| Profit median | ${p_med:,.0f} | ${baseline['profit_median']:,.0f} | {fmt_delta(p_med, baseline['profit_median'])} |",
        f"| Profit min | ${p_min:,.0f} | ${baseline['profit_min']:,.0f} | {fmt_delta(p_min, baseline['profit_min'])} |",
        f"| Profit max | ${p_max:,.0f} | ${baseline['profit_max']:,.0f} | {fmt_delta(p_max, baseline['profit_max'])} |",
        f"| Mean idle min | {result['mean_idle_min']:.0f} | {baseline['mean_idle_min']:.0f} | {fmt_delta(result['mean_idle_min'], baseline['mean_idle_min'], lower_is_better=True)} |",
        f"| Mean setups | {result['mean_setup_events']:.1f} | {baseline['mean_setup_events']:.1f} | {fmt_delta(result['mean_setup_events'], baseline['mean_setup_events'], lower_is_better=True)} |",
        f"| Mean restocks | {result['mean_restock_count']:.1f} | {baseline['mean_restock_count']:.1f} | {fmt_delta(result['mean_restock_count'], baseline['mean_restock_count'], lower_is_better=True)} |",
        f"| Mean PSC | {result['mean_psc']:.1f} | {baseline['mean_psc']:.1f} | {fmt_delta(result['mean_psc'], baseline['mean_psc'])} |",
        f"| Mean tard cost | ${result['mean_tard_cost']:,.0f} | ${baseline['mean_tard_cost']:,.0f} | {fmt_delta(result['mean_tard_cost'], baseline['mean_tard_cost'], lower_is_better=True)} |",
        "",
        f"**Tool distribution**: {tool_str}",
        "",
    ]
    if diagnosis:
        lines.append("### Diagnosis")
        lines.append(diagnosis)
        lines.append("")
    if action:
        lines.append("### Next action (cycle %d)" % (cycle + 1))
        lines.append(action)
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)
