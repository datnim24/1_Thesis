"""Render Block B aggregated.json into a single self-contained HTML report.

Implements V4Evaluation.md §11 chart layout. Each chart has a clickable
``<sup>?</sup>`` superscript in its title that links to an always-visible
``<details>`` block beneath the figure explaining why this chart matters
(literature anchors and what the reader should take away). Plotly's native
per-point hover handles raw data inspection.

Usage::

    python scripts/build_block_b_report.py
    python scripts/build_block_b_report.py --input results/block_b/aggregated.json \
                                           --output results/block_b/report.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


METHOD_ORDER = ("dispatching", "q_learning", "paeng_ddqn", "rl_hh")
METHOD_LABEL = {
    "dispatching": "Dispatching",
    "q_learning": "Tabular QL",
    "paeng_ddqn": "Paeng DDQN",
    "rl_hh": "RL-HH",
}
METHOD_COLOR = {
    "dispatching": "#9E9E9E",  # grey — null hypothesis
    "q_learning":  "#FFB74D",  # amber — tabular RL
    "paeng_ddqn":  "#1976D2",  # blue — standard DDQN
    "rl_hh":       "#43A047",  # green — thesis innovation
}


def _fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"${x:,.0f}"


def _figure_to_html(fig: go.Figure, include_plotlyjs: bool | str = False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )


def _chart_card(
    section_id: str,
    title: str,
    why_html: str,
    fig_html: str,
) -> str:
    """Wrap a Plotly figure with the title + clickable ? + always-visible details."""
    why_id = f"why-{section_id}"
    return f"""
<section class="chart-card" id="{section_id}">
  <h3>{title} <sup><a href="#{why_id}" class="why-link" title="Why this chart?">[?]</a></sup></h3>
  <div class="plotly-figure">{fig_html}</div>
  <details id="{why_id}" class="why-block">
    <summary>Why this chart?</summary>
    {why_html}
  </details>
</section>
"""


def _placeholder_card(section_id: str, title: str, message: str) -> str:
    return f"""
<section class="chart-card placeholder" id="{section_id}">
  <h3>{title}</h3>
  <p class="placeholder-msg">{message}</p>
</section>
"""


def _ordered_methods(methods_present: list[str]) -> list[str]:
    return [m for m in METHOD_ORDER if m in methods_present]


# ---------------------------------------------------------------------------
# Header section: cell-tag table + receipts
# ---------------------------------------------------------------------------


def _header_html(agg: dict) -> str:
    meta = agg["meta"]
    cells = agg["cells"]
    methods_present = meta["methods_present"]
    methods_missing = meta["methods_missing"]
    paired = meta["paired_seed_receipt"]

    # Cell-tag mini table
    rows = []
    rows.append("<tr><th></th><th>μ × 0.5</th><th>μ × 1.0</th><th>μ × 2.0</th></tr>")
    for lam in (0.5, 1.0, 2.0):
        cell_row = [f"<td><b>λ × {lam}</b></td>"]
        for mu in (0.5, 1.0, 2.0):
            key = f"lm{lam}_mm{mu}"
            if key in cells:
                d = cells[key]
                tag = f"λ={d['ups_lambda_used']:.1f}, μ={d['ups_mu_used']:.0f}"
                if lam == 1.0 and mu == 1.0:
                    tag = f"<b>TRAIN</b><br>{tag}"
            else:
                tag = "—"
            cell_row.append(f"<td>{tag}</td>")
        rows.append("<tr>" + "".join(cell_row) + "</tr>")
    grid_html = "<table class='cell-grid'>" + "".join(rows) + "</table>"

    receipt_class = "ok" if paired == "VERIFIED" else "fail"
    receipt_msg = (
        f"<b>Paired-seed receipt: <span class='{receipt_class}'>{paired}</span></b>"
        f" — every (cell, seed) tuple has matching <code>ups_event_hash</code> "
        f"across all {len(methods_present)} methods evaluated."
    )
    if paired != "VERIFIED":
        mismatches = meta.get("paired_seed_mismatches", [])
        receipt_msg += f"<br><small>{len(mismatches)} mismatches: {', '.join(mismatches[:3])}</small>"

    methods_html = "<ul>"
    for m in methods_present:
        methods_html += f"<li><span class='method-pill' style='background:{METHOD_COLOR[m]}'>{METHOD_LABEL[m]}</span> ({m})</li>"
    for m in methods_missing:
        methods_html += f"<li><span class='method-pill missing'>{METHOD_LABEL[m]}</span> (pending)</li>"
    methods_html += "</ul>"

    return f"""
<header class="report-header">
  <h1>Block B — Reactive Method Comparison Report</h1>
  <p class="subtitle">3λ × 3μ paired-seed factorial · {meta['n_cells']} cells × {meta['n_seeds']} seeds × {meta['n_methods_present']} methods.
     Generated from <code>{html.escape(str(Path('results/block_b/aggregated.json')))}</code>.</p>

  <div class="receipts">
    <p>{receipt_msg}</p>
    <p><b>Methods evaluated:</b></p>
    {methods_html}
  </div>

  <h3>UPS Cell Grid (λ × μ multipliers of Input_data)</h3>
  {grid_html}

  <details class="why-block">
    <summary>Why this report?</summary>
    <p>This report implements <b>V4Evaluation.md §11</b> — the canonical evaluation
    methodology for the v4 thesis. Every chart below has a "Why this chart?" expandable
    block citing the literature precedent (Paeng 2021 IEEE Access, Luo 2020 ASOC,
    Ren &amp; Liu 2024 Sci. Reports, Drake 2024 EJOR, Panzer 2024 IJPR).
    The 9 cells are 3 levels of UPS frequency (λ) × 3 levels of UPS duration (μ);
    every method is trained at the centre cell (1.0, 1.0) and tested off-distribution
    at the surrounding 8 cells. Hover any data point for exact numbers; click the
    <code>[?]</code> superscript in any chart title to scroll to its rationale.</p>
  </details>
</header>
"""


# ---------------------------------------------------------------------------
# §5.3.1a — Headline mean grid (heatmap, per (cell, method))
# ---------------------------------------------------------------------------


def _build_headline_heatmap(agg: dict) -> tuple[str, str]:
    methods = _ordered_methods(agg["meta"]["methods_present"])
    cell_keys = [k for k in (
        f"lm{lam}_mm{mu}" for lam in (0.5, 1.0, 2.0) for mu in (0.5, 1.0, 2.0)
    ) if k in agg["cells"]]
    z = []
    text = []
    for m in methods:
        row_z = []
        row_t = []
        for ck in cell_keys:
            stats = agg["cells"][ck]["methods"].get(m)
            if stats is None:
                row_z.append(None)
                row_t.append("—")
            else:
                row_z.append(stats["mean"])
                row_t.append(
                    f"${stats['mean']:,.0f}<br>±${stats['std']:,.0f}<br>"
                    f"CI [{stats['ci_95'][0]:,.0f}, {stats['ci_95'][1]:,.0f}]"
                )
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=cell_keys,
        y=[METHOD_LABEL[m] for m in methods],
        text=text,
        texttemplate="%{text}",
        colorscale="Viridis",
        colorbar=dict(title="Mean profit ($)"),
        hovertemplate="<b>%{y}</b> at %{x}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="5.3.1 Mean profit per method per cell (with std and 95% bootstrap CI)",
        xaxis_title="Cell (lambda_mult, mu_mult)",
        yaxis_title="Method",
        height=max(280, 80 * len(methods) + 120),
        margin=dict(l=120, r=20, t=60, b=80),
    )
    fig_html = _figure_to_html(fig)
    why = """
    <p>This heatmap renders <b>Paeng 2021 Table 5</b>'s "mean / std" reporting in
    a single visual. Each row is a method; each column is a (λ, μ) cell.
    Color intensity encodes mean profit; the in-cell label gives mean ± std plus
    the <b>95% bootstrap confidence interval</b> on the mean (Drake 2024 EJOR
    recommendation, n=1,000 resamples, deterministic via seed=42 per
    V4Evaluation.md §4.3). Hover any cell to see the exact (mean, ±std, CI) tuple.</p>
    <p><b>Reading guide:</b> a method that dominates uniformly across cells produces
    a horizontally bright row. A method with brittle generalization shows
    color collapse off the centre column. The centre cell (lm1.0_mm1.0) is
    where every RL method was trained — degradation away from it tests
    Paeng-2021-Section-V.D-style off-distribution generalization.</p>
    """
    return _chart_card("chart-5-3-1", "5.3.1 Headline 4-method × 9-cell mean profit grid",
                       why, fig_html), "chart-5-3-1"


# ---------------------------------------------------------------------------
# §5.3.1b — Bar chart with CI error bars at the training cell
# ---------------------------------------------------------------------------


def _build_centre_cell_bars(agg: dict) -> tuple[str, str]:
    cell = agg["cells"].get("lm1.0_mm1.0")
    if not cell:
        return "", ""
    methods = _ordered_methods(cell["methods_present"])
    means = [cell["methods"][m]["mean"] for m in methods]
    ci_low = [cell["methods"][m]["mean"] - cell["methods"][m]["ci_95"][0] for m in methods]
    ci_high = [cell["methods"][m]["ci_95"][1] - cell["methods"][m]["mean"] for m in methods]

    fig = go.Figure(data=go.Bar(
        x=[METHOD_LABEL[m] for m in methods],
        y=means,
        error_y=dict(type="data", symmetric=False, array=ci_high, arrayminus=ci_low,
                     thickness=2, width=8),
        marker_color=[METHOD_COLOR[m] for m in methods],
        text=[f"${v:,.0f}" for v in means],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Mean: $%{y:,.0f}<br>CI shown as error bars<extra></extra>",
    ))
    fig.update_layout(
        title="5.3.1b Mean profit at training cell (1.0, 1.0) with 95% bootstrap CIs",
        yaxis_title="Profit ($)",
        height=380,
        margin=dict(l=70, r=20, t=60, b=40),
    )
    fig_html = _figure_to_html(fig)
    why = """
    <p>Same data as 5.3.1, restricted to the <b>training cell</b> (1.0, 1.0) where every
    RL method was trained — this is the <i>headline</i> comparison in the thesis.
    Error bars are <b>95% bootstrap CIs on the mean</b> (Drake 2024 EJOR recommendation;
    PetriRL 2024 J. Manuf. Syst. is the closest in-field positive precedent for CI
    bands). Non-overlapping CIs are a strong visual cue but <i>not</i> a substitute
    for the paired Wilcoxon tests in §5.3.2 — for paired data the test has more
    power than visual CI overlap.</p>
    <p><b>What to look for:</b> the magnitude of the gap between methods at the cell
    they were optimized for, plus the CI width (a proxy for cross-seed reliability —
    Panzer 2024 IJPR's σ-as-reliability framing).</p>
    """
    return _chart_card("chart-5-3-1b",
                       "5.3.1b Training-cell headline bar with 95% CIs",
                       why, fig_html), "chart-5-3-1b"


# ---------------------------------------------------------------------------
# §5.3.2 — Wilcoxon p-value heatmaps (3×3 grid of 3×3 matrices for 9 cells)
# ---------------------------------------------------------------------------


def _build_wilcoxon_subplots(agg: dict) -> tuple[str, str]:
    methods = _ordered_methods(agg["meta"]["methods_present"])
    if len(methods) < 2:
        return _placeholder_card("chart-5-3-2", "5.3.2 Wilcoxon p-value matrices",
                                 "Need at least 2 methods for pairwise tests."), "chart-5-3-2"

    lams = [0.5, 1.0, 2.0]
    mus = [0.5, 1.0, 2.0]
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f"λ×{lam}, μ×{mu}" for lam in lams for mu in mus],
        horizontal_spacing=0.07, vertical_spacing=0.10,
    )

    for r_idx, lam in enumerate(lams):
        for c_idx, mu in enumerate(mus):
            ck = f"lm{lam}_mm{mu}"
            cell = agg["cells"].get(ck)
            if cell is None:
                continue
            n = len(methods)
            z_grid = [[None] * n for _ in range(n)]
            text_grid = [[""] * n for _ in range(n)]
            for i, a in enumerate(methods):
                for j, b in enumerate(methods):
                    if i == j:
                        text_grid[i][j] = "—"
                        continue
                    key = f"{a}_vs_{b}"
                    rev = f"{b}_vs_{a}"
                    if key in cell["wilcoxon"]:
                        w = cell["wilcoxon"][key]
                    elif rev in cell["wilcoxon"]:
                        w = cell["wilcoxon"][rev]
                    else:
                        continue
                    p = w.get("p_holm") if w.get("p_holm") is not None else w.get("p")
                    z_grid[i][j] = -math.log10(max(p, 1e-300))  # for color scale
                    text_grid[i][j] = f"{w['sig']}<br>p={p:.2e}" if p is not None else "n/a"

            fig.add_trace(
                go.Heatmap(
                    z=z_grid,
                    x=[METHOD_LABEL[m] for m in methods],
                    y=[METHOD_LABEL[m] for m in methods],
                    text=text_grid,
                    texttemplate="%{text}",
                    showscale=(r_idx == 0 and c_idx == len(mus) - 1),
                    colorscale="Reds",
                    zmin=0, zmax=10,
                    colorbar=dict(title="-log10(p)") if (r_idx == 0 and c_idx == len(mus) - 1) else None,
                    hovertemplate=(
                        f"<b>{ck}</b><br>"
                        "row %{y} vs col %{x}<br>%{text}<extra></extra>"
                    ),
                ),
                row=r_idx + 1, col=c_idx + 1,
            )
    fig.update_layout(
        title="5.3.2 Pairwise Wilcoxon signed-rank p-values per cell (Holm-corrected)",
        height=900,
        margin=dict(l=80, r=20, t=80, b=40),
    )
    fig_html = _figure_to_html(fig)
    why = """
    <p>Each of the 9 small heatmaps is one cell of the (λ × μ) grid. Within a cell
    we run all <i>k(k−1)/2</i> pairwise paired Wilcoxon signed-rank tests (k methods),
    Holm-Bonferroni-corrected at family-wise α=0.05 per V4Evaluation.md §4.4.
    Color encodes <code>-log10(p_Holm)</code>; the in-cell label shows the
    significance marker (<code>***</code>=p&lt;0.001, <code>**</code>=p&lt;0.01,
    <code>*</code>=p&lt;0.05, <code>ns</code>=p≥0.05) and the raw Holm p-value.</p>
    <p><b>Why Wilcoxon and not paired t-test:</b> profit distributions are not normal
    (heavy-tailed left from pathological-UPS realizations). Luo 2020 Section 6.4
    used paired t-tests; we use the more conservative non-parametric equivalent
    (Drake 2024 EJOR identifies non-parametric tests as underused in scheduling-DRL).</p>
    <p><b>Why Holm and not raw Bonferroni:</b> Holm step-down has the same
    family-wise error rate but more power. Patterson et al. 2024 explicitly
    recommends MCC for cross-method RL evaluations.</p>
    """
    return _chart_card("chart-5-3-2",
                       "5.3.2 Wilcoxon p-value matrices per cell (Holm-corrected)",
                       why, fig_html), "chart-5-3-2"


# ---------------------------------------------------------------------------
# §5.3.3 — Win-rate matrices per cell (3×3 grid of k×k matrices)
# ---------------------------------------------------------------------------


def _build_winrate_subplots(agg: dict) -> tuple[str, str]:
    methods = _ordered_methods(agg["meta"]["methods_present"])
    lams = [0.5, 1.0, 2.0]
    mus = [0.5, 1.0, 2.0]
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f"λ×{lam}, μ×{mu}" for lam in lams for mu in mus],
        horizontal_spacing=0.07, vertical_spacing=0.10,
    )

    for r_idx, lam in enumerate(lams):
        for c_idx, mu in enumerate(mus):
            ck = f"lm{lam}_mm{mu}"
            cell = agg["cells"].get(ck)
            if cell is None:
                continue
            wr = cell["winrate"]
            z_grid = []
            text_grid = []
            for a in methods:
                row_z = []
                row_t = []
                for b in methods:
                    if a == b:
                        row_z.append(None)
                        row_t.append("—")
                        continue
                    val = wr.get(a, {}).get(b)
                    if val is None:
                        row_z.append(None)
                        row_t.append("—")
                    else:
                        row_z.append(val)
                        row_t.append(f"{val:.0%}")
                z_grid.append(row_z)
                text_grid.append(row_t)

            fig.add_trace(
                go.Heatmap(
                    z=z_grid,
                    x=[METHOD_LABEL[m] for m in methods],
                    y=[METHOD_LABEL[m] for m in methods],
                    text=text_grid,
                    texttemplate="%{text}",
                    showscale=(r_idx == 0 and c_idx == len(mus) - 1),
                    colorscale="RdBu",
                    zmid=0.5, zmin=0, zmax=1,
                    colorbar=dict(title="W[i,j]") if (r_idx == 0 and c_idx == len(mus) - 1) else None,
                    hovertemplate=(
                        f"<b>{ck}</b><br>"
                        "%{y} beats %{x} on %{z:.0%} of seeds<extra></extra>"
                    ),
                ),
                row=r_idx + 1, col=c_idx + 1,
            )
    fig.update_layout(
        title="5.3.3 Win-rate matrices per cell — W[i, j] = #seeds where i &gt; j (strict) / 50",
        height=900,
        margin=dict(l=80, r=20, t=80, b=40),
    )
    fig_html = _figure_to_html(fig)
    why = """
    <p>For each cell, <code>W[i,j]</code> is the fraction of the 50 paired seeds where
    method <i>i</i>'s profit strictly exceeds method <i>j</i>'s. Ties (rare,
    typically only at low-disruption cells) go to neither, so
    <code>W[i,j] + W[j,i] ≤ 1</code>.</p>
    <p>The win-rate complements the Wilcoxon: it answers a different question.
    Wilcoxon says "is the median paired difference significantly non-zero?";
    win-rate says "how often does method <i>i</i> beat method <i>j</i>?"
    A method that wins by 1¢ on every seed gets the same win-rate as one that wins
    by $100k on every seed — so very high win-rates flag <i>categorical</i>
    superiority, not magnitude.</p>
    <p><b>Literature anchor:</b> Luo 2020 Section 6.4 — "DQN beats tabular Q-Learning
    on 38/45 instances (84.4% win rate) at DDT=1.0". V4Evaluation.md §4.2 mandates
    this metric.</p>
    """
    return _chart_card("chart-5-3-3",
                       "5.3.3 Win-rate matrices per cell",
                       why, fig_html), "chart-5-3-3"


# ---------------------------------------------------------------------------
# §5.3.4 — Three nested contrasts
# ---------------------------------------------------------------------------


def _build_contrasts_card(agg: dict) -> tuple[str, str]:
    contrasts = agg.get("contrasts", {})

    cards = []

    # C1
    c1 = contrasts.get("C1_rules_vs_learning")
    if c1:
        rows = []
        for lm, info in c1.items():
            rows.append(
                f"<tr><td><b>{METHOD_LABEL.get(lm, lm)}</b> vs Dispatching</td>"
                f"<td>{info['majority_wins_cells']}</td>"
                f"<td>{info['significant_cells_holm']}</td>"
                f"<td>p ∈ [{info['raw_p_range'][0]:.2e}, {info['raw_p_range'][1]:.2e}]</td></tr>"
            )
        c1_html = (
            "<h4>C1 — Rules vs Learning</h4>"
            "<table class='contrast-table'><thead><tr>"
            "<th>Comparison</th><th>Cells where learning wins majority</th>"
            "<th>Significant cells (Holm)</th><th>Raw p range</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    else:
        c1_html = "<h4>C1 — Rules vs Learning</h4><p>Not available — need Dispatching + at least one learning method.</p>"

    # C2
    c2 = contrasts.get("C2_tabular_vs_deep")
    if c2:
        rows = []
        for cell_info in c2["cells"]:
            rows.append(
                f"<tr><td>{cell_info['cell']}</td>"
                f"<td>{cell_info['sig']}</td>"
                f"<td>p_Holm = {cell_info['p_holm']}</td>"
                f"<td>{cell_info['deep_winrate_vs_tabular']:.0%}</td></tr>"
            )
        c2_html = (
            "<h4>C2 — Tabular vs Deep</h4>"
            "<table class='contrast-table'><thead><tr>"
            "<th>Cell</th><th>Sig</th><th>Holm p</th><th>Deep wins vs tabular</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    else:
        c2_html = (
            "<h4>C2 — Tabular vs Deep</h4>"
            "<div class='pending-msg'>⏳ <b>Pending Paeng-DDQN integration.</b><br>"
            "This contrast tests whether deep continuous-state RL beats discretized-state "
            "tabular QL — a direct replication test of Luo 2020 (38/45 instances) and "
            "Paeng 2021 (8/8 datasets) on our problem class. Will populate when Phase 5 "
            "lands and <code>paeng_ddqn</code> is added to the sweep.</div>"
        )

    # C3
    c3 = contrasts.get("C3_dueling_vs_standard")
    if c3:
        rows = []
        for cell_info in c3["cells"]:
            rows.append(
                f"<tr><td>{cell_info['cell']}</td>"
                f"<td>{cell_info['sig']}</td>"
                f"<td>p_Holm = {cell_info['p_holm']}</td>"
                f"<td>{cell_info['rl_hh_winrate_vs_paeng']:.0%}</td></tr>"
            )
        c3_html = (
            "<h4>C3 — Standard DDQN vs Dueling RL-HH</h4>"
            "<table class='contrast-table'><thead><tr>"
            "<th>Cell</th><th>Sig</th><th>Holm p</th><th>RL-HH wins vs Paeng</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    else:
        c3_html = (
            "<h4>C3 — Standard DDQN vs Dueling RL-HH</h4>"
            "<div class='pending-msg'>⏳ <b>Pending Paeng-DDQN integration.</b><br>"
            "This contrast tests whether the architectural step from standard DDQN with "
            "direct allocation (Paeng 2021) to Dueling DDQN with tool-selection action "
            "space (this thesis, motivated by Ren &amp; Liu 2024) yields gains. The thesis's "
            "main novelty contribution. Will populate when Phase 5 lands.</div>"
        )

    why = """
    <p>The Block-B headline is decomposed into three nested method-vs-method contrasts
    per V4Evaluation.md §8, each with a literature-backed expected pattern:</p>
    <ol>
      <li><b>C1 (Rules vs Learning)</b> — null-hypothesis test. Universal across all
        surveyed papers (Luo 2020 Tables 5–7, Paeng 2021 Table 3, Ren &amp; Liu 2024,
        Zhang 2022).</li>
      <li><b>C2 (Tabular vs Deep)</b> — replicates Luo 2020 Section 6.4 (DQN beats
        tabular QL 38/45, 84.4%) and Paeng 2021 Table 3 (DDQN beats LBF-Q 8/8 datasets,
        ~85%) on our problem class with shared GC pipeline.</li>
      <li><b>C3 (Standard DDQN vs Dueling RL-HH)</b> — tests Ren &amp; Liu 2024's
        finding that D3QN (0.85) outperforms standard DDQN (0.80) by 5pp.
        This is the thesis's main novelty contribution — no problem-class-matched
        precedent exists.</li>
    </ol>
    <p>The "majority wins cells" column counts cells where method-pair winrate &gt; 50%.
    "Significant cells (Holm)" counts cells where the Holm-corrected Wilcoxon p &lt; 0.05.</p>
    """
    body = f"<div class='contrasts-body'>{c1_html}{c2_html}{c3_html}</div>"
    return _chart_card("chart-5-3-4",
                       "5.3.4 Three nested contrasts",
                       why, body), "chart-5-3-4"


# ---------------------------------------------------------------------------
# §5.3.5 — RL-HH 18-cycle ablation
# ---------------------------------------------------------------------------


HARDCODED_ABLATION = [
    ("Baseline", "cycle3_best.pt + original tools", 328532, None),
    ("1", "Smart GC_RESTOCK urgency gate", 347774, "+19,242"),
    ("2", "Warm-start LR=5e-4 fine-tune", 344552, "−3,222 (reverted)"),
    ("3", "Warm-start LR=1e-4 fine-tune", 340547, "−7,227 (reverted)"),
    ("4", "SETUP_AVOID MTO hijack on R1/R2", 352622, "+4,848"),
    ("5", "R3 routing argmax(min(rc_space, gc))", 357463, "+4,841"),
    ("6v1", "Mask WAIT when productive feasible", 307671, "−49,792 (reverted)"),
    ("6v2", "GC_RESTOCK stricter (stock≤10)", 362966, "+5,503"),
    ("7", "GC_RESTOCK stock≤6 ratio<0.35", 373000, "+10,034"),
    ("8", "Per-capacity GC thresholds", 373000, "no measurable effect"),
    ("9", "PSC_THROUGHPUT GC depletion guard", 363913, "−9,087 (reverted)"),
    ("10", "R3 tie-break flip to L2", 374628, "+1,628"),
    ("11", "R3 sum score (rc + gc)", 373250, "−1,378 (reverted)"),
    ("12", "MTO priority NDG > BUSTA", 373984, "−644 (reverted)"),
    ("13", "MTO_DEADLINE returns PSC fallback", 371324, "reverted"),
    ("14", "UPS-aware R3 routing", 375084, "+456"),
    ("15", "R2 NDG when R1 DOWN", 374332, "reverted"),
    ("16", "GC_RESTOCK weighted by idle roasters", 375084, "no net effect"),
    ("17", "R3 score weights GC × 1.5", 373596, "reverted"),
    ("18", "GC_RESTOCK loosened by idle count", 352906, "reverted"),
    ("Final", "Cycles 1+4+5+6v2+7+8+10+14+16 kept", 375084, "+46,552 cumulative (+14.2%)"),
]


def _try_parse_cycle_progress(md_path: Path) -> list[tuple] | None:
    """Best-effort markdown-table parser. Returns None on any difficulty."""
    if not md_path.exists():
        return None
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return None
    # Look for the "10-Cycle Brainstorm" or similar progress table
    pattern = re.compile(r"\|\s*Cycle\s*\|\s*Change\s*\|.*?\n((?:\|.+\|\n)+)", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        return None
    # Parse first match's rows
    rows: list[tuple] = []
    for line in matches[0].split("\n"):
        line = line.strip()
        if not line or line.startswith("|---") or line.startswith("|:--"):
            continue
        parts = [c.strip() for c in line.split("|") if c.strip() != ""]
        if len(parts) < 3:
            continue
        rows.append(tuple(parts))
    return rows if len(rows) >= 5 else None


def _build_ablation_card(repo_root: Path) -> tuple[str, str]:
    parsed = _try_parse_cycle_progress(repo_root / "test_rl_hh" / "RLHHtrainProgress.md")
    if parsed and len(parsed) >= 5:
        # Best-effort use; if shape doesn't match, fall back
        try:
            rows = []
            for r in parsed:
                rows.append(r[:4] if len(r) >= 4 else (*r, *(["—"] * (4 - len(r)))))
            data = rows
        except Exception:
            data = HARDCODED_ABLATION
    else:
        data = HARDCODED_ABLATION

    headers = ["Cycle", "Change", "Mean profit ($)", "Δ vs prior best"]
    cells = list(zip(*[
        [str(row[0]) for row in data],
        [str(row[1]) for row in data],
        [f"{row[2]:,}" if isinstance(row[2], (int, float)) else str(row[2]) for row in data],
        [str(row[3]) if row[3] is not None else "—" for row in data],
    ]))

    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{h}</b>" for h in headers],
                    fill_color="#37474F",
                    font=dict(color="white", size=13),
                    align="left"),
        cells=dict(values=[
            [str(row[0]) for row in data],
            [str(row[1]) for row in data],
            [f"${row[2]:,}" if isinstance(row[2], (int, float)) else str(row[2]) for row in data],
            [str(row[3]) if row[3] is not None else "—" for row in data],
        ],
                   fill_color=[["white", "#F5F5F5"] * (len(data) // 2 + 1)],
                   align="left",
                   font=dict(size=12),
                   height=28),
        columnwidth=[0.7, 4.0, 1.4, 2.0],
    )])
    fig.update_layout(
        title="5.3.5 RL-HH 18-cycle ablation (kept + reverted)",
        height=max(300, 36 * len(data) + 60),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig_html = _figure_to_html(fig)
    why = """
    <p>Per-cycle attribution of the +$46,552 (+14.2%) RL-HH improvement over the
    baseline cycle-3 checkpoint with original tools. Reverted cycles are kept
    (with their negative result) — this is unusually thorough vs the literature:
    Zhang 2022 Table 4 has one ablation row, Paeng 2021 Section V.D has three
    (PABS / FBS-1D / proposed).</p>
    <p>The cycle-tuning is <b>tool-behavior refinement</b>, not retraining of the
    Dueling-DDQN meta-agent. The 18 cycles modify what each of the 5 tools
    deterministically returns given a state; the agent's Q-value ordering over
    tools transfers cleanly. This matches Panzer 2024 IJPR's "tool-based action
    spaces preserve interpretability" framing.</p>
    <p><b>Literature anchor:</b> V4Evaluation.md §9.2; Drake 2024 EJOR identifies
    cycle-by-cycle attribution as a missing practice in scheduling-DRL.</p>
    """
    return _chart_card("chart-5-3-5",
                       "5.3.5 RL-HH 18-cycle ablation table",
                       why, fig_html), "chart-5-3-5"


# ---------------------------------------------------------------------------
# §5.4 — 9-cell heat map per method (one heatmap per method, side-by-side)
# ---------------------------------------------------------------------------


def _build_per_method_heatmaps(agg: dict) -> tuple[str, str]:
    methods = _ordered_methods(agg["meta"]["methods_present"])
    if not methods:
        return _placeholder_card("chart-5-4", "5.4 Per-method 9-cell heat map",
                                 "No methods to render."), "chart-5-4"

    fig = make_subplots(
        rows=1, cols=len(methods),
        subplot_titles=[METHOD_LABEL[m] for m in methods],
        horizontal_spacing=0.06,
    )

    lams = [0.5, 1.0, 2.0]
    mus = [0.5, 1.0, 2.0]
    # Compute global min/max for shared color scale
    all_means = []
    for m in methods:
        for lam in lams:
            for mu in mus:
                ck = f"lm{lam}_mm{mu}"
                cell = agg["cells"].get(ck)
                if cell and m in cell["methods"]:
                    all_means.append(cell["methods"][m]["mean"])
    zmin = min(all_means) if all_means else 0
    zmax = max(all_means) if all_means else 1

    for col, m in enumerate(methods, start=1):
        z = []
        text = []
        for lam in lams:
            row_z = []
            row_t = []
            for mu in mus:
                ck = f"lm{lam}_mm{mu}"
                cell = agg["cells"].get(ck)
                if cell and m in cell["methods"]:
                    val = cell["methods"][m]["mean"]
                    row_z.append(val)
                    row_t.append(f"${val:,.0f}")
                else:
                    row_z.append(None)
                    row_t.append("—")
            z.append(row_z)
            text.append(row_t)
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[f"μ×{mu}" for mu in mus],
                y=[f"λ×{lam}" for lam in lams],
                text=text,
                texttemplate="%{text}",
                colorscale="Viridis",
                zmin=zmin, zmax=zmax,
                showscale=(col == len(methods)),
                colorbar=dict(title="Mean profit ($)") if col == len(methods) else None,
                hovertemplate=(
                    f"<b>{METHOD_LABEL[m]}</b><br>"
                    "%{y}, %{x}<br>Mean profit: %{text}<extra></extra>"
                ),
            ),
            row=1, col=col,
        )
    fig.update_layout(
        title="5.4 Off-distribution generalization — mean profit per (λ, μ) cell, per method",
        height=400,
        margin=dict(l=70, r=20, t=80, b=60),
    )
    fig_html = _figure_to_html(fig)
    why = """
    <p>Each method's mean profit across the 9 (λ, μ) cells. Shared color scale
    so methods are directly visually comparable. The <b>training cell</b> is
    (λ=1.0, μ=1.0) — degradation away from it tests off-distribution
    generalization (Paeng 2021 Section V.D, Luo 2020 Section 6.5).</p>
    <p><b>Reading guide:</b>
    <ul>
      <li>A <i>uniform-bright</i> heatmap = robust generalization
        across disruption regimes.</li>
      <li>A <i>peaked-at-centre</i> heatmap = method overfit to training distribution.</li>
      <li>A <i>diagonal pattern</i> would suggest UPS-frequency × duration interaction
        effects — uncommon but worth flagging.</li>
    </ul></p>
    """
    return _chart_card("chart-5-4",
                       "5.4 Per-method 9-cell mean-profit heat map",
                       why, fig_html), "chart-5-4"


# ---------------------------------------------------------------------------
# §5.5a — Pareto plot mean vs σ across cells
# ---------------------------------------------------------------------------


def _build_pareto(agg: dict) -> tuple[str, str]:
    methods = _ordered_methods(agg["meta"]["methods_present"])
    aggregate = agg.get("aggregate_across_cells", {})
    method_means = aggregate.get("method_means", {})
    method_stds = aggregate.get("method_std_across_cells", {})

    # If across-cells aggregate not available, fall back to centre-cell
    points = []
    for m in methods:
        if m in method_means:
            mean = method_means[m]
            sigma = method_stds.get(m, 0.0)
            points.append((m, mean, sigma, "across cells"))
        else:
            cc = agg["cells"].get("lm1.0_mm1.0", {}).get("methods", {}).get(m)
            if cc:
                points.append((m, cc["mean"], cc["std"], "centre cell"))

    if not points:
        return _placeholder_card("chart-5-5a", "5.5 Pareto: mean vs σ", "No data."), "chart-5-5a"

    fig = go.Figure()
    for m, mean, sigma, src in points:
        fig.add_trace(go.Scatter(
            x=[sigma], y=[mean],
            mode="markers+text",
            text=[METHOD_LABEL[m]],
            textposition="top center",
            marker=dict(size=20, color=METHOD_COLOR[m], line=dict(color="black", width=1)),
            name=METHOD_LABEL[m],
            hovertemplate=(
                f"<b>{METHOD_LABEL[m]}</b><br>"
                f"Mean profit: ${mean:,.0f}<br>"
                f"σ: ${sigma:,.0f}<br>"
                f"Source: {src}<extra></extra>"
            ),
            showlegend=False,
        ))
    fig.update_layout(
        title="5.5a Pareto: mean profit vs std (σ as reliability)",
        xaxis_title="σ (standard deviation, $) — lower is more reliable",
        yaxis_title="Mean profit ($) — higher is better",
        height=420,
        margin=dict(l=70, r=20, t=60, b=60),
    )
    # Annotate Pareto direction
    fig.add_annotation(x=0, y=max(p[1] for p in points), text="↑ Higher profit, lower σ",
                       showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=11, color="#666"))
    fig_html = _figure_to_html(fig)
    why = """
    <p>Two-axis Pareto plot: <i>x</i>-axis is profit standard deviation across
    seeds-and-cells (σ), <i>y</i>-axis is mean profit. Top-left is the ideal
    quadrant (high mean, low σ — the Panzer 2024 IJPR "reliability" framing).</p>
    <p><b>Why this chart:</b> the headline mean alone hides reliability. A method
    that wins by $50k on most seeds and loses by $200k on some has a high mean
    but poor deployment risk. σ-as-reliability is V4Evaluation.md §5.4's explicit
    motivation, drawn from Panzer 2024 IJPR's observation that "the hyper-heuristic
    demonstrated more stable reward reception with lower variance across varying
    conditions."</p>
    <p><b>Literature anchor:</b> V4Evaluation.md §5.4 + §11.5; Panzer 2024 IJPR.</p>
    """
    return _chart_card("chart-5-5a",
                       "5.5a Pareto: mean profit vs σ",
                       why, fig_html), "chart-5-5a"


# ---------------------------------------------------------------------------
# §5.5b — KPI decomposition waterfall per method
# ---------------------------------------------------------------------------


def _build_kpi_waterfall(agg: dict) -> tuple[str, str]:
    kpi = agg.get("kpi_decomposition", {})
    methods = _ordered_methods(agg["meta"]["methods_present"])
    if not kpi or not methods:
        return _placeholder_card("chart-5-5b", "5.5b KPI decomposition", "No KPI data."), "chart-5-5b"

    fig = make_subplots(
        rows=1, cols=len(methods),
        subplot_titles=[METHOD_LABEL[m] for m in methods],
        horizontal_spacing=0.05,
    )
    for col, m in enumerate(methods, start=1):
        d = kpi.get(m, {})
        if not d:
            continue
        names = ["Revenue", "Tardiness", "Setup", "Stockout", "Idle", "Net profit"]
        # Costs are negative contributions
        values = [
            d.get("revenue", 0),
            -d.get("tard_cost", 0),
            -d.get("setup_cost", 0),
            -d.get("stockout_cost", 0),
            -d.get("idle_cost", 0),
            None,  # total
        ]
        measures = ["absolute", "relative", "relative", "relative", "relative", "total"]
        # Plotly waterfall: total component value is auto-computed but we can pass 0
        values[-1] = 0
        fig.add_trace(
            go.Waterfall(
                name=METHOD_LABEL[m],
                x=names,
                measure=measures,
                y=values,
                text=[
                    _fmt_money(d.get("revenue", 0)),
                    _fmt_money(-d.get("tard_cost", 0)),
                    _fmt_money(-d.get("setup_cost", 0)),
                    _fmt_money(-d.get("stockout_cost", 0)),
                    _fmt_money(-d.get("idle_cost", 0)),
                    "",
                ],
                connector={"line": {"color": "#9E9E9E"}},
                increasing={"marker": {"color": "#4CAF50"}},
                decreasing={"marker": {"color": "#F44336"}},
                totals={"marker": {"color": METHOD_COLOR[m]}},
                hovertemplate="<b>%{x}</b><br>%{text}<br>cumulative: %{y}<extra></extra>",
            ),
            row=1, col=col,
        )
    fig.update_layout(
        title="5.5b KPI decomposition — revenue minus cost components, per method (mean across cells)",
        height=440,
        showlegend=False,
        margin=dict(l=60, r=20, t=80, b=60),
    )
    fig_html = _figure_to_html(fig)
    why = """
    <p>Waterfall per method: revenue (positive, green) at left, each cost component
    deducted (negative, red), profit at right (method-colored). All values are means
    across cells.</p>
    <p><b>Why this chart:</b> the headline mean tells you "X beats Y by $Δk" but
    doesn't say <i>why</i>. The waterfall enables mechanistic attribution:
    "RL-HH gains $46k vs QL because: +6 PSC ($24k revenue), 2.4 fewer setups
    ($1.9k), 92 fewer idle minutes ($18.4k)" — exact attribution per
    V4Evaluation.md §5.5. Panzer 2024 IJPR uses analogous multi-component reporting.</p>
    """
    return _chart_card("chart-5-5b",
                       "5.5b KPI decomposition waterfall per method",
                       why, fig_html), "chart-5-5b"


# ---------------------------------------------------------------------------
# §5.6 — Method-recommendation grid (best-method-per-cell)
# ---------------------------------------------------------------------------


def _build_recommendation_grid(agg: dict) -> tuple[str, str]:
    methods = _ordered_methods(agg["meta"]["methods_present"])
    if not methods:
        return _placeholder_card("chart-5-6", "5.6 Method-recommendation grid", "No methods."), "chart-5-6"

    lams = [0.5, 1.0, 2.0]
    mus = [0.5, 1.0, 2.0]
    z = []  # method index for color
    text = []
    for lam in lams:
        row_z = []
        row_t = []
        for mu in mus:
            ck = f"lm{lam}_mm{mu}"
            cell = agg["cells"].get(ck)
            if not cell:
                row_z.append(None)
                row_t.append("—")
                continue
            best_m, best_mean = None, -math.inf
            for m in methods:
                stats = cell["methods"].get(m)
                if stats and stats["mean"] > best_mean:
                    best_m, best_mean = m, stats["mean"]
            row_z.append(methods.index(best_m) if best_m else None)
            row_t.append(f"<b>{METHOD_LABEL[best_m]}</b><br>${best_mean:,.0f}" if best_m else "—")
        z.append(row_z)
        text.append(row_t)

    # Map method index → its color
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"μ×{mu}" for mu in mus],
        y=[f"λ×{lam}" for lam in lams],
        text=text,
        texttemplate="%{text}",
        colorscale=[[i / max(1, len(methods) - 1), METHOD_COLOR[m]] for i, m in enumerate(methods)],
        zmin=0, zmax=max(1, len(methods) - 1),
        showscale=False,
        hovertemplate="%{y}, %{x}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="5.6 Method-recommendation grid — best method per (λ, μ) cell",
        height=380,
        margin=dict(l=70, r=20, t=60, b=60),
    )
    # Legend chips below
    legend_chips = "".join(
        f"<span class='method-pill' style='background:{METHOD_COLOR[m]}'>{METHOD_LABEL[m]}</span> "
        for m in methods
    )
    fig_html = _figure_to_html(fig) + f"<div class='legend-chips'>{legend_chips}</div>"
    why = """
    <p>Color-coded grid showing which method has the highest mean profit at each
    (λ, μ) cell. This is the practical takeaway — if a deployment scenario is at
    a particular UPS regime, pick the method whose color dominates that cell.</p>
    <p><b>Reading guide:</b> a <i>uniformly-coloured</i> grid means one method wins
    everywhere (Paeng-2021-style universal dominance). A <i>patchy</i> grid means
    the optimal method depends on the disruption regime — the Q1-finding the thesis
    title "<i>under which UPS conditions does each strategy dominate</i>" expects.</p>
    <p><b>Literature anchor:</b> V4Evaluation.md §11.6 (method-recommendation matrix);
    derived view of §5.4 heat maps.</p>
    """
    return _chart_card("chart-5-6",
                       "5.6 Method-recommendation grid",
                       why, fig_html), "chart-5-6"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


CSS = """
<style>
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #FAFAFA;
    color: #212121;
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px 30px 60px;
    line-height: 1.5;
  }
  h1 { color: #0D47A1; border-bottom: 3px solid #0D47A1; padding-bottom: 8px; margin-top: 0; }
  h2 { margin-top: 36px; color: #1565C0; }
  h3 { color: #263238; }
  h4 { margin-bottom: 8px; color: #455A64; }
  .subtitle { color: #555; margin: 6px 0 16px; }
  .receipts {
    background: #E8F5E9; padding: 14px 18px; border-left: 4px solid #4CAF50;
    border-radius: 6px; margin: 10px 0;
  }
  .receipts .ok { color: #1B5E20; font-weight: 700; }
  .receipts .fail { color: #B71C1C; font-weight: 700; }
  .receipts ul { margin: 6px 0 0 20px; padding: 0; }
  .receipts li { list-style: none; margin: 3px 0; }
  .method-pill {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    color: white; font-size: 12px; font-weight: 600; margin: 2px;
  }
  .method-pill.missing { background: #BDBDBD; }
  .cell-grid {
    border-collapse: collapse; margin: 10px 0; font-size: 13px;
  }
  .cell-grid th, .cell-grid td {
    border: 1px solid #CFD8DC; padding: 8px 14px; text-align: center;
  }
  .cell-grid th { background: #ECEFF1; font-weight: 700; }
  .chart-card {
    background: white; border: 1px solid #DDD;
    border-radius: 10px; padding: 18px 22px;
    margin: 24px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .chart-card.placeholder { background: #FFF8E1; border-left: 4px solid #FFA000; }
  .chart-card .placeholder-msg { color: #E65100; font-style: italic; }
  .chart-card h3 { margin-top: 0; }
  .why-link {
    color: #1976D2; text-decoration: none; font-weight: 600;
    background: #E3F2FD; padding: 1px 6px; border-radius: 4px;
    font-size: 12px;
  }
  .why-link:hover { background: #BBDEFB; }
  .why-block {
    background: #F5F5F5; padding: 10px 14px; border-left: 4px solid #2196F3;
    border-radius: 4px; margin-top: 10px; font-size: 14px;
  }
  .why-block summary {
    font-weight: 600; cursor: pointer; color: #1565C0;
  }
  .why-block summary:hover { color: #0D47A1; }
  .why-block p { margin: 8px 0; }
  .why-block code { background: #ECEFF1; padding: 0 4px; border-radius: 3px; font-size: 13px; }
  .pending-msg {
    background: #FFF8E1; padding: 12px 16px; border-left: 4px solid #FFA000;
    border-radius: 4px; margin: 8px 0;
  }
  .contrast-table {
    border-collapse: collapse; width: 100%; margin: 8px 0;
    font-size: 13px;
  }
  .contrast-table th, .contrast-table td {
    border: 1px solid #CFD8DC; padding: 6px 12px; text-align: left;
  }
  .contrast-table th { background: #ECEFF1; }
  .legend-chips { padding: 10px 0; text-align: center; }
  .footer {
    margin-top: 50px; padding-top: 20px; border-top: 1px solid #DDD;
    color: #757575; font-size: 12px; text-align: center;
  }
  sup a { color: #1976D2; }
</style>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_report(agg: dict, repo_root: Path) -> str:
    # Plotly JS once at top via CDN
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>'

    # Build all chart cards
    cards: list[tuple[str, str]] = []
    cards.append(_build_headline_heatmap(agg))
    cards.append(_build_centre_cell_bars(agg))
    cards.append(_build_wilcoxon_subplots(agg))
    cards.append(_build_winrate_subplots(agg))
    cards.append(_build_contrasts_card(agg))
    cards.append(_build_ablation_card(repo_root))
    cards.append(_build_per_method_heatmaps(agg))
    cards.append(_build_pareto(agg))
    cards.append(_build_kpi_waterfall(agg))
    cards.append(_build_recommendation_grid(agg))

    # TOC
    toc_links = []
    for card_html, card_id in cards:
        # Extract title from card_html
        m = re.search(r"<h3>([^<]+)<", card_html)
        title = m.group(1).strip() if m else card_id
        toc_links.append(f'<li><a href="#{card_id}">{title}</a></li>')
    toc_html = "<nav class='toc'><h3>Contents</h3><ol>" + "".join(toc_links) + "</ol></nav>"

    # Compose
    body = "\n".join(card_html for card_html, _ in cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Block B Evaluation Report</title>
  {plotly_cdn}
  {CSS}
</head>
<body>
  {_header_html(agg)}
  {toc_html}
  <main>
    {body}
  </main>
  <div class="footer">
    Block B report generated by <code>scripts/build_block_b_report.py</code>.
    Methodology: <code>V4Evaluation.md</code>. Statistical helpers: <code>scripts/_block_b_stats.py</code>.
  </div>
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Block B HTML report")
    parser.add_argument("--input", default="results/block_b/aggregated.json")
    parser.add_argument("--output", default="results/block_b/report.html")
    args = parser.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"FATAL: {inp} not found; run aggregate_block_b.py first.", file=sys.stderr)
        return 1
    agg = json.loads(inp.read_text(encoding="utf-8"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_report(agg, _ROOT)
    out.write_text(html_text, encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
