"""Master evaluation orchestrator — runs all methods sequentially under a shared run name.

Invokes each method's CLI as a subprocess with a uniform time budget and seed,
then aggregates the latest Results/ subdirs into a single master folder
``Results/<YYYYMMDD_HHMMSS>_MasterEval_<RunName>/`` containing a comparison
markdown and copies of every method's result.json + report.html.

Usage:
    python evaluation/master_eval.py --name FirstRun --time 600 --seed 42
    python evaluation/master_eval.py --name Quick --time 60 --skip rlhh
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.result_schema import make_run_dir

METHOD_LABELS = {
    "cpsat": ("CPSAT", "CP-SAT (optimal ceiling)"),
    "ql": ("QLearning", "Q-Learning (tabular)"),
    "rlhh": ("RLHH", "RL-HH (Dueling DDQN)"),
    "paengv2": ("PaengDDQNv2", "Paeng DDQN v2 (in dev)"),
}


def _elapsed(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def _run_method(method: str, run_name: str, time_budget: int, seed: int) -> dict | None:
    """Invoke a method's CLI, return its result.json dict if found."""
    method_label = METHOD_LABELS[method][0]
    sub_run = f"{run_name}_{method_label}"
    print(f"\n{'='*70}\n[master] Running {method_label} (budget={time_budget}s, seed={seed})\n{'='*70}")

    if method == "cpsat":
        cmd = [
            sys.executable, "-m", "cpsat_pure.runner",
            "--name", sub_run, "--time", str(time_budget),
            "--seed", str(seed),
        ]
    elif method == "ql":
        cmd = [
            sys.executable, "-m", "q_learning.train",
            "--name", sub_run, "--time", str(time_budget),
        ]
    elif method == "rlhh":
        cmd = [
            sys.executable, "-m", "rl_hh.train",
            "--name", sub_run, "cycle",
            "--cycle", "1", "--time-sec", str(time_budget),
        ]
    elif method == "paengv2":
        cmd = [
            sys.executable, "-m", "paeng_ddqn_v2.train",
            "--name", sub_run, "--time", str(time_budget), "--seed", str(seed),
        ]
    else:
        print(f"[master] Unknown method '{method}' — skipping", file=sys.stderr)
        return None

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    print(f"[master] {method_label} returned exit={proc.returncode} in {_elapsed(elapsed)}")

    # Find the most recent Results/<*_<Label>_<sub_run>...>/ folder.
    # Note: q_learning appends a "_profit{N}" suffix after training, so we
    # match folders whose name *contains* the run name (not endswith).
    results_root = _PROJECT_ROOT / "Results"
    candidates = sorted(
        [
            d for d in results_root.iterdir()
            if d.is_dir()
            and f"_{method_label}_" in d.name
            and sub_run in d.name
        ],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"[master] No Results/ folder found for {method_label}_{sub_run}", file=sys.stderr)
        return None

    method_dir = candidates[0]
    json_files = list(method_dir.glob("result*.json"))
    if not json_files:
        print(f"[master] No result.json in {method_dir}", file=sys.stderr)
        return None

    with json_files[0].open("r", encoding="utf-8") as f:
        return {"path": method_dir, "result": json.load(f)}


def _make_summary_md(out_dir: Path, results: dict[str, dict | None], run_name: str, seed: int) -> Path:
    md_path = out_dir / "Master_Evaluation.md"
    successful = [m for m in results.keys() if results[m]]
    lines = [
        f"# Master Evaluation — `{run_name}`",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Seed: {seed}",
        "",
        "## KPI Comparison",
        "",
        "| Metric | " + " | ".join(METHOD_LABELS[m][1] for m in successful) + " |",
        "|--------|" + "|".join("---" for _ in successful) + "|",
    ]

    rows = [
        ("Net Profit", "net_profit", "${:,.0f}"),
        ("Revenue", "total_revenue", "${:,.0f}"),
        ("PSC Batches", "psc_count", "{:.0f}"),
        ("NDG Batches", "ndg_count", "{:.0f}"),
        ("BUSTA Batches", "busta_count", "{:.0f}"),
        ("Tardiness Cost", "tard_cost", "${:,.0f}"),
        ("Setup Cost", "setup_cost", "${:,.0f}"),
        ("Stockout Cost", "stockout_cost", "${:,.0f}"),
        ("Idle Cost", "idle_cost", "${:,.0f}"),
        ("Restocks", "restock_count", "{:.0f}"),
    ]
    for label, key, fmt in rows:
        cells = []
        for m in successful:
            r = results[m]
            val = r["result"].get("kpi", {}).get(key)
            cells.append(fmt.format(val) if val is not None else "N/A")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    if not successful:
        lines.append("| (no successful methods) | |")
    lines.append("")
    failed = [m for m in results.keys() if not results[m]]
    if failed:
        lines.append("## Failed methods")
        lines.append("")
        for m in failed:
            lines.append(f"- {METHOD_LABELS[m][1]}")

    lines += [
        "",
        "## Reports",
        "",
    ]
    for fname in sorted(out_dir.glob("*_report.html")):
        lines.append(f"- [{fname.name}]({fname.name})")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Master evaluation orchestrator")
    parser.add_argument("--name", required=True, help="Run name (master folder suffix)")
    parser.add_argument("--time", type=int, default=600, help="Time budget per method (sec)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=list(METHOD_LABELS.keys()),
        help="Methods to skip (cpsat|ql|rlhh|paengv2)",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, choices=list(METHOD_LABELS.keys()),
        help="Run only these methods (overrides --skip)",
    )
    args = parser.parse_args(argv)

    out_dir = make_run_dir("MasterEval", args.name)
    print(f"[master] Output -> {out_dir}")

    methods = list(METHOD_LABELS.keys())
    if args.only:
        methods = args.only
    else:
        methods = [m for m in methods if m not in args.skip]

    # paeng_ddqn_v2 is a stub — skip by default unless --only
    if not args.only and "paengv2" not in args.skip:
        methods = [m for m in methods if m != "paengv2"]
        print("[master] paengv2 is a stub; skipping. Pass --only paengv2 to test the stub.")

    results: dict[str, dict | None] = {}
    for m in methods:
        res = _run_method(m, args.name, args.time, args.seed)
        results[m] = res
        if res is None:
            continue
        # Copy result.json + report.html into master dir
        method_label = METHOD_LABELS[m][0]
        src_dir = res["path"]
        for name in src_dir.iterdir():
            if name.suffix in (".json", ".html") and name.is_file():
                dest = out_dir / f"{method_label}_{name.name}"
                shutil.copy2(name, dest)

    md_path = _make_summary_md(out_dir, results, args.name, args.seed)
    print(f"\n[master] Summary written: {md_path}")
    print(f"[master] All artefacts in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
