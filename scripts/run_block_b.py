"""One-shot Block B orchestrator: sweep -> aggregate -> render -> open browser.

This is the user-facing entry point. Equivalent to running each script in turn
manually, but with a single command and a unified status line.

Usage::

    python scripts/run_block_b.py                     # full pipeline
    python scripts/run_block_b.py --force             # re-run sweep ignoring existing JSONs
    python scripts/run_block_b.py --no-open           # don't auto-open browser
    python scripts/run_block_b.py --methods rl_hh     # only sweep one method
    python scripts/run_block_b.py --skip-sweep        # use existing JSONs (re-aggregate + re-render only)
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import aggregate_block_b, build_block_b_report, run_block_b_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block B one-shot pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Pass --force to the sweep runner (overwrite existing JSONs)")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not auto-open the report in a browser")
    parser.add_argument("--methods", default=None,
                        help="Subset of methods to sweep (e.g., rl_hh,dispatching)")
    parser.add_argument("--cells", default=None,
                        help="Subset of cells to sweep (e.g., '1.0,1.0;2.0,2.0')")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip the sweep phase; just re-aggregate and re-render")
    parser.add_argument("--output-root", default="results/block_b")
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    aggregated_json = output_root / "aggregated.json"
    report_html = output_root / "report.html"

    t_total = time.perf_counter()

    # 1. Sweep
    if not args.skip_sweep:
        print("\n" + "=" * 70)
        print(" [1/3] Sweep — running evaluate_100_seeds for each (method, cell)")
        print("=" * 70)
        sweep_argv: list[str] = []
        if args.force:
            sweep_argv.append("--force")
        if args.methods:
            sweep_argv += ["--methods", args.methods]
        if args.cells:
            sweep_argv += ["--cells", args.cells]
        sweep_argv += ["--output-root", str(output_root)]
        rc = run_block_b_sweep.main(sweep_argv)
        if rc != 0:
            print(f"\nSweep returned exit code {rc}; aborting.", file=sys.stderr)
            return rc
    else:
        print("\n[1/3] Sweep — SKIPPED (--skip-sweep)")

    # 2. Aggregate
    print("\n" + "=" * 70)
    print(" [2/3] Aggregate — pooling JSONs, computing stats, writing aggregated.json")
    print("=" * 70)
    rc = aggregate_block_b.main([
        "--root", str(output_root),
        "--output", str(aggregated_json),
    ])
    if rc != 0:
        print(f"\nAggregator returned exit code {rc}; aborting.", file=sys.stderr)
        return rc

    # 3. Render report
    print("\n" + "=" * 70)
    print(" [3/3] Render — building HTML report with hover tooltips and 'why' blocks")
    print("=" * 70)
    rc = build_block_b_report.main([
        "--input", str(aggregated_json),
        "--output", str(report_html),
    ])
    if rc != 0:
        print(f"\nReport builder returned exit code {rc}; aborting.", file=sys.stderr)
        return rc

    elapsed = time.perf_counter() - t_total
    print(f"\nPipeline complete in {elapsed:.1f}s")
    print(f"Report: {report_html.resolve()}")

    if not args.no_open:
        try:
            webbrowser.open(report_html.resolve().as_uri())
        except Exception as exc:  # noqa: BLE001
            print(f"(could not open browser: {exc})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
