"""CLI entry point for MILP_Test_v5."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

try:
    from .data import load
    from .model import build
    from . import solver as solver_module
except ImportError:
    from data import load
    from model import build
    import solver as solver_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="roast_milp_v5",
        description="MILP deterministic baseline with RC + GC inventory management and no UPS.",
    )
    parser.add_argument(
        "--input-dir",
        default="Input_data",
        help="Path to folder containing CSV input files. Default: ./Input_data",
    )
    parser.add_argument(
        "--solver",
        choices=["CBC", "HiGHS"],
        default=None,
        help="Override solver_name from solver_config.csv",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Override time_limit_sec.",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=None,
        help="Override mip_gap_target (fraction).",
    )
    parser.add_argument(
        "--presolve-mode",
        choices=["on", "off", "choose"],
        default=None,
        help="Override HiGHS presolve mode.",
    )
    parser.add_argument(
        "--r3-flex",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override allow_r3_flexible_output. 0=fixed to L2, 1=flexible.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full solver log.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Python logging level.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        metavar="PATH",
        help="Write result dict to this JSON file path.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Suppress the stdout report.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate and open the interactive HTML report automatically after a successful solve.",
    )
    return parser.parse_args()


def _default_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output_json:
        json_path = Path(args.output_json)
        if not json_path.is_absolute():
            json_path = PROJECT_ROOT / json_path
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = RESULTS_DIR / f"milp_v5_result_{timestamp}.json"
    html_path = RESULTS_DIR / f"{json_path.stem}_plot.html"
    return json_path, html_path


def _generate_html_report(json_path: Path, html_path: Path, logger: logging.Logger) -> bool:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "plot_result.py"),
        str(json_path),
        "--output",
        str(html_path),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        logger.exception("Failed to generate interactive HTML report from %s", json_path)
        logger.error("plot_result.py exited with code %s", exc.returncode)
        return False
    return True


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def print_report(result: dict) -> None:
    width = 62
    print("=" * width)
    print("  MILP_TEST_V5 - DETERMINISTIC INVENTORY-MANAGED BASELINE")
    print("=" * width)
    print(f"  Solver        : {result['solver_name']}   Status: {result['status']}")
    if result["gap_pct"] is not None:
        print(f"  Solve         : {result['solve_time']:.2f}s   MIP gap: {result['gap_pct']:.4f}%")
    else:
        print(f"  Solve         : {result['solve_time']:.2f}s   MIP gap: N/A")
    node_count = result.get("node_count")
    print(f"  Node count    : {node_count if node_count is not None else 'N/A'}")
    print(f"  Objective     : ${result.get('objective_profit', 0):>12,.0f}")
    print(f"  Reported P/L  : ${result['net_profit']:>12,.0f}")
    if result["lp_bound"] is not None:
        print(f"  LP bound      : ${result['lp_bound']:>12,.0f}")
    print(f"  R3 mode       : {'flexible' if result['allow_r3_flex'] else 'fixed'}")
    print(f"  Restocks      : {result['restock_count']}")
    print("-" * width)

    print(f"\n  SCHEDULE ({len(result['schedule'])} active batches):")
    header = (
        f"  {'Batch':<14} {'SKU':<6} {'Rstr':<5} {'Start':>5} {'End':>5}  "
        f"{'Pipe':<4} {'Out':<3} {'Setup':<8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for entry in result["schedule"]:
        out = entry["output_line"] if entry["output_line"] else "--"
        print(
            f"  {entry['batch_id']:<14} {entry['sku']:<6} {entry['roaster']:<5} "
            f"{entry['start']:>5} {entry['end']:>5}  "
            f"{entry['pipeline']:<4} {out:<3} {entry['setup']:<8}"
        )

    if result["restocks"]:
        print("\n  RESTOCKS:")
        for rst in result["restocks"]:
            print(
                f"  {rst['line_id']}_{rst['sku']:<6} start={rst['start']:>3} "
                f"end={rst['end']:>3} qty={rst['qty']}"
            )

    print("\n" + "-" * width)
    print("  KPI REPORT")
    print("-" * width)
    print(f"  Revenue PSC   : ${result['revenue_psc']:>12,.0f} ({result['psc_count']} batches)")
    print(f"  Revenue NDG   : ${result['revenue_ndg']:>12,.0f} ({result['ndg_count']} batches)")
    print(f"  Revenue BUSTA : ${result['revenue_busta']:>12,.0f} ({result['busta_count']} batches)")
    print(f"  Total revenue : ${result['total_revenue']:>12,.0f}")
    print(f"  Tardiness cost: ${result['tard_cost']:>12,.0f}  {result['tardiness_min']}")
    print(f"  Setup cost    : ${result['setup_cost']:>12,.0f}  ({result['setup_events']} events)")
    print(f"  Idle cost     : ${result['idle_cost']:>12,.0f}  ({result['idle_min']:.0f} min)")
    print(f"  Overflow cost : ${result['over_cost']:>12,.0f}  ({result['over_min']:.0f} min)")
    print(f"  Total costs   : ${result['total_costs']:>12,.0f}")
    print(f"  Net profit    : ${result['net_profit']:>12,.0f}")

    print("\n  FINAL INVENTORY")
    print(f"  RC final      : {result['rc_final']}")
    print(f"  GC final      : {result['gc_final']}")

    if result.get("model_notes"):
        print("\n  MODEL NOTES")
        for note in result["model_notes"]:
            print(f"  - {note}")
    print("=" * width)


def main() -> int:
    args = parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    configure_logging(args.log_level)
    logger = logging.getLogger("main")
    total_start = time.time()

    overrides = {}
    if args.solver is not None:
        overrides["solver_name"] = args.solver
    if args.time_limit is not None:
        overrides["time_limit"] = args.time_limit
    if args.gap is not None:
        overrides["mip_gap"] = args.gap
    if args.presolve_mode is not None:
        overrides["presolve_mode"] = args.presolve_mode
    if args.r3_flex is not None:
        overrides["allow_r3_flex"] = bool(args.r3_flex)

    logger.info("=== MILP_Test_v5 ===")
    logger.info("Input dir : %s", args.input_dir)
    if overrides:
        logger.info("Overrides : %s", overrides)

    t0 = time.time()
    data = load(input_dir=args.input_dir, overrides=overrides or None)
    logger.info("Data load  : %.2fs", time.time() - t0)

    t1 = time.time()
    prob, vars_dict = build(data)
    logger.info("Model build: %.2fs", time.time() - t1)

    solver_module.set_verbose(args.verbose)
    t2 = time.time()
    result = solver_module.solve(data, prob, vars_dict)
    logger.info("Solve time : %.2fs", time.time() - t2)

    if result is None:
        logger.error("No solution obtained.")
        logger.info("Total time : %.2fs", time.time() - total_start)
        return 1

    if not args.no_report:
        print_report(result)

    output_path, html_path = _default_output_paths(args)
    result["_path"] = output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in result.items() if k != "_path"}, handle, indent=2)
    logger.info("Results written to: %s", output_path)

    if not args.no_plot:
        logger.info("Generating interactive HTML report...")
        if _generate_html_report(output_path, html_path, logger):
            logger.info("HTML report saved to: %s", html_path)
        else:
            return 1
    else:
        logger.info("Auto-plot disabled (--no-plot).")

    logger.info("Total time : %.2fs", time.time() - total_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
