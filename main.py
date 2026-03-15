"""CLI entry point for the deterministic roasting MILP baseline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from data import load
from model import build
import solver as solver_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="roast_milp",
        description="MILP deterministic baseline solver - Nestle Tri An roasting schedule",
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
        help="Override time_limit_sec. E.g. --time-limit 60",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=None,
        help="Override mip_gap_target (fraction). E.g. --gap 0.005 for 0.5%%",
    )
    parser.add_argument(
        "--r3-flex",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override allow_r3_flexible_output. 0=fixed to L2, 1=flexible",
    )
    parser.add_argument(
        "--disruptions",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override enable_disruptions. 0=ignore disruption file, 1=load it",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full solver log (CBC/HiGHS internal output)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Python logging level. Default: INFO",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        metavar="PATH",
        help="Write results dict to this JSON file path",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Suppress profit report from stdout (useful for batch runs)",
    )
    return parser.parse_args()


def configure_logging(level_name: str):
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def print_report(results: dict):
    report = results
    width = 55

    print("=" * width)
    print("  ROASTING SCHEDULE - DETERMINISTIC BASELINE")
    print("=" * width)
    print(f"  Solver  : {report['solver_name']}   Status: {report['status']}")
    if report["gap_pct"] is not None:
        print(f"  Solve   : {report['solve_time']:.2f}s   MIP gap: {report['gap_pct']:.4f}%")
    else:
        print(f"  Solve   : {report['solve_time']:.2f}s   MIP gap: N/A")
    if report["lp_bound"] is not None:
        print(f"  LP bound: ${report['lp_bound']:>12,.0f}")
    else:
        print("  LP bound: N/A")
    print(f"  R3 mode : {'flexible' if report['allow_r3_flex'] else 'fixed'}")
    print("-" * width)

    print(f"\n  SCHEDULE ({len(report['schedule'])} active batches):")
    header = (
        f"  {'Batch':<14} {'SKU':<6} {'Rstr':<5} {'Start':>5} "
        f"{'End':>5}  {'Pipeline':<12} {'Out':>5}  {'Setup'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for entry in report["schedule"]:
        output_line = entry["output_line"] if entry["output_line"] else " -- "
        print(
            f"  {entry['batch_id']:<14} {entry['sku']:<6} {entry['roaster']:<5} "
            f"{entry['start']:>5} {entry['end']:>5}  {entry['pipeline']:<12} "
            f"{output_line:>5}  {entry['setup']}"
        )

    print(f"\n  Active PSC  : {report['psc_count']}")
    print(
        f"  MTO (NDG)   : {report['ndg_count']}  "
        f"(J1: {report['tardiness_min'].get('J1', 0):.0f} min late)"
    )
    print(
        f"  MTO (BUSTA) : {report['busta_count']}  "
        f"(J2: {report['tardiness_min'].get('J2', 0):.0f} min late)"
    )
    print(
        f"  Total active: {report['psc_count'] + report['ndg_count'] + report['busta_count']}"
    )

    psc_unit = report["sku_revenue"].get("PSC", 0.0)
    ndg_unit = report["sku_revenue"].get("NDG", 0.0)
    busta_unit = report["sku_revenue"].get("BUSTA", 0.0)
    tard_unit = report.get("cost_tardiness", 0.0)
    idle_unit = report.get("cost_idle", 0.0)
    over_unit = report.get("cost_overflow", 0.0)
    total_tardiness = report["tardiness_min"].get("J1", 0.0) + report["tardiness_min"].get("J2", 0.0)

    print("\n" + "-" * width)
    print("  PROFIT REPORT")
    print("-" * width)
    print("  Revenue:")
    print(
        f"    PSC   {report['psc_count']:>3} x ${psc_unit:,.0f}  "
        f"= ${report['revenue_psc']:>12,.0f}"
    )
    print(
        f"    NDG   {report['ndg_count']:>3} x ${ndg_unit:,.0f}  "
        f"= ${report['revenue_ndg']:>12,.0f}"
    )
    print(
        f"    BUSTA {report['busta_count']:>3} x ${busta_unit:,.0f}  "
        f"= ${report['revenue_busta']:>12,.0f}"
    )
    print(f"    {'Total revenue':30}  ${report['total_revenue']:>12,.0f}")
    print("  Costs:")
    print(
        f"    Tardiness  {total_tardiness:>5.0f} min x ${tard_unit:,.0f} = "
        f"${report['tard_cost']:>10,.0f}"
    )
    print(
        f"    Safety-idle {report['idle_min']:>4.0f} min x ${idle_unit:,.0f}   = "
        f"${report['idle_cost']:>10,.0f}"
    )
    print(
        f"    Over-idle  {report['over_min']:>4.0f} min x ${over_unit:,.0f}    = "
        f"${report['over_cost']:>10,.0f}"
    )
    print(f"    {'Total costs':30}  ${report['total_costs']:>12,.0f}")
    print("-" * width)
    print(f"  NET PROFIT                            ${report['net_profit']:>12,.0f}")
    if report["lp_bound"] is not None:
        print(f"  LP bound (approx)                     ${report['lp_bound']:>12,.0f}")
    if report["gap_pct"] is not None:
        print(f"  Optimality gap                             {report['gap_pct']:>8.4f}%")
    print("=" * width)


def main() -> int:
    args = parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    configure_logging(args.log_level)
    logger = logging.getLogger("main")
    t_total = time.time()

    overrides = {}
    if args.solver is not None:
        overrides["solver_name"] = args.solver
    if args.time_limit is not None:
        overrides["time_limit"] = args.time_limit
    if args.gap is not None:
        overrides["mip_gap"] = args.gap
    if args.r3_flex is not None:
        overrides["allow_r3_flex"] = bool(args.r3_flex)
    if args.disruptions is not None:
        overrides["enable_disruptions"] = bool(args.disruptions)

    logger.info("=== Nestlé Trị An Roasting MILP - Deterministic Baseline ===")
    logger.info("Input dir : %s", args.input_dir)
    if overrides:
        logger.info("Overrides : %s", overrides)

    t0 = time.time()
    d = load(input_dir=args.input_dir, overrides=overrides)
    logger.info("Data load : %.2fs", time.time() - t0)

    t1 = time.time()
    prob, vars = build(d)
    logger.info("Model build: %.2fs", time.time() - t1)

    solver_module.set_verbose(args.verbose)
    t2 = time.time()
    results = solver_module.solve(d, prob, vars)
    logger.info("Solve time : %.2fs", time.time() - t2)

    if results is None:
        logger.error("No solution obtained. Exiting.")
        logger.info("Total elapsed: %.2fs", time.time() - t_total)
        return 1

    if not args.no_report:
        print_report(results)

    if args.output_json:
        output_path = Path(args.output_json)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        logger.info("Results written to: %s", output_path)

    logger.info("Total elapsed: %.2fs", time.time() - t_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
