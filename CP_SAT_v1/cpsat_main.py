"""CLI entry point for the CP-SAT roasting scheduler."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from data import load
from cpsat_model import build
import cpsat_solver as cpsat_solver_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="roast_cpsat",
        description="CP-SAT deterministic/reactive solver - Nestle Tri An",
    )
    parser.add_argument("--input-dir", default="Input_data")
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--gap", type=float, default=None)
    parser.add_argument("--r3-flex", type=int, choices=[0, 1], default=None)
    parser.add_argument("--disruptions", type=int, choices=[0, 1], default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument("--output-json", default=None, metavar="PATH")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Parallel search workers. Default=1 (deterministic). "
            "Set 4+ for faster search on multi-core (non-deterministic)."
        ),
    )
    parser.add_argument(
        "--lp-only",
        action="store_true",
        help=(
            "Compute LP relaxation bound only (no MIP search). "
            "Useful for fast upper-bound estimation."
        ),
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
    width = 60

    print("=" * width)
    print("  ROASTING SCHEDULE - CP-SAT DETERMINISTIC BASELINE")
    print("=" * width)
    print(f"  Engine  : {report['solver_engine']}")
    print(f"  Status  : {report['status']}")
    print(
        f"  Solve   : {report['solve_time']:.2f}s | "
        f"Incumbents: {report['num_incumbents']}"
    )
    if report["gap_pct"] is not None:
        print(f"  MIP gap : {report['gap_pct']:.4f}%")
    if report["best_bound"] is not None:
        print(f"  Bound   : ${report['best_bound']:>12,.0f}")
    print(f"  R3 mode : {'flexible' if report['allow_r3_flex'] else 'fixed'}")
    print("-" * width)

    history = report.get("solution_history", [])
    if history:
        print(f"\n  INCUMBENT HISTORY ({len(history)} solutions):")
        print(f"  {'#':>3}  {'Obj':>12}  {'Bound':>12}  {'Gap%':>8}  {'Time(s)':>8}")
        print("  " + "-" * 48)
        shown = ([history[0]] if len(history) > 5 else []) + history[-5:]
        seen = set()
        for item in shown:
            key = item["incumbent"]
            if key in seen:
                continue
            seen.add(key)
            print(
                f"  {item['incumbent']:>3}  {item['obj']:>12,.0f}  "
                f"{item['bound']:>12,.0f}  {item['gap_pct']:>8.3f}  "
                f"{item['elapsed_s']:>8.2f}"
            )

    print(f"\n  SCHEDULE ({len(report['schedule'])} active batches):")
    header = (
        f"  {'Batch':<16} {'SKU':<6} {'Rstr':<5} {'Start':>5} "
        f"{'End':>5}  {'Pipeline':<14} {'Out':>5}  Setup"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for entry in report["schedule"]:
        output_line = entry["output_line"] if entry["output_line"] else " -- "
        print(
            f"  {entry['batch_id']:<16} {entry['sku']:<6} {entry['roaster']:<5} "
            f"{entry['start']:>5} {entry['end']:>5}  {entry['pipeline']:<14} "
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
    total_tardiness = sum(report["tardiness_min"].values())

    print("\n" + "-" * width)
    print("  PROFIT REPORT")
    print("-" * width)
    print("  Revenue:")
    print(
        f"    PSC   {report['psc_count']:>3} x ${psc_unit:,.0f} = "
        f"${report['revenue_psc']:>12,.0f}"
    )
    print(
        f"    NDG   {report['ndg_count']:>3} x ${ndg_unit:,.0f} = "
        f"${report['revenue_ndg']:>12,.0f}"
    )
    print(
        f"    BUSTA {report['busta_count']:>3} x ${busta_unit:,.0f} = "
        f"${report['revenue_busta']:>12,.0f}"
    )
    print(f"    {'Total revenue':35} ${report['total_revenue']:>12,.0f}")
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
    print(f"    {'Total costs':35} ${report['total_costs']:>12,.0f}")
    print("-" * width)
    print(f"  NET PROFIT                                ${report['net_profit']:>12,.0f}")
    if report.get("best_bound") is not None:
        print(f"  CP-SAT bound                              ${report['best_bound']:>12,.0f}")
    if report.get("gap_pct") is not None:
        print(f"  Optimality gap                                 {report['gap_pct']:>8.4f}%")
    print("=" * width)


def main() -> int:
    args = parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    configure_logging(args.log_level)
    logger = logging.getLogger("cpsat_main")

    if args.num_workers < 1:
        logger.error("--num-workers must be >= 1")
        return 2

    overrides = {}
    if args.time_limit is not None:
        overrides["time_limit"] = args.time_limit
    if args.gap is not None:
        overrides["mip_gap"] = args.gap
    if args.r3_flex is not None:
        overrides["allow_r3_flex"] = bool(args.r3_flex)
    if args.disruptions is not None:
        overrides["enable_disruptions"] = bool(args.disruptions)

    logger.info("=== Nestle Tri An CP-SAT Solver ===")
    logger.info("Input dir : %s", args.input_dir)
    logger.info("Workers   : %d", args.num_workers)
    if overrides:
        logger.info("Overrides : %s", overrides)

    t_total = time.time()

    t0 = time.time()
    d = load(input_dir=args.input_dir, overrides=overrides)
    logger.info("Data load  : %.2fs", time.time() - t0)

    t1 = time.time()
    model, cp_vars = build(d)
    logger.info("Model build: %.2fs", time.time() - t1)

    cpsat_solver_module.set_verbose(args.verbose)

    if args.lp_only:
        logger.info("--lp-only: computing LP relaxation bound")
        t2 = time.time()
        lp_bound = cpsat_solver_module.solve_lp_relaxation(
            d,
            model,
            time_limit=int(d["time_limit"]),
        )
        logger.info(
            "LP bound   : %s  (%.2fs)",
            f"{lp_bound:,.0f}" if lp_bound is not None else "N/A",
            time.time() - t2,
        )
        logger.info("Total elapsed: %.2fs", time.time() - t_total)
        return 0

    t2 = time.time()
    results = cpsat_solver_module.solve(
        d,
        model,
        cp_vars,
        num_workers=args.num_workers,
    )
    logger.info("Solve time : %.2fs", time.time() - t2)
    logger.info("Total elapsed: %.2fs", time.time() - t_total)

    if results is None:
        logger.error("No solution obtained. Exiting.")
        return 1

    if not args.no_report:
        print_report(results)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        logger.info("Results written to: %s", args.output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
