"""CLI entry point and experiment runner for Reactive CP-SAT Oracle.

Usage:
    python -m Reactive_CPSAT.runner --time 120 --name "test" --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("reactive_cpsat.runner")


def _setup_logging(log_path: Path, log_level: str) -> None:
    """Configure root logger with file + stdout handlers."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    sh.setFormatter(fmt)
    root.addHandler(sh)


def _heartbeat_thread(
    stop_event: threading.Event,
    state: dict[str, Any],
    interval_sec: int = 600,
) -> None:
    """Background thread that logs progress every `interval_sec` seconds."""
    while not stop_event.wait(timeout=interval_sec):
        i = state.get("current_index", 0)
        n = state.get("total_events", 0)
        elapsed_min = (time.perf_counter() - state["start_time"]) / 60
        last_profit = state.get("last_profit", "N/A")
        last_gap = state.get("last_gap", "N/A")
        logger.info(
            "Progress: %d/%d UPS events solved | elapsed: %.1fm | "
            "last oracle: %s | gap: %s%%",
            i, n, elapsed_min, last_profit, last_gap,
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full reactive oracle pipeline."""

    # ── Output directory ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.name}"
    output_dir = Path(__file__).resolve().parent / "results" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ──────────────────────────────────────────────────────────
    _setup_logging(output_dir / "run.log", args.log_level)
    logger.info("=" * 60)
    logger.info("Reactive CP-SAT Oracle Runner")
    logger.info("=" * 60)
    logger.info("Args: %s", vars(args))
    logger.info("Output: %s", output_dir)

    # ── Step 1: Load all parameters from input data ────────────────────
    logger.info("Step 1: Loading parameters from %s ...", args.input_dir)

    from Reactive_CPSAT.data import load as cpsat_load
    from env.data_bridge import get_sim_params

    sim_params = get_sim_params(args.input_dir)

    # Resolve every parameter: input data first, CLI override if explicitly provided
    lambda_rate = args.lambda_rate if args.lambda_rate is not None else float(sim_params.get("ups_lambda", 2.0))
    mu_mean = args.mu_mean if args.mu_mean is not None else float(sim_params.get("ups_mu", 20.0))
    r3_flex = bool(args.r3_flex) if args.r3_flex is not None else bool(sim_params.get("allow_r3_flex", False))

    # time_limit: read from solver_config.csv via cpsat_load (will be loaded below)
    # Use a temporary load just to get the configured value
    _tmp = cpsat_load(input_dir=args.input_dir)
    time_limit = args.time if args.time is not None else int(_tmp.get("time_limit", 120))

    logger.info(
        "Resolved params (input data + CLI overrides): "
        "lambda=%.2f, mu=%.1f, time_limit=%ds, r3_flex=%s, seed=%d",
        lambda_rate, mu_mean, time_limit, r3_flex, args.seed,
    )

    base_params = cpsat_load(
        input_dir=args.input_dir,
        overrides={
            "time_limit": time_limit,
            "allow_r3_flex": r3_flex,
        },
    )
    logger.info(
        "Loaded: %d roasters, %d jobs, %d MTO batches, %d PSC pool, SL=%d",
        len(base_params["roasters"]),
        len(base_params["jobs"]),
        len(base_params["mto_batches"]),
        len(base_params["psc_pool"]),
        base_params["shift_length"],
    )

    # ── Step 2: Generate UPS events ──────────────────────────────────────
    logger.info(
        "Step 2: Generating UPS events (lambda=%.2f, mu=%.1f, seed=%d) ...",
        lambda_rate, mu_mean, args.seed,
    )

    from env.ups_generator import generate_ups_events

    ups_events = generate_ups_events(
        lambda_rate=lambda_rate,
        mu_mean=mu_mean,
        seed=args.seed,
        shift_length=int(base_params["shift_length"]),
        roasters=base_params["roasters"],
    )
    logger.info("Generated %d UPS events", len(ups_events))
    for ev in ups_events:
        logger.info("  UPS: t=%d, roaster=%s, duration=%d", ev.t, ev.roaster_id, ev.duration)

    # ── Step 3: Run baseline simulation ──────────────────────────────────
    logger.info("Step 3: Running baseline simulation (DispatchingHeuristic) ...")

    from dispatch.dispatching_heuristic import DispatchingHeuristic
    from env.simulation_engine import SimulationEngine
    engine = SimulationEngine(sim_params)
    strategy = DispatchingHeuristic(sim_params)
    kpi, state = engine.run(strategy, ups_events)

    baseline_kpi = kpi.to_dict()
    logger.info("Baseline result: net_profit=$%s", f'{baseline_kpi["net_profit"]:,.0f}')
    logger.info(
        "  PSC=%d, NDG=%d, BUSTA=%d, setups=%d, stockout_events=%s",
        baseline_kpi.get("psc_count", 0),
        baseline_kpi.get("ndg_count", 0),
        baseline_kpi.get("busta_count", 0),
        baseline_kpi.get("setup_events", 0),
        baseline_kpi.get("stockout_events", {}),
    )
    logger.info("UPS events actually fired: %d", len(state.ups_events_fired))

    # Serialize baseline schedule for full-shift Gantt
    baseline_schedule = [
        {
            "batch_id": str(b.batch_id),
            "sku": b.sku,
            "roaster": b.roaster,
            "start": int(b.start),
            "end": int(b.end),
            "output_line": b.output_line,
            "is_mto": bool(b.is_mto),
            "status": "completed",
        }
        for b in state.completed_batches
    ]
    baseline_cancelled = [
        {
            "batch_id": str(b.batch_id),
            "sku": b.sku,
            "roaster": b.roaster,
            "start": int(b.start),
            "end": int(b.end),
            "output_line": b.output_line,
            "is_mto": bool(b.is_mto),
            "status": "cancelled",
        }
        for b in state.cancelled_batches
    ]
    baseline_restocks = [
        {
            "line_id": r.line_id,
            "sku": r.sku,
            "start": int(r.start),
            "end": int(r.end),
            "qty": int(r.qty),
        }
        for r in state.completed_restocks
    ]
    baseline_ups = [
        {
            "t": int(ev.t),
            "roaster_id": str(ev.roaster_id),
            "duration": int(ev.duration),
        }
        for ev in state.ups_events_fired
    ]

    # ── Step 4: Oracle solves ────────────────────────────────────────────
    logger.info("Step 4: Running oracle for each UPS event ...")

    from .oracle import ReactiveCPSATOracle

    oracle = ReactiveCPSATOracle(base_params, time_limit_sec=time_limit)

    # Heartbeat thread
    heartbeat_state: dict[str, Any] = {
        "start_time": time.perf_counter(),
        "total_events": len(state.ups_events_fired),
        "current_index": 0,
    }
    stop_heartbeat = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_thread,
        args=(stop_heartbeat, heartbeat_state),
        daemon=True,
    )
    hb_thread.start()

    # Global deadline: if set, shrink per-solve TL as time runs out
    global_deadline = getattr(args, "global_deadline", None)

    oracle_results: list[dict[str, Any]] = []
    for i, ev in enumerate(state.ups_events_fired):
        # Check global deadline
        if global_deadline is not None:
            remaining_global = global_deadline - time.perf_counter()
            if remaining_global < 30:
                logger.warning(
                    "Global deadline reached — skipping remaining %d/%d UPS events",
                    len(state.ups_events_fired) - i, len(state.ups_events_fired),
                )
                break
            # Shrink per-solve TL to fit remaining events within budget
            events_left = len(state.ups_events_fired) - i
            adaptive_tl = max(30, int(remaining_global * 0.9 / events_left))
            oracle.time_limit_sec = min(time_limit, adaptive_tl)
            logger.info(
                "Adaptive TL: %ds (%.0fs remaining, %d events left)",
                oracle.time_limit_sec, remaining_global, events_left,
            )

        t0 = int(ev.t)
        if t0 >= len(state.trace):
            logger.error("UPS at t=%d outside trace range (%d) — skipping", t0, len(state.trace))
            continue

        trace_at_t0 = state.trace[t0]

        from .snapshot import reconstruct_mto_remaining

        mto_remaining = reconstruct_mto_remaining(
            completed_batches=state.completed_batches,
            cancelled_batches=state.cancelled_batches,
            trace_at_t0=trace_at_t0,
            base_params=base_params,
            ups_roaster=str(ev.roaster_id),
        )

        result = oracle.solve_from_snapshot(trace_at_t0, ev, mto_remaining)
        result["ups_index"] = i
        oracle_results.append(result)

        # Update heartbeat state
        heartbeat_state["current_index"] = i + 1
        if result["oracle_profit"] is not None:
            heartbeat_state["last_profit"] = f"${result['oracle_profit']:,.0f}"
            heartbeat_state["last_gap"] = result.get("gap_pct", "N/A")

    stop_heartbeat.set()
    hb_thread.join(timeout=2)

    # ── Step 5: Aggregate results ────────────────────────────────────────
    logger.info("Step 5: Aggregating results ...")

    profits = [r["oracle_profit"] for r in oracle_results if r["oracle_profit"] is not None]
    solve_times = [r["solve_time"] for r in oracle_results]
    feasible_count = sum(1 for r in oracle_results if r["oracle_status"] != "Infeasible")
    optimal_count = sum(1 for r in oracle_results if r["oracle_status"] == "Optimal")

    aggregate = {
        "mean_oracle_profit": round(sum(profits) / len(profits), 2) if profits else None,
        "mean_solve_time": round(sum(solve_times) / len(solve_times), 2) if solve_times else 0.0,
        "feasible_solves": feasible_count,
        "optimal_solves": optimal_count,
    }

    # ── Build full result ────────────────────────────────────────────────
    full_result: dict[str, Any] = {
        "run_name": args.name,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "lambda_rate": lambda_rate,
        "mu_mean": mu_mean,
        "time_limit_sec": time_limit,
        "r3_flex": r3_flex,
        "baseline_kpi": baseline_kpi,
        "baseline_schedule": baseline_schedule,
        "baseline_cancelled": baseline_cancelled,
        "baseline_restocks": baseline_restocks,
        "baseline_ups": baseline_ups,
        "ups_events_total": len(state.ups_events_fired),
        "oracle_results": oracle_results,
        "aggregate": aggregate,
    }

    # ── Step 6: Save outputs ─────────────────────────────────────────────
    result_path = output_dir / "reactive_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, default=str)
    logger.info("Saved: %s", result_path)

    # ── Step 7: Generate HTML report via plot_result.py ─────────────────
    try:
        import plot_result

        report_path = output_dir / "report.html"
        html = plot_result._build_html(
            plot_result._load_any_result(result_path),
            compare=None,
            offline=True,
        )
        report_path.write_text(html, encoding="utf-8")
        logger.info("Report: %s", report_path)
        plot_result._open_in_browser(report_path)
    except Exception as exc:
        logger.warning("Report generation failed: %s", exc, exc_info=True)

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.perf_counter() - heartbeat_state["start_time"]
    logger.info("=" * 60)
    logger.info("DONE in %.1f minutes", total_time / 60)
    logger.info("Baseline profit:       $%s", f'{baseline_kpi["net_profit"]:,.0f}')
    if aggregate["mean_oracle_profit"] is not None:
        logger.info("Mean oracle profit:    $%s", f'{aggregate["mean_oracle_profit"]:,.0f}')
    logger.info("Feasible/Optimal:      %d / %d", feasible_count, optimal_count)
    logger.info("Output directory:       %s", output_dir)
    logger.info("=" * 60)

    return full_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reactive CP-SAT Oracle — offline re-optimisation at each UPS event",
    )
    # All defaults are None → loaded from input data at runtime
    parser.add_argument("--time", type=int, default=None, help="CP-SAT time limit per solve (seconds). Default: from solver_config.csv")
    parser.add_argument("--name", type=str, default="run", help="Run name for output folder")
    parser.add_argument("--lambda-rate", type=float, default=None, help="UPS inter-arrival rate. Default: from shift_parameters.csv (ups_lambda)")
    parser.add_argument("--mu-mean", type=float, default=None, help="Mean UPS duration (minutes). Default: from shift_parameters.csv (ups_mu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for UPS generation")
    parser.add_argument("--r3-flex", type=int, default=None, choices=[0, 1], help="R3 routing mode. Default: from solver_config.csv")
    parser.add_argument("--input-dir", type=str, default="Input_data", help="Input data directory")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
