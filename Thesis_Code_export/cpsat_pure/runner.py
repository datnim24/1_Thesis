"""Runner + CLI for the pure CP-SAT solver.

Orchestrates data loading + UPS-as-downtime injection + model build + solve into
a single call that returns a result dict compatible with master_eval. Also
exposes a CLI:

    python -m cpsat_pure.runner --name SmokeTest --time 60 --seed 42

Outputs land in ``Results/<YYYYMMDD_HHMMSS>_CPSAT_<RunName>/`` (via
``evaluation.result_schema.make_run_dir``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Make the project root importable so ``evaluation.*`` resolves whether this
# file is run as a script or as a module.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .solver import build as build_model
from .solver import load as load_data
from .solver import solve as solve_model

logger = logging.getLogger("cpsat_pure_runner")


def _merge_ups_as_downtime(d: dict[str, Any], ups_events: Iterable[Any]) -> int:
    """Add each UPS event as extra minutes in ``downtime_slots`` for its roaster.

    Mirrors the SimulationEngine's effective DOWN window semantics: the engine
    runs ``_step_roaster_timers`` in the same slot UPS fires (one tick burned
    before next decision point), so the observed DOWN span is
    ``duration - 1`` minutes starting at ``t``.
    """
    shift_length = int(d["shift_length"])
    downtime = d["downtime_slots"]
    applied = 0
    for ev in ups_events:
        if ev is None:
            continue
        roaster_id = getattr(ev, "roaster_id", None) if not isinstance(ev, dict) else ev.get("roaster_id")
        t = getattr(ev, "t", None) if not isinstance(ev, dict) else ev.get("t")
        duration = getattr(ev, "duration", None) if not isinstance(ev, dict) else ev.get("duration")
        if roaster_id is None or t is None or duration is None:
            continue
        if roaster_id not in downtime:
            continue
        effective_duration = max(0, int(duration) - 1)
        start_min = max(0, int(t))
        end_min = min(shift_length, int(t) + effective_duration)
        if end_min <= start_min:
            continue
        downtime[roaster_id].update(range(start_min, end_min))
        applied += 1
    return applied


def run_pure_cpsat(
    time_limit_sec: int | float,
    ups_events: Iterable[Any] | None = None,
    input_dir: str = "Input_data",
    num_workers: int = 8,
    mip_gap: float | None = None,
) -> dict[str, Any]:
    """Load Input_data, merge UPS as planned stops, build and solve CP-SAT."""
    overrides: dict[str, Any] = {"time_limit": int(max(1, time_limit_sec))}
    if mip_gap is not None:
        overrides["mip_gap"] = float(mip_gap)

    d = load_data(input_dir=input_dir, overrides=overrides)

    applied = 0
    ups_list = list(ups_events) if ups_events else []
    if ups_list:
        applied = _merge_ups_as_downtime(d, ups_list)
        logger.info("Merged %d UPS events as planned downtime.", applied)
    d["ups_events_applied"] = applied
    d["ups_events_list"] = ups_list

    t_build = time.perf_counter()
    model, cp_vars = build_model(d)
    build_time = time.perf_counter() - t_build

    result = solve_model(d, model, cp_vars, num_workers=int(num_workers))
    if result is None:
        raise RuntimeError("Pure CP-SAT returned no feasible solution within the time budget.")

    result["ups_events_applied"] = applied
    result["build_time_sec"] = round(build_time, 3)
    return result


def _generate_ups_events(seed: int) -> list[dict]:
    """Generate UPS events for a given seed using env's UPS generator."""
    from env import generate_ups_events, get_sim_params

    params = get_sim_params()
    events = generate_ups_events(
        lambda_rate=float(params.get("ups_lambda", 0)),
        mu_mean=float(params.get("ups_mu", 20)),
        seed=seed,
        shift_length=int(params.get("SL", 480)),
        roasters=list(params.get("roasters", ["R1", "R2", "R3", "R4", "R5"])),
    )
    return [
        {"roaster_id": ev.roaster_id, "t": int(ev.t), "duration": int(ev.duration)}
        for ev in events
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pure CP-SAT solver runner")
    parser.add_argument("--name", required=True, help="Run name (used in output folder)")
    parser.add_argument("--time", type=int, default=300, help="Solver time budget in seconds")
    parser.add_argument("--seed", type=int, default=None, help="UPS seed (omit for deterministic, no UPS)")
    parser.add_argument("--workers", type=int, default=8, help="CP-SAT workers")
    parser.add_argument("--mip-gap", type=float, default=None, help="MIP gap target")
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from evaluation.result_schema import make_run_dir

    out_dir = make_run_dir("CPSAT", args.name)
    print(f"[cpsat] Output -> {out_dir}")

    ups_events = _generate_ups_events(args.seed) if args.seed is not None else None
    if ups_events:
        print(f"[cpsat] Generated {len(ups_events)} UPS events for seed {args.seed}")

    t0 = time.perf_counter()
    result = run_pure_cpsat(
        time_limit_sec=args.time,
        ups_events=ups_events,
        num_workers=args.workers,
        mip_gap=args.mip_gap,
    )
    elapsed = time.perf_counter() - t0

    # Convert raw CP-SAT result to universal schema before saving.
    from evaluation.result_schema import convert_legacy_result
    universal = convert_legacy_result(result, source="auto")

    result_path = out_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(universal, f, indent=2, default=str)
    print(f"[cpsat] Wrote {result_path}")

    if not args.no_report:
        try:
            from evaluation.plot_result import _build_html

            html = _build_html(universal, compare=None, offline=True)
            (out_dir / "report.html").write_text(html, encoding="utf-8")
            print(f"[cpsat] Wrote {out_dir / 'report.html'}")
        except Exception as exc:  # pragma: no cover - best-effort report
            print(f"[cpsat] Report generation failed: {exc}", file=sys.stderr)

    print(f"[cpsat] Done in {elapsed:.1f}s — net_profit = ${result.get('net_profit', 'N/A')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
