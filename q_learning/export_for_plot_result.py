"""Q-learning single-seed exporter — writes a universal-schema JSON consumable
by ``plot_result.py``, optionally also writing the rendered HTML next to it.

Companion to ``q_learning_run.py`` (which renders HTML directly). Use this when
you want a JSON artifact you can re-render or pass to ``plot_result.py FILE.json``.

Usage:
    python -m q_learning.export_for_plot_result \
        --checkpoint q_learning/ql_results/30_03_2026_1230_*/q_table_Test_restockpatch6.pkl \
        --seed 42 \
        --output q_learning/ql_results/30_03_2026_1230_*/ql_seed42_result.json \
        --html
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.data_bridge import get_sim_params
from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events

from q_learning.q_strategy import QStrategy, load_q_table
from q_learning.q_learning_run import _assemble_result, _resolve_file_arg


def _json_safe(obj):
    """Recursively coerce non-JSON-native types (sets, tuples, tuple-keyed
    dicts) so result_schema dicts serialize cleanly. Same pattern as
    ``plot_result._json_safe``.
    """
    if isinstance(obj, dict):
        return {
            ("_".join(str(p) for p in k) if isinstance(k, tuple) else str(k) if not isinstance(k, (str, int, float, bool)) and k is not None else k): _json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        try:
            return sorted(obj)
        except TypeError:
            return [_json_safe(x) for x in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Q-learning single-seed evaluation exported as universal-schema JSON.",
    )
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to Q-table .pkl (default: latest under ql_results/).")
    parser.add_argument("--seed", type=int, default=42, help="UPS realization seed.")
    parser.add_argument("--ups-lambda", type=float, default=None,
                        help="Override UPS lambda (default: Input_data value).")
    parser.add_argument("--ups-mu", type=float, default=None,
                        help="Override UPS mu (default: Input_data value).")
    parser.add_argument("--lambda-mult", type=float, default=1.0,
                        help="Multiplier on Input_data UPS lambda (overrides --ups-lambda if both set).")
    parser.add_argument("--mu-mult", type=float, default=1.0,
                        help="Multiplier on Input_data UPS mu.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path (will also write a sibling _report.html if --html).")
    parser.add_argument("--html", action="store_true",
                        help="Also generate a plot_result HTML next to the JSON.")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open the HTML in browser.")
    args = parser.parse_args(argv)

    params = get_sim_params()
    q_path, log_path, meta, result_dir = _resolve_file_arg(args.checkpoint)
    if q_path is None:
        return 1

    q_table = load_q_table(str(q_path))
    print(f"Loaded Q-table: {q_path} ({len(q_table):,} entries)")

    base_lam = float(params.get("ups_lambda", 0.0))
    base_mu = float(params.get("ups_mu", 0.0))
    lam = float(args.ups_lambda) if args.ups_lambda is not None else base_lam * args.lambda_mult
    mu = float(args.ups_mu) if args.ups_mu is not None else base_mu * args.mu_mult

    print(f"Running seed={args.seed}, lambda={lam:.2f} (x{args.lambda_mult}), mu={mu:.2f} (x{args.mu_mult})")
    ups_events = generate_ups_events(lam, mu, args.seed)
    engine = SimulationEngine(params)
    strategy = QStrategy(params, q_table=q_table)
    kpi, state = engine.run(strategy, ups_events)
    net_profit = float(kpi.net_profit())
    print(f"  Net profit: ${net_profit:,.0f}")

    result: dict[str, Any] = _assemble_result(
        kpi, state, ups_events, params, lam, mu, args.seed, q_path, meta,
    )
    # Tag the multipliers used so the JSON is Block-B-cell-aware
    result["metadata"]["lambda_mult"] = round(args.lambda_mult, 4)
    result["metadata"]["mu_mult"] = round(args.mu_mult, 4)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, indent=2)
    print(f"  JSON saved to: {out_path}")

    if args.html:
        from plot_result import _build_html
        training_info: dict[str, Any] | None = None
        if log_path and Path(log_path).exists():
            training_info = {
                "log_path": str(log_path),
                "epsilon_start": (meta or {}).get("epsilon_start"),
                "epsilon_end": (meta or {}).get("epsilon_end"),
            }
        html = _build_html(result, None, offline=True, training_info=training_info)
        html_path = out_path.with_suffix("").with_name(out_path.stem + "_report.html")
        html_path.write_text(html, encoding="utf-8")
        print(f"  HTML report: {html_path}")
        if not args.no_open:
            try:
                webbrowser.open(html_path.resolve().as_uri())
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
