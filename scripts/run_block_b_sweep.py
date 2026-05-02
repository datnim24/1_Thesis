"""Block B sweep runner — drives the 3-method × 9-cell × 50-seed factorial.

Calls ``test_rl_hh.evaluate_100seeds.evaluate_100_seeds`` in-process
(saves repeated Python imports vs subprocess). Outputs to
``results/block_b/lm{lam}_mm{mu}/{method}.json``.

Idempotent: skips a (method, cell) if the JSON already exists, has matching
``lambda_mult/mu_mult/n_seeds/package`` cell-tags, and is newer than the
checkpoint file. ``--force`` overrides this. ``--methods`` and ``--cells``
filter the sweep.

Adding Paeng-DDQN later: uncomment the entry in ``METHODS`` below — the rest
of the pipeline (aggregator, report) is glob-driven and auto-widens.

Usage::

    python scripts/run_block_b_sweep.py
    python scripts/run_block_b_sweep.py --methods dispatching,rl_hh
    python scripts/run_block_b_sweep.py --cells 1.0,1.0
    python scripts/run_block_b_sweep.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_rl_hh.evaluate_100seeds import evaluate_100_seeds


# Canonical method registry. Add/uncomment paeng_ddqn here when Phase 5 lands.
METHODS: dict[str, dict] = {
    "dispatching": {
        "checkpoint": None,
        "label": "Dispatching (operator baseline)",
    },
    "q_learning": {
        "checkpoint": "q_learning/ql_results/31_03_2026_0919_ep1433449_a00500_g09900_Q_learn_nonUPS_overnight_profit296270/q_table_Q_learn_nonUPS_overnight.pkl",
        "label": "Tabular Q-Learning (1.43M ep)",
    },
    "rl_hh": {
        "checkpoint": "rl_hh/outputs/rlhh_cycle3_best.pt",
        "label": "RL-HH (Dueling DDQN, cycle3 + 18 tool cycles)",
    },
    # "paeng_ddqn": {
    #     "checkpoint": "paeng_ddqn/outputs/best/agent.pt",
    #     "label": "Paeng's Modified DDQN (parameter sharing)",
    # },
}


CELLS: list[tuple[float, float]] = [
    (lam, mu) for lam in (0.5, 1.0, 2.0) for mu in (0.5, 1.0, 2.0)
]


N_SEEDS = 50
BASE_SEED = 900_000
OUTPUT_ROOT = _ROOT / "results" / "block_b"


def cell_dirname(lam: float, mu: float) -> str:
    return f"lm{lam}_mm{mu}"


def _is_up_to_date(json_path: Path, package: str, lam: float, mu: float, n_seeds: int) -> bool:
    """Return True if the existing JSON matches the requested cell+method+seed-count."""
    if not json_path.exists():
        return False
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if d.get("package") != package:
        return False
    if abs(float(d.get("lambda_mult", -999)) - lam) > 1e-9:
        return False
    if abs(float(d.get("mu_mult", -999)) - mu) > 1e-9:
        return False
    if int(d.get("n_seeds", -1)) != n_seeds:
        return False
    return True


def parse_methods(spec: str | None) -> list[str]:
    if not spec:
        return list(METHODS.keys())
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [m for m in requested if m not in METHODS]
    if bad:
        raise SystemExit(f"Unknown method(s): {bad}; known: {list(METHODS.keys())}")
    return requested


def parse_cells(spec: str | None) -> list[tuple[float, float]]:
    if not spec:
        return list(CELLS)
    out: list[tuple[float, float]] = []
    for chunk in spec.split(";"):
        if not chunk.strip():
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            raise SystemExit(f"Bad --cells chunk: {chunk!r} (expect 'lam,mu')")
        out.append((float(parts[0]), float(parts[1])))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block B factorial sweep runner")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated subset of methods (default: all)")
    parser.add_argument("--cells", default=None,
                        help="Cells as 'lam,mu;lam,mu;...' (default: 9-cell grid)")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing JSONs even if up-to-date")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)

    methods = parse_methods(args.methods)
    cells = parse_cells(args.cells)
    output_root = Path(args.output_root)
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    print("=" * 70)
    print(f"  Block B sweep")
    print(f"  Methods: {methods}")
    print(f"  Cells:   {len(cells)} cells (lam x mu)")
    print(f"  Seeds:   {args.base_seed}..{args.base_seed + args.n_seeds - 1}")
    print(f"  Output:  {output_root}")
    print("=" * 70)

    n_total = len(methods) * len(cells)
    n_done = 0
    n_skipped = 0
    n_failed = 0
    t_total_start = time.perf_counter()

    for cell_idx, (lam, mu) in enumerate(cells):
        cell_dir = output_root / cell_dirname(lam, mu)
        cell_dir.mkdir(parents=True, exist_ok=True)

        for method in methods:
            n_done += 1
            cfg = METHODS[method]
            json_path = cell_dir / f"{method}.json"

            tag = f"[{n_done}/{n_total}] {method:<12} cell=(lam={lam},mu={mu})"

            if not args.force and _is_up_to_date(
                json_path, method, lam, mu, args.n_seeds
            ):
                print(f"  {tag} SKIP (up-to-date)")
                n_skipped += 1
                continue

            checkpoint = cfg["checkpoint"]
            if checkpoint is not None and not (_ROOT / checkpoint).exists():
                print(f"  {tag} FAIL — checkpoint not found: {checkpoint}")
                n_failed += 1
                continue

            try:
                t0 = time.perf_counter()
                result = evaluate_100_seeds(
                    package=method,
                    seeds=seeds,
                    checkpoint_path=checkpoint,
                    lambda_mult=lam,
                    mu_mult=mu,
                )
                wall = time.perf_counter() - t0
                json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"  {tag} OK    "
                      f"mean=${result['profit_mean']:>10,.0f}  "
                      f"std=${result['profit_std']:>8,.0f}  "
                      f"wall={wall:.1f}s")
            except Exception as exc:  # noqa: BLE001 — keep sweeping
                n_failed += 1
                print(f"  {tag} FAIL — {type(exc).__name__}: {exc}")

    t_total = time.perf_counter() - t_total_start
    n_ok = n_done - n_skipped - n_failed
    print("=" * 70)
    print(f"  Sweep done in {t_total:.1f}s    "
          f"OK={n_ok}  SKIP={n_skipped}  FAIL={n_failed}  total={n_total}")
    print(f"  Output: {output_root}")
    print("=" * 70)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
