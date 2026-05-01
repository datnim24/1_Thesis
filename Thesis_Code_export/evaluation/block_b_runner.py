"""Block B factorial driver: 3 lambda × 3 mu × 4 methods × N reps.

Orchestrates 36 cells (4 methods × 9 (λ_mult, μ_mult) cells), each producing
a 100-seed JSON aggregate via test_rl_hh.evaluate_100seeds. Same paired seeds
across cells/methods so per-seed differences are directly attributable to
method × cell, not seed variance.

Usage:
    python block_b_runner.py --reps 100 --output results/block_b_<ts>/
    python block_b_runner.py --reps 5  --dry-run    # skeleton sanity
    python block_b_runner.py --methods rl_hh paeng_ddqn  # restrict to subset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ROOT = _PROJECT_ROOT  # kept for backward-compatible references below
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval_100seeds import evaluate_100_seeds


LAMBDA_MULTS = (0.5, 1.0, 2.0)
MU_MULTS = (0.5, 1.0, 2.0)

DEFAULT_METHODS = ("dispatching", "q_learning", "rl_hh")

# Checkpoint paths must be supplied via --checkpoint-<method> at runtime
# because the new Results/ folders use the timestamp convention.
CHECKPOINTS: dict[str, str | None] = {
    "dispatching":  None,
    "q_learning":   None,
    "rl_hh":        None,
}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {("_".join(str(p) for p in k) if isinstance(k, tuple) else str(k) if not isinstance(k, (str, int, float, bool)) and k is not None else k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        try:
            return sorted(obj)
        except TypeError:
            return [_json_safe(x) for x in obj]
    return obj


def run_block_b(
    methods: tuple[str, ...],
    output_dir: Path,
    n_seeds: int = 100,
    base_seed: int = 900_000,
    dry_run: bool = False,
) -> dict:
    """Loop over (method × λ_mult × μ_mult); save 1 JSON per cell."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(base_seed, base_seed + n_seeds))
    n_cells = len(methods) * len(LAMBDA_MULTS) * len(MU_MULTS)
    cell = 0
    t0 = time.perf_counter()
    cell_files: list[str] = []

    for method in methods:
        ckpt = CHECKPOINTS.get(method)
        if method != "dispatching" and ckpt is not None:
            ckpt_p = _ROOT / ckpt if not Path(ckpt).is_absolute() else Path(ckpt)
            if not ckpt_p.exists():
                print(f"[block-b] WARNING: checkpoint missing for {method}: {ckpt_p}")
                if method == "paeng_ddqn":
                    print("  → train Paeng first (Task 9), then re-run block_b_runner")
                    continue
                if method == "q_learning":
                    print("  → checkpoint required, skipping method")
                    continue

        for lm in LAMBDA_MULTS:
            for mm in MU_MULTS:
                cell += 1
                cell_path = output_dir / f"{method}_lm{lm}_mm{mm}.json"
                print(f"\n[block-b cell {cell}/{n_cells}] {method}  lambda x{lm}  mu x{mm}")
                if dry_run:
                    print(f"  [dry-run] would write {cell_path}")
                    continue

                t_cell = time.perf_counter()
                try:
                    result = evaluate_100_seeds(
                        package=method, seeds=seeds, checkpoint_path=ckpt,
                        lambda_mult=lm, mu_mult=mm,
                    )
                except Exception as exc:
                    print(f"  [error] {method} cell ({lm}, {mm}) failed: {exc}")
                    continue
                dt = time.perf_counter() - t_cell

                with open(cell_path, "w", encoding="utf-8") as f:
                    json.dump(_json_safe(result), f, indent=2)
                cell_files.append(str(cell_path))
                print(f"  mean ${result['profit_mean']:,.0f}  std ${result['profit_std']:,.0f}  ({dt:.1f}s)")

    total_wall = time.perf_counter() - t0
    summary = {
        "timestamp":     datetime.now().isoformat(),
        "n_cells":       cell,
        "methods":       list(methods),
        "lambda_mults":  list(LAMBDA_MULTS),
        "mu_mults":      list(MU_MULTS),
        "n_seeds":       n_seeds,
        "base_seed":     base_seed,
        "total_wall_sec": round(total_wall, 1),
        "cell_files":    cell_files,
    }
    summary_path = output_dir / "block_b_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[block-b] done. {cell} cells in {total_wall:.0f}s. Summary: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Block B 3λ × 3μ × M-method × N-rep factorial.")
    parser.add_argument("--name", default="default", help="Run name (BlockB folder suffix).")
    parser.add_argument("--reps", type=int, default=100, help="Seeds per cell.")
    parser.add_argument("--base-seed", type=int, default=900_000)
    parser.add_argument("--output", type=str, default=None,
                        help="Override output dir (default: Results/<ts>_BlockB_<name>/).")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        choices=list(DEFAULT_METHODS), help="Methods to run.")
    parser.add_argument("--checkpoint-q-learning", default=None,
                        help="Path to Q-learning .pkl (required if 'q_learning' in --methods).")
    parser.add_argument("--checkpoint-rl-hh", default=None,
                        help="Path to RL-HH .pt (required if 'rl_hh' in --methods).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the heavy eval; print cell layout only.")
    args = parser.parse_args()

    # Inject checkpoint overrides
    if args.checkpoint_q_learning:
        CHECKPOINTS["q_learning"] = args.checkpoint_q_learning
    if args.checkpoint_rl_hh:
        CHECKPOINTS["rl_hh"] = args.checkpoint_rl_hh

    from evaluation.result_schema import make_run_dir

    if args.output:
        out_dir = Path(args.output)
        if not out_dir.is_absolute():
            out_dir = _PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = make_run_dir("BlockB", getattr(args, "name", "default"))

    run_block_b(
        methods=tuple(args.methods),
        output_dir=out_dir,
        n_seeds=args.reps,
        base_seed=args.base_seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
