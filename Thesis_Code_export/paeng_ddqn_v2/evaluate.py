"""Paeng DDQN v2 evaluation. [STUB]

Placeholder companion to ``paeng_ddqn_v2.train``. Expected behaviour:
load a trained checkpoint, run a greedy episode through SimulationEngine,
write ``result.json`` + ``report.html`` to the Results/ run directory.

Usage (planned):
    python -m paeng_ddqn_v2.evaluate --checkpoint <path> --seed 42 --name SmokeEval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate paeng_ddqn_v2 checkpoint. [STUB]"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dir", default=None)
    args = parser.parse_args(argv)

    from evaluation.result_schema import make_run_dir

    out_dir = make_run_dir("Eval_PaengDDQNv2", args.name)
    print(f"[paeng_ddqn_v2.evaluate] Output dir reserved: {out_dir}")
    print("[paeng_ddqn_v2.evaluate] STATUS: in development — implementation not yet landed.")
    raise NotImplementedError(
        "paeng_ddqn_v2 evaluation is in active development."
    )


if __name__ == "__main__":
    sys.exit(main())
