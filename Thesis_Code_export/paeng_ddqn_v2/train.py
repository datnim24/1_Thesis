"""Paeng DDQN v2 training (period-based, parameter-sharing). [STUB]

This file is a placeholder for the in-development paeng_ddqn_v2 algorithm. The
CLI surface is fixed so master_eval and scripts can call it without changes
once the implementation lands.

Planned design (Paeng 2021):
- Period-based env wrapper (decisions every 5 minutes, not every slot).
- Family-Based State (FBS) features — per-job-family aggregates.
- Parameter-sharing Dueling DDQN across roasters (single shared network).
- Reward shaping per Paeng 2021 §3 (utilization + tardiness blend).

Usage (planned):
    python -m paeng_ddqn_v2.train --name SmokeTest --time 600 --seed 42
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
        description="Train paeng_ddqn_v2 (period-based DDQN). [STUB — not yet implemented]"
    )
    parser.add_argument("--name", required=True, help="Run name (used in Results/ folder)")
    parser.add_argument("--time", type=int, default=600, help="Wall-clock budget in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dir", default=None)
    args = parser.parse_args(argv)

    from evaluation.result_schema import make_run_dir

    out_dir = make_run_dir("PaengDDQNv2", args.name)
    print(f"[paeng_ddqn_v2.train] Output dir reserved: {out_dir}")
    print("[paeng_ddqn_v2.train] STATUS: in development — implementation not yet landed.")
    raise NotImplementedError(
        "paeng_ddqn_v2 is in active development. See module docstring for the "
        "planned design. Use rl_hh.train for the current canonical RL method."
    )


if __name__ == "__main__":
    sys.exit(main())
