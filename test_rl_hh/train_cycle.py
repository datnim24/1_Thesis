"""Time-budgeted single-cycle trainer for test_rl_hh.

Runs one training cycle from a fresh agent for a fixed wall-clock budget.
Saves `cycle<N>_best.pt` (best training-profit episode) and `cycle<N>_final.pt`.

Usage:
    python -m test_rl_hh.train_cycle --cycle 1 --time-sec 1800
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data

from . import configs as C
from .fast_loop import run_fast_episode
from .meta_agent import DuelingDDQNAgent
from .numpy_net import NumpyDuelingDDQN
from .tools import ToolKit


def train_one_cycle(
    cycle: int,
    time_sec: float,
    output_dir: str | Path = "test_rl_hh/outputs",
    episode_budget_for_eps: int = 300_000,
    warm_start: str | Path | None = None,
    epsilon_start: float | None = None,
    lr: float | None = None,
) -> dict:
    """Train one cycle within wall-clock budget.

    warm_start: optional checkpoint path to load as starting weights (fine-tune mode).
    epsilon_start: override initial ε (useful for fine-tune with low exploration).
    episode_budget_for_eps is used only for the ε-decay schedule (should match
    the expected episode count so ε lands at ε_end by the end).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    params = data.to_env_params()
    ups_lambda = data.ups_lambda
    ups_mu = data.ups_mu
    SL = int(params["SL"])
    roasters = list(params["roasters"])

    engine = SimulationEngine(params)
    toolkit = ToolKit(engine, params)
    agent = DuelingDDQNAgent(lr=lr if lr is not None else C.LR)
    if lr is not None:
        print(f"  Override LR = {lr}")
    if warm_start:
        print(f"  Warm-starting from {warm_start}")
        agent.load_checkpoint(warm_start)
        # load_checkpoint restores optimizer state with OLD lr; reset lr explicitly
        if lr is not None:
            for g in agent.optimizer.param_groups:
                g["lr"] = lr
    if epsilon_start is not None:
        agent.epsilon = epsilon_start
        agent._eps_start = epsilon_start
        print(f"  Override initial eps = {epsilon_start}")
    np_net = NumpyDuelingDDQN(agent.online_net)

    log_path = output_dir / f"cycle{cycle}_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "net_profit", "best_profit",
            "epsilon", "buffer_size", "wall_sec", "tool_counts",
        ])

    t0 = time.perf_counter()
    episode = 0
    best_profit = -float("inf")
    best_ep = 0

    best_ckpt = output_dir / f"cycle{cycle}_best.pt"
    final_ckpt = output_dir / f"cycle{cycle}_final.pt"

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= time_sec:
            break

        ups = generate_ups_events(ups_lambda, ups_mu, episode + cycle * 1_000_000, SL, roasters)
        kpi, n_dec, tc = run_fast_episode(
            engine, agent, toolkit, data,
            agent.replay_buffer, ups, training=True, np_net=np_net,
        )
        if len(agent.replay_buffer) >= C.BATCH_SIZE:
            for _ in range(C.TRAINS_PER_EP):
                agent.train_step()
            np_net.sync(agent.online_net)

        agent.decay_epsilon(episode, episode_budget_for_eps)
        episode += 1

        ep_profit = float(kpi.net_profit())
        if ep_profit > best_profit:
            best_profit = ep_profit
            best_ep = episode
            agent.save_checkpoint(best_ckpt)

        if episode % C.LOG_INTERVAL == 0:
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            print(
                f"  [C{cycle} ep={episode:>7d}] "
                f"profit={ep_profit:>10.1f}  "
                f"best={best_profit:>10.1f}  "
                f"eps={agent.epsilon:.3f}  "
                f"buf={len(agent.replay_buffer):>6d}  "
                f"tools={tc}  "
                f"wall={elapsed:.0f}s  "
                f"({eps_per_sec:.0f} ep/s)"
            )
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, round(ep_profit, 2), round(best_profit, 2),
                    round(agent.epsilon, 4), len(agent.replay_buffer),
                    round(elapsed, 1), json.dumps(tc),
                ])

    agent.save_checkpoint(final_ckpt)
    elapsed = time.perf_counter() - t0
    summary = {
        "cycle": cycle,
        "episodes": episode,
        "wall_sec": round(elapsed, 1),
        "best_training_profit": round(best_profit, 2),
        "best_training_episode": best_ep,
        "final_epsilon": round(agent.epsilon, 4),
        "best_ckpt": str(best_ckpt),
        "final_ckpt": str(final_ckpt),
    }
    print(f"\n  Cycle {cycle} training complete: {episode} episodes in {elapsed:.0f}s")
    print(f"  Best training profit: {best_profit:.1f} at ep {best_ep}")
    print(f"  Saved: {best_ckpt} and {final_ckpt}")

    with open(output_dir / f"cycle{cycle}_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train one RL-HH cycle (time-budgeted)")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--time-sec", type=float, required=True)
    parser.add_argument("--output-dir", type=str, default="test_rl_hh/outputs")
    parser.add_argument("--eps-budget", type=int, default=300_000,
                        help="Total episodes that ε-decay schedule targets")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Checkpoint path to load as starting weights")
    parser.add_argument("--epsilon-start", type=float, default=None,
                        help="Override initial ε (e.g. 0.3 for warm-start fine-tune)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: configs.LR=5e-4)")
    args = parser.parse_args()
    train_one_cycle(
        args.cycle, args.time_sec, args.output_dir, args.eps_budget,
        warm_start=args.warm_start, epsilon_start=args.epsilon_start,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
