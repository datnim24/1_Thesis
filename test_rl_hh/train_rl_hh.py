"""Training script for RL-HH Dueling DDQN.

Uses fast_loop (engine-direct) for ~10-20x speedup over the Gym wrapper.
Trains every TRAIN_EVERY decisions.  Logs and checkpoints periodically.
"""

from __future__ import annotations

import argparse
import csv
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


def train(
    num_episodes: int = C.NUM_EPISODES,
    ups_lambda: float | None = None,
    ups_mu: float | None = None,
    output_dir: str | Path = "test_rl_hh/outputs",
    input_dir: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    data = load_data(input_dir)
    params = data.to_env_params()
    if ups_lambda is None:
        ups_lambda = data.ups_lambda
    if ups_mu is None:
        ups_mu = data.ups_mu

    engine = SimulationEngine(params)
    toolkit = ToolKit(engine, params)
    agent = DuelingDDQNAgent()
    np_net = NumpyDuelingDDQN(agent.online_net)

    shift_length = int(params["SL"])
    roasters = list(params["roasters"])

    log_path = output_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "net_profit", "epsilon", "buffer_size",
            "loss", "decisions", "wall_sec",
        ])

    t0 = time.perf_counter()
    best_profit = -float("inf")
    total_decisions = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for episode in range(num_episodes):
        ups = generate_ups_events(ups_lambda, ups_mu, episode, shift_length, roasters)

        kpi, n_dec, tool_counts = run_fast_episode(
            engine, agent, toolkit, data,
            agent.replay_buffer, ups, training=True,
            np_net=np_net,
        )
        total_decisions += n_dec

        # Train: fixed small number of gradient steps per episode
        if len(agent.replay_buffer) >= C.BATCH_SIZE:
            for _ in range(C.TRAINS_PER_EP):
                agent.train_step()
            np_net.sync(agent.online_net)

        agent.decay_epsilon(episode, num_episodes)

        net_profit = float(kpi.net_profit())

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if episode % C.LOG_INTERVAL == 0:
            elapsed = time.perf_counter() - t0
            eps_per_sec = max(1, episode) / elapsed if elapsed > 0 else 0
            eta_h = (num_episodes - episode) / max(1, eps_per_sec) / 3600
            print(
                f"[{episode:>7d}/{num_episodes}]  "
                f"profit={net_profit:>10.1f}  "
                f"eps={agent.epsilon:.3f}  "
                f"buf={len(agent.replay_buffer):>6d}  "
                f"loss={agent._last_loss:.4f}  "
                f"dec={n_dec:>4d}  "
                f"tools={tool_counts}  "
                f"wall={elapsed:.0f}s  "
                f"({eps_per_sec:.0f} ep/s, ETA {eta_h:.1f}h)"
            )
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, round(net_profit, 2), round(agent.epsilon, 4),
                    len(agent.replay_buffer), round(agent._last_loss, 6),
                    n_dec, round(elapsed, 1),
                ])

        # Checkpoints
        if episode > 0 and episode % C.CHECKPOINT_INTERVAL == 0:
            ckpt_path = output_dir / f"rlhh_ep{episode}.pt"
            agent.save_checkpoint(ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")

        if net_profit > best_profit:
            best_profit = net_profit
            agent.save_checkpoint(output_dir / "rlhh_best.pt")

    # Final checkpoint
    agent.save_checkpoint(output_dir / "rlhh_final.pt")
    elapsed = time.perf_counter() - t0
    print(f"\nTraining complete.  {num_episodes} episodes in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Best profit: {best_profit:.1f}  |  Total decisions: {total_decisions}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RL-HH Dueling DDQN")
    parser.add_argument("--episodes", type=int, default=C.NUM_EPISODES)
    parser.add_argument("--ups-lambda", type=float, default=None, help="UPS lambda (default: from input data)")
    parser.add_argument("--ups-mu", type=float, default=None, help="UPS mu (default: from input data)")
    parser.add_argument("--output-dir", type=str, default="test_rl_hh/outputs")
    parser.add_argument("--input-dir", type=str, default=None)
    args = parser.parse_args()
    train(
        num_episodes=args.episodes,
        ups_lambda=args.ups_lambda,
        ups_mu=args.ups_mu,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
    )


if __name__ == "__main__":
    main()
