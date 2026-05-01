"""Paeng DDQN v2 training CLI.

Period-based DDQN training with 100k episode target per Paeng 2021 paper.

Usage:
    python -m paeng_ddqn_v2.train_v2 --time-sec 10800 --output-dir paeng_ddqn_v2/outputs/cycle1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from PPOmask.Engine.data_loader import load_data

from paeng_ddqn_v2.agent_v2 import PaengAgentV2, PaengConfigV2
from paeng_ddqn_v2.strategy_v2 import PaengStrategyV2


def train(
    time_sec: float,
    output_dir: Path,
    seed_base: int = 42,
    use_ups: bool = True,
    target_episode_estimate: int = 100_000,
    snapshot_every: int = 0,
    rolling_window: int = 0,
    restore_drop_threshold: float = 50_000.0,
    load_ckpt: str | None = None,
    initial_epsilon: float | None = None,
    run_seed: int | None = None,
) -> dict:
    """Time-budgeted training loop for Paeng DDQN v2."""

    if run_seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(run_seed)
        _np.random.seed(run_seed)
        try:
            import torch as _torch
            _torch.manual_seed(run_seed)
        except ImportError:
            pass

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    ups_lambda = float(data.ups_lambda)
    ups_mu = float(data.ups_mu)

    cfg = PaengConfigV2()
    cfg.target_episode_estimate = target_episode_estimate
    agent = PaengAgentV2(cfg)

    if load_ckpt is not None:
        agent.load_checkpoint(load_ckpt)
        print(f"[paeng-train-v2] warm-start: loaded {load_ckpt}; epsilon={agent.epsilon:.3f}")

    if initial_epsilon is not None:
        agent.epsilon = float(initial_epsilon)
        print(f"[paeng-train-v2] initial epsilon override: {agent.epsilon:.3f}")

    strategy = PaengStrategyV2(agent, data, training=True, params=params)
    engine = SimulationEngine(params)

    log_path = output_dir / "training_log.csv"
    summary_path = output_dir / "training_summary.json"
    best_ckpt = output_dir / "paeng_v2_best.pt"
    final_ckpt = output_dir / "paeng_v2_final.pt"
    best_rolling_ckpt = output_dir / "paeng_v2_best_rolling.pt"
    snapshot_dir = output_dir / "snapshots"
    if snapshot_every > 0:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths: list[str] = []

    rolling: deque[float] = deque(maxlen=rolling_window) if rolling_window > 0 else deque()
    best_rolling_mean = -float("inf")
    best_rolling_episode = -1
    n_restores = 0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "episode", "profit", "tardiness_cost", "period_decisions", "epsilon",
            "buffer_size", "wall_sec", "loss_avg", "action_dist",
        ])

    t0 = time.perf_counter()
    episode = 0
    best_profit = -float("inf")
    best_episode = 0
    losses_window: list[float] = []
    period_decisions_per_ep = 0

    print(f"[paeng-train-v2] starting: budget={time_sec:.0f}s, target_episodes={target_episode_estimate}, "
          f"ups={use_ups}, lambda={ups_lambda}, mu={ups_mu}, seed_base={seed_base}")

    try:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= time_sec or episode >= target_episode_estimate:
                break

            seed = seed_base + episode
            ups = generate_ups_events(ups_lambda, ups_mu, seed, SL, roasters) if use_ups else []

            strategy.reset_episode()
            strategy.action_counts = {i: 0 for i in range(9)}

            kpi, sim_state = engine.run(strategy, ups)
            final_profit = float(kpi.net_profit())
            tardiness_cost = float(kpi.tard_cost)
            period_decisions_per_ep = sum(strategy.action_counts.values())

            strategy.end_episode(sim_state, final_profit, tardiness_cost)

            if initial_epsilon is None:
                agent.decay_epsilon(episode, cfg.target_episode_estimate)

            if episode % cfg.freq_target_episodes == 0:
                agent.update_target_network()

            if len(agent.replay) >= cfg.batch_size and agent.timestep > cfg.warmup_timesteps:
                losses_window.append(agent.train_step())
                if len(losses_window) > 100:
                    losses_window.pop(0)
            loss_avg = sum(losses_window) / len(losses_window) if losses_window else 0.0

            if final_profit > best_profit:
                best_profit = final_profit
                best_episode = episode
                agent.save_checkpoint(best_ckpt)

            # Rolling-mean collapse-restore
            if rolling_window > 0:
                rolling.append(final_profit)
                if len(rolling) >= rolling_window:
                    rolling_mean = sum(rolling) / len(rolling)
                    if rolling_mean > best_rolling_mean:
                        best_rolling_mean = rolling_mean
                        best_rolling_episode = episode
                        agent.save_checkpoint(best_rolling_ckpt)
                    # Restore if rolling mean drops too far
                    if (
                        agent.epsilon <= cfg.eps_end + 1e-9
                        and best_rolling_mean > -1e9
                        and rolling_mean < best_rolling_mean - restore_drop_threshold
                        and best_rolling_ckpt.exists()
                    ):
                        agent.load_checkpoint(best_rolling_ckpt)
                        rolling.clear()
                        n_restores += 1
                        print(f"  [restore #{n_restores}] ep{episode}  rolling=${rolling_mean:>10,.0f}  "
                              f"best=${best_rolling_mean:>10,.0f} (ep{best_rolling_episode})")

            # Snapshots
            if snapshot_every > 0 and agent.epsilon <= cfg.eps_end + 1e-9 and episode % snapshot_every == 0 and episode > 0:
                snap_path = snapshot_dir / f"ckpt_ep{episode}.pt"
                agent.save_checkpoint(snap_path)
                snapshot_paths.append(str(snap_path))

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    episode,
                    round(final_profit, 2),
                    round(tardiness_cost, 2),
                    period_decisions_per_ep,
                    round(agent.epsilon, 4),
                    len(agent.replay),
                    round(elapsed, 1),
                    round(loss_avg, 6),
                    json.dumps(dict(strategy.action_counts)),
                ])

            if episode % 50 == 0 or elapsed > time_sec - 5:
                eps_per_sec = (episode + 1) / max(elapsed, 1e-3)
                print(f"  ep {episode:>5d}  profit=${final_profit:>10,.0f}  "
                      f"best=${best_profit:>10,.0f} (ep {best_episode})  "
                      f"eps={agent.epsilon:.3f}  buf={len(agent.replay):>6d}  "
                      f"loss={loss_avg:.4f}  wall={elapsed:.0f}s  ({eps_per_sec:.1f} ep/s)")

            episode += 1

    except KeyboardInterrupt:
        print("[paeng-train-v2] interrupted by user; saving final checkpoint anyway")

    agent.save_checkpoint(final_ckpt)
    elapsed = time.perf_counter() - t0
    summary = {
        "episodes": episode,
        "wall_sec": round(elapsed, 1),
        "best_profit": round(best_profit, 2),
        "best_episode": best_episode,
        "final_epsilon": round(agent.epsilon, 4),
        "final_buffer_size": len(agent.replay),
        "best_ckpt": str(best_ckpt),
        "final_ckpt": str(final_ckpt),
        "config": {
            "action_dim": cfg.action_dim,
            "state_rows": cfg.state_rows,
            "state_cols": cfg.state_cols,
            "sf_dim": cfg.sf_dim,
            "lr": cfg.lr,
            "gamma": cfg.gamma,
            "batch_size": cfg.batch_size,
            "buffer_size": cfg.buffer_size,
            "eps_start": cfg.eps_start,
            "eps_end": cfg.eps_end,
            "eps_ratio": cfg.eps_ratio,
            "is_double": cfg.is_double,
            "is_duel": cfg.is_duel,
            "tau": cfg.tau,
            "huber_delta": cfg.huber_delta,
        },
        "use_ups": use_ups,
        "ups_lambda": ups_lambda,
        "ups_mu": ups_mu,
        "seed_base": seed_base,
        "best_rolling_mean": round(best_rolling_mean, 2) if best_rolling_mean > -1e9 else None,
        "best_rolling_episode": best_rolling_episode if best_rolling_episode >= 0 else None,
        "best_rolling_ckpt": str(best_rolling_ckpt) if rolling_window > 0 else None,
        "n_restores": n_restores,
        "snapshot_every": snapshot_every,
        "snapshot_paths": snapshot_paths,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[paeng-train-v2] done.  episodes={episode}  best=${best_profit:,.0f} "
          f"(ep {best_episode})  wall={elapsed:.0f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Paeng DDQN v2 (faithful to 2021 paper).")
    parser.add_argument("--time-sec", type=float, required=True)
    parser.add_argument("--output-dir", type=str, default="paeng_ddqn_v2/outputs")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--no-ups", action="store_true")
    parser.add_argument("--target-episodes", type=int, default=100_000)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--rolling-window", type=int, default=0)
    parser.add_argument("--restore-drop-threshold", type=float, default=50_000.0)
    parser.add_argument("--load-ckpt", type=str, default=None)
    parser.add_argument("--initial-epsilon", type=float, default=None)
    parser.add_argument("--run-seed", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    train(
        time_sec=args.time_sec,
        output_dir=out_dir,
        seed_base=args.seed_base,
        use_ups=not args.no_ups,
        target_episode_estimate=args.target_episodes,
        snapshot_every=args.snapshot_every,
        rolling_window=args.rolling_window,
        restore_drop_threshold=args.restore_drop_threshold,
        load_ckpt=args.load_ckpt,
        initial_epsilon=args.initial_epsilon,
        run_seed=args.run_seed,
    )


if __name__ == "__main__":
    main()
