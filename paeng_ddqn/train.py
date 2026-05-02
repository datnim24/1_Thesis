"""Paeng's Modified DDQN — training CLI.

Mirrors `Paeng_DRL_Github/main.py:130-180` adapted to our engine + UPS.

Usage:
    # 4-hour main training run
    python -m paeng_ddqn.train --time-sec 14400 --output-dir paeng_ddqn/outputs

    # 10-minute smoke (Task 8.5)
    python -m paeng_ddqn.train --time-sec 600 --output-dir paeng_ddqn/outputs/smoke

    # No-UPS smoke (Task 7)
    python -m paeng_ddqn.train --time-sec 60 --output-dir paeng_ddqn/outputs/smoke_noups --no-ups

Per decision D1: a fresh seed is drawn from `seed_base + episode_idx` each
episode so the agent generalizes across UPS realizations (Paeng's robustness
test pattern, Table 5).
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

from paeng_ddqn.agent import PaengAgent, PaengConfig
from paeng_ddqn.strategy import PaengStrategy


def train(
    time_sec: float,
    output_dir: Path,
    seed_base: int = 42,
    use_ups: bool = True,
    target_episode_estimate: int | None = None,
    snapshot_every: int = 0,
    rolling_window: int = 0,
    restore_drop_threshold: float = 250_000.0,
    load_ckpt: str | None = None,
    initial_epsilon: float | None = None,
    warm_restart_period: int = 0,
    warm_restart_peak: float = 0.15,
    warm_restart_decay_episodes: int = 100,
    run_seed: int | None = None,
) -> dict:
    # Cycle 46 (2026-04-29): explicit run-RNG seed. Cycle 45 confirmed cycle 13's
    # +$45k was an unseeded-RNG accident. Setting these seeds at run start makes
    # each (run_seed, seed_base) pair fully reproducible, turning seed search into
    # a controlled sweep instead of a coin flip.
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
    """Time-budgeted training loop with per-episode seed rotation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    ups_lambda = float(data.ups_lambda)
    ups_mu = float(data.ups_mu)

    cfg = PaengConfig()
    if target_episode_estimate is not None:
        cfg.target_episode_estimate = target_episode_estimate
    agent = PaengAgent(cfg)
    if load_ckpt is not None:
        agent.load_checkpoint(load_ckpt)
        print(f"[paeng-train] warm-start: loaded {load_ckpt}; epsilon={agent.epsilon:.3f}")
    if initial_epsilon is not None:
        agent.epsilon = float(initial_epsilon)
        print(f"[paeng-train] initial epsilon override: {agent.epsilon:.3f}")
    strategy = PaengStrategy(agent, data, training=True)

    engine = SimulationEngine(params)

    log_path = output_dir / "training_log.csv"
    summary_path = output_dir / "training_summary.json"
    best_ckpt = output_dir / "paeng_best.pt"
    final_ckpt = output_dir / "paeng_final.pt"
    best_rolling_ckpt = output_dir / "paeng_best_rolling.pt"
    snapshot_dir = output_dir / "snapshots"
    if snapshot_every > 0:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths: list[str] = []

    rolling: deque[float] = deque(maxlen=rolling_window) if rolling_window > 0 else deque()
    best_rolling_mean = -float("inf")
    best_rolling_episode = -1
    n_restores = 0

    # Cycle 27 (2026-04-29): ε warm-restart scaffolding.
    last_warm_restart_at = -10**9
    warm_restart_remaining = 0     # ep countdown of manual decay window
    warm_restart_decay_per_ep = 0.0
    n_warm_restarts = 0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "episode", "profit", "epsilon", "buffer_size",
            "wall_sec", "loss_avg", "action_dist",
        ])

    t0 = time.perf_counter()
    episode = 0
    best_profit = -float("inf")
    best_episode = 0
    losses_window: list[float] = []

    print(f"[paeng-train] starting: budget={time_sec:.0f}s, ups={use_ups}, "
          f"lambda={ups_lambda}, mu={ups_mu}, seed_base={seed_base}")

    try:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= time_sec:
                break

            seed = seed_base + episode
            ups = generate_ups_events(ups_lambda, ups_mu, seed, SL, roasters) if use_ups else []

            strategy.reset_episode()
            strategy.action_counts = {i: 0 for i in range(8)}

            kpi, sim_state = engine.run(strategy, ups)
            final_profit = float(kpi.net_profit())
            strategy.end_episode(sim_state, final_profit)

            if initial_epsilon is None:
                # Cycle 27: warm-restart cycle. Trigger when ε is at floor and period elapsed.
                if (
                    warm_restart_period > 0
                    and agent.epsilon <= cfg.eps_end + 1e-9
                    and warm_restart_remaining == 0
                    and (episode - last_warm_restart_at) >= warm_restart_period
                ):
                    agent.epsilon = warm_restart_peak
                    warm_restart_remaining = warm_restart_decay_episodes
                    warm_restart_decay_per_ep = (
                        (warm_restart_peak - cfg.eps_end) / max(1, warm_restart_decay_episodes)
                    )
                    last_warm_restart_at = episode
                    n_warm_restarts += 1
                    print(f"  [warm-restart #{n_warm_restarts}] ep{episode} eps -> {warm_restart_peak:.3f}")
                if warm_restart_remaining > 0:
                    agent.epsilon = max(cfg.eps_end, agent.epsilon - warm_restart_decay_per_ep)
                    warm_restart_remaining -= 1
                else:
                    agent.decay_epsilon(episode, cfg.target_episode_estimate)
            # else: epsilon pinned to initial_epsilon (cycle 21 warm-start mode)
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

            # Cycle 17 (2026-04-28): rolling-mean tracking + collapse-restore.
            if rolling_window > 0:
                rolling.append(final_profit)
                if len(rolling) >= rolling_window:
                    rolling_mean = sum(rolling) / len(rolling)
                    if rolling_mean > best_rolling_mean:
                        best_rolling_mean = rolling_mean
                        best_rolling_episode = episode
                        agent.save_checkpoint(best_rolling_ckpt)
                    # Restore only after ε floor (let exploration play out first).
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
                              f"best=${best_rolling_mean:>10,.0f} (ep{best_rolling_episode})  "
                              f"drop=${best_rolling_mean - rolling_mean:>10,.0f}")

            # Snapshot: dump a checkpoint every snapshot_every episodes once ε is at floor.
            if (
                snapshot_every > 0
                and agent.epsilon <= cfg.eps_end + 1e-9
                and episode % snapshot_every == 0
                and episode > 0
            ):
                snap_path = snapshot_dir / f"ckpt_ep{episode}.pt"
                agent.save_checkpoint(snap_path)
                snapshot_paths.append(str(snap_path))

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    episode, round(final_profit, 2), round(agent.epsilon, 4),
                    len(agent.replay), round(elapsed, 1),
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
        print("[paeng-train] interrupted by user; saving final checkpoint anyway")

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
        "config": cfg.__dict__,
        "use_ups": use_ups,
        "ups_lambda": ups_lambda,
        "ups_mu": ups_mu,
        "seed_base": seed_base,
    }
    summary["config"]["hid_dims"] = list(summary["config"].get("hid_dims", []))
    summary["snapshot_paths"] = snapshot_paths
    summary["snapshot_every"] = snapshot_every
    summary["rolling_window"] = rolling_window
    summary["restore_drop_threshold"] = restore_drop_threshold
    summary["best_rolling_mean"] = round(best_rolling_mean, 2) if best_rolling_mean > -1e9 else None
    summary["best_rolling_episode"] = best_rolling_episode if best_rolling_episode >= 0 else None
    summary["n_restores"] = n_restores
    summary["best_rolling_ckpt"] = str(best_rolling_ckpt) if rolling_window > 0 else None
    summary["warm_restart_period"] = warm_restart_period
    summary["warm_restart_peak"] = warm_restart_peak
    summary["warm_restart_decay_episodes"] = warm_restart_decay_episodes
    summary["n_warm_restarts"] = n_warm_restarts
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[paeng-train] done.  episodes={episode}  best=${best_profit:,.0f} "
          f"(ep {best_episode})  wall={elapsed:.0f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Paeng's Modified DDQN with seed rotation.")
    parser.add_argument("--time-sec", type=float, required=True)
    parser.add_argument("--output-dir", type=str, default="paeng_ddqn/outputs")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--no-ups", action="store_true")
    parser.add_argument("--target-episodes", type=int, default=None)
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="Save a checkpoint every N episodes once ε is at its floor (0 = disabled).")
    parser.add_argument("--rolling-window", type=int, default=0,
                        help="Cycle 17: rolling-mean window size for collapse-restore (0 = disabled).")
    parser.add_argument("--restore-drop-threshold", type=float, default=250_000.0,
                        help="Cycle 17: restore weights if rolling mean drops > this $ below best.")
    parser.add_argument("--load-ckpt", type=str, default=None,
                        help="Cycle 21: warm-start from a checkpoint path (overrides random init).")
    parser.add_argument("--initial-epsilon", type=float, default=None,
                        help="Cycle 21: override the agent's starting epsilon (e.g., 0.05 for refine mode).")
    parser.add_argument("--warm-restart-period", type=int, default=0,
                        help="Cycle 27: episodes between ε warm-restarts (0 = disabled). Triggers after ε floor.")
    parser.add_argument("--warm-restart-peak", type=float, default=0.15,
                        help="Cycle 27: ε value to reset to on warm-restart.")
    parser.add_argument("--warm-restart-decay-episodes", type=int, default=100,
                        help="Cycle 27: episodes to manually decay ε from peak back to eps_end after a restart.")
    parser.add_argument("--run-seed", type=int, default=None,
                        help="Cycle 46: explicit RNG seed (random, numpy, torch) for reproducibility.")
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
        warm_restart_period=args.warm_restart_period,
        warm_restart_peak=args.warm_restart_peak,
        warm_restart_decay_episodes=args.warm_restart_decay_episodes,
        run_seed=args.run_seed,
    )


if __name__ == "__main__":
    main()
