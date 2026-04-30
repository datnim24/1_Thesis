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
) -> dict:
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
    strategy = PaengStrategy(agent, data, training=True)

    engine = SimulationEngine(params)

    log_path = output_dir / "training_log.csv"
    summary_path = output_dir / "training_summary.json"
    best_ckpt = output_dir / "paeng_best.pt"
    final_ckpt = output_dir / "paeng_final.pt"

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
    )


if __name__ == "__main__":
    main()
