"""Monitored RL-HH training with hourly health checks.

Runs training for up to MAX_WALL_HOURS, checking every CHECK_INTERVAL_SEC.
At each check:
  - Evaluates 20 greedy episodes
  - If profit regressed or all-WAIT, terminates and restarts a new cycle
  - Exports best result as plot_result.py-compatible JSON + HTML
  - Maximum MAX_CYCLES restarts

Usage:
    python -m rl_hh.train_monitored
"""

from __future__ import annotations

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
from .export_result import run_and_export
from .fast_loop import run_fast_episode
from .meta_agent import DuelingDDQNAgent
from .numpy_net import NumpyDuelingDDQN
from .tools import ToolKit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_WALL_HOURS = 9.0
CHECK_INTERVAL_SEC = 3600        # 1 hour
EVAL_EPISODES = 20               # quick eval at each checkpoint
MAX_CYCLES = 20
MIN_PROFIT_THRESHOLD = -500_000  # below this = clearly broken (UPS adds variance)
TOOL_DIVERSITY_MIN = 0.02        # each tool must be > 2% of selections
PROGRESS_WINDOW = 3              # check last N evals for stagnation

OUTPUT_DIR = Path("rl_hh/outputs")
TOOL_NAMES = C.TOOL_NAMES


def _evaluate_quick(
    engine, agent, toolkit, data, np_net, n_eps=EVAL_EPISODES,
    ups_lambda: float = 0.0, ups_mu: float = 0.0, base_seed: int = 800_000,
):
    """Run quick greedy evaluation.  Returns (mean_profit, tool_dist, profits)."""
    profits = []
    tool_total = [0] * C.N_TOOLS
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    SL = int(engine.params["SL"])
    roasters = list(engine.params["roasters"])

    for ep in range(n_eps):
        ups = generate_ups_events(ups_lambda, ups_mu, base_seed + ep, SL, roasters)
        kpi, n_dec, tc = run_fast_episode(
            engine, agent, toolkit, data,
            agent.replay_buffer, ups, training=False, np_net=np_net,
        )
        profits.append(float(kpi.net_profit()))
        for i in range(C.N_TOOLS):
            tool_total[i] += tc[i]

    agent.epsilon = old_eps
    total = max(1, sum(tool_total))
    tool_dist = [c / total for c in tool_total]
    mean_profit = sum(profits) / len(profits)
    return mean_profit, tool_dist, profits


def _check_health(
    mean_profit: float,
    tool_dist: list[float],
    eval_history: list[float],
) -> tuple[bool, str]:
    """Return (healthy, reason).  healthy=False means restart."""
    # Check 1: profit floor
    if mean_profit < MIN_PROFIT_THRESHOLD:
        return False, f"Profit {mean_profit:.0f} below threshold {MIN_PROFIT_THRESHOLD}"

    # Check 2: tool diversity (at least some non-WAIT usage)
    non_wait_total = sum(tool_dist[:-1])
    if non_wait_total < 0.10:
        return False, f"Tool diversity too low: WAIT={tool_dist[-1]*100:.1f}%, non-WAIT={non_wait_total*100:.1f}%"

    # Check 3: stagnation — no improvement in last PROGRESS_WINDOW evals
    if len(eval_history) >= PROGRESS_WINDOW:
        recent = eval_history[-PROGRESS_WINDOW:]
        if all(r <= eval_history[-PROGRESS_WINDOW] + 500 for r in recent):
            best_ever = max(eval_history)
            if mean_profit < best_ever - 20_000:
                return False, f"Stagnation: last {PROGRESS_WINDOW} evals flat, regressed from best {best_ever:.0f}"

    return True, "OK"


def _export_html(checkpoint_path: Path, output_dir: Path, cycle: int, tag: str):
    """Export result JSON and generate HTML report."""
    result_path = output_dir / f"rlhh_cycle{cycle}_{tag}_result.json"
    try:
        result = run_and_export(
            checkpoint_path=str(checkpoint_path),
            output_path=str(result_path),
            seed=42,
        )
        # Generate HTML via plot_result.py
        from plot_result import _build_html
        html = _build_html(result, None, offline=True)
        html_path = output_dir / f"rlhh_cycle{cycle}_{tag}_report.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"  HTML report: {html_path}")
        return str(html_path)
    except Exception as exc:
        print(f"  Export failed: {exc}")
        return None


def train_monitored():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    params = data.to_env_params()
    SL = int(params["SL"])
    roasters = list(params["roasters"])
    ups_lambda = data.ups_lambda
    ups_mu = data.ups_mu

    master_log_path = OUTPUT_DIR / "monitored_log.csv"
    with open(master_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cycle", "episode", "wall_sec", "eval_profit", "epsilon",
            "tool_dist", "healthy", "reason",
        ])

    overall_best_profit = -float("inf")
    overall_best_ckpt = None
    global_t0 = time.perf_counter()

    for cycle in range(1, MAX_CYCLES + 1):
        elapsed_total = time.perf_counter() - global_t0
        if elapsed_total >= MAX_WALL_HOURS * 3600:
            print(f"\n=== Wall time limit reached ({MAX_WALL_HOURS}h). Stopping. ===")
            break

        remaining_sec = MAX_WALL_HOURS * 3600 - elapsed_total
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle}/{MAX_CYCLES}  |  Remaining: {remaining_sec/3600:.1f}h")
        print(f"{'='*60}")

        # Fresh agent per cycle
        engine = SimulationEngine(params)
        toolkit = ToolKit(engine, params)
        agent = DuelingDDQNAgent()
        np_net = NumpyDuelingDDQN(agent.online_net)

        cycle_t0 = time.perf_counter()
        episode = 0
        eval_history: list[float] = []
        cycle_best_profit = -float("inf")
        cycle_best_ep = 0
        last_check_time = cycle_t0

        # Training loop
        while True:
            elapsed_total = time.perf_counter() - global_t0
            if elapsed_total >= MAX_WALL_HOURS * 3600:
                break

            # Run one episode
            ups = generate_ups_events(ups_lambda, ups_mu, episode + cycle * 1_000_000, SL, roasters)
            kpi, n_dec, tc = run_fast_episode(
                engine, agent, toolkit, data,
                agent.replay_buffer, ups, training=True, np_net=np_net,
            )
            if len(agent.replay_buffer) >= C.BATCH_SIZE:
                for _ in range(C.TRAINS_PER_EP):
                    agent.train_step()
                np_net.sync(agent.online_net)
            agent.decay_epsilon(episode, C.NUM_EPISODES)
            episode += 1

            # Track training profit
            ep_profit = float(kpi.net_profit())
            if ep_profit > cycle_best_profit:
                cycle_best_profit = ep_profit
                cycle_best_ep = episode
                agent.save_checkpoint(OUTPUT_DIR / f"rlhh_cycle{cycle}_best.pt")

            # Periodic logging
            if episode % C.LOG_INTERVAL == 0:
                elapsed_cycle = time.perf_counter() - cycle_t0
                eps_per_sec = episode / elapsed_cycle if elapsed_cycle > 0 else 0
                print(
                    f"  [C{cycle} ep={episode:>7d}] "
                    f"profit={ep_profit:>10.1f}  "
                    f"best={cycle_best_profit:>10.1f}  "
                    f"eps={agent.epsilon:.3f}  "
                    f"buf={len(agent.replay_buffer):>6d}  "
                    f"tools={tc}  "
                    f"({eps_per_sec:.0f} ep/s)"
                )

            # Hourly health check
            now = time.perf_counter()
            if now - last_check_time >= CHECK_INTERVAL_SEC:
                last_check_time = now
                elapsed_cycle = now - cycle_t0
                print(f"\n  --- Health check at {elapsed_cycle/3600:.1f}h into cycle {cycle}, ep={episode} ---")

                mean_profit, tool_dist, profits = _evaluate_quick(
                    engine, agent, toolkit, data, np_net,
                    ups_lambda=ups_lambda, ups_mu=ups_mu,
                )
                eval_history.append(mean_profit)

                dist_str = ", ".join(
                    f"{TOOL_NAMES[i]}={tool_dist[i]*100:.1f}%"
                    for i in range(C.N_TOOLS)
                )
                print(f"  Eval profit: {mean_profit:>10.1f} (std={max(profits)-min(profits):.0f})")
                print(f"  Tool dist:   {dist_str}")

                healthy, reason = _check_health(mean_profit, tool_dist, eval_history)
                print(f"  Health:      {'OK' if healthy else 'FAIL'} — {reason}")

                with open(master_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        cycle, episode, round(now - global_t0, 1),
                        round(mean_profit, 2), round(agent.epsilon, 4),
                        json.dumps({TOOL_NAMES[i]: round(tool_dist[i], 4) for i in range(C.N_TOOLS)}),
                        healthy, reason,
                    ])

                # Export checkpoint report
                ckpt_path = OUTPUT_DIR / f"rlhh_cycle{cycle}_best.pt"
                if ckpt_path.exists():
                    _export_html(ckpt_path, OUTPUT_DIR, cycle, f"check{len(eval_history)}")

                if mean_profit > overall_best_profit:
                    overall_best_profit = mean_profit
                    overall_best_ckpt = str(ckpt_path)
                    agent.save_checkpoint(OUTPUT_DIR / "rlhh_overall_best.pt")
                    print(f"  >> New overall best: {overall_best_profit:.1f}")

                if not healthy:
                    print(f"  >> RESTARTING (reason: {reason})")
                    # Save final state before restart
                    agent.save_checkpoint(OUTPUT_DIR / f"rlhh_cycle{cycle}_final.pt")
                    break

        # End of cycle: save and export
        agent.save_checkpoint(OUTPUT_DIR / f"rlhh_cycle{cycle}_final.pt")
        print(f"\n  Cycle {cycle} done: {episode} episodes, best training profit={cycle_best_profit:.1f} (at ep {cycle_best_ep})")

        # Final eval + export for this cycle
        mean_profit, tool_dist, _ = _evaluate_quick(
            engine, agent, toolkit, data, np_net,
            ups_lambda=ups_lambda, ups_mu=ups_mu,
        )
        print(f"  Final eval profit: {mean_profit:.1f}")
        if mean_profit > overall_best_profit:
            overall_best_profit = mean_profit
            overall_best_ckpt = str(OUTPUT_DIR / f"rlhh_cycle{cycle}_best.pt")
            agent.save_checkpoint(OUTPUT_DIR / "rlhh_overall_best.pt")

        best_ckpt = OUTPUT_DIR / f"rlhh_cycle{cycle}_best.pt"
        if best_ckpt.exists():
            _export_html(best_ckpt, OUTPUT_DIR, cycle, "final")

    # ---------------------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------------------
    total_elapsed = time.perf_counter() - global_t0
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total wall time: {total_elapsed/3600:.1f}h")
    print(f"  Overall best eval profit: {overall_best_profit:.1f}")
    if overall_best_ckpt:
        print(f"  Best checkpoint: {overall_best_ckpt}")
        # Final export with the absolute best
        _export_html(Path(OUTPUT_DIR / "rlhh_overall_best.pt"), OUTPUT_DIR, 0, "overall_best")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_monitored()
