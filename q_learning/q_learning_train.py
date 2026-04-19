"""Tabular Q-Learning training for reactive roasting scheduling.

Usage (from project root):
    python -m q_learning.q_learning_train --name first_try
    python -m q_learning.q_learning_train --time 3600 --alpha 0.05 --gamma 0.99 --name second_try
    python -m q_learning.q_learning_train --episodes 5000 --name quick_test

Outputs go to:  q_learning/ql_results/<timestamp>_ep<N>_a<alpha>_g<gamma>_<name>/
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.data_bridge import get_sim_params
from env.simulation_engine import SimulationEngine
from env.ups_generator import generate_ups_events
from dispatch.dispatching_heuristic import DispatchingHeuristic

from q_learning.q_strategy import (
    ACTION_MAP,
    WAIT_ACTION,
    discretize_roaster_state,
    get_valid_roaster_actions,
    save_q_table,
)

# ---------------------------------------------------------------------------
# Hyper-parameters (defaults)
# ---------------------------------------------------------------------------

ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
NUM_EPISODES = 20_000

# Robust-mode scenario grid (only used with --robust flag)
ROBUST_LAMBDA = [0, 1, 2, 5, 10, 20]
ROBUST_MU = [5, 10, 20, 30]

RESULTS_ROOT = Path(__file__).resolve().parent / "ql_results"


# ---------------------------------------------------------------------------
# Result folder naming
# ---------------------------------------------------------------------------

def _make_result_dir(name: str, episodes: int, alpha: float, gamma: float) -> Path:
    ts = datetime.now().strftime("%d_%m_%Y_%H%M")
    a_str = f"{alpha:.4f}".replace(".", "")
    g_str = f"{gamma:.4f}".replace(".", "")
    parts = [ts, f"ep{episodes}", f"a{a_str}", f"g{g_str}"]
    if name:
        parts.append(name)
    path = RESULTS_ROOT / "_".join(parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rename_with_profit(result_dir: Path, final_profit: float) -> Path:
    new_path = result_dir.parent / (result_dir.name + f"_profit{int(round(final_profit))}")
    try:
        result_dir.rename(new_path)
        return new_path
    except OSError:
        return result_dir


# ---------------------------------------------------------------------------
# Manual episode runner - gives direct KPI access for slot-aligned rewards
# ---------------------------------------------------------------------------

def _train_one_episode(
    params: dict,
    engine: SimulationEngine,
    q_table: dict,
    epsilon: float,
    ups_events: list,
    allow_flex: bool,
) -> tuple[list[tuple], float]:
    """Run one shift with epsilon-greedy decisions and slot-aligned rewards.

    Returns (transitions, episode_profit) where each transition is:
    (state, action_id, reward_delta, next_state, next_valid_actions)

    Only per-roaster roasting decisions are Q-learned.
    Restock decisions are delegated to the dispatching heuristic so
    training matches evaluation and the Q-table stays roaster-only.
    """
    state = engine._initialize_state()
    kpi = engine._make_kpi_tracker()

    ups_by_time: dict[int, list] = defaultdict(list)
    for event in sorted(ups_events, key=lambda e: (e.t, e.roaster_id, e.duration)):
        ups_by_time[event.t].append(event)

    transitions: list[tuple] = []
    prev_state_disc: tuple | None = None
    prev_action: int | None = None
    pending_reward: float = 0.0

    # Track SETUP->IDLE so the state can distinguish "freshly configured"
    # from "already producing this SKU".
    prev_status = {r: "IDLE" for r in engine.roasters}
    just_set_up = {r: False for r in engine.roasters}
    restock_helper = DispatchingHeuristic(params)

    def _close_pending(next_state: tuple, next_valid_actions: list[int]) -> None:
        nonlocal pending_reward
        if prev_state_disc is None:
            return
        transitions.append(
            (prev_state_disc, prev_action, pending_reward, next_state, tuple(next_valid_actions))
        )
        pending_reward = 0.0

    def _choose_action(state_key: tuple, valid_actions: list[int]) -> int:
        if not valid_actions:
            return WAIT_ACTION
        if random.random() < epsilon:
            return random.choice(valid_actions)
        q_vals = [(action_id, q_table.get((state_key, action_id), 0.0)) for action_id in valid_actions]
        return max(q_vals, key=lambda item: item[1])[0]

    for slot in range(params["SL"]):
        state.t = slot
        slot_profit_before = kpi.net_profit()

        for event in ups_by_time.get(slot, []):
            engine._process_ups(state, event, None, kpi)

        engine._step_roaster_timers(state, kpi)
        engine._step_pipeline_and_restock_timers(state, kpi)
        engine._process_consumption_events(state, kpi)
        engine._track_stockout_duration(state, kpi)
        engine._accrue_idle_penalties(state, kpi)

        for roaster_id in engine.roasters:
            cur = state.status[roaster_id]
            if prev_status[roaster_id] == "SETUP" and cur == "IDLE":
                just_set_up[roaster_id] = True
            elif cur == "RUNNING":
                just_set_up[roaster_id] = False
            prev_status[roaster_id] = cur

        # Restock handled by heuristic, not by Q-learning.
        # This matches the engine.run() restock phase and keeps the
        # Q-table focused on roaster decisions only.
        engine._process_restock_decision_point(state, restock_helper, kpi)

        for roaster_id in engine.roasters:
            if state.status[roaster_id] != "IDLE" or not state.needs_decision[roaster_id]:
                continue

            s_new = discretize_roaster_state(
                state,
                roaster_id,
                params,
                just_set_up=just_set_up[roaster_id],
            )
            valid_actions = get_valid_roaster_actions(engine, state, roaster_id, allow_flex)
            _close_pending(s_new, valid_actions)

            action_idx = _choose_action(s_new, valid_actions)
            action_tuple = ACTION_MAP[action_idx]
            engine._apply_action(state, roaster_id, action_tuple, kpi)

            if state.status[roaster_id] == "RUNNING":
                just_set_up[roaster_id] = False
            prev_status[roaster_id] = state.status[roaster_id]

            prev_state_disc = s_new
            prev_action = action_idx

        pending_reward += kpi.net_profit() - slot_profit_before

    if prev_state_disc is not None:
        transitions.append((prev_state_disc, prev_action, pending_reward, None, ()))

    episode_profit = kpi.net_profit()
    reward_sum = sum(reward for _, _, reward, _, _ in transitions)
    if abs(reward_sum - episode_profit) > 1e-9:
        raise RuntimeError(
            f"Training reward mismatch: transitions sum to {reward_sum}, "
            f"but episode profit is {episode_profit}"
        )

    return transitions, episode_profit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train tabular Q-learning agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m q_learning.q_learning_train --name first_try\n"
            "  python -m q_learning.q_learning_train --time 3600 --name long_run\n"
            "  python -m q_learning.q_learning_train --episodes 5000 --alpha 0.05 --name quick\n"
        ),
    )
    parser.add_argument("--name", "-n", type=str, default="", help="Run name (used in folder and Q-table filename)")
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=None,
        help=f"Max episodes (default: {NUM_EPISODES} if no --time)",
    )
    parser.add_argument(
        "--time",
        "-t",
        type=float,
        default=None,
        help="Time limit in seconds (preferred over --episodes)",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA, help=f"Learning rate (default: {ALPHA})")
    parser.add_argument("--gamma", type=float, default=GAMMA, help=f"Discount factor (default: {GAMMA})")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N episodes (0=disabled)",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Train on a grid of UPS scenarios instead of the env default (lambda, mu)",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip automatic evaluation after training")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Episodes per scenario for auto-eval (default: 50)",
    )
    args = parser.parse_args(argv)

    use_time = args.time is not None
    if use_time:
        max_episodes = 10_000_000
        time_budget = args.time
    else:
        max_episodes = args.episodes if args.episodes is not None else NUM_EPISODES
        time_budget = float("inf")

    params = get_sim_params()
    q_table: dict = defaultdict(float)
    alpha = args.alpha
    gamma = args.gamma
    allow_flex = params.get("allow_r3_flex", True)
    run_name = args.name

    robust_mode = args.robust
    env_lambda = params.get("ups_lambda", 2.0)
    env_mu = params.get("ups_mu", 20.0)
    if robust_mode:
        lambda_options = ROBUST_LAMBDA
        mu_options = ROBUST_MU
    else:
        lambda_options = [env_lambda]
        mu_options = [env_mu]

    rewards_log: list[float] = []
    t_start = time.perf_counter()

    stop_reason = "episodes"
    if use_time:
        print(f"Training Q-learning: time budget {time_budget:.0f}s  (name={run_name or '(none)'})")
    else:
        print(f"Training Q-learning: {max_episodes} episodes  (name={run_name or '(none)'})")
    if robust_mode:
        print(f"  UPS mode: ROBUST - lambda={lambda_options}, mu={mu_options}")
    else:
        print(f"  UPS mode: env default - lambda={env_lambda}, mu={env_mu}")
    print(f"  alpha={alpha}  gamma={gamma}  linear epsilon decay {EPSILON_START} -> {EPSILON_END}")

    ep = 0
    while ep < max_episodes:
        elapsed_now = time.perf_counter() - t_start
        if use_time and elapsed_now >= time_budget:
            stop_reason = f"time limit ({time_budget:.0f}s)"
            break

        if use_time:
            progress = min(1.0, elapsed_now / (0.7 * time_budget))
        else:
            progress = min(1.0, ep / (0.7 * max_episodes))
        epsilon = max(EPSILON_END, EPSILON_START - progress * (EPSILON_START - EPSILON_END))

        lam = random.choice(lambda_options)
        mu = random.choice(mu_options)
        seed = random.randint(0, 2**31 - 1)
        ups_events = generate_ups_events(lam, mu, seed)

        engine = SimulationEngine(params)
        transitions, episode_profit = _train_one_episode(
            params, engine, q_table, epsilon, ups_events, allow_flex,
        )

        for s, a, r, s_next, next_valid_actions in transitions:
            old_q = q_table[(s, a)]
            if s_next is not None and next_valid_actions:
                max_next_q = max(
                    (q_table.get((s_next, a2), 0.0) for a2 in next_valid_actions),
                    default=0.0,
                )
            else:
                max_next_q = 0.0
            q_table[(s, a)] = old_q + alpha * (r + gamma * max_next_q - old_q)

        rewards_log.append(episode_profit)
        ep += 1

        if ep % 500 == 0:
            recent = rewards_log[-500:]
            avg = sum(recent) / len(recent)
            elapsed = time.perf_counter() - t_start
            eps_per_sec = ep / elapsed
            remaining = ""
            if use_time:
                left = max(0, time_budget - elapsed)
                remaining = f"  {left:.0f}s left"
            print(
                f"  ep {ep:>6} | "
                f"eps={epsilon:.4f} | "
                f"avg500=${avg:>10,.0f} | "
                f"Q-size={len(q_table):>8,} | "
                f"{eps_per_sec:.1f} ep/s{remaining}"
            )

        if args.checkpoint_interval > 0 and ep % args.checkpoint_interval == 0:
            ckpt_dir = RESULTS_ROOT / "_checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tag = f"_{run_name}" if run_name else ""
            save_q_table(dict(q_table), str(ckpt_dir / f"q_table_ckpt_{ep}{tag}.pkl"))

    total_episodes = ep
    elapsed = time.perf_counter() - t_start
    final_avg = sum(rewards_log[-1000:]) / max(1, min(1000, len(rewards_log)))

    if use_time:
        final_progress = min(1.0, elapsed / (0.7 * time_budget))
    else:
        final_progress = min(1.0, total_episodes / (0.7 * max_episodes))
    final_epsilon = max(EPSILON_END, EPSILON_START - final_progress * (EPSILON_START - EPSILON_END))

    result_dir = _make_result_dir(run_name, total_episodes, alpha, gamma)

    qtable_filename = f"q_table_{run_name}.pkl" if run_name else "q_table.pkl"
    save_q_table(dict(q_table), str(result_dir / qtable_filename))

    log_filename = f"training_log_{run_name}.pkl" if run_name else "training_log.pkl"
    with open(result_dir / log_filename, "wb") as f:
        pickle.dump(rewards_log, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "name": run_name,
        "episodes": total_episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_schedule": "linear (reaches eps_end at 70% of budget)",
        "final_epsilon": round(final_epsilon, 6),
        "state_formulation": "roaster_tabular_with_heuristic_restock",
        "ups_mode": "robust" if robust_mode else "env_default",
        "ups_lambda": lambda_options if robust_mode else env_lambda,
        "ups_mu": mu_options if robust_mode else env_mu,
        "elapsed_seconds": round(elapsed, 1),
        "stop_reason": stop_reason,
        "q_table_entries": len(q_table),
        "final_avg_profit_1000": round(final_avg, 2),
        "q_table_file": qtable_filename,
        "training_log_file": log_filename,
        "timestamp": datetime.now().isoformat(),
    }
    with open(result_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    latest_q = Path(__file__).resolve().parent / "q_table.pkl"
    save_q_table(dict(q_table), str(latest_q))
    latest_log = Path(__file__).resolve().parent / "training_log.pkl"
    with open(latest_log, "wb") as f:
        pickle.dump(rewards_log, f, protocol=pickle.HIGHEST_PROTOCOL)

    result_dir = _rename_with_profit(result_dir, final_avg)

    print(f"\nTraining complete ({stop_reason})")
    print(f"  Episodes: {total_episodes:,}  |  Wall time: {elapsed:.1f}s  |  {total_episodes / elapsed:.1f} ep/s")
    print(f"  Q-table: {len(q_table):,} entries")
    print(f"  Final epsilon: {final_epsilon:.4f}")
    print(f"  Final avg profit (last 1000): ${final_avg:,.0f}")
    print(f"  Result folder: {result_dir}")
    print(f"  Latest copy:   {latest_q}")

    if not args.no_eval:
        q_table_path = str(result_dir / qtable_filename)
        print(f"\n{'=' * 72}")
        print("AUTO-EVALUATION: generating report...")
        print(f"{'=' * 72}\n")
        from q_learning.q_learning_run import main as run_main

        run_main(["--file", q_table_path, "--episodes", str(args.eval_episodes)])


if __name__ == "__main__":
    main()
