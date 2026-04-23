"""Training callbacks: progress logging, best-model saving, violation tracking, and diagnostic outputs.

Outputs produced at end of training:
  - {run_dir}/training_log.pkl   : full episode history for plot_result.py
  - {run_dir}/training_stats.json: summary stats + grade assessment
"""
from __future__ import annotations

import json
import pickle
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback


def _compute_grade(avg_profit: float, viol_rate_last100: float) -> str:
    if avg_profit >= 280_000 and viol_rate_last100 < 0.05:
        return "A"
    if avg_profit >= 240_000 and viol_rate_last100 < 0.10:
        return "B"
    if avg_profit >= 200_000 and viol_rate_last100 < 0.20:
        return "C"
    if avg_profit >= 150_000 and viol_rate_last100 < 0.40:
        return "D"
    return "F"


class TrainingProgressCallback(BaseCallback):
    """Track per-episode metrics, log custom TensorBoard scalars, dump training_log.pkl.

    Terminal info fields consumed from env._info() (when terminated=True):
        kpi_psc_count, kpi_stockout_events, episode_action_counts, violation, violation_type
    These are set in RoastingMaskEnv._info() only when the episode is over.
    """

    def __init__(
        self,
        run_dir: Path,
        time_budget_seconds: float | None = None,
        print_interval_seconds: float = 30.0,
        rolling_window: int = 100,
        best_model_dir: Path | None = None,
        best_model_stem: str = "best_training_profit_model",
    ) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.time_budget_seconds = time_budget_seconds
        self.print_interval_seconds = print_interval_seconds
        self.rolling_window = max(1, rolling_window)
        self.best_model_dir = best_model_dir
        self.best_model_stem = best_model_stem

        # Full episode history (for training_log.pkl)
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int] = []
        self._ep_violations: list[bool] = []
        self._ep_action_counts: list[dict] = []
        self._ep_psc_counts: list[int] = []
        self._ep_stockouts: list[int] = []

        # Rolling windows for live stats
        self._reward_window: deque[float] = deque(maxlen=self.rolling_window)
        self._violation_window: deque[bool] = deque(maxlen=self.rolling_window)

        # Violation type aggregates
        self._violation_type_counts: dict[str, int] = defaultdict(int)

        # Public state (read by train_maskedppo.py after training)
        # latest_profit / best_profit track SHAPED episode return (reward sum).
        # latest_net_profit / best_net_profit track TRUE KPI net_profit (unshaped).
        self.latest_profit = 0.0
        self.best_profit = float("-inf")
        self.latest_net_profit = 0.0
        self.best_net_profit = float("-inf")
        self._net_profit_window: deque[float] = deque(maxlen=self.rolling_window)
        self.completed_episodes = 0
        self.stop_reason = "timesteps"

        self._started_at = 0.0
        self._last_print_at = 0.0

        self.best_model_zip_path = (
            (self.best_model_dir / self.best_model_stem).with_suffix(".zip")
            if self.best_model_dir is not None
            else None
        )

    # ------------------------------------------------------------------
    # SB3 lifecycle
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._started_at = time.perf_counter()
        self._last_print_at = self._started_at

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is None:
                continue

            reward = float(episode["r"])
            length = int(episode["l"])
            self.completed_episodes += 1
            self.latest_profit = reward
            net_profit = float(info.get("net_profit", reward))
            self.latest_net_profit = net_profit
            if net_profit > self.best_net_profit:
                self.best_net_profit = net_profit

            # Store full history
            self._ep_rewards.append(reward)
            self._ep_lengths.append(length)
            self._reward_window.append(reward)
            self._net_profit_window.append(net_profit)

            # Violation info from terminal step
            was_violation = info.get("violation", False)
            self._ep_violations.append(was_violation)
            self._violation_window.append(was_violation)
            if was_violation:
                vtype = info.get("violation_type", "unknown")
                self._violation_type_counts[vtype] += 1

            # KPI / action fields exposed by RoastingMaskEnv._info() at terminal
            self._ep_psc_counts.append(int(info.get("kpi_psc_count", 0)))
            self._ep_stockouts.append(int(info.get("kpi_stockout_events", 0)))
            self._ep_action_counts.append(info.get("episode_action_counts", {}))

            # Per-episode TensorBoard scalars
            self._log_episode_scalars(reward, length, was_violation, info)

            if reward > self.best_profit:
                self.best_profit = reward
                self._save_best_training_model(reward)

        now = time.perf_counter()
        if now - self._last_print_at >= self.print_interval_seconds:
            self._print_progress(now)
            self._last_print_at = now

        if self.time_budget_seconds is not None and now - self._started_at >= self.time_budget_seconds:
            self.stop_reason = f"time budget ({self.time_budget_seconds:.0f}s)"
            print(
                f"[ppo-progress] stopping at {self.num_timesteps:,} timesteps after "
                f"{now - self._started_at:.1f}s due to time budget"
            )
            return False
        return True

    def _on_rollout_end(self) -> None:
        # Rolling window profit stats
        if self._reward_window:
            self.logger.record("rollout/mean_reward", statistics.mean(self._reward_window))
        self.logger.record("rollout/best_profit", self.best_profit if self.completed_episodes else 0.0)
        self.logger.record("rollout/completed_episodes", self.completed_episodes)

        # Violation rate (rolling window)
        if self._violation_window:
            viol_rate = sum(self._violation_window) / len(self._violation_window)
            self.logger.record("violations/episode_rate", viol_rate)

        # Cumulative violation type counts
        for vtype, count in self._violation_type_counts.items():
            safe = vtype.replace("-", "_").replace("/", "_")
            self.logger.record(f"violations/type_{safe}", count)

    def _on_training_end(self) -> None:
        self._print_progress(time.perf_counter(), force=True)
        self._save_training_log()
        self._save_training_stats()

    # ------------------------------------------------------------------
    # Custom TensorBoard scalars (per episode)
    # ------------------------------------------------------------------

    def _log_episode_scalars(
        self, reward: float, length: int, violation: bool, info: dict
    ) -> None:
        # reward is the shaped episode return (sum of per-step rewards incl. shaping
        # bonuses like mto_completion_bonus, rc_maintenance_bonus, completion_bonus).
        # The env exposes the unshaped KPI profit via info["net_profit"].
        self.logger.record("kpi/reward_sum", reward)
        self.logger.record("kpi/net_profit", float(info.get("net_profit", reward)))
        self.logger.record("kpi/episode_length", length)
        self.logger.record("kpi/psc_count", float(info.get("kpi_psc_count", 0)))
        self.logger.record("kpi/stockout_count", float(info.get("kpi_stockout_events", 0)))

        # Violation type scalars (1 if this episode had that type, 0 otherwise)
        vtype = info.get("violation_type", "") if violation else ""
        self.logger.record("violations/rc_negative", int(vtype.startswith("rc_negative")))
        self.logger.record("violations/gc_negative", int(vtype.startswith("gc_negative")))
        self.logger.record("violations/rc_overflow", int(vtype.startswith("rc_overflow")))
        self.logger.record("violations/invalid_action", int(vtype.startswith("invalid_action")))

        # Action distribution scalars
        action_counts = info.get("episode_action_counts", {})
        if action_counts:
            total = sum(action_counts.values()) or 1
            wait_frac = action_counts.get(20, 0) / total
            # Actions 0-8 are batch-start (PSC/roaster starts)
            psc_frac = sum(action_counts.get(i, 0) for i in range(9)) / total
            self.logger.record("actions/wait_fraction", wait_frac)
            self.logger.record("actions/psc_fraction", psc_frac)

    # ------------------------------------------------------------------
    # Diagnostic output files
    # ------------------------------------------------------------------

    def _save_training_log(self) -> None:
        """Save full episode history as training_log.pkl."""
        log_data = {
            "episode_rewards": self._ep_rewards,
            "episode_lengths": self._ep_lengths,
            "episode_violations": self._ep_violations,
            "episode_actions": self._ep_action_counts,
            "episode_psc_count": self._ep_psc_counts,
            "episode_stockouts": self._ep_stockouts,
        }
        log_path = self.run_dir / "training_log.pkl"
        with open(log_path, "wb") as f:
            pickle.dump(log_data, f)
        print(
            f"[ppo-progress] training_log.pkl saved ({len(self._ep_rewards)} episodes): {log_path}"
        )

    def _save_training_stats(self) -> None:
        """Compute summary stats + grade and save as training_stats.json."""
        n = len(self._ep_rewards)
        if n == 0:
            self.run_dir.joinpath("training_stats.json").write_text(
                json.dumps({"grade": "F", "n_episodes": 0}, indent=2), encoding="utf-8"
            )
            return

        rewards = self._ep_rewards
        violations = self._ep_violations
        psc_counts = self._ep_psc_counts

        # Reward slope (linear trend on last 25% of episodes)
        slope = 0.0
        if n >= 20:
            tail = max(10, n // 4)
            ys = rewards[-tail:]
            xs = list(range(len(ys)))
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            denom = sum((x - mx) ** 2 for x in xs)
            if denom > 0:
                slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom

        # Violation rates
        viol_rate_overall = sum(violations) / n
        tail100 = violations[-100:]
        viol_rate_last100 = sum(tail100) / len(tail100)

        # Final avg reward over last 1000 episodes
        final_1000 = rewards[-1000:]
        final_avg_reward_1000 = sum(final_1000) / len(final_1000)

        # WAIT fraction (average across episodes)
        wait_fracs = []
        psc_fracs = []
        for ac in self._ep_action_counts:
            total = sum(ac.values())
            if total > 0:
                wait_fracs.append(ac.get(20, 0) / total)
                psc_fracs.append(sum(ac.get(i, 0) for i in range(9)) / total)
        mean_wait_frac = sum(wait_fracs) / len(wait_fracs) if wait_fracs else 0.0
        mean_psc_frac = sum(psc_fracs) / len(psc_fracs) if psc_fracs else 0.0

        mean_psc_count = sum(psc_counts) / n if psc_counts else 0.0
        mean_stockouts = sum(self._ep_stockouts) / n if self._ep_stockouts else 0.0

        grade = _compute_grade(final_avg_reward_1000, viol_rate_last100)

        stats = {
            "grade": grade,
            "n_episodes": n,
            "final_avg_reward_1000": final_avg_reward_1000,
            "best_profit": self.best_profit,  # shaped reward
            "best_net_profit": self.best_net_profit,  # true KPI net_profit
            "final_avg_net_profit_1000": (
                sum(list(self._net_profit_window)[-1000:]) / min(len(self._net_profit_window), 1000)
                if self._net_profit_window else 0.0
            ),
            "reward_slope": slope,
            "violation_rate_overall": viol_rate_overall,
            "violation_rate_last100": viol_rate_last100,
            "violation_type_counts": dict(self._violation_type_counts),
            "mean_wait_fraction": mean_wait_frac,
            "mean_psc_fraction": mean_psc_frac,
            "mean_psc_count": mean_psc_count,
            "mean_stockouts": mean_stockouts,
        }
        stats_path = self.run_dir / "training_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(
            f"[ppo-progress] training_stats.json: grade={grade} "
            f"final_avg=${final_avg_reward_1000:,.0f} "
            f"viol_rate_last100={viol_rate_last100:.1%} "
            f"wait={mean_wait_frac:.1%} psc={mean_psc_count:.1f}"
        )

    # ------------------------------------------------------------------
    # Best-model persistence
    # ------------------------------------------------------------------

    def _save_best_training_model(self, profit: float) -> None:
        if self.best_model_dir is None or self.best_model_zip_path is None:
            return
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        save_base = self.best_model_dir / self.best_model_stem
        try:
            self.model.save(str(save_base))
            (self.best_model_dir / "best_training_profit.json").write_text(
                json.dumps(
                    {
                        "profit": profit,
                        "timesteps": int(self.num_timesteps),
                        "completed_episodes": int(self.completed_episodes),
                        "model_path": str(self.best_model_zip_path),
                        "violation_terminations": int(sum(self._ep_violations)),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[ppo-progress] warning: could not save best training model: {exc}")

    # ------------------------------------------------------------------
    # Console progress
    # ------------------------------------------------------------------

    def _print_progress(self, now: float, force: bool = False) -> None:
        elapsed = now - self._started_at
        if not force and self.completed_episodes == 0:
            print(
                f"[ppo-progress] steps={self.num_timesteps:,} elapsed={elapsed:.1f}s "
                "episodes=0 waiting for first episode..."
            )
            return
        n = self.completed_episodes
        mean_r = statistics.mean(self._reward_window) if self._reward_window else 0.0
        mean_np = statistics.mean(self._net_profit_window) if self._net_profit_window else 0.0
        viol_rate = (
            f"{sum(self._violation_window) / len(self._violation_window):.0%}"
            if self._violation_window
            else "n/a"
        )
        total_viols = sum(self._ep_violations)
        mean_psc = sum(self._ep_psc_counts[-100:]) / len(self._ep_psc_counts[-100:]) if self._ep_psc_counts else 0.0
        print(
            f"[ppo-progress] steps={self.num_timesteps:,} elapsed={elapsed:.1f}s "
            f"eps={n:,} latest_r=${self.latest_profit:,.0f} best_r=${self.best_profit:,.0f} "
            f"latest_np=${self.latest_net_profit:,.0f} best_np=${self.best_net_profit:,.0f} "
            f"mean_r_last{min(n, self.rolling_window)}=${mean_r:,.0f} "
            f"mean_np_last{min(n, self.rolling_window)}=${mean_np:,.0f} "
            f"viols={total_viols} viol_rate={viol_rate} psc={mean_psc:.1f}"
        )

    # ------------------------------------------------------------------
    # Properties for train_maskedppo.py to read after training
    # ------------------------------------------------------------------

    @property
    def violation_terminations(self) -> int:
        return int(sum(self._ep_violations))

    @property
    def total_violations(self) -> int:
        return sum(self._violation_type_counts.values())

    @property
    def final_avg_reward_1000(self) -> float:
        if not self._ep_rewards:
            return 0.0
        tail = self._ep_rewards[-1000:]
        return sum(tail) / len(tail)

    @property
    def violation_rate_last100(self) -> float:
        if not self._ep_violations:
            return 1.0
        tail = self._ep_violations[-100:]
        return sum(tail) / len(tail)


class AdaptiveEntropyCallback(BaseCallback):
    """Monitor policy entropy and boost ent_coef when it collapses.

    Checks `approx_kl` at the end of each rollout. If KL drops below
    `kl_floor` for `patience` consecutive rollouts, multiply ent_coef by
    `boost_factor`. Caps ent_coef at `max_ent_coef` to prevent instability.
    """

    def __init__(
        self,
        kl_floor: float = 1e-6,
        entropy_floor: float = 0.05,
        patience: int = 5,
        boost_factor: float = 1.5,
        max_ent_coef: float = 0.2,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.kl_floor = kl_floor
        self.entropy_floor = entropy_floor
        self.patience = patience
        self.boost_factor = boost_factor
        self.max_ent_coef = max_ent_coef
        self._stall_count = 0
        self._boost_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Read latest approx_kl from SB3 logger (set during train())
        approx_kl = self.model.logger.name_to_value.get("train/approx_kl", None)
        entropy = self.model.logger.name_to_value.get("train/entropy_loss", None)

        # Convert entropy_loss to positive (SB3 logs negative entropy)
        abs_entropy = abs(entropy) if entropy is not None else 1.0

        stalled = False
        if approx_kl is not None and approx_kl < self.kl_floor:
            stalled = True
        if abs_entropy < self.entropy_floor:
            stalled = True

        if stalled:
            self._stall_count += 1
        else:
            self._stall_count = 0

        if self._stall_count >= self.patience:
            old_ent = self.model.ent_coef
            new_ent = min(old_ent * self.boost_factor, self.max_ent_coef)
            if new_ent > old_ent:
                self.model.ent_coef = new_ent
                self._boost_count += 1
                if self.verbose:
                    print(
                        f"[adaptive-entropy] BOOST #{self._boost_count}: "
                        f"ent_coef {old_ent:.4f} -> {new_ent:.4f} "
                        f"(kl={approx_kl:.2e}, entropy={abs_entropy:.4f}, "
                        f"stall={self._stall_count})"
                    )
            self._stall_count = 0  # reset after boost

        # Log current ent_coef to TensorBoard
        self.logger.record("adaptive/ent_coef", self.model.ent_coef)
        self.logger.record("adaptive/stall_count", self._stall_count)
        self.logger.record("adaptive/boost_count", self._boost_count)


class ValueHeadPerturbCallback(BaseCallback):
    """Inject noise into the value network when the policy gradient stalls.

    When `approx_kl` stays below `kl_floor` for `patience` consecutive rollouts,
    add Gaussian noise to the value function head weights. This de-converges
    the value function, making advantage estimates noisy again, which restores
    the policy gradient signal.
    """

    def __init__(
        self,
        kl_floor: float = 1e-4,
        ev_ceiling: float = 0.5,
        patience: int = 3,
        noise_std: float = 0.1,
        max_perturbs: int = 50,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.kl_floor = kl_floor
        self.ev_ceiling = ev_ceiling
        self.patience = patience
        self.noise_std = noise_std
        self.max_perturbs = max_perturbs
        self._stall_count = 0
        self._perturb_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        import torch
        approx_kl = self.model.logger.name_to_value.get("train/approx_kl", None)
        expl_var = self.model.logger.name_to_value.get("train/explained_variance", None)

        # Stall = KL too low OR explained_variance too high (value converged)
        kl_stalled = approx_kl is not None and approx_kl < self.kl_floor
        ev_stalled = expl_var is not None and expl_var > self.ev_ceiling

        if kl_stalled or ev_stalled:
            self._stall_count += 1
        else:
            self._stall_count = max(0, self._stall_count - 1)  # decay instead of hard reset

        if self._stall_count >= self.patience and self._perturb_count < self.max_perturbs:
            # Perturb value function head weights
            vf_net = self.model.policy.value_net
            with torch.no_grad():
                for param in vf_net.parameters():
                    noise = torch.randn_like(param) * self.noise_std
                    param.add_(noise)
            self._perturb_count += 1
            if self.verbose:
                print(
                    f"[value-perturb] PERTURB #{self._perturb_count}: "
                    f"added noise (std={self.noise_std}) to value head "
                    f"(kl={approx_kl:.2e}, ev={expl_var:.3f}, "
                    f"stall={self._stall_count})"
                )
            self._stall_count = 0

        self.logger.record("perturb/stall_count", self._stall_count)
        self.logger.record("perturb/perturb_count", self._perturb_count)


def build_training_callbacks(
    run_dir: Path,
    eval_env,
    eval_freq: int,
    checkpoint_freq: int,
    deterministic_eval: bool = True,
    time_budget_seconds: float | None = None,
    print_interval_seconds: float = 30.0,
    adaptive_entropy: bool = True,
    value_perturb: bool = False,
):
    callbacks = [
        TrainingProgressCallback(
            run_dir=run_dir,
            time_budget_seconds=time_budget_seconds,
            print_interval_seconds=print_interval_seconds,
            best_model_dir=run_dir / "checkpoints",
        )
    ]
    if adaptive_entropy:
        callbacks.append(AdaptiveEntropyCallback())
    if value_perturb:
        callbacks.append(ValueHeadPerturbCallback())
    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, checkpoint_freq),
                save_path=str(run_dir / "checkpoints" / "periodic"),
                name_prefix="maskedppo",
            )
        )
    if eval_env is not None:
        callbacks.append(
            MaskableEvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "checkpoints"),
                log_path=str(run_dir / "checkpoints" / "eval_logs"),
                eval_freq=max(1, eval_freq),
                deterministic=deterministic_eval,
                render=False,
            )
        )
    return CallbackList(callbacks)
