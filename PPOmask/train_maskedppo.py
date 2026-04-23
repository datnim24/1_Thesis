"""Train a MaskablePPO policy for the thesis roasting environment.

Usage:
    python -m PPOmask.train_maskedppo --timesteps 40000 --n-envs 8 --seed 111 --run-name PPO_NonUPS_2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from PPOmask.Engine.callbacks import build_training_callbacks
from PPOmask.Engine.data_loader import load_data
from PPOmask.Engine.roasting_env import RoastingMaskEnv

_PPO_ROOT = Path(__file__).resolve().parent


def _mask_fn(env) -> object:
    return env.unwrapped.action_masks()


def make_env_factory(data, scenario_seed: int, ups_lambda: float | None, ups_mu: float | None):
    def _factory():
        env = RoastingMaskEnv(
            data=data, scenario_seed=scenario_seed,
            ups_lambda=ups_lambda, ups_mu=ups_mu,
        )
        env = Monitor(env)
        env = ActionMasker(env, _mask_fn)
        return env
    return _factory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a MaskablePPO policy.")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--time", "-t", type=float, default=None, help="Wall-clock budget in seconds.")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=123)
    # Learning rate: --lr is a short alias for --learning-rate
    p.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--net-arch", type=str, default="256,256",
                   help="Comma-separated hidden layer sizes, e.g. '256,256'.")
    p.add_argument("--separate-networks", action="store_true",
                   help="Use separate policy and value networks (no shared backbone). "
                        "Prevents value convergence from killing policy gradient.")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--checkpoint-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--run-name", default="", help="Run name suffix.")
    p.add_argument("--ups-lambda", type=float, default=None)
    p.add_argument("--ups-mu", type=float, default=None)
    p.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv instead of DummyVecEnv.")
    p.add_argument("--smoke", action="store_true", help="Tiny smoke-training budget.")
    p.add_argument("--no-eval", action="store_true", help="Skip post-training evaluation.")
    p.add_argument("--no-open-report", action="store_true")
    p.add_argument("--no-tensorboard", action="store_true", help="Don't auto-launch TensorBoard.")
    p.add_argument("--rc-maintenance-bonus", type=float, default=None,
                   help="Per-slot reward for each line with rc_stock >= safety_stock. "
                        "Overrides shift_parameters.csv value.")
    p.add_argument("--violation-penalty", type=float, default=None,
                   help="Override violation penalty from shift_parameters.csv.")
    p.add_argument("--completion-bonus", type=float, default=None,
                   help="Lump-sum reward for completing the full shift without violation.")
    p.add_argument("--mto-completion-bonus", type=float, default=None,
                   help="Dense per-batch reward when an NDG or BUSTA batch finishes. "
                        "Converts the sparse end-of-shift skip penalty into direct "
                        "credit-assignable signal (C27 fix for multi-seed training).")
    p.add_argument("--progress-print-seconds", type=float, default=30.0)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a .zip model to resume training from (loads weights into new model).")
    p.add_argument("--lr-schedule", choices=["constant", "linear"], default="linear",
                   help="Learning rate schedule. 'linear' decays to 0 over training.")
    p.add_argument("--target-kl", type=float, default=None,
                   help="Target KL divergence for early stopping of epoch loop. "
                        "Recommended: 0.01-0.05. Prevents policy over-optimization.")
    p.add_argument("--no-adaptive-entropy", action="store_true",
                   help="Disable adaptive entropy boosting callback.")
    p.add_argument("--value-perturb", action="store_true",
                   help="Enable value head perturbation when gradient stalls.")
    p.add_argument("--normalize-reward", action="store_true",
                   help="Wrap VecEnv with VecNormalize(norm_reward=True). "
                        "Normalizes rewards to unit variance for cleaner advantages.")
    return p.parse_args(argv)


def _build_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Format: DateTime_PPO_{index} where index comes from --run-name (e.g. "1", "2")
    # or just "PPO" if no run-name given
    suffix = args.run_name or "PPO"
    run_dir = _PPO_ROOT / "outputs" / f"{timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # TensorBoard dir: shared parent outputs/ so ALL runs appear in one TB session
    (_PPO_ROOT / "outputs" / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def _find_tensorboard_exe() -> str:
    """Return the tensorboard executable path (handles Windows Scripts/ layout)."""
    from pathlib import Path as _Path
    scripts = _Path(sys.executable).parent / "Scripts" / "tensorboard.exe"
    if scripts.exists():
        return str(scripts)
    # Fallback: let the shell find it on PATH
    return "tensorboard"


def _launch_tensorboard(log_dir: Path, port: int = 6006) -> subprocess.Popen | None:
    import socket
    import time as _time
    import webbrowser
    tb_exe = _find_tensorboard_exe()
    try:
        # Kill any stale TensorBoard processes so we can bind to the port
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "tensorboard.exe"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _time.sleep(1)  # brief pause for port release
        except Exception:
            pass
        proc = subprocess.Popen(
            [tb_exe, "--logdir", str(log_dir), "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        url = f"http://localhost:{port}"
        print(f"[train] TensorBoard launching (PID {proc.pid}) at {url} ...")
        # Poll until TensorBoard actually accepts connections (up to 30 s)
        deadline = _time.time() + 30
        while _time.time() < deadline:
            _time.sleep(1)
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    break   # port is open — TensorBoard is ready
            except OSError:
                continue
        else:
            print(f"[train] Warning: TensorBoard did not bind on port {port} within 30 s")
            return proc
        print(f"[train] TensorBoard ready — opening {url}")
        webbrowser.open(url)
        return proc
    except Exception as exc:
        print(f"[train] Warning: Could not launch TensorBoard: {exc}")
        return None


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.smoke:
        args.timesteps = 2048
        args.eval_freq = 512
        args.checkpoint_freq = 512

    requested_timesteps = args.timesteps
    data = load_data(args.input_dir)
    if args.rc_maintenance_bonus is not None:
        data.rc_maintenance_bonus = args.rc_maintenance_bonus
    if args.violation_penalty is not None:
        data.violation_penalty = args.violation_penalty
    if args.completion_bonus is not None:
        data.completion_bonus = args.completion_bonus
    if args.mto_completion_bonus is not None:
        data.mto_completion_bonus = args.mto_completion_bonus
    set_random_seed(args.seed)

    run_dir = _build_run_dir(args)
    print(f"[train] Output dir: {run_dir}")
    print(f"[train] Violation penalty: {data.violation_penalty}, termination: {data.episode_termination_on_violation}")

    # Create vectorized training environments
    ups_lam = args.ups_lambda if args.ups_lambda is not None else data.ups_lambda
    ups_mu_val = args.ups_mu if args.ups_mu is not None else data.ups_mu
    env_fns = [
        make_env_factory(data, args.seed + i, ups_lam, ups_mu_val)
        for i in range(args.n_envs)
    ]
    VecEnvCls = SubprocVecEnv if args.subproc else DummyVecEnv
    vec_env = VecEnvCls(env_fns)
    if args.normalize_reward:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        print("[train] VecNormalize enabled: norm_reward=True, clip_reward=10.0")

    # Create eval environment
    eval_env = DummyVecEnv([make_env_factory(data, args.seed + 1000, ups_lam, ups_mu_val)])
    if args.normalize_reward:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    # Launch TensorBoard — shared logdir so all runs appear together
    tb_logdir = _PPO_ROOT / "outputs" / "tensorboard"
    tb_proc = None
    if not args.no_tensorboard:
        tb_proc = _launch_tensorboard(tb_logdir)

    # Parse net_arch
    net_arch_list = [int(x.strip()) for x in args.net_arch.split(",")]
    if args.separate_networks:
        net_arch = {"pi": net_arch_list, "vf": net_arch_list}
        print(f"[train] Separate policy/value networks: pi={net_arch_list}, vf={net_arch_list}")
    else:
        net_arch = net_arch_list
    policy_kwargs = {"net_arch": net_arch}

    # Learning rate schedule
    base_lr = args.learning_rate
    if args.lr_schedule == "linear":
        def lr_schedule(progress_remaining: float) -> float:
            """Linear decay: progress_remaining goes from 1.0 to 0.0."""
            return base_lr * max(progress_remaining, 0.05)  # floor at 5% of base
        effective_lr = lr_schedule
        print(f"[train] LR schedule: linear decay {base_lr} -> {base_lr * 0.05}")
    else:
        effective_lr = base_lr

    # Build model (or resume from checkpoint)
    if args.resume_from:
        resume_path = args.resume_from
        print(f"[train] Resuming from: {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=vec_env,
            learning_rate=effective_lr,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            tensorboard_log=str(tb_logdir),
        )
        model.seed = args.seed
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=effective_lr,
            gamma=args.gamma,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gae_lambda=args.gae_lambda,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(tb_logdir),
        )
    if args.target_kl:
        print(f"[train] Target KL: {args.target_kl} (early stopping of epoch loop)")

    # Build callbacks
    callbacks = build_training_callbacks(
        run_dir=run_dir,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        time_budget_seconds=args.time,
        print_interval_seconds=args.progress_print_seconds,
        adaptive_entropy=not args.no_adaptive_entropy,
        value_perturb=args.value_perturb,
    )

    # Train
    print(f"[train] Starting training: {requested_timesteps:,} steps, {args.n_envs} envs, seed={args.seed}")
    t0 = time.perf_counter()
    model.learn(total_timesteps=requested_timesteps, callback=callbacks, tb_log_name=run_dir.name)
    elapsed = time.perf_counter() - t0
    print(f"[train] Training complete in {elapsed:.1f}s")

    # Save final model
    final_path = run_dir / "checkpoints" / "final_model"
    model.save(str(final_path))
    print(f"[train] Final model saved: {final_path}.zip")

    # Save training metadata (diagnostic protocol fields)
    progress_cb = callbacks.callbacks[0]  # TrainingProgressCallback

    # Aggregate action distribution across all episodes
    action_dist: dict[str, int] = {}
    for ep_ac in progress_cb._ep_action_counts:
        for aid, cnt in ep_ac.items():
            key = str(aid)
            action_dist[key] = action_dist.get(key, 0) + cnt

    meta = {
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "seed": args.seed,
        "timesteps_requested": requested_timesteps,
        "timesteps_actual": int(model.num_timesteps),
        "elapsed_seconds": elapsed,
        # Hyperparameters
        "n_envs": args.n_envs,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gae_lambda": args.gae_lambda,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "net_arch": net_arch,
        "ent_coef": args.ent_coef,
        "clip_range": args.clip_range,
        "ups_lambda": ups_lam,
        "ups_mu": ups_mu_val,
        # Constraint config
        "violation_penalty": data.violation_penalty,
        "episode_termination_on_violation": data.episode_termination_on_violation,
        "rc_maintenance_bonus": data.rc_maintenance_bonus,
        "completion_bonus": data.completion_bonus,
        "mto_completion_bonus": data.mto_completion_bonus,
        # Episode stats
        "completed_episodes": progress_cb.completed_episodes,
        "best_profit": progress_cb.best_profit if progress_cb.completed_episodes else None,
        "final_avg_reward_1000": progress_cb.final_avg_reward_1000,
        "violation_terminations": progress_cb.violation_terminations,
        "violation_rate_last100": progress_cb.violation_rate_last100,
        "violation_counts": dict(progress_cb._violation_type_counts),
        "action_distribution": action_dist,
        "early_termination_rate": progress_cb.violation_rate_last100,
        "stop_reason": progress_cb.stop_reason,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    vec_env.close()
    eval_env.close()

    # Auto-run evaluation
    if not args.no_eval:
        print("\n[train] Running post-training evaluation...")
        try:
            eval_argv = [
                "--model-path", str(final_path) + ".zip",
                "--run-dir", str(run_dir),
                "--seed", str(args.seed),
                "--deterministic-runs", str(args.eval_episodes),
                "--stochastic-runs", "0",
                "--no-open-report",
            ]
            if ups_lam > 0:
                eval_argv += ["--ups-lambda", str(ups_lam), "--ups-mu", str(ups_mu_val)]
            from PPOmask.evaluate_maskedppo import main as eval_main
            eval_main(eval_argv)
        except Exception as exc:
            print(f"[train] Warning: Post-training evaluation failed: {exc}")

    return run_dir


if __name__ == "__main__":
    main()
