"""Evaluate C26 PPO (best+final) + QL + RL-HH on seeds 1..10.
Cross-reference C25 numbers from results/10seedseval/summary.json for delta analysis.
Output: results/10seedseval_c26/
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MASTER_RUN = _ROOT / "results" / "20260419_233328_7800x_6h_1"
Q_TABLE = MASTER_RUN / "q_table_master.pkl"
RLHH_CKPT = MASTER_RUN / "rlhh_best.pt"

C25_SUMMARY = _ROOT / "results" / "10seedseval" / "summary.json"

# C26 paths — resolved at runtime by scanning PPOmask/outputs/*c26_multiseed*
C26_RUN_PATTERN = "*c26_multiseed"

OUT_DIR = _ROOT / "results" / "10seedseval_c26"
SEEDS = list(range(1, 11))


def _find_c26_run() -> Path:
    candidates = sorted((_ROOT / "PPOmask" / "outputs").glob(C26_RUN_PATTERN))
    if not candidates:
        raise FileNotFoundError(f"No C26 run dir matching {C26_RUN_PATTERN}")
    return candidates[-1]


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _generate_html(result: dict, output_path: Path) -> None:
    from plot_result import _build_html
    html = _build_html(result, compare=None, offline=True)
    output_path.write_text(html, encoding="utf-8")


def _build_result(solver_name, engine_tag, kpi, state, params, data, seed, model_path):
    from result_schema import create_result
    from rl_hh.export_result import _batch_to_schedule_entry, _restock_to_entry, _ups_to_entry
    from master_eval import _build_export_params

    schedule = [_batch_to_schedule_entry(b, params) for b in state.completed_batches]
    cancelled = [_batch_to_schedule_entry(b, params) for b in state.cancelled_batches]
    for c in cancelled:
        c["status"] = "cancelled"
    restocks = [_restock_to_entry(r) for r in state.completed_restocks]
    ups_entries = [_ups_to_entry(e) for e in state.ups_events_fired]

    return create_result(
        metadata={
            "solver_engine": engine_tag, "solver_name": solver_name,
            "status": "Completed", "solve_time_sec": 0.0,
            "input_dir": str(data.input_dir), "model_path": str(model_path),
            "notes": "C26 held-out eval — per-episode seed randomization in training.",
        },
        experiment={
            "lambda_rate": float(params.get("ups_lambda", 0)),
            "mu_mean": float(params.get("ups_mu", 0)), "seed": seed,
        },
        kpi=kpi.to_dict(), schedule=schedule, cancelled_batches=cancelled,
        ups_events=ups_entries, restocks=restocks,
        parameters=_build_export_params(params),
    )


def run_ql(params, data, engine, ups):
    from q_learning.q_strategy import QStrategy, load_q_table
    strategy = QStrategy(params, q_table=load_q_table(str(Q_TABLE)))
    return engine.run(strategy, ups)


def run_ppo_factory(model_path: Path):
    def _run(params, data, engine, ups):
        from PPOmask.Engine.ppo_strategy import PPOStrategy
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(model_path))
        strategy = PPOStrategy(data, model, deterministic=True)
        return engine.run(strategy, ups)
    return _run


def run_rlhh(params, data, engine, ups):
    from rl_hh.meta_agent import DuelingDDQNAgent
    from rl_hh.rl_hh_strategy import RLHHStrategy
    agent = DuelingDDQNAgent()
    agent.load_checkpoint(str(RLHH_CKPT))
    agent.epsilon = 0.0
    strategy = RLHHStrategy(agent, data, training=False)
    return engine.run(strategy, ups)


def _summarize(d: dict) -> dict:
    tm = d.get("tardiness_min", {})
    return {
        "net_profit": d.get("net_profit", 0), "revenue": d.get("total_revenue", 0),
        "psc": d.get("psc_count", 0), "ndg": d.get("ndg_count", 0),
        "busta": d.get("busta_count", 0),
        "tard_cost": d.get("tard_cost", 0), "setup_cost": d.get("setup_cost", 0),
        "setup_events": d.get("setup_events", 0),
        "idle_cost": d.get("idle_cost", 0), "restocks": d.get("restock_count", 0),
        "j1_late": tm.get("J1", 0), "j2_late": tm.get("J2", 0),
    }


def main() -> None:
    from env.data_bridge import get_sim_params
    from env.simulation_engine import SimulationEngine
    from env.ups_generator import generate_ups_events
    from PPOmask.Engine.data_loader import load_data

    c26_run = _find_c26_run()
    c26_best = c26_run / "checkpoints" / "best_training_profit_model.zip"
    c26_final = c26_run / "checkpoints" / "final_model.zip"
    if not c26_best.exists():
        raise FileNotFoundError(f"C26 best not found: {c26_best}")
    if not c26_final.exists():
        raise FileNotFoundError(f"C26 final not found: {c26_final}")
    print(f"C26 run: {c26_run.name}")
    print(f"  best  : {c26_best.name}")
    print(f"  final : {c26_final.name}")

    params = get_sim_params()
    data = load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    methods = [
        ("ql", "Q-Learning", "qlearning", run_ql, str(Q_TABLE)),
        ("rlhh", "RL-HH (Dueling DDQN)", "rl_hh", run_rlhh, str(RLHH_CKPT)),
        ("ppo_c26_best", "MaskedPPO (C26 best)", "ppo_mask", run_ppo_factory(c26_best), str(c26_best)),
        ("ppo_c26_final", "MaskedPPO (C26 final)", "ppo_mask", run_ppo_factory(c26_final), str(c26_final)),
    ]

    all_runs: dict[str, list[dict]] = {tag: [] for tag, *_ in methods}

    print(f"\n{'='*110}\n10-Seed Eval | Output: {OUT_DIR}\n{'='*110}")
    print(f"{'Seed':>4} | {'Method':<24} | {'NetProfit':>11} | {'Rev':>8} | "
          f"{'PSC/NDG/BUS':>11} | {'Tard':>8} | {'Setup':>6} | {'Idle':>7} | Restocks")
    print("-" * 110)

    for seed in SEEDS:
        ups = generate_ups_events(
            params.get("ups_lambda", 0), params.get("ups_mu", 0),
            seed=seed, shift_length=int(params["SL"]),
            roasters=list(params["roasters"]),
        )
        seed_dir = OUT_DIR / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for tag, solver_name, engine_tag, runner, mdl_path in methods:
            engine = SimulationEngine(params)
            kpi, state = runner(params, data, engine, list(ups))
            result = _build_result(solver_name, engine_tag, kpi, state, params, data, seed, mdl_path)
            s = _summarize(kpi.to_dict())
            all_runs[tag].append({"seed": seed, **s})

            _save_json(result, seed_dir / f"{tag}_result.json")
            _generate_html(result, seed_dir / f"{tag}_report.html")

            print(f"{seed:>4} | {solver_name:<24} | ${s['net_profit']:>10,.0f} | "
                  f"${s['revenue']:>7,.0f} | {s['psc']:>3}/{s['ndg']}/{s['busta']:<3} | "
                  f"${s['tard_cost']:>7,.0f} | ${s['setup_cost']:>5,.0f} | "
                  f"${s['idle_cost']:>6,.0f} | {s['restocks']}")

    # --- Aggregate ---
    print(f"\n{'='*110}\nAVERAGES across seeds 1..10\n{'='*110}")
    agg: dict[str, dict] = {}
    nets_per_seed = {seed: {} for seed in SEEDS}
    for tag, runs in all_runs.items():
        solver_name = next(m[1] for m in methods if m[0] == tag)
        nets = [r["net_profit"] for r in runs]
        busta = [r["busta"] for r in runs]
        tards = [r["tard_cost"] for r in runs]
        for r in runs:
            nets_per_seed[r["seed"]][tag] = r["net_profit"]
        agg[tag] = {
            "solver_name": solver_name, "avg_net": statistics.mean(nets),
            "std_net": statistics.stdev(nets) if len(nets) > 1 else 0.0,
            "min_net": min(nets), "max_net": max(nets),
            "avg_busta": statistics.mean(busta), "avg_tard": statistics.mean(tards),
            "avg_revenue": statistics.mean(r["revenue"] for r in runs),
            "avg_setup": statistics.mean(r["setup_cost"] for r in runs),
            "avg_idle": statistics.mean(r["idle_cost"] for r in runs),
            "avg_restocks": statistics.mean(r["restocks"] for r in runs),
            "per_seed": runs,
        }

    wins = {tag: 0 for tag in agg}
    for seed, nets in nets_per_seed.items():
        winner = max(nets, key=nets.get)
        wins[winner] += 1

    print(f"{'Method':<24} | {'Avg Net':>11} | {'Std':>10} | {'Min':>11} | "
          f"{'Max':>11} | BUSTA | Tard  | Wins")
    print("-" * 110)
    for tag, a in agg.items():
        print(f"{a['solver_name']:<24} | ${a['avg_net']:>10,.0f} | ${a['std_net']:>9,.0f} | "
              f"${a['min_net']:>10,.0f} | ${a['max_net']:>10,.0f} | "
              f"{a['avg_busta']:.1f}/5 | ${a['avg_tard']:>8,.0f} | {wins[tag]}/10")

    # --- Load C25 for comparison ---
    c25_agg = None
    if C25_SUMMARY.exists():
        with open(C25_SUMMARY, "r", encoding="utf-8") as f:
            c25_data = json.load(f)
        if "ppo" in c25_data.get("methods", {}):
            c25_agg = c25_data["methods"]["ppo"]
            print(f"\nC25 PPO baseline (from results/10seedseval): "
                  f"avg=${c25_agg['avg_net']:,.0f}, std=${c25_agg['std_net']:,.0f}, "
                  f"min=${c25_agg['min_net']:,.0f}, max=${c25_agg['max_net']:,.0f}")

    # --- Write summary files ---
    summary_json = {
        "c26_run": str(c26_run),
        "seeds": SEEDS,
        "methods": {tag: {k: v for k, v in a.items() if k != "per_seed"} for tag, a in agg.items()},
        "per_seed_net_profits": nets_per_seed,
        "wins": wins,
        "c25_baseline": c25_agg,
    }
    _save_json(summary_json, OUT_DIR / "summary.json")

    md = ["# C26 (multi-seed) 10-Seed Evaluation", "",
          f"C26 run: `{c26_run.name}`  ",
          f"Seeds: {SEEDS}  |  UPS λ={params.get('ups_lambda')}, μ={params.get('ups_mu')}min",
          "",
          "## Averages (this run)", "",
          "| Method | Avg Net | Std Net | Min | Max | BUSTA μ | Tard μ | Wins |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, a in agg.items():
        md.append(f"| {a['solver_name']} | ${a['avg_net']:,.0f} | ${a['std_net']:,.0f} | "
                  f"${a['min_net']:,.0f} | ${a['max_net']:,.0f} | {a['avg_busta']:.1f}/5 | "
                  f"${a['avg_tard']:,.0f} | {wins[tag]}/10 |")

    if c25_agg is not None:
        best_c26 = max(agg["ppo_c26_best"]["avg_net"], agg["ppo_c26_final"]["avg_net"])
        best_c26_std = min(agg["ppo_c26_best"]["std_net"], agg["ppo_c26_final"]["std_net"])
        delta_avg = best_c26 - c25_agg["avg_net"]
        delta_std = best_c26_std - c25_agg["std_net"]
        md += ["", "## C25 (single-seed) vs C26 (multi-seed) — PPO only", "",
               "| Version | Avg Net | Std | Min | Max |",
               "|---|---:|---:|---:|---:|",
               f"| C25 (8 fixed seeds) | ${c25_agg['avg_net']:,.0f} | ${c25_agg['std_net']:,.0f} | "
               f"${c25_agg['min_net']:,.0f} | ${c25_agg['max_net']:,.0f} |",
               f"| C26 best (≈3.8M seeds) | ${agg['ppo_c26_best']['avg_net']:,.0f} | "
               f"${agg['ppo_c26_best']['std_net']:,.0f} | "
               f"${agg['ppo_c26_best']['min_net']:,.0f} | ${agg['ppo_c26_best']['max_net']:,.0f} |",
               f"| C26 final (≈3.8M seeds) | ${agg['ppo_c26_final']['avg_net']:,.0f} | "
               f"${agg['ppo_c26_final']['std_net']:,.0f} | "
               f"${agg['ppo_c26_final']['min_net']:,.0f} | ${agg['ppo_c26_final']['max_net']:,.0f} |",
               "",
               f"**Δ best-C26 − C25:** avg = ${delta_avg:+,.0f}, std = ${delta_std:+,.0f}"]

    md += ["", "## Per-Seed Net Profit", "",
           "| Seed | QL | RL-HH | C26 best | C26 final | Winner |",
           "|---:|---:|---:|---:|---:|---|"]
    for seed in SEEDS:
        nets = nets_per_seed[seed]
        winner = max(nets, key=nets.get)
        winner_label = next(m[1] for m in methods if m[0] == winner)
        md.append(f"| {seed} | ${nets['ql']:,.0f} | ${nets['rlhh']:,.0f} | "
                  f"${nets['ppo_c26_best']:,.0f} | ${nets['ppo_c26_final']:,.0f} | {winner_label} |")

    md += ["", "## Per-Method Detail", ""]
    for tag, a in agg.items():
        md += [f"### {a['solver_name']}", "",
               "| Seed | Net | Rev | PSC/NDG/BUS | Tard | Setup | Idle | Restocks |",
               "|---:|---:|---:|---|---:|---:|---:|---:|"]
        for r in a["per_seed"]:
            md.append(f"| {r['seed']} | ${r['net_profit']:,.0f} | ${r['revenue']:,.0f} | "
                      f"{r['psc']}/{r['ndg']}/{r['busta']} | ${r['tard_cost']:,.0f} | "
                      f"${r['setup_cost']:,.0f} | ${r['idle_cost']:,.0f} | {r['restocks']} |")
        md.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote summary.md and summary.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
