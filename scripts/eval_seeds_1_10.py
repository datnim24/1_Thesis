"""Evaluate QL, PPO (C25), RL-HH on seeds 1..10. Save per-seed JSON + HTML reports
and a summary.md / summary.json under results/10seedseval/.
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

C25_RUN = _ROOT / "PPOmask" / "outputs" / "20260420_193326_c25_busta_fix"
PPO_MODEL = C25_RUN / "checkpoints" / "final_model.zip"

OUT_DIR = _ROOT / "results" / "10seedseval"
SEEDS = list(range(1, 11))


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _generate_html(result: dict, output_path: Path) -> None:
    from plot_result import _build_html
    html = _build_html(result, compare=None, offline=True)
    output_path.write_text(html, encoding="utf-8")


def _build_result(solver_name: str, engine_tag: str, kpi, state, params, data, seed: int,
                  model_path: str | None) -> dict:
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
            "solver_engine": engine_tag,
            "solver_name": solver_name,
            "status": "Completed",
            "solve_time_sec": 0.0,
            "input_dir": str(data.input_dir),
            "model_path": model_path or "",
            "notes": "Evaluation-only rerun on held-out seed.",
        },
        experiment={
            "lambda_rate": float(params.get("ups_lambda", 0)),
            "mu_mean": float(params.get("ups_mu", 0)),
            "seed": seed,
        },
        kpi=kpi.to_dict(),
        schedule=schedule,
        cancelled_batches=cancelled,
        ups_events=ups_entries,
        restocks=restocks,
        parameters=_build_export_params(params),
    )


def run_ql(params, data, engine, ups):
    from q_learning.q_strategy import QStrategy, load_q_table
    q_table = load_q_table(str(Q_TABLE))
    strategy = QStrategy(params, q_table=q_table)
    return engine.run(strategy, ups)


def run_ppo(params, data, engine, ups):
    from PPOmask.Engine.ppo_strategy import PPOStrategy
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(str(PPO_MODEL))
    strategy = PPOStrategy(data, model, deterministic=True)
    return engine.run(strategy, ups)


def run_rlhh(params, data, engine, ups):
    from rl_hh.meta_agent import DuelingDDQNAgent
    from rl_hh.rl_hh_strategy import RLHHStrategy
    agent = DuelingDDQNAgent()
    agent.load_checkpoint(str(RLHH_CKPT))
    agent.epsilon = 0.0
    strategy = RLHHStrategy(agent, data, training=False)
    return engine.run(strategy, ups)


METHODS = [
    ("ql",   "Q-Learning", "qlearning", run_ql,   str(Q_TABLE)),
    ("ppo",  "MaskedPPO (C25)", "ppo_mask", run_ppo, str(PPO_MODEL)),
    ("rlhh", "RL-HH (Dueling DDQN)", "rl_hh", run_rlhh, str(RLHH_CKPT)),
]


def _summarize_kpi(d: dict) -> dict:
    tm = d.get("tardiness_min", {})
    return {
        "net_profit": d.get("net_profit", 0),
        "revenue": d.get("total_revenue", 0),
        "psc": d.get("psc_count", 0),
        "ndg": d.get("ndg_count", 0),
        "busta": d.get("busta_count", 0),
        "tard_cost": d.get("tard_cost", 0),
        "setup_cost": d.get("setup_cost", 0),
        "setup_events": d.get("setup_events", 0),
        "stockout_cost": d.get("stockout_cost", 0),
        "idle_cost": d.get("idle_cost", 0),
        "restocks": d.get("restock_count", 0),
        "j1_late": tm.get("J1", 0),
        "j2_late": tm.get("J2", 0),
    }


def main() -> None:
    from env.data_bridge import get_sim_params
    from env.simulation_engine import SimulationEngine
    from env.ups_generator import generate_ups_events
    from PPOmask.Engine.data_loader import load_data

    params = get_sim_params()
    data = load_data()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {tag: [] for tag, *_ in METHODS}

    print(f"\n{'='*100}")
    print(f"10-Seed Evaluation: QL | PPO (C25) | RL-HH  |  Output: {OUT_DIR}")
    print(f"{'='*100}")
    print(f"{'Seed':>4} | {'Method':<20} | {'NetProfit':>11} | {'Rev':>8} | "
          f"{'PSC/NDG/BUS':>11} | {'Tard':>8} | {'Setup':>6} | {'Idle':>7} | Restocks")
    print("-" * 100)

    for seed in SEEDS:
        ups = generate_ups_events(
            params.get("ups_lambda", 0), params.get("ups_mu", 0),
            seed=seed, shift_length=int(params["SL"]),
            roasters=list(params["roasters"]),
        )
        seed_dir = OUT_DIR / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for tag, solver_name, engine_tag, runner, mdl_path in METHODS:
            engine = SimulationEngine(params)
            kpi, state = runner(params, data, engine, list(ups))
            result = _build_result(solver_name, engine_tag, kpi, state, params, data, seed, mdl_path)
            s = _summarize_kpi(kpi.to_dict())
            all_results[tag].append({"seed": seed, **s})

            _save_json(result, seed_dir / f"{tag}_result.json")
            _generate_html(result, seed_dir / f"{tag}_report.html")

            print(f"{seed:>4} | {solver_name:<20} | ${s['net_profit']:>10,.0f} | "
                  f"${s['revenue']:>7,.0f} | {s['psc']:>3}/{s['ndg']}/{s['busta']:<3} | "
                  f"${s['tard_cost']:>7,.0f} | ${s['setup_cost']:>5,.0f} | "
                  f"${s['idle_cost']:>6,.0f} | {s['restocks']}")

    # --- Aggregate ---
    print(f"\n{'='*100}\nAVERAGES across seeds 1..10\n{'='*100}")
    agg: dict[str, dict] = {}
    print(f"{'Method':<20} | {'Avg Net':>11} | {'Std Net':>10} | {'Min Net':>11} | "
          f"{'Max Net':>11} | {'BUSTA μ':>7} | {'Tard μ':>9} | Wins")
    print("-" * 110)

    nets_per_seed = {seed: {} for seed in SEEDS}
    for tag, runs in all_results.items():
        solver_name = next(m[1] for m in METHODS if m[0] == tag)
        nets = [r["net_profit"] for r in runs]
        busta = [r["busta"] for r in runs]
        tards = [r["tard_cost"] for r in runs]
        for r in runs:
            nets_per_seed[r["seed"]][tag] = r["net_profit"]
        agg[tag] = {
            "solver_name": solver_name,
            "avg_net": statistics.mean(nets),
            "std_net": statistics.stdev(nets) if len(nets) > 1 else 0.0,
            "min_net": min(nets),
            "max_net": max(nets),
            "avg_busta": statistics.mean(busta),
            "avg_tard": statistics.mean(tards),
            "avg_revenue": statistics.mean(r["revenue"] for r in runs),
            "avg_setup": statistics.mean(r["setup_cost"] for r in runs),
            "avg_idle": statistics.mean(r["idle_cost"] for r in runs),
            "avg_restocks": statistics.mean(r["restocks"] for r in runs),
            "per_seed": runs,
        }

    # Per-seed wins
    wins = {tag: 0 for tag in agg}
    for seed, nets in nets_per_seed.items():
        winner = max(nets, key=nets.get)
        wins[winner] += 1

    for tag, a in agg.items():
        print(f"{a['solver_name']:<20} | ${a['avg_net']:>10,.0f} | ${a['std_net']:>9,.0f} | "
              f"${a['min_net']:>10,.0f} | ${a['max_net']:>10,.0f} | "
              f"{a['avg_busta']:>6.1f}/5 | ${a['avg_tard']:>8,.0f} | {wins[tag]}/10")

    # Head-to-head PPO vs RL-HH
    ppo_wins_over_rlhh = sum(1 for s in SEEDS if nets_per_seed[s]["ppo"] > nets_per_seed[s]["rlhh"])
    print(f"\nPPO beats RL-HH on {ppo_wins_over_rlhh}/10 seeds.")
    avg_gap = agg["ppo"]["avg_net"] - agg["rlhh"]["avg_net"]
    print(f"Avg gap (PPO - RL-HH): ${avg_gap:,.0f}")

    # --- Write summary files ---
    summary_json = {
        "seeds": SEEDS,
        "methods": {tag: {k: v for k, v in a.items() if k != "per_seed"} for tag, a in agg.items()},
        "per_seed_net_profits": nets_per_seed,
        "wins": wins,
        "ppo_wins_over_rlhh": ppo_wins_over_rlhh,
        "avg_gap_ppo_minus_rlhh": avg_gap,
    }
    _save_json(summary_json, OUT_DIR / "summary.json")

    md_lines = [
        "# 10-Seed Evaluation — QL vs PPO (C25) vs RL-HH",
        "",
        f"Evaluated on seeds {SEEDS[0]}..{SEEDS[-1]} (UPS λ={params.get('ups_lambda')}, μ={params.get('ups_mu')}min).",
        "",
        "## Averages",
        "",
        "| Method | Avg Net | Std Net | Min | Max | BUSTA μ | Tard μ | Wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, a in agg.items():
        md_lines.append(
            f"| {a['solver_name']} | ${a['avg_net']:,.0f} | ${a['std_net']:,.0f} | "
            f"${a['min_net']:,.0f} | ${a['max_net']:,.0f} | {a['avg_busta']:.1f}/5 | "
            f"${a['avg_tard']:,.0f} | {wins[tag]}/10 |"
        )
    md_lines += [
        "",
        f"**Head-to-head:** PPO beats RL-HH on **{ppo_wins_over_rlhh}/10** seeds. "
        f"Avg net gap = ${avg_gap:,.0f}.",
        "",
        "## Per-Seed Net Profit",
        "",
        "| Seed | Q-Learning | PPO (C25) | RL-HH | Winner |",
        "|---:|---:|---:|---:|---|",
    ]
    for seed in SEEDS:
        nets = nets_per_seed[seed]
        winner = max(nets, key=nets.get)
        winner_label = next(m[1] for m in METHODS if m[0] == winner)
        md_lines.append(
            f"| {seed} | ${nets['ql']:,.0f} | ${nets['ppo']:,.0f} | ${nets['rlhh']:,.0f} | {winner_label} |"
        )

    md_lines += [
        "",
        "## Per-Method Detail",
        "",
    ]
    for tag, a in agg.items():
        md_lines.append(f"### {a['solver_name']}")
        md_lines.append("")
        md_lines.append("| Seed | Net | Rev | PSC/NDG/BUS | Tard | Setup | Idle | Restocks |")
        md_lines.append("|---:|---:|---:|---|---:|---:|---:|---:|")
        for r in a["per_seed"]:
            md_lines.append(
                f"| {r['seed']} | ${r['net_profit']:,.0f} | ${r['revenue']:,.0f} | "
                f"{r['psc']}/{r['ndg']}/{r['busta']} | ${r['tard_cost']:,.0f} | "
                f"${r['setup_cost']:,.0f} | ${r['idle_cost']:,.0f} | {r['restocks']} |"
            )
        md_lines.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote summary.md and summary.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
