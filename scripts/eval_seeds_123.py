"""Re-evaluate trained QL/PPO/RL-HH models on seeds 1,2,3 to check PPO is not an outlier.

Uses artifacts from results/20260419_233328_7800x_6h_1/.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RUN_DIR = _ROOT / "results" / "20260419_233328_7800x_6h_1"
Q_TABLE = RUN_DIR / "q_table_master.pkl"
PPO_MODEL = RUN_DIR / "ppo_final_model.zip"
RLHH_CKPT = RUN_DIR / "rlhh_best.pt"

SEEDS = [1, 2, 3]


def _kpi_summary(kpi) -> dict:
    d = kpi.to_dict() if hasattr(kpi, "to_dict") else kpi
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
    }


def _line_breakdown(state, params) -> dict:
    """Batches per line for behavioral diagnosis."""
    counts = {}
    for b in state.completed_batches:
        line = getattr(b, "line_id", None) or getattr(b, "line", None) or "?"
        counts[line] = counts.get(line, 0) + 1
    return counts


def _roaster_breakdown(state) -> dict:
    counts = {}
    for b in state.completed_batches:
        r = getattr(b, "roaster", None) or getattr(b, "roaster_id", None) or "?"
        counts[r] = counts.get(r, 0) + 1
    return counts


def eval_ql(params, engine, ups):
    from q_learning.q_strategy import QStrategy, load_q_table
    q_table = load_q_table(str(Q_TABLE))
    strategy = QStrategy(params, q_table=q_table)
    kpi, state = engine.run(strategy, ups)
    return kpi, state


def eval_ppo(params, engine, ups):
    from PPOmask.Engine.data_loader import load_data
    from PPOmask.Engine.ppo_strategy import PPOStrategy
    from sb3_contrib import MaskablePPO
    data = load_data()
    model = MaskablePPO.load(str(PPO_MODEL))
    strategy = PPOStrategy(data, model, deterministic=True)
    kpi, state = engine.run(strategy, ups)
    return kpi, state


def eval_rlhh(params, engine, ups):
    from rl_hh.rl_hh_strategy import RLHHStrategy
    from rl_hh.meta_agent import DuelingDDQNAgent
    from PPOmask.Engine.data_loader import load_data
    data = load_data()
    agent = DuelingDDQNAgent()
    agent.load_checkpoint(str(RLHH_CKPT))
    agent.epsilon = 0.0
    strategy = RLHHStrategy(agent, data, training=False)
    kpi, state = engine.run(strategy, ups)
    return kpi, state


def main() -> None:
    from env.data_bridge import get_sim_params
    from env.simulation_engine import SimulationEngine
    from env.ups_generator import generate_ups_events

    params = get_sim_params()

    methods = [
        ("Q-Learning", eval_ql),
        ("MaskedPPO ", eval_ppo),
        ("RL-HH     ", eval_rlhh),
    ]

    print(f"{'Seed':>4} | {'Method':<10} | {'NetProfit':>11} | {'Revenue':>9} | "
          f"{'PSC/NDG/BUS':>11} | {'Tard':>7} | {'Setup':>7} | {'Idle':>8} | Lines | Roasters")
    print("-" * 130)

    results = {}
    for seed in SEEDS:
        ups = generate_ups_events(
            params.get("ups_lambda", 0), params.get("ups_mu", 0),
            seed=seed, shift_length=int(params["SL"]),
            roasters=list(params["roasters"]),
        )
        for name, fn in methods:
            engine = SimulationEngine(params)
            try:
                kpi, state = fn(params, engine, list(ups))
                s = _kpi_summary(kpi)
                lines = _line_breakdown(state, params)
                roasters = _roaster_breakdown(state)
                print(
                    f"{seed:>4} | {name} | ${s['net_profit']:>10,.0f} | ${s['revenue']:>8,.0f} | "
                    f"{s['psc']:>3}/{s['ndg']}/{s['busta']:<3} | ${s['tard_cost']:>6,.0f} | "
                    f"${s['setup_cost']:>6,.0f} | ${s['idle_cost']:>7,.0f} | "
                    f"{dict(sorted(lines.items()))} | {dict(sorted(roasters.items()))}"
                )
                results.setdefault(name.strip(), []).append((seed, s, lines, roasters))
            except Exception as exc:
                print(f"{seed:>4} | {name} | ERROR: {exc}")
                import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    print("AVERAGES across seeds 1,2,3")
    print("=" * 60)
    for method, runs in results.items():
        n = len(runs)
        if n == 0:
            continue
        avg_net = sum(r[1]["net_profit"] for r in runs) / n
        avg_rev = sum(r[1]["revenue"] for r in runs) / n
        avg_tard = sum(r[1]["tard_cost"] for r in runs) / n
        avg_idle = sum(r[1]["idle_cost"] for r in runs) / n
        line_totals = {}
        for _, _, lines, _ in runs:
            for k, v in lines.items():
                line_totals[k] = line_totals.get(k, 0) + v
        print(f"  {method:<10} avg_net=${avg_net:>10,.0f}  avg_rev=${avg_rev:>9,.0f}  "
              f"avg_tard=${avg_tard:>7,.0f}  avg_idle=${avg_idle:>8,.0f}  "
              f"lines_sum={line_totals}")


if __name__ == "__main__":
    main()
