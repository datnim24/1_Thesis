"""Evaluate C25's best+final PPO models on seeds 1,2,3 and compare to broken baseline."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

C25_RUN = _ROOT / "PPOmask" / "outputs" / "20260420_193326_c25_busta_fix"
C25_FINAL = C25_RUN / "checkpoints" / "final_model.zip"
C25_BEST = C25_RUN / "checkpoints" / "best_training_profit_model.zip"

SEEDS = [1, 2, 3]


def eval_ppo_model(model_path, params, engine, ups):
    from PPOmask.Engine.data_loader import load_data
    from PPOmask.Engine.ppo_strategy import PPOStrategy
    from sb3_contrib import MaskablePPO
    data = load_data()
    model = MaskablePPO.load(str(model_path))
    strategy = PPOStrategy(data, model, deterministic=True)
    kpi, state = engine.run(strategy, ups)
    return kpi, state


def _s(kpi):
    d = kpi.to_dict() if hasattr(kpi, "to_dict") else kpi
    tm = d.get("tardiness_min", {})
    return {
        "net": d.get("net_profit", 0),
        "rev": d.get("total_revenue", 0),
        "psc": d.get("psc_count", 0),
        "ndg": d.get("ndg_count", 0),
        "busta": d.get("busta_count", 0),
        "tard": d.get("tard_cost", 0),
        "setup": d.get("setup_cost", 0),
        "idle": d.get("idle_cost", 0),
        "restocks": d.get("restock_count", 0),
        "j1_late": tm.get("J1", 0),
        "j2_late": tm.get("J2", 0),
    }


def main():
    from env.data_bridge import get_sim_params
    from env.simulation_engine import SimulationEngine
    from env.ups_generator import generate_ups_events

    params = get_sim_params()

    print(f"\n{'='*110}")
    print(f"C25 PPO Eval | seeds=1,2,3 | model=final_model.zip ({C25_FINAL.name})")
    print(f"{'='*110}")
    print(f"{'Seed':>4} | {'NetProfit':>11} | {'Rev':>9} | {'PSC/NDG/BUS':>11} | "
          f"{'Tard':>7} | {'Setup':>7} | {'Idle':>8} | {'J2min':>5} | Restocks")
    print("-" * 110)

    all_runs = {}
    for label, mdl in [("FINAL", C25_FINAL), ("BEST", C25_BEST)]:
        if not mdl.exists():
            print(f"Missing {label}: {mdl}")
            continue
        print(f"\n--- C25 {label} ({mdl.name}) ---")
        runs = []
        for seed in SEEDS:
            ups = generate_ups_events(
                params.get("ups_lambda", 0), params.get("ups_mu", 0),
                seed=seed, shift_length=int(params["SL"]),
                roasters=list(params["roasters"]),
            )
            engine = SimulationEngine(params)
            kpi, state = eval_ppo_model(mdl, params, engine, list(ups))
            s = _s(kpi)
            runs.append(s)
            print(
                f"{seed:>4} | ${s['net']:>10,.0f} | ${s['rev']:>8,.0f} | "
                f"{s['psc']:>3}/{s['ndg']}/{s['busta']:<3} | ${s['tard']:>6,.0f} | "
                f"${s['setup']:>6,.0f} | ${s['idle']:>7,.0f} | {s['j2_late']:>5.0f} | {s['restocks']}"
            )
        n = len(runs)
        avg_net = sum(r["net"] for r in runs) / n
        avg_tard = sum(r["tard"] for r in runs) / n
        avg_busta = sum(r["busta"] for r in runs) / n
        avg_psc = sum(r["psc"] for r in runs) / n
        print(f"\nC25 {label} AVG: net=${avg_net:,.0f} tard=${avg_tard:,.0f} "
              f"BUSTA={avg_busta:.1f}/5 PSC={avg_psc:.1f}")
        all_runs[label] = (avg_net, avg_tard, avg_busta, avg_psc)

    print(f"\n{'='*110}")
    print("COMPARISON vs old broken model")
    print(f"{'='*110}")
    print(f"{'Model':<15} | {'Avg Net':>12} | {'Avg Tard':>11} | {'BUSTA avg':>10} | {'PSC avg':>9}")
    print("-" * 70)
    print(f"{'OLD (broken)':<15} | {'$47,100':>12} | {'$341,000':>11} | {'1.7/5':>10} | {'102.7':>9}")
    for label, (net, tard, busta, psc) in all_runs.items():
        print(f"{'C25 ' + label:<15} | {'$' + format(int(net), ','):>12} | "
              f"{'$' + format(int(tard), ','):>11} | {busta:.1f}/5{'':>5} | {psc:.1f}")


if __name__ == "__main__":
    main()
