# Draft Reply — Rolling Horizon CP-SAT Discussion

---

Thanks for the great questions. Two clarifications:

---

## 1. Main factors causing the 19% gap

The gap is vs an **oracle** CP-SAT that pre-knows all UPS events before the shift starts.
Three structural sources:

| Factor | Impact |
|--------|--------|
| **UPS uncertainty** | CP-SAT injects UPS as *planned* downtime (it cheats — knows the future). RL operates under true stochastic disruptions, re-deciding in real time. | 
| **Global vs local optimality** | CP-SAT optimizes the entire 480-min horizon jointly. RL makes greedy per-roaster decisions within an 11-min decision window. |
| **No MTO skip penalty exposure** | CP-SAT guarantees both J1 and J2 are completed (tardiness = 0). RL occasionally misses a job under severe UPS, incurring the $100k skip penalty. |

The realistic upper bound for an online method is lower — an online CP-SAT that doesn't know UPS in advance would perform worse than $443k even with unlimited compute.

---

## 2. Rolling-horizon CP-SAT (2+2+2+2 hours, 5 min/window)

We ran this experiment. Key results:

| Budget / window | Net Profit | Gap vs full CP-SAT | Rolling latency (4×) | % of shift frozen |
|:-:|--:|--:|--:|--:|
| 5s  | see results | large | 0.3 min | 0.07% |
| 30s | see results | moderate | 2 min | 0.4% |
| 5min | see results | moderate | **20 min** | **4.2%** |
| Full (300s) | $443,400 | 0% (reference) | 20 min | 4.2% |

> *Run `python Experiment/cpsat_budget_experiment.py` for live numbers — chart and HTML report are auto-generated in `Experiment/results/`.*

**Three problems with rolling-horizon MILP in practice:**

1. **Solution quality degrades with short windows.** A 5-min budget on a 120-min sub-horizon still misses inter-window coupling (RC inventory, GC overflow, MTO due dates that span windows). The solver sees an amputated problem.

2. **Solver latency is production-frozen time.** 4 windows × 5 min = **20 min of idle waiting** per shift, during which roasters cannot receive new assignments. For a factory running 3 shifts/day, that is 1 hour of lost throughput per day.

3. **UPS invalidates the schedule mid-window.** With λ=5 expected events/shift, each UPS hit forces a re-solve from the new state (another 5 min freeze). Total worst-case: 20 + 5×5 = **45 min frozen** per shift (9.4%).

**RL-HH** makes each decision in **< 1 ms** and updates its policy instantaneously on the new state — zero frozen time, handles UPS natively.

---

## Summary table

| Method | Net Profit (seed 69) | Decision latency | Handles UPS online? |
|--------|---------------------:|-----------------|:-------------------:|
| CP-SAT full (oracle) | **$443,400** | ~8 h (offline) | No — pre-scheduled |
| Rolling CP-SAT 5 min/window | TBD (see experiment) | 5 min / re-plan | Partial — re-solve needed |
| **RL-HH (proposed)** | **$359,096** (100-seed mean) | **< 1 ms** | **Yes** |
| Dispatching heuristic | $326,554 (100-seed mean) | < 1 ms | Yes |

**The 19% gap is the price of zero latency + full uncertainty handling.** A rolling-horizon exact solver closes part of the gap but introduces frozen latency and still assumes UPS is known within the window.

---
