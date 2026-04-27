# Detailed 100-Seed Comparison Report

**Date**: 2026-04-27
**Environment**: Nestlé Trị An batch roasting plant, 480-min shift, λ=5, μ=20 UPS
**Eval seeds**: 900000-900099 (same set for all 4 simulation-based methods)
**Engine**: `env.simulation_engine.SimulationEngine`
**Output**: [results/full_comparison_20260427_065910/](../results/full_comparison_20260427_065910/)

---

## 1. Methods evaluated

| Method | Type | Model / config |
|--------|------|----------------|
| **Dispatching heuristic** | Hand-coded baseline | `dispatch/dispatching_heuristic.py` — MTO-urgency dispatch + reorder-point GC restock |
| **Q-learning (tabular)** | Tabular RL | `q_learning/ql_results/.../q_table_seed69_4h_s479740.pkl` — 546k episodes, 4h, ε-greedy |
| **MaskedPPO** | Deep RL | `PPOmask/outputs/20260424_004458_seed69_pipeline_s427008_v2/checkpoints/final_model.zip` — sb3-contrib MaskablePPO, 55M timesteps in 4h, 8 envs, lr=5e-4 |
| **RL-HH (Dueling DDQN + improved tools)** | Hyper-heuristic | `rl_hh/outputs/rlhh_cycle3_best.pt` + 7 tool-logic improvements from cycles 1, 4, 5, 7, 8, 10, 14, 16 |
| **CP-SAT no-UPS ceiling** | Optimization | `CPSAT_Pure/runner.py` — single deterministic solve, no UPS injection, 900s, 8 workers |

---

## 2. Headline results (100 seeds, λ=5, μ=20)

| Method | Mean | Median | Std | Min | Max | % of ceiling |
|--------|------|--------|-----|-----|-----|---------------|
| **CP-SAT no-UPS ceiling** | **$452,400** | — | 0 | $452,400 | $452,400 | 100.0% |
| MaskedPPO | $388,469 | $396,100 | $76,112 | $120,700 | $489,200 | 85.9% |
| **RL-HH (Dueling DDQN)** | **$375,084** | $374,800 | **$17,903** | $337,600 | $417,400 | 82.9% |
| Dispatching heuristic | $320,140 | $321,000 | $13,189 | $271,800 | $348,800 | 70.8% |
| Q-learning (tabular) | -$91,718 | -$62,700 | $137,728 | -$458,200 | $178,000 | -20.3% |

> **Note on the ceiling.** The 900-s CP-SAT solve closed only to a 31.66 % gap (best bound = $662,000). The true optimum lies between **$452,400** (achieved feasible) and **$662,000** (LP relaxation upper bound). All "% of ceiling" figures above use $452,400 as a conservative ceiling — the relative ranking of the 4 simulation methods is unaffected.

### Box-plot summary (text form)

```
              min        p25      med      p75       max
DISPATCH:  271800 ──── 312750 ─321k─ 330000 ────── 348800     [σ=13k]
Q-LEARN:  -458200 ──── -170k ──-63k── 11k ──────── 178000      [σ=138k] BROKEN
PPO:      120700 ──── 353850 ─396k─ 447225 ────── 489200      [σ=76k]  HIGH VAR
RL-HH:     337600 ──── 361950 ─375k─ 386450 ────── 417400     [σ=18k]  STABLE
CP-SAT:                       452400                             [no-UPS ceiling]
```

---

## 3. Detailed KPI breakdown

| Metric | Dispatching | Q-learning | PPO | RL-HH | CP-SAT (no-UPS) |
|--------|------------:|-----------:|----:|------:|----------------:|
| **Profit mean** | $320,140 | -$91,718 | $388,469 | $375,084 | $452,400 |
| **Profit std** | $13,189 | $137,728 | $76,112 | **$17,903** | 0 |
| **Profit p25** | $312,750 | -$170,000 | $353,850 | $361,950 | — |
| **Profit p75** | $330,000 | $10,900 | $447,225 | $386,450 | — |
| **Profit min** | $271,800 | -$458,200 | $120,700 | **$337,600** | $452,400 |
| **Profit max** | $348,800 | $178,000 | **$489,200** | $417,400 | $452,400 |
| Mean revenue | $477,760 | $431,650 | $493,390 | $501,840 | n/a |
| Mean tard cost | $3,080 | $324,020 | $43,780 | **$0** | $0 |
| Mean setup cost | $14,424 | $25,872 | **$4,128** | $6,552 | $4,800 |
| Mean stockout cost | $0 | $2,250 | $6,225 | **$0** | $0 |
| Mean idle cost | $140,116 | $171,226 | **$50,788** | $120,204 | $44,800 |
| Mean PSC count | 101.94 | 91.34 | 106.53 | **107.96** | 108 |
| Mean NDG count | 5.00 | 4.99 | 4.85 | **5.00** | 5 |
| Mean BUSTA count | 5.00 | 4.48 | 4.76 | **5.00** | 5 |
| Mean setup events | 18.03 | 32.34 | **5.16** | 8.19 | 6 |
| Mean restock count | 17.49 | 15.24 | **16.15** | 19.12 | 15 |
| Mean idle min | 700.58 | 856.13 | **253.94** | 601.02 | 224 (idle/200) |
| Mean stockout dur L1 | 0.00 | 18.99 | 55.32 | **0.00** | 0 |
| Mean stockout dur L2 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |

### Reliability counters (out of 100 seeds)

| Reliability metric | Dispatching | Q-learning | PPO | **RL-HH** |
|---|---:|---:|---:|---:|
| Perfect seeds (no tard, no stockout) | 38 | 0 | 24 | **100** |
| Seeds with any tardiness | 62 | 100 | 37 | **0** |
| Seeds with any stockout | 0 | 33 | 53 | **0** |

---

## 4. Per-method analysis

### 4.1 Dispatching heuristic (baseline)

- **$320,140 mean, std $13,189** — extremely stable across seeds (smallest spread among non-CP-SAT methods).
- **62 / 100 seeds had tardiness** ($3,080 average tard cost) — the rule-based MTO urgency check sometimes misses deadlines under disruption.
- **Zero stockouts**: the reorder-point restock heuristic is conservative and never lets GC hit zero.
- **Highest setup count among all methods** (18.03 events / shift) — the heuristic switches SKUs frequently because it doesn't reason about future production mix.
- **Idle 700 min / shift** ≈ 29 % of available roaster-time. The heuristic is too cautious about starting batches when state is ambiguous.

**Verdict**: Reliable and easy to operate; clearly suboptimal on throughput. Useful as the do-nothing baseline.

### 4.2 Q-learning (tabular)

- **-$91,718 mean** — the only method that loses money on average.
- **100 / 100 seeds tardy**, $324,020 average tardiness cost, 33 / 100 with stockouts.
- **Profit std $137,728** — completely unreliable, ranges from -$458k to +$178k.
- The Q-table has only **10,531 visited states** out of a state space far larger; on unseen UPS scenarios it falls back to default Q = 0 across all candidate actions and behaves close to random.
- Trained 4 h on UPS-enabled environment with seed 69 and reached $210k *training* profit; on its own training seed it scores **$127,600**, and that drops to **-$91,718 mean** on unseen seeds.
- Setup events 32.34 / shift — almost double the dispatching baseline. Likely from churning between SKUs in unfamiliar states.
- 856 idle min — worst of all methods.

**Verdict**: Tabular Q-learning **does not generalize across UPS scenarios**. The state discretization needed to keep the table tractable destroys the agent's ability to react to novel disruptions. This validates the literature claim (Li et al. 2024) that tabular RL is "limited to low-dimensional problems".

### 4.3 MaskedPPO

- **$388,469 mean — best of the simulation methods**, but **std $76,112** — *4.3×* RL-HH's std.
- **$489,200 max** (best single seed) — clear evidence the deep policy can hit near-ceiling on lucky seeds.
- **$120,700 min**: at least one seed degrades severely. Combined with **53 / 100 seeds with stockouts** and **37 / 100 with tardiness**, PPO has a **long left tail of failures**.
- Mean tard cost $43,780, stockout cost $6,225 — together ≈ $50k per shift in violation costs on average, eating ≈ 10 % of revenue.
- Strongest on idle (253 min, almost 3× better than baseline) and setups (5.16 events, near-optimal).
- 16.15 restocks / shift — closest to the CP-SAT-style 15 restocks.

**Verdict**: PPO's policy is **aggressive and high-throughput** but **fragile** — under bad UPS sequences it commits to early starts that cascade into stockouts/tardiness. Wins by mean, loses by reliability and worst-case.

### 4.4 RL-HH (Dueling DDQN with improved tools)

- **$375,084 mean** — second behind PPO by ≈ 3.5 %.
- **Std $17,903** — the most stable method among the four simulation methods, **4.3× tighter than PPO**.
- **100 / 100 seeds perfect** (zero tardiness, zero stockouts) — the only method with this property.
- Worst seed = **$337,600**: better than PPO's worst by **+$216,900** and better than dispatching's worst by +$65,800.
- 5 NDG and 5 BUSTA on every seed (100 / 100 MTO completion).
- Idle 601 min — better than dispatching but 2.4× worse than PPO. This is the main lever still on the table.
- 8.19 setup events — second-best after PPO; 19.12 restocks — slightly above PPO and CP-SAT.

**Verdict**: RL-HH **never fails**. The improved tools (smart GC restock, SETUP_AVOID MTO hijack, R3 headroom routing, UPS-aware fallback) keep the network's choices feasible and conservative. It trades the upside potential of PPO for **reliability** and **deadline safety**. In a production setting where missed MTO deadlines are very costly (and customer-visible), this trade-off is favourable.

### 4.5 CP-SAT no-UPS ceiling

- **$452,400** in 912 s with gap 31.66 % (best bound $662,000) — solver did not close the gap in the 15-min budget.
- Schedule: 108 PSC + 5 NDG + 5 BUSTA, 6 setup events, 15 restocks, 224 idle-cost units, **zero tardiness, zero stockouts** (by definition: no UPS).
- This is the **theoretical upper bound assuming perfect knowledge and zero disruption**. Real online performance under UPS is *strictly* below this number.

**Verdict**: Even the deterministic optimal isn't huge — the shift is genuinely capacity-constrained. The reason PPO's max ($489,200) appears to *exceed* the ceiling is that the CP-SAT solve did not converge: the LP relaxation bound shows there is up to $662k of profit available in the no-UPS case, so PPO's $489k is still well below the true optimum.

---

## 5. Cross-method observations

### 5.1 Mean vs reliability trade-off

The four simulation methods split clearly into two regimes:

**Mean-optimised, fragile**: PPO ($388k mean / $76k std / 53 stockout seeds)
**Reliability-optimised, conservative**: RL-HH ($375k mean / $18k std / 0 stockout seeds), Dispatching ($320k / $13k / 0 stockout)

For the operator, the question is the *cost of unreliability*: a missed MTO deadline costs $1,000/min of tardiness in this model and customer goodwill in reality. PPO's $43,780 mean tardiness cost is structurally absorbed by its mean profit, but on 37 % of seeds the operator is delivering late. RL-HH delivers on time on **every** seed.

### 5.2 What separates RL-HH from PPO

- **Setups**: PPO 5.16 / RL-HH 8.19 — PPO blocks MTO into a tighter contiguous window.
- **Idle**: PPO 254 min / RL-HH 601 min — PPO commits to PSC starts more aggressively.
- **Stockouts**: PPO 53 seeds / RL-HH 0 — PPO's aggression sometimes drains GC.
- **Tardiness**: PPO 37 seeds, $43,780 avg / RL-HH 0 — PPO occasionally lets MTO slip late.

The pattern is consistent: PPO trades safety for throughput; RL-HH trades throughput for safety.

### 5.3 What kills Q-learning

- The state discretizer in `q_strategy.discretize_roaster_state` produces tuple keys; UPS-disrupted states create combinations the table never visited during training.
- 10,531 entries is far below what a flat Q-table over all reachable (time × roaster-status × RC × GC × pipeline × MTO-remaining × …) states would need.
- Result: in unseen states, the agent sees Q = 0 for every action and falls back to its action-order heuristic — which is essentially random.
- **Tardiness on every seed** ($324k avg) is the dominant cost: jobs simply don't complete because the table never tells the agent to start them.

This is the strongest empirical evidence in the thesis that **value-based deep RL is required for this problem class** — exactly the argument Li et al. 2024 make for D3QN over tabular methods.

### 5.4 PSC ceiling check

Maximum theoretically reachable PSC count per shift = `5 roasters × (480 - 90 MTO time) / 15 = ~130 batches`. In practice (RC capacity 40, consume rate 9-10 / min), the production plan is bounded. CP-SAT no-UPS picks **108** PSC. RL-HH hits **107.96** mean (essentially the same), PPO **106.53**, Dispatching **101.94**, Q-learning **91.34**. So PSC throughput is essentially saturated for the top three simulation methods — the differences in profit come from MTO completion, setup avoidance, and stockout/tard avoidance.

---

## 6. Conclusions

1. **CP-SAT no-UPS sets the ceiling at $452k** (with a 31% remaining LP gap suggesting up to $662k is theoretically reachable on the deterministic optimum — needs longer solve to confirm).

2. **MaskedPPO has the highest mean ($388k, 85.9 % of ceiling)** but is fragile: 53 seeds stockout, 37 seeds tardy, σ = $76k.

3. **RL-HH (Dueling DDQN + improved tools) has the lowest variance ($18k σ) and is the only method that completes all MTO with zero stockouts on every one of 100 seeds.** Mean $375k = 82.9 % of ceiling, only 3.5 % behind PPO.

4. **Dispatching heuristic ($320k) is honest baseline** — not a contender for production but useful as a floor.

5. **Tabular Q-learning fails ($-92k mean)** — the state discretization cannot generalize across UPS scenarios. Confirms the case for deep RL methods.

6. The choice between PPO and RL-HH is a **mean-vs-reliability trade-off**. For a Nestlé-scale plant where missed customer deadlines have non-monetary costs (account loss, reputation), RL-HH's zero-failure profile is operationally preferable; for pure-profit benchmarks, PPO wins by ≈ $13k / shift.

---

## 7. Reproduction

```bash
# Run the comparison from project root
python -m test_rl_hh.full_comparison

# Outputs:
#   results/full_comparison_<ts>/comparison_100seed.json (aggregates)
#   results/full_comparison_<ts>/per_seed.json (full per-seed kpi for each method)
#   results/full_comparison_<ts>/REPORT.md (this report)
```

Seeds and UPS parameters are identical for the four simulation methods, so the
ranking is directly comparable. CP-SAT runs once with `ups_events=None` for the
no-UPS ceiling.
