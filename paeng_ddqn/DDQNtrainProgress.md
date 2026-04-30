# Paeng DDQN Training Progress Log

**Environment**: Nestlé Trị An roasting plant reactive scheduling (480-min shift, 5 roasters × 2 lines, mixed PSC/NDG/BUSTA, UPS λ=5 μ=20)
**Algorithm**: Modified DDQN with parameter sharing (Paeng et al. 2021, IEEE Access) — see `PORT_NOTES.md` for the port history
**Hardware**: Intel i3-9100F (4 cores), 16 GB RAM, no GPU

**Stopping condition**: 100-seed mean profit ∈ **[$340,000, $370,000]** OR 50 cycles reached.
**Per-cycle budget**: 0.5–4 hours wall-clock training each.
**Reference baselines** (100 seeds, λ=5, μ=20):
- CP-SAT no-UPS ceiling: $452,400
- RL-HH (Phase 6, Dueling DDQN + tools): **$375,084** ± $17,903
- Dispatching heuristic: $320,140 ± $13,189
- Q-learning (1.43M ep nonUPS): $237,376 ± $70,378
- **Paeng cycle-0 (this baseline below): -$1,582,570 ± $9,086**

The target band [$340k, $370k] sits between dispatching and RL-HH — a reasonable spot for Phase 5 to land if the Q-collapse can be fixed without inheriting RL-HH's tool-selection action space. Hitting it would empirically demonstrate Paeng's standard DDQN as a competitive Phase 5 anchor without needing the dueling step (which is Phase 6's contribution).

---

## Background — what's broken in cycle 0

### Symptom
After 4 hours / 4,358 episodes of training the Paeng DDQN policy collapses
to **WAIT-only** at greedy evaluation. 100-seed mean = -$1,582,570 (best
checkpoint, ep 80) and -$1,462,936 (final checkpoint, ep 4,357).

### Schedule analysis (paeng_final.pt on seed 42)
- **PSC: 19** (RL-HH does ~108)
- **NDG: 0, BUSTA: 0** — MTO never started → $1M tardiness penalty
- **First batch starts at t=248** — first half of 8h shift entirely idle
- **RC trajectory**: crashes through zero around t=80, never recovers ($118.5k stockout)
- **Idle min: 2,033 / 2,400** (85% idle)
- **GC silos**: NDG and BUSTA never depleted (because never used); PSC silo drains slowly

### Root cause (3 compounding factors — see PORT_NOTES.md §10 diagnosis)

1. **Reward signal magnitude**. Per-decision reward is raw `kpi.net_profit() − prev_profit`. PSC revenue +$4,000 per completion, MTO +$7,000, tardiness $1,000/min, stockout $1,500/event — span 4+ orders of magnitude with γ=0.99. Q-values blow up; Huber loss saturates; gradient scale mismatched with Paeng's `lr=0.0025` (which his repo paired with a normalized tardiness reward via `rnorm`).

2. **Replay buffer dominance after fill**. Buffer hits 100k cap by ep ~95 (each episode = ~1,300 decisions). After that, uniform sampling never displaces early WAIT-heavy transitions, so target-Q regression stays anchored to the bad-policy distribution.

3. **WAIT shortcut**. WAIT is the only always-feasible action. When productive Q-values are noisy/underestimated (early training), masked argmax defaults to WAIT systematically. WAIT episodes accumulate `reward ≈ 0` transitions (no profit changes during waiting), reinforcing `Q(WAIT) ≈ 0` while productive Q-values remain noisy with negative pulls from upfront costs (consume time, setup time) before delayed revenue is credited.

### Action distribution at cycle 0 greedy (paeng_final.pt seed 42)
```
0 (PSC):    19 selections   → some recovery late-shift
1 (NDG):     0   ← never picked
2 (BUSTA):   0   ← never picked
3 (WAIT): ~1100 selections   ← dominant
4 (RST L1_PSC):   0
5 (RST L1_NDG):   0
6 (RST L1_BUSTA): 0
7 (RST L2_PSC):   12
```

The agent can't even consistently pick PSC, let alone the time-pressured MTO actions or the line-balanced restocks.

---

## Cycle 0 — Baseline (no fix, just the v4 build training output)

| Metric | Value |
|---|---|
| Training | 4,358 episodes / 4 h |
| Best training profit | $57,300 (ep 80; before Q-collapse) |
| Final ε | 0.079 (decay over 6,000-ep budget; never reached eps_end=0.05) |
| 100-seed mean (best.pt) | **-$1,582,570** ± $9,086 |
| 100-seed mean (final.pt) | **-$1,462,936** ± $23,646 |
| 100-seed PSC count | best: 0, final: 19 (vs RL-HH ~108) |
| 100-seed tard cost | $1,000,000 (MTO never starts) |
| 100-seed stockout cost | ~$118,500 |
| Greedy action dist | WAIT ≈ 90%, PSC 5%, RST_L2_PSC 1%, MTO 0%, all others ≈ 0% |

**Verdict**: cycle 0 is far below dispatching ($320k). Need fundamental fixes before considering hyperparameter tuning.

---
