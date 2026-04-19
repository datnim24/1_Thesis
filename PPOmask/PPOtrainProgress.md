# PPO Training Progress Log

**Environment**: Nestle Tri An roasting plant reactive scheduling
**Algorithm**: MaskablePPO (sb3-contrib)
**Hardware**: Intel i3-9100F (4 cores), 16GB RAM, no GPU
**Objective**: Grade A (eval profit >= $280k, violation rate < 5%)
**Grade scale**: A >= $280k/<5% viol | B >= $240k/<10% | C >= $200k/<20% | D >= $150k/<40% | F = else

**Current Best**: C20 — eval=$470,600, GRADE A, 0 violations, 120 PSC, 17 restocks

---

## Cycle 1 — Baseline (2026-04-10, 10:30-14:22)

**Wall-clock**: 3h50m (13,800s) | **Episodes**: 29,847 | **Timesteps**: 19.9M | **FPS**: ~1,440

### Configuration
```
--seed 42 --n-envs 4 --lr 3e-4 --n-steps 2048 --batch-size 256 --n-epochs 10
--gamma 0.995 --ent-coef 0.01 --clip-range 0.2 --rc-maintenance-bonus 1.0
--net-arch 256,256 --time 13800
```

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $54,900 |
| Eval det restock | 0 |
| Eval det PSC | 40 |
| Eval det violation | rc_negative_L2 |
| Eval violation rate | 100% |
| Training best profit | $82,729 |
| Training final avg (last 1000) | $29,316 |
| Training reward slope | +0.15/ep |
| Training violation rate (last 100) | 100% |

### Timeline & Symptoms
- **0-30min**: `ep_rew_mean` rose from -$66k to -$40k. `entropy_loss=-0.275`. Policy exploring.
- **30min-1h**: Reward climbing. `entropy_loss` dropping toward -0.1. `approx_kl` falling below 1e-6.
- **1h-2h**: `entropy_loss=-0.024` (collapsed). `approx_kl=1.9e-6 → ~1e-10`. `clip_fraction=0`. Policy frozen.
- **2h-3h50m**: No change in policy. Value function converged (`explained_variance=0.952`). Reward plateaued at ~$29k avg. All 5 best episodes ($185k, $136k, $129k, $128k, $126k) were from ep 6368-6977, showing late stochastic exploration found good policies but argmax locked early.

### Diagnosis
- `ent_coef=0.01` too low — entropy collapsed before the agent explored restock actions
- `rc_maintenance_bonus=1.0` is noise ($1/slot) vs PSC revenue ($1,500/batch)
- Deterministic strategy: start 40 PSC batches, WAIT everything else, violate when L2 RC hits 0
- The policy literally stopped changing after ~1h (approx_kl=0, clip_fraction=0)

### Action Taken
- Increase `ent_coef` to 0.05 (5x)
- Increase `rc_maintenance_bonus` to 5.0 (5x)
- Increase `lr` to 5e-4

**Run dir**: `PPOmask/outputs/20260410_103041_PPO_1/`

---

## Cycle 2 — Higher LR + RC Bonus (2026-04-10, 14:24-16:14, killed early)

**Wall-clock**: ~1h50m (killed at ~5,400s of 17,400s budget) | **Episodes**: ~11,713 | **FPS**: ~1,440

### Configuration Changes from C1
| Parameter | C1 | C2 | Why |
|-----------|----|----|-----|
| lr | 3e-4 | **5e-4** | Faster learning |
| ent_coef | 0.01 | **0.05** | Prevent entropy collapse |
| rc_maintenance_bonus | 1.0 | **5.0** | Meaningful restocking reward |
| gamma | 0.995 | 0.995 | Same |

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $10,900 |
| Eval det restock | 0 |
| Training best profit | **$291,200** |
| Training ep_rew_mean at kill | $85,900 |
| Training violation rate | 98% |

### Timeline & Symptoms
- **0-30min**: Learning fast. `ep_rew_mean` rose from -$66k to +$80k. Entropy healthy at -0.282.
- **30min-1h**: Best episode $291,200 found (stochastically). `ep_rew_mean=$72,400`. `violation_rate=98%` — first time below 100%! `approx_kl=4.7e-10` — already collapsing.
- **1h-1h50m**: `approx_kl=3.3e-10`. Flat reward. TensorBoard showed clear plateau. **Killed manually.**

### Diagnosis
- Stochastic exploration found **$291,200** — proved Grade A is achievable in this environment
- But eval collapsed to $10,900 — deterministic policy converged to a bad strategy
- `approx_kl` fell to ~1e-10 by 1h despite higher entropy start
- The higher LR helped initial learning but didn't prevent collapse

### Action Taken
- Created `AdaptiveEntropyCallback` — monitors KL, boosts ent_coef when stalled
- Added `--lr-schedule linear` to train_maskedppo.py
- Plan: test adaptive entropy in C3

**Run dir**: `PPOmask/outputs/20260410_142416_PPO_2/`

---

## Cycle 3 — Adaptive Entropy Callback (2026-04-10, 15:59-16:30)

**Wall-clock**: 30min (1,800s) | **Episodes**: 6,785 | **Timesteps**: 2.8M | **FPS**: ~1,530

### Code Changes
- **New**: `AdaptiveEntropyCallback` in `callbacks.py` — monitors `approx_kl`, boosts `ent_coef` by 1.5x when stalled for 5 consecutive rollouts. Caps at `max_ent_coef=0.2`.
- **New**: `--lr-schedule linear` flag — LR decays linearly to 5% of base over training.
- **New**: `--resume-from` flag (prepared for C4).

### Configuration Changes from C2
| Parameter | C2 | C3 | Why |
|-----------|----|----|-----|
| lr | 5e-4 | **3e-4** (mistake!) | Intended to be cautious, was actually harmful |
| lr_schedule | constant | **linear** | Decay over training |
| ent_coef | 0.05 | 0.05 | Same |
| rc_maintenance_bonus | 5.0 | 5.0 | Same |
| time | 17,400s | **1,800s** | Quick 30min test |

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | **-$27,500** |
| Eval det restock | 0 |
| Training best profit | $21,900 |
| Training final avg | -$54,373 |
| Training violation rate | 100% |

### Timeline & Symptoms
- **0-15min**: `explained_variance=1.79e-7` (value function not learning at all!). `entropy_loss=-0.294` (healthy). `approx_kl=7.7e-7`.
- **15-30min**: Still `explained_variance~0`. `ep_rew_mean=-$62k`. No improvement in rewards.

### Diagnosis
- **Regression** — worst eval yet. The value function never learned (explained_variance stuck at 0).
- Root cause: `lr=3e-4` (reduced from C2's 5e-4) was too low for the value function to converge. Without a working value function, advantage estimates are random noise — the policy gets no useful gradient signal.
- Adaptive entropy callback worked correctly (entropy stayed healthy) but couldn't fix a broken value network.
- Lesson: **lr=5e-4 is necessary, not optional**

### Action Taken
- Return `lr` to 5e-4 for all future cycles
- Try resuming from C2's best model ($291k checkpoint) instead of training from scratch

**Run dir**: `PPOmask/outputs/20260410_155857_PPO_3/`

---

## Cycle 4 — Resume from C2 Best (2026-04-10, 16:31-17:33)

**Wall-clock**: 1h (3,600s) | **Episodes**: 6,129 | **Timesteps**: 5.4M | **FPS**: ~1,510

### Code Changes
- **New**: `--resume-from` flag implemented in `train_maskedppo.py` — uses `MaskablePPO.load()` to load weights.

### Configuration Changes from C3
| Parameter | C3 | C4 | Why |
|-----------|----|----|-----|
| lr | 3e-4 | **5e-4** | Back to proven LR |
| resume | fresh | **C2 best model** | Start from $291k checkpoint |
| time | 1,800s | **3,600s** | 1h test |

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $60,900 |
| Eval det restock | 0 |
| Eval det PSC | 35 |
| Eval det setup | 1 |
| Eval det violation | rc_negative_L2 |
| Training best profit | $285,200 |
| Training final avg | $79,150 |
| Training violation rate overall | 97.3% |
| Training violation rate last 100 | 99% |

### Timeline & Symptoms
- **0-15min**: `ep_rew_mean=$79k`, `best=$177,700` in 30 seconds! Resume working. But `approx_kl=1.0e-9` — **policy gradient already dead from the start.**
- **15-45min**: `explained_variance` climbed from 0.217 → 0.533 → 0.775. Value function learning but policy frozen.
- **45min-1h**: `violation_rate=0.95` (best at the time). Reward stable at ~$83k avg. No policy improvement.

### Diagnosis
- Resuming from a collapsed policy loads dead gradients — `approx_kl=1e-9` from step 1
- The value function continues learning (0.217 → 0.775) but the policy network outputs are frozen
- This $60,900 deterministic strategy became the locked baseline for C4/C5/C6: 35 PSC, 1 setup, 0 restock, die on L2
- **Lesson: resuming from a collapsed policy is pointless for policy improvement**

### Action Taken
- Increase `rc_maintenance_bonus` to 20.0 to see if stronger signal can break through

**Run dir**: `PPOmask/outputs/20260410_163136_PPO_4/`

---

## Cycle 5 — Stronger RC Bonus (2026-04-10, 17:34-19:35)

**Wall-clock**: 2h (7,200s) | **Episodes**: 11,934 | **Timesteps**: ~10M | **FPS**: ~1,475

### Configuration Changes from C4
| Parameter | C4 | C5 | Why |
|-----------|----|----|-----|
| rc_maintenance_bonus | 5.0 | **20.0** | Each slot where both lines healthy = $40 reward |
| resume | C2 best | **C4 best** | Continue chain |
| time | 3,600s | **7,200s** | 2h run |

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $60,900 | **Identical to C4** |
| Eval det restock | 0 |
| Training best profit | **$308,400** (new record!) |
| Training final avg | $88,000 |
| Training violation rate | 97% |

### Stochastic Eval of Best Model (10 det + 10 stoch)
| Mode | Mean profit | Violations | Restocks |
|------|------------|------------|----------|
| Deterministic | $60,900 | 10/10 | 0 |
| **Stochastic** | **$121,200** | **10/10** | **6-14 per ep** |

Best stochastic episode: $206,500 (86 PSC, 14 restocks, 26 setups)

### Timeline & Symptoms
- **0-30min**: `ep_rew_mean=$78k`. `explained_variance=0.887`. Policy frozen from start.
- **30min-1h**: `best=$292k` found stochastically. `violation_rate=0.97`.
- **1h-2h**: `best=$308k`. Reward stable. No policy change. `approx_kl=0.0`.

### Diagnosis
- Deterministic policy **identical** to C4 despite 4x stronger bonus
- But stochastic eval revealed the key insight: **the policy knows restocking is good** — it samples restock actions randomly (6-14 per episode) and earns $121k mean. The restock logits are just below the PSC/WAIT argmax threshold.
- $308k stochastic best = Grade A. The environment supports it. The agent understands it. The argmax just won't pick it.

### Action Taken
- Increase bonus further to 50.0 for C6
- Begin accepting that resume approach is fundamentally limited

**Run dir**: `PPOmask/outputs/20260410_173401_PPO_5/`

---

## Cycle 6 — Maximum RC Bonus (2026-04-10, 19:37-22:38)

**Wall-clock**: 3h (10,800s) | **Episodes**: 17,738 | **FPS**: ~1,517

### Configuration Changes from C5
| Parameter | C5 | C6 | Why |
|-----------|----|----|-----|
| rc_maintenance_bonus | 20.0 | **50.0** | $100/slot when both lines healthy (~$48k/episode) |
| resume | C4 best | **C5 best** | Continue chain |
| time | 7,200s | **10,800s** | 3h run |

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $60,900 | **STILL identical to C4/C5** |
| Eval det restock | 0 |
| Training best profit | **$312,800** |
| Training violation rate overall | 96.6% |
| Training violation rate last 100 | 97% |

### Timeline & Symptoms
- **0-1h**: `explained_variance=0.945` (best ever). `violation_rate=0.95` (best ever at the time). But `approx_kl=7.3e-12`.
- **1h-2h**: `violation_rate` **increased** from 0.95 to 0.98. Policy regressing. `entropy=-0.154` dropping.
- **2h-3h**: Fully frozen. `approx_kl=0.0`. No improvement.

### Diagnosis
- **Three consecutive cycles (C4/C5/C6) with identical $60,900 eval.** The resumed policy is permanently locked.
- Even at `rc_bonus=50` ($48k/episode potential), the frozen policy gradient can't move the restock logits
- `violation_rate` actually got *worse* from 1h to 2h (0.95→0.98) — value function over-fitted without policy updates

### Action Taken
- **Abandon resume approach entirely** — train from scratch with new seed
- Try smaller batch_size to create noisier gradients that might escape the saddle point

**Run dir**: `PPOmask/outputs/20260410_193659_PPO_6/`

---

## Cycle 7 — Fresh Start, Smaller Batch (2026-04-10, 22:39 → 2026-04-11, 02:40)

**Wall-clock**: 4h (14,400s) | **Episodes**: 19,127 | **Timesteps**: 13.2M | **FPS**: ~953

### Configuration Changes from C6
| Parameter | C6 | C7 | Why |
|-----------|----|----|-----|
| seed | 42 | **99** | Fresh initialization to escape C4-C6 saddle |
| batch_size | 256 | **64** | 4x noisier gradients |
| clip_range | 0.2 | **0.3** | Allow larger policy updates |
| gamma | 0.995 | **0.99** | Shorter horizon for faster credit assignment |
| resume | C5 best | **NO — FRESH START** | Escape frozen policy |
| time | 10,800s | **14,400s** | 4h |

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $73,900 |
| Eval det restock | 0 |
| Eval det PSC | 20 |
| Eval det setup | 2 |
| Eval det violation | **rc_negative_L1** (different from C1-C6!) |
| Training best profit | $317,800 |
| Training final avg | $99,835 |
| Training reward slope | +3.7/ep |

### Timeline & Symptoms
- **0-1min**: `policy_gradient_loss=-0.00092` — **1000x stronger than any C1-C6 gradient!** Batch=64 creates much noisier gradients. `psc_fraction=0.218` (exploring more aggressively).
- **0-30min**: Rapid learning. `ep_rew_mean` went from -$64k to +$56k. `approx_kl=6.8e-9` — higher than C4-C6 at this point but already declining. `entropy=-0.126` — lowest so far.
- **30min-1.5h**: `approx_kl=3.2e-9`. `explained_variance=0.981`. Gradient weakening.
- **1.5h-4h**: `entropy=-0.121`. Policy frozen. `best=$317k` found at ~3h but couldn't consolidate.

### Diagnosis
- Fresh start + batch=64 found a **completely different local optimum**: 20 PSC (not 35-40), 2 setups, die on L1 (not L2)
- The initial gradient signal (`pg_loss=-0.00092`) was 1000x stronger than resumed policies, confirming the saddle point escape
- But batch=64 means 4x more gradient steps per rollout (8192/64×10=1280), which still causes over-optimization → KL collapse by 30min
- `fps=953` — slower because more gradient steps per env step

### Action Taken
- **Reduce `n_epochs` from 10 to 3** — the key structural fix. With 10 epochs, each rollout does 1280 gradient steps, over-optimizing the policy to a fixed point
- Add `target_kl=0.02` to hard-stop the epoch loop when KL gets too high
- Increase batch to 128 and n_steps to 4096 for better balance

**Run dir**: `PPOmask/outputs/20260410_223914_PPO_7/`

---

## Cycle 8 — Structural Fix: n_epochs=3, target_kl (2026-04-11, 02:41-04:43)

**Wall-clock**: 2h (7,200s) | **Episodes**: 26,466 | **Timesteps**: 15.4M | **FPS**: ~2,050

### Code Changes
- **New**: `--target-kl` CLI flag in `train_maskedppo.py` — passes to `MaskablePPO(target_kl=...)`. When `approx_kl > target_kl` during epoch loop, SB3 stops iterating early.

### Configuration Changes from C7
| Parameter | C7 | C8 | Why |
|-----------|----|----|-----|
| n_epochs | 10 | **3** | Reduce gradient steps: 1280 → 384 per rollout |
| target_kl | none | **0.02** | Safety limit on policy update size |
| batch_size | 64 | **128** | Balance noise/stability |
| n_steps | 2048 | **4096** | More diverse rollout data per update |

Gradient steps per rollout: `4_envs × 4096 / 128 × 3_epochs = 384` (vs C7's 1280, C1-C6's 640)

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | **$95,900** ← best at the time |
| Eval det restock | 0 |
| Eval det PSC | 40 |
| Eval det setup | 6 |
| Eval det violation | rc_negative_L2 |
| Training best profit | $202,900 |
| Training final avg | $52,601 |
| Training violation rate | 100% |

### Timeline & Symptoms
- **0-1min**: `policy_gradient_loss=-1.8e-5`, `entropy=-0.297` (max). `fps=2050` — 2x faster than C7 (fewer gradient steps).
- **0-30min**: Fast learning. `best=$139k`. `stall_count=0`. But `approx_kl=2.6e-6` already weakening.
- **30min-1h**: `approx_kl=8.1e-10` — collapsed. `entropy=-0.104` (worst ever at the time). `explained_variance=0.512`. `eval/mean_reward=$45,900`.
- **1h-2h**: Frozen. Policy settled into the $95.9k strategy.

### Diagnosis
- **Best eval yet** ($95.9k) — a *third* distinct strategy: 40 PSC, 6 setups (more SKU diversity than C1-C7)
- `n_epochs=3` produced a different and better policy than `n_epochs=10`, even though KL still collapsed at 1h
- The initial gradient window (0-30min) was enough to push the policy to a better region before freezing
- `target_kl=0.02` never actually fired — the problem is KL going too *low*, not too high
- 2x fps means 2x more environment experience per hour

### Action Taken
- Add **danger zone penalty** to the reward function — escalating penalty when `rc_stock < safety_stock/2`
- Resume from C8 to test the new reward signal

**Run dir**: `PPOmask/outputs/20260411_024100_PPO_8/`

---

## Cycle 9 — Danger Zone Penalty (2026-04-11, 04:43-07:45)

**Wall-clock**: 3h (10,800s) | **Episodes**: 28,457 | **FPS**: ~1,920

### Code Changes
- **Modified**: `roasting_env.py` `_advance_until_decision()` — added escalating penalty in the RC maintenance reward block:
```python
# 3 zones per line per slot:
# rc >= safety_stock (20):    +bonus ($50)
# 10 <= rc < safety_stock:    $0 (neutral)
# rc < safety_stock/2 (0-9):  penalty = -3 × bonus × (1 - rc/10)
#   At rc=0: -$150/slot per line
#   At rc=5: -$75/slot per line
```

### Configuration Changes from C8
| Parameter | C8 | C9 | Why |
|-----------|----|----|-----|
| resume | fresh | **C8 final model** | Test danger zone penalty on existing policy |
| time | 7,200s | **10,800s** | 3h |

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $95,900 | **Identical to C8** (frozen) |
| Eval det restock | 0 |
| Training best profit | $176,495 |
| Training final avg | $56,136 |
| **L1 violations** | **10** (vs C8's 13,877!) |
| L2 violations | 28,447 |

### Timeline & Symptoms
- **0-1min**: `approx_kl=2.4e-10` — dead from start (resumed). `explained_variance=0.919`.
- **0-1h**: `violation_rate=100%`. L1 violations dropping rapidly (3,433 at 30min → eventually 10 total).
- **1h-3h**: Policy frozen. Stochastic training found better L1 management but deterministic unchanged.

### Diagnosis
- Danger zone penalty **massively effective for L1**: violations dropped from 13,877 to just 10 (99.93% reduction!)
- But the resumed policy was frozen — couldn't learn to protect L2
- The reward signal works; it just needs a live gradient to act on
- **The danger zone penalty is a keeper** — must combine with fresh start

### Action Taken
- Combine ALL working ingredients in a fresh start: n_epochs=3, target_kl=0.02, danger zone penalty, rc_bonus=50, fresh seed

**Run dir**: `PPOmask/outputs/20260411_044343_PPO_9/`

---

## Cycle 10 — All Fixes Combined, Fresh (2026-04-11, 07:46-11:47)

**Wall-clock**: 4h (14,400s) | **Episodes**: 37,430 | **Timesteps**: 28.5M | **FPS**: ~1,950

### Configuration Changes from C9
| Parameter | C9 | C10 | Why |
|-----------|----|----|-----|
| seed | 99 | **200** | Fresh initialization |
| resume | C8 final | **NO — FRESH START** | Combine all fixes from scratch |

All active: n_epochs=3, target_kl=0.02, danger zone penalty, rc_bonus=50, batch=128, n_steps=4096, clip=0.3, ent=0.05, lr=5e-4

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $99,100 |
| Eval det restock | 0 |
| Eval det PSC | 40 |
| Training best profit | $287,185 |
| Training final avg | $83,613 |
| **Training violation rate overall** | **91.2%** ← BREAKTHROUGH |
| **Training violation rate last 100** | **86%** |

### Stochastic Eval of Best Model (10 det + 20 stoch)
| Mode | Mean profit | Zero-violation eps | Restocks |
|------|------------|-------------------|----------|
| Deterministic | $99,100 | 0/10 | 0 |
| **Stochastic** | **$173,065** | **5/20 (25%)** | **16-20** |

Best stochastic episodes: $241,400 (100 PSC, 17 restocks, 0 violations), $240,200 (96 PSC, 18 restocks, 0 violations) — **Grade B!**

### Timeline & Symptoms
- **0-1h**: `policy_gradient_loss=-6.5e-5` (strongest at the time). `violation_rate=92%` (first below 100% at 1h!). `best=$268k`. `stall_count=0`.
- **1h-2h**: `violation_rate=84%` (16% non-violating!). `ep_rew_mean` doubled $42k→$82k. `explained_variance=0.846`. But `approx_kl=1.1e-10` — gradient dying.
- **2h-4h**: `violation_rate` stabilized at 84-86%. Gradient dead. `explained_variance=0.939`. No further improvement.

### Diagnosis
- **FIRST CYCLE WITH VIOLATION RATE BELOW 100%** — 16% of training episodes completed without violations
- 25% of stochastic eval episodes achieved zero violations with $221-241k profit (Grade B!)
- The combination of danger zone + structural fix + fresh start enabled genuine learning
- Gradient lasted ~2h before `explained_variance` hit 0.846 and killed it
- The policy learned significantly in the 0-2h window before freezing at a much better point than any previous cycle

### Action Taken
- Try resuming to push further (C11 — spoiler: didn't work)
- Then try 8h fresh runs with different seeds to find one where gradient lives longer

**Run dir**: `PPOmask/outputs/20260411_074550_PPO_10/`

---

## Cycle 11 — Resume from C10 (abandoned) (2026-04-11, 11:50-14:00)

**Wall-clock**: ~2h (killed) | **Episodes**: ~13,800

### Configuration
Same as C10, but `--resume-from C10/best_training_profit_model.zip`

### Results
- Killed at 2h. `approx_kl=3.6e-10` from step 1. `violation_rate=88-90%`.
- No improvement over C10.

### Diagnosis
**4th confirmation that resume never works.** Stopped wasting time on resume approach.

### Action Taken
- Switch to 8h fresh runs to give more time for the initial learning window

**Run dir**: `PPOmask/outputs/20260411_115028_PPO_11/` (incomplete)

---

## Cycle 12 — Lucky Seed 300, 8h (2026-04-11, 13:50 → 2026-04-11, 21:52)

**Wall-clock**: 8h (28,800s) | **Episodes**: 93,295 | **Timesteps**: ~50M | **FPS**: ~1,830

### Configuration Changes from C10
| Parameter | C10 | C12 | Why |
|-----------|----|----|-----|
| seed | 200 | **300** | Try new seed |
| time | 14,400s | **28,800s** | 8h — longest fresh run |

### Results
| Metric | Value |
|--------|-------|
| **Eval det profit (final model)** | **$120,700** |
| **Eval det restock (final model)** | **2** ← FIRST EVER! |
| **Eval det profit (best model)** | **$127,900** |
| **Eval det restock (best model)** | **7** |
| Eval det PSC (best model) | 41 |
| Eval det setup (best model) | 5 |
| Training best profit | $269,665 |
| Training violation rate overall | 99.9% |

### Stochastic Eval of Best Model
| Mode | Mean profit | Violations | Restocks |
|------|------------|------------|----------|
| Deterministic | $127,900 | 10/10 | 7 |
| Stochastic | $137,640 | 20/20 | 7-17 |

### Timeline & Symptoms — THE BREAKTHROUGH
- **0-1h**: `approx_kl=8.3e-6` at 1h — alive! (C10 was dead by 1h). `explained_variance=0` (value function not converging!). `best=$141k`. `pg_loss=-5.4e-6`.
- **1h-2h**: `approx_kl=7.6e-6` — **STILL ALIVE at 2h!** (All previous cycles dead by 30min-1h). `explained_variance` still 0! `pg_loss=-2.8e-5` — gradient **strengthening**. `best=$235k`. `ep_rew_mean` dipped to -$13k (heavy exploration).
- **2h-4h**: `approx_kl=1.3e-4` at 4h — **strongest KL measurement ever!** `pg_loss=2.85e-5` (positive — policy being pushed). `entropy=-0.044` (near-uniform — maximum exploration). `ep_len_mean` dropped from 803 to 501 (trying radically different strategies). `best=$270k`.
- **4h-6h**: **Gradient died.** `approx_kl=3.5e-11`. `explained_variance` jumped from 0 to 0.644. Value function finally converged. `eval/mean_reward=$49,600`.
- **6h-8h**: Frozen. `explained_variance=0.994`. Policy locked.

### Diagnosis
- **FIRST DETERMINISTIC RESTOCK EVER** — the argmax policy chose restock at 7 decision points
- Seed 300 kept `explained_variance=0` for **4 full hours** — 1000x longer gradient lifespan than any other seed
- When `explained_variance=0`, advantage estimates are pure noise → policy gradient stays nonzero → policy keeps changing
- The 4h of active learning was enough to push restock logits above the argmax threshold at some decision points
- Between 2-4h, the policy was in "maximum exploration mode" (entropy=-0.044, near-uniform) — this is what enabled the breakthrough

### Key Discovery
**Gradient lifespan = time until value function converges.** `explained_variance > 0.5` → gradient dies. Seed 300's random initialization produced a value network that inherently took 4h to converge. This was initialization luck, not a configurable parameter.

### Action Taken
- Test other seeds to see if reproducible (C13)
- Try `vf_coef=0.1` to deliberately slow value convergence (C14)
- Try larger network (C15)

**Run dir**: `PPOmask/outputs/20260411_135044_PPO_12/` ← **CURRENT BEST**

---

## Cycle 13 — Seed 400 Test (2026-04-11, 21:52 → 2026-04-12, 05:55)

**Wall-clock**: 8h (28,800s) | **Episodes**: 69,691 | **FPS**: ~1,740

### Configuration Changes from C12
| Parameter | C12 | C13 | Why |
|-----------|----|----|-----|
| seed | 300 | **400** | Test different seed |

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $62,700 |
| Eval det restock | 0 |
| Training best profit | $226,135 |
| Training violation rate | 100% |

### Timeline & Symptoms
- **0-2h**: `explained_variance=0.497` at 2h — **converged too fast** (C12 was 0 at 2h). `approx_kl=2.5e-11`. Gradient dead.
- **2h-8h**: Completely frozen. No improvement. `best=$226k` never changed past 2h.

### Diagnosis
- Seed 400 unfavorable — value function converged in <2h vs C12's 4h
- Confirmed: **seed 300 was special**, not the configuration
- 6h of compute wasted on a frozen policy

**Run dir**: `PPOmask/outputs/20260411_215219_PPO_13/`

---

## Cycle 14 — Lower vf_coef Experiment (2026-04-12, 05:56-13:58)

**Wall-clock**: 8h (28,800s) | **Episodes**: ~68k | **FPS**: ~1,740

### Configuration Changes from C12
| Parameter | C12 | C14 | Why |
|-----------|----|----|-----|
| vf_coef | 0.5 | **0.1** | Hypothesis: lower weight on value loss → slower convergence |

Seed=300 (same as C12 for controlled comparison)

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $127,500 |
| Eval det restock | 1 |
| Eval det PSC | 44 |
| Training best profit | $272k |

### Timeline & Symptoms
- **0-2h**: `explained_variance=0.803` at 2h — **converged just as fast as C13!** (C12 was 0 at 2h). `approx_kl=9.9e-9`. Gradient dead.
- **2h-8h**: Frozen. Value function hit 0.994 by end.

### Diagnosis
- **Hypothesis disproved.** `vf_coef=0.1` doesn't slow value convergence — `explained_variance=0.803` at 2h
- The value network learns because even 0.1x weight on value loss produces sufficient gradients
- But eval=$127,500 nearly matches C12's $127,900 — **seed 300 consistently produces ~$127k policies** regardless of `vf_coef`
- C12's 4h gradient was truly initialization luck, not reproducible with the same seed + different params

**Run dir**: `PPOmask/outputs/20260412_055557_PPO_14/`

---

## Cycle 15 — Larger Network (2026-04-12, 13:58-22:00)

**Wall-clock**: 8h (28,800s) | **Episodes**: ~73k | **FPS**: ~1,260 (slower — bigger net)

### Configuration Changes from C12
| Parameter | C12 | C15 | Why |
|-----------|----|----|-----|
| net_arch | 256,256 | **512,512,256** | Bigger network might resist value convergence |
| lr | 5e-4 | **3e-4** | Lower for stability with larger network |

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $54,900 |
| Eval det restock | 0 |
| Training best profit | $229,475 |

### Timeline & Symptoms
- **0-2h**: `approx_kl=1.5e-4` at 2h — **alive, 20x stronger than C12!** `explained_variance=0` (unconverged). `pg_loss=2.5e-5`. This confirmed: bigger network delays value convergence.
- **2h-4h**: Gradient died. `approx_kl=0.0`. `explained_variance=0.74`. `entropy=-0.149`. Policy froze in a bad region.
- **4h-8h**: Frozen. No improvement.

### Diagnosis
- Bigger network **did delay value convergence by ~1h** (died at ~3h vs C12's 4h for 256,256)
- But the extended exploration period with near-uniform policy (`entropy=-0.044`) pushed the policy into a **worse** region
- The frozen policy at $54.9k was worse than all C8+ cycles
- **Larger network = longer gradient but worse final policy.** The extra capacity created a more complex loss landscape that the optimizer couldn't navigate usefully.
- Also 33% slower fps (1260 vs 1830) — less environment experience per hour

### Action Taken
- Return to proven 256,256 architecture
- Implement **ValueHeadPerturbCallback** — inject noise into value network when gradient stalls, to artificially de-converge the value function and restore gradient signal

**Run dir**: `PPOmask/outputs/20260412_135816_PPO_15/`

---

## Cycle 16 — Value Head Perturbation v1 (2026-04-12, 22:02 → 2026-04-13, 06:03)

**Wall-clock**: 8h (28,800s) | **Episodes**: 101,954 | **Timesteps**: ~56M | **FPS**: ~1,980

### Code Changes
- **New**: `ValueHeadPerturbCallback` in `callbacks.py`:
  - When `approx_kl < 1e-6` for 10 consecutive rollouts → add Gaussian noise (std=0.1) to value network head weights
  - Logs `perturb/stall_count` and `perturb/perturb_count` to TensorBoard
- **New**: `--value-perturb` CLI flag in `train_maskedppo.py`

### Configuration
```
--seed 300 --n-envs 4 --lr 5e-4 --lr-schedule linear --n-steps 4096
--batch-size 128 --n-epochs 3 --gamma 0.99 --ent-coef 0.05
--clip-range 0.3 --target-kl 0.02 --rc-maintenance-bonus 50.0
--net-arch 256,256 --value-perturb --time 28800
```

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $120,700 |
| Eval det restock | 2 |
| Eval det PSC | 30 |
| Eval det setup | 2 |
| Eval det violation | rc_negative_L1 |
| Training best profit | $269,665 |
| Training final avg | $43,075 |
| Training violation rate | 99.9% |
| **perturb_count** | **0 (never fired!)** |

### Timeline & Symptoms
- **0-23min**: `approx_kl=0.0014`, `clip_fraction=0.00285` (**first ever non-zero clipping!**), `pg_loss=0.0001`. Strongest gradient signals ever recorded. `explained_variance=0`.
- **23min-1h50m**: `approx_kl=0.000336`, `clip_fraction=0.000936`, `pg_loss=0.000252`. Gradient oscillating, not monotonically decaying. `explained_variance` still 0.
- **1h50m-2h**: `approx_kl` bounced to 0.00197, then 0.000449, then 0.000292. Oscillating near 1e-4 to 1e-3 range. `pg_loss=-0.000328` (strongest negative ever).
- **2h-4h**: `approx_kl=1.46e-6` at 4h — weakened but still alive. `explained_variance` still 0. `best=$270k`. `ep_len_mean=172, psc_fraction=0.339` — heavy exploration mode. Policy trying radically different strategies.
- **4h-6h**: **Gradient collapsed.** `approx_kl=1.24e-10`. `explained_variance=0.855`. `entropy=-0.044` (near-uniform). `perturb_count=0` — **callback never fired!**
- **6h-8h**: Frozen. No improvement.

### Diagnosis
- Gradient was alive for ~5h (longest ever!) with unprecedented strength (KL 45-260x higher than C12)
- **Perturbation callback NEVER FIRED** — design flaw: `patience=10` with `kl_floor=1e-6` was too conservative. KL oscillated between 1e-6 and 1e-3, resetting the stall counter each time it spiked above 1e-6. By the time KL truly died (1.24e-10), it happened too fast for 10 consecutive readings.
- Final eval $120.7k with 2 restocks = same as C12's final model (not best model)
- The 5h of active gradient produced the same quality policy as C12's 4h — no improvement from the extra hour

### Action Taken
- **Fixed ValueHeadPerturbCallback**:
  - `kl_floor` raised from 1e-6 to **1e-4** (trigger much earlier)
  - `ev_ceiling=0.5` added — also triggers when `explained_variance > 0.5`
  - `patience` lowered from 10 to **3**
  - Stall counter now **decays by 1** instead of hard reset (brief KL spikes don't fully reset)
  - `max_perturbs=50` safety cap

**Run dir**: `PPOmask/outputs/20260412_220152_PPO_16/`

---

## Cycle 17 — Fixed Value Head Perturbation (2026-04-13, 06:10, RUNNING)

**Wall-clock**: 8h budget (28,800s) | Running

### Code Changes from C16
- **Fixed**: `ValueHeadPerturbCallback`:
  - `kl_floor`: 1e-6 → **1e-4** (trigger when KL drops below 0.0001, not 0.000001)
  - `ev_ceiling`: none → **0.5** (also trigger when value function converges)
  - `patience`: 10 → **3** (fire after 3 stalled rollouts, not 10)
  - Stall counter: hard reset → **decay by 1** (more persistent tracking)
  - `max_perturbs`: none → **50** (safety cap)

### Configuration
Same as C16:
```
--seed 300 --n-envs 4 --lr 5e-4 --lr-schedule linear --n-steps 4096
--batch-size 128 --n-epochs 3 --gamma 0.99 --ent-coef 0.05
--clip-range 0.3 --target-kl 0.02 --rc-maintenance-bonus 50.0
--net-arch 256,256 --value-perturb --time 28800
```

### Results (killed at 2h)
| Metric | Value |
|--------|-------|
| perturb_count | **50 (hit max_perturbs cap!)** |
| approx_kl | 1.3e-9 — dead |
| explained_var | 0.896 — converged despite 50 perturbations |
| best_profit | $245,965 |
| ep_rew_mean | $25,700 |
| violation_rate | 100% |

### Timeline & Symptoms
- **0-3min**: Perturbation #1 fired at `kl=4.65e-6, stall=3`. `kl_floor=1e-4` was too aggressive — triggered almost immediately before learning could start.
- **3min-2h**: Perturbation kept firing on nearly every rollout. Hit `max_perturbs=50` cap. Value network recovered from each noise injection within a few gradient steps. `explained_variance` climbed to 0.896 despite 50 perturbations.
- **At 2h**: Gradient dead. Perturbation exhausted. Killed.

### Diagnosis
- **Value perturbation approach fundamentally flawed** for this problem. The value network self-heals faster than we can disrupt it. Each perturbation (std=0.1) is undone by 3 epochs of gradient updates.
- `kl_floor=1e-4` was too aggressive — triggered before the policy had any learning time
- Even 50 perturbations couldn't prevent value convergence
- **Conclusion: cannot solve gradient death through value network noise injection**

### Action Taken
- Abandoned value perturbation approach entirely
- New strategy: **10x violation penalty ($500k)** to make violation avoidance the dominant learning signal
- Added `--violation-penalty` CLI flag to `train_maskedppo.py`

**Run dir**: `PPOmask/outputs/20260413_061002_PPO_17/` (incomplete, killed at 2h)

---

## Cycle 18 — 10x Violation Penalty (2026-04-13, RUNNING)

**Wall-clock**: 8h budget (28,800s) | Running

### Code Changes
- **New**: `--violation-penalty` CLI flag in `train_maskedppo.py` — overrides `data.violation_penalty`

### Configuration Changes from C16
| Parameter | C16 | C18 | Why |
|-----------|----|----|-----|
| violation_penalty | 50,000 | **500,000** | Make violation catastrophic — force survival learning |
| value-perturb | yes | **no** | Abandoned — doesn't work |

### Rationale
With $50k penalty, the agent earns ~$60-95k in PSC revenue before violating — net positive even with the penalty. At $500k, **any violation is a net -$400k+ catastrophe**. The only way to get positive reward is to not violate. This should force the deterministic policy to choose restock at the critical moments.

### Results (killed at ~40min)
- `ep_rew_mean=-$527,000` — every episode is catastrophic
- `best_profit=-$459,000` — even best episode deeply negative
- $500k penalty too harsh — no gradient toward "better violations," just uniform catastrophe
- **Killed early — approach abandoned**

**Run dir**: `PPOmask/outputs/20260413_075905_PPO_18/` (incomplete)

---

## Brainstorm #1 (after C18) — Root Cause Analysis

After 18 cycles, the user identified the **true root cause**: SB3's default shared backbone.

**Shared backbone** (`net_arch=[256,256]`): One feature extractor feeds both policy and value heads. Value loss (MSE, initially ~4×10¹⁰) dominates the gradient on shared weights → features optimize for prediction → when value converges, features freeze → policy head can't change features → gradient dies.

**Separate networks** (`net_arch={"pi":[256,256], "vf":[256,256]}`): Policy and value have independent weights. Value convergence freezes Wv but Wπ remains free. Advantage signal persists because V(s) predicts AVERAGE return but per-action advantages are still nonzero. Cost: ~2x params (153k vs 80k), ~10-15% fps loss. **This is the correct fix.**

---

## Cycle 19 — SEPARATE NETWORKS (2026-04-13, 09:29, RUNNING)

**Wall-clock**: 4h budget (14,400s) | Running

### Code Changes
- **New**: `--separate-networks` flag in `train_maskedppo.py`
- Changes `policy_kwargs` from `{"net_arch": [256,256]}` to `{"net_arch": {"pi": [256,256], "vf": [256,256]}}`
- Total params: ~153k (vs 80k shared). FPS impact: ~10-15% slower.

### Configuration
```
--seed 42 --n-envs 4 --lr 5e-4 --lr-schedule linear --n-steps 4096
--batch-size 128 --n-epochs 3 --gamma 0.99 --ent-coef 0.05
--clip-range 0.3 --target-kl 0.02 --rc-maintenance-bonus 50.0
--net-arch 256,256 --separate-networks --time 14400
```

**Seed=42 deliberately** — not lucky seed 300. If separate networks fix the gradient death, ANY seed should work. Using seed 42 proves the fix is architectural, not initialization-dependent.

### Key Test Criteria
```
@2h: explained_var > 0.5 AND approx_kl > 1e-4 → CONFIRMED FIX
@2h: approx_kl < 1e-8 AND explained_var > 0.8 → FAILED, kill
@4h: approx_kl > 1e-6 → gradient alive past C12's death point → BREAKTHROUGH
```

### Results
| Metric | Value |
|--------|-------|
| Eval det profit | $54,900 |
| Eval det restock | 0 |
| Eval det PSC | 40 |
| Training best profit | $217,995 |
| Training final avg | $64,786 |
| Training violation rate | 100% |

### Timeline & Symptoms
- **0-18min**: `approx_kl=0.003`, `clip_fraction=0.005`, `pg_loss=0.000822` — **record-breaking gradient** (8x stronger than any prior cycle). `entropy=-0.363` (highest ever — separate networks give policy more freedom).
- **18-30min**: `approx_kl=0.00547`, `clip_fraction=0.0446` (4.5%!), `pg_loss=0.00356` — **35x stronger gradient than any C1-C18 at 30min**. `explained_var=0`. `best_profit` climbing rapidly.
- **30min-1h**: Gradient still strong. `eval/mean_reward` crossed **positive ($85)** at ~1h8min. `best=$218k`. `ep_rew_mean=$76k`.
- **1h-1.5h**: `explained_variance` jumped from 0 to **0.596**. `pg_loss` dropped from 7.4e-5 to **1.1e-6** — dying. Value network converged **independently** (not through shared weights).
- **1.5h-4h**: Gradient dead. Policy regressed. Final eval $54.9k (worse than C12's $127.9k).

### Diagnosis
- **Separate networks partially worked**: gradient was 35x stronger and lasted ~1.5h (vs <1h shared). `clip_fraction` was non-zero for the first time ever.
- **But didn't solve the root cause**: the value network converges independently through its own weights, and once V(s) is accurate, advantage estimates A(s,a) = R + γV(s') - V(s) ≈ 0 for all actions → gradient dies.
- **The problem is NOT shared weights** — it's that accurate value predictions kill per-action advantages regardless of architecture.
- Eval briefly hit $85 (positive!) at 1h, but 3h of frozen training pushed policy to the $54.9k local optimum.

### Key Insight
The advantage A(s,a) approaches zero when V(s) is accurate because **all actions from the same state lead to similar outcomes** (the agent hasn't learned to differentiate restock vs WAIT yet). The value function correctly predicts the *average* future return, but per-action advantages are too small for the policy gradient to act on.

**Run dir**: `PPOmask/outputs/20260413_093001_PPO_19/`

---

## Cycle 20 — Plan (Brainstorm #2 due)

### What we know after 19 cycles

1. **Hyperparameter tuning**: exhausted (C1-C8)
2. **Resume**: never works (C4-C6, C9, C11)
3. **Value perturbation**: value self-heals too fast (C16-C17)
4. **Separate networks**: stronger gradient but still dies when value converges (C19)
5. **Lucky seeds**: not reproducible or scalable (C12 vs C13-C14)
6. **Larger network**: worse final policy (C15)

### Root cause (refined)
The advantage A(s,a) ≈ 0 for all actions once V(s) is accurate. This is because from any given state, the *immediate* reward difference between restock and WAIT is small (~$50 bonus vs $0), while the *long-term* consequence (avoiding -$50k violation 100 slots later) is heavily discounted by γ=0.99 (0.99^100 = 0.37). The credit assignment problem is too hard for PPO's GAE.

### C20 Options
- **A: VecNormalize(norm_reward=True)** + separate networks — normalize rewards to unit variance, making small advantages relatively larger ← **CHOSEN**
- B: Lower gamma (0.95) + separate networks — reduce horizon so restock credit assignment is easier
- C: Accept C12 as best PPO result, move to thesis writing

---

## Cycle 20 — Separate Networks + VecNormalize (2026-04-13, 19:11, RUNNING)

**Wall-clock**: 4h budget (14,400s) | Running

### Code Changes
- **New**: `--normalize-reward` flag in `train_maskedppo.py`
- Wraps training env: `VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)`
- Wraps eval env: `VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)` (for sync compatibility)
- Effect: raw rewards $[-50k, +300k]$ → normalized to ~$[-2, +3]$. Loss scale drops from ~3×10⁷ to ~0.04.

### Configuration
```
--seed 42 --n-envs 4 --lr 5e-4 --lr-schedule linear --n-steps 4096
--batch-size 128 --n-epochs 3 --gamma 0.99 --ent-coef 0.05
--clip-range 0.3 --target-kl 0.02 --rc-maintenance-bonus 50.0
--net-arch 256,256 --separate-networks --normalize-reward --time 14400
```

### Bug Fix
First launch crashed: `AssertionError: expected the eval env to be a VecEnvWrapper`. SB3's `MaskableEvalCallback` calls `sync_envs_normalization()` which requires eval env to also be wrapped in `VecNormalize` (with `training=False, norm_reward=False`).

### Live Status (at ~2min, 126s elapsed)
| Metric | C20 @2min | C19 @2min | Significance |
|--------|----------|----------|-------------|
| approx_kl | **0.0048** | 0.0023 | 2x higher |
| clip_fraction | **0.0138** | 0.005 | 2.8x |
| **explained_var** | **0.899** | 0 | **Already converged!** |
| **approx_kl with ev>0.5** | **YES** | NO (ev was 0) | **★ KEY TEST PASSED ★** |
| loss | **0.04** | 3.3e+7 | 6 orders of magnitude smaller |
| value_loss | **0.15** | 6.6e+7 | Normalized scale |
| ep_rew_mean | **$17,600** | -$75,600 | Already positive |
| best_profit | **$74,600** | -$16,000 | Way ahead |

### ★ KEY TEST RESULT ★
**`explained_variance = 0.899 AND approx_kl = 0.0048`** — BOTH conditions met simultaneously!

This has NEVER happened in C1-C19. In all prior cycles, `explained_var > 0.5` meant `approx_kl < 1e-8` (dead). VecNormalize breaks this coupling by keeping the loss scale small enough that even when explained_variance is high, the normalized advantages produce meaningful gradients.

### Progress Updates

**@2min**: `explained_var=0.899 AND approx_kl=0.0048` — KEY TEST PASSED. Gradient alive despite value convergence.

**@30min**: `best=$406k`, `ep_rew_mean=$323k`, `violation_rate=34%` (66% non-violating!). `approx_kl=0.0065`. Gradient stronger than at 2min.

**@1h8m**: `best=$463k` 🚀, `ep_rew_mean=$373k`, `violation_rate=23%` (77% non-violating!). `approx_kl=0.0024`, `clip_fraction=0.004`. Latest episode: 114 PSC, 0 stockouts, $417k.

### Mid-training Eval (best_training_profit_model, at ~1h)
| Mode | Mean Profit | PSC | Restock | Violations | Best Episode |
|------|------------|-----|---------|------------|-------------|
| **Deterministic** | **$292,900** | **65** | **7** | 5/5 (rc_negative_L1) | $292,900 |
| Stochastic | $370,380 | 60-115 | 6-18 | 3/5 | **$466,400** (0 violations!) |

**$292,900 deterministic = Grade A profit!** (above $280k threshold). Still has 1 violation per episode (L1), but the scheduling policy is real: 65 PSC, 7 restocks, 6 setups.

**$466,400 stochastic best = Grade A+++ with zero violations!** 115 PSC, 18 restocks. Full episode completion.

### Final Results (4h complete)
| Metric | Value |
|--------|-------|
| **Eval det profit** | **$470,600 — GRADE A** |
| **Eval det violations** | **0/100 — ZERO** |
| Eval det PSC | 120 |
| Eval det restock | 17 |
| Eval det setup | 4 |
| Eval det stockouts | 0 |
| Training best profit | $481,745 |
| Training final avg (last 1000) | $425,916 |
| Training violation rate last 100 | **2%** |
| Training violation rate overall | 20.5% |
| Training episodes | 27,189 |
| Grade | **A** |

### Full Timeline
- **0-2min**: KEY TEST PASSED — `explained_var=0.899 AND approx_kl=0.0048` (first time ever both conditions met)
- **2min-30min**: `best=$406k`, `violation_rate=34%→23%`, `ep_rew_mean=$323k`, gradient strong
- **30min-1h**: `best=$463k`, `violation_rate=23%`, eval crossed positive ($85→$292.9k mid-training)
- **1h-2h**: `best=$463k→$468k`, `violation_rate=26%` stabilized, gradient weakening but still active
- **2h-4h**: Policy consolidated. Violation rate dropped to **2% (last 100)**. Best reached $481.7k. Final model learned full-shift completion.

### Schedule Analysis (from eval)
- **All 5 roasters active** across the full 480-slot shift
- **120 PSC batches** — near theoretical maximum
- **17 restocks** — proactive inventory management throughout the shift
- **4 setup events** — minimal SKU changeovers
- **RC stock stays positive** for all 480 slots on both lines
- **Zero stockouts, zero violations**

### What Made C20 Work
The combination of **separate policy/value networks** + **VecNormalize(norm_reward=True)** solved the gradient death problem:
1. Separate networks: policy weights independent from value convergence
2. VecNormalize: keeps loss scale at ~0.04 (vs ~30M raw), so advantages remain meaningful even when value is accurate
3. Result: `explained_var=0.979 AND approx_kl=0.0065` simultaneously — impossible in C1-C19

**Run dir**: `PPOmask/outputs/20260413_191311_PPO_20/`

### Post-C20 Discovery: MTO Skip Penalty Missing!
After examining the Gantt chart, found that C20's policy produced **0 BUSTA batches** (5 required). The simulation engine had NO penalty for skipping MTO jobs entirely — only tardiness for late completion. The agent exploited this loophole.

**Fix applied** (affects ALL solvers, not just PPO):
- Added `_penalize_skipped_mto()` to `env/simulation_engine.py` — $50k per unfinished MTO batch at end of shift
- Added `mto_skip_penalty_per_batch=50000` to `Input_data/shift_parameters.csv`
- Applied penalty at all termination points in PPO env (violation + end-of-shift)
- Also added `--completion-bonus` CLI flag ($100k for completing full shift without violation)

**C20's corrected profit**: $470,600 - (5 BUSTA × $50k) = **$220,600** (below Q-learning's $250k)

---

## Cycle 21 — Corrected Rewards + Completion Bonus (2026-04-14, RUNNING)

**Wall-clock**: 4h budget (14,400s) | Running

### Code Changes
- **Fixed**: `env/simulation_engine.py` — added `_penalize_skipped_mto()` method
- **Fixed**: `env/kpi_tracker.py` — added `mto_skipped` field
- **Fixed**: `env/data_bridge.py` — reads `mto_skip_penalty_per_batch` from CSV, passes as `c_skip_mto`
- **Fixed**: `PPOmask/Engine/roasting_env.py` — calls skip penalty at all termination points
- **New**: `--completion-bonus` flag in train_maskedppo.py

### Configuration
```
--seed 42 --n-envs 4 --lr 5e-4 --lr-schedule linear --n-steps 4096
--batch-size 128 --n-epochs 3 --gamma 0.99 --ent-coef 0.05
--clip-range 0.3 --target-kl 0.02 --rc-maintenance-bonus 50.0
--completion-bonus 100000 --separate-networks --normalize-reward --time 14400
```
MTO skip penalty ($50k/batch) active via simulation engine (no CLI override needed).

### Incentive Structure
```
Skip BUSTA + violate:   65 PSC - $50k viol - $250k skip = -$202,500  ← BAD
Skip BUSTA + survive:   120 PSC - $250k skip + $100k bonus = -$30,000  ← STILL BAD
Do BUSTA + survive:     110 PSC + 5 NDG + 5 BUSTA - costs + $100k bonus ≈ $300k+ ← GOOD
```
The ONLY path to high profit is completing ALL MTO jobs + full shift.

### Live Status (at ~2min)
- `approx_kl=0.0055`, `clip_fraction=0.0381`, `pg_loss=-0.00236` (12x stronger than C20!)
- `explained_var=0.923` + gradient alive = confirmed fix still works
- `ep_rew_mean=-$454k` (deeply negative due to MTO skip penalty — must learn BUSTA)

**Run dir**: `PPOmask/outputs/20260414_000042_PPO_21/` (killed, replaced by C22)

---

## Cycle 22 — 12h Full Run with Corrected MTO (2026-04-14, 00:01, RUNNING)

**Wall-clock**: 12h budget (43,200s) | Currently at ~6h40m

### Configuration
Same as C21 but 12h:
```
--seed 42 --n-envs 4 --lr 5e-4 --lr-schedule linear --n-steps 4096
--batch-size 128 --n-epochs 3 --gamma 0.99 --ent-coef 0.05
--clip-range 0.3 --target-kl 0.02 --rc-maintenance-bonus 50.0
--completion-bonus 100000 --separate-networks --normalize-reward --time 43200
```
MTO skip penalty ($50k/batch) active. All fixes from C20 (separate nets + VecNormalize).

### Progress Updates

**@2min**: `approx_kl=0.0055`, `clip_fraction=0.0381`, `pg_loss=-0.00236`. Gradient 12x stronger than C20. `ep_rew_mean=-$461k` (MTO penalty biting).

**@1h**: `eval=$163,005`. `target_kl FIRED` for first time ever — policy changing so fast it hit the 0.02 KL ceiling. Gradient strong.

**@3h**: `best=$521,940`, `ep_rew_mean=$427k`, `violation_rate=7%`. 93% non-violating. `approx_kl=0.005`. Agent scheduling 115 PSC with restocks.

**@6h**: `best=$554,000`, `ep_rew_mean=$416k`, **`violation_rate=0%` (last 100)**. `approx_kl=0.005` — **GRADIENT STILL ALIVE AT 6h!** Unprecedented. `pg_loss=0.00578` strongest ever at 6h mark in any cycle.

### Mid-training Eval (@6h40m, best_training_profit_model)
| Mode | Mean Profit | PSC | Restock | Setup | Violations | Tardiness |
|------|------------|-----|---------|-------|------------|-----------|
| **Deterministic** | **$422,400** | **116** | **16** | **7** | **0/5** | **0 min** |
| Stochastic | $409,420 | 98-117 | 13-17 | 7-9 | 1/5 | 0 min |

**Grade A confirmed: $422,400, zero violations, 7 setup events (doing NDG + BUSTA!), 16 restocks, 0 tardiness.** With the corrected MTO skip penalty this is genuine.

### Schedule Analysis (mid-training, best model)
- **116 PSC, 16 restocks, 7 setups** — agent does all 3 SKUs (PSC + NDG + BUSTA)
- **0 tardiness** — MTO jobs completed before due time (slot 240)
- **0 violations** — RC stock stays positive on both lines for full 480 slots
- **Full shift completion** — all 480 slots utilized

### Gradient Health at 6h
| Metric | Value |
|--------|-------|
| approx_kl | 0.00492 |
| clip_fraction | 0.00953 |
| pg_loss | 0.00578 |
| explained_var | 0.994 |
| entropy | -0.018 |

Gradient alive at 6h with `explained_var=0.994` — VecNormalize + separate networks confirmed as permanent fix.

**Awaiting 9h and 12h checkpoints + final eval.**

**Run dir**: `PPOmask/outputs/20260414_000110_PPO_22/`

---

## Gradient Lifespan Analysis

The core bottleneck across all 16 cycles: the policy gradient dies when the value function converges.

| Cycle | Net arch | Seed | explained_var @2h | KL @2h | Gradient lifespan | Eval |
|-------|----------|------|-------------------|--------|-------------------|------|
| C1 | 256,256 | 42 | high | ~0 | ~30min | $54.9k |
| C7 | 256,256 | 99 | 0.981 | 3.2e-9 | ~30min | $73.9k |
| C8 | 256,256 | 99 | 0.512 | 8.1e-10 | ~1h | $95.9k |
| C10 | 256,256 | 200 | 0.846 | 1.1e-10 | ~2h | $99.1k |
| **C12** | **256,256** | **300** | **0** | **7.6e-6** | **~4h** | **$127.9k** |
| C13 | 256,256 | 400 | 0.497 | 2.5e-11 | <2h | $62.7k |
| C14 | 256,256 | 300 | 0.803 | 9.9e-9 | <2h | $127.5k |
| C15 | 512,512,256 | 300 | 0 | 1.5e-4 | ~3h | $54.9k |
| C16 | 256,256 | 300 | 0 | 3.4e-4 | ~5h | $120.7k |
| C17 | 256,256 | 300 | 0.896 | 1.3e-9 | <2h (50 perturbs, all failed) | killed |
| C18 | 256,256 | 300 | — | — | killed (penalty=500k too harsh) | killed |
| C19 | 256,256 SEPARATE | 42 | 0→0.596 | 0.005→1e-6 | ~1.5h (35x stronger but still died) | $54.9k |
| C20 | 256,256 SEP+VECNORM | 42 | 0.994 | 0.0065 | ALIVE 4h+ | $470,600 (no MTO penalty) |
| **C22** | **256,256 SEP+VECNORM** | **42** | **0.994** | **0.005** | **ALIVE 6h+ (ongoing!)** | **$422,400 GRADE A (w/ MTO fix)** |

**Pattern**: Longer gradient lifespan correlates with better eval — except when the gradient period pushes the policy into a bad exploration region (C15). C16 had the longest gradient (5h) but perturbation never fired due to callback design flaw.

---

## Confirmed Rules

1. **Never resume** — gradient is dead on load (C4/C5/C6/C9/C11)
2. **n_epochs=3** beats n_epochs=10 every time (C8 vs C1-C7)
3. **Danger zone penalty** eliminates L1 violations (C9)
4. **rc_maintenance_bonus=50** is the right scale (C6+)
5. **256,256 > 512,512,256** — bigger network hurt (C15 vs C12)
6. **lr=5e-4** is necessary (C3 proved 3e-4 too low)
7. **Fresh start always** — each new seed has a chance at a good initialization
8. **Seed matters enormously** — seed 300 consistently hits $127k; other seeds vary wildly
