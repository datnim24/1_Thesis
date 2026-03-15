# Audit Resolution — All Open Items
## Cross-referenced to `cost.md`, `mathematical_model_complete.md`, and `simulation_logic_complete.md`

> **Purpose:** This document resolves every open item from the comprehensive audit. After this, there are **zero blocking items** — implementation can begin.

> **Canonical objective source:** As of the cost-based reformulation, `cost.md` is the single source of truth for economic coefficients ($c^{tard}, c^{stock}, c^{idle}, c^{over}$). Any older weight-based notation (for example $w^{stock}=10$) should be treated as historical prototype notation, not current thesis notation.

---

## Item 1 ✅ RESOLVED — RC Max Buffer

$$\overline{B}_l = \frac{20{,}000 \text{ kg (4 silos × 5,000 kg)}}{500 \text{ kg/batch}} = 40 \text{ batches per line}$$

**Implications:**
- A full buffer covers $40 \times 5.1 = 204$ minutes (~3.4 hours) of PSC demand with zero production
- Overflow constraint (C9) rarely binds under normal conditions
- After extended high-throughput with low consumption, overflow can occur — scheduler must pace production
- RC stock range: $[0, 40]$ per line (integer)

**Update needed in:** math model §2.2 (replace "TBD"), simulation logic §2.4, problem description §2.5.4, DRL observation normalization (index 16–17: divide by 40).

---

## Item 2 ✅ RESOLVED — Stockout Cost Coefficient (Cost-Based Formulation)

$$c^{stock} = \$1{,}500 	ext{/min/line}$$

**Calibration logic:**
- 1 minute of stockout stops the entire PSC packaging line
- The direct and indirect cost includes idle labor, energy waste, schedule disruption, and downstream service loss
- At $\$1{,}500$/min, a 5-minute stockout costs $\$7{,}500$, which is larger than the revenue of one PSC batch ($\$4{,}000$)
- This preserves the intended factory priority: **PSC continuity > MTO on-time > throughput smoothing**

**Cost hierarchy:** $c^{stock} (\$1{,}500) > c^{tard} (\$1{,}000) > c^{idle} (\$200) > c^{over} (\$50)$

This means the reactive solver will:
1. First priority: avoid stockout duration
2. Second priority: avoid MTO tardiness
3. Third priority: avoid unnecessary idle / overflow-blocked time while still maximizing profitable output

**Sensitivity analysis:** Optional secondary sensitivity can test $c^{stock} \in \{\$1{,}000, \$1{,}500, \$2{,}000\}$ if time permits. Not part of the primary factorial unless explicitly added.

---

## Item 3 ✅ RESOLVED — Initial `last_sku[r]` at Shift Start

$$\text{last\_sku}[r] = k^{PSC} \quad \forall r \in \mathcal{R}$$

All roasters start the shift as if their last batch was PSC. This means:
- First PSC batch on any roaster: **no setup** (same SKU)
- First NDG batch on R1 or R2: **5 min setup** required
- First Busta batch on R2: **5 min setup** required

**Rationale:** The factory runs PSC as the default product. Between shifts, roasters are left in "PSC-ready" state. Starting MTO requires switching.

**Impact on scheduling:** In a shift with 4 MTO batches (3 NDG + 1 Busta):
- R1 starts NDG: setup at [0, 5), first NDG at [5, 20). Loses 5 min.
- R2 starts Busta: setup at [0, 5), Busta at [5, 20). Loses 5 min.
- Alternative: R1 starts PSC first (no setup), then switches to NDG later. Trades early throughput for deferred setup cost.

The CP-SAT solver will find the optimal sequencing. The DRL agent must learn it. The dispatching heuristic needs a rule (see Item 5 below).

**For CP-SAT:** Add a "phantom" initial batch on each roaster: $\text{sku} = k^{PSC}$, $e_b = 0$, that participates in the NoOverlap + transition matrix. This forces setup when the first real batch differs from PSC.

---

## Item 4 ✅ RESOLVED — DRL Action Space Architecture: Per-Roaster

**Decision:** The DRL agent is called **per-roaster** at each decision point. When roaster $r$ becomes IDLE, the agent is called with $r$ as context and selects an action for **that roaster only**.

**If two roasters become IDLE at the same time slot:** The agent is called twice, sequentially. The second call sees the updated state (including the first call's action). Order: process roasters in fixed order R1 → R2 → R3 → R4 → R5.

**Justification:** Simultaneous IDLE events are rare (batch completion times differ by at least 3 slots due to pipeline staggering). The per-roaster design keeps the action space small (17 actions) and training tractable. A global action space would be $17^5$ in the worst case — intractable.

**Documented as design choice** in simulation_logic_complete.md §6.

---

## Item 5 ✅ RESOLVED — Complete Dispatching Heuristic Specification

This is the full, unambiguous specification of the dispatching baseline. Every edge case covered.

### 5.1 Overview

The dispatching heuristic represents **what a competent operator would do** using simple rules and no optimization. It is the **control baseline** — both CP-SAT and DRL must beat it to justify their complexity.

### 5.2 Decision Function

Called whenever roaster $r$ becomes IDLE at time $t$:

```
DISPATCH(r, t, state) → action

Step 1: CAN I DO MTO?
  Let eligible_mto_jobs = [j for j in MTO_jobs 
                           if mto_remaining[j] > 0 
                           AND sku(j) ∈ eligible_skus[r]]
  
  If eligible_mto_jobs is not empty:
    # Check if there's enough time to complete MTO before due date
    # Account for setup time if needed
    setup_needed = (last_sku[r] != sku(j)) ? σ : 0
    latest_start = D_MTO - P = 240 - 15 = 225
    
    If t + setup_needed ≤ latest_start:
      # There's time. But should I do it NOW or defer?
      # Rule: do MTO if remaining MTO time pressure is high
      
      total_mto_time_needed = Σ over eligible_mto_jobs:
        mto_remaining[j] × P + (setup_needed_for_j ? σ : 0)
      
      time_available = D_MTO - t = 240 - t
      
      # MTO urgency ratio: if > 0.7, do MTO now
      urgency = total_mto_time_needed / time_available
      
      If urgency > 0.7 OR t + setup_needed + mto_remaining[j_best] × P > 230:
        → Select MTO job (see Step 1a)
        → Go to Step 3
      
      # Not urgent yet — check if RC stock needs attention
      → Fall through to Step 2
    
    Else:
      # Past the point where MTO can be completed — still try (soft constraint)
      If mto_remaining[j] > 0 AND t ≤ D_MTO:
        → Select MTO job (see Step 1a)
        → Go to Step 3
      Else:
        → Fall through to Step 2  # MTO deadline passed, focus on PSC

Step 1a: SELECT MTO JOB (when multiple MTO jobs are eligible)
  # Priority: whichever job has MORE remaining batches
  # Tie-break: Busta first (more constrained — only R2)
  j_best = max(eligible_mto_jobs, key=lambda j: (mto_remaining[j], sku(j) == BUS))
  → action = START sku(j_best) on r

Step 2: PSC SCHEDULING
  # Default action: produce PSC
  
  # Check overflow: will RC overflow if this batch completes?
  l_out = output_line(r)  # for R1,R2 → L1. For R4,R5 → L2. For R3 → see Step 2a.
  completion_time = t + (σ if last_sku[r] != PSC else 0) + P
  future_consumption = |{τ ∈ E[l_out] : t < τ ≤ completion_time}|
  future_completions = (count of in-progress PSC batches on l_out completing before completion_time)
  projected_rc = rc_stock[l_out] + future_completions - future_consumption + 1
  
  If projected_rc > max_buffer:
    → action = WAIT  # RC would overflow, wait for consumption to make room
  Else:
    → action = START PSC on r
    → Go to Step 3

Step 2a: R3 ROUTING (only when r = R3)
  # Route R3 output to whichever line has LOWER RC stock
  If rc_stock[L1] ≤ rc_stock[L2]:
    l_out = L1  (y_b = 1)
  Else:
    l_out = L2  (y_b = 0)
  
  # Then check overflow on l_out (same as Step 2)
  # If l_out would overflow, try the other line
  # If both would overflow: WAIT

Step 3: PIPELINE CHECK
  # Before executing the action, check pipeline availability
  l_pipe = pipe(r)
  setup_needed = (last_sku[r] != sku(action)) ? σ : 0
  
  If setup_needed > 0:
    # Need setup first. Enter SETUP state.
    # Pipeline doesn't need to be free during setup.
    → Execute: enter SETUP for σ minutes, then start batch when IDLE again
  Else:
    # No setup. Pipeline must be free NOW.
    If pipeline_busy[l_pipe] > 0:
      → action = WAIT  # try again next slot when pipeline might be free
    Else:
      → Execute: start batch immediately

Step 4: DOWNTIME CHECK
  start_time = t + setup_needed
  If [start_time, start_time + P - 1] ∩ D[r] ≠ ∅:
    → action = WAIT  # can't fit batch before downtime
```

### 5.3 Worked Example — Dispatching Decisions

```
Shift: 3 NDG + 1 Busta. ρ_L1 = 5.1, ρ_L2 = 4.8.
Initial: all roasters last_sku = PSC. RC: L1=12, L2=15.

t=0: R1 becomes IDLE (shift start).
  Step 1: eligible MTO = [j1(NDG,3)]. R1 can do NDG.
    setup_needed = 5 (PSC→NDG). latest_start = 225.
    t + setup = 0 + 5 = 5 ≤ 225 → time OK.
    urgency = (3×15 + 5) / (240 - 0) = 50/240 = 0.21 → not urgent.
    Fall through to Step 2.
  Step 2: PSC on R1. No setup (last_sku=PSC). No overflow risk (RC=12 << 40).
  Step 3: Pipeline L1 free? Yes.
  → ACTION: Start PSC on R1 at t=0.

t=0: R2 becomes IDLE (shift start).
  Step 1: eligible MTO = [j1(NDG,3), j2(Busta,1)]. R2 can do both.
    setup_needed = 5 for either (PSC→NDG or PSC→Busta).
    urgency = (3×15 + 5 + 1×15 + 5) / 240 = 70/240 = 0.29 → not urgent.
    Fall through to Step 2.
  Step 2: PSC on R2. No setup. 
  Step 3: Pipeline L1 busy (R1 consuming [0,2]). 
  → ACTION: WAIT.

t=3: R2 still IDLE, pipeline L1 now free.
  Same evaluation → Start PSC on R2 at t=3.

t=15: R1 completes PSC. IDLE.
  Step 1: eligible MTO = [j1(NDG,3)]. urgency = 50/225 = 0.22 → not urgent.
  Step 2: Start PSC on R1. Pipeline check → may need to wait for R2's next consume.
  → Start PSC on R1 (when pipeline free).

... (PSC continues on both R1, R2 until urgency threshold hit) ...

t=120: R1 completes PSC. IDLE.
  Step 1: eligible MTO = [j1(NDG,3)]. 
    urgency = (3×15 + 5) / (240 - 120) = 50/120 = 0.42 → not urgent yet.
  Step 2: PSC.

t=160: R1 completes PSC. IDLE.
  Step 1: eligible MTO = [j1(NDG,3)].
    urgency = 50 / (240 - 160) = 50/80 = 0.625 → not urgent yet.
  Step 2: PSC.

t=175: R1 completes PSC. IDLE.
  Step 1: eligible MTO = [j1(NDG,3)].
    urgency = 50 / (240 - 175) = 50/65 = 0.77 → URGENT! > 0.7
  → Step 1a: select j1 (NDG, 3 remaining). 
  → Step 3: setup needed (PSC→NDG, 5 min). Enter SETUP [175, 180).
  
t=180: R1 SETUP complete. IDLE.
  → Start NDG batch. Pipeline check → start when free.
  → NDG batch [180+wait, 195+wait]. Then NDG [195+, 210+]. Then NDG [210+, 225+].
  → 3 NDG batches complete by ~slot 225. Tardiness = 0. ✓

t=175: R2 still doing PSC. Eventually R2 will pick up Busta.
  At some point urgency for j2(Busta,1) hits threshold.
  j2 needs: 1×15 + 5 = 20 min. With 65 min left at t=175, urgency = 20/65 = 0.31.
  At t=210: urgency = 20/30 = 0.67 → not yet.
  At t=215: urgency = 20/25 = 0.80 → URGENT.
  → R2 enters SETUP [215, 220). Start Busta at 220. Completes at 235. Tardiness = 0. ✓
```

### 5.4 Dispatching vs. Optimal: Where Dispatching Loses

The dispatching heuristic produces **reasonable** schedules but has systematic weaknesses:

1. **Myopic R3 routing:** Routes R3 to the currently-lower line. But CP-SAT/DRL might route R3 to the currently-higher line if they foresee that line's stock will drop faster (e.g., Line 2 has higher consumption rate).

2. **Late MTO scheduling:** The urgency threshold (0.7) means MTO batches are deferred until they're "urgent." This is safe but wasteful — it concentrates MTO batches in a narrow window, creating setup time clusters. CP-SAT can spread MTO batches more efficiently (e.g., do NDG first on R1 while R2 does PSC, eliminating the late-shift setup crunch).

3. **No pipeline coordination:** Dispatching doesn't coordinate pipeline usage across roasters. If R1 and R2 both become IDLE at the same slot, dispatching processes them sequentially — the second roaster always waits 3 min. CP-SAT can stagger batch starts by 3 slots to eliminate pipeline contention entirely.

4. **No UPS anticipation:** After a UPS, dispatching applies the same rules as before — no adaptation to the changed state beyond the immediate effect. DRL can learn post-UPS patterns (e.g., "after R4 goes down, shift R3 output to L2 preemptively").

---

## Item 6 ✅ RESOLVED — R3 Routing in DRL Action Space

**Decision:** Bake R3 routing into the action space.

**Revised action space (17 actions):**

| ID | Action | Eligibility |
|----|--------|-------------|
| 0 | PSC on R1 → L1 | R1 eligible |
| 1 | PSC on R2 → L1 | R2 eligible |
| 2 | PSC on R3 → **L1** | R3 eligible ($y_b = 1$) |
| 3 | PSC on R3 → **L2** | R3 eligible ($y_b = 0$) |
| 4 | PSC on R4 → L2 | R4 eligible |
| 5 | PSC on R5 → L2 | R5 eligible |
| 6 | NDG on R1 | R1 eligible, NDG remaining |
| 7 | NDG on R2 | R2 eligible, NDG remaining |
| 8 | Busta on R2 | R2 eligible, Busta remaining |
| 9–15 | *(permanently masked: NDG on R3-R5, Busta on R1/R3-R5)* | Never valid |
| 16 | WAIT | Always valid |

**Changes from old 16-action space:**
- Old action 2 ("PSC on R3") split into action 2 ("PSC on R3 → L1") and action 3 ("PSC on R3 → L2")
- Old actions 3, 4 (PSC on R4, R5) shifted to IDs 4, 5
- Old actions 5–14 (NDG/Busta variants) shifted to IDs 6–15
- 7 actions are permanently masked (ineligible SKU-roaster combos)
- Effective valid actions at any decision point: typically 2–4 (roaster-specific)

**Impact on DRL observation:** No change to observation vector (21 features). The R3 routing decision is now embedded in the action, not a separate step.

**Impact on dispatching:** No change — dispatching uses the lowest-stock rule (Step 2a) independently of the action space encoding.

**Impact on CP-SAT:** No change — $y_b$ is already a decision variable in the model.

---

## Item 7 ✅ RESOLVED — CP-SAT Re-Solve Trigger: UPS Only

**Explicit design choice:** CP-SAT re-solves **only** when a UPS event occurs. No periodic re-planning, no stock-level triggers.

**Implications:**
- At $\lambda = 0$ (no UPS): CP-SAT solves once at shift start and the initial schedule runs to completion without any re-solve. This is the "perfect information" baseline.
- At $\lambda = 5$: CP-SAT may re-solve 5+ times per shift (once per UPS, possibly more if UPS events cluster).
- The DRL agent, by contrast, makes a decision at **every decision point** (every time any roaster becomes IDLE) — even without UPS. This is a structural difference between the strategies.
- Dispatching also decides at every decision point, making it more comparable to DRL in decision frequency.

**Why not periodic re-solve?** The model is small enough (~200 variables) that re-solving frequently wouldn't be computationally expensive. But periodic re-solve would blur the comparison: is CP-SAT better because of optimization quality or because it re-plans more often? By limiting re-solve to UPS events, we isolate the effect of optimization quality vs. learned policy quality.

**Documented as explicit design choice.** Could be extended to periodic re-solve as future work.

---

## Item 8 ✅ RESOLVED — End-of-Shift Constraint

**Rule:** No batch may start after slot **465**.

$$s_b \leq 480 - P = 480 - 15 = 465 \quad \forall b : a_b = 1$$

A batch starting at slot 465 ends at slot 480 — exactly at shift end. A batch starting at 466 would end at 481 — past the shift boundary and should not be counted.

**In CP-SAT:** Set the domain of $s_b$ to $[0, 465]$.

**In simulation:** Action mask checks `t + (setup if needed) + P ≤ 480`. If the batch can't finish by slot 480, mask it out.

**In dispatching:** Same check — if `t + setup + P > 480`, WAIT (which effectively means "shift is over for this roaster").

**Edge case — batch in progress at t=479:** If a batch started at slot 465, it's RUNNING with remaining > 0 at t=479. The simulation should let it complete (it finishes "at" t=480, which is the shift boundary). RC is credited. This is consistent with $e_b = 465 + 15 = 480 \leq 480$.

---

## Item 9 — DRL Training Hyperparameters (Deferred)

Not blocking. Will be decided during implementation. Starting points from SB3 documentation:

| Parameter | Starting value | Rationale |
|-----------|---------------|-----------|
| Learning rate | 3e-4 | SB3 default for PPO |
| Batch size | 64 | Small environment, small network |
| n_steps | 2048 | Steps before each PPO update |
| n_epochs | 10 | PPO epochs per update |
| gamma | 1.0 | Finite horizon, no discounting |
| ent_coef | 0.01 | Encourage exploration |
| Network | MlpPolicy [64, 64] | 2 hidden layers, sufficient for 21-dim observation |
| Total timesteps | 500,000–1,000,000 | ~10,000–20,000 episodes |
| Training UPS distribution | Uniform mix of $\lambda \in \{1,2,3,5\}$ | Train on variety, not single intensity |
| R3 mode during training | Flexible (decision variable) | Train the more general agent; can fix R3 at eval time |

**Two agents or one?** Train ONE agent with flexible R3. At evaluation time, for the "fixed R3" experimental condition, override R3 actions by always selecting "PSC on R3 → L2" (action 3) — or re-train a second agent with R3 actions masked to L2 only. The first approach is simpler; the second is cleaner. Decide during implementation.

---

## Item 10 — Statistical Analysis Plan (Deferred)

Not blocking. Preliminary plan:

**Primary analysis:** For each of the 90 cells, compute mean and standard deviation of each KPI across 100 replications. Present as:
- **Heatmap:** rows = ($\lambda$, $\mu$) pairs (15 rows), columns = strategies (3), cell color = mean throughput. Separate heatmaps for fixed vs. flexible R3.
- **Line plots:** Throughput vs. $\lambda$ at each $\mu$ level, one line per strategy. Shows degradation curves.
- **Paired difference tests:** For each cell, compute $\Delta_{CP-SAT} = \text{throughput}_{CP-SAT} - \text{throughput}_{Dispatching}$ and $\Delta_{DRL} = \text{throughput}_{DRL} - \text{throughput}_{Dispatching}$ across the 100 paired replications. Report mean difference with 95% confidence interval.
- **Wilcoxon signed-rank test** per cell (non-parametric, doesn't assume normality) to test if the strategy difference is statistically significant.

**Secondary analysis:**
- R3 routing value: compare flexible vs. fixed across all cells. Is the throughput gain from flexibility larger under high $\lambda$?
- Computation time: boxplot of CP-SAT re-solve time vs. DRL inference time.
- Stockout analysis: in which cells does each strategy first encounter stockout? How does stockout severity grow with $\lambda$?

---

## Item 11 — Literature Search (Still Open)

⚠️ **Still blocks Chapter 2 writing.** Not blocking implementation.

**Minimum references needed before Chapter 2:**
- 2 reactive scheduling surveys (Pillar 2)
- 2 RL for scheduling papers (Pillar 3)
- 1 CP for scheduling (Pillar 4 — Naderi et al. 2023 likely works)
- Bektur & Saraç 2019 (Pillar 1 — already confirmed)

**Recommended action:** Use the paper-scout skill in a dedicated session after instructor approval.

---

## Item 12 ✅ RESOLVED — Cross-File Consistency Correction Sheet (Non-Blocking but Apply Before Final Thesis Draft)

The following corrections are **not blocking implementation**, but they should be applied so that `cost.md`, `mathematical_model_complete.md`, and `Thesis_Problem_Description_v2.md` stop speaking in slightly different dialects. Tiny contradiction goblins love exactly this kind of gap.

| Issue | Exact sentence/formula to replace | Recommended final version | Where to apply |
|---|---|---|---|
| **12.1 Stockout penalty mixes event-based and duration-based definitions** | In `cost.md`, replace `- \text{stockout}_{l,t} = \max(0, -B_l(t)) — deficit at consumption events (soft)` | `- \text{stockout}_{l,t} \in \{0,1\} : 1 iff line l is in stockout at minute t, i.e., B_l(t) < 0; 0 otherwise. The reactive objective penalizes stockout duration (minutes), while stockout count at consumption events is reported separately as a KPI.` | `cost.md` reactive objective bullet list |
|  | In `cost.md` reward pseudocode, replace `if new_state.rc_stock[l] < 0 and t in E[l]: reward -= 1500` | `if new_state.rc_stock[l] < 0: reward -= 1500` | `cost.md` DRL reward pseudocode |
|  | In `mathematical_model_complete.md`, replace `\text{stockout}_{l,t} = \max(0, -B_l(t)) \quad \forall l, t` | `\text{stockout}_{l,t} \in \{0,1\}, \quad \text{stockout}_{l,t}=1 \iff B_l(t) < 0 \quad \forall l,t` | `mathematical_model_complete.md` §4.1 Stockout Variable |
|  | In `Thesis_Problem_Description_v2.md`, replace `w^{stock}: penalty weight per stockout event-minute.` | `c^{stock}: penalty cost per stockout minute. The solver penalizes stockout duration over time; stockout count at consumption events is reported separately as an evaluation KPI.` | `Thesis_Problem_Description_v2.md` around the reactive objective explanation |
| **12.2 Deterministic objective is inconsistent about overflow-idle** | In `cost.md`, replace `- No overstock penalty ($c^{over}$) — overflow is a hard constraint ($B_l(t) \leq 40$), enforced by not allowing batch starts that would overflow. The resulting idle time falls under safety-idle if RC is also below threshold (unlikely — overflow and understock don't coexist), otherwise it's just unpenalized idle` | `- Overflow remains a hard constraint ($B_l(t) \leq 40$), but overflow-idle cost ($c^{over}$) is still active in deterministic mode. It penalizes roaster minutes lost because the assigned output buffer is full, allowing the model to distinguish between otherwise feasible schedules that create unnecessary blockage.` | `cost.md` deterministic-mode bullets |
|  | In `Thesis_Problem_Description_v2.md`, replace `\text{Maximize:} \quad \underbrace{\sum_{b \in \mathcal{B}^{PSC}} a_b}_{\text{PSC throughput}} \;\;-\;\; w^{tard} \cdot \underbrace{\sum_{j \in \mathcal{J}^{MTO}} \text{tard}_j}_{\text{MTO tardiness}}` | `\text{Maximize:} \quad \underbrace{\sum_{b: a_b=1} R_{\text{sku}(b)}}_{\text{Revenue}} \; - \; \underbrace{c^{tard} \sum_j \text{tard}_j}_{\text{Tardiness}} \; - \; \underbrace{c^{idle} \sum_{r,t} \text{idle}_{r,t}}_{\text{Safety-Idle}} \; - \; \underbrace{c^{over} \sum_{r,t} \text{over}_{r,t}}_{\text{Overflow-Idle}}` | `Thesis_Problem_Description_v2.md` deterministic objective section |
| **12.3 R3 overflow-idle is described correctly in prose but not fully encoded in the math** | In `mathematical_model_complete.md`, replace the generic C15 block beginning with `\text{over}_{r,t} \geq (1 - \text{busy}_{r,t}) + \text{full}_{l,t} - 1` | `For r \in \{R1,R2\}: \quad \text{over}_{r,t} \geq (1-\text{busy}_{r,t}) + \text{full}_{L1,t} - 1\n\nFor r \in \{R4,R5\}: \quad \text{over}_{r,t} \geq (1-\text{busy}_{r,t}) + \text{full}_{L2,t} - 1\n\nFor r = R3: \quad \text{over}_{R3,t} \geq (1-\text{busy}_{R3,t}) + \text{full}_{L1,t} + \text{full}_{L2,t} - 2` | `mathematical_model_complete.md` C15 |
|  | In `cost.md` reward pseudocode, replace `if new_state.status[r] == IDLE and new_state.rc_stock[output_line(r)] >= 40: reward -= 50` | `if r in [R1, R2] and new_state.status[r] == IDLE and new_state.rc_stock[L1] >= 40: reward -= 50\nif r in [R4, R5] and new_state.status[r] == IDLE and new_state.rc_stock[L2] >= 40: reward -= 50\nif r == R3 and new_state.status[r] == IDLE and new_state.rc_stock[L1] >= 40 and new_state.rc_stock[L2] >= 40: reward -= 50` | `cost.md` DRL reward pseudocode |
| **12.4 KPI layer should explicitly separate optimized quantity from reported KPI** | In `Thesis_Problem_Description_v2.md`, replace KPI rows `Stockout count` and `Stockout duration` | `Stockout count = number of consumption events \tau \in E_l such that B_l(\tau) < 0` and `Stockout duration = total minutes with B_l(t) < 0 across both lines`. Add one sentence below the table: `The optimization objective penalizes stockout duration; stockout count is reported only as a secondary KPI.` | `Thesis_Problem_Description_v2.md` §10.3 KPIs |
| **12.5 Retire old weight-based notation in narrative and appendices** | Replace any remaining narrative phrases like `w^{stock}=10`, `w^{tard}=5`, or `maximize throughput minus weighted penalties` | `Use the cost-based notation consistently: c^{stock}, c^{tard}, c^{idle}, c^{over}, and the corresponding profit objective from cost.md.` | All thesis drafts, slides, notes, and appendices |

**Implementation impact:** None of the five corrections above blocks coding. They are specification-cleanup edits so the written thesis and the implemented environment tell the same story.

---

## Master Resolution Summary

| # | Item | Resolution | Status |
|---|------|-----------|--------|
| 1 | RC max_buffer | $\overline{B}_l = 40$ batches per line | ✅ |
| 2 | $c^{stock}$ | $c^{stock} = \$1{,}500$/min/line (cost-based objective) | ✅ |
| 3 | Initial last_sku | PSC for all roasters (setup needed for first MTO batch) | ✅ |
| 4 | DRL action architecture | Per-roaster, sequential if simultaneous IDLE, fixed order R1→R5 | ✅ |
| 5 | Dispatching heuristic | Fully specified: urgency threshold 0.7, most-remaining MTO priority, lowest-stock R3 routing, overflow/downtime/pipeline checks | ✅ |
| 6 | R3 routing in DRL | Baked into action space: 17 actions (PSC-R3 split into →L1 and →L2) | ✅ |
| 7 | CP-SAT re-solve trigger | UPS-only (explicit design choice, not periodic) | ✅ |
| 8 | End-of-shift | $s_b \leq 465$, batch at 465 completes at 480 (shift boundary) | ✅ |
| 9 | DRL hyperparameters | Deferred — starting values documented | 🔵 Later |
| 10 | Statistical analysis | Deferred — preliminary plan documented | 🔵 Later |
| 11 | Literature search | Still open — blocks Ch 2 only | ⚠️ Open |
| 12 | Cross-file consistency patch | Exact replacements documented for stockout, overflow-idle, R3 C15, KPI wording, and notation cleanup | ✅ |

**Zero blocking items remain. Implementation can begin.**
