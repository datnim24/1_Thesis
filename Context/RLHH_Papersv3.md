# Reinforcement Learning-Based Hyper-Heuristics for Production Scheduling: Literature Review

> **Version:** 3.0 (April 2026)
> **Previous:** v2 (March 2026)
> **Purpose:** Canonical Q1-verified literature foundation for the RL-HH method in the Nestlé Trị An batch roasting thesis.

---

## CHANGELOG v2 → v3

| # | Change | Reason |
|---|---|---|
| 1 | **Paeng et al. (2021), IEEE Access promoted to primary key reference (Core Paper #1)** | Problem-structure match is structurally identical: UPMSP with sequence-dependent family setups, total-tardiness objective minimized by DDQN with parameter sharing, with explicit Q-Learning baseline (LBF-Q from Zhang 2007) that the deep method beats by ~85%. This is the closest published precedent to our problem class. |
| 2 | **Ren & Liu (2024) demoted to supporting reference** | Retained as architectural justification for Dueling DDQN over standard DDQN (D3QN: 0.85 > DDQN: 0.80 > DQN: 0.70 on-time completion). The thesis innovation = one architectural step beyond Paeng (dueling + tool-selection), motivated by Ren & Liu's hierarchy. But Ren & Liu's problem (DFJSP with multi-operation jobs and AGV transport) is structurally further from ours than Paeng's. |
| 3 | **Luo (2020), Applied Soft Computing promoted from contextual to supporting reference** | Provides the RL-HH paradigm precedent (DQN selecting from 6 composite dispatching rules) that our 5-tool action space inherits. Also provides the tabular Q-Learning baseline experimental protocol (SOM-discretization, 38/45 instance win) that we replicate in Phase 4. |
| 4 | **MaskedPPO references removed throughout** | PPO/MaskedPPO no longer in thesis scope after gradient death was observed. The PPO-related references (Lassoued, Hubbs) become Chapter 6 future-work citations, not method-defense citations. |
| 5 | **Lassoued et al. (2026) status unchanged** | Still arXiv preprint, still NOT Q1. Still a single-citation supplementary reference, but no longer needed to defend "PPO baseline" since PPO is dropped. |
| 6 | **Q1 verification table updated** | Paeng now ★1; Ren & Liu now ★2 (supporting); Panzer demoted from ★ to supporting (still cited but no longer co-primary). |

---

## Q1 Verification Summary (All Core References)

| # | Paper | Year | Journal | Scimago Q | Role in thesis |
|---|---|---|---|---|---|
| ★1 | **Paeng, Park, Park** | 2021 | IEEE Access | **Q1** | **Primary key reference** — UPMSP-SDFST architecture, parameter-sharing DDQN, LBF-Q baseline experimental design. The closest published problem-structure match to ours. |
| ★2 | **Ren & Liu** | 2024 | Scientific Reports | **Q1** | **Supporting key reference** — D3QN architecture justification, grouped network design, dueling-architecture motivation. The "one architectural step beyond Paeng" comes from this paper. |
| ★3 | **Luo, S.** | 2020 | Applied Soft Computing | **Q1** | **Supporting key reference** — RL-HH dispatching-rule paradigm precedent (6 CDRs). Tabular Q-Learning baseline experimental protocol (Section 6.4 / Table 8). State-feature normalization template ([0,1] features). |
| 4 | Panzer, Bender, Gronau | 2024 | International Journal of Production Research | **Q1** | DRL-HH for production control with disruptions — cite for tool-based action space defense and multi-objective reward template |
| 5 | Zhang Y., Bai, Qu, Tu, Jin | 2022 | European Journal of Operational Research | **Q1** | DDQN-HH under uncertainty — methodological support |
| 6 | Karimi-Mamaghan et al. | 2023 | European Journal of Operational Research | **Q1** | Q-Learning-HH framework — methodology lineage |
| 7 | Zhang Z.-Q. et al. | 2023 | Expert Systems with Applications | **Q1** | Q-Learning-HH for FJSP — supplementary |
| 8 | **Zhang, Z., Zheng, L., Weng, M. X.** | 2007 | International Journal of Advanced Manufacturing Technology | Q1/Q2 | **Original tabular Q-Learning baseline** for parallel machine scheduling. Cited as the lineage origin of our tabular Q-Learning Phase 4. Both Paeng (2021) LBF-Q and Luo (2020) SOM-Q-Learning trace to this paper. |
| ⚪ | Lassoued et al. | 2026 | arXiv preprint | **NOT Q1** | Contextual only; PPO-HH paradigm reference (now Chapter 6 future-work cite) |
| – | Li, C. et al. | 2024 | PeerJ Computer Science | Q2 | Survey — taxonomy citation |
| – | Cheng et al. | 2022 | Swarm and Evolutionary Computation | **Q1** | Multi-objective Q-learning HH — supplementary |
| – | Gui et al. | 2023 | Computers & Industrial Engineering | **Q1** | DDPG with continuous rule weights — supplementary |
| ◇ | Hubbs et al. | 2020 | Computers & Chemical Engineering | **Q1** | End-to-end DRL for batch chemical (Chapter 6 future work — links to abandoned MaskedPPO direction) |
| ◇ | Hu et al. | 2020 | Journal of Manufacturing Systems | **Q1** | DRL + Petri net + GCN (alternative state representation, future work) |

★ = Primary key reference for thesis defense
⚪ = Contextual only, NOT Q1
◇ = Adjacent literature (not RL-HH but topically connected)

---

## Field Context and Survey

**Li, C., Wei, X., Wang, J., Wang, S., & Zhang, S. (2024).** A review of reinforcement learning based hyper-heuristics. *PeerJ Computer Science*, 10, e2141. https://doi.org/10.7717/peerj-cs.2141

This 2024 survey represents the **first comprehensive review** of reinforcement learning based hyper-heuristics, systematically analyzing over 80 representative papers from six major academic databases. The authors present a hierarchical two-tier taxonomy dividing RL-HH into **Value-Based approaches** (subdivided into Traditional RL-HH using Q-learning/MAB methods, and Deep RL-HH using DQN/DDQN/D3QN) and **Policy-Based approaches** (using PPO/DPPO). For production scheduling specifically, the survey documents that Q-learning-based approaches have proven particularly effective, with multiple successful implementations across workshop scheduling, semiconductor manufacturing, and mixed shop problems. The review identifies key advantages of RL-HH: no requirement for deep prior domain knowledge, robust generalization capabilities, and automatic adaptation to dynamic environments through learning. Current limitations include lack of theoretical convergence analysis, high computational costs, and limited integration of value-based and policy-based strengths. This survey establishes RL-HH as a mature field with substantial production scheduling applications from 2020-2024.

**Why this survey matters for the thesis:** Table 1 of Li et al. (2024) classifies the value-based DRL-HH lineage as DQN → DDQN → D3QN (Dueling Double DQN). Our Dueling DDQN meta-agent sits in the D3QN class — the survey's frontier category. The survey also explicitly defends value-based DRL over policy-based PPO for hyper-heuristic control: *"PRL... requires more samples to learn effective policies, making them inferior to value-based methods in terms of sample efficiency. Additionally, PRL is also more susceptible to training instabilities."* This justifies our choice to use Dueling DDQN throughout — both for the Paeng-style DDQN baseline (Phase 5) and for the RL-HH meta-agent (Phase 6) — rather than a policy-gradient method.

---

## Core Papers: Detailed Analysis

### ★1. Paeng, Park, Park (2021) — Modified DDQN with Parameter Sharing for UPMSP-SDFST — **PRIMARY KEY REFERENCE**

**Paeng, B., Park, I.-B., & Park, J. (2021).** Deep Reinforcement Learning for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups. *IEEE Access*, 9, 101390–101401. https://doi.org/10.1109/ACCESS.2021.3097254

**Q1 Verification:** IEEE Access — Q1 (Scimago 2024) in Computer Science (miscellaneous), Engineering (miscellaneous), Materials Science (miscellaneous). Impact Factor 3.9.

**Problem Type:** Unrelated Parallel Machine Scheduling Problem (UPMSP) with **Sequence-Dependent Family Setup Times (SDFST)**, minimizing total tardiness. Real-world context: semiconductor wafer preparation in South Korea. The problem: N_J jobs to be allocated across N_M unrelated parallel machines, each job belonging to one of N_F families with its own due-date. When a job of family f is assigned to a machine with current setup status g (where f ≠ g), a sequence-dependent family setup time σ_{f,g} is incurred before processing.

**RL Algorithm:** Double Deep Q-Network (DDQN) trained with target network synchronization and Huber loss. Standard DDQN architecture (NOT dueling) but with a novel **parameter-sharing network** to handle dimension-invariance.

**Key innovation — Dimension-invariant state representation:** Paeng's headline contribution is a state representation whose dimension is independent of N_J (job count) and N_M (machine count). Achieved through:
- **Family-based binning:** state has N_F rows (one per family), not N_J rows (one per job)
- **Time-bucket binning:** column n of waiting-job matrix S_w counts waiting jobs whose due-date proximity falls in bucket n, capped at H_w buckets (optimal H_w = 6 found via sensitivity sweep)
- **Parameter sharing across families:** one shared 3-layer block (64 → 32 → 16 nodes, ReLU) processes each family's row, drastically reducing network parameters

This means a single trained DDQN can be deployed across scheduling problems with different N_J and different due-date distributions **without re-training** — a property critical for real industrial deployment.

**State Representation (5 matrices + 1 vector):**
- **S_w** (N_F × 2H_w): Waiting job status, binned by due-date proximity
- **S_p** (N_F × H_p): In-progress job status, binned by remaining processing time
- **S_s** (N_F × N_F): Setup time matrix (σ_{f,g} for feasible actions; σ_max otherwise)
- **S_u** (N_F × 3): Utilization history per family — total processing time, setup time, finished count
- **S_a** (N_F × 2): Last action (one-hot encoded for both family and setup status)
- **S_f** (3): Last reward, period index, terminal flag

**Action Space:** Tuple **(f, g)** = (family of job to assign, machine setup status to assign on). Action space size = N_F × N_F (10 × 10 = 100 in their largest setup), independent of N_J and N_M.

**Reward Function:** Clipped tardiness reward computed per period (T = (2/3)·p̄ for small datasets, (1/2)·p̄ for large datasets). The total sum of rewards across all periods equals the total tardiness TT — a clean theoretical property the authors prove.

**Network Architecture (Figure 2):** Parameter-sharing fully-connected network. Input matrix → N_F parallel shared blocks (3 hidden layers, 64 → 32 → 16, ReLU) → concatenate all block outputs with system-level vector S_f → final fully-connected layer → Q-values for all N_F × N_F actions.

**Key Results (Table 3, the result we will replicate on our problem class):**

Compared against 7 alternative methods on 8 datasets (each with 300 training problems + 30 held-out validation problems, production requirements perturbed by ≥37%):

| Dataset | LBF-Q (tabular Q-Learning baseline) | IG (metaheuristic, 1h budget) | Paeng's DDQN |
|---|---|---|---|
| 1 | 9.079 | 1.744 | **1.243** |
| 2 | 11.339 | 2.867 | **2.490** |
| 3 | 10.191 | 1.585 | **1.196** |
| 4 | 11.340 | 2.719 | **2.404** |
| 5 | 8.763 | 2.440 | **1.796** |
| 6 | 9.527 | 3.520 | **2.934** |
| 7 | 7.322 | 2.440 | **1.572** |
| 8 | 10.486 | 3.669 | **2.724** |

- **Paeng's DDQN wins on every single dataset (8/8)**
- **Beats LBF-Q (Zhang 2007 lineage tabular Q-Learning) by ~85% on average** — this is the result we expect to replicate on our problem class
- **Beats IG (1-hour-budget metaheuristic) by 13–55%**
- **Computation time:** 3.26–17.6 seconds per problem (vs. IG's 3600s) — fast enough for real-time deployment

**Robustness Test (Table 5):** Tested under stochastic processing/setup times (Uniform[0.8x, 1.2x] noise on each parameter). Paeng's DDQN achieved both lowest mean AND lowest standard deviation across 30 random seeds. This is the published precedent for our paired-seed UPS robustness testing in Block B.

**Why this is the PRIMARY KEY REFERENCE for our thesis:**

| Component | Our `paeng_ddqn/` design | Paeng et al. (2021) |
|---|---|---|
| Topology | Unrelated Parallel Machines (5 roasters, 2 lines) | Unrelated Parallel Machines |
| Setups | Sequence-dependent SKU setups (5 min + $800) | Sequence-dependent family setups (σ_{f,g}) |
| State representation | Family-based binning, dimension-invariant | Family-based binning, dimension-invariant |
| Network | Parameter-sharing fully-connected blocks | Parameter-sharing fully-connected blocks (3 hidden layers, 64-32-16) |
| Action space | (SKU, target_pool) tuple | (family, setup_status) tuple |
| Training algorithm | DDQN + target sync + Huber loss + RMSProp | DDQN + target sync + Huber loss + RMSProp |
| Q-Learning baseline | Tabular Q-Learning (this thesis Phase 4) | LBF-Q (Zhang 2007) |
| Disruption test | UPS events (paired seeds) | Stochastic processing/setup times (paired seeds, Table 5 robustness) |

The first six rows are direct architectural inheritance. The disruption-test row shows methodological inheritance with adaptation — Paeng tests stochastic noise on processing times; we test discrete UPS events that cancel running batches. Our Block B follows Paeng's paired-seed protocol exactly.

**Where Paeng falls short (= our contribution boundaries):**
1. **No shared bottleneck resource** — Paeng's machines are independent. Our shared GC pipeline is the central novelty.
2. **No mixed MTS/MTO demand** — Paeng has uniform job classes; we have PSC (MTS) + NDG/Busta (MTO).
3. **No buffer/inventory dynamics** — Paeng doesn't track RC buffers or GC silos.
4. **Standard DDQN, not dueling** — Paeng's DDQN does not separate state-value from action-advantage. The Dueling DDQN step (motivated by Ren & Liu 2024) is one architectural rung beyond Paeng.
5. **Direct (family, setup) action space, not tool-selection** — Paeng's action space is direct allocation. The tool-selection action space (RL-HH paradigm from Luo 2020 / Ren & Liu 2024) is a different abstraction level.

These five gaps define our contribution: we transfer Paeng's architectural framework to a more complex problem class (mixed demand + shared bottleneck + within-shift disruptions) and add the architectural step from Ren & Liu (2024) on top.

**Thesis Connection:**
- Chapter 1 (Introduction) — Cite as the closest published precedent to our problem class
- Chapter 2 (Literature Review) — Anchor for "DDQN for UPMSP-SDFST" subsection. Replicate their Table 1 RL-based dynamic scheduling comparison with our 4 reactive methods.
- Chapter 3 (Methodology) — Justifies every Modified DDQN architectural choice in Section 3.4.3; cite for parameter-sharing design and dimension-invariance philosophy
- Chapter 4 (Implementation) — Reference architecture for Section 4.7 (paeng_ddqn/)
- Chapter 5 (Discussion) — Replication of Paeng's headline result (~85% improvement over tabular Q-Learning under disruption) is Finding #2.1 in the Block B comparison

---

### ★2. Ren & Liu (2024) — D3QN with MachineRank for Dynamic Flexible Job Shop Scheduling — **SUPPORTING KEY REFERENCE (Architectural Justification for Dueling DDQN)**

**Ren, F., & Liu, H. (2024).** Dynamic scheduling for flexible job shop based on MachineRank algorithm and reinforcement learning. *Scientific Reports*, 14, 29741. https://doi.org/10.1038/s41598-024-79593-8

**Q1 Verification:** Scientific Reports — Q1 (Scimago 2024, Multidisciplinary). Published by Nature Portfolio. Impact Factor 3.9.

**Role in our thesis:** Justifies the architectural step from Paeng's standard DDQN to our Dueling DDQN RL-HH. Where Paeng provides the problem-class precedent, Ren & Liu provide the value-based DRL state-of-the-art (D3QN = Dueling Double DQN) and the empirical evidence that the dueling architecture beats standard DDQN on a comparable scheduling problem with disruptions.

**Problem Type:** Dynamic Flexible Job Shop Scheduling Problem (DFJSP) with new job insertions following Poisson arrivals, machine breakdowns, processing time changes, and AGV (Automated Guided Vehicle) transportation constraints. Objectives include minimizing maximum completion time (makespan) and improving on-time completion rates in highly dynamic manufacturing environments.

**RL Algorithm:** Dueling Double Deep Q-Network (D3QN), which reduces overestimation of Q-values compared to DQN/DDQN and improves training stability. D3QN separates state value V(s) and advantage A(s,a) estimation in the network architecture, combining them as Q(s,a) = V(s) + A(s,a) − mean(A) to produce final Q-values, enabling more robust learning in complex scheduling environments.

**Low-Level Heuristics:** The D3QN agent selects from **7 Composite Dispatching Rules (CDRs)** based on the novel **MachineRank (MR)** algorithm, calculated iteratively as MR(n) = a×M×MR(n−1) + (1−a)×MV. The 7 CDRs are: (1) CDR1 — Minimum Average Idle Time + Minimum MR; (2) CDR2 — Maximum Estimated Tardiness + Minimum MR; (3) CDR3 — Tardiness-aware + Completion Rate + Minimum MR; (4) CDR4 — Tardiness-aware + Relaxation Time Ratio + Minimum MR; (5) CDR5 — Weighted Tardiness-Completion + Minimum MR; (6) CDR6 — Random Job + Minimum MR; (7) CDR7 — Random Job + Earliest Available Machine. Each CDR integrates job selection logic with machine assignment strategy in a single integrated decision.

**State Representation:** Eight state features divided into machine-level and job-level groups, all normalized to [0,1]: average machine utilization rate, standard deviation of machine utilization, maximum difference in machine utilization, average job completion rate, standard deviation of job completion rate, average operation completion rate, estimated tardiness rate, and actual tardiness rate.

**Network Architecture (Fig. 1b):** **Grouped network**, not fully connected. Machine-level features and job-level features are processed in separate sub-networks for the first three hidden layers, then merged. This grouped architecture reduces parameter count and prevents cross-group feature interference, achieving faster convergence than fully connected networks.

**Reward Design:** Multi-component reward considering actual tardiness rate, MR values, and estimated job tardiness rate. Reward +1 if MR(sₜ) ≤ MR(sₜ₊₁) AND Tard_e(sₜ) > Tard_e(sₜ₊₁); Reward −1 if Tard_e(sₜ) ≤ Tard_e(sₜ₊₁); Reward +1 if Tard_a(sₜ) ≥ Tard_a(sₜ₊₁); Reward −1 if MR(sₜ) > MR(sₜ₊₁) AND Tard_a(sₜ) < Tard_a(sₜ₊₁); Otherwise reward = 0.

**Key Results — the architectural justification we cite:** D3QN converged to **0.85 on-time completion rate** compared to **0.80 for DDQN** and **0.70 for DQN**. This is the central empirical evidence that the dueling architecture step (V/A decomposition) yields a 5-percentage-point improvement over standard DDQN on a dynamic scheduling problem with disruptions. The approach significantly outperformed single scheduling rules (FIFO, EDD, MRT, SPT, LPT, Random) across all metrics. After machine breakdown rescheduling, the system recovered with only 6-second delay versus the original schedule.

**Why this is the SUPPORTING KEY REFERENCE for our thesis:**

| Component | Our `rl_hh/` design | Ren & Liu 2024 |
|---|---|---|
| Algorithm | Dueling DDQN | D3QN (= Dueling Double DQN) |
| Network | Grouped (Roaster / Inventory / Context) | Grouped (machine-level / job-level), Fig. 1(b) |
| State features | 33, all normalized [0,1] | 8, all normalized [0,1] |
| Action space | 5 dispatching tools | 7 composite dispatching rules (CDRs) |
| Disruption type | UPS (machine stoppages) | Machine breakdowns + new job insertions |
| Compared against | Tabular Q-Learning, Paeng DDQN, dispatching | DQN, DDQN, single rules |

This paper is the architectural blueprint for our RL-HH meta-agent specifically — the dueling V/A decomposition and the grouped network design come directly from Ren & Liu (2024) Figure 1(b). However, the **problem-class precedent** comes from Paeng (2021), not from Ren & Liu (2024), because Paeng's UPMSP-SDFST is structurally identical to ours while Ren & Liu's DFJSP (multi-operation jobs + AGVs) is structurally further away.

**The two-paper inheritance pattern:**
- **From Paeng (2021):** problem class (UPMSP-SDFST), parameter-sharing architecture, dimension-invariant state, DDQN training, paired-seed disruption testing
- **From Ren & Liu (2024):** dueling architecture (V/A decomposition), grouped network design, [0,1] feature normalization, RL-HH dispatching-rule action paradigm

Our innovation = these two inheritances combined and applied to a problem class with a shared bottleneck resource (GC pipeline) absent in both precedents.

**Thesis Connection:**
- Chapter 2 (Literature Review) — Anchor for "DRL-HH for dynamic scheduling under disruptions" subsection. Cite for D3QN > DDQN > DQN performance hierarchy.
- Chapter 3 (Methodology) — Justifies the dueling architectural step from Paeng's standard DDQN (Section 3.4.4)
- Chapter 4 (Implementation) — Reference architecture for `rl_hh/network.py`
- Chapter 5 (Discussion) — D3QN > DDQN > DQN convergence ordering provides the expected pattern; we test whether the analogous "RL-HH (Dueling DDQN) > Paeng (Standard DDQN) > Q-Learning (Tabular)" pattern holds on our problem class

---

### ★3. Luo, S. (2020) — DQN-HH with Composite Dispatching Rules for DFJSP — **SUPPORTING KEY REFERENCE (RL-HH Paradigm Precedent + Q-Learning Baseline Protocol)**

**Luo, S. (2020).** Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning. *Applied Soft Computing*, 91, 106208. https://doi.org/10.1016/j.asoc.2020.106208

**Q1 Verification:** Applied Soft Computing — Q1 (Scimago 2024) in Computer Science Applications, Software.

**Role in our thesis:** Two distinct contributions — (1) provides the **RL-HH dispatching-rule paradigm** that our 5-tool action space inherits, and (2) provides the **tabular Q-Learning baseline experimental protocol** (Section 6.4 / Table 8) that our Phase 4 replicates.

**Problem Type:** Dynamic Flexible Job Shop Scheduling Problem (DFJSP) with new job insertions following Poisson arrivals. Direct predecessor of Ren & Liu (2024) — same problem family, same architecture lineage (DQN → Double DQN → Dueling DDQN). Where Luo uses Double-DQN, Ren & Liu add the Dueling architecture.

**RL Algorithm:** DQN trained with Double DQN + soft target weight update (τ = 0.01). Standard fully-connected network: 7 input → 5 hidden layers × 30 nodes (tansig) → 6 output (purelin).

**Low-Level Heuristics — RL-HH paradigm precedent:** DQN selects from **6 Composite Dispatching Rules (CDRs)**. Each rule combines a job-selection criterion (slack-based, critical ratio, estimated tardiness, completion-rate weighted, random) with a machine-assignment criterion (earliest-available, lowest utilization, lowest workload, random). This is the **template our 5-tool RL-HH action space inherits**: each tool combines a job-selection logic with a machine-assignment logic.

**State Representation:** 7 generic features ∈ [0,1]^7:
1. U_ave: average machine utilization
2. U_std: std deviation of machine utilization
3. CRO_ave: average operation completion rate
4. CRJ_ave: average job completion rate
5. CRJ_std: std deviation of job completion rate
6. Tard_e: estimated tardiness rate
7. Tard_a: actual tardiness rate

**Action selection at deployment:** Softmax with entropy parameter μ = 1.6 (NOT argmax) — promotes diversity, avoids local optima. Sensitivity sweep μ from 1.0 to 2.0 in steps of 0.1 found μ = 1.6 optimal.

**Reward Function:** Hierarchical priority logic — actual tardiness > estimated tardiness > utilization. Reward +1/-1/0 based on consecutive-state deltas with a 5% tolerance band on U_ave.

**Key Results — Section 6.4 / Table 8 (the Q-Learning baseline protocol we replicate):**

Compared against tabular Q-Learning with **Self-Organizing Map (SOM) based 9-state discretization** of the same 7 features. Q-table: 9 states × 6 actions = 54 cells.

- **DQN beats tabular Q-Learning on 38/45 instances (84.4% win rate)** at DDT = 1.0
- Pairwise t-tests at 5% significance confirm dominance on most instances
- Author's interpretation: *"compulsive discretization of continuous state features is too rough... fails to accurately distinguish different production statuses of DFJSP"*

This 38/45 result is one of the two pieces of literature evidence (alongside Paeng 2021's 8/8 wins) that establish the expected pattern for our Block B comparison: deep value-based RL beats discretized-state tabular Q-Learning under disruption by large margins.

**Vs. composite rules (Tables 5–7):** DQN wins ~61.5% of instances on average. Strongest performance under tighter due dates.
**Vs. well-known rules (Tables 9–11):** DQN wins ~83.7% on average. At DDT = 1.5 (loose due dates, more realistic), DQN wins ~89% of instances.

**Why this is a SUPPORTING KEY REFERENCE for our thesis:**

| Element | Our adoption | Luo (2020) |
|---|---|---|
| RL-HH paradigm (DQN selects from CDRs) | 5 dispatching tools | 6 composite dispatching rules |
| State feature normalization | All 33 features ∈ [0,1] | All 7 features ∈ [0,1] |
| Tabular Q-Learning baseline experimental design | Phase 4 replicates Section 6.4 protocol | 9-state SOM discretization, 6-action Q-table, paired t-tests at 5% |
| Performance hierarchy expectation | Expected: Dueling DDQN > Paeng DDQN > Tabular Q-Learning | Empirical: DQN > Tabular Q-Learning on 38/45 (84.4%) |
| Composite-rule design pattern | Each tool = job-selection criterion + machine-assignment criterion | Each CDR = job-selection criterion + machine-assignment criterion |

**Thesis Connection:**
- Chapter 2 (Literature Review) — Cite for the RL-HH composite-rule paradigm origin and the tabular Q-Learning baseline lineage
- Chapter 3 (Methodology) — Reference for the [0,1] feature-normalization design philosophy and the composite-rule action space template
- Chapter 4 (Implementation) — Cite Section 6.4 protocol for Phase 4 (Tabular Q-Learning) experimental design
- Chapter 5 (Discussion) — DQN vs. tabular Q-Learning on 38/45 instances is the empirical pattern we expect to reproduce on our problem class

---

### 4. Panzer, Bender, Gronau (2024) — DQN Hyper-Heuristic for Modular Production Control — **Supporting reference (production-control context)**

**Panzer, M., Bender, B., & Gronau, N. (2024).** A deep reinforcement learning based hyper-heuristic for modular production control. *International Journal of Production Research*, 62(8), 2747-2768. https://doi.org/10.1080/00207543.2023.2233641

**Q1 Verification:** International Journal of Production Research — Q1 (Scimago 2024) in Industrial and Manufacturing Engineering, Management Science and Operations Research, Strategy and Management.

**Problem Type:** Semi-heterarchical modular production control in job shop-like environments with highly configurable products and reentrant flows. The multi-layered system includes shop-floor manufacturing agents controlling production operations and transport-layer distribution agents managing material flow between pre-defined modules. Challenges addressed include demand fluctuations, **unforeseen machine failures**, **rush orders**, prioritized order processing, and large control solution spaces due to flexible routing capabilities.

**RL Algorithm:** Deep Q-Network (DQN), value-based, model-free, off-policy. Initial training required ~36 hours for the first module, with re-training time of only ~4 hours when adding new modules (transfer learning capability).

**Low-Level Heuristics:** The DQN-HH selects from production dispatching rules including: **FIFO** (local + global variants), **SPT** (Shortest Processing Time), **EDD** (Earliest Due Date), and **HP** (Highest Priority). Each heuristic maps to a specific DQN action, providing quick execution, established production logic, comprehensibility (unlike pure neural decisions), and **scenario-independent action space that doesn't increase with layout size**.

**State Representation:** Min-max normalized features including job/order information (current job characteristics, queue lengths, waiting jobs, order priorities, urgency flags, due dates, remaining processing requirements), machine/station status, and system-level metrics (WIP levels, buffer states, inventory positions, queue information). Distribution agents in D1, D1.1, D1.2 deploy 88, 84, and 52 input neurons; manufacturing agents in M1.1.1 and M1.1.2 deploy 80 and 60 neurons.

**Action Space:** Discrete action space where each action corresponds to selection of one low-level dispatching rule. The action set is fixed in size and does not scale with production layout. Actions are triggered at specific events: new order arrivals, agent arrival at destination, system changes (machine failure, rush order insertion), agents with no assigned task, or machine idle with jobs waiting in queue.

**Reward Design:** Multi-objective reward targeting: (1) **throughput time minimization** (mean throughput time, flow time optimization), (2) **tardiness reduction** (minimizing late deliveries, adherence to due dates), and (3) **priority order processing** (rush orders, prioritized orders, order urgency consideration as customer-centric objective). The total reward is a composite of these criteria for the chosen order.

**Key Results:** Tested in discrete-event simulation and validated in a hybrid production environment at the Center for Industry 4.0. The hyper-heuristic achieved 12.7%–18.6% better throughput time than worst conventional heuristics (FIFO/FCFS as benchmark, plus SPT, EDD, STD, LWT, NV individually tested). Robustness analysis showed conventional rules exhibited significant reward fluctuations while the hyper-heuristic demonstrated more stable reward reception with lower variance across varying conditions. Real-world validation successfully transferred from simulation to physical system with autonomous mobile robots.

**Why this is a SUPPORTING reference for our thesis:**

Panzer et al. provide the production-control context anchor. Where Paeng (2021) provides the problem-class precedent and Ren & Liu (2024) provides the algorithm template, Panzer et al. provide the proof that DRL-HH actually works for **production scheduling under industrial disruptions** (machine failures, rush orders, demand changes) — not just abstract job-shop benchmarks. Three critical contributions to our defense:

1. **Action-space defense.** The paper states explicitly: *"due to the pre-defined set of dispatching rules, the action space does not increase with large layout sizes and there is no need to introduce masked actions for learning as the logic is mapped intrinsically"* (Section on action space design). This sentence directly defends our 5-tool RL-HH action space — tool-based action spaces stay small and explainable regardless of plant complexity, in contrast to direct allocation in Paeng.

2. **Multi-objective reward template.** Our incremental-profit reward (revenue − tardiness − stockout − setup penalties) is structurally a Panzer-style composite reward. We can cite Panzer to justify the multi-objective formulation rather than Hubbs's pure-profit approach.

3. **Industrial validation.** Panzer transferred their DQN-HH from simulation to a real Industry 4.0 testbed. This is the precedent we cite when arguing that Nestlé Trị An deployment is a realistic future-work step, not science fiction.

**Thesis Connection:**
- Chapter 1 (Introduction) — Cite for "DRL-HH has been applied to production control with disruptions" framing
- Chapter 2 (Literature Review) — Core reference for "DRL-HH in production scheduling" subsection
- Chapter 3 (Methodology) — Justifies tool-based action space, multi-objective reward design, situation-specific heuristic selection
- Chapter 5 (Discussion) — Industrial transferability framing; sim-to-real precedent

---

### 3. Zhang Y. et al. (2022) — DDQN-HH for Combinatorial Optimisation under Uncertainty

**Zhang, Y., Bai, R., Qu, R., Tu, C., & Jin, J. (2022).** A deep reinforcement learning based hyper-heuristic for combinatorial optimisation with uncertainties. *European Journal of Operational Research*, 300(2), 418-427. https://doi.org/10.1016/j.ejor.2021.10.032

**Q1 Verification:** European Journal of Operational Research — Q1 (Scimago 2024) in Management Science and Operations Research, Information Systems and Management, Modeling and Simulation. SJR 2.239, Impact Factor 7.40. Top-tier OR journal.

**Problem Type:** Online combinatorial optimization with uncertainty — parameters are revealed sequentially during decision-making rather than known in advance. Demonstrated on (1) real-world container terminal truck routing with uncertain crane service times and (2) online 2D strip packing.

**RL Algorithm:** Double Deep Q-Network (DDQN) with experience replay. Per Li et al. (2024) Table 1 taxonomy, this is the canonical DDQN-HH paper. Training stability is the explicit motivation for choosing DDQN over plain DQN.

**Low-Level Heuristics:** **10 parameter-controlled heuristics**, inspired by the manual operations heuristic of Chen et al. (2016). The heuristics differ across three dimensions (distance, work-queue imbalance, urgency), each with 3 threshold settings, yielding 3×3 = 9 parameterized heuristics plus 1 manual baseline.

**State Representation:** Concatenation of four state vectors: (1) remaining number of tasks per ship crane (length of work queue), (2) distance between current truck and source nodes of first tasks of every work queue, (3) **predicted number of trucks** to serve every ship crane (model-derived), and (4) **predicted number of tasks** to be finished per ship crane in the next 10 minutes (model-derived using moving average). The first two are explicit observable states; the latter two are model-derived predictions — a hybrid state representation philosophy.

**Action Space:** Two-level. The agent selects an *agent_action* index ∈ {0, 1, …, 9} corresponding to one of 10 LLHs. The selected LLH then executes the *heuristic_action* — assigning the current truck to a specific work queue.

**Reward Design:** Episodic reward computed retrospectively. The negative time-gap (crane idle time between consecutive tasks) is used as the per-decision reward, since the problem is to minimize aggregated crane waiting time. Rewards cannot be computed immediately at decision time because they depend on the actual completion of subsequent tasks.

**Key Results:** Achieved **7.39%–8.20% improvement** over manual baseline heuristics on real-world container-terminal data. Scalability tests (48–288 task instances) showed 2.9%–4.5% improvement maintained across problem sizes. Training was 3.7×–4.1× faster than end-to-end DRL approaches. Spectrum analysis of trained DDQN decisions demonstrated improved interpretability over conventional DRL.

**Key quote for the uncertainty-handling argument:** *"The solutions derived from these deterministic approaches can rapidly deteriorate during execution due to the over-optimisation without explicit consideration of the uncertainties... it is crucial to develop alternative methodologies that can accommodate uncertainties as well as make decisions that can sufficiently balance the conflicts between solution optimality and its resilience to unpredictable (sometimes even disruptive) changes."*

**Why this paper supports our thesis:**

This paper replaces the role Lassoued (2026) was meant to play — it is a Q1-published DDQN-HH paper with explicit uncertainty handling. Three direct connections to our work:

1. **Uncertainty handling.** Zhang Y. et al. solve combinatorial optimization where parameters arrive stochastically. This is the cleanest published analog to our reactive scheduling under exponential UPS inter-arrival times. The argument that *deterministic approaches deteriorate under stochastic disruptions* is the published version of our motivation for not using only CP-SAT.

2. **DDQN family validation.** The DDQN core of our Dueling-DDQN meta-agent has its uncertainty-handling pedigree established here. Cite Zhang Y. (2022) when defending Double-Q learning specifically.

3. **Interpretability defense.** Spectrum analysis of trained Q-values shows that DDQN-HH decisions correspond to interpretable patterns over LLH selections. We can replicate this analysis on our 5-tool meta-agent for Chapter 5 discussion (e.g., showing that GC_RESTOCK is selected with high probability when GC silo levels drop below threshold X).

**Thesis Connection:**
- Chapter 1 (Introduction) — Cite for "deterministic approaches deteriorate under disruptions" motivation
- Chapter 2 (Literature Review) — Anchor for "DRL-HH under uncertainty" subsection
- Chapter 3 (Methodology) — Justifies DDQN component of Dueling DDQN; supports hybrid state representation philosophy
- Chapter 5 (Discussion) — Spectrum-analysis interpretability framing

---

### 4. Karimi-Mamaghan et al. (2023) — Q-Learning + Iterated Greedy for Permutation Flow Shop Scheduling

**Karimi-Mamaghan, M., Mohammadi, M., Pasdeloup, B., & Meyer, P. (2023).** Learning to select operators in meta-heuristics: An integration of Q-learning into the iterated greedy algorithm for the permutation flowshop scheduling problem. *European Journal of Operational Research*, 304(3), 1296-1330. https://doi.org/10.1016/j.ejor.2022.03.054

**Q1 Verification:** European Journal of Operational Research — Q1 (same journal as Zhang 2022).

**Problem Type:** Permutation Flowshop Scheduling Problem (PFSP) with makespan minimization (Cmax). Jobs must be processed on M machines in series with identical permutation ordering — a classic NP-hard combinatorial optimization problem.

**RL Algorithm:** Tabular Q-learning integrated as an Adaptive Operator Selection (AOS) mechanism within the perturbation phase of Iterated Greedy. The framework (QILS — Q-learning Iterated Local Search) learns online during the search process, updating Q-values in real-time based on operator performance and search status.

**Low-Level Heuristics:** Portfolio of perturbation operators following the destruction-reconstruction paradigm. Multiple destruction operators remove d jobs (varying d gives different perturbation strengths); reconstruction operators reinsert removed jobs at positions minimizing makespan increase (NEH-style). Estimated 3–5 operators with varying perturbation strengths.

**State Representation:** Search status features including current iteration/stage, solution quality metrics (current makespan, best makespan found, recent improvement trends), operator performance history, and search trajectory indicators. Continuous state space discretized for Q-table storage.

**Action Space:** Discrete; each action corresponds to selecting one perturbation operator. ε-greedy or similar exploration-exploitation strategy.

**Reward Design:** Solution quality improvement after applying the selected operator. Positive rewards for makespan reduction, negative or zero rewards for no improvement. Credit assignment is immediate.

**Key Results:** On Taillard's benchmark instances (120 instances up to 500 jobs × 20 machines) and VRF benchmarks, QILS achieved better optimality gaps than non-learning IG versions and seven state-of-the-art algorithms. Faster convergence, lower computational overhead, and statistically significant improvements demonstrated through comprehensive Wilcoxon signed-rank testing.

**Position in our literature:** The most-cited foundational Q-learning hyper-heuristic paper for scheduling. Cite as the originator of the *"machine learning at the service of meta-heuristics"* philosophy that motivates our entire RL-HH approach. Less direct algorithmic match than Ren & Liu (since Karimi-Mamaghan uses tabular Q-learning, not DRL), but stronger philosophical anchor.

**Thesis Connection:**
- Chapter 2 (Literature Review) — Origin reference for tabular RL-HH lineage; bridges to deep variants (Zhang 2022, Ren 2024)
- Chapter 3 (Methodology) — Cite for the philosophy of "RL selects WHICH heuristic; heuristic decides HOW"

---

### 5. Zhang Z.-Q. et al. (2023) — Q-Learning HH for Distributed FJSP with Crane Transportation

**Zhang, Z.-Q., Wu, F.-C., Qian, B., Hu, R., Wang, B., Jin, H.-P., & Yang, J.-B. (2023).** A Q-learning-based hyper-heuristic evolutionary algorithm for the distributed flexible job-shop scheduling problem with crane transportation. *Expert Systems with Applications*, 234, 121050. https://doi.org/10.1016/j.eswa.2023.121050

**Q1 Verification:** Expert Systems with Applications — Q1 (Scimago 2024) in Computer Science Applications, Engineering (miscellaneous).

**RL Algorithm:** Q-learning-based High-Level Strategy (QHLS) within a Hyper-Heuristic Evolutionary Algorithm (QHHEA) framework.

**Low-Level Heuristics:** **6 neighborhood heuristics** for solution modification: (1) Factory-Based Swap, (2) Factory-Based Insert, (3) Operation-Machine Swap, (4) Operation-Machine Insert, (5) Job Sequence Swap, (6) Job Sequence Insert. Operating across three decision dimensions: factory assignment, machine assignment, job sequencing.

**State Representation:** Two-dimensional discretized state — (population diversity, average fitness). Captures both convergence and diversity aspects of evolutionary search.

**Action Space:** 6 actions corresponding to 6 LLHs. ε-greedy with dynamic adaptive ε to prevent premature convergence.

**Reward Design:** Relative improvement: R = (f_before − f_after) / f_before. Direct effectiveness signal.

**Key Results:** On 36 benchmark instances (10×5, 20×10, 30×15), QHHEA achieved 12.3% better makespan than GA, 9.7% better than ABC, 8.1% better than IG. Wilcoxon signed-rank test confirmed statistical significance (p < 0.05). Ablation showed adaptive Q-learning selection outperformed both random and fixed LLH sequences.

**Position in our literature:** Adjacent rather than central. Useful for: (a) showing that Q-learning HH works for FJSP variants (which validates our Q-Learning baseline as a sensible method ladder rung), (b) the statistical-significance-test methodology (Wilcoxon) for our Chapter 5 results.

**Thesis Connection:**
- Chapter 2 (Literature Review) — Supplementary citation for Q-learning HH track
- Chapter 5 (Results) — Statistical methodology reference (Wilcoxon signed-rank)

---

## Contextual References (NOT RL-HH, but topically connected)

These three papers appear in many RL-HH literature reviews but they are **not actually hyper-heuristics**. We cite them carefully in the appropriate sections, never as RL-HH support.

### ◇ Hubbs et al. (2020) — End-to-end DRL for Chemical Production Scheduling

**Hubbs, C. D., Li, C., Sahinidis, N. V., Grossmann, I. E., & Wassick, J. M. (2020).** A deep reinforcement learning approach for chemical production scheduling. *Computers & Chemical Engineering*, 141, 106982. https://doi.org/10.1016/j.compchemeng.2020.106982

**Q1 Verification:** Computers & Chemical Engineering — Q1 (Scimago 2024) in Chemical Engineering (miscellaneous).

**What this paper actually is:** End-to-end DRL using **Advantage Actor-Critic (A2C)** — a policy-gradient method. The agent directly outputs production decisions (what product to make next, batch size). This is **NOT a hyper-heuristic** — there are no low-level heuristics to choose from. The agent learns the entire scheduling policy from scratch.

**Why we cite it (and where):** Industrial applicability anchor for end-to-end DRL on **batch chemical scheduling under disruptions** (production delays, plant shutdowns, rush orders, fluctuating prices, shifting demand). This is the closest published analog to the **end-to-end DRL paradigm** (PPO, A2C) applied to batch process manufacturing — the direction we explored in preliminary trials but ultimately abandoned due to gradient death. Cite in Chapter 6 (Future Work) when discussing why the value-based RL-HH paradigm was selected over end-to-end policy-gradient DRL for this thesis. Do NOT cite as RL-HH support; Hubbs et al. explicitly do NOT use dispatching rules as actions — they let the policy network output production decisions directly.

**Important warning for the thesis:** Do not cite this as RL-HH support. Do not use it to defend the RL-HH design. Hubbs et al. explicitly do NOT use dispatching rules as actions — they let the policy network output production decisions directly.

### ◇ Hu et al. (2020) — DRL with Graph Convolutional Network on Petri Nets

**Hu, L., Liu, Z., Hu, W., Wang, Y., Tan, J., & Wu, F. (2020).** Petri-net-based dynamic scheduling of flexible manufacturing system via deep reinforcement learning with graph convolutional network. *Journal of Manufacturing Systems*, 55, 1-14. https://doi.org/10.1016/j.jmsy.2020.02.004

**Q1 Verification:** Journal of Manufacturing Systems — Q1 (Scimago 2024) in Industrial and Manufacturing Engineering, Control and Systems Engineering, Software, Hardware and Architecture.

**What this paper actually is:** DQN with a custom Petri-net Convolution (PNC) layer using Graph Convolutional Network principles. The agent directly fires Petri-net transitions (= directly executes scheduling actions) based on the marking (token distribution). This is **NOT a hyper-heuristic** — there are no dispatching rules being selected.

**Why we cite it (and where):** Reference for **alternative state representations** in DRL scheduling. The Petri-net + GCN approach handles shared resources, route flexibility, and stochastic arrivals — a problem structure with similarities to our shared-pipeline GC bottleneck. Cite as future-work option in Chapter 6 ("alternative state encodings such as Petri nets with GCNs could potentially capture the shared-pipeline structural constraints more naturally than the flat 33-feature vector"). Do not cite as RL-HH support; their action space is not heuristic selection.

> **Note on Luo (2020):** Luo, S. (2020). *Applied Soft Computing*, 91, 106208 — previously listed in this Contextual References section in v2. **Promoted to ★3 Supporting Key Reference in v3** (see top section of this document). The detailed analysis of Luo (2020) now appears immediately after Ren & Liu (2024) in the Core Papers section.

---

## Additional High-Quality Papers (Supplementary)

### Multi-Objective Q-Learning for Energy-Aware Mixed Shop

**Cheng, L., Tang, Q., Zhang, L., & Zhang, Z. (2022).** Multi-objective Q-learning-based hyper-heuristic with Bi-criteria selection for energy-aware mixed shop scheduling. *Swarm and Evolutionary Computation*, 69, 100985. https://doi.org/10.1016/j.swevo.2021.100985

**Q1 Verification:** Swarm and Evolutionary Computation — Q1 (Scimago 2024) in Computer Science Applications, Theoretical Computer Science.

This paper addresses **mixed shop scheduling** combining job-shop and flow-shop production systems with energy optimization through speed-scaling policies. The QHH-BS algorithm uses Q-learning to select from **four solution updating heuristics** within a multi-objective evolutionary framework. Key innovation is the **bi-criteria selection strategy** combining Pareto-based selection (diversity) with indicator-based selection (convergence). Reward design considers improvement in both Pareto front proximity and solution distribution. Useful supporting reference for the multi-objective dimension of our profit/throughput/stockout/tardiness evaluation.

### DDPG with Continuous Rule Weights for Dynamic Flexible Job Shop

**Gui, Y., Tang, D., Zhu, H., Zhang, Y., & Zhang, Z. (2023).** Dynamic scheduling for flexible job shop using a deep reinforcement learning approach. *Computers & Industrial Engineering*, 180, 109255. https://doi.org/10.1016/j.cie.2023.109255

**Q1 Verification:** Computers & Industrial Engineering — Q1 (Scimago 2024) in Computer Science Applications, Industrial and Manufacturing Engineering.

This paper innovates by using **Deep Deterministic Policy Gradient (DDPG)** with **continuous action spaces** for dynamic FJSP with new job arrivals (Poisson distribution) and machine breakdowns. Instead of discrete heuristic selection, DDPG learns **continuous weight variables (w₁, w₂, w₃, w₄)** for four single dispatching rules (SPT, EDD, FIFO, MWKR), computing Composite_Rule = w₁×SPT + w₂×EDD + w₃×FIFO + w₄×MWKR. State representation uses 8 normalized features identical to Ren & Liu (2024). Useful supporting reference for the discrete-vs-continuous action-space discussion in Chapter 3.

---

## Contextual Reference (NOT Q1, NOT for primary citation)

### ⚪ Lassoued et al. (2026) — Policy-Based DRL Hyperheuristic for JSSP

**Lassoued, S., Gobachew, A., Lier, S., & Schwung, A. (2026).** Policy-Based Deep Reinforcement Learning Hyperheuristics for Job-Shop Scheduling Problems. *arXiv preprint* arXiv:2601.11189. **NOT Q1-published. NOT peer-reviewed.**

**What this paper is:** MaskablePPO selecting among 6 dispatching rules (FIFO, SPT, SPS, LTWR, SPSP, LPTN), with action masking via Petri net guard functions. Tested on 80 Taillard JSSP instances; achieved 1.4–4.1% improvement over best heuristic per instance.

**Why we cannot use it as a primary key reference:** Lassoued et al. (2026) is an arXiv preprint as of April 2026 and has not yet appeared in a peer-reviewed Q1 journal. Citing it as a primary methodological anchor for a Q1-grade thesis would be a defensible weakness for the committee. The same authors' prior work (Lassoued & Schwung, 2024, "Introducing PetriRL," *Journal of Manufacturing Systems*, Q1) is peer-reviewed but covers MaskablePPO end-to-end on JSSP without the hyper-heuristic layer.

**How we cite it in our thesis:** Once, in Chapter 2 literature review, as supplementary evidence of the policy-based DRL-HH paradigm — explicitly labeled *"arXiv preprint, not yet peer-reviewed"*. Use Zhang Y. et al. (2022) EJOR as the actual Q1 anchor for DRL-HH instead.

**Selected useful concepts (cite carefully):**
- Action masking via Petri-net guard functions (Eq. 14): *"the Petri net pre-filters the invalid action"* — methodologically aligned with our tool-mask approach
- 6 dispatching rules as LLHs — same range as our 5-tool design
- Commitment mechanism (heuristic applied for k consecutive steps) — interesting future-work direction we have not implemented

---

## Synthesis: The Two-Pillar Inheritance Pattern and Our Position

### Two-pillar inheritance

The thesis innovation rests on **two complementary literature lineages** that converge in our problem class:

```
PILLAR 1 — Problem-class lineage (UPMSP-SDFST with sequence-dependent setups)
─────────────────────────────────────────────────────────────────────────────
Zhang, Zheng, Weng (2007, IJAMT)        Tabular Q-Learning for parallel machine scheduling
        ↓ (function approximation step)
LBF-Q baseline (Zhang 2007 lineage)     Q-Learning with linear basis functions
        ↓ (deep RL step)
PAENG, Park, Park (2021, IEEE Access)   ★ DDQN with parameter sharing for UPMSP-SDFST
                                          → ~85% improvement over LBF-Q on 8/8 datasets
                                          → robustness under stochastic processing/setup times
        ↓ (this thesis: shared bottleneck + UPS extension)
THIS THESIS (Phase 5: paeng_ddqn/)      DDQN with parameter sharing on Nestlé Trị An UPMSP


PILLAR 2 — RL-HH paradigm lineage (RL selecting from dispatching rules)
─────────────────────────────────────────────────────────────────────────────
Karimi-Mamaghan et al. (2023, EJOR)     Tabular Q-Learning HH framework
        ↓
LUO (2020, ASOC)                        ★ DQN + 6 CDRs for DFJSP, with tabular Q-Learning
                                          baseline (Section 6.4: 38/45 instance win)
        ↓
Panzer, Bender, Gronau (2024, IJPR)     DQN-HH for production control with disruptions
        ↓ (dueling architecture step)
REN & LIU (2024, Sci Reports)           ★ D3QN (Dueling DDQN) + 7 CDRs + grouped network
                                          → D3QN 0.85 > DDQN 0.80 > DQN 0.70 on-time rate
        ↓ (this thesis: applied to UPMSP-SDFST + shared pipeline)
THIS THESIS (Phase 6: rl_hh/)           Dueling DDQN + 5 tools + grouped network
```

### How the two pillars combine in our thesis

- **From Pillar 1 (Paeng 2021):** problem class (UPMSP-SDFST), parameter-sharing architecture, dimension-invariant state representation, DDQN training algorithm, paired-seed disruption testing protocol, tabular Q-Learning baseline experimental design (LBF-Q lineage).
- **From Pillar 2 (Ren & Liu 2024 + Luo 2020):** dueling architecture (V/A decomposition), grouped network design, [0,1] feature normalization, RL-HH dispatching-rule action paradigm.
- **Our innovation:** combine both inheritances and apply to a problem class with a shared bottleneck resource (GC pipeline) absent in both precedents, plus mixed MTS/MTO demand and within-shift UPS disruptions.

### Algorithmic positioning

The field employs value-based methods (Q-learning, DQN, DDQN, D3QN) for discrete heuristic selection. Our Dueling DDQN choice sits in the value-based stream, on the D3QN frontier (Li 2024 survey, Table 1). Both Phase 5 (Paeng's Modified DDQN) and Phase 6 (RL-HH Dueling DDQN) are value-based, providing methodological consistency across the architectural ladder.

### Low-Level Heuristic Design Philosophy

Successful implementations leverage domain knowledge through established dispatching rules (SPT, EDD, FIFO, LTWR), problem-specific operators, and composite rules integrating multiple criteria (MachineRank-based CDRs in Ren & Liu; tool-style heuristics in our work). The key insight from the literature is that **LLH quality fundamentally determines hyper-heuristic performance** — intelligent selection from well-designed operators outperforms sophisticated RL selecting from poor operators. Our 5 tools (PSC_THROUGHPUT, GC_RESTOCK, MTO_DEADLINE, SETUP_AVOID, WAIT) are designed to encode the Nestlé Trị An operator's domain expertise, with GC_RESTOCK as a novel inventory-replenishment tool not present in JSSP/FJSP literature.

### Production problem coverage

Applications span classic problems (PFSP, JSSP, FJSP) to complex variants (distributed scheduling with crane transportation, modular production control, energy-aware mixed shop). Paeng (2021) addresses the closest variant — UPMSP-SDFST in semiconductor wafer preparation — but with deterministic execution within episodes. Our contribution extends the DRL-HH paradigm to **batch roasting with shared pipeline constraints under within-shift stochastic equipment failures (UPS) and mixed MTS/MTO demand** — a problem structure not previously addressed in the RL-HH literature.

### Hybrid intelligence

The dominant paradigm combines human domain expertise (proven heuristics, problem structure) with machine learning adaptability (state-dependent selection, online learning), achieving superior performance versus both pure end-to-end DRL (which lacks domain knowledge) and static heuristics (which lack adaptability). Our 4-method ladder (Dispatching → Tabular Q-Learning → Paeng DDQN → RL-HH) is designed to test the architectural progression rigorously: each level adds one architectural element with a published precedent, and the comparison tests whether each step yields proportional gains on our problem class.

### Interpretability Emphasis

Recent work (Panzer 2024, Zhang Y. 2022) prioritizes explainability through named dispatching rules, transparent decision processes, and spectrum analysis of learned policies. Our RL-HH inherits this advantage — *"the agent chose GC_RESTOCK because GC silo level dropped below 12 batches"* is a defensible explanation that direct allocation methods (Paeng's Modified DDQN, end-to-end DRL approaches) cannot produce. This interpretability is a qualitative differentiator independent of headline performance numbers.

### Our specific position

This thesis sits at the intersection of two literature pillars: it adopts Paeng (2021)'s problem class and DDQN architecture, then takes one architectural step beyond by introducing Ren & Liu (2024)'s dueling V/A decomposition and Luo (2020)'s tool-selection action paradigm. The methodological contribution is not algorithmic but **comparative** — establishing under which UPS intensities each of the four reactive methods dominates, with explicit literature-aligned baselines at each rung of the architectural ladder.
