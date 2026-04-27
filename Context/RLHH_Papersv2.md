# Reinforcement Learning-Based Hyper-Heuristics for Production Scheduling: Literature Review

> **Version:** 3.0 (April 2026)
> **Previous:** v2 (March 2026)
> **Purpose:** Canonical Q1-verified literature foundation for the RL-HH method in the Nestlé Trị An batch roasting thesis.

---

## CHANGELOG v2 → v3

| # | Change | Reason |
|---|---|---|
| 1 | **Lassoued et al. (2026) demoted from core to contextual reference** | arXiv preprint, not Q1-published. Cannot anchor a Q1-grade thesis defense. |
| 2 | **Zhang Y. et al. (2022) EJOR added as Core Paper #3** | Q1-verified DDQN-HH paper with explicit uncertainty handling — fills the role Lassoued could not. |
| 3 | **Two Primary Key References designated** | Ren & Liu (2024) for algorithm + Panzer et al. (2024) for production-control context. |
| 4 | **Three contextual references added** | Hubbs 2020 (end-to-end DRL for chemical batch — links to MaskedPPO baseline), Hu 2020 (Petri net + GCN — alternative state representation), Luo 2020 (DQN-HH precursor to Ren & Liu's D3QN). |
| 5 | **Q1 verification table added** | Explicit Scimago/JCR status for every cited journal so committee questions can be answered with one glance. |

---

## Q1 Verification Summary (All Core References)

| # | Paper | Year | Journal | Scimago Q | Role in thesis |
|---|---|---|---|---|---|
| ★1 | **Ren & Liu** | 2024 | Scientific Reports | **Q1** | **Primary algorithm reference** — D3QN architecture, grouped network, CDR-based action space |
| ★2 | **Panzer, Bender, Gronau** | 2024 | International Journal of Production Research | **Q1** | **Primary application-context reference** — DRL-HH for production control with disruptions |
| 3 | Zhang Y., Bai, Qu, Tu, Jin | 2022 | European Journal of Operational Research | **Q1** | DDQN-HH under uncertainty — methodological support |
| 4 | Karimi-Mamaghan et al. | 2023 | European Journal of Operational Research | **Q1** | Q-Learning-HH framework — methodology lineage |
| 5 | Zhang Z.-Q. et al. | 2023 | Expert Systems with Applications | **Q1** | Q-Learning-HH for FJSP — supplementary |
| ⚪ | Lassoued et al. | 2026 | arXiv preprint | **NOT Q1** | Contextual only; PPO-HH paradigm reference |
| – | Li, C. et al. | 2024 | PeerJ Computer Science | Q2 | Survey — taxonomy citation |
| – | Cheng et al. | 2022 | Swarm and Evolutionary Computation | **Q1** | Multi-objective Q-learning HH — supplementary |
| – | Gui et al. | 2023 | Computers & Industrial Engineering | **Q1** | DDPG with continuous rule weights — supplementary |
| ◇ | Hubbs et al. | 2020 | Computers & Chemical Engineering | **Q1** | End-to-end DRL for batch chemical (links to MaskedPPO baseline) |
| ◇ | Hu et al. | 2020 | Journal of Manufacturing Systems | **Q1** | DRL + Petri net + GCN (alternative state representation, future work) |
| ◇ | Luo | 2020 | Applied Soft Computing | **Q1** | DQN-HH for DFJSP — direct predecessor of Ren & Liu (2024) |

★ = Primary key reference for thesis defense
⚪ = Contextual only, NOT Q1
◇ = Adjacent literature (not RL-HH but topically connected)

---

## Field Context and Survey

**Li, C., Wei, X., Wang, J., Wang, S., & Zhang, S. (2024).** A review of reinforcement learning based hyper-heuristics. *PeerJ Computer Science*, 10, e2141. https://doi.org/10.7717/peerj-cs.2141

This 2024 survey represents the **first comprehensive review** of reinforcement learning based hyper-heuristics, systematically analyzing over 80 representative papers from six major academic databases. The authors present a hierarchical two-tier taxonomy dividing RL-HH into **Value-Based approaches** (subdivided into Traditional RL-HH using Q-learning/MAB methods, and Deep RL-HH using DQN/DDQN/D3QN) and **Policy-Based approaches** (using PPO/DPPO). For production scheduling specifically, the survey documents that Q-learning-based approaches have proven particularly effective, with multiple successful implementations across workshop scheduling, semiconductor manufacturing, and mixed shop problems. The review identifies key advantages of RL-HH: no requirement for deep prior domain knowledge, robust generalization capabilities, and automatic adaptation to dynamic environments through learning. Current limitations include lack of theoretical convergence analysis, high computational costs, and limited integration of value-based and policy-based strengths. This survey establishes RL-HH as a mature field with substantial production scheduling applications from 2020-2024.

**Why this survey matters for the thesis:** Table 1 of Li et al. (2024) classifies the value-based DRL-HH lineage as DQN → DDQN → D3QN (Dueling Double DQN). Our Dueling DDQN meta-agent sits in the D3QN class — the survey's frontier category. The survey also explicitly defends value-based DRL over policy-based PPO for hyper-heuristic control: *"PRL... requires more samples to learn effective policies, making them inferior to value-based methods in terms of sample efficiency. Additionally, PRL is also more susceptible to training instabilities."* This directly justifies our choice to use Dueling DDQN for the RL-HH meta-agent rather than another PPO variant after the gradient-death observed in our PPO end-to-end cycles.

---

## Core Papers: Detailed Analysis

### ★1. Ren & Liu (2024) — D3QN with MachineRank for Dynamic Flexible Job Shop Scheduling — **PRIMARY KEY REFERENCE**

**Ren, F., & Liu, H. (2024).** Dynamic scheduling for flexible job shop based on MachineRank algorithm and reinforcement learning. *Scientific Reports*, 14, 29741. https://doi.org/10.1038/s41598-024-79593-8

**Q1 Verification:** Scientific Reports — Q1 (Scimago 2024, Multidisciplinary). Published by Nature Portfolio. Impact Factor 3.9.

**Problem Type:** Dynamic Flexible Job Shop Scheduling Problem (DFJSP) with new job insertions following Poisson arrivals, machine breakdowns, processing time changes, and AGV (Automated Guided Vehicle) transportation constraints. Objectives include minimizing maximum completion time (makespan) and improving on-time completion rates in highly dynamic manufacturing environments.

**RL Algorithm:** Dueling Double Deep Q-Network (D3QN), which reduces overestimation of Q-values compared to DQN/DDQN and improves training stability. D3QN separates state value V(s) and advantage A(s,a) estimation in the network architecture, combining them as Q(s,a) = V(s) + A(s,a) − mean(A) to produce final Q-values, enabling more robust learning in complex scheduling environments.

**Low-Level Heuristics:** The D3QN agent selects from **7 Composite Dispatching Rules (CDRs)** based on the novel **MachineRank (MR)** algorithm, calculated iteratively as MR(n) = a×M×MR(n−1) + (1−a)×MV. The 7 CDRs are: (1) CDR1 — Minimum Average Idle Time + Minimum MR; (2) CDR2 — Maximum Estimated Tardiness + Minimum MR; (3) CDR3 — Tardiness-aware + Completion Rate + Minimum MR; (4) CDR4 — Tardiness-aware + Relaxation Time Ratio + Minimum MR; (5) CDR5 — Weighted Tardiness-Completion + Minimum MR; (6) CDR6 — Random Job + Minimum MR; (7) CDR7 — Random Job + Earliest Available Machine. Each CDR integrates job selection logic with machine assignment strategy in a single integrated decision.

**State Representation:** Eight state features divided into machine-level and job-level groups, all normalized to [0,1]: average machine utilization rate, standard deviation of machine utilization, maximum difference in machine utilization, average job completion rate, standard deviation of job completion rate, average operation completion rate, estimated tardiness rate, and actual tardiness rate.

**Network Architecture (Fig. 1b):** **Grouped network**, not fully connected. Machine-level features and job-level features are processed in separate sub-networks for the first three hidden layers, then merged. This grouped architecture reduces parameter count and prevents cross-group feature interference, achieving faster convergence than fully connected networks.

**Reward Design:** Multi-component reward considering actual tardiness rate, MR values, and estimated job tardiness rate. Reward +1 if MR(sₜ) ≤ MR(sₜ₊₁) AND Tard_e(sₜ) > Tard_e(sₜ₊₁); Reward −1 if Tard_e(sₜ) ≤ Tard_e(sₜ₊₁); Reward +1 if Tard_a(sₜ) ≥ Tard_a(sₜ₊₁); Reward −1 if MR(sₜ) > MR(sₜ₊₁) AND Tard_a(sₜ) < Tard_a(sₜ₊₁); Otherwise reward = 0.

**Key Results:** D3QN converged to 0.85 on-time completion rate compared to 0.80 for DDQN and 0.70 for DQN. The approach significantly outperformed single scheduling rules (FIFO, EDD, MRT, SPT, LPT, Random) across all metrics. After machine breakdown rescheduling, the system recovered with only 6-second delay versus the original schedule.

**Why this is the PRIMARY KEY REFERENCE for our thesis:**

| Component | Our `rl_hh/` design | Ren & Liu 2024 |
|---|---|---|
| Algorithm | Dueling DDQN | D3QN (= Dueling Double DQN) |
| Network | Grouped (Roaster / Inventory / Context) | Grouped (machine-level / job-level), Fig. 1(b) |
| State features | 33, all normalized [0,1] | 8, all normalized [0,1] |
| Action space | 5 dispatching tools | 7 composite dispatching rules (CDRs) |
| Disruption type | UPS (machine stoppages) | Machine breakdowns + new job insertions |
| Compared against | Q-Learning, MaskedPPO, dispatching | DQN, DDQN, single rules |

This paper is the architectural blueprint for our RL-HH meta-agent. Our network groups (Roaster / Inventory-Flow / Context) directly replicate Ren & Liu's Fig. 1(b) grouped architecture, our Dueling-DDQN class is exactly Ren & Liu's D3QN, and our use of dispatching tools maps to their CDRs. Cite as the primary algorithm reference in Chapter 3 (Methodology), Section 3.4.4 (RL-HH design).

**Thesis Connection:**
- Chapter 2 (Literature Review) — Anchor for "DRL-HH for dynamic scheduling under disruptions" subsection
- Chapter 3 (Methodology) — Justifies every Dueling DDQN architectural choice; cite for grouped network design and CDR-as-action paradigm
- Chapter 4 (Implementation) — Reference architecture for Section 4.8
- Chapter 5 (Discussion) — Comparison framework; D3QN > DDQN > DQN convergence ordering provides expected pattern

---

### ★2. Panzer, Bender, Gronau (2024) — DQN Hyper-Heuristic for Modular Production Control — **PRIMARY KEY REFERENCE**

**Panzer, M., Bender, B., & Gronau, N. (2024).** A deep reinforcement learning based hyper-heuristic for modular production control. *International Journal of Production Research*, 62(8), 2747-2768. https://doi.org/10.1080/00207543.2023.2233641

**Q1 Verification:** International Journal of Production Research — Q1 (Scimago 2024) in Industrial and Manufacturing Engineering, Management Science and Operations Research, Strategy and Management.

**Problem Type:** Semi-heterarchical modular production control in job shop-like environments with highly configurable products and reentrant flows. The multi-layered system includes shop-floor manufacturing agents controlling production operations and transport-layer distribution agents managing material flow between pre-defined modules. Challenges addressed include demand fluctuations, **unforeseen machine failures**, **rush orders**, prioritized order processing, and large control solution spaces due to flexible routing capabilities.

**RL Algorithm:** Deep Q-Network (DQN), value-based, model-free, off-policy. Initial training required ~36 hours for the first module, with re-training time of only ~4 hours when adding new modules (transfer learning capability).

**Low-Level Heuristics:** The DQN-HH selects from production dispatching rules including: **FIFO** (local + global variants), **SPT** (Shortest Processing Time), **EDD** (Earliest Due Date), and **HP** (Highest Priority). Each heuristic maps to a specific DQN action, providing quick execution, established production logic, comprehensibility (unlike pure neural decisions), and **scenario-independent action space that doesn't increase with layout size**.

**State Representation:** Min-max normalized features including job/order information (current job characteristics, queue lengths, waiting jobs, order priorities, urgency flags, due dates, remaining processing requirements), machine/station status, and system-level metrics (WIP levels, buffer states, inventory positions, queue information). Distribution agents in D1, D1.1, D1.2 deploy 88, 84, and 52 input neurons; manufacturing agents in M1.1.1 and M1.1.2 deploy 80 and 60 neurons.

**Action Space:** Discrete action space where each action corresponds to selection of one low-level dispatching rule. The action set is fixed in size and does not scale with production layout. Actions are triggered at specific events: new order arrivals, agent arrival at destination, system changes (machine failure, rush order insertion), agents with no assigned task, or machine idle with jobs waiting in queue.

**Reward Design:** Multi-objective reward targeting: (1) **throughput time minimization** (mean throughput time, flow time optimization), (2) **tardiness reduction** (minimizing late deliveries, adherence to due dates), and (3) **priority order processing** (rush orders, prioritized orders, order urgency consideration as customer-centric objective). The total reward is a composite of these criteria for the chosen order.

**Key Results:** Tested in discrete-event simulation and validated in a hybrid production environment at the Center for Industry 4.0. The hyper-heuristic achieved 12.7%–18.6% better throughput time than worst conventional heuristics (FIFO/FCFS as benchmark, plus SPT, EDD, STD, LWT, NV individually tested). Robustness analysis showed conventional rules exhibited significant reward fluctuations while the hyper-heuristic demonstrated more stable reward reception with lower variance across varying conditions. Real-world validation successfully transferred from simulation to physical system with autonomous mobile robots.

**Why this is the PRIMARY KEY REFERENCE for our thesis:**

This is the production-control context anchor. Where Ren & Liu provide the algorithm template, Panzer et al. provide the proof that DRL-HH actually works for **production scheduling under industrial disruptions** (machine failures, rush orders, demand changes) — not just abstract job-shop benchmarks. Three critical contributions to our defense:

1. **Action-space defense.** The paper states explicitly: *"due to the pre-defined set of dispatching rules, the action space does not increase with large layout sizes and there is no need to introduce masked actions for learning as the logic is mapped intrinsically"* (Section on action space design). This sentence directly defends our 5-tool action space against the criticism *"why not learn 21 raw actions like PPO does?"* — the answer is exactly Panzer's argument: tool-based action spaces stay small and explainable regardless of plant complexity.

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

**Why we cite it (and where):** Industrial applicability anchor for end-to-end DRL on **batch chemical scheduling under disruptions** (production delays, plant shutdowns, rush orders, fluctuating prices, shifting demand). This is the closest published analog to our **MaskedPPO baseline** — also end-to-end policy-gradient DRL — applied to batch process manufacturing. Cite in the MaskedPPO subsection (Section 3.4.3 / 4.7), not in the RL-HH subsection. Useful for arguing that our MaskedPPO baseline is a credible representative of the end-to-end DRL category, since Hubbs et al. show A2C/policy-gradient approaches are competitive with shrinking-horizon MILP for chemical batch scheduling.

**Important warning for the thesis:** Do not cite this as RL-HH support. Do not use it to defend the RL-HH design. Hubbs et al. explicitly do NOT use dispatching rules as actions — they let the policy network output production decisions directly.

### ◇ Hu et al. (2020) — DRL with Graph Convolutional Network on Petri Nets

**Hu, L., Liu, Z., Hu, W., Wang, Y., Tan, J., & Wu, F. (2020).** Petri-net-based dynamic scheduling of flexible manufacturing system via deep reinforcement learning with graph convolutional network. *Journal of Manufacturing Systems*, 55, 1-14. https://doi.org/10.1016/j.jmsy.2020.02.004

**Q1 Verification:** Journal of Manufacturing Systems — Q1 (Scimago 2024) in Industrial and Manufacturing Engineering, Control and Systems Engineering, Software, Hardware and Architecture.

**What this paper actually is:** DQN with a custom Petri-net Convolution (PNC) layer using Graph Convolutional Network principles. The agent directly fires Petri-net transitions (= directly executes scheduling actions) based on the marking (token distribution). This is **NOT a hyper-heuristic** — there are no dispatching rules being selected.

**Why we cite it (and where):** Reference for **alternative state representations** in DRL scheduling. The Petri-net + GCN approach handles shared resources, route flexibility, and stochastic arrivals — a problem structure with similarities to our shared-pipeline GC bottleneck. Cite as future-work option in Chapter 6 ("alternative state encodings such as Petri nets with GCNs could potentially capture the shared-pipeline structural constraints more naturally than the flat 33-feature vector"). Do not cite as RL-HH support; their action space is not heuristic selection.

### ◇ Luo (2020) — DQN with Composite Dispatching Rules for DFJSP

**Luo, S. (2020).** Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning. *Applied Soft Computing*, 91, 106208. https://doi.org/10.1016/j.asoc.2020.106208

**Q1 Verification:** Applied Soft Computing — Q1 (Scimago 2024) in Computer Science Applications, Software.

**What this paper is:** DQN + Double DQN + soft target weight update, selecting from **6 composite dispatching rules** for DFJSP with new job insertions. Seven generic state features, all normalized to [0,1]. Softmax action-selection at evaluation time.

**Position in our literature:** This is the **direct predecessor of Ren & Liu (2024)**. Same problem family (DFJSP with disruptions), same architecture family (DQN→Double→Dueling lineage), same action paradigm (composite dispatching rules), same state-feature philosophy ([0,1] normalized features for cross-instance generality). Where Luo uses Double-DQN, Ren & Liu add the Dueling architecture — yielding D3QN. Cite as the lineage origin if the thesis discusses why we chose D3QN specifically over plain DQN.

**Thesis Connection:**
- Chapter 2 (Literature Review) — Cite as "Luo (2020)... was extended to D3QN by Ren & Liu (2024) by adding dueling architecture"
- Chapter 3 (Methodology) — Reference for the [0,1] feature-normalization design philosophy

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

## Synthesis: The RL-HH Lineage and Our Position

The DRL-HH lineage relevant to our thesis can be summarized as:

```
Karimi-Mamaghan (2023, EJOR Q1)         Tabular Q-learning HH
        ↓
Luo (2020, ASOC Q1)                     DQN + Double + soft target → 6 CDRs (DFJSP)
        ↓
Zhang Y. (2022, EJOR Q1)                DDQN-HH with uncertainty handling (10 LLHs, container routing)
        ↓
Panzer (2024, IJPR Q1)                  DQN-HH for production control (FIFO, SPT, EDD, HP)
        ↓
Ren & Liu (2024, Sci Rep Q1)            D3QN (Dueling DDQN) + 7 CDRs + grouped network (DFJSP)
        ↓
=========================================
THIS THESIS (Nestlé Trị An, 2026)        Dueling DDQN + 5 tools + grouped network (Batch Roasting, UPS)
=========================================
```

**Algorithmic Diversity:** The field employs value-based methods (Q-learning, DQN, DDQN, D3QN) for discrete heuristic selection and policy-based methods (PPO, DDPG) for both discrete and continuous action spaces. Our Dueling DDQN choice sits in the value-based stream, on the D3QN frontier (Li 2024 survey, Table 1). The PPO stream (Lassoued 2026 preprint, Kallestad 2023) is a parallel paradigm — relevant to our MaskedPPO baseline rather than to our RL-HH method.

**Low-Level Heuristic Design Philosophy:** Successful implementations leverage domain knowledge through established dispatching rules (SPT, EDD, FIFO, LTWR), problem-specific operators, and composite rules integrating multiple criteria (MachineRank-based CDRs in Ren & Liu; tool-style heuristics in our work). The key insight from the literature is that **LLH quality fundamentally determines hyper-heuristic performance** — intelligent selection from well-designed operators outperforms sophisticated RL selecting from poor operators. Our 5 tools (PSC_THROUGHPUT, GC_RESTOCK, MTO_DEADLINE, SETUP_AVOID, WAIT) are designed to encode the Nestlé Trị An operator's domain expertise, with GC_RESTOCK as a novel inventory-replenishment tool not present in JSSP/FJSP literature.

**Production Problem Coverage:** Applications span classic problems (PFSP, JSSP, FJSP) to complex variants (distributed scheduling with crane transportation, modular production control, energy-aware mixed shop). Our contribution applies the DRL-HH paradigm to **batch roasting with shared pipeline constraints under stochastic equipment failures (UPS)** — a problem structure not previously addressed in the RL-HH literature.

**Hybrid Intelligence:** The dominant paradigm combines human domain expertise (proven heuristics, problem structure) with machine learning adaptability (state-dependent selection, online learning), achieving superior performance versus both pure end-to-end DRL (which lacks domain knowledge) and static heuristics (which lack adaptability). Our 4-method ladder (Dispatching → Q-Learning → MaskedPPO → RL-HH) is designed to test exactly this claim — does the structured-learning RL-HH outperform both the pure-rules baseline and the brute-force end-to-end MaskedPPO?

**Interpretability Emphasis:** Recent work (Panzer 2024, Zhang Y. 2022) prioritizes explainability through named dispatching rules, transparent decision processes, and spectrum analysis of learned policies. Our RL-HH inherits this advantage — *"the agent chose GC_RESTOCK because GC silo level dropped below 12 batches"* is a defensible explanation that the MaskedPPO baseline cannot produce.

**Our specific position:** This thesis applies the Ren & Liu (2024) D3QN + grouped-network architecture to a new problem domain (batch roasting), with a domain-novel tool (GC_RESTOCK) and a novel application context (cross-line R3 routing under shared pipeline). The methodological contribution is not algorithmic but **comparative** — establishing under which UPS intensities each of the four reactive methods dominates.
