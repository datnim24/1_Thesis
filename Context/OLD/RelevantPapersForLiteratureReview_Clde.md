# Closest structural matches for the Nestlé Trị An roasting-line scheduling problem

**No single paper in the literature covers all five novel features of this problem—dedicated-silo pipeline mutex, cross-line routing, MTO/MTS inventory coupling, batch-pool activation, and SKU reassignment—simultaneously.** The thesis sits at the intersection of process-industry batch scheduling (chemical engineering tradition) and parallel-machine lot-sizing with inventory (OR tradition), a gap that five papers collectively bracket. The strongest structural overlap comes from Ferrer-Nadal et al. (2008) on shared-transfer mutex constraints and Terrazas-Moreno et al. (2012) on dedicated tank-farm assignment—together they supply roughly 60% of the formulation skeleton. The remaining 40% (safety-stock penalties, MTO/MTS mixing, batch-pool pre-generation) must be assembled from the lot-sizing and hybrid-production literatures. Below is a scored, verified bibliography with adaptation plans.

---

## Top 5 papers: full deep dives

### Paper 1 — Ferrer-Nadal, Capón-García, Méndez & Puigjaner (2008)

**Citation:** Ferrer-Nadal, S., Capón-García, E., Méndez, C. A., & Puigjaner, L. (2008). Material transfer operations in batch scheduling: A critical modeling issue. *Industrial & Engineering Chemistry Research*, *47*(20), 7721–7732. DOI: 10.1021/ie800075u

| Rubric dimension | Score | Justification |
|---|---|---|
| Storage realism | 3 | Models intermediate storage vessels with material tracking; not dedicated single-SKU silos |
| Shared bottleneck | **5** | Material transfer = mutex; during a transfer, both sending and receiving units are blocked; synchronization constraints enforce exclusive pipeline use |
| Scheduling richness | 4 | Multipurpose batch units, sequence-dependent changeovers, general-precedence binary variables |
| Demand/inventory | 1 | No explicit demand or inventory penalty structure |
| Method transferability | 4 | MILP continuous-time formulation; synchronization constraints directly map to CP-SAT `AddNoOverlap` on transfer-interval variables |
| **Total** | **17** | |

**Formulation summary.** The key decision variables are binary transfer-task indicators $w_{ii'tt'}$ that lock both the source unit $i$ and destination unit $i'$ during time window $[t,t']$. The synchronization constraint states that if a transfer from unit $i$ to $i'$ begins at event point $n$, then no other task may execute on either $i$ or $i'$ until the transfer completes. The objective is makespan minimization subject to recipe precedence, changeover sequencing, and storage capacity. Solved with CPLEX.

**Problem match map:**

| Thesis constraint | Paper equivalent | Gap |
|---|---|---|
| Shared GC pipeline (mutex) | Transfer task locks both sender and receiver | ✅ Direct analog — thesis pipeline locks one line's consume/replenish/dump operations |
| Variable-duration dump | Transfer duration is recipe-dependent (variable) | ✅ Partially covered — dump duration $5 + \lceil q/100 \rceil$ can use same variable-duration transfer framework |
| Dedicated silo (no-mixing) | Storage vessels hold intermediates but allow mixing | ❌ Not modeled; thesis needs single-SKU per silo |
| Sequence-dependent setups | Changeovers between product tasks | ✅ Covered |
| MTO/MTS inventory coupling | Not modeled | ❌ |

**Gap analysis.** This paper provides the strongest template for encoding the shared-pipeline mutex, which is the hardest novel constraint in the thesis. The variable-duration transfer framework can be extended to model the variable-duration dump operation (5 + ⌈q/100⌉ time slots). **What the thesis adds:** dedicated silo assignment with no-mixing, the three-operation mutex (consume/replenish/dump rather than just transfer), and inventory-penalty coupling.

**Adaptation plan.** (1) Adapt the synchronization binary $w_{ii'tt'}$ into a set of interval variables per pipeline operation, each associated with a `NoOverlap` constraint on the shared pipeline resource in CP-SAT. (2) For variable-duration dump, define the interval's `size` as $5 + \lceil q_s / 100 \rceil$ where $q_s$ is the current silo inventory—this creates a coupling between inventory state and interval duration that requires linearization in MILP or an element constraint in CP-SAT. (3) The three operation types (consume, replenish, dump) become three families of interval variables all competing for the same no-overlap resource.

**Cite reason:** "Ferrer-Nadal et al. (2008) established that material-transfer operations require explicit synchronization constraints to avoid infeasible schedules, providing the foundational mutex formulation we extend to the shared GC pipeline with three mutually exclusive operation types."

---

### Paper 2 — Terrazas-Moreno, Grossmann & Wassick (2012)

**Citation:** Terrazas-Moreno, S., Grossmann, I. E., & Wassick, J. M. (2012). A mixed-integer linear programming model for optimizing the scheduling and assignment of tank farm operations. *Industrial & Engineering Chemistry Research*, *51*(18), 6441–6454. DOI: 10.1021/ie202217v

| Rubric dimension | Score | Justification |
|---|---|---|
| Storage realism | **5** | Dedicated tanks (each holds exactly one product — no mixing), tank capacity constraints, tank-product assignment as binary decision variable |
| Shared bottleneck | 1 | No explicit shared pipeline or mutex constraint |
| Scheduling richness | 3 | Parallel continuous production lines, priority-slot sequencing (MOS model), but no explicit sequence-dependent setups |
| Demand/inventory | 3 | Inventory balance at time points, shipping schedules, blocking (unsatisfied demand) minimization |
| Method transferability | 4 | MILP with ~2,500 binary variables for industrial instance; tank assignment variables $z_{j,k}$ directly transferable |
| **Total** | **16** | |

**Formulation summary.** Binary variables $z_{j,k} \in \{0,1\}$ assign product $j$ to tank $k$. Inventory balance $s_{j,k,t} = s_{j,k,t-1} + c_{j,k,t-1} \cdot \Delta t - p_{j,k,l,t-1} \cdot \Delta t$ tracks material in each dedicated tank over time. Tank capacity constraint $s_{j,k,t} \leq v_k$. A knapsack-type inequality limits transfer rates. The objective maximizes satisfied demand (equivalently minimizes blocking). The model uses the multi-operation sequencing (MOS) continuous-time framework and is solved with CPLEX/Gurobi. Results validated against a discrete-event simulation of the Dow Chemical tank farm.

**Problem match map:**

| Thesis constraint | Paper equivalent | Gap |
|---|---|---|
| Dedicated GC/RC silos (no mixing) | Dedicated tank assignment $z_{j,k}$ | ✅ Direct analog |
| Silo capacity (3,000 kg GC; 5,000 kg RC) | Tank capacity $s_{j,k,t} \leq v_k$ | ✅ Direct analog |
| SKU reassignment when empty | Not explicitly modeled (assignment fixed for horizon) | ⚠️ Partial — thesis allows dynamic reassignment |
| Inventory tracking per silo | Inventory balance equations per tank-product pair | ✅ Direct analog |
| Shared GC pipeline (mutex) | Not modeled | ❌ |
| MTO tardiness / MTS shortage penalties | Blocking minimization (similar but different penalty) | ⚠️ Partial |

**Gap analysis.** This is the **storage-realism champion**—the dedicated-tank binary assignment with no-mixing and capacity constraints can be lifted almost directly into the thesis model. The MILP + DES validation approach is also relevant. **What the thesis adds:** dynamic SKU reassignment (when a silo empties, it can be reassigned), the shared-pipeline constraint, and the multi-objective penalty structure replacing simple blocking.

**Adaptation plan.** (1) Reuse the $z_{j,k,t}$ assignment variable but index it over time to allow reassignment: $z_{j,k,t} = 1$ iff silo $k$ holds SKU $j$ at time $t$, with a constraint that $z_{j,k,t}$ can only change from $j$ to $j'$ when inventory hits zero and a dump has completed. (2) Add inventory balance equations per GC-silo and RC-silo, replacing the paper's continuous-rate model with the thesis's discrete 1-minute slots. (3) Replace the blocking objective with the thesis's weighted penalty terms.

**Cite reason:** "The tank-farm assignment model of Terrazas-Moreno et al. (2012) provides the binary dedicated-storage formulation ($z_{j,k}$ variables with no-mixing and capacity constraints) that we adapt for both GC-silo and RC-silo allocation in the roasting plant."

---

### Paper 3 — Bektur & Saraç (2019)

**Citation:** Bektur, G., & Saraç, T. (2019). A mathematical model and heuristic algorithms for an unrelated parallel machine scheduling problem with sequence-dependent setup times, machine eligibility restrictions and a common server. *Computers & Operations Research*, *103*, 46–63. DOI: 10.1016/j.cor.2018.10.010

| Rubric dimension | Score | Justification |
|---|---|---|
| Storage realism | 0 | No storage modeling |
| Shared bottleneck | **5** | Common server serves one machine at a time for setup operations — exactly a mutex/unary resource |
| Scheduling richness | **5** | Unrelated parallel machines, sequence-dependent setup times, machine eligibility, total weighted tardiness |
| Demand/inventory | 2 | Tardiness minimization only, no inventory or MTS |
| Method transferability | 4 | MILP model + TS/SA metaheuristics; machine environment is the closest match to the 5 roasters |
| **Total** | **16** | |

**Formulation summary.** The MILP uses assignment variables $x_{ijk} = 1$ if job $j$ is processed immediately after job $i$ on machine $k$, and server-synchronization variables that prevent two machines from using the common server simultaneously. The key constraint is: if machine $k$ requires setup for its next job during interval $[t, t+s_{ijk}]$, then no other machine $k'$ can have its server-assisted setup overlap that interval. Setup times $s_{ijk}$ are sequence- and machine-dependent. Objective: minimize $\sum_j w_j T_j$ (total weighted tardiness). Due to NP-hardness, tabu search and simulated annealing with a modified ATCS dispatching rule provide solutions for large instances.

**Problem match map:**

| Thesis constraint | Paper equivalent | Gap |
|---|---|---|
| 5 unrelated parallel roasters | $m$ unrelated parallel machines | ✅ Direct match |
| Sequence-dependent 5-min setup | Sequence-dependent setup times $s_{ijk}$ (machine-dependent) | ✅ Direct match (thesis uses uniform 5 min, simpler) |
| Machine eligibility (R3 cross-line) | Machine eligibility restrictions | ✅ Analogous |
| Common server / shared pipeline | Common server: only one machine setup at a time | ✅ Strong structural analog |
| Storage / inventory | Not modeled | ❌ |
| MTO tardiness | Total weighted tardiness | ✅ Direct match |
| MTS safety stock | Not modeled | ❌ |

**Gap analysis.** This paper provides the **strongest scheduling-layer match**: unrelated parallel machines with sequence-dependent setups and a shared server as mutex, minimizing weighted tardiness. The common-server constraint (one machine setup at a time) is structurally identical to the shared-pipeline constraint (one pipeline operation at a time). **What the thesis adds:** the common server in the thesis gates not just setups but three different operation types (consume, replenish, dump) with different durations; plus storage constraints and inventory coupling.

**Adaptation plan.** (1) The assignment and sequencing variables $x_{ijk}$ can be directly adopted for roaster scheduling. (2) The server-synchronization constraint can be generalized: instead of gating only setups, gate all three pipeline operations. In CP-SAT, this maps to creating interval variables for consume, replenish, and dump operations and placing them all on a single `NoOverlap` constraint representing the pipeline. (3) Add a second layer of constraints for storage/inventory on top of this scheduling skeleton.

**Cite reason:** "Bektur and Saraç (2019) formalized the MILP for unrelated parallel machines with sequence-dependent setups and a common server, providing the scheduling backbone we extend with storage constraints and inventory-penalty objectives."

---

### Paper 4 — Larroche, Bellenguez-Morineau & Massonnet (2022)

**Citation:** Larroche, F., Bellenguez-Morineau, O., & Massonnet, G. (2022). Clustering-based solution approach for a capacitated lot-sizing problem on parallel machines with sequence-dependent setups. *International Journal of Production Research*, *60*(21), 6573–6596. DOI: 10.1080/00207543.2021.1995792

| Rubric dimension | Score | Justification |
|---|---|---|
| Storage realism | 1 | No explicit storage tank or silo constraints |
| Shared bottleneck | 0 | No shared resource |
| Scheduling richness | **5** | Parallel (unrelated) machines, non-uniform sequence-dependent setups, lot sizing as decision variable, capacity with overtime |
| Demand/inventory | **5** | Explicit **safety stock violation penalties**, lost sales, inventory holding costs, deterministic demand |
| Method transferability | 4 | Two MIP formulations, Relax-and-Fix + Fix-and-Optimize heuristics; food industry origin |
| **Total** | **15** | |

**Formulation summary.** Binary production variables $y_{imt}$ indicate whether product $i$ is produced on machine $m$ in period $t$. Continuous lot-size variables $q_{imt}$ determine production quantities. Setup state variables $\gamma_{ijmt}$ track transitions between products. Inventory balance: $I_{it} = I_{i,t-1} + \sum_m q_{imt} - d_{it}$. **Safety stock deficit variable** $\delta_{it} = \max(0, SS_i - I_{it})$ is penalized in the objective. The objective minimizes $\sum_{i,t} (h_i I_{it} + p_i \delta_{it} + \pi_i L_{it} + \sum_m c_{ijm} \gamma_{ijmt})$ where $L_{it}$ is lost demand and $p_i$ is the safety stock violation penalty. A novel clustering method groups products with similar setup costs to reduce the combinatorial complexity of sequence-dependent setups.

**Problem match map:**

| Thesis constraint | Paper equivalent | Gap |
|---|---|---|
| Parallel roasters | Parallel machines with lot sizing | ✅ Direct match |
| Sequence-dependent setups | Non-uniform sequence-dependent setups $c_{ijm}$ | ✅ Direct match (thesis is simpler: uniform 5 min) |
| Safety stock threshold (10,000 kg) | Safety stock penalty variable $\delta_{it}$ | ✅ **Direct match** — identical modeling approach |
| MTS shortage penalty (600K VND/min) | Lost sales penalty $\pi_i L_{it}$ | ✅ Direct analog |
| MTO tardiness | Not modeled (pure MTS focus) | ❌ |
| Storage silos / no-mixing | Not modeled | ❌ |
| Shared pipeline | Not modeled | ❌ |
| Batch pool pre-generation | Lot size is continuous variable (not discrete batch count) | ⚠️ Different |

**Gap analysis.** This is the **demand/inventory champion**—the safety-stock penalty formulation ($\delta_{it}$ soft constraint) is exactly the approach needed for MTS shortage modeling. The parallel-machine environment with sequence-dependent setups also matches the scheduling layer. **What the thesis adds:** dedicated storage constraints, shared pipeline, MTO jobs with hard due dates, discrete batch sizes (not continuous lot sizes), and consumption events every 10 minutes.

**Adaptation plan.** (1) Adopt the inventory balance and safety stock deficit formulation directly: $I_{s,t} = I_{s,t-1} + \text{production}_{s,t} - \text{consumption}_{s,t}$ and $\delta_{s,t} \geq SS_s - \sum_k I_{s,k,t}$. (2) Modify the consumption term from a constant demand rate to discrete **10-minute consumption events** with known per-SKU rates. (3) Add MTO tardiness penalties as an additional objective term. (4) Replace continuous lot sizes with discrete batch activation binaries. (5) The Relax-and-Fix heuristic is directly applicable for the thesis MILP if CP-SAT struggles at scale.

**Cite reason:** "Larroche et al. (2022) introduced the safety-stock deficit penalization framework on parallel machines with sequence-dependent setups that we adopt for the MTS/PSC inventory objective, extending it with discrete 10-minute consumption events and MTO tardiness penalties."

---

### Paper 5 — Kopanos, Puigjaner & Georgiadis (2010)

**Citation:** Kopanos, G. M., Puigjaner, L., & Georgiadis, M. C. (2010). Optimal production scheduling and lot-sizing in dairy plants: The yogurt production line. *Industrial & Engineering Chemistry Research*, *49*(2), 701–718. DOI: 10.1021/ie901013k

| Rubric dimension | Score | Justification |
|---|---|---|
| Storage realism | 3 | Intermediate storage tanks between fermentation and packaging stages with capacity limits |
| Shared bottleneck | 3 | Parallel packaging units share common resources (fruit-mixers); resource-constrained formulation |
| Scheduling richness | 4 | Parallel units, product families, sequence-dependent changeover times and costs, lot-sizing, daily shutdown |
| Demand/inventory | 3 | Demand fulfillment, inventory costs, production planning integrated with scheduling |
| Method transferability | 4 | Mixed discrete/continuous-time MILP; real-life dairy plant case study; solves large instances to optimality |
| **Total** | **17** | |

**Formulation summary.** Products are grouped into families. Binary immediate-precedence variables $z_{ff'mt}$ indicate that family $f'$ is produced immediately after family $f$ on machine $m$ in period $t$. Lot-size variables $q_{fmt}$ determine quantities per family per machine per period. Changeover time constraint: if families $f$ and $f'$ are sequenced consecutively, a setup time $\sigma_{ff'}$ is added. Storage balance: intermediate tank inventory tracked across time periods with capacity bounds. Shared resource constraint: at most one packaging unit can use the fruit-mixer at any time. Objective minimizes total production, changeover, and inventory costs.

**Problem match map:**

| Thesis constraint | Paper equivalent | Gap |
|---|---|---|
| Parallel roasters across 2 lines | Parallel packaging units sharing resources | ✅ Structural analog |
| Product families / SKU types | Product families (yogurt types) | ✅ Direct analog |
| Sequence-dependent changeovers | Sequence-dependent changeover times $\sigma_{ff'}$ | ✅ Direct match |
| Intermediate storage (tanks) | Fermentation tanks with capacity | ✅ Partial analog to GC/RC silos |
| Shared resource (fruit-mixer) | Shared fruit-mixer constraint | ✅ Analog to shared pipeline |
| Lot sizing | Production quantity decision | ✅ Analog to batch count decision |
| Downstream packaging consumption | Packaging stage demand | ✅ Structural analog — packaging draws from storage |
| No-mixing (dedicated silo) | Tanks can hold multiple products (not dedicated) | ❌ |
| MTO/MTS mix | Not modeled (all MTS-like) | ❌ |

**Gap analysis.** This is the **best food-industry structural analog**—a real dairy plant with parallel processing units, shared resources, intermediate storage, product families with changeovers, and downstream packaging demand. The multi-stage structure (processing → storage → packaging) mirrors the thesis's (roasting → RC silos → PSC packaging line). **What the thesis adds:** strictly dedicated silos with no-mixing, explicit MTO/MTS demand mix, safety stock penalties, and the three-operation pipeline mutex.

**Adaptation plan.** (1) Use the product-family-based sequencing framework to group coffee SKUs into families, reducing binary variable count. (2) Adapt the intermediate storage tracking to dedicated RC silos with the consumption-priority logic (buffer silo first, then highest-stock). (3) Use the shared-resource constraint pattern for the GC pipeline. (4) Extend the packaging-stage demand model to include 10-minute discrete consumption events with shortage penalties.

**Cite reason:** "Kopanos et al. (2010) demonstrated a production-scheduling-and-lot-sizing MILP for a real-life multi-stage dairy facility with shared resources, intermediate storage, and downstream packaging—the closest food-industry analog to the roasting-storage-packaging structure of the Nestlé Trị An plant."

---

## Skim list: papers scoring 10–14

### 6. Kondili, Pantelides & Sargent (1993) — STN foundation

Kondili, E., Pantelides, C. C., & Sargent, R. W. H. (1993). A general algorithm for short-term scheduling of batch operations—I. MILP formulation. *Computers & Chemical Engineering*, *17*(2), 211–227. DOI: 10.1016/0098-1354(93)80015-F

The foundational State-Task Network paper. Introduces discrete-time MILP with allocation binaries $w_{i,n}$ (task $i$ starts at time $n$) and storage state variables $S_{s,n}$. Directly provides the discrete-time indexing scheme and material balance framework that underpins any time-indexed formulation of the thesis problem. **Essential Chapter 2 anchor for introducing batch scheduling formulations.**

### 7. Méndez, Cerdá, Grossmann, Harjunkoski & Fahl (2006) — Definitive review

Méndez, C. A., Cerdá, J., Grossmann, I. E., Harjunkoski, I., & Fahl, M. (2006). State-of-the-art review of optimization methods for short-term scheduling of batch processes. *Computers & Chemical Engineering*, *30*(6–7), 913–946. DOI: 10.1016/j.compchemeng.2006.02.008

The most comprehensive review of batch scheduling. Classifies storage policies (UIS/FIS/NIS/ZW/dedicated), time representations (discrete vs. continuous), and formulation types. Essential for positioning the thesis within the literature and justifying the choice of discrete-time slots. **Chapter 2 anchor for literature taxonomy and storage policy classification.**

### 8. Janak, Lin & Floudas (2004) — Mixed storage policies

Janak, S. L., Lin, X., & Floudas, C. A. (2004). Enhanced continuous-time unit-specific event-based formulation for short-term scheduling of multipurpose batch processes: Resource constraints and mixed storage policies. *Industrial & Engineering Chemistry Research*, *43*(10), 2516–2533. DOI: 10.1021/ie0341597

Explicitly handles **mixed storage policies** (UIS, FIS, NIS, ZW) and shared discrete resources in a single continuous-time MILP. The dedicated-storage (FIS) formulation with resource constraints beyond equipment is directly relevant to encoding the silo assignment and pipeline constraints. **Key reference for comparing FIS formulation approaches.**

### 9. Laborie, Rogerie, Shaw & Vilím (2018) — CP Optimizer scheduling primitives

Laborie, P., Rogerie, J., Shaw, P., & Vilím, P. (2018). IBM ILOG CP Optimizer for scheduling. *Constraints*, *23*(2), 210–250. DOI: 10.1007/s10601-018-9281-x

Describes interval variables, no-overlap constraints, cumulative functions, and state functions—the modeling primitives that CP-SAT inherits. Essential for explaining why CP-SAT is an appropriate solver: `new_interval_var` for batch operations, `add_no_overlap` for pipeline mutex and roaster sequencing, `add_cumulative` for silo capacity tracking. **Chapter 3 anchor for explaining the CP-SAT modeling approach.**

### 10. Maravelias & Grossmann (2004) — Hybrid MILP/CP for batch plants

Maravelias, C. T., & Grossmann, I. E. (2004). A hybrid MILP/CP decomposition approach for the continuous time scheduling of multipurpose batch plants. *Computers & Chemical Engineering*, *28*(10), 1921–1949. DOI: 10.1016/j.compchemeng.2004.03.016

Decomposes batch scheduling into MILP master (assignment/batch sizing) and CP subproblem (sequencing/timing). Handles multiple storage policies and resource constraints. **Key reference if the thesis adopts a decomposition strategy—MILP for silo-assignment master, CP-SAT for scheduling subproblem.**

### 11. Soman, van Donk & Gaalman (2004) — MTO/MTS food framework

Soman, C. A., van Donk, D. P., & Gaalman, G. (2004). Combined make-to-order and make-to-stock in a food production system. *International Journal of Production Economics*, *90*(2), 223–235. DOI: 10.1016/S0925-5273(02)00376-6

Develops a hierarchical planning framework for **hybrid MTO/MTS in food processing** with a bottleneck between processing and packaging stages—directly analogous to the thesis's roaster-to-packaging coupling. Uses run-out-time scheduling heuristics for MTS. **Chapter 2 anchor for justifying the MTO/MTS framework in food manufacturing.**

### 12. Absi & Kedad-Sidhoum (2009) — Safety stock as soft constraint

Absi, N., & Kedad-Sidhoum, S. (2009). The multi-item capacitated lot-sizing problem with safety stocks and demand shortage costs. *Computers & Operations Research*, *36*(11), 2926–2936. DOI: 10.1016/j.cor.2009.01.007

Treats safety stock as an **objective-function penalty** (soft constraint with deficit variable $\delta_{it}$) rather than a hard constraint—exactly the approach needed for the thesis MTS penalty structure. Uses Lagrangian relaxation of capacity constraints. **Key formulation reference for encoding the 10,000 kg safety stock threshold as a penalty.**

### 13. Naderi, Ruiz & Roshanaei (2023) — MIP vs CP benchmark

Naderi, B., Ruiz, R., & Roshanaei, V. (2023). Mixed-integer programming vs. constraint programming for shop scheduling problems: New results and outlook. *INFORMS Journal on Computing*, *35*(4), 817–843. DOI: 10.1287/ijoc.2023.1287

Systematic comparison of MIP and CP across 12 scheduling variants showing CP dominates for most scheduling problems at scale. Provides the empirical justification for choosing CP-SAT as primary solver over MILP. **Chapter 3 anchor for defending the dual CP-SAT/MILP methodology.**

### 14. Harjunkoski et al. (2014) — Industrial scheduling applications

Harjunkoski, I., Maravelias, C. T., Bongers, P., Castro, P. M., Engell, S., Grossmann, I. E., Hooker, J., Méndez, C., Sand, G., & Wassick, J. (2014). Scope for industrial applications of production scheduling models and solution methods. *Computers & Chemical Engineering*, *62*, 161–193. DOI: 10.1016/j.compchemeng.2013.12.001

Discusses industrial scheduling across chemicals, food/consumer goods, steel, and pharmaceuticals with practical lessons from Dow, Unilever, and BASF. Covers dedicated storage, shared infrastructure, and gap between academic models and real implementation. **Chapter 1/2 anchor for motivating industrial relevance.**

### 15. Sato, Maeda, Toshima, Nagasawa, Morikawa & Takahashi (2023) — MTO/MTS switching

Sato, Y., Maeda, H., Toshima, R., Nagasawa, K., Morikawa, K., & Takahashi, K. (2023). Switching decisions in a hybrid MTS/MTO production system comprising multiple machines considering setup. *International Journal of Production Economics*, *263*, 108877. DOI: 10.1016/j.ijpe.2023.108877

Models **multiple parallel machines** that dynamically switch between MTS and MTO modes with setup costs. MDP-based. Key finding: switching one machine at a time is sufficient. **Provides the dynamic switching intuition for the thesis's R3 cross-line routing decision.**

### 16. Marchetti & Cerdá (2009) — Resource-constrained batch scheduling

Marchetti, P. A., & Cerdá, J. (2009). A general resource-constrained scheduling framework for multistage batch facilities with sequence-dependent changeovers. *Computers & Chemical Engineering*, *33*(4), 871–886. DOI: 10.1016/j.compchemeng.2008.12.007

Introduces overlapping variables to track parallel resource usage for unary (mutex) and finite-capacity resources alongside sequence-dependent changeovers. The overlapping-variable approach for shared resources can encode the pipeline mutex elegantly in MILP. **Formulation reference for the MILP encoding of the pipeline constraint.**

### 17. Hooker (2007) — Logic-based Benders decomposition

Hooker, J. N. (2007). Planning and scheduling by logic-based Benders decomposition. *Operations Research*, *55*(3), 588–602. DOI: 10.1287/opre.1060.0371

Seminal paper on LBBD combining MILP master (allocation) with CP subproblem (scheduling). If the thesis problem proves intractable for monolithic CP-SAT, LBBD offers a principled decomposition: MILP assigns batches to roasters and silos to SKUs, while CP-SAT sequences and times operations. **Key fallback method reference if CP-SAT alone cannot solve full instances.**

---

## Method map: which papers to read by solver choice

| If you choose… | Read these first | They provide… |
|---|---|---|
| **CP-SAT (primary)** | Laborie et al. (2018), Naderi et al. (2023), Bektur & Saraç (2019) | Interval variables/no-overlap primitives; CP vs MIP benchmarks; common-server as no-overlap constraint |
| **MILP (reference)** | Terrazas-Moreno et al. (2012), Ferrer-Nadal et al. (2008), Larroche et al. (2022) | Tank assignment binaries; transfer-sync mutex in MILP; safety stock penalty formulation |
| **Hybrid MILP+CP decomposition** | Maravelias & Grossmann (2004), Hooker (2007) | LBBD framework; assignment master + sequencing subproblem |
| **Heuristic/metaheuristic** | Bektur & Saraç (2019), Larroche et al. (2022) | Tabu search/SA for common-server scheduling; Relax-and-Fix for lot-sizing |
| **Rolling horizon** | Larroche et al. (2022), Harjunkoski et al. (2014) | Fix-and-Optimize for large horizons; industrial rolling horizon practices |

---

## Suggested Chapter 2 outline

**Section 2.1 — Batch scheduling foundations** anchored by Kondili et al. (1993) for the STN framework and Méndez et al. (2006) for the comprehensive taxonomy of time representations and storage policies. This section establishes the discrete-time MILP foundation and classifies the thesis problem within the UIS/FIS/NIS/dedicated storage spectrum.

**Section 2.2 — Scheduling with dedicated storage and tank assignment** anchored by Terrazas-Moreno et al. (2012) for dedicated-tank MILP formulation and Janak et al. (2004) for mixed storage policies with resource constraints. This section develops the silo-assignment formulation with no-mixing and capacity constraints.

**Section 2.3 — Shared resources and pipeline constraints** anchored by Ferrer-Nadal et al. (2008) for material-transfer mutex constraints and Bektur & Saraç (2019) for common-server parallel-machine scheduling. Marchetti & Cerdá (2009) provides the overlapping-variable MILP encoding. This section formalizes the three-operation pipeline mutex.

**Section 2.4 — Integrated production scheduling and inventory management** anchored by Larroche et al. (2022) for safety-stock penalties on parallel machines, Absi & Kedad-Sidhoum (2009) for the soft-constraint safety-stock formulation, and Kopanos et al. (2010) for food-industry scheduling with storage and packaging demand. This section establishes the inventory-coupling formulation and penalty structure.

**Section 2.5 — Hybrid MTO/MTS production systems** anchored by Soman et al. (2004) for the food-industry MTO/MTS framework and Sato et al. (2023) for dynamic switching on parallel machines. This section justifies the mixed MTO/MTS objective with tardiness and shortage penalties.

**Section 2.6 — Solution methods: CP-SAT and decomposition** anchored by Laborie et al. (2018) for CP scheduling primitives, Naderi et al. (2023) for MIP-vs-CP benchmarking, and Maravelias & Grossmann (2004) and Hooker (2007) for hybrid decomposition. Harjunkoski et al. (2014) provides industrial context. This section justifies CP-SAT as primary solver and MILP as mathematical reference.

**Section 2.7 — Research gap and positioning.** Synthesizes the preceding sections to show that no existing paper simultaneously addresses dedicated-silo pipeline mutex + cross-line routing + MTO/MTS inventory coupling + batch-pool activation + SKU reassignment. The thesis fills this gap by assembling formulation elements from the above streams into a unified CP-SAT model for the coffee roasting domain—an application area with almost no formal scheduling optimization literature.

---

## Verification notes

All 17 papers above have been verified via Google Scholar, publisher pages (ACS Publications, ScienceDirect, Springer, INFORMS), or institutional repositories. DOIs have been confirmed through cross-referencing citation records. The Giraldo H. (2020) coffee-roasting paper mentioned in some search results (DOI: 10.1007/978-3-030-23816-2_46) was not included in the Top 5 because it is a short conference proceeding with limited formulation detail, but it could be cited as the only known coffee-specific scheduling reference if desired.