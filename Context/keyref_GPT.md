# Research-Backed Reference Set for a Buffer-Coupled, Disruption-Prone Batch Scheduling Thesis

## Problem-to-literature mapping

Your thesis problem has an unusually “coupled” structure: batches must be assigned/sequenced on **multiple parallel batch machines** (roasters), but feasibility and performance depend just as much on (i) **sequence-dependent setup/cleaning** on the roasters and (ii) **intermediate inventory/buffer dynamics** downstream (to avoid stockouts, congestion, or starvation). On top of that, stochastic **downtime/disruptions** pushes the work from “static scheduling” into **reactive/rescheduling** or **stochastic/robust evaluation**.

Across the scheduling literature, this combination most naturally maps to three overlapping modeling traditions:

- **Hybrid/flexible flow shop with limited buffers (LB)**: strong on parallel machines, blocking/limited buffers, sequence/assignment decisions, and metaheuristics tailored to NP-hardness (often with multiobjective variants). Examples include re-entrant and stocker-buffer variants that are structurally similar to “roast → buffer → pack” lines. citeturn25view1turn11view1turn28view2turn11view2  
- **Process-systems short-term batch scheduling (STN/RTN MILP)**: strong on **material/inventory coupling**, storage policies, batch sizing, and multi-stage batch-network structure; also the most direct route to a formulation that “looks like” a roasting + buffer + downstream line. citeturn27view0turn29view0  
- **Reactive scheduling under disruptions**: focuses on how to update a baseline schedule when **machine breakdowns** and **rush orders** occur, often with MILP-based rescheduling frameworks. citeturn29view0turn26view1  

The 10 papers below were selected to collectively cover **all five** of your defining features (parallel machines, sequence-dependent setups, intermediate buffers/inventory, discrete batch processing, and disruptions/downtime), while also meeting your “≥5 papers after 2018” constraint (7/10 are 2019+).

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["hybrid flow shop limited buffer diagram","state task network batch plant scheduling diagram","coffee roasting production line roaster cooling packaging buffer"],"num_per_query":1}

## Search and selection method

Selection emphasized papers that *explicitly* model (not just mention) at least two of these: **limited buffers / inventory coupling**, **sequence-dependent setups/changeovers**, **parallel units per stage**, **finite-horizon short-term scheduling**, and **disruptions (breakdowns, maintenance windows, rush orders, stochasticity)**.

To keep the set “thesis-useful,” preference was given to papers that provide either:
- a **mathematical programming formulation** (MILP/MIP, continuous-time, discrete-time), or  
- a **well-specified algorithmic framework** (metaheuristic with clear encoding/decoding, simheuristics), together with performance evidence.

Sources used are publisher pages, university research portals, and accessible bibliographic/abstract records for each paper. citeturn25view1turn28view3turn28view2turn11view2turn33view0turn28view0turn28view1turn26view1turn29view0turn27view0  

## Annotated list of ten closely related papers

### Re-entrant / hybrid flow shop with limited buffers and setups

**Paper 1**  
**APA citation.** entity["people","Qianqian Zheng","author; rhfsp 2024"], entity["people","Yu Zhang","author; shop scheduling"], entity["people","Hongwei Tian","author; flow shop"], & entity["people","Lijun He","author; metaheuristics"]. (2024). *A cooperative adaptive genetic algorithm for reentrant hybrid flow shop scheduling with sequence-dependent setup time and limited buffers*. **Complex & Intelligent Systems, 10**, 781–809. https://doi.org/10.1007/s40747-023-01147-8 citeturn25view1  
**Problem type.** Re-entrant **hybrid flow shop** (multiple stages with unrelated parallel machines) with **SDST** and **limited buffers**. citeturn25view1  
**Key constraints modeled.** Re-entrance; unrelated parallel machines per stage; **sequence-dependent setup time**; **limited intermediate buffers**. citeturn25view1  
**Solution method.** Mathematical model + **cooperative adaptive genetic algorithm (CAGA)**. citeturn12search11turn25view1  
**Why it’s relevant to roasting.** This is one of the closest direct matches to “multiple parallel roasters + setup-dependent changeovers + finite intermediate buffer.” Even if your roasting line is not formally “re-entrant,” the *buffer + SDST + parallel machines* coupling is highly analogous. citeturn25view1  

**Paper 2**  
**APA citation.** entity["people","Christian Klanke","author; make-and-pack 2021"], entity["people","Vassilios Yfantis","author; process scheduling"], entity["people","Francesc Corominas","author; operations research"], & entity["people","Sebastian Engell","process systems engineer"]. (2021). *Short-term scheduling of make-and-pack processes in the consumer goods industry using discrete-time and precedence-based MILP models*. **Computers & Chemical Engineering, 154**, 107453. https://doi.org/10.1016/j.compchemeng.2021.107453 citeturn28view3  
**Problem type.** Two-stage **make-and-pack** short-term scheduling with a **finite intermediate buffer** and **sequence-dependent changeovers**. citeturn11view1  
**Key constraints modeled.** Finite intermediate buffer; sequence-dependent changeovers; stage-dependent bottlenecks; short-term horizon; coupling/decoupling decisions via buffer. citeturn11view1  
**Solution method.** **Discrete-time MILP** plus a second **precedence-based MILP** in a **two-step + decomposition strategy** to address tractability on realistic horizons. citeturn11view1  
**Why it’s relevant to roasting.** Coffee roasting lines are often “process stage → buffer → downstream stage,” with the buffer acting as a decoupler but also a constraint. This paper is especially valuable as a *playbook for large-scale, shift-horizon MILP scheduling when intermediate storage is central and the full model is otherwise intractable*. citeturn11view1  

**Paper 3**  
**APA citation.** Qian-Qian Zheng, Yu Zhang, Hong-Wei Tian, & Li-Jun He. (2021). *An effective hybrid meta-heuristic for flexible flow shop scheduling with limited buffers and step-deteriorating jobs*. **Engineering Applications of Artificial Intelligence, 106**, 104503. https://doi.org/10.1016/j.engappai.2021.104503 citeturn28view2  
**Problem type.** **Flexible flow shop** with **limited buffers** and multiple non-identical parallel machines. citeturn28view2  
**Key constraints modeled.** Limited intermediate buffers; non-identical parallel machines; step deterioration (time-dependent processing extension); multiobjective criteria. citeturn28view2  
**Solution method.** Mixed-integer model + a hybrid metaheuristic (GVNSA) combining **GA + VNS + SA**, with explicit encoding/decoding and embedded heuristics. citeturn28view2  
**Why it’s relevant to roasting.** Even if you do not use step deterioration, this paper is a strong template for: (i) buffer-limited flow-shop modeling, (ii) dual objectives involving **tardiness** plus another operational measure, and (iii) practical decoding under buffer constraints (often the “hard part” for metaheuristics). citeturn28view2  

**Paper 4**  
**APA citation.** entity["people","Chun-Cheng Lin","author; reentrant stockers 2020"], entity["people","Wan-Yu Liu","author; manufacturing scheduling"], & entity["people","Yu-Hsiang Chen","author; industrial engineering"]. (2020). *Considering stockers in reentrant hybrid flow shop scheduling with limited buffer capacity*. **Computers & Industrial Engineering, 139**, 106154. https://doi.org/10.1016/j.cie.2019.106154 citeturn11view2turn28view4  
**Problem type.** Re-entrant **hybrid flow shop** with **limited buffers** and centralized buffers (“stockers”). citeturn11view2turn28view4  
**Key constraints modeled.** Limited buffer capacity at machines plus stockers (centralized inventory buffer space); re-entrance; multi-stage routing; NP-hard objective setting. citeturn11view2turn28view4  
**Solution method.** Hybrid metaheuristic combining **harmony search + genetic algorithm (HHSGA)**; explicitly notes decoding complexity due to limited buffers/stockers. citeturn11view2  
**Why it’s relevant to roasting.** If your roasting line has (or could have) a “central WIP buffer” that feeds multiple downstream operations, the stocker abstraction is close to “roasted-bean buffer bins/silos” that smooth roaster-to-packaging mismatch. citeturn11view2turn28view4  

### Hybrid flow shop with limited buffers and performance tradeoffs

**Paper 5**  
**APA citation.** entity["people","Shenglong Jiang","author; hfsp energy 2019"] & Long Zhang. (2019). *Energy-oriented scheduling for hybrid flow shop with limited buffers through efficient multi-objective optimization*. **IEEE Access, 7**, 34477–34487. https://doi.org/10.1109/ACCESS.2019.2904848 citeturn33view0turn19view0  
**Problem type.** **Hybrid flow shop** with **limited buffers**, treated as a multiobjective scheduling environment. citeturn19view0  
**Key constraints modeled.** Limited intermediate buffers; hybrid flow-shop structure; explicit buffering constraints; multiobjective performance (tardiness + energy). citeturn19view0  
**Solution method.** MILP formulation + **MOEA/D-style** multiobjective evolutionary optimization with decoding that uses discrete-event simulation and post-shift logic. citeturn19view0  
**Why it’s relevant to roasting.** Even if energy is not your main objective, the paper is useful for **multiobjective design under limited buffers** and for the “simulation-assisted decoding” idea, which is often directly transferable when buffer evolution is hard to linearize tightly. citeturn19view0  

### Stochasticity, downtime, and maintenance under limited buffers

**Paper 6**  
**APA citation.** entity["people","Rooeinfar R","author; stochastic ffs 2019"], entity["people","Raissi S","author; simulation optimization"], & entity["people","Ghezavati V R","author; industrial engineering"]. (2019). *Stochastic flexible flow shop scheduling problem with limited buffers and fixed interval preventive maintenance: a hybrid approach of simulation and metaheuristic algorithms*. **Simulation, 95**(6), 509–528. https://doi.org/10.1177/0037549718809542 citeturn28view0  
**Problem type.** **Stochastic flexible flow shop** with **limited buffers** and **maintenance-driven downtime**. citeturn28view0  
**Key constraints modeled.** Limited buffers; uncertainty/stochastic decision setting; fixed-interval preventive maintenance (explicit downtime); comparative evaluation across methods. citeturn28view0  
**Solution method.** Hybrid “HSIM-META”: **simulation outputs used within metaheuristics** (GA, SA, PSO), and comparisons against non-simulation variants. citeturn28view0  
**Why it’s relevant to roasting.** This is directly aligned with your “stochastic disruptions/downtime” requirement, and it explicitly shows how to combine **buffer constraints** with **downtime/maintenance** in a computational approach that remains usable when exact models become brittle or slow. citeturn28view0  

**Paper 7**  
**APA citation.** entity["people","R. Wallrath","author; time-bucket milp 2023"], entity["people","F. Seeanner","author; batch optimization"], entity["people","M. Lampe","author; process scheduling"], & entity["people","M. B. Franke","author; chemical engineering"]. (2023). *A time-bucket MILP formulation for optimal lot-sizing and scheduling of real-world chemical batch plants*. **Computers & Chemical Engineering, 177**, 108341. https://doi.org/10.1016/j.compchemeng.2023.108341 citeturn28view1  
**Problem type.** **Multi-stage batch plant** lot-sizing + scheduling with a time representation designed to scale. citeturn28view1  
**Key constraints modeled.** Multi-stage manufacturing; industrial lot-sizing/scheduling integration; time-bucket representation combining fixed macroperiods with flexible microperiods (bridging discrete and continuous-time strengths). citeturn28view1  
**Solution method.** **Time-bucket MILP**, with explicit emphasis on scalability and parameter effects; includes a real-world case study (formulation/filling). citeturn28view1  
**Why it’s relevant to roasting.** If your thesis needs to integrate “daily/shift planning” with “within-shift sequencing,” time-bucket MILP is a strong structural option: it provides a natural place to model **buffer inventory** as balance constraints across buckets while still capturing within-bucket sequencing decisions at a workable resolution. citeturn28view1  

**Paper 8**  
**APA citation.** entity["people","M. Gholami","author; hfsp breakdowns"], entity["people","M. Zandieh","author; scheduling"], & entity["people","A. Alem-Tabriz","author; manufacturing"]. (2009). *Scheduling hybrid flow shop with sequence-dependent setup times and machines with random breakdowns*. **The International Journal of Advanced Manufacturing Technology, 42**, 189–201. https://doi.org/10.1007/s00170-008-1577-3 citeturn26view1  
**Problem type.** **Hybrid flow shop** with both **sequence-dependent setup times** and **stochastic/random breakdowns**. citeturn26view1  
**Key constraints modeled.** SDST; random breakdown behavior; expected-performance reasoning (the paper positions breakdowns as a first-class modeling element in the scheduling problem). citeturn21view9turn26view1  
**Solution method.** Evolutionary/metaheuristic approach (genetic algorithm-based) for an NP-hard hybrid flow shop with breakdowns and SDST. citeturn21view9turn26view1  
**Why it’s relevant to roasting.** This paper is the cleanest bridge between your **setup-driven roasting changeovers** and **stochastic disruptions** on roasters, in a flow-shop-like setting. Even if your final formulation uses different machinery, the modeling logic for “expected performance under breakdowns + SDST sequencing” is directly applicable. citeturn26view1  

### Reactive rescheduling in batch plants when disruptions happen

**Paper 9**  
**APA citation.** entity["people","Jeetmanyu P. Vin","author; rescheduling 2000"] & entity["people","Marianthi G. Ierapetritou","chemical engineer; scheduling"]. (2000). *A new approach for efficient rescheduling of multiproduct batch plants*. **Industrial and Engineering Chemistry Research, 39**(11), 4228–4238. https://doi.org/10.1021/ie000233z citeturn29view0  
**Problem type.** **Reactive/rescheduling** in multiproduct batch plants. citeturn29view0  
**Key constraints modeled.** Disturbances explicitly include **machine breakdown** and **rush order arrival**; rescheduling is formulated as a continuous-time MILP for computational efficiency and policy closeness to the base schedule. citeturn29view0  
**Solution method.** Two-stage approach: compute deterministic schedule, then solve an **MILP-based rescheduling** model that systematically considers rescheduling alternatives. citeturn29view0  
**Why it’s relevant to roasting.** If your thesis must produce **a realizable within-shift schedule** and also define how to **repair/re-optimize** after downtime events, this is a direct methodological guide for structuring the “baseline schedule → disturbance → reschedule” workflow with objective terms that trade off profitability/service vs stability. citeturn29view0  

### Core batch-scheduling formulation with inventory coupling and sequence-dependent changeovers

**Paper 10**  
**APA citation.** entity["people","Christos T. Maravelias","chemical engineering; scheduling"] & entity["people","Ignacio E. Grossmann","chemical engineer; optimization"]. (2003). *New general continuous-time state–task network formulation for short-term scheduling of multipurpose batch plants*. **Industrial and Engineering Chemistry Research, 42**(13), 3056–3074. https://doi.org/10.1021/ie020923y citeturn27view0  
**Problem type.** General **multipurpose batch plant** short-term scheduling using a continuous-time **state–task network (STN)** MILP. citeturn27view0  
**Key constraints modeled.** Variable batch sizes/processing times; **resource/utility constraints**; multiple storage policies; **batch mixing/splitting**; **sequence-dependent changeover times**; plus valid inequalities to strengthen the LP relaxation. citeturn27view0  
**Solution method.** Continuous-time STN **MILP formulation**, positioned as general and computationally efficient relative to other STN/event-driven formulations. citeturn27view0  
**Why it’s relevant to roasting.** Your roasting line is naturally expressible as a material-transforming network with storage/buffer states. This paper provides a thesis-grade “mother formulation” for inventory-coupled batch scheduling—exactly the modeling layer you need to enforce buffer feasibility and represent changeovers rigorously. citeturn27view0  

## Comparison matrix against the thesis needs

Legend: ✅ = explicitly modeled; ◐ = partially/indirectly addressed; — = not the main focus / not explicit in the cited source.

| Paper | Year | Primary modeling “lens” | Parallel machines | Seq.-dep. setup/changeover | Intermediate buffer / inventory coupling | Disruptions / downtime (stochastic or explicit) | Batch / discrete processing | Main objective examples (as stated) | Method type | Optimality / convergence signal | Fit vs. roasting thesis |
|---|---:|---|:---:|:---:|:---:|:---:|:---:|---|---|---|---|
| Zheng et al. | 2024 | Re-entrant hybrid flow shop | ✅ | ✅ | ✅ | — | ✅ | Total weighted completion time (stated) | Metaheuristic (CAGA) | Heuristic (no global optimality guarantee stated) | **Very high**: SDST + LB + parallel machines citeturn25view1 |
| Klanke et al. | 2021 | Make-and-pack, short-term scheduling | ✅ (multi-line) | ✅ (changeovers) | ✅ (finite buffer) | — | ◐ (process modeled as stages; scheduling is discrete) | Productivity/downtime improvements; short-term feasibility | MILP + decomposition | Exact per subproblem; decomposition for tractability | **High**: finite buffer + changeovers + shift-horizon framing citeturn11view1 |
| Zheng et al. | 2021 | Flexible flow shop (LB) | ✅ | — | ✅ | ◐ (time-dependence via deterioration) | ✅ | Makespan + total tardiness (stated) | MIP + hybrid metaheuristic (GVNSA) | Heuristic | **High**: LB + tardiness + encoding/decoding guidance citeturn28view2 |
| Lin et al. | 2020 | Re-entrant hybrid flow shop with stockers | ✅ | — | ✅ (buffers + stockers) | — | ✅ | Makespan + mean flowtime (stated) | Hybrid harmony search + GA | Heuristic | **Medium–high**: strong buffer mechanics (central buffer concept) citeturn11view2turn28view4 |
| Jiang & Zhang | 2019 | Hybrid flow shop (LB) multiobjective | ✅ | — | ✅ | — | ✅ | Total weighted tardiness + non-processing energy (stated) | MILP + MOEA/D-style evolutionary + simulation decoding | Heuristic (Pareto search) | **Medium–high**: strong LB handling; offers sim-based decoding idea citeturn19view0turn33view0 |
| Rooeinfar et al. | 2019 | Stochastic flexible flow shop + LB + PM | ✅ | — | ✅ | ✅ (PM downtime + stochasticity) | ✅ | Comparative performance (not a single stated objective in snippet) | Simulation + GA/SA/PSO hybrids | Heuristic; evaluated via simulation | **High for disruptions**: shows how to combine LB with downtime/uncertainty citeturn28view0 |
| Wallrath et al. | 2023 | Batch plant lot-sizing + scheduling | ◐ (multi-stage resources) | — | ◐ (inventory via lot-sizing/scheduling coupling) | — | ✅ | Industrial lot-sizing/scheduling objectives (implied) | Time-bucket MILP | Exact MILP (subject to size) | **Medium–high**: strong template for shift-horizon + inventory balances citeturn28view1 |
| Gholami et al. | 2009 | Hybrid flow shop with SDST + breakdowns | ✅ | ✅ | — | ✅ (random breakdowns) | ✅ | Expected makespan-style objective framing | GA-based metaheuristic | Heuristic | **High for disruption+SDST**: bridges setup sequencing and stochastic downtime citeturn26view1turn21view9 |
| Vin & Ierapetritou | 2000 | Reactive rescheduling of batch plants | ✅ (multipurpose plant units) | ◐ (via MILP sequencing; not emphasized in snippet) | ◐ (batch plant material coupling) | ✅ (breakdown + rush orders) | ✅ | Profitability/operability; closeness to base schedule | Continuous-time MILP rescheduling | Exact MILP per rescheduling solve | **Very high for “reactive layer”**: a direct rescheduling template under breakdowns citeturn29view0 |
| Maravelias & Grossmann | 2003 | STN continuous-time batch scheduling | ✅ | ✅ | ✅ (storage policies; inventories) | — | ✅ | General ST scheduling; due dates/changeovers supported | Continuous-time STN MILP | Exact MILP (subject to size) | **Very high for formulation**: strongest “inventory-coupled batch MILP” backbone citeturn27view0 |

Interpretation note: Several flow-shop papers treat buffers as “limited WIP between stages” (often inducing blocking/starvation logic), while batch-plant STN papers treat buffers as explicit **material states** with **inventory balance constraints**. For your roasting system, both viewpoints can be useful: WIP blocking logic captures physical congestion constraints, while explicit inventory states capture “avoid stockout” targets. citeturn11view2turn27view0  

## Key methodological reference selection

### Recommended key reference

**Key methodological reference for the thesis:** **Maravelias & Grossmann (2003)** *New general continuous-time state–task network formulation for short-term scheduling of multipurpose batch plants*. citeturn27view0  

### Justification against your criteria

**Similarity of constraints.** This paper explicitly supports (within one coherent MILP framework) the kinds of constraints that dominate a coffee roasting line model: multiple batch units, material states, storage policies (your buffer), and **sequence-dependent changeovers**. citeturn27view0  

**Modeling structure.** The **STN** representation is a direct conceptual match to “green coffee → roasting task → roasted coffee buffer → downstream tasks,” because it forces you to represent **inventory coupling** (buffer feasibility) through state balances instead of treating buffers as an afterthought. citeturn27view0  

**Solution approach.** It is a **continuous-time MILP** designed for short-term scheduling and includes formulation-strengthening ideas (valid inequalities, tightened matching structure) intended to improve computational behavior—useful for finite shift horizons where you may want high fidelity. citeturn27view0  

**Citation impact.** The Princeton-hosted record reports **Scopus citations (279)** for this article, indicating sustained impact and high visibility in the batch scheduling community. citeturn27view0  

## How the key paper can guide your formulation and solution approach

### Formulation blueprint for a roasting line

A practical way to use Maravelias & Grossmann as your “formulation backbone” is to translate the roasting line into an STN with:

- **States (inventories):** green-bean lots, roasted-bean buffer (possibly by roast level/SKU), packaging-ready buffer, finished goods. The STN model explicitly accommodates storage policies and resource constraints as part of the scheduling formulation. citeturn27view0  
- **Tasks (operations):** roast batch on roaster \(r\), cool/degass (if modeled), transfer to buffer, package batch, QA/cleaning tasks as needed. The STN framework is intended to model batch tasks that transform states and consume resources. citeturn27view0  
- **Resources / units:** each roaster is a unit; packaging lines are downstream units; utilities (e.g., labor teams, shared conveyors) can be modeled as non-equipment resources. citeturn27view0turn11view1  

In this structure, your **intermediate buffer constraints** become inventory bounds on the roasted-bean state (capacity) and (if you want stockout avoidance) lower bounds / safety stocks or penalty terms.

### How to incorporate the thesis-specific “disruption layer”

The Maravelias–Grossmann backbone is deterministic. To address your requirement of **stochastic disruptions/downtime**, you can add a second, explicitly reactive layer guided by Vin & Ierapetritou (2000), which provides a template for:

1. computing a baseline schedule, then  
2. rescheduling via an MILP when a disruption occurs—explicitly including **machine breakdown** and **rush order arrival** disturbances. citeturn29view0  

If you want stochasticity studied “a priori,” Rooeinfar et al. (2019) demonstrates a pattern where buffer-limited flow-shop schedules are evaluated and improved using simulation integrated with metaheuristics under preventive-maintenance downtime and uncertainty. citeturn28view0  

### How this guides your solution method choices

A thesis-ready, defensible strategy—anchored to the above literature—often looks like:

- **Primary model:** STN-based MILP (Maravelias & Grossmann) for the coupled batch + buffer structure. citeturn27view0  
- **Scalability enhancement:** decomposition or two-stage modeling (as in Klanke et al.) if the full MILP is too slow for shift-level instances. citeturn11view1  
- **Disruption handling:** reactive MILP rescheduling (Vin & Ierapetritou) for breakdown/rush orders, optionally paired with simulation-based evaluation (Rooeinfar et al.) to quantify performance under stochastic downtime. citeturn29view0turn28view0  
- **Heuristic alternative (if needed):** use the encoding/decoding structures from the limited-buffer hybrid/flexible flow shop papers (Zheng 2024; Zheng 2021; Lin 2020; Jiang & Zhang 2019) when exact optimization is computationally prohibitive but you still need high-quality schedules quickly. citeturn25view1turn28view2turn11view2turn19view0  

This combination gives you a clean thesis narrative: **(i) rigorous formulation for the coupled system**, **(ii) computational strategy for realistic horizons**, and **(iii) explicitly modeled response to downtime/disruptions**, all grounded in peer-reviewed scheduling literature. citeturn27view0turn11view1turn29view0turn28view0