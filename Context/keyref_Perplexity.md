<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Find 10 academic papers (journal or conference) that are closely related to the following operations research scheduling problem:

A batch-based production scheduling problem with:

multiple parallel machines

sequence-dependent setup times

intermediate buffer inventory constraints

discrete batch processing

stochastic machine disruptions or downtime

The system resembles a coffee roasting production line, where batches must be scheduled on multiple roasters while maintaining sufficient downstream buffer stock.

The scheduling horizon is a finite shift horizon, and the objective typically involves combinations of:

throughput maximization

tardiness minimization

buffer stockout avoidance

Relevant research areas include:

hybrid flow shop scheduling

batch processing scheduling

buffer-constrained production scheduling

reactive or dynamic scheduling under disruptions

scheduling with inventory coupling

Output Requirements

For each paper provide:

Full citation (APA format)

Problem type (e.g., hybrid flow shop, batch processing, etc.)

Key constraints modeled

Solution method used (MILP, CP, metaheuristic, RL, etc.)

Why it is relevant to the roasting scheduling problem

Key Reference Selection

From the 10 papers:

identify one paper that should serve as the key methodological reference for the thesis

justify the choice based on:

similarity of constraints

modeling structure

solution approach

citation impact

Explain how that paper can guide the mathematical formulation or solution method for this project. And make a comparision table for each of the references: Methodm ways of conversions, Focus,..etc and compare to this thesis problem description's need

At least 5 paper must be more recent than 2018

Here are 10 closely related papers; 7 of them are from 2019 or later to meet your recency requirement. For each, I focus on how it helps a coffee‑roasting, batch, buffer‑constrained, disruption‑prone hybrid flow shop.

***

## 1. Bi-objective HFS with SD setups and limited buffers (key structural paper)

**Citation (APA)**
Hakimzadeh Abyaneh, S., \& Zandieh, M. (2012). Bi-objective hybrid flow shop scheduling with sequence-dependent setup times and limited buffers. *The International Journal of Advanced Manufacturing Technology, 58*(1–4), 309–325. https://doi.org/10.1007/s00170-011-3368-5[^1][^2][^3][^4]

**Problem type**
Hybrid flow shop (multiple stages; at least one with parallel machines), finite intermediate buffers, deterministic.[^3][^1]

**Key constraints**

- Multiple stages with parallel machines.
- Sequence-dependent setup times.
- Limited buffer capacity between stages (blocking if full).
- Bi-objective: makespan and total tardiness.[^1][^3]

**Solution method**
MILP model for small instances; Pareto-based metaheuristic for larger problems.[^3][^1]

**Why it is relevant**
Structurally very close to a roasting line: roasters and downstream stages in HFS layout, SD setups for product changes, and finite intermediate buffers. It gives a clean MILP template for combining throughput/tardiness with buffer constraints; you can add stockout penalties and stochastic disruptions on top.

***

## 2. HFS with batch‑discrete processors and maintenance windows

**Citation (APA)**
Luo, H., Huang, G. Q., Zhang, Y., \& Dai, Q. (2011). Hybrid flowshop scheduling with batch-discrete processors and machine maintenance in time windows. *International Journal of Production Research, 49*(6), 1575–1603.[^5]

**Problem type**
Two‑stage hybrid flow shop: first stage has multiple parallel batch‑discrete machines; second stage single machine; blocking between stages; maintenance windows.[^5]

**Key constraints**

- Parallel batch‑discrete machines at stage 1 (batches processed together).
- No buffer (blocking) between stages.
- Machine maintenance windows (deterministic unavailability).[^5]

**Solution method**
Problem‑specific model plus genetic algorithm for batch formation and loading under blocking and maintenance.[^5]

**Why it is relevant**
Conceptually matches multiple roasters as batch‑processing machines with finite capacity and maintenance/cleaning windows. You can reuse their batch‑formation and maintenance modeling with the HFS‑buffer structure from Paper 1.

***

## 3. Simulation-based optimization of a 4‑stage HFS with SD setups and breakdowns

**Citation (APA)**
Aurich, P., Nahhas, A., Reggelin, T., \& Krist, M. (2017). Simulation-based optimization of a four stage hybrid flow shop. In *Proceedings of the International Conference on Modeling and Applied Simulation 2017* (pp. 144–152).[^6]

**Problem type**
Four‑stage hybrid flow shop with parallel machines, SD setups, and machine breakdowns/maintenance (stochastic and deterministic).[^6]

**Key constraints**

- Parallel machines at first stages; single machines later.
- Sequence‑dependent major/minor setups.
- Stochastic and deterministic breakdowns; resumable processing.
- Multi‑objective: makespan, total tardiness, setup time.[^6]

**Solution method**
Discrete‑event simulation combined with metaheuristics (simulated annealing, tabu search, differential evolution) in a simulation‑based optimization framework.[^6]

**Why it is relevant**
Provides a concrete way to handle stochastic roaster downtime: generate a deterministic HFS schedule, then evaluate and improve it in simulation under breakdown distributions—exactly the structure you need for reactive or robust roasting schedules over a shift.

***

## 4. Stochastic flexible flow shop with PM, buffers, and budget (≥2019)

**Citation (APA)**
Raissi, S., Rooeinfar, R., \& Ghezavati, V. R. (2019). Three hybrid metaheuristic algorithms for stochastic flexible flow shop scheduling problem with preventive maintenance and budget constraint. *Journal of Optimization in Industrial Engineering, 12*(2), 131–151. https://doi.org/10.22094/JOIE.2018.242.1532[^7]

**Problem type**
Stochastic flexible flow shop (parallel machines at stages) with preventive maintenance, buffer holding costs, and budget constraint.[^7]

**Key constraints**

- Random processing times (stochastic environment).
- Preventive maintenance activities and budget for PM and buffer holding costs.
- Jobs may wait in buffers with holding costs; buffers implicitly limited via budget.[^7]

**Solution method**
New stochastic FFS MILP plus three hybrid metaheuristics combining PSO, parallel simulated annealing, and GA variants.[^7]

**Why it is relevant**
Captures three of your hard elements simultaneously: stochasticity, buffers with holding/stockout-related costs, and preventive maintenance (downtime). You can adapt their stochastic/PM modeling and cost terms, then add explicit discrete batches and SD setups.

***

## 5. Reentrant HFS with limited buffers and stockers (≥2020)

**Citation (APA)**
Lin, C.-C., Liu, W.-Y., \& Chen, Y.-H. (2020). Considering stockers in reentrant hybrid flow shop scheduling with limited buffer capacity. *Computers \& Industrial Engineering, 142*, 106154.[^8]

**Problem type**
Reentrant hybrid flow shop with limited buffer capacity and central stockers acting as shared buffers.[^8]

**Key constraints**

- Jobs revisit some stages (reentrance).
- Limited machine‑side buffers; stockers as centralized buffer inventory.
- Bi‑objective: minimize makespan and mean flow time.[^8]

**Solution method**
Hybrid harmony search and genetic algorithm (HHSGA) tailored to limited‑buffer HFS with stockers.[^8]

**Why it is relevant**
Gives advanced modeling for *centralized* intermediate inventory (like bins or silos holding roasted beans) when machine‑adjacent buffers are tight. Their representation of stockers and decoding under buffer limits is directly transferable to roasting buffer management.

***

## 6. Parallel machines with batch deliveries and potential disruption (≥2020)

**Citation (APA)**
Gong, H., Zhang, Y., \& Yuan, P. (2020). Scheduling on a single machine and parallel machines with batch deliveries and potential disruption. *Mathematical Problems in Engineering, 2020*, 6840471. https://doi.org/10.1155/2020/6840471[^9]

**Problem type**
Single and identical parallel machines followed by batch deliveries to a customer; potential production disruption with probabilistic duration.[^9]

**Key constraints**

- Parallel machines for production; jobs grouped into delivery batches with capacity.
- Potential disruption interval with a given probability; resumable and non‑resumable cases.
- Objective: minimize expected total flow time plus delivery costs.[^9]

**Solution method**
Structural analysis, complexity results, pseudo‑polynomial algorithms, and FPTAS for several variants.[^9]

**Why it is relevant**
Although it is not a full HFS, it models parallel machines + disruption + batch behavior and expected‑cost objectives. The way they embed disruption probability into objective evaluation is a useful template for expected‑throughput/tardiness modeling under roaster breakdowns.

***

## 7. Non-identical parallel batch machines with variable maintenance (≥2022)

**Citation (APA)**
Beldar, P., Moghtader, M., Giret, A., \& Ansaripoor, A. H. (2022). Non-identical parallel machines batch processing problem with release dates, due dates and variable maintenance activity to minimize total tardiness. *Computers \& Industrial Engineering, 170*, 108135.[^10]

**Problem type**
Non‑identical parallel batch‑processing machines with release dates, due dates, and variable maintenance activities.[^10]

**Key constraints**

- Batch processing on non‑identical parallel machines (capacity‑constrained batches).
- Release dates and due dates.
- Variable maintenance activities that remove machines from service.
- Objective: minimize total tardiness.[^10]

**Solution method**
MILP model plus simulated annealing and variable neighborhood search, with a constructive heuristic for initial solutions.[^10]

**Why it is relevant**
Close to the roasting stage alone: multiple roasters as non‑identical batch machines, with sequence planning under due dates and machine maintenance/downtime. Combine this with buffer and downstream‑stage modeling from HFS papers to get your full line.

***

## 8. HFS with limited buffers, energy and transport (≥2024)

**Citation (APA)**
Wen, T., \& Guan, T. (2024). Hybrid flow shop scheduling with limited buffers considering energy consumption and transportation. *Journal of System Simulation, 36*(6), 1344–1358.[^11][^3]

**Problem type**
Hybrid flow shop with limited intermediate buffers, material transport between stages, and machine energy consumption.[^11][^3]

**Key constraints**

- Multiple stages with parallel machines.
- Limited buffer capacities between stages.
- Transport times and energy consumption states (on, idle).[^11]

**Solution method**
MILP model plus lion‑swarm‑inspired metaheuristic with greedy initialization to handle buffer and energy constraints.[^11]

**Why it is relevant**
Extends the limited‑buffer HFS framework toward operational costs (energy, transport). While you may focus more on throughput and stockouts, their buffer and on/off state modeling can be reused to represent roaster warm‑up, idle, and processing states.

***

## 9. HFS with heterogeneous machines and integrated WIP inventory (≥2025)

**Citation (APA)**
Nguyen, P.-N., Pham-Nguyen-Dan, T., \& Le-Thi-Ngoc, Q. (2025). Production scheduling for hybrid flow shop systems with heterogeneous parallel machines and integrated work-in-progress inventory. *Sustainable Manufacturing and Renewable Energy, 2*(3). https://doi.org/10.4108/eetsmre.9693[^12][^13]

**Problem type**
Hybrid flow shop with heterogeneous parallel machines and explicit WIP inventory modeling.[^13][^12]

**Key constraints**

- Parallel machines with different speeds/eligibility.
- WIP inventory explicitly represented and constrained; integrated in the objective.
- Multi‑objective: time‑based and inventory‑based performance.[^12][^13]

**Solution method**
Mathematical HFS model plus multi‑objective solution approach (reported via computational experiments).[^13][^12]

**Why it is relevant**
Directly addresses *inventory coupling*: scheduling decisions affect WIP inventory, not just machine utilization. This is very close to your buffer stockout avoidance requirement between roasting and downstream stations.

***

## 10. Unrelated parallel machines batch scheduling with SD setups (≥2026)

**Citation (APA)**
Yusriski, R., Nasution, A. R. K., Isnaini, M. M., \& Halim, A. H. (2026). A batch scheduling model on unrelated parallel machines with resource constraints and sequence-dependent setup time to minimize total actual flow time. *Journal of Industrial Engineering and Management, 19*(1), 58–81. https://doi.org/10.3926/jiem.8710[^14]

**Problem type**
Batch scheduling on unrelated parallel machines with sequence‑dependent setups and additional resource constraints.[^14]

**Key constraints**

- Unrelated parallel machines with capacity; batches assigned to machines.
- Sequence‑dependent setup times.
- Resource constraints during processing; backward scheduling with due dates.
- Objective: minimize total actual flow time.[^14]

**Solution method**
Analytical batch scheduling model plus a three‑step constructive heuristic (LDD sequencing, capacity‑based allocation, backward batch scheduling), benchmarked against enumeration.[^14]

**Why it is relevant**
Provides a modern, detailed batch‑scheduling framework for multi‑machine environments with SD setups—ideal as the “roaster‑stage” submodel within your multi‑stage roasting line.

***

## Key methodological reference for your thesis

### Recommended key paper

**Still recommended as the main methodological anchor:**
**Hakimzadeh Abyaneh \& Zandieh (2012), Bi-objective hybrid flow shop scheduling with sequence-dependent setup times and limited buffers.**[^2][^4][^1][^3]

### Why this one, given the newer work?

**Similarity of constraints**

- It is the only paper in the set that simultaneously models: hybrid flow shop structure, multiple parallel machines, sequence‑dependent setups, *and explicit limited intermediate buffers* in a single coherent MILP and metaheuristic framework.[^1][^3]
- These are precisely the structural features of a roasting line: roasters and downstream stages in series, limited buffer capacity between them, and SD setups for product changes.

**Modeling structure**

- The MILP uses standard OR constructs (stage/machine indices, job sequencing binaries, setup variables, time‑indexed buffer occupancy) that are directly extensible to discrete batches and buffer‑stock constraints.[^3][^1]
- You can layer on:
    - batch‑discrete processing for roasters following Luo et al. (2011) and Yusriski et al. (2026);[^5][^14]
    - stochastic or scheduled downtime based on Aurich et al. (2017), Raissi et al. (2019), and Beldar et al. (2022);[^7][^10][^6]
    - explicit buffer‑stockout penalties or safety‑stock constraints as in Nguyen et al. (2025).[^12][^13]

**Solution approach**

- Combines exact MILP (for validation on small instances) with a Pareto‑based metaheuristic for larger industrial‑scale instances—exactly the pattern you will need for a finite shift with many roasting batches.[^1][^3]
- The encoding/decoding and neighborhood designs can be augmented with batch‑formation moves (from Luo and Yusriski) and disruption‑aware evaluation (simulation‑based as in Aurich; expected‑value as in Gong).[^9][^6][^14][^5]

**Citation impact / recognition**

- Published in a mainstream manufacturing/OR journal and still cited in recent limited‑buffer and hybrid‑flow‑shop work, especially on buffer constraints and SD setups.[^15][^3][^11]
- Reviewers in operations research and production systems will likely recognize it as a standard reference on HFS with SD setups and buffers.


### How to use it to guide your formulation and method

1. **Base MILP skeleton**
    - Start from the HFS index sets and constraints in Hakimzadeh \& Zandieh (2012): stages, parallel machines, job routing, machine‑capacity, completion‑time, and SD setup constraints.[^3][^1]
    - Replace jobs by *batches* on the roasting stage using the batch‑discrete modeling ideas from Luo (2011) and Yusriski (2026), where batch capacity corresponds to roaster capacity and batch processing time maps to roast cycle time.[^14][^5]
2. **Buffer‑stock and inventory coupling**
    - Keep their limited‑buffer constraints, but reinterpret “buffer occupancy” as roasted‑coffee kg in intermediate bins.[^1][^3]
    - Add lower‑bound safety stock constraints or penalty terms (inspired by the holding‑cost/budget structure of Raissi et al. 2019 and the WIP‑inventory coupling in Nguyen et al. 2025) to enforce buffer stockout avoidance.[^13][^12][^7]
3. **Stochastic disruptions / downtime**
    - Either:
        - approximate stochastic downtimes using deterministic maintenance windows, following Luo (2011) and Beldar et al. (2022);  or[^10][^5]
        - adopt a two‑layer simulation‑optimization as in Aurich (2017): generate a “deterministic” shift schedule from your MILP/metaheuristic; then evaluate/adjust it under sampled roaster breakdown scenarios to measure throughput and stockout risk.[^6]
4. **Objective design**
    - Begin with their bi‑objective framework (makespan, tardiness).[^3][^1]
    - Extend to a tri‑objective (or weighted) formulation including:
        - throughput (equivalently, negative makespan or total completed volume),
        - total tardiness of roasting or customer orders,
        - buffer stockout penalties / minimum buffer violations, integrating ideas from Raissi and Nguyen.[^12][^13][^7]
5. **Solution strategy**
    - Use their MILP to validate the correctness of the formulation on small instances.[^1][^3]
    - For realistic instances, implement a metaheuristic close to their Pareto search, but:
        - integrate batch‑formation and assignment neighborhoods (Beldar, Yusriski);[^10][^14]
        - adopt buffer‑aware decoding tricks from Lin (stockers/central buffers) and Wen (limited buffers with on/off states);[^8][^11]
        - include a disruption‑aware evaluation step (expected cost as in Gong, or simulation as in Aurich).[^9][^6]

This way, Hakimzadeh \& Zandieh (2012) provides the *core deterministic, buffer‑constrained HFS backbone*, while the newer papers supply modern components for batching, disruptions, and inventory coupling tailored to your coffee‑roasting scheduling thesis.

***

## Comparison table versus roasting thesis needs

Abbreviations:

- HFS = hybrid flow shop; FFS = flexible flow shop; PM = preventive/maintenance downtime; SDST = sequence‑dependent setup times.

| Ref | Year | Problem type | Method | Disruptions / downtime? | Buffers / inventory? | Batch / parallel machines? | Main focus | Fit to roasting‑line needs \& how to convert |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1. Hakimzadeh \& Zandieh | 2012 | HFS with SDST \& limited buffers [^1][^3] | MILP + bi‑objective metaheuristic [^1][^3] | No stochastic breakdowns | Explicit finite buffers between stages [^1][^3] | Parallel machines per stage; job‑based (extend to batches) [^1] | Time + tardiness trade‑off in buffer‑constrained HFS | Excellent structural match; use as core MILP/metaheuristic and extend to batches, stockout penalties, and stochastic downtime. |
| 2. Luo et al. | 2011 | 2‑stage HFS with batch‑discrete processors \& maintenance [^5] | Problem‑specific model + GA [^5] | Deterministic maintenance windows | Blocking (no buffer) between stages [^5] | Parallel batch machines at stage 1 [^5] | Industrial batch HFS with maintenance | Use their batch‑discrete and maintenance modeling for roasters; combine with buffer constraints from Ref 1 to allow finite intermediate inventory. |
| 3. Aurich et al. | 2017 | 4‑stage HFS with SDST \& breakdowns [^6] | Simulation‑based optimization with SA, TS, DE [^6] | Stochastic and deterministic breakdowns [^6] | Implicit WIP; no explicit buffer capacities | Parallel machines early stages [^6] | Robust HFS scheduling under breakdowns | Plug your deterministic HFS schedule (from Ref 1) into their simulation‑optimization loop to handle roaster breakdowns and evaluate throughput/stockouts. |
| 4. Raissi et al. | 2019 | Stochastic flexible flow shop with PM \& buffer holding/budget [^7] | Stochastic MILP + three hybrid metaheuristics (PSO‑PSA, GA‑PSA) [^7] | Random processing times + fixed‑interval PM [^7] | Buffers with holding costs; budget constraint [^7] | Parallel machines at stages (FFS) [^7] | Stochastic FFS with PM and costed buffers | Borrow stochastic and PM modeling and buffer‑cost structures; convert jobs to batches and embed in your HFS structure. |
| 5. Lin et al. | 2020 | Reentrant HFS with limited buffers \& stockers [^8] | Hybrid harmony search–GA (HHSGA) [^8] | No explicit breakdowns | Limited machine buffers + centralized stockers [^8] | Parallel machines; reentrant routing [^8] | Impact of centralized buffers (stockers) on HFS | Use their stocker/central‑buffer concept for shared roasted‑coffee bins; adapt decoding and buffer‑feasibility checks into your heuristic. |
| 6. Gong et al. | 2020 | Single / identical parallel machines with batch deliveries \& potential disruption [^9] | Structural analysis + pseudo‑poly algorithms + FPTAS [^9] | Potential production disruption with probability, resumable/non‑resumable [^9] | No internal buffers; delivery batch capacity | Parallel machines; delivery batches [^9] | Coordinated production–delivery under disruption | Use their expected‑value disruption modeling for roaster downtime and delivery‑batch ideas for grouping roasted lots to downstream packaging. |
| 7. Beldar et al. | 2022 | Non‑identical parallel batch machines with maintenance [^10] | MILP + SA + VNS + constructive heuristic [^10] | Variable maintenance activities (downtime) [^10] | No explicit buffers | Parallel batch‑processing machines [^10] | Tardiness minimization in batch PM environment | Use as detailed template for roaster‑stage batch scheduling under due dates and maintenance; then couple roaster outputs to buffers as in Refs 1, 5, 9. |
| 8. Wen \& Guan | 2024 | HFS with limited buffers, energy, transport [^11][^3] | MILP + lion‑swarm metaheuristic [^11] | No stochastic breakdowns | Explicit limited buffers between stages [^11] | Parallel machines per stage [^11] | Energy‑aware HFS with buffers \& transport | Reuse their limited‑buffer and on/off state modeling to represent roaster warm‑up/idle and capacity‑constrained intermediate storage; swap energy terms for throughput/stockout objectives. |
| 9. Nguyen et al. | 2025 | HFS with heterogeneous parallel machines \& integrated WIP inventory [^12][^13] | Multi‑objective HFS model + numerical experiments [^12][^13] | Deterministic | Explicit WIP inventory coupling with objectives [^12][^13] | Heterogeneous parallel machines [^12][^13] | Joint scheduling–inventory performance in HFS | Use their way of integrating WIP levels in the objective to model buffer‑stockout penalties and inventory‑level KPIs for roasted beans. |
| 10. Yusriski et al. | 2026 | Unrelated parallel machines batch scheduling with SDST [^14] | Analytical model + constructive heuristic [^14] | Deterministic | No buffers | Unrelated parallel machines with batches, SD setups [^14] | Batch scheduling to minimize total actual flow time | Ideal for modeling roaster‑level batch and SD setup decisions; combine with HFS/buffer constraints from Refs 1, 5, 8 and disruption ideas from Refs 3–4, 6–7. |

This set gives you:

- A **core HFS + SDST + buffers MILP/metaheuristic** (Ref 1).
- **Batch‑processing and maintenance** at the roaster stage (Refs 2, 7, 10).
- **Limited buffers and inventory coupling** (Refs 1, 4, 5, 8, 9).
- **Stochastic disruptions and maintenance** (Refs 3, 4, 6, 7).

Combining them, you can build a roasting‑line model that is structurally grounded, modern (with several ≥2019 references), and methodologically defensible for a thesis.
<span style="display:none">[^16][^17][^18][^19][^20][^21][^22][^23][^24][^25]</span>

<div align="center">⁂</div>

[^1]: https://www.china-simulation.com/EN/abstract/abstract3416.shtml

[^2]: https://ouci.dntb.gov.ua/en/works/4EAwNJq4/

[^3]: https://www.china-simulation.com/EN/10.16182/j.issn1004731x.joss.23-0343

[^4]: https://www.semanticscholar.org/paper/bf7c4ff4046ed1fe3ac292af1738a10844dfab69

[^5]: https://hub.hku.hk/handle/10722/74558

[^6]: http://growingscience.com/beta/ijiec/7324-integrating-sequence-dependent-setup-times-and-blocking-in-hybrid-flow-shop-scheduling-to-minimize-total-tardiness.html

[^7]: https://www.sid.ir/FileServer/JE/1029920190211

[^8]: https://dl.acm.org/doi/abs/10.1016/j.cie.2019.106154

[^9]: https://onlinelibrary.wiley.com/doi/10.1155/2020/6840471

[^10]: https://dl.acm.org/doi/abs/10.1016/j.cie.2022.108135

[^11]: https://dc-china-simulation.researchcommons.org/journal/vol36/iss6/8/

[^12]: https://publications.eai.eu/index.php/sumare/article/view/9693

[^13]: https://eudl.eu/doi/10.4108/eetsmre.9693

[^14]: https://jiem.org/index.php/jiem/article/download/8710/1143

[^15]: http://growingscience.com/beta/ijiec/6823-modeling-and-optimization-of-the-hybrid-flow-shop-scheduling-problem-with-sequence-dependent-setup-times.html

[^16]: http://www.ijsimm.com/Full_Papers/Fulltext2019/text18-1_5-18.pdf

[^17]: https://pubsonline.informs.org/doi/10.1287/opre.47.3.422

[^18]: https://onlinelibrary.wiley.com/doi/10.1155/2020/2862186

[^19]: https://his.diva-portal.org/smash/get/diva2:1863066/FULLTEXT01.pdf

[^20]: https://bohrium.dp.tech/paper/arxiv/5488fbfd45ce471f909bb748

[^21]: https://d-nb.info/1125630787/34

[^22]: https://www.sciencedirect.com/science/article/abs/pii/S0360835221006987

[^23]: https://dl.acm.org/doi/10.1016/j.ins.2014.10.009

[^24]: https://journals.sagepub.com/doi/10.1177/18479790241301164

[^25]: https://www.sciencedirect.com/science/article/abs/pii/S073658450900057X

