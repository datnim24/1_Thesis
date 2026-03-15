# Project Navigator — Nestlé Trị An Roasting Scheduling Thesis

This README is a navigator for the current thesis project files. It tells you **what each file is for, when to read it, and how the files connect**, so you do not have to play archaeological expedition in the folder.

---

## 1. Project at a glance

**Thesis theme:** Dynamic batch roasting scheduling under shared pipeline constraints and unplanned disruptions.

**Core workflow:**

1. Define the real factory system and scope.
2. Formalize the deterministic optimization model.
3. Define the dynamic simulation and reactive re-scheduling loop.
4. Use the cost structure as the single source of truth for objective/reward values.
5. Support thesis writing with contextual/background notes and literature references.
6. Format the final thesis using IU-IEM guidelines.

---

## 2. Master file map

| File | Role | Use this when you need… | Main dependency / relationship |
|---|---|---|---|
| [`Thesis_Problem_Description_v2.md`](./Thesis_Problem_Description_v2.md) | **System and problem definition** | The physical system, roasters, pipeline logic, RC stock behavior, UPS concept, and the business meaning of the problem | Feeds the math model and simulation logic |
| [`cost.md`](./cost.md) | **Single source of truth for objective values** | Revenue, tardiness penalty, stockout penalty, safety-idle, overflow-idle, and profit objective | Referenced by problem description, math model, and future code |
| [`mathematical_model_complete.md`](./mathematical_model_complete.md) | **Formal optimization model** | Sets, parameters, variables, constraints, objective, and solver roles | Built from the problem description and cost structure |
| [`event_simulation_logic_complete.md`](./event_simulation_logic_complete.md) | **Dynamic simulation / reactive scheduling engine** | How time advances, how UPS events are processed, how state changes, and how Dispatching / CP-SAT / DRL plug into the simulator | Operational wrapper around the mathematical model |
| [`Surrounding_information_introduction_v2.md`](./Surrounding_information_introduction_v2.md) | **Thesis writing reference** | Chapter framing, company background, contribution statement, scope, limitations, methodology framing, experiment structure | Synthesizes the project into thesis-ready narrative |
| [`keyref_GPT.md`](./keyref_GPT.md) | **Curated literature mapping** | A more structured literature-reference set for the thesis, with relevance explanations | Supports Chapter 2 and method justification |
| [`keyref_Perplexity.md`](./keyref_Perplexity.md) | **Alternative literature scouting draft** | A second AI-generated paper list and comparison view that can still be mined for useful references or phrasing | Use as supporting material, but verify papers manually |
| [`IEM-IU Thesis Guidelines (Updated-032023) (3) (2) (1) - Copy.docx`](./IEM-IU%20Thesis%20Guidelines%20%28Updated-032023%29%20%283%29%20%282%29%20%281%29%20-%20Copy.docx) | **University formatting guide** | Structure, formatting, required thesis sections, and presentation rules for final submission | Applies when writing and formatting the actual thesis document |

---

## 3. Which files are “source of truth” for what?

### A. Physical system and business logic
- **Primary file:** [`Thesis_Problem_Description_v2.md`](./Thesis_Problem_Description_v2.md)
- Use this for:
  - roaster eligibility
  - shared pipeline logic
  - RC stock logic
  - MTO vs MTS interpretation
  - UPS behavior at system level

### B. Cost / reward / objective values
- **Primary file:** [`cost.md`](./cost.md)
- Use this for:
  - revenue per completed batch
  - stockout event penalty
  - tardiness penalty
  - safety-idle penalty
  - overflow-idle penalty
- If another file seems to disagree with this one, **`cost.md` should win** unless you intentionally revise the project standard.

### C. Optimization model
- **Primary file:** [`mathematical_model_complete.md`](./mathematical_model_complete.md)
- Use this for:
  - symbols and notation
  - decision variables
  - hard/soft constraints
  - deterministic vs reactive model interpretation
  - solver-ready formulation

### D. Event-driven execution logic
- **Primary file:** [`event_simulation_logic_complete.md`](./event_simulation_logic_complete.md)
- Use this for:
  - simulator state definitions
  - slot-by-slot update order
  - UPS interruption handling
  - CP-SAT re-solve behavior
  - strategy interface for Dispatching / CP-SAT / DRL

### E. Thesis narrative and chapter framing
- **Primary file:** [`Surrounding_information_introduction_v2.md`](./Surrounding_information_introduction_v2.md)
- Use this for:
  - Chapter 1 introduction framing
  - Chapter 2 company/context write-up
  - methodology contribution statement
  - scope and limitations
  - experiment framing

### F. Literature review support
- **Primary files:**
  - [`keyref_GPT.md`](./keyref_GPT.md)
  - [`keyref_Perplexity.md`](./keyref_Perplexity.md)
- Use these to:
  - identify core papers
  - justify CP-SAT / MILP / reactive scheduling / buffer-constrained scheduling
  - build Chapter 2 comparison tables
- These are **reference-support files**, not the governing definition of the thesis model.

### G. Final formatting and submission compliance
- **Primary file:** [`IEM-IU Thesis Guidelines (Updated-032023) (3) (2) (1) - Copy.docx`](./IEM-IU%20Thesis%20Guidelines%20%28Updated-032023%29%20%283%29%20%282%29%20%281%29%20-%20Copy.docx)
- Use this when producing the final thesis manuscript.

---

## 4. Recommended reading paths

## Path A — If you are about to start coding

Read in this order:

1. [`Thesis_Problem_Description_v2.md`](./Thesis_Problem_Description_v2.md)
2. [`cost.md`](./cost.md)
3. [`mathematical_model_complete.md`](./mathematical_model_complete.md)
4. [`event_simulation_logic_complete.md`](./event_simulation_logic_complete.md)

Why this order works:
- First understand the factory logic.
- Then lock down the cost semantics.
- Then implement the formal model.
- Then wrap it inside the simulation loop.

---

## Path B — If you are writing the thesis document

Read in this order:

1. [`Surrounding_information_introduction_v2.md`](./Surrounding_information_introduction_v2.md)
2. [`Thesis_Problem_Description_v2.md`](./Thesis_Problem_Description_v2.md)
3. [`keyref_GPT.md`](./keyref_GPT.md)
4. [`keyref_Perplexity.md`](./keyref_Perplexity.md)
5. [`IEM-IU Thesis Guidelines (Updated-032023) (3) (2) (1) - Copy.docx`](./IEM-IU%20Thesis%20Guidelines%20%28Updated-032023%29%20%283%29%20%282%29%20%281%29%20-%20Copy.docx)

Why this order works:
- Start with the narrative/frame.
- Then pull the exact system details.
- Then support Chapter 2 with references.
- Finally format the thesis to university requirements.

---

## Path C — If you are checking logic consistency before implementation

Read in this order:

1. [`Thesis_Problem_Description_v2.md`](./Thesis_Problem_Description_v2.md)
2. [`cost.md`](./cost.md)
3. [`mathematical_model_complete.md`](./mathematical_model_complete.md)
4. [`event_simulation_logic_complete.md`](./event_simulation_logic_complete.md)
5. [`Surrounding_information_introduction_v2.md`](./Surrounding_information_introduction_v2.md)

Focus questions:
- Does the math model reflect the physical system exactly?
- Does the simulator implement the same logic as the math model?
- Do all penalties and KPIs match `cost.md`?
- Do the thesis framing and limitations still match the actual model?

---

## 5. Dependency view

```text
Thesis_Problem_Description_v2.md
        │
        ├──> cost.md
        │
        ├──> mathematical_model_complete.md
        │         │
        │         └──> event_simulation_logic_complete.md
        │
        └──> Surrounding_information_introduction_v2.md
                  │
                  ├──> keyref_GPT.md
                  ├──> keyref_Perplexity.md
                  └──> IEM-IU Thesis Guidelines (...).docx
```

Interpretation:
- The **problem description** is the root of the technical story.
- The **cost file** standardizes the economics.
- The **math model** formalizes the scheduling problem.
- The **simulation file** executes that model dynamically under disruptions.
- The **surrounding info** file helps convert the project into thesis chapters.
- The **keyref** files support literature review.
- The **guidelines docx** governs final formatting, not technical logic.

---

## 6. Practical file usage notes

### `Thesis_Problem_Description_v2.md`
Best for human understanding. If someone new joins the project, start them here.

### `mathematical_model_complete.md`
Best for solver implementation and model auditing. This is the most important technical file once the problem is already understood.

### `event_simulation_logic_complete.md`
Best for coding the environment and reactive comparison experiments. If the solver is the brain, this file is the nervous system.

### `cost.md`
Do not casually override numbers from memory. Tiny inconsistency here turns into thesis goblin behavior later.

### `Surrounding_information_introduction_v2.md`
Best for writing chapters, justifying scope, and making the thesis defensible in front of a committee.

### `keyref_GPT.md` and `keyref_Perplexity.md`
Useful for literature discovery and comparison tables, but they are still secondary to manual paper verification.

### `IEM-IU Thesis Guidelines (...).docx`
This is your formatting referee. Boring, yes. Also the kind of boring that can cost points.

---

## 7. Fast-start checklist

### If you are coding next
- Read the problem description.
- Lock the objective values from `cost.md`.
- Implement the math model.
- Implement the simulator exactly in the phase order defined in the simulation file.

### If you are writing next
- Use the surrounding-info file as chapter scaffold.
- Use the problem description for system details.
- Use the key references to justify the methodology.
- Conform to the IU-IEM thesis format at the end, not as an afterthought.

---

## 8. Suggested “canonical core set”

If you only keep **four files open at all times**, keep these:

1. [`Thesis_Problem_Description_v2.md`](./Thesis_Problem_Description_v2.md)
2. [`cost.md`](./cost.md)
3. [`mathematical_model_complete.md`](./mathematical_model_complete.md)
4. [`event_simulation_logic_complete.md`](./event_simulation_logic_complete.md)

That quartet is the actual machine.
Everything else helps explain it, justify it, or format it.

