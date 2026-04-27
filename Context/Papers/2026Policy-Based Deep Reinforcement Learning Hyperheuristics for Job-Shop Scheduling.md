## Policy-Based Deep Reinforcement Learning Hyperheuristics for Job-Shop Scheduling Problems

Sofiene Lassoued            - [a], Asrat Gobachew b, Stefan Lier b, Andreas Schwung a


_aSouth Westphalia University of Applied Sciences, Automation Technology and Learning Systems Lübecker Ring 2, Soest, 59494, North_
_Rhine-Westphalia, Germany_
_bSouth Westphalia University of Applied Sciences, Logistik und Supply Chain Management Lindenstr.53, Meschede, 59872, North Rhine-Westphalia, Germany_


**Abstract**


This paper proposes a policy-based deep reinforcement learning hyper-heuristic framework for solving the Job Shop Scheduling
Problem. The hyper-heuristic agent learns to switch scheduling rules based on the system state dynamically. We extend the hyperheuristic framework with two key mechanisms. First, action prefiltering restricts decision-making to feasible low-level actions,
enabling low-level heuristics to be evaluated independently of environmental constraints and providing an unbiased assessment.
Second, a commitment mechanism regulates the frequency of heuristic switching. We investigate the impact of different commitment strategies, from step-wise switching to full-episode commitment, on both training behavior and makespan. Additionally, we
compare two action selection strategies at the policy level: deterministic greedy selection and stochastic sampling. Computational
experiments on standard JSSP benchmarks demonstrate that the proposed approach outperforms traditional heuristics, metaheuristics, and recent neural network-based scheduling methods.


_Keywords:_ Hyper-heuristics, Job Shop Scheduling, Policy-based Reinforcement learning, Petri nets,



**1.** **Introduction**


The Shop Scheduling Problem (JSSP) is a fundamental and
widely studied combinatorial optimization problem with significant practical relevance in many domains [1]. Despite its
widespread use, it remains computationally challenging due to
its NP-complete nature [2]. Consequently, no algorithm can
solve all JSSP instances optimally in polynomial time, despite
numerous solution approaches ranging from exact to heuristics,
metaheuristics, and learning-based approaches [3].
Exact solutions, as Branch and Bound [4] Mixed-Integer
Linear Programming[5], Constraint Programming [6], and dynamic programming [7], suffer from the dimensionality curse
with impractical solving time even for moderately sized problems. This led to a shift to near-optimal yet fast heuristic approaches. However, handcrafted heuristics are typically
domain-specific and require substantial problem expertise [8].
For instance, a heuristic might excel in one scenario but fail to
generalize to others.
Given these limitations, scientists experimented with natureinspired metaheuristics to solve JSSPs, like Tabu search [9, 10],
genetic algorithms [11], Simulated Annealing [12], Ant Colony
Optimization [13] and many other approaches. Still, metaheuristics require substantial algorithmic expertise, with parameter tuning often unrelated to the problem itself [14], creating
a domain barrier between optimization expertise and problemdomain expertise.


∗Corresponding author: Sofiene Lassoued, email: lassoued.sofiene@fhswf.de



To address this domain barrier, the hyper-heuristic (HH)
framework has been proposed. Widely used in scheduling
problems [15], high-level HH operate on low-level heuristics
(LLHs) that construct solutions rather than search the solution
space directly, thereby enabling improved generalization [16].
The high-level HH can take many forms, ranging from metaheuristics, primarily genetic programming (GP-HH), to more
recently introduced deep reinforcement learning approaches
(DRL-HH). Starting with the more widely used GP-HH, the authors of [14] examined the advantages and limitations of using
GP as an HH. As an application, [17] employed GP-based HH
to solve the Uncertain Capacitated Arc Routing Problem, while

[18, 19, 20] applied genetic HH to tackle Dynamic Flexible JobShop Scheduling problems.

Despite their advantages, GP-HH has several inherent limitations. First, each execution of GP can yield a different
"best-of-run" heuristic, raising concerns about the approach’s
repeatability and predictability. Second, GP often requires unintuitive manual parameter tuning through trial-and-error to
achieve competitive results [14]. In contrast, the knowledgeretention capability of Reinforcement Learning, implemented
with a fixed seed, enables reproducible and consistent results.
Moreover, unlike GP, RL hyperparameters are more theoretically grounded and easier to interpret.

Deep reinforcement learning (DRL) has been successfully
applied as a standalone optimization approach for the JSSP,
achieving competitive performance with architectures ranging
from graph neural networks [21, 22], to attention- and transformer based models [23, 24]. However, despite their effectiveness, deep neural network–based methods alone generally


suffer from limited explainability: unlike heuristic rules that
are readily interpretable by domain experts, learned policy networks typically function as black boxes. This limitation motivates the integration of DRL with hyper-heuristics. On the
one hand, HHs preserve domain interpretability by operating
on low-level heuristics. On the other hand, DRL overcomes
the limitations of traditional GP-HH by learning effective highlevel heuristic-selection policies.
This paper builds on the potential of DRL-HH by addressing inherent challenges arising from combining the two hyperheuristics and Deep Reinforcement learning frameworks, such
as unbiased performance evaluation and credit assignment. The
main contributions of this paper are summarized as follows:


1. We propose a novel policy-based deep reinforcement
Learning Hyper-heuristics framework. The JSSP RL environment is modeled as a timed colored Petri net that explicitly captures scheduling dynamics and manufacturing
constraints.
2. We leverage the Petri net guard to ensure that only feasible
low-level heuristics are considered in each state, allowing
LLH performance to be evaluated independently of environmental constraints.
3. We introduce _commitment_ . This temporal abstraction
mechanism improves credit assignment in long-horizon
scheduling by aggregating feedback over multiple consecutive timesteps, thereby enabling more stable learning.
4. Our framework provides multi-level interpretability: action selections are inherently explainable by the selected
dispatching rule, and real-time Petri net visualization further enables verification of the agent’s decision-making
process.
5. The approach achieves state-aware adaptive dispatching that learns optimal heuristic selection across different scheduling contexts. Experimental results show our
method outperforms static dispatching rules, metaheuristic algorithms, and end-to-end neural network approaches
in a head-to-head comparison on the Taillard Benchmark.


This paper is structured as follows. Section 2 reviews relevant literature and identifies the research gap addressed by this
work. Section 3 establishes the theoretical foundations, presenting the JSSP formulation, the timed colored Petri net representation, and the reinforcement learning framework based on
the PPO algorithm. Section 4 introduces our Hyper-heuristic
methodology, detailing the JSSP environment design, the proposed approach, and its key advantages. Section 5 analyzes
and discusses the results, examining the contribution of individual components of the proposed framework. Finally, Section 6 summarizes the main findings and discusses directions
for future research.


**2.** **Related Work**


Hyper-heuristics (HH) aim at interchanging different solvers
while solving a problem. The idea is to determine the best approach for solving a problem at its current state [25]. The HH



framework separates the optimization process into two distinct
domains: a control domain and a problem domain, separated by
a domain barrier [26, 27]. This architectural separation offers
several advantages. First, the problem domain is governed by
low-level heuristics that are interpretable by domain experts,
thereby preserving explainability. Second, this separation enables a modular design, allowing the high-level control strategy
to be interchanged, for example, evolving from evolutionary
methods to genetic algorithms or deep reinforcement learning,
without modifying the problem-domain heuristic [28].
To examine the different subcategories of HH, we adopt the
classification proposed in [29], which organizes methods along
two dimensions: the nature of the heuristic search space and the
type of feedback mechanism.
With respect to the search space, hyper-heuristics can be categorized into selection and generation approaches. Selection
HH choose from a predefined set of low-level heuristics (LLHs)

[30]. Generation HHs, by contrast, construct new heuristics
by combining components of existing LLHs. In both categories, solutions can be built incrementally using constructive
strategies or refined from a complete solution using perturbative
strategies.
The second dimension concerns the feedback mechanism
guiding the search. Three main categories can be distinguished:
online learning, offline learning, and non-learning approaches

[26]. In this paper, we focus on selection-based, learning-based
hyper-heuristics, in which the optimizer is a deep reinforcement learning (DRL) agent. Inherited from the DRL framework, DRL-HH can be divided into value-based (VRL-HH) and
policy-based (PRL-HH) approaches [31].
Value-based methods span from traditional reinforcement
learning–based hyper-heuristics (TRL-HH) to deep reinforcement learning (DRL) approaches. TRL-HH relies on techniques such as Q-table updates [32], bandit-based updates [33],
and value estimation schemes [34] to evaluate and select the
most suitable low-level heuristic (LLH) for a given state. In
contrast, DRL-based methods integrate reinforcement learning
with deep neural networks to handle larger and more complex state spaces. Examples include Deep Q-Networks (DQN)
applied to routing problems [35], Double Deep Q-Networks
(DDQN) for 2D packing problems [36], and Dueling Double
Deep Q-Networks (D3QN) for online packing problems [37].
Owing to their simplicity, traditional reinforcement learning approaches dominate the literature, followed by Deep Qlearning. In contrast, policy-based approaches remain relatively
scarce, despite their potential advantages.
On the policy-based hyper-heuristics side, the authors [38]
trained a Proximal Policy Optimization (PPO) hyper-heuristic
agent to select generalized constructive low-level heuristics
for combinatorial optimization problems, reporting improvements of up to 98% over benchmark results. However, their
constraint-handling strategy relies on post hoc penalization, in
which solutions are first generated and then penalized for violating constraints, resulting in unnecessary computational effort
spent evaluating invalid actions.
The authors of [39] also applied PPO-based hyper-heuristics
to solve various combinatorial problems and compared their



2


performance against Adaptive Large Neighborhood Search
(ALNS). The authors introduced a "wasted action" mechanism
to prevent the repeated selection of deterministic heuristics that
do not change the system state, thereby avoiding an infinite
loop. However, their choice of "insert" and "remove" operators as low-level heuristics limits the expressiveness of decisions relative to dispatching rules such as MTWR (Most total
work remaining) or FIFO (First in, first out), thereby weakening one of hyper-heuristics’s key advantages: interpretability
for domain experts.
Applied to the trading domain, the authors [40] proposed a
DRL-HH framework for multi-period portfolio optimization,
reporting notable performance gains over state-of-the-art trading strategies and traditional DRL baselines. Although their approach employs rich state representations, the state-transition
dynamics remain largely opaque; this is standard and acceptable in trading, where the environment is inherently stochastic and partially observable, and performance metrics such as
return and risk are prioritized over interpretability. However,
such opacity is less suitable for Job Shop Scheduling, where
explainable system state evolution is of primary importance.
To address the explainable system state evolution, we combined in our previous work [41, 42], the modeling capabilities
of timed colored Petri nets with the dynamic decision-making
and knowledge retention of DRL. We also leveraged the Petri
net’s guard functions to mask invalid actions. Despite the improved explainability provided by the Petri net’s graphical interface, the policy network itself remained a black box, leaving
the decision-making process largely opaque. This motivates integrating our previous approach with a hyper-heuristic framework, which offers complementary benefits: hyper-heuristics
LLH, enhance explainability. At the same time, dynamic action
masking in Petri nets can improve the evaluation of low-level
heuristics independently of the environment’s constraints.


Research on DRL-based hyper-heuristics (DRL-HH) for the
Job Shop Scheduling Problem faces several gaps and open challenges yet to be addressed.
First, most existing HH approaches rely on tabular or scorebased selection mechanisms, with limited use of actor–critic
DRL models that can leverage rich state representations. Second, in many HH frameworks (e.g., VRL-HH), LLHs are penalized when proposed moves are rejected by move-acceptance
strategies, even when rejections are caused by environmental
constraints rather than poor heuristic choices, leading to biased
performance signals; prefiltering infeasible actions is therefore
necessary for fair LLH evaluation. Third, the literature lacks a
systematic analysis of heuristic switching frequency, despite its
strong impact on performance stability and credit assignment.
Fourth, although HH frameworks offer inherent interpretability through domain-understandable LLHs, existing work does
not explicitly capitalize on or enhance this explainability. Finally, direct and controlled comparisons between metaheuristics, DRL methods, and DRL-based HH on common JSSP
benchmarks remain scarce.
To address these gaps, this paper employs a selection-based
hyper-heuristic framework with a constructive methodology,



using a policy-based deep reinforcement learning agent as an
optimization algorithm. Data are collected through interactions with an environment that models the Job Shop Scheduling
Problem using a colored timed Petri net. Additionally, the Petri
net guard functions provide a natural pre-filter for move acceptance, thereby avoiding the evaluation of invalid actions.


**3.** **Preliminaries**


In this section, we present the formulation and theoretical
foundations of our approach. We begin by defining the Job
Shop Scheduling Problem, including its objectives and constraints. We then introduce the mathematical formulation of
the Petri net model and the reinforcement learning framework,
followed by the theoretical background of Proximal Policy Optimization (PPO), which is used as the RL optimization algorithm.


_3.1._ _Job shop scheduling problem definition_


The Job Shop Scheduling Problem (JSSP) concerns the allocation of a set of _n_ jobs J = {1, 2, . . ., _n_ } to a set of _m_ machines
M = {1, 2, . . ., _m_ }. Each job _j_ ∈J consists of an ordered sequence of operations _O j_ 1, _O_ _j_ 2, . . ., _O j_ ℓ _j_, where ℓ _j_ denotes the
number of operations in job _j_ . Each operation _O_ _jk_ must be processed on a specific machine _M_ _jk_ ∈M for a given processing
time _p jk_ - 0. The goal is to determine feasible start times _S_ _jk_
for each operation so as to minimize the makespan _C_ max, defined as the maximum completion time among all operations.


(1) _S_ _j_, _k_ +1 ≥ _S_ _jk_ + _p jk_, ∀ _j_ ∈J, _k_ < ℓ _j_ .

(2) _S_ _jk_ ≥ _S_ _j_ [′] _k_ [′] + _p j_ [′] _k_ [′] or _S_ _j_ [′] _k_ [′] ≥ _S_ _jk_ + _p jk_,

∀( _j_, _k_ ) � ( _j_ [′], _k_ [′] ), _M jk_ = _M_ _j_ [′] _k_ [′] .

(3) _C_ max ≥ _S_ _jk_ + _p jk_, ∀ _j_ ∈J, _k_ ≤ ℓ _j_ .


Here, Equation (1) enforces the processing order of operations within each job, Equation (2) ensures that no two operations overlap on the same machine, and Equation (3) defines
the makespan as the maximum completion time across all operations.


_3.2._ _Colored timed Petrinets_


Petri nets provide a formal graphical framework for modeling
discrete-event systems characterized by concurrency, synchronization, and resource sharing. A Petri net is defined by the
pair (G, µ0), where G is a bipartite graph of places P and transitions T, and µ0 denotes the initial marking. Tokens represent
resources or job states and move through the net when transitions fire. For any node _n_ ∈P ∪T, let π( _n_ ) and σ( _n_ ) denote
its input and output sets. A transition _t_ is enabled if every input
place _p_ ∈ π( _t_ ) contains at least one token, and its firing updates
the marking according to:



µ˜( _p_ ) =



µ( _p_ ) − 1 _p_ ∈ π( _t_ ),
µ( _p_ ) + 1 _p_ ∈ σ( _t_ ), (1)

µ( _p_ ) otherwise.



3


Colored Petri nets (CPNs) extend this structure by associating data values (colors) with tokens, enabling compact representations of systems with repeated but heterogeneous job
structures. A CPN is defined as


CPN = (P, T, A, Σ, _C_, _N_, _E_, _G_, _I_ ), (2)


where Σ is the color set, _C_ assigns colors to nodes, _N_ specifies arc directions, _E_ defines arc expressions, _G_ encodes guard
conditions, and _I_ provides the initial marking.
Colored timed Petri nets (CTPNs) further extend CPNs by
associating a transition with firing delays, such that a transition
cannot fire until a token has spent the required sojourn time in
its upstream place. This allows for modeling the processing
times of a given operation.


_3.3._ _Reinforcement learning framework_


Reinforcement Learning (RL) provides a framework for
training agents to make sequential decisions through environmental interaction. An RL problem is formalized as a Markov
Decision Process (MDP) defined by the tuple (S, A, P, R, γ),
where S represents the state space, A the action space,
P( _s_ [′] | _s_, _a_ ) the transition dynamics, R( _s_, _a_, _s_ [′] ) the reward function, and γ ∈ [0, 1) the discount factor. The agent learns a policy
π( _a_ | _s_ ) that maximizes the expected discounted return :



with λ ∈ [0, 1] controlling the bias-variance tradeoff.
To stabilize learning, PPO introduces a clipped surrogate objective based on the probability ratio between the probability of
an action in the old and new policy:


_rt_ (θ) = [π][θ][(] _[a][t]_ [|] _[s][t]_ [)] (7)

πθold( _at_ | _st_ ) [.]


The clipped objective is defined as:


_L_ [CLIP] (θ) = E _t_  - min [�] _rt_ (θ) _A_ [ˆ] _t_, clip( _rt_ (θ), 1 − ϵ, 1 + ϵ) _A_ [ˆ] _t_  - [�], (8)


which discourages excessively large policy updates. The full
PPO loss includes a value function loss and an entropy bonus:


         -         _L_ (θ, ϕ) = E _t_ _L_ [CLIP] (θ) − _c_ 1( _V_ ϕ( _st_ ) − _Rt_ ) [2] + _c_ 2 _S_ [πθ]( _st_ ), (9)


where _Rt_ denotes the bootstrapped return used for value function training. The parameters (θ, ϕ) are optimized jointly using
stochastic gradient descent.


**4.** **The PetriRL Hyper-Heuristic Framework**


Since our hyperheuristic (HH) framework is built on reinforcement learning, it relies on the interaction between an agent
and its environment. In this section, we first describe the environment, which captures the dynamics and constraints of
the scheduling problem, and then introduce the HH agent that
learns to select the most appropriate scheduling rule based on
the observed system state. We conclude the section with the
advantages of the proposed approach.


_4.1._ _The JSSP Environment_

The environment consists of a JSSP model and a reward
function. We start with the JSSP model: following [41], we employ a colored timed Petri net (CTPN) to simulate the dynamics of the scheduling problem while enforcing all constraints.
In Figure 1, we depict a 5-job, 4-machine shop floor modeled
using a CTPN.
Starting from the top of the figure, each job is represented as
an ordered sequence of colored tokens. As introduced in Section 3.2, firing a transition moves a token from its parent place
to a child place if the necessary conditions are met. Controllable
transitions define the RL agent’s action space, and the sequence
of transition firings determines the job scheduling order. Once
the RL agent selects the processing sequence, the remaining
operations are managed automatically by the Petri net. Colored
transitions route tokens to the appropriate machine buffers. If
a machine is available, the token is processed. The processing
time is captured by the sojourn time of the token at the machine
place, which must elapse before the corresponding timed transition can fire.
The set of conditions, including the presence of a token in the
parent place, the matching of the token’s color, and the elapse of
the sojourn time, defines the Petri net’s guard function, which
determines which transitions are enabled based on the current
system state. This function is valuable not only as a constraint
enforcer but also as an action-masking provider. By masking



_Gt_ =



∞

- γ _[k]_ _Rt_ + _k_ +1. (3)

_k_ =0



RL algorithms can be categorized into value-based methods
that estimate expected returns using value functions _V_ ( _s_ ) or
_Q_ ( _s_, _a_ ), policy-based methods that directly optimize the policy parameters, and actor-critic methods that combine both approaches [43]. In this work, we adopt Proximal Policy Optimization (PPO) [44] . PPO employs a clipped surrogate objective to constrain policy updates, ensuring stable training by
preventing excessively large parameter changes.


_3.4._ _Proximal Policy Optimization algorithm_


The objective of the PPO agent is to maximize the expected
discounted return:


                  -                  - [∞]                  _J_ (θ) = Eτ∼πθ γ _[t]_ _r_ ( _st_, _at_ ), (4)

_t_ =0


where τ denotes a trajectory generated by πθ. The policy gradient is estimated using the advantage-weighted logprobabilities:


               -                E _st_, _at_ ∼πθ ∇θ log πθ( _at_ | _st_ ) _A_ [ˆ] _t_, (5)


where _A_ [ˆ] _t_ is an estimate of the advantage function. Advantage
estimates are computed using a learned value function _V_ ϕ( _s_ )
and Generalized Advantage Estimation (GAE):



_A_ ˆ _t_ =



∞


(γλ) _[l]_ [�] _rt_ + _l_ + γ _V_ ϕ( _st_ + _l_ +1) − _V_ ϕ( _st_ + _l_ ) [�], (6)
_l_ =0



4


combinatorial action space, the agent operates at a higher level
of abstraction by selecting dispatching rules from a predefined
set of low-level heuristics. As depicted in Figure 2, the agent’s
policy network observes the current system state and selects
the most appropriate dispatching rule for the current scheduling context. The selected heuristic is then applied to determine
which job-machine assignment should be executed, thereby updating the system state.

































Figure 1: Representation of a 5-job x 4-machine job shop scheduling problem
modeled using a timed colored Petri net.


invalid actions, it significantly improves the efficiency and stability of the training process.
The key elements to define in any reinforcement learning environment are the observation space, action space, and reward
function. As mentioned previously, the action space consists
of the controllable transitions in the Petri net. The observation space comprises the distribution of tokens across places
in the Petri net, also known as the _marking_, augmented with
additional temporal features such as elapsed time, remaining
processing time, and machine availability.
Concerning the reward function, we implement a terminal
makespan penalty where the agent is trained to minimize the
final makespan. The reward function is defined as:







Figure 2: PetriRL Hyper-heuristic framework: The policy network (left) selects
an appropriate heuristic based on the current state. The chosen heuristic then
decides the next action, which is executed in the colored timed Petri net environment modeling the JSSP (right).


In the proposed approaches, two action spaces are involved:
the hyper-heuristics action space, composed of all the dispatching rules as low-level heuristics _ht_ ∈H, and the environmentlevel action _at_ ∈A produced by applying that heuristic rule to
the current state _st_ in the Petri net environment, for example fire
transition T2 that allocates and operation from job 1 to machine
2. Let the set of low-level heuristics be :


H = { _h_ 1, _h_ 2, . . ., _hN_ }. (11)


Each heuristic _hk_ ∈H is a deterministic mapping


_hk_ : S →A. (12)


The RL agent selects a heuristic according to the policy:


πθ( _k_ | _st_ ) = _P_ θ( _hk_ | _st_ ), (13)


and the chosen heuristic produces the actual dispatching action under the dynamic masking induced by the Petri net guard
function, this means the Petri net Per-filter the invalid action
and provides the low-level heuristic only with possible actions
to decide from :


_at_ = _hk_ ( _st_ | mask( _st_ )). (14)


A commitment parameter is introduced: once a heuristic _hk_
is selected, it is applied for _x_ consecutive steps. The pseudo
code describing the approach is below:



_rt_ =




 - _C_ max, _t_ = _T_,
(10)
0, otherwise.



Despite its simplicity, this reward function consistently provided the most robust results compared to more complex reward
formulations. As Equation (10) shows, this reward structure
is sparse, providing feedback only at the end of each episode.
We experimented with various reward shaping techniques to
provide more frequent signals throughout the episode. However, most shaped rewards suffered from reward hacking, where
the agent exploited shortcuts to maximize intermediate rewards
while deteriorating the final makespan. The sparse terminal reward, though challenging for credit assignment, proved more
reliable in guiding the agent toward genuinely optimal scheduling policies.


_4.2._ _PetriRL Hyper Heuristic overview_

In the proposed Hyperheuristic approach, instead of directly
selecting job-machine assignments from a potentially large



5


**Algorithm 1** PPO Hyperheuristic for JSSP

1: **Input:** Environment E, heuristics H, policy πθ, commitment length _x_, episodes _N_
2: **Output:** Trained policy π [∗] θ
3: **for** episode = 1 to _N_ **do**
4: Reset E and observe _s_ 0
5: **while** not done **do**
6: Sample heuristic index _k_ ∼ πθ( _k_ | _st_ )
7: Commit to _hk_ for _x_ steps
8: **for** _i_ = 0 to _x_  - 1 **do**
9: Compute valid actions: Avalid ←{ _a_ : _G_ ( _a_ ) = 1}
10: Select _at_ + _i_ ← _hk_ ( _st_ + _i_, Avalid)

11: Execute _at_ + _i_, observe _rt_ + _i_ and _st_ + _i_ +1
12: _Rt_ ← _Rt_ + _rt_ + _i_
13: **end for**
14: Store ( _st_, _k_, _Rt_ )
15: _t_ ← _t_ + _x_
16: **end while**

17: Update πθ using PPO
18: **end for**
19: **return** π [∗] θ


_4.3._ _Analysis of the approach’s features_
The proposed approach offers several advantages, which we
will evaluate in the results section. These include a fixedsize action space, improved credit assignment, adaptive statedependent switching, enhanced interpretability, and the ability
to learn mixtures of heuristics.


_4.3.1._ _Reduced and fixed Action Space size_
Without Hyper-heuristics, the RL action space consists of the
controllable Petri net transitions, which represent decisions for
job selection and machine allocation:


|A| = (jobs waiting) × (machines), (15)


Growing combinatorially with problem size. By switching to
a hand-selected set of LLHs, the action space is drastically reduced.The dimension of the action space becomes :


|A| = |H|, H = { _h_ 1, _h_ 2, . . ., _hN_ }. (16)


This not only drastically reduces the complexity of the learning problem but also creates a fixed action space that is independent of the number of jobs or machines. Consequently, the
learned PPO policy is size-agnostic and can generalize across
different problem sizes, while benefiting from improved stability and sample efficiency.


_4.3.2._ _Policy Search over Heuristics_
Since each dispatching heuristic _hk_ can be represented as a
deterministic HH policy that selects the same heuristic at every decision point, the HH policy class strictly subsumes the
set of single-heuristic policies. Therefore, in principle, the optimal HH policy cannot perform worse than the best individual
heuristic:
max [≥] [max] _J_ ( _hk_ ), (17)
π∈Π _[J]_ [(][π][)] _k_



where Π denotes the set of all HH policies.
In practice, whether a learned HH actually outperforms the
best heuristic depends on the optimization algorithm, instance
diversity, and problem dynamics. Empirical studies suggest that
adaptive selection often yields higher returns than fixed heuristics [45]. We evaluate this behavior in our experiments and
present the results in Section 5.


_4.3.3._ _State-Dependent Switching_

The authors of [8] used simulation software to model and
analyze 44 dispatching rules, consisting of 14 single rules and
30 hybrid rules across many performance criteria. They found
that no single dispatching rule achieves optimal performance
across all measured criteria. Nevertheless, the study highlighted
key insights: SPT (Shortest Processing Time) excels at minimizing queue time, SPS (Shortest Process Sequence) performs
well in reducing WIP (Work In Process), and LWT (Longest
Waiting Time) is effective for minimizing queue length. Although MTWR (Most Total Work Remaining) emerged as the
most consistently effective single rule overall, these findings
demonstrate the potential for intelligent, state-dependent rule
selection, where an adaptive agent can dynamically choose the
most suitable rule based on the current system state, seamlessly
switching from one dispatching rule to another as conditions
change.


_4.3.4._ _Rule commitment and e_ ff _ect Credit Assignment_
In this section, we discuss the effect of introducing rule commitment on the credit assignment problem. In Section 4.1, we
established the motivation for using the makespan as a reward
function. However, this creates a credit assignment challenge.
Without commitment, the policy gradient becomes :



∇θ _J_ ≈



_T_ −1

- ∇θ log πθ( _at_ | _st_ ) �−γ _[T]_ [−] _[t]_ _C_ max − _V_ ϕ( _st_ )� . (18)

_t_ =0



Each timestep receives an advantage signal diluted across all _T_
decisions, making it difficult to identify which heuristic choices
actually contributed to the final makespan.
By committing to heuristic _hk_ for _x_ consecutive steps, we reduce the effective decision horizon from _T_ to _M_ = _T_ / _x_ . And
each decision now collects the advantages from all x commitment steps. The policy gradient under commitment becomes:



_x_ −1

- _Atm_ + _i_ . (19)

_i_ =0



∇θ _J_ commit ≈



_M_ −1


∇θ log πθ( _h_ _[m]_ _k_ [|] _[s][t]_ _m_ [)]
_m_ =0



Commitment introduces a form of temporal abstraction,
transforming the decision process from per-event control to
higher-level heuristic selection, conceptually related to temporally extended actions in the Options framework [46]. The commitment mechanism can improve credit assignment through
two complementary effects. First, aggregating advantages over
_x_ consecutive steps provides a more robust evaluation signal for
each heuristic choice. Instead of assessing a heuristic based on



6


a single timestep’s outcome, which may be noisy or unrepresentative, the agent evaluates it based on its cumulative performance over multiple timesteps. Second, reducing the number
of sequential decisions from _T_ to _T_ / _x_ simplifies the attribution
problem.
We empirically validate these benefits in Section 5.4, comparing heuristic commitment versus per-step switching on identical JSSP instances.


_4.3.5._ _Interpretability_
In the framework, we leverage one of the biggest advantages of HH: using low-level, explainable heuristics that are
often handcrafted by domain experts, without sacrificing performance. For instance, each action corresponds to a known
scheduling rule _hk_, thereby allowing the agent’s behavior to be
explained in a natural and interpretable manner. For example, in
state _st_, the agent may select the MTWR rule with probability
πθ( _k_ | _st_ ) = 80%, indicating a preference for prioritizing jobs
with the largest remaining workload under the current system
conditions.
This interpretability is further enhanced by our specific use
of HH within a colored timed Petri net framework. The Petri net
provides a real-time, human-understandable graphical representation of the system state previously discussed in section 5.4,
enabling users to verify the agent’s decisions visually. In particular, the user can directly observe that the selected action targets
the job with the most remaining work, as indicated by the job
place with the most significant number of tokens waiting in the
queue.


**5.** **Results and Discussion**


_5.1._ _Experimental Setup_
We evaluated our approach against a diverse set of methods from the literature for solving Job Shop Scheduling Problems. All methods were benchmarked on the widely used
Taillard dataset [47] to enable a direct comparison of performance. The Taillard benchmark comprises eight groups of instances with varying sizes, ranging from 15 jobs × 15 machines
to 100 jobs×20 machines. Within each size group, ten instances
are provided, each differing in processing times, job sequences,
and machine allocations .
The competing algorithms include learning-based neural network models such as the Disjunctive Graph Embedded Recurrent Decoding Transformer (DGERD) [23], Gated-Attention
Model (GAM) [24], and Graph Isomorphism Network (GIN)

[21]. Furthermore, we considered nature-inspired metaheuristics and evolutionary algorithms, including Tabu with the improved iterated greedy algorithm (TMMIG) [9], coevolutionary quantum genetic algorithm (CQGA) [11], genetic algorithm
combined with simulated annealing (HGSA) [12], and a hybrid genetic algorithm with tabu search (GA–TS) [10]. We
also compared our approach to classic dispatching rules such
as FIFO, SPT, SPS, LTWR, SPSP, and LPTN. Finally, we compared against our previous work, PetriRL, in which we used a
masked PPO agent to solve the JSSP problem in the traditional
standalone DRL framework [41].


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
|||~~(a)~~|||
||||||
|||||Raw<br>|
|||||~~Smoothed~~|


|Col1|Col2|Col3|Col4|Raw<br>Smoothed|
|---|---|---|---|---|
|||(c)|||
||||||
||||||



Figure 3: Agent training performances on a 20 jobs x 20 machines instance . ( **a** )
the episode mean reward, ( **b** ) the combined training loss,( **c** ) the entropy loss ( **d** )
the clipping range. The agent is trained for **1e6** steps with 5-step commitment
using the maskable PPO algorithm.


Subplot (a) shows a steady increase in the collected reward,
which directly reflects the makespan. In an effort to reduce the
incurred penalty, the agent works to minimize the makespan
while respecting the constraints imposed by the environment.
Subplot (b) depicts the total loss, combining the losses from
the policy network and the critic. This confirms that the critic
is becoming better at correctly predicting the values needed to



**Heuristics**


FIFO First-In-First-Out.
SPT Shortest Processing Time.
SPS Shortest Processing Sequence.
LTWR Longest Total Work Remaining.
SPSR Shortest Processing Sequence Remaining.
LPTN Longest Processing Time of Next Operation.
LWT Longest Waiting Time.


**Metaheuristics**


TMIIG Tabu Search with Modified Iterated Greedy Algorithm.
CQGA Coevolutionary Quantum Genetic Algorithm.
HGSA Hybrid Genetic Algorithm with Simulated Annealing.
TSGA Tabu Search combined with Genetic Algorithm.


**Learning-Based**


GIN Graph Isomorphism Network.
GAM Gated Attention Model.
DGERD Disjunctive Graph Embedded Recurrent Decoding
Transformer.
MPPO Maskable Proximal Policy Optimization.
integrated with Petri net (our previous work).


Table 1: Descriptions of all contending algorithms.


_5.2._ _Training performance_

Figure 3 illustrates the training performance of an agent
trained on a 20 jobs × 20 machines Job Shop Scheduling Problem . Four key metrics are reported to assess exploration, exploitation, and learning progress: episode mean reward, combined training loss, entropy loss, and clipping range.




_−_ 1960


_−_ 1980


_−_ 2000


_−_ 2020


_−_ 2040


_−_ 2 _._ 2


_−_ 2 _._ 4


_−_ 2 _._ 6













1000


800


600


400


200


0


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0





0 2 4 6 8 10
Step _×_ 10 [5]



0 2 4 6 8 10
Step _×_ 10 [5]



7


Inst Size FIFO SPT SPS LTWR SPSR LPTN LWT Heur- Best- Ours Gap
Avg Heuristic


ta01 15x15 1486 1454 1486 1454 1486 1639 1486 1761 **1454** **1454** 0.0%
ta11 20x15 1701 1771 1701 1771 1671 1712 1701 2083 1671 **1648** -1.4%
ta21 20x20 2089 2114 2089 2114 2111 2016 2089 2345 2016 **1977** -1.9%
ta31 30x15 2277 2312 2277 2312 2277 2260 2277 2517 2260 **2219** -1.8%
ta41 30x20 2543 2661 2543 2661 2543 2634 2543 3080 **2543** **2543** 0.0%
ta51 50x15 3590 3564 3590 3561 3590 3664 3590 3792 3496 **3433** -1.8%
ta61 50x20 3690 3619 3690 3619 3690 3572 3690 4097 3572 **3427** -4.1%
ta71 100x20 6270 6359 6270 6312 6270 6282 6270 6857 6248 **6190** -1.6%


**Average** - 2956 2982 2956 2976 2955 2972 2956 2976 2908 **2860** **-1.6%**


Table 2: Performance comparison of heuristic rules and the proposed super-heuristic on Taillard benchmark instances.


Inst Size DGERD GIN GAM TMIIG CQGA HGSA TSGA MPPO Best- Ours
Heur


ta01 15x15 1711 1547 1530 1486 1486 1324 **1282** 1436 1454 1454
ta11 20x15 1833 1775 1798 2011 2044 1713 1622 **1568** 1671 1648
ta21 20x20 2146 2128 2086 2973 2973 2331 2331 2064 2016 **1977**
ta31 30x15 2383 2379 2342 3161 3161 2731 2730 2166 2260 2219
ta41 30x20 **2541** 2604 2604 4274 4274 3198 3100 2576 2543 2543
ta51 50x15 3763 3394 3344 6129 6129 4105 4064 **3272** 3496 3433
ta61 50x20 3633 3594 3534 6397 6397 5536 5502 3505 3572 **3427**
ta71 100x20 6321 6098 6027 8077 8077 5964 **5962** 6366 6248 6190


**Average**  - 3041 2940 2908 4314 4318 3363 3324 2869 2908 **2860**


Table 3: Comparison of the proposed super-heuristic with representative approaches from the literature on Taillard benchmarks.



calculate the advantage, a crucial component of the policy network’s gradient.
Finally, subplots (c) and (d) show exploration behavior during the first 200,000 steps, characterized by high policy entropy.
This led the agent to clip more (approximately 30%) to maintain training stability, as demonstrated in subplot (d). Both subplots indicate a transition toward exploitation, with decreased
entropy and fewer action updates being clipped.


_5.3._ _Results analysis_


In line with the claims from the literature discussed in subsection 4.3.2, we empirically confirm that the learned HH outperforms the best single heuristic on the scheduling task, as
evaluated on the Taillard benchmark instances. Table 2 presents
the makespan results across problem instances ranging from
15 × 15 to 100 × 20 jobs and machines. Across all instance
sizes, the HH performed similarly to or better than the static
heuristics. The improvement ranges from 1.4% to 4.1%, with
an average improvement of 1.6% relative to the best heuristic
and 4% relative to the average performance of the heuristics.
This improvement can be explained by the state-dependent
switching behavior discussed in sub-section 4.3.3. Unlike static
low-level heuristics, HH can adapt dynamically to the current
system state, thereby improving condition handling on the shop
floor.



After comparing the performance of the HH algorithm with
traditional heuristic rules and demonstrating that it consistently
matches or outperforms the best heuristic for any given instance of the Taillard benchmark, we now turn to benchmarking
against approaches from the literature listed in the Table 1.
On average across all instance sizes, the HH approach
outperforms all benchmarked algorithms, achieving the lowest average makespan of 2860 steps. While its dominance
over fixed heuristics is clear, comparisons with metaheuristics
and learning-based approaches are more competitive, as each
method may excel on specific instance sizes but not consistently
across all instances. Nevertheless, the HH retains a distinct advantage in explainability. Unlike neural networks, which often function as black-box mappings from states to actions, HH
makes decisions using a set of interpretable dispatching rules.
For instance, if the agent allocates an operation from job two
under the MTWR rule, the decision can be directly traced to the
job with the most remaining work. This transparency provides
a more interpretable decision-making process while sustaining
strong performance across diverse scheduling instances.


_5.4._ _Ablation_


In the ablation study, we first qualitatively analyze the impact
of employing a hyper-heuristic, rather than acting directly in
the environment’s action space, on training performance. This
is followed by a quantitative assessment of the impact of the



8


number of commitment steps on the resulting makespan. Additionally, we evaluate the effects of different action-selection
strategies during inference, considering both a greedy and a
sampling-based approach.


_5.4.1._ _Commitment steps_



1960


1980


2000


2020


2040


2080


2100


2120


2140


|Col1|R<br>S|aw<br>moothed|
|---|---|---|
||(b)||
||||
||||











40000


30000


20000


10000


0


75000


50000





25000


0


|Col1|R|aw|
|---|---|---|
||S|moothed|
||S||
||(d)||
||||
||||



0 2 4 Step 6 8 ×10105



0 2 4 Step 6 8 ×10105



Table 4: efect of commitment horizon on performance

Instance size 1-step 5-steps 1000-steps


15×15 1486 1454 1454
20×15 1671 1648 1671
20×20 1979 1977 2016
30×15 2217 2219 2260
30×20 2554 2530 2543
50×15 3528 3433 3496
50×20 3718 3417 3572
100×20 6170 6190 6248


Average 2915 **2860** 2908


heuristic and the continuous switching baseline. Consistent with our previous findings that intermediate commitment
lengths provide superior training stability, this stability translated into a lower average makespan than continuous switching, while maintaining the flexibility to adapt heuristic selection throughout the scheduling process. In contrast, a 1000-step
commitment eliminates this adaptability, reducing performance
to that of a fixed heuristic policy.


_5.4.2._ _Action selection_
Finally, we tested the effect of two action selection methods
during inference: greedy deterministic selection and sampling.
In the greedy deterministic method, the algorithm selects the
action with the highest softmax probability, i.e., the action with
the highest logit value, using the argmax function. On the other
hand, in the sampling method, the action is chosen by sampling
from the categorical probability distribution produced by the
softmax function, introducing some randomness into the decision process. We used the deterministic method throughout the
study.
The results show that the makespan using sampling or deterministic selection yielded similar results, and we can attribute
this phenomenon to two main reasons. First, in multinomial
sampling, each action _ai_ is selected according to the probability
_pi_ as follows: _P_ (ˆ _a_ = _ai_ ) = _pi_ for _i_ = 1, 2, . . ., _n_ . If the agent is
trained sufficiently, the probability distribution of a given action
becomes so dominant that even with sampling, the agent almost
always selects that action.
Additionally, some heuristic rules yield the same decision
under a given state, making them equivalent. This can be seen
when analyzing the probability distribution in a given state. In
this case, two actions are co-dominant with similar probabilities. Despite the fact that the RL agent alternates between selecting the two actions, the final makespan remains the same. In
other words, the two different heuristics, under the given constraints, select the same action (e.g., selecting the same job for a
machine), especially in highly constrained problems like JSSP,
the valid action selection could be limited.
Exploration is a cornerstone of the RL framework. Like in
inference, environments with discrete action spaces often rely
on multinomial sampling to explore the solution space during
training. However, multinomial sampling can become limit


Figure 4: Comparison of two control strategies applied to the same environment
and problem instance (20 jobs x 20 machines): a hyperheuristic agent selecting
among a set of heuristic rules (blue) and a reinforcement learning agent acting
directly on the environment’s action space (red). Subfigures show: (a) rewards
collected by the hyperheuristic agent, (b) its associated training loss function,
(c) rewards collected by the direct-action RL agent, and (d) its corresponding
training loss function.


In Figure 4, we compare solving a 20 jobs × 20 machines
JSSP using two cases. In the first case, a PPO agent acts directly on the environment’s action space, whereas in the second
case, an HH agent operates on a heuristic action space with a
5-step commitment. Qualitatively, both the collected reward
function and the training loss are noticeably smoother in this
case, indicating more stable training. The commitment mechanism improves credit assignment by effectively shortening the
original non-heuristic trajectory length by a factor equal to the
number of commitment steps.
Quantitatively, the HH agent achieves a maximum reward of
approximately −1960 (corresponding to a makespan of 1960)
and stabilizes after roughly 4 × 10 [4] steps. In contrast, the direct
RL agent peaks at approximately -2080 and exhibits substantially more unstable performance.
We also evaluated the impact of commitment length by comparing three configurations: 1-step, 5-step, and 1000-step commitment. The 1-step commitment represents the lower bound
of our approach, equivalent to continuous rule switching at every timestep. The 5-step commitment is our default configuration used throughout this paper. The 1000-step commitment
represents the upper bound, effectively degenerating to a fixed
single-heuristic policy, since all benchmark instances have trajectories requiring fewer than 1000 action selections, forcing
the agent to commit to one rule for the entire episode.
The results demonstrate that the 5-step commitment achieves
the best performance, outperforming both the single best static



9


ing if one action’s probability dominates the distribution early,
thereby hindering the agent from exploring other parts of the
solution space. Further studying this phenomenon and finding
alternatives could enhance the exploration phase in RL.


**6.** **Conclusion**


In this paper, we addressed the Job Shop Scheduling Problem using a deep policy-based reinforcement learning hyperheuristic (DRL-HH) framework. To this end, we extended the
classical hyper-heuristic (HH) framework by two key mechanisms: action masking and action commitment.
Invalid actions are pre-filtered before reaching the lowlevel heuristics (LLHs) for decision-making. This is achieved
through the Petri net’s dynamic action masking capability,
which disables transitions according to guard-function rules.
Unlike traditional HH approaches that assess action validity
post hoc via move-acceptance strategies, often penalizing LLHs
for infeasible decisions, our method ensures that LLHs are evaluated exclusively on valid actions. This separation enables the
high-level policy to learn heuristic selection independently of
hard environmental constraints, thereby allowing a more objective and unbiased evaluation of the different LLHs.
We evaluated the proposed framework against a broad range
of competing methods, including classical heuristics, metaheuristics, and recent deep reinforcement learning approaches
such as Graph Isomorphism Networks, Gated Attention Models, and Transformer-based architectures. Across multiple Taillard benchmark instances, our method consistently achieved a
superior average makespan.
In addition, we introduced action commitment, inspired by
temporally extended actions in the Options framework (macroactions). Per-step switching tends to exacerbate credit assignment issues, while rigidly committing to a single heuristic for
an entire episode limits adaptability and often leads to suboptimal performance. As an alternative, we define a commitment
horizon as a tunable hyperparameter that controls the duration
for which a selected heuristic is applied.
Ablation studies show that intermediate commitment lengths
provide the best trade-off. Specifically, they qualitatively
improve training stability and quantitatively achieve a lower
makespan than both per-step heuristic switching and fullepisode commitment.
Finally, we investigated different action selection strategies
at the policy level, comparing deterministic greedy selection
with stochastic sampling. The results were essentially identical, which we attribute to two combined effects. First, early
dominance of specific action probabilities in the softmax output
causes multinomial sampling to collapse to near-deterministic
behavior. The second structural constraint imposed by the Petri
net leads multiple LLHs to produce identical low-level actions
under the same state.
Early probability dominance can limit exploration during
training, particularly in the initial learning phase. Addressing
this issue represents a promising direction for future work, potentially enabling richer exploration and the discovery of more
diverse scheduling strategies.



Overall, this work demonstrates that structurally informed
action filtering and temporal abstraction can enhance reinforcement learning–based hyper-heuristics.


**References**


[1] X. Zhang, G.-Y. Zhu, A literature review of reinforcement learning methods applied to job-shop scheduling
problems, Computers & Operations Research 175 (2025)
106929. `[doi:10.1016/j.cor.2024.106929](https://doi.org/10.1016/j.cor.2024.106929)` .


[2] M. R. Garey, D. S. Johnson, Computers and intractability:
A guide to the theory of NP-completeness, A Series of
books in the mathematical sciences, W. H. Freeman, San
Francisco, 1979.


[3] I. A. Chaudhry, A. A. Khan, A research survey: review
of flexible job shop scheduling techniques, International
Transactions in Operational Research 23 (3) (2016) 551–
591. `[doi:10.1111/itor.12199](https://doi.org/10.1111/itor.12199)` .


[4] P. Brucker, B. Jurisch, B. Sievers, A branch and bound
algorithm for the job-shop scheduling problem, Discrete
Applied Mathematics 49 (1-3) (1994) 107–127. `[doi:10.](https://doi.org/10.1016/0166-218X(94)90204-6)`
`[1016/0166-218X(94)90204-6](https://doi.org/10.1016/0166-218X(94)90204-6)` .


[5] W.-Y. Ku, J. C. Beck, Mixed integer programming models for job shop scheduling: A computational analysis,
Computers & Operations Research 73 (2016) 165–173.
`[doi:10.1016/j.cor.2016.04.006](https://doi.org/10.1016/j.cor.2016.04.006)` .


[6] J. C. Beck, T. K. Feng, J.-P. Watson, Combining constraint
programming and local search for job-shop scheduling,
INFORMS Journal on Computing 23 (1) (2011) 1–14.
`[doi:10.1287/ijoc.1100.0388](https://doi.org/10.1287/ijoc.1100.0388)` .


[7] J. A. Gromicho, J. J. van Hoorn, F. Saldanha-da Gama,
G. T. Timmer, Solving the job-shop scheduling problem optimally by dynamic programming, Computers &
Operations Research 39 (12) (2012) 2968–2977. `[doi:](https://doi.org/10.1016/j.cor.2012.02.024)`
`[10.1016/j.cor.2012.02.024](https://doi.org/10.1016/j.cor.2012.02.024)` .


[8] A. K. Kaban, Z. Othman, D. S. Rohmah, Comparison of
dispatching rules in job-shop scheduling problem using
simulation: a case study, International Journal of Simulation Modelling 11 (3) (2012) 129–140.


[9] J.-Y. Ding, S. Song, J. N. Gupta, R. Zhang, R. Chiong,
C. Wu, An improved iterated greedy algorithm with a
tabu-based reconstruction strategy for the no-wait flowshop scheduling problem, Applied Soft Computing 30
(2015) 604–613. `[doi:10.1016/j.asoc.2015.02.006](https://doi.org/10.1016/j.asoc.2015.02.006)` .


[10] M. S. Umam, M. Mustafid, S. Suryono, A hybrid genetic algorithm and tabu search for minimizing makespan
in flow shop scheduling problem, Journal of King Saud
University   - Computer and Information Sciences 34 (9)
(2022) 7459–7467. `[doi:10.1016/j.jksuci.2021.](https://doi.org/10.1016/j.jksuci.2021.08.025)`
`[08.025](https://doi.org/10.1016/j.jksuci.2021.08.025)` .



10


[11] G. Deng, M. Wei, Q. Su, M. Zhao, An effective coevolutionary quantum genetic algorithm for the no-wait
flow shop scheduling problem, Advances in Mechanical Engineering 7 (12) (2015) 1–10. `[doi:10.1177/](https://doi.org/10.1177/1687814015622900)`
`[1687814015622900](https://doi.org/10.1177/1687814015622900)` .


[12] H. Wei, S. Li, H. Jiang, J. Hu, J. Hu, Hybrid genetic
simulated annealing algorithm for improved flow shop
scheduling with makespan criterion, Applied Sciences
8 (12) (2018) 2621. `[doi:10.3390/app8122621](https://doi.org/10.3390/app8122621)` .


[13] K.-L. Huang, C.-J. Liao, Ant colony optimization combined with taboo search for the job shop scheduling problem, Computers & Operations Research 35 (4) (2008)
1030–1046. `[doi:10.1016/j.cor.2006.07.003](https://doi.org/10.1016/j.cor.2006.07.003)` .


[14] E. K. Burke, M. R. Hyde, G. Kendall, G. Ochoa, E. Ozcan, J. R. Woodward, Exploring hyper-heuristic methodologies with genetic programming, in: J. Kacprzyk,
L. C. Jain, C. L. Mumford (Eds.), Computational
Intelligence, Vol. 1 of Intelligent Systems Reference Library, Springer Berlin Heidelberg, Berlin,
Heidelberg, 2009, pp. 177–201. `[doi:10.1007/](https://doi.org/10.1007/978-3-642-01799-5{_}6)`
`[978-3-642-01799-5{\textunderscore}6](https://doi.org/10.1007/978-3-642-01799-5{_}6)` .


[15] A. Vela, G. H. Valencia-Rivera, J. M. Cruz-Duarte, J. C.
Ortiz-Bayliss, I. Amaya, Hyper-heuristics and scheduling
problems: Strategies, application areas, and performance
metrics, IEEE Access 13 (2025) 14983–14997. `[doi:10.](https://doi.org/10.1109/ACCESS.2025.3532201)`
`[1109/ACCESS.2025.3532201](https://doi.org/10.1109/ACCESS.2025.3532201)` .


[16] W. BOUAZZA, Hyper-heuristics applications to manufacturing scheduling: overview and opportunities, IFACPapersOnLine 56 (2) (2023) 935–940. `[doi:10.1016/j.](https://doi.org/10.1016/j.ifacol.2023.10.1685)`
`[ifacol.2023.10.1685](https://doi.org/10.1016/j.ifacol.2023.10.1685)` .


[17] Y. Liu, Y. Mei, M. Zhang, Z. Zhang, Automated heuristic design using genetic programming hyper-heuristic for
uncertain capacitated arc routing problem, in: P. A. N.
Bosman (Ed.), Proceedings of the Genetic and Evolutionary Computation Conference, ACM, New York, NY,
USA, 2017, pp. 290–297. `[doi:10.1145/3071178.](https://doi.org/10.1145/3071178.3071185)`
`[3071185](https://doi.org/10.1145/3071178.3071185)` .


[18] H. Guo, J. Liu, Y. Wang, C. Zhuang, An improved genetic
programming hyper-heuristic for the dynamic flexible job
shop scheduling problem with reconfigurable manufacturing cells, Journal of Manufacturing Systems 74 (2024)
252–263. `[doi:10.1016/j.jmsy.2024.03.009](https://doi.org/10.1016/j.jmsy.2024.03.009)` .


[19] F. Zhang, Y. Mei, S. Nguyen, K. C. Tan, M. Zhang,
Multitask genetic programming-based generative hyperheuristics: A case study in dynamic scheduling, IEEE
transactions on cybernetics 52 (10) (2022) 10515–10528.
`[doi:10.1109/TCYB.2021.3065340](https://doi.org/10.1109/TCYB.2021.3065340)` .


[20] F. Zhang, Y. Mei, S. Nguyen, M. Zhang, Evolving
scheduling heuristics via genetic programming with feature selection in dynamic flexible job-shop scheduling,
IEEE transactions on cybernetics 51 (4) (2021) 1797–
1811. `[doi:10.1109/TCYB.2020.3024849](https://doi.org/10.1109/TCYB.2020.3024849)` .




[21] C. Zhang, W. Song, Z. Cao, J. Zhang, P. S. Tan, X. Chi,
Learning to dispatch for job shop scheduling via deep
reinforcement learning, Advances in neural information
processing systems 33 (2020) 1621–1632.


[22] M. S. A. Hameed, A. Schwung, Graph neural networksbased scheduler for production planning problems using
reinforcement learning, Journal of Manufacturing Systems 69 (2023) 91–102. `[doi:10.1016/j.jmsy.2023.](https://doi.org/10.1016/j.jmsy.2023.06.005)`
`[06.005](https://doi.org/10.1016/j.jmsy.2023.06.005)` .


[23] R. Chen, W. Li, H. Yang, A deep reinforcement learning
framework based on an attention mechanism and disjunctive graph embedding for the job-shop scheduling problem, IEEE Transactions on Industrial Informatics 19 (2)
(2023) 1322–1331. `[doi:10.1109/TII.2022.3167380](https://doi.org/10.1109/TII.2022.3167380)` .


[24] G. Gebreyesus, G. Fellek, A. Farid, S. Fujimura,
O. Yoshie, Gated–attention model with reinforcement
learning for solving dynamic job shop scheduling problem, IEEJ Transactions on Electrical and Electronic Engineering 18 (6) (2023) 932–944. `[doi:10.1002/tee.](https://doi.org/10.1002/tee.23788)`
`[23788](https://doi.org/10.1002/tee.23788)` .


[25] M. Sanchez, J. M. Cruz-Duarte, J. C. Ortiz-Bayliss,
H. Ceballos, H. Terashima-Marin, I. Amaya, A systematic review of hyper-heuristics on combinatorial optimization problems, IEEE Access 8 (2020) 128068–128095.
`[doi:10.1109/ACCESS.2020.3009318](https://doi.org/10.1109/ACCESS.2020.3009318)` .


[26] E. K. Burke, M. Gendreau, M. Hyde, G. Kendall,
G. Ochoa, E. Özcan, R. Qu, Hyper-heuristics: a survey
of the state of the art, Journal of the Operational Research
Society 64 (12) (2013) 1695–1724. `[doi:10.1057/jors.](https://doi.org/10.1057/jors.2013.71)`
`[2013.71](https://doi.org/10.1057/jors.2013.71)` .


[27] E. Özcan, M. Misir, G. Ochoa, E. K. Burke, A reinforcement learning - great-deluge hyper-heuristic for examination timetabling, International Journal of Applied Metaheuristic Computing 1 (1) (2010) 39–59. `[doi:10.4018/](https://doi.org/10.4018/jamc.2010102603)`
`[jamc.2010102603](https://doi.org/10.4018/jamc.2010102603)` .


[28] N. Pillay, R. Qu, Hyper-Heuristics: Theory and Applications, Springer International Publishing, Cham, 2018.
`[doi:10.1007/978-3-319-96514-7](https://doi.org/10.1007/978-3-319-96514-7)` .


[29] E. K. Burke, M. R. Hyde, G. Kendall, G. Ochoa,
E. Özcan, J. R. Woodward, A classification of hyperheuristic approaches: Revisited, in: M. Gendreau, J.Y. Potvin (Eds.), Handbook of Metaheuristics, Vol.
272 of International Series in Operations Research
& Management Science, Springer International Publishing, Cham, 2019, pp. 453–477. `[doi:10.1007/](https://doi.org/10.1007/978-3-319-91086-4{_}14)`
`[978-3-319-91086-4{\textunderscore}14](https://doi.org/10.1007/978-3-319-91086-4{_}14)` .


[30] J. H. Drake, A. Kheiri, E. Özcan, E. K. Burke, Recent
advances in selection hyper-heuristics, European Journal
of Operational Research 285 (2) (2020) 405–428. `[doi:](https://doi.org/10.1016/j.ejor.2019.07.073)`
`[10.1016/j.ejor.2019.07.073](https://doi.org/10.1016/j.ejor.2019.07.073)` .



11


[31] C. Li, X. Wei, J. Wang, S. Wang, S. Zhang, A [review](https://peerj.com/articles/cs-2141/)
of reinforcement [learning](https://peerj.com/articles/cs-2141/) based hyper-heuristics, PeerJ
Computer Science 10 (2024) e2141. `[doi:10.7717/](https://doi.org/10.7717/peerj-cs.2141)`
`[peerj-cs.2141](https://doi.org/10.7717/peerj-cs.2141)` .
URL `[https://peerj.com/articles/cs-2141/](https://peerj.com/articles/cs-2141/)`


[32] S. S. Choong, L.-P. Wong, C. P. Lim, Automatic design
of hyper-heuristic based on reinforcement learning, Information Sciences 436-437 (2018) 89–107. `[doi:10.1016/](https://doi.org/10.1016/j.ins.2018.01.005)`
`[j.ins.2018.01.005](https://doi.org/10.1016/j.ins.2018.01.005)` .


[33] A. S. Ferreira, R. A. Goncalves, A. Trinidad Ramirez
Pozo, A multi-armed bandit hyper-heuristic, in: 2015
Brazilian Conference on Intelligent Systems (BRACIS),
IEEE, 2015, pp. 13–18. `[doi:10.1109/BRACIS.2015.](https://doi.org/10.1109/BRACIS.2015.31)`
`[31](https://doi.org/10.1109/BRACIS.2015.31)` .


[34] A. Lamghari, R. Dimitrakopoulos, Hyper-heuristic approaches for strategic mine planning under uncertainty,
Computers & Operations Research 115 (2020) 104590.
`[doi:10.1016/j.cor.2018.11.010](https://doi.org/10.1016/j.cor.2018.11.010)` .


[35] A. Dantas, A. F. d. Rego, A. Pozo, Using deep q-network
for selection hyper-heuristics, in: F. Chicano, K. Krawiec
(Eds.), Proceedings of the Genetic and Evolutionary Computation Conference Companion, ACM, New York, NY,
USA, 2021, pp. 1488–1492. `[doi:10.1145/3449726.](https://doi.org/10.1145/3449726.3463187)`
`[3463187](https://doi.org/10.1145/3449726.3463187)` .


[36] Y. Zhang, R. Bai, R. Qu, C. Tu, J. Jin, A deep reinforcement learning based hyper-heuristic for combinatorial optimisation with uncertainties, European Journal of
Operational Research 300 (2) (2022) 418–427. `[doi:](https://doi.org/10.1016/j.ejor.2021.10.032)`
`[10.1016/j.ejor.2021.10.032](https://doi.org/10.1016/j.ejor.2021.10.032)` .


[37] C. Tu, R. Bai, U. Aickelin, Y. Zhang, H. Du, A deep reinforcement learning hyper-heuristic with feature fusion for
online packing problems, Expert Systems with Applications 230 (2023) 120568. `[doi:10.1016/j.eswa.2023.](https://doi.org/10.1016/j.eswa.2023.120568)`
`[120568](https://doi.org/10.1016/j.eswa.2023.120568)` .


[38] O. Udomkasemsub, B. Sirinaovakul, T. Achalakul, Phh:
Policy-based hyper-heuristic with reinforcement learning,
IEEE Access 11 (2023) 52026–52049. `[doi:10.1109/](https://doi.org/10.1109/ACCESS.2023.3277953)`
`[ACCESS.2023.3277953](https://doi.org/10.1109/ACCESS.2023.3277953)` .


[39] J. Kallestad, R. Hasibi, A. Hemmati, K. Sörensen, A general deep reinforcement learning hyperheuristic framework for solving combinatorial optimization problems,
European Journal of Operational Research 309 (1) (2023)
446–468. `[doi:10.1016/j.ejor.2023.01.017](https://doi.org/10.1016/j.ejor.2023.01.017)` .


[40] T. Cui, N. Du, X. Yang, S. Ding, Multi-period portfolio optimization using a deep reinforcement learning
hyper-heuristic approach, Technological Forecasting and
Social Change 198 (2024) 122944. `[doi:10.1016/j.](https://doi.org/10.1016/j.techfore.2023.122944)`
`[techfore.2023.122944](https://doi.org/10.1016/j.techfore.2023.122944)` .


[41] S. Lassoued, A. Schwung, Introducing petrirl: An innovative framework for jssp resolution integrating petri nets



and event-based reinforcement learning, Journal of Manufacturing Systems 74 (2024) 690–702. `[doi:10.1016/](https://doi.org/10.1016/j.jmsy.2024.04.028)`
`[j.jmsy.2024.04.028](https://doi.org/10.1016/j.jmsy.2024.04.028)` .


[42] S. Lassoued, L. S. Baheti, N. Weiß-Borkowski, S. Lier,
A. Schwung, Flexible manufacturing systems intralogistics: Dynamic optimization of agvs and tool sharing using
colored-timed petri nets and actor–critic rl with actions
masking, Journal of Manufacturing Systems 82 (2025)
405–419. `[doi:10.1016/j.jmsy.2025.06.017](https://doi.org/10.1016/j.jmsy.2025.06.017)` .


[43] S. Richard, B. Andrew, Reinforcement learning: An introduction, A Bradford Book, Cambridge, MA, USA, 1998.
`[doi:10.5555/3312046](https://doi.org/10.5555/3312046)` .


[44] J. Schulman, F. Wolski, P. Dhariwal, A. Radford,
O. Klimov, Proximal policy optimization algorithms,
arXiv preprint (2017).


[45] P. Cowling, G. Kendall, E. Soubeiga, A hyperheuristic
approach to scheduling a sales summit, in: Edmund K.
Burke, Wilhelm Erben (Eds.), Practice and Theory of Automated Timetabling III, Third International conference
PATAT, Lecture Notes in Computer Science, Springer,
Konstanz Germany, 2001, pp. 176–190. `[doi:10.1007/](https://doi.org/10.1007/3-540-44629-X{_}11)`
`[3-540-44629-X{\textunderscore}11](https://doi.org/10.1007/3-540-44629-X{_}11)` .


[46] R. S. Sutton, D. Precup, S. Singh, Between mdps and
semi-mdps: A framework for temporal abstraction in
reinforcement learning, Artificial Intelligence 112 (12) (1999) 181–211. `[doi:10.1016/S0004-3702(99)](https://doi.org/10.1016/S0004-3702(99)00052-1)`
`[00052-1](https://doi.org/10.1016/S0004-3702(99)00052-1)` .


[47] E. Taillard, Benchmarks for basic scheduling problems,
European Journal of Operational Research 64 (2) (1993)
278–285. `[doi:10.1016/0377-2217(93)90182-m](https://doi.org/10.1016/0377-2217(93)90182-m)` .



12


