European Journal of [Operational](https://doi.org/10.1016/j.ejor.2021.10.032) Research 300 (2022) 418–427


Contents lists available at [ScienceDirect](http://www.ScienceDirect.com)

# ~~European Journal of Operational Research~~


journal homepage: [www.elsevier.com/locate/ejor](http://www.elsevier.com/locate/ejor)


Discrete Optimization

## A deep reinforcement learning based hyper-heuristic for combinatorial optimisation with uncertainties


Yuchang Zhang [a], Ruibin Bai [a][,][∗], Rong Qu [b], Chaofan Tu [a], Jiahuan Jin [a]


a _School_ _of_ _Computer_ _Science,_ _University_ _of_ _Nottingham_ _Ningbo_ _China,_ _Ningbo_ _315100,_ _China_
b _School_ _of_ _Computer_ _Science,_ _University_ _of_ _Nottingham,_ _Nottingham_ _NG8_ _1BB,_ _UK_



a r t i c l e i n f 

_Article_ _history:_
Received 10 November 2020
Accepted 13 October 2021
Available online 21 October 2021


_Keywords:_
Transportation
2D packing
Hyper-heuristics
Deep reinforcement learning
Container truck routing


**1.** **Introduction**



a b s t r a c t


In the past decade, considerable advances have been made in the field of computational intelligence and
operations research. However, the majority of these optimisation approaches have been developed for
deterministically formulated problems, the parameters of which are often assumed perfectly predictable
prior to problem-solving. In practice, this strong assumption unfortunately contradicts the reality of many
real-world problems which are subject to different levels of uncertainties. The solutions derived from
these deterministic approaches can rapidly deteriorate during execution due to the over-optimisation
without explicit consideration of the uncertainties. To address this research gap, a deep reinforcement
learning based hyper-heuristic framework is proposed in this paper. The proposed approach enhances the
existing hyper-heuristics with a powerful data-driven heuristic selection module in the form of deep reinforcement learning on parameter-controlled low-level heuristics, to substantially improve their handling
of uncertainties while optimising across various problems. The performance and practicality of the proposed hyper-heuristic approach have been assessed on two combinatorial optimisation problems: a realworld container terminal truck routing problem with uncertain service times and the well-known online
2D strip packing problem. The experimental results demonstrate its superior performance compared to
existing solution methods for these problems. Finally, the increased interpretability of the proposed deep
reinforcement learning hyper-heuristic has been exhibited in comparison with the conventional deep reinforcement learning methods.


© 2021 Elsevier B.V. All rights reserved.



Research on combinatorial optimisation problems is of vital
importance because of their broad applications in various realworld scenarios, including transportation, logistics, production, resource allocation, timetabling, digital services, finance and numerous other domains. Despite the recent advances, existing studies
have largely focused on algorithmic development on the deterministic variant of the problems, in which the parameters to define
the problems are assumed to be known in advance. It is not always possible to acquire accurate information of all the problem
characteristics in most real-life scenarios. Instead, the real values
of problem parameters are often sequentially revealed over time
during decision making. In such situations, solutions generated


∗ Corresponding author.
_E-mail_ _addresses:_ [Yuchang.Zhang@nottingham.edu.cn](mailto:Yuchang.Zhang@nottingham.edu.cn) (Y. Zhang),
[Ruibin.Bai@nottingham.edu.cn](mailto:Ruibin.Bai@nottingham.edu.cn) (R. Bai), [Rong.Qu@nottingham.ac.uk](mailto:Rong.Qu@nottingham.ac.uk) (R. Qu),
[Chaofan.Tu@nottingham.edu.cn](mailto:Chaofan.Tu@nottingham.edu.cn) (C. Tu), [Jiahuan.Jin@nottingham.edu.cn](mailto:Jiahuan.Jin@nottingham.edu.cn) (J. Jin).


[https://doi.org/10.1016/j.ejor.2021.10.032](https://doi.org/10.1016/j.ejor.2021.10.032)
0377-2217/© 2021 Elsevier B.V. All rights reserved.



as a priori often encounter various issues such as inferior service quality, increased costs, and infeasibile solutions, all of which
would lead to substantial losses. For example, in Uncertain Capacitated Arc Routing Problem studied in Mei, Tang, & Yao (2010), offline methods showed to generate infeasible solutions when the
actual demand of a task exceeds the remaining capacity of the vehicle, and consequently, the vehicle cannot fully serve the task as
expected by the offline methods.
Thus, it is crucial to develop alternative methodologies that
can accommodate uncertainties as well as make decisions that
can sufficiently balance the conflicts between solution optimality
and its resilience to unpredictable (sometimes even disruptive)
changes. This research focuses on adaptive algorithms that are
trained offline but deployed in real-time, so that decisions are
dynamically made in sequence pursuant to the actual situations
revealed. This provides decision-makers with the maximum flexibility to react to changes while maintaining the high quality from
the resulting solution.
One of the possible methods for this purpose is the hyperheuristics, originally proposed as a high-level search paradigm that


_Y._ _Zhang,_ _R._ _Bai,_ _R._ _Qu_ _et_ _al._ _European_ _Journal_ _of_ _Operational_ _Research_ _300_ _(2022)_ _418–427_



aims to achieve an increased generality in performance across
different problem instances and problem domains. Different from
the metaheuristic approaches that operate directly in the space
of solutions, hyper-heuristics search in the heuristic space, the
landscape of which is considered less problem-dependent than
that of the solution search space (Burke et al., 2013; Pillay &
Qu, 2018a). The idea of hyper-heuristic originated from Fisher &
Thompson (1963) in the 1960s before the term was formally introduced by Cowling, Kendall, & Soubeiga (2000). Hyper-heuristics
have been studied to solve various combinatorial optimisation
problems, such as timetabling (Soria-Alcaraz et al., 2014), production scheduling (Rahimian, Akartunalı, & Levine, 2017) and vehicle
routing (Ahmed, Mumford, & Kheiri, 2019), and potentially can be
a promising candidate framework to address online optimisation
problems.
This work is motivated by the demand of advanced algorithms for solving challenging online combinatorial optimisation
problems, taking advantage of both the known problem structures as well as a large amount of unlabelled historical data
reflecting the uncertainties. To be more adaptable to industrial
applications, the proposed algorithms must also cater to a certain level of interpretability. Bearing these requirements in mind,
we propose a new hyper-heuristic method that uses a double deep _Q_ -network (DDQN) (Van Hasselt, Guez, & Silver, 2016)
to train a heuristic selection module from a set of low-level,
human-interpretable heuristics in different problem-solving scenarios. The DDQN offers good performance and training stability,
and it has been used to solve problems in different fields, such as
edge computing (Chen et al., 2018) and recommendation systems
(Zheng et al., 2018).
Our proposed method extends the previous research in the followings: (1) different from the DRL methods in Mnih et al. (2013),
Mnih et al. (2015) and Van Hasselt et al. (2016), which are designed for solving gaming problems without obvious mathematical formulations, our DRL-HH is applied to classical combinatorial optimisation problems which have rich literature, especially
for their offline version of the problems. (2) compared with previous hyper-heuristics, the simple _Q_ -learning mechanism is replaced
with a much powerful DDQN, which provides much better ability
to handle high dimensionality data; (3) the proposed framework
is now applicable for online combinatorial optimisation problems
while the previous _Q_ -learning based hyper-heuristics are designed
for offline optimisation.
Compared to online genetic programming (GP) hyper-heuristic
methods (e.g. Chen, Bai, Qu, Dong, & Chen, 2020; MacLachlan, Mei,
Branke, & Zhang, 2020), our proposed method could offer better
performance handling a large amount of training data with much
higher dimensionality of features, on which the evolution of a GP
decision tree is often highly challenging (if not impossible). A GP
based method will need a set of pre-defined features as well as
a set of customised operators. Compared to supervised learning
methods heavily used in data predictions and pattern recognition,
our proposed method does not require pre-labelled data by experts. In fact, most of the problem instances addressed in this paper do not normally have ground-truth solutions due to their difficulty. The decisions made in practice are not optimal because of
the problem complexity.
The remainder of this paper is organised as follows:
Section 2 reviews hyper-heuristics and related problems.
Section 3 describes the proposed DRL hyper-heuristic framework. In Sections 4 and 5, the proposed framework is evaluated
by solving two considerably different combinatorial optimisation
problems with uncertainties, followed with discussions of the experimental design and results analysis. Finally, Section 6 concludes
the paper.



**2.** **Literature** **review**


Boosted by the increased computing power and more sophisticated optimisation algorithms that are now capable of exploiting more advanced problem structures, possibilities exist to tackle
optimisation problems of considerably larger sizes and to obtain
solutions of substantially higher quality in terms of stated objectives. Among these methods, hyper-heuristics have been explored
with the primary goal of raising the generality of the performance
of optimisation methods across different problem domains and instances. In broad terms, hyper-heuristics can be defined as “heuristics to select or generate heuristics” (Burke et al., 2010). The selection hyper-heuristic, being the focus of this paper, can be further divided into two types: construction-based and perturbationbased. Selection perturbative hyper-heuristics start from a complete solution and then iteratively select a low-level heuristic
among a set of perturbative low-level heuristics (often neighbourhood operators) that can efficiently search the solution space
and rapidly improve the incumbent solution. Selection constructive
hyper-heuristics learn to select among a set of constructive lowlevel heuristics at each point of solution construction, incrementally building a complete solution to a given optimisation problem
(Pillay & Qu, 2018b).
Since the initial introduction in 2000, hyper-heuristics have
received progressively increasing research attention, especially in
the past few years. The number of yearly publications on hyperheuristics is close to 100 in the Web of Science database. Among
these publications, the majority of existing literature on selection
hyper-heuristics is perturbative hyper-heuristics, which operate on
sets of perturbative low-level heuristics searching upon complete
solutions for optimisation problems (Bai, Blazewicz, Burke, Kendall,
& McCollum, 2012; Drake, Kheiri, Özcan, & Burke, 2020). Studies have been conducted exploring different pairwise combinations of selection and move acceptance (Burke et al., 2013). A
Modified Choice Function was chosen by Choong, Wong, & Lim
(2019) to solve TSP by choosing between low-level heuristics
within a swarm-based evolutionary algorithm. In Zamli, Alkazemi,
& Kendall (2016), a tabu search hyper-heuristic was utilised for
combinatorial interaction testing, choosing among four low-level
metaheuristics for _t_ -way test suite generation, and obtaining good
results for problems with up to 6-way interactions. Traditional reinforcement learning was adopted in selection perturbative hyperheuristics. Kheiri & Keedwell (2017) proposed a sequence-based
selection hyper-heuristic that maintains scores representing the
probability of choosing a low-level heuristic. The scores are updated by employing a reinforcement learning strategy during the
search. The selection perturbative hyper-heuristics can perform
well for problems with perfect and complete information, but
could potentially suffer from performance degradation when dealing with uncertain factors or dynamic events which impact on the
performance of decisions obtained offline.
As depicted in Fig. 1, selection constructive hyper-heuristics
operate upon a set of constructive low-level heuristics to incrementally build solutions. A constructive heuristic can be certain
rules, suitable building blocks/patterns and partial optimal solutions premised on certain mathematical models. There can also be
some random assignments in certain cases to diversify the search.
A key decision in constructive hyper-heuristics is to intelligently
select the most suitable constructive heuristic(s) in the heuristic
space at each step of solution construction while at the same time
satisfying various constraints. The ultimate goal is to acquire an
optimal (or near-optimal) sequence of constructive heuristics that
builds a high-quality solution in the solution space. The challenge
is to build a reliable mapping between the problem states and constructive heuristics. The interface between the solution space and



419


_Y._ _Zhang,_ _R._ _Bai,_ _R._ _Qu_ _et_ _al._ _European_ _Journal_ _of_ _Operational_ _Research_ _300_ _(2022)_ _418–427_



**Fig.** **1.** A selection constructive hyper-heuristic framework.


hyper-heuristics renders the possibility of building a system that
performs effectively across various solution spaces.
Evolutionary algorithms have been the most adopted methods
to explore sequences of low-level heuristics for solution construction. With low-level graph colouring heuristics, Burke, McCollum,
Meisels, Petrovic, & Qu (2007) proposed a constructive selection
hyper-heuristic to solve educational timetabling problems. Soghier
& Qu (2013) presented a hybrid approach for exam timetabling
utilising classic graph colouring heuristics to choose and assign an
exam before using bin packing heuristics to allocate a time slot
and room. To generate probability distributions of sequences of
low-level heuristics at different stages of a search, Qu, Pham, Bai,
& Kendall (2015) applied a Univariate Marginal Distribution Algorithm (UMDA). Gomez & Terashima-Marín (2018) used an evolutionary algorithm to evolve rules to select low-level heuristics for
solving multi-objective two-dimensional bin packing problems.
In these selection constructive hyper-heuristics, a decision is
made step by step at each decision point, offering the potential
to solve online combinatorial optimisation problems. However, the
aforementioned methods can only perform effectively when certain vital information of problem characteristics is known beforehand. For example, in 2-D bin packing problems, the items (rectangles) need to be known in advance. After being sorted, the rectangles are fed to the algorithm. In timetabling problems, all the
events that are not yet scheduled similarly need to be known in
advance, and are then ordered by certain low-level heuristics. For
instance, events are ordered according to the number of feasible
timeslots available in the partial solution at that time, or in terms
of the number of conflicts they have with those already scheduled
in the timetable. For real-world online problems, this required information is quite often not available.
To handle uncertainties in real-world problems in a very flexible way, Chen, Bai, Dong, Qu, & Kendall (2016) established a manually crafted dynamic heuristic based on the human experience.
The algorithm obtained solutions that are superior to those used
in practice. This type of manual heuristics can be used as baselines for performance evaluations but are often far from optimality.
MacLachlan et al. (2020) proposed a Genetic Programming HyperHeuristic (GPHH) to develop routing policy for the Uncertain Capacitated Arc Routing Problem, while Chen et al. (2020) proposed
a data-driven genetic programming heuristic that evolved different
decision-making rules (heuristics) in solving real-world truck routing problems in a container port. Superior results were reported in
comparison with those from Chen et al. (2016).


420



The GP based hyper-heuristics used by MacLachlan et al.
(2020) and Chen et al. (2020) showed to improve the manual
heuristic in Chen et al. (2016), as the resulting heuristic (in the
form of a GP decision tree) is evolved with a large number of instances, thus of better average performance than human-designed
ones. However, the evolution process of GP can be extremely timeconsuming on hundreds of training instances, and the resulting
tree can be too large to be used in practice with a large number of
features and terminal operators.
In the present research, a Deep Reinforcement Learning (DRL)
based selection constructive hyper-heuristic is proposed to solve
difficult online combinatorial optimisation problems with uncertain variables revealed over time. Compared against the existing
hyper-heuristics, the proposed hyper-heuristic method can take advantage of large scale historical data of the random variables to
train the heuristic selection module so that robustness can be built
into the constructed solution. Furthermore, we note that the solutions obtained by the proposed method is of improved interpretability, compared with those obtained by traditional deep reinforcement learning. This is due to: Firstly, the underlying actions
in the proposed framework are human-understandable heuristics
(rather than direct variable fixing actions in conventional deep reinforcement learning); Secondly, through the spectrum analysis of
the state-action pairs, we can identify decision patterns in which
the agent prefers to choose certain low-level heuristics (actions) at
specific decision points (states).


**3.** **The** **proposed** **DRL** **hyper-heuristic** **framework**


In the new hyper-heuristic framework for online combinatorial optimisation problems, a deep reinforcement learning is introduced into an existing selection constructive hyperheuristic framework. Specifically, a double deep _Q_ -network (DDQN)
(Van Hasselt et al., 2016) was utilised to train the present heuristic
selection module exhibited in Fig. 1. Details are described in the
following sub-sections.
DRL combines reinforcement learning (RL) and deep learning.
Since it was first proposed by Mnih et al. (2013), DRL has attracted
intensive attention, with the highlight of AlphaGo (Silver et al.,
2016) which beat the world Go champion Lee Sedol in 2016. The
deep neural networks in DRL are capable of perceiving and extracting advanced features from data automatically; while RL can
iteratively improve the decision-making thereof by ’trial and error’ interactions with the problem model. Compared with some
greedy and myopic online algorithms, such as best-fit for online bin-packing and nearest neighbour heuristic for TSP, the
proposed method can strategically give up some current rewards
to obtain a bigger reward in the future thanks to the intelligence
built in the DRL agent through offline training based on the large
amount of data containing hidden information regarding uncertainties. It can, therefore, more effectively handle problems with
uncertainties.


_3.1._ _DRL_ _based_ _hyper-heuristics_


The proposed DRL based constructive hyper-heuristic framework is depicted in Fig. 2. In the framework, the actions are a set
of parameterised heuristics (often referred to as low-level heuristics in hyper-heuristics). The selection of these actions/heuristics
is based on two state-vectors (see Section 3.3 for details) and the
historical experience of the DRL agent.
The DRL agent is represented as a value function _Q_ _(s,_ _a)_ with
respect to state _s_ and action _a_, and corresponds to the heuristic
selection module in the constructive hyper-heuristic. At each decision point, the agent selects and then executes an action in accordance with the states of the partial solution, and acquires reward


_Y._ _Zhang,_ _R._ _Bai,_ _R._ _Qu_ _et_ _al._ _European_ _Journal_ _of_ _Operational_ _Research_ _300_ _(2022)_ _418–427_


egy that substantially improved the performance of the basic DQN.
The target network (with parameters _θt_ [−][)] [and] [the] [online] [network]
(with parameters _θt_ ) share the same structure, but the parameters
of the target network are only updated every _τ_ steps from the online network. Here, _τ_ is a parameter to define how frequently the
target network parameters ( _θt_ [−][)] [are] [updated.] [The] [target] [used] [by]
DQN can be described by Eq. (2).

_Yt_ _[DQN]_ ≡ _Rt_ +1 + _γ_ max _a_ _[Q]_ _[(][S][t]_ [+][1] _[,]_ _[a]_ [;] _[θ]_ _t_ [ −] _[)]_ (2)


Owing to the max operator in the standard DQN in Eq. (2), the
DQN agent is more likely to select overestimated values, leading to
over-optimistic value estimates. Double-DQN (DDQN) is hence introduced to reduce over-estimations by decomposing the max operation in the target into action selection and action evaluation.
Then, the target used by DDQN is changed to Eq. (3):



**Fig.** **2.** The DRL hyper-heuristic framework.


feedback from the problem model. The _Q_ function is defined as
the expectation of discounted cumulative rewards, as denoted in
Eq. (1), where _γ_ is the discounted factor, _t_ is the time step, _R_ is
the reward and _π_ is the policy.

_Q_ _[π]_ _(s,_ _a)_ = E[ _Rt_ + _γ Rt_ +1 + _γ_ [2] _Rt_ +2 + _._ _._ _._ | _St_ = _s,_ _At_ = _a,_ _π_ ] (1)


Notably, in the DRL framework, the training data of the states, actions and rewards is generated during the interactions between the
agent and the problem model with random parameters. Thus, the
difficulty in building the training data or the issues of low-quality
labels in supervised learning can be avoided.
The proposed DRL based constructive hyper-heuristic method
capitalises on both the problem mathematical model (via model
derived solution states) as domain knowledge and the large
amount of training data possibly available from previous real-life
experience and/or simulations. The mathematical model provides
the main structures and properties of the problem while the uncertainties are not modelled mathematically. Instead, it is assumed
that all the information of uncertainties is implicitly given in the
form of the training data and the proposed algorithm is expected
to perform well across all uncertain scenarios. Thus, this framework is believed to be relatively easier to train than the previous
data-driven methods due to the use of additional information as
pre-knowledge from the mathematical model.


_3.2._ _Offline_ _training_ _of_ _double_ _deep_ Q _network_ _(DDQN)_


Deep _Q_ -network (DQN) is a widely used robust DRL method,
which combines a deep neural network function approximator
with the classical _Q_ -learning algorithm to learn a state-action value
function. By acting greedily, a policy can be iteratively acquired
(Mnih et al., 2013). Numerous methods to enhance the performance of the original DQN have been proposed in prior literature.
In the present framework, the double deep _Q_ network method
(DDQN) (Van Hasselt et al., 2016) with the experience replay strategy (Lin, 1993) is adopted because of their performance consistency and fast convergence. Although the availabilities of other DRL
methods that may be better than DDQN are noted, the focus in this
research is on the interactions between DRL and hyper-heuristics,
rather than exploring the best DRL method in the context of hyperheuristics. Particular focus is centred on the hybridisation of datadriven and model-driven schemes as well as the interpretability of
the proposed DRL hyper-heuristic framework.
DQN is a deep neural network that outputs a vector of actions’ (i.e. low-level heuristics) preference values _Q_ _(s,_ •; _θ_ _)_ given
state _s_, where _θ_ is the set of parameters of the network that can
be trained to help select the most appropriate heuristics. Mnih
et al. (2015) utilised a target network and experience replay strat


_Yt_ _[DDQN]_ ≡ _Rt_ +1 + _γ Q_ _(St_ +1 _,_ arg max _Q_ _(St_ +1 _,_ _a_ ; _θt_ _)_ ; _θt_ [−] _[)]_ (3)
_a_


In this research, during the interactions with the problem model
with random parameters, at each time step _t_, the DRL agent acquires an explicit partial solution state _sa_ from the problem model
directly; then the model derived solution state _sb_ is calculated pursuant to the dynamics of the problem model. After that, the state
_s_ is denoted as the concatenation of _sa_ and _sb_ . In accordance with
the state, the DRL agent takes an action _a_ (a low-level heuristic)
and obtains a reward _r_ . Meanwhile, the action takes the partial solution to a new state _s_ [′], which contains _s_ [′] _a_ [and] _[s]_ [′] _b_ [,] [as] [mentioned]
above. In this case, a piece of data _e_ is obtained, indicated as a tuple _(s,_ _a,_ _r,_ _s_ [′] _)_ . At each time step, _e_ is added into the data pool _M_ .
During the training process, the experience replay mechanism is
applied. Every time, a mini-batch of training data (a random set of
experiences) is sampled from the data pool _M_ . The details of the
training process are shown in Algorithm 1 in Appendix A.


_3.3._ _Solution_ _state_


The proposed DRL hyper-heuristic constructs a solution step by
step by repeatedly calling a chosen constructive heuristic. At each
step, two state vectors on the current partial solution are passed
to the hyper-heuristic as reference information for decision making (i.e. choice of the most appropriate constructive heuristic from
the set of heuristics at the disposal thereof). The first state vector (the explicit partial solution states) contains all the necessary
information about the current partial solution, including any constraints and efficiency indicators of key resources of the problem
under concern. The second state vector (the model derived solution states) is on the projected solution states at any future point,
being estimated through the deterministic model of the problem.
In an effort to raise the generality of hyper-heuristics, the use of
both explicit state vector and model-derived state vector are considered as a distinctive algorithm design philosophy.


_3.4._ _Actions_


In the context of hyper-heuristics, the actions are, in most cases,
various heuristic rules used in practice. The actions can also be
more sophisticated model-based strategies for making multiple decisions at each step. Action set design is problem-specific, and
there is no generic design suitable for all problems. As a general
guideline, an action set design should satisfy both reachability and
interpretability. Reachability means that there should exist at least
one combination of these heuristics through which the optimal solution can be reached. Meanwhile, interpretability requires a certain level of convenience to interpret and evaluate these heuristics.
In the following two sections, we demonstrate how the proposed hyper-heuristic method can be used to solve two differ


421


_Y._ _Zhang,_ _R._ _Bai,_ _R._ _Qu_ _et_ _al._ _European_ _Journal_ _of_ _Operational_ _Research_ _300_ _(2022)_ _418–427_


**Fig.** **3.** Typical layout of a container terminal.


ent combinatorial optimisation problems with challenging uncertainties and evaluate its performance in comparison with existing
methods.


**4.** **Application** **to** **online** **container** **terminal** **truck** **routing**
**problem**


_4.1._ _Online_ _container_ _truck_ _routing_ _problem_ _description_



The real-world problem considered in this paper is a container
truck routing problem faced by one of the largest international
ports. At the same time, it is also a type of problem faced by
many maritime ports, airports and logistics centres. The problem
is concerned with the optimal truck assignments for a list of predefined container transportation between the vessels (seaside) and
the container yard in a container terminal (see a typical layout
in Fig. 3). On each day, the terminal is visited by several vessels
with a list of predefined containers to be loaded and/or unloaded.
Cranes are required to handle the operations at both the seaside
(ship cranes) and the yard area (yard cranes). The yard area consists of a number of yard blocks, each of which with a unique yard
block ID (e.g. A1–A6 in Fig. 3), and is equipped with a single yard
crane. A fleet of homogeneous trucks, based at the depot initially,
transport containers between ship cranes and yard blocks. In this
problem, the depot, each ship crane and each yard block is represented as a node.
As the most valuable resources in a container terminal, ship
cranes are often considered as the primary focuses in operations
optimisation. Therefore, in this study, the objective is set to be the
minimisation of the total ship crane waiting time between two
consecutive operations, which is mostly caused by late truck arrivals.
Different from other container truck routing problem studies, the problem considered in this study is modelled as an online problem to realistically formulate the uncertainties of the
crane operation times (i.e. loading/unloading time) caused by the
complexities in container stacking requirements, operator proficiency, weather conditions and differences among cranes. Meanwhile, since each crane can only handle one operation at any time,
it is extremely challenging if ever possible to deal with the truck
queues at both ship and yard cranes with deterministic problem
formulation.


422



**Fig.** **4.** The flow of handling a task.


Each time when a vessel arrives, high-level decisions are made
in terms of the assignments of the berth, the ship cranes and
the yard blocks for this vessel. For practicality, these decisions are
made separately from the truck routing problem concerned in this
paper. Additionally, for each assigned ship crane, a load balance
planner is used to generate a _work_ _queue_ that specifies the operation sequence of the containers to be loaded and unloaded from
this vessel. Again, practical rules require that each ship crane is responsible for one type of operations only (i.e. either loading or unloading but not both). Containers are either in small size (20-inch)
or in large size (40-inch). We use _task_ to define a standard operation unit consisting of either two small containers of the same
Source-Destination pairs or one large container. Each task is then
defined by a source node (SN), a destination node (DN) and the
details of the corresponding container(s). A task is serviced by one
truck exactly. The dashed line in Fig. 3 represents a transportation
task from a ship crane CR07 to yard block D6.
The detailed events of handling a task are shown in Fig. 4. The
timings of these events for all tasks define a full truck dispatching
solution. The ship cranes strictly follow the sequences defined by
the work queues while operations at each yard crane adopt the
First-Come-First-Serve (FCFS) policy.


_4.2._ _Problem_ _formulation_


This problem is initially formulated in Chen et al. (2016). Here,
we provide a slightly different formulation.


_Y._ _Zhang,_ _R._ _Bai,_ _R._ _Qu_ _et_ _al._ _European_ _Journal_ _of_ _Operational_ _Research_ _300_ _(2022)_ _418–427_



The problem can be defined with a directed graph _G_ = _(N,_ _A)_,
where vertices _N_ = {0} [�] _N_ _[ship]_ [�] _N_ _[yard]_ are the union of the depot
(node 0), ship cranes _N_ _[ship]_, and yard cranes _N_ _[yard]_ . Set _A_ denotes
the arcs between the nodes. Let _W_ _Ql_ denote the work queue list
associated with ship crane _l_ ∈ _N_ _[ship]_ and _nl_ be the size of _W_ _Ql_ . Denote _q_ _[h]_ _l_ [be] [the] _[h]_ [th] [task] [in] [work] [queue] _[W]_ _[Q][l]_ [.] [For] [each] [task] _[i]_ [from]

some work queue, denote _t(i)_ _[ship]_ (respectively _tt(i)_, _t(i)_ _[yard]_ ) be its
operation time at the ship crane (respectively its transportation
time, and operation time at the yard crane). Denote _td(i,_ _j)_ be the
deadheading time from the destination of task _i_ to source of task _j_
when task _j_ is serviced immediately after task _i_ by the same truck.
Denote _K_ be the set of homogeneous trucks to be dispatched. Let
_Q_ = [�] { _W_ _Ql_ } be the set of all tasks in all work queues.


_4.2.1._ _Decision_ _variables_
There are two sets of decision variables. The first set is the truck
assignment decisions _x(i,_ _j)_ _[k]_, ∀ _i,_ _j_ ∈ _Q,_ _k_ ∈ _K_, which takes value 1
if task _j_ is immediately serviced after task _i_ by truck _k_, and 0
otherwise. The second variable set determines the operation start
times at the ship and yard cranes (respectively denoted as _T_ _(i)_ _[ship]_,
_T_ _(i)_ _[yard]_ ) for each task _i_ ∈ _Q_ . For the ease of mathematical formulation, we also use _T_ _(i)_ _[SN]_ and _T_ _(i)_ _[DN]_ to denote the operation start
time of task _i_ at the source and destination nodes, respectively.
Similarly, we use _t(i)_ _[SN]_ and _t(i)_ _[DN]_ to stand for the service times
for task _i_ at the source and destination nodes, respectively.


_4.2.2._ _Objective_ _function_
The objective of the problem is to minimise the aggregated ship
crane waiting times between two consecutive tasks, which equals
to the time difference between the operation start time of the current task and the completion time of the previous task, expressed
as shown in (4).



_n_ - _l_ −1




 min




[ _T_ _(q_ _[h]_ _l_ [+][1] _[)][ship]_ [−] _[T]_ _[(][q][h]_ _l_ _[)][ship]_ [−] _[t][(][q][h]_ _l_ _[)][ship]_ []] (4)
_h_ =1



_l_ ∈ _N_ _[ship]_



_4.2.3._ _Constraints_
The following constraints need to be satisfied to ensure the feasibility of the solution. Constraint (5) ensure that all tasks are serviced exactly in the order specified by the work queues. Constraint
(6) ensures the feasible operation start times for any two consecutive tasks assigned to the same truck. Constraints (7) and (8) make
sure that each task is serviced by exactly one truck. Another important constraint is the FCFS policy at each yard crane which has
an effect on a truck’s waiting time before the assigned tasks can
be started. Its mathematical representation is not included due to
its expression complexities and page limitation.


_T_ _(q_ _[h]_ _l_ [+][1] _[)][ship]_ [≥] _[T]_ _[(][q][h]_ _l_ _[)][ship]_ [+] _[t][(][q][h]_ _l_ _[)][ship][,]_ [∀] _[q][h]_ _l_ [∈] _[W]_ _[Q][l][,]_ _[l]_ [∈] _[N][ship]_ (5)


_T_ _( j)_ _[SN]_ ≥ _(T_ _(i)_ _[DN]_ + _t(i)_ _[DN]_ + _td(i,_ _j))x(i,_ _j)_ _[k]_ ∀ _i,_ _j_ ∈ _Q,_ _k_ ∈ _K_ (6)











_x(i,_ _j)_ _[k]_ = 1 ∀ _i_ ∈ _Q_ (7)

_k_ ∈ _K_



_j_ ∈ _Q_









_x(i,_ _j)_ _[k]_ = 1 ∀ _j_ ∈ _Q_ (8)

_k_ ∈ _K_



_i_ ∈ _Q_



Note that in this study, the crane operation times ( _t(i)_ _[ship]_,
_t(i)_ _[yard]_ ) are subject to uncertainties and are assumed to be revealed in an online fashion. We model this online combinatorial
problem as a sequential decision problem (i.e., multi-stage). Once
a decision is made, no change can be made later on to reverse the


423



decision. This way, the problem can then be solved within a reinforcement learning framework. Nevertheless, the DRL agent is exposed to the nature of the uncertainties through training instances;
while its performance is evaluated on a set of independently generated instances.


_4.3._ _Implementation_ _details_


This section describes the implementation details of the proposed DDQN based hyper-heuristic method, including the state design, the action sets (low-level heuristics) and the reward design.


_4.3.1._ _State_ _design_
Unlike in most reinforcement learning problems, such as playing Atari games, where screen images can be directly fed as a state
to the neural network, a real-world problem like the container
truck routing in a port terminal requires professional advice on selecting features to encode a state. The following features have been
seen as vital by our collaborators in the port when dispatching a
task, being subsequently selected to be part of the state.


_•_ The remaining number of tasks each ship crane needs to finish
(i.e. the length of the work queue),

_•_ The distance between the current position of the truck to be
dispatched and the source nodes of the first tasks of every work
queue,

_•_ The predicted number of trucks to serve every ship crane, including trucks already dispatched and to be dispatched in the
near future (i.e. the supply),

_•_ The predicted number of tasks to be finished at every ship
crane in the near future (i.e. demand). In this application, this
is set to 10 minutes.


The first two features are explicit partial solution states which
can be directly acquired from the problem environment, denoted
as [ _rn_ 1 _,_ _rn_ 2 _,_ _._ _._ _._ _,_ _rnm_ ] and [ _d_ 1 _,_ _d_ 2 _,_ _._ _._ _._ _,_ _dm_ ], respectively. The latter
two need to be estimated by the problem mathematical model.
For example, a moving-average method is applied to predict the
service time of a task in a ship crane. Algorithm 2 in Appendix
A denotes how the problem mathematical model calculates the
predicted number of tasks to be finished at every ship crane in
10 minutes.
The last two state vectors can be expressed as

[ _wn_ 1 _,_ _wn_ 2 _,_ _._ _._ _._ _,_ _wnm_ ] and [ _pn_ 1 _,_ _pn_ 2 _,_ _._ _._ _._ _,_ _pnm_ ], respectively. Thus,
the final state is a concatenation of all the four items, represented
as: [[ _rn_ 1 _,_ _rn_ 2 _,_ _. . ._ _,_ _rnm_ ] _,_ [[ _d_ 1 _,_ _d_ 2 _,_ _._ _._ _._ _,_ _dm_ ] _,_ [ _wn_ 1 _,_ _wn_ 2 _,_ _._ _._ _._ _,_ _wnm_ ] _,_

[ _pn_ 1 _,_ _pn_ 2 _,_ _._ _._ _._ _,_ _pnm_ ]].


_4.3.2._ _Actions_
In this proposed framework, there are two level of actions. An
agent performs an _agent_ _action_ to select a low-level heuristic, i.e.
_Aagent_ = { _agent_ _action_ _i_ | _i_ ∈ _(_ 0 _,_ 1 _,_ 2 _,_ _._ _._ _._ _,_ 9 _)_ }. Once agent action _i_ is
selected, the corresponding _heuristic_ _action_ is taken to assign a task
to the current truck under consideration.
The set of low-level heuristics used in this paper is inspired by
the manual heuristic in Chen et al. (2016). Essentially, these heuristics are different rules to sort the active work queues ( _W_ _Ql,_ _l_ ∈
_N_ _[ship]_ ) attached to ship cranes. This is because the task sequencing
in each work queue is already decided separately and is not part
of optimisation in our problem. The low-level heuristics considered three factors: distances between the current action-triggering
truck and the candidate work queues, the degree of unbalance of
work queues, and the degree of urgency of all candidate work
queues. For each factor, three possible thresholds are assigned to
decide whether the corresponding factor becomes active in work
queue sorting. This leads to 3 × 3 = 9 low-level heuristics. Finally,
Chen et al. (2016)’s manual heuristic is also included as a low-level


_Y._ _Zhang,_ _R._ _Bai,_ _R._ _Qu_ _et_ _al._ _European_ _Journal_ _of_ _Operational_ _Research_ _300_ _(2022)_ _418–427_



heuristic in our DRL-HH. See Appendix B for more details of the
low-level heuristics. This leads to a total of 10 low-level heuristics.


_4.3.3._ _Rewards_
Typically an immediate reward _rt_ is a scalar value that the
agent receives after taking the chosen action in the environment
at each time step _t_ . Since the objective of this problem is to minimise the aggregated ship crane waiting times between two consecutive tasks, when a task _q_ _[h]_ [+][1] is selected, we set the reward as
_l_
the time gap (i.e. crane idle time) between the completion time of
the previous task _q_ _[h]_ _l_ [and] [the] [start] [time] [of] [the] [current] [task] _[q][h]_ _l_ [+][1],

i.e. _T_ _(q_ _[h]_ _l_ [+][1] _[)][ship]_ [−] _[T]_ _[(][q][h]_ _l_ _[)][ship]_ [−] _[t][(][q][h]_ _l_ _[)][ship]_ [.] [Since] [this] [problem] [is] [a] [min-]
imisation problem and DRL normally aims to maximise the accumulative reward, we chose the negative time gap as the reward in
order to minimise the accumulated ship crane idle time between
tasks.
Note that, when computing the reward of a specific task assignment, the corresponding ship crane waiting time cannot be computed immediately after the assignment because the previous task
may not have been completed yet, or the current truck has not
reached the assigned ship crane. Therefore, the evaluation is done
episodically. That is, at each episode, when all the tasks in a given
data set are dispatched and finished (i.e. an episode is finished),
the rewards are calculated retrospectively.


_4.4._ _Experiment_ _design_ _and_ _results_ _analysis_


To evaluate the performance and robustness of the proposed
DRL hyper- heuristic (DRL-HH), several benchmark problem instances of different sizes are extracted from real-life data to serve
as a test bed. The methods to solve this kind of online combinatorial optimisation problem are scarce due to its huge solution space
and a relatively low response time required. The manually crafted
heuristic (Chen et al., 2016) showed to generate solutions that are
superior to those used in practice, and is used as a baseline for our
proposed method. The proposed DRL-HH is also compared with
the data-driven genetic programming hyper-heuristic (Data-driven
GP) Chen et al. (2020).


_4.4.1._ _Datasets_
The experiment datasets were drawn from real-world problems
with a small adaptation. Two datasets (small_basic and big_basic)
were used in the initial experiments, both contain 120 problem
instances. In the scalability test experiments, 10 further datasets
were generated based on small_basic and big_basic. In all problem
instances, the crane operation times ( _t(i)_ _[ship]_ _,_ _t(i)_ _[yard]_ ) are drawn
from four different Gaussian distributions to sufficiently simulate
the complexity of the real-life data (see Appendix C.1 for more details). Their real values are revealed dynamically over time.


_4.4.2._ _Experiment_ _settings_
We adopted a four-layer DRL similar to the one used in Chen
& Tian (2019), where a DRL was used to solve some deterministic
combinatorial optimisation problems. Due to space limitations, the
details of our DRL settings are given in Appendix C.2.


_4.4.3._ _Experimental_ _results_
There are stochastic components in our algorithm, such as the
initial values of the weights, biases in the neural networks and the
randomness in the low-level heuristics. We therefore repeated the
experiment (i.e. both training and testing) 10 times. During the
training, our DRL agent converges after about 2000 episodes on
both small_basic and big_basic. In each episode, the entire training
dataset (20 problem instances) was used to train the agents. The
average total crane waiting time of the present DRL-HH for both


424



datasets during the training process is shown in two figures in Appendix D.1.
Once the training is completed, the resulting DRL-HH method is
evaluated for its performance, scalability and relative performance
against the simple DRL method. In each test experiment, the algorithms were used to solve 100 test instances. For each test instance, the algorithms were run 200 times with different random
seeds.
The average total crane waiting time over 100 problem instances was adopted as the performance indicator. Meanwhile,
rank tests were conducted to fully compare the performance of the
proposed method and the other two benchmark methods, results
shown in Table 1. It was shown that combining multiple heuristics is beneficial than applying any of them alone (Burke et al.,
2007; Pillay & Qu, 2021). The comparison of our proposed DRLHH against each individual heuristic is presented in Appendix D.2,
and results confirm this finding in the literature.
It can be seen that the data-driven GP is marginally better
than the manual heuristic, while DRL-HH performs the best among
the three methods. This was expected because our DRL-HH can
balance the long-term and short-term rewards with DQN training. Both the manual heuristic and data-driven GP may suffer
from solving extreme test instances. On average, DRL-HH obtains
a significant improvement when compared with manual heuristic
(7.37% for small_basic and 8.20% for big_basic).
_Scalability_ _evaluations_ In real-world scenarios, the generality of
the trained model is of high importance. In the problem faced by
Ningbo Port, the number of tasks in the datasets and the number
of work queues (which is equal to the number of ship cranes) may
change at some point. Therefore, two groups of experiments (Scalability Experiment 1 and Scalability Experiment 2) were conducted
to evaluate the generality of the trained DRL-HH agent.
In Scalability Experiment 1, DRL-HH was tested on ‘small48T’,
‘small96T’, ‘big144T’ and ‘big288T’, where the number of tasks in a
problem instance is different from those of the training instances
(See Appendix C.1 for details). The experimental results are presented in Table 2.
In Scalability Experiment 2, DRL-HH was tested on ‘small3WQ’,
‘small2 WQ’, ‘big5WQ’, where the number of work queues involved is decreased (See Appendix C.1 for details), results shown
in Table 3. Note that here we did not test our trained algorithm
on instances with increased ship cranes due to two reasons: First,
we foresee poor results because the DRL-HH agent knows nothing about these newly added ship cranes during the training. Second, for a real-world port, the maximum number of ship cranes
stays unchanged. If we train our DRL-HH with the maximum ship
cranes, the model would still perform well for instances with fewer
ship cranes because it has seen all the information.
In these two groups of scalability experiments, the changes of
both the number of tasks in the datasets and the number of work
queues rendered significantly in the problem structure. Hence, obtaining a model that could perform well in different situations was
considerably difficult. The results of manual heuristic and datadriven GP are still relatively close: data-driven GP is slightly better than manual heuristic (except for the task set with 48 tasks in
Table 2).
Notably, in Tables 2 and 3, the results of data-driven GP in every row were obtained by rules developed separately for different
situations. Using the results obtained by a single rule in the datadriven GP would be unfair. Yet, in real world, when faced with
continuously changing scenarios, data-driven GP would experience
difficulties in automatically making a choice to call different rules.
In DRL-HH, since the deep neural network can learn non-linear
relationships and encode kinds of knowledge in different problem instances, good performance with only one well-trained model
could be achieved. DRL-HH is better than manual heuristic in all


_Y._ _Zhang,_ _R._ _Bai,_ _R._ _Qu_ _et_ _al._ _European_ _Journal_ _of_ _Operational_ _Research_ _300_ _(2022)_ _418–427_


**Table** **1**
The Performance of DRL-HH in comparison with manual heuristic and a data-driven GP method. In _x/y_ : _x_
is the average waiting time, and _y_ is the average rank from the rank test.


Average total crane waiting time (s) / Average rank Imp% of DRL-HH
Datasets over manual
Manual heuristic (baseline) Data-driven GP DRL-HH heuristic


Small_Basic 1808.02/2.85 1786.32/2.04 1674.40/1.11 7.39%
Big_Basic 6392.29/2.87 6333.48/2.03 5868.30/1.10 8.20%


**Table** **2**
Results of scalability experiment 1. In _x/y_, _x_ is the average waiting time, and _y_ is the average rank from
the rank test.


Average total crane waiting time (s) / Average rank Imp% of DRL-HH
Datasets over manual
Manual heuristic (baseline) Data-driven GP DRL-HH heuristic


Small48T 1017.06/2.85 1037.40/2.06 971.44/1.09 4.52%
Small96T 2207.33/2.91 2192.76/1.95 2130.67/1.14 3.51%
Big144T 3601.26/2.83 3580.37/2.12 3450.30/1.05 4.20%
Big288T 8489.78/2.90 8423.56/2.02 8175.28/1.08 3.71%


**Table** **3**
Results of scalability experiment 2. In _x/y_, _x_ is the average waiting time, and _y_ is the average rank from
the rank test.


Average total crane waiting time (s) / Average rank Imp% of DRL-HH
Datasets over manual
Manual heuristic (baseline) Data-driven GP DRL-HH heuristic


Small3WQ 1360.45/2.93 1354.74/1.99 1313.10/1.08 3.42%
Small2WQ 895.43/2.87 889.79/1.99 869.92/1.14 2.93%
Big5WQ 4850.87/2.89 4805.76/2.04 4646.37/1.07 4.22%
Big4WQ 3920.40/2.79 3898.45/2.05 3774.21/1.16 3.76%
Big3WQ 2620.34/2.81 2612.49/2.04 2533.54/1.15 3.32%
Big2WQ 1745.55/2.89 1735.95/1.97 1692.46/1.14 3.08%



**Table** **4**
Average training time and number of training episodes for two methods to
achieve the same results .



Task sets Methods



Average training
time (minute) & Average num of
StdDev episodes & StdDev



Small_basic DRL-HH 87 (6) 1933 (61)
Simple DRL 324 (11) 7740 (209)
Big_Basic DRL-HH 397 (17) 1992 (75)
Simple DRL 1640 (45) 8864 (218)


experiments (from 2.9% to about 4.5%), as observed from the two
tables.
_DRL_ _VS._ _DRL-HH_ Finally, the direct use of DRL to choose work
queues was tested against the DRL-HH for both the small_basic
and big_basic task sets. In the experiment where the DRL directly
chooses work queue, all the hyper-parameters are the same as
those in the DRL-HH experiments. Again, the simple DRL was run
10 times with the random seeds. In the simple DRL method, the
agent may choose an action that violates some of the constraints
(e.g. going to work queue without tasks). If this happens, the agent
heavily to make the decision extremely unpopular, namely giving
a big negative reward. In this experiments, we give the agent rewards of -10000 and -20000 which are far smaller than a normal
reward signal, for small_basic and big_basic datasets, respectively.
However, although the punishment strategy addresses this
problem to a certain extent, the training time increases greatly, as
shown in Table 4 for simple DRL to obtain similar results to those
by DRL-HH. The simple DRL consumes about 3.7 (for small_basic)
and 4.1 (for big_basic) times training time, respectively, compared
with that of DRL-HH. The simple DRL also requires about 4 and 4.5
times training iterations compared with that of DRL-HH. Using DRL
directly suffers from considerably slow convergence.


425



It is worth noting that although the training of DRL-HH takes
a lot of time, it is very fast in testing/execution once trained. The
average execution time for a trained DRL-HH agent to solve a small
instance (containing 72 tasks) is 0.165 seconds and 0.575 seconds
for a large instance (with 216 tasks).


_4.5._ _States_ _spectral_ _analysis_


Models trained with data-driven methods are often concerned
with their interpretability. Complex models acquired through traditional machine learning methods, such as deep neural networks,
are difficult to comprehend. The low-level heuristics in the present
hyper-heuristics were manually designed. The proposed DRL hyperheuristic thus provides a certain level of interpretability. To further
understand the trained model, in the test phase, spectrum analysis
of the states was conducted to identify possible patterns between
the states and the corresponding actions.
We collected 12,000 states and removed duplication. The remaining 11,035 states were then partitioned into nine groups pursuant to their corresponding actions executed, as shown in Fig. 5.
It can be observed that out of the total 11,035 states, actions 0, 1,
3 and 6 are among the most frequently used heuristics.
Spectrum analysis was conducted on each group of these classified states. Taking action 0 in Fig. 6 as an example, where the
_x_ -axis represents the state elements discussed in Section 4.3.1. The
urgent degree (see Appendix B for details) was employed to replace the supply and the demand as it is a more intuitive indicator
and more conducive to discover certain patterns. The range of values of different state elements was normalised to [−50, 50]. For
the elements of work queue length, the closer it was to −50, the
smaller the work queue length was. When it was equal to −50,
the work queue was empty. The colour represents the frequency
of the elements that fall into a specific area exhibited as the colour
bar on the right side of the figure. In Fig. 6 and Figure D.3 in Ap

_Y._ _Zhang,_ _R._ _Bai,_ _R._ _Qu_ _et_ _al._ _European_ _Journal_ _of_ _Operational_ _Research_ _300_ _(2022)_ _418–427_


lines. The first is best-fit with a fixed strategy, which only selects the “leftmost” placement throughout the whole packing process. The second is best-fit with a stochastic strategy, which selects different placements at different decision points. At the same
time, we also included in the comparison the grouping super harmonic algorithm (GSHA) proposed in Han, Iwama, Ye, & Zhang
(2016), which is commonly considered as a state of the art online
method. More details of this problem and the experiments are reported in Appendix E.

Table 5 displays the average results (i.e. heights) achieved by
the proposed DRL-HH in comparison with the other three methods. An observation can be made that the present DRL-HH method
outperformed the best-fit algorithm on average, and ranked the
highest (1.22). The recently proposed GSHA method, to our surprise, did not perform well. This is probably due to its lack of internal learning mechanism to the given problem data set. The results
demonstrate the generality of the proposed method across different types of online optimisation problems.


**Fig.** **5.** The distribution of the actions corresponding to the 11,035 states.


**6.** **Conclusions** **and** **future** **work**



pendix D.3, when focusing on the work queue length elements, it
can be observed that DRL-HH tended to choose actions 0 and 3
at the middle and late stages of an episode (work queue length in

[−50, 2] and [−50, 26], respectively). However, at the early stage
of an episode, actions 0 and 3 were chosen with a low frequency.
When attention shifts to the urgent degree, Figures D.3 and D.4 in
Appendix D.3 indicate that compared with action 4, action 3 was
more likely to be chosen when the urgent degree was high.
The results reveal that certain patterns exist between the states
and the corresponding actions. In real life, people instinctively reject decisions that are hard to understand. This presents a challenge for the use of DRL in an industrial environment like Ningbo
Port. DRL-HH can fully utilise the powerful learning and exploration of DRL, but more importantly, can also provide explainable
solutions to some extent, allowing decision-makers in industry to
understand and thus accept the suggestions provided by the algorithm more easily.


**5.** **Application** **of** **DRL-HH** **to** **online** **2-D** **strip** **packing** **problem**


The proposed DRL-HH was further evaluated on a classic online
2-D strip packing problem. Two online variants of the best-fit
algorithm (Burke, Kendall, & Whitwell, 2004) were used as base


Real-world combinatorial optimisation problems are frequently
featured with uncertainties. This poses a major challenge to traditional optimisation algorithms. In this paper, we propose a deep
reinforcement learning (DRL) based hyper-heuristic framework. For
the first time, DRL was introduced into a constructive hyperheuristics framework to address the challenging online combinatorial optimisation problems. Experimental results highlight several
advantages with this new framework. Firstly, it shows better performance compared with the existing state of the art methods on
both a real-world truck routing problem and a 2D strip packing
problem with uncertainties. Secondly, it shows a good scalability
when the problem sizes change. Thirdly, compared with traditional
DRL methods, it holds better convergence. Finally, the proposed
approach showed to improve the interpretability of the solutions,
thus is more acceptable in real-life applications.
In our future work, first, we can explore a better neural network
structure with variable input dimensions to adapt to the changing
number of ship cranes or other relevant elements. Second, it would
be interesting to investigate whether the ‘patterns’ or ‘knowledge’
obtained from the state spectral analysis can be fed to the learning agent at the early training process to improve the training efficiency.



**Fig.** **6.** The spectrogram of states for action 0.


426


_Y._ _Zhang,_ _R._ _Bai,_ _R._ _Qu_ _et_ _al._ _European_ _Journal_ _of_ _Operational_ _Research_ _300_ _(2022)_ _418–427_


**Table** **5**
The average heights achieved by four approaches and their scores in the rank test for the online 2-D strip packing problem.



Average heights of packings / Rank score
Datasets



Best-fit with fixed strategy
Burke et al. (2004)



Best-fit with
Stochastic strategy GSHA Han et al.
Burke et al. (2004) DRL-HH (2016)



Training 271.8 272.2 269.3 374.3
Test 271.7/2.07 271.9/2.29 269.1/1.22 372.8/3.78



**Acknowledgement**


This work is supported by the National Natural Science Foundation of China (grant number 72071116) and the Ningbo Science
and Technology Bureau (grant numbers 2019B10026, 2017D10034).


**Supplementary** **material**


Supplementary material associated with this article can be
found, in the online version, at [10.1016/j.ejor.2021.10.032](https://doi.org/10.1016/j.ejor.2021.10.032)


**References**


[Ahmed,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0001) L., [Mumford,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0001) C., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0001) [Kheiri,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0001) A. (2019). Solving urban transit route design problem using selection [hyper-heuristics.](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0001) _European_ _Journal_ _of_ _Operational_ _Research,_
_274_ (2), 545–559.
Bai, R., Blazewicz, J., Burke, E. K., Kendall, G., & McCollum, B. (2012). A simulated annealing hyper-heuristic methodology for flexible decision support. _4OR-_
_A_ _Quarterly_ _Journal_ _of_ _Operations_ _[Research,](https://doi.org/10.1007/s10288-011-0182-8)_ _10_ (1), 43–66. https://doi.org/10.1007/
s10288-011-0182-8.
[Burke,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0003) E. K., [Gendreau,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0003) M., [Hyde,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0003) M., [Kendall,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0003) G., [Ochoa,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0003) G., [Özcan,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0003) E., et al. (2013).
Hyper-heuristics: A survey of the [state](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0003) of the art. _Journal_ _of_ _the_ _Operational_ _Re-_
_search_ _Society,_ _64_ (12), 1695–1724.
[Burke,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0004) E. K., [Hyde,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0004) M., [Kendall,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0004) G., [Ochoa,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0004) G., [Özcan,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0004) E., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0004) [Woodward,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0004) J. R. (2010).
A classification of hyper-heuristic [approaches.](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0004) In _Handbook_ _of_ _metaheuristics_
(pp. 449–468). Springer.
[Burke,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0005) E. K., [Kendall,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0005) G., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0005) [Whitwell,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0005) G. (2004). A new placement heuristic for the
orthogonal stock-cutting problem. _Operations_ _Research,_ _52_ (4), 655–671.
[Burke,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0006) E. K., [McCollum,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0006) B., [Meisels,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0006) A., [Petrovic,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0006) S., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0006) [Qu,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0006) R. (2007). A graph-based
hyper-heuristic for educational [timetabling](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0006) problems. _European_ _Journal_ _of_ _Oper-_
_ational_ _Research,_ _176_ (1), 177–192.
[Chen,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0007) J., [Bai,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0007) R., [Dong,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0007) H., [Qu,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0007) R., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0007) [Kendall,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0007) G. (2016). A dynamic truck dispatching
problem in marine container terminal. _2016_ _IEEE_ _symposiums_ _on_ _computational_
_intelligence_ _in_ _scheduling_ _and_ _network_ _design_ _(CISND2017),_ _6–9_ _December,_ _2016_
_Athens,_ _Greece_ .
[Chen,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0008) X., [Bai,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0008) R., [Qu,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0008) R., [Dong,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0008) H., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0008) [Chen,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0008) J. (2020). A data-driven genetic programming heuristic for real-world [dynamic](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0008) seaport container terminal truck
dispatching. In _2020_ _IEEE_ _congress_ _on_ _evolutionary_ _computation_ _(CEC)_ (pp. 1–8).
[Chen,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0009) X., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0009) [Tian,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0009) Y. (2019). Learning to perform local rewriting for combinatorial
optimization. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems,_ _32_, 6281–6292.
[Chen,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0010) X., [Zhang,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0010) H., [Wu,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0010) C., [Mao,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0010) S., Ji, Y., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0010) [Bennis,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0010) M. (2018). Optimized computation offloading performance in virtual edge computing systems via deep
reinforcement learning. _IEEE_ _Internet_ _of_ _Things_ _Journal,_ _6_ (3), 4005–4018.
[Choong,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0011) S. S., [Wong,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0011) L.-P., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0011) Lim, C. P. (2019). An artificial bee colony algorithm
with a modified choice function for [the](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0011) traveling salesman problem. _Swarm_ _and_
_Evolutionary_ _Computation,_ _44_, 622–635.
[Cowling,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0012) P., [Kendall,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0012) G., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0012) [Soubeiga,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0012) E. (2000). A hyperheuristic approach to scheduling a sales summit. In _International_ _[conference](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0012)_ _on_ _the_ _practice_ _and_ _theory_ _of_ _au-_
_tomated_ _timetabling_ (pp. 176–190). Springer.


427



[Drake,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0013) J. H., [Kheiri,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0013) A., [Özcan,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0013) E., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0013) [Burke,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0013) E. K. (2020). Recent advances in selection
hyper-heuristics. _European_ _Journal_ _of_ _Operational_ _Research,_ _285_ (2), 405–428.
[Fisher,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0014) R. D., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0014) [Thompson,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0014) G. L. (1963). Probabilistic learning combinations of local
job-shop scheduling rules. In _Industrial_ _scheduling_ (pp. 225–251).
[Gomez,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0015) J. C., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0015) [Terashima-Marín,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0015) H. (2018). Evolutionary hyper-heuristics for tackling bi-objective 2D bin packing [problems.](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0015) _Genetic_ _Programming_ _and_ _Evolvable_
_Machines,_ _19_ (1–2), 151–181.
[Han,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0016) X., [Iwama,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0016) K., Ye, D., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0016) [Zhang,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0016) G. (2016). Approximate strip packing: Revisited.
_Information_ _and_ _Computation,_ _249_, 110–120.
[Kheiri,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0017) A., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0017) [Keedwell,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0017) E. (2017). A hidden Markov model approach to the problem of heuristic selection in [hyper-heuristics](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0017) with a case study in high school
timetabling problems. _Evolutionary_ _computation,_ _25_ (3), 473–501.
Lin, L.-J. (1993). Reinforcement learning for robots using neural networks. _Technical_
_Report_ . Carnegie-Mellon Univ Pittsburgh PA School of Computer Science.
[MacLachlan,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0019) J., [Mei,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0019) Y., [Branke,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0019) J., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0019) [Zhang,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0019) M. (2020). Genetic programming hyper-heuristics with vehicle collaboration for uncertain capacitated arc routing problems. _Evolutionary_ _Computation,_ _28_ (4), 563–593 MIT Press One Rogers
Street, Cambridge, MA 02142-1209, USA.
[Mei,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0020) Y., [Tang,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0020) K., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0020) [Yao,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0020) X. (2010). Capacitated arc routing problem in uncertain
environments. In _IEEE_ _congress_ _on_ _evolutionary_ _computation_ (pp. 1–8). IEEE.
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., & Wierstra, D. et al.
(2013). Playing Atari with deep [reinforcement](http://arxiv.org/abs/1312.5602) learning. arXiv preprint arXiv:
1312.5602.
[Mnih,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0022) V., [Kavukcuoglu,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0022) K., [Silver,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0022) D., Rusu, A. A., [Veness,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0022) J., [Bellemare,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0022) M. G.,
et al. (2015). Human-level control through deep reinforcement learning. _Nature,_
_518_ (7540), 529–533.
[Pillay,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0023) N., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0023) [Qu,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0023) R. (2018a). _Hyper-heuristics:_ _[Theory](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0023)_ _and_ _applications_ . Springer.
Pillay, N., & Qu, R. (2018b). Selection constructive hyper-heuristics, pp. 7–16.
10.1007/978-3-319-96514-7_2
[Pillay,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0025) N., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0025) [Qu,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0025) R. (2021). _Rigorous_ _performance_ _analysis_ _of_ _hyper-heuristics_ . Springer
Natural Computing Series.
[Qu,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0026) R., [Pham,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0026) N., [Bai,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0026) R., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0026) [Kendall,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0026) G. (2015). Hybridising heuristics within an estimation distribution algorithm for [examination](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0026) timetabling. _Applied_ _Intelligence,_
_42_ (4), 679–693.
[Rahimian,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0027) E., [Akartunalı,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0027) K., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0027) [Levine,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0027) J. (2017). A hybrid integer programming and
variable neighbourhood search algorithm [to](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0027) solve nurse rostering problems. _Eu-_
_ropean_ _Journal_ _of_ _Operational_ _Research,_ _258_ (2), 411–423.
[Silver,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0028) D., [Huang,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0028) A., [Maddison,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0028) C. J., [Guez,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0028) A., [Sifre,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0028) L., Van Den [Driessche,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0028) G.,
et al. (2016). Mastering the game of go with deep neural networks and tree
search. _Nature,_ _529_ (7587), 484–489.
[Soghier,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0029) A., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0029) [Qu,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0029) R. (2013). Adaptive selection of heuristics for assigning time slots
and rooms in exam timetables. _Applied_ _Intelligence,_ _39_ (2), 438–450.
[Soria-Alcaraz,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0030) J. A., [Ochoa,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0030) G., [Swan,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0030) J., [Carpio,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0030) M., [Puga,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0030) H., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0030) [Burke,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0030) E. K. (2014).
Effective learning hyper-heuristics for the [course](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0030) timetabling problem. _European_
_Journal_ _of_ _Operational_ _Research,_ _238_ (1), 77–86.
Van [Hasselt,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0031) H., [Guez,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0031) A., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0031) [Silver,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0031) D. (2016). Deep reinforcement learning with double _Q_ -learning. _Thirtieth_ _AAAI_ _conference_ _on_ _artificial_ _intelligence_ .
[Zamli,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0032) K. Z., [Alkazemi,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0032) B. Y., [&](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0032) [Kendall,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0032) G. (2016). A tabu search hyper-heuristic strategy for _t_ -way test suite generation. _Applied_ _Soft_ _Computing,_ _44_, 57–74.
[Zheng,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0033) G., [Zhang,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0033) F., [Zheng,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0033) Z., [Xiang,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0033) Y., [Yuan,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0033) N. J., [Xie,](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0033) X., et al. (2018). DRN: A
deep reinforcement learning framework [for](http://refhub.elsevier.com/S0377-2217(21)00882-1/sbref0033) news recommendation. In _Proceed-_
_ings_ _of_ _the_ _2018_ _world_ _wide_ _web_ _conference_ (pp. 167–176).


