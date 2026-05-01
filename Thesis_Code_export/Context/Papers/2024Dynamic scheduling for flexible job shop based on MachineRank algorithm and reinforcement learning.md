## **OPEN**



[www.nature.com/scientificreports](http://www.nature.com/scientificreports)

# **Dynamic scheduling for flexible** **job shop based on MachineRank** **algorithm and reinforcement** **learning**

**Fujie Ren & Haibin Liu** []


**This paper investigates the Dynamic Flexible Job Shop Scheduling Problem (DFJSP), which is based**
**on new job insertion, machine breakdowns, changes in processing time, and considering the state**
**of Automated Guided Vehicles (AGVs). The objective is to minimize the maximum completion time**
**and improve on-time completion rates. To address the continuous production status and learn**
**the most suitable actions (scheduling rules) at each rescheduling point, a Dueling Double Deep Q**
**Network (D3QN) is developed to solve this problem. To improve the quality of the model solutions,**
**a MachineRank algorithm (MR) is proposed, and based on the MR algorithm, seven composite**
**scheduling rules are introduced. These rules aim to select and execute the optimal operation each**
**time an operation is completed or a new disturbance occurs. Additionally, eight general state features**
**are proposed to represent the scheduling status at the rescheduling point. By using continuous**
**state features as the input to the D3QN, state-action values (Q-values) for each scheduling rule can**
**be obtained. Numerical experiments were conducted on a large number of instances with different**
**production configurations, and the results demonstrated the superiority and generality of the D3QN**
**compared to various composite rules, other advanced scheduling rules, and standard Q-learning**
**agents. The effectiveness and rationality of the dynamic scheduling trigger rules were also validated.**



The FJSP which is an extension of the classical JSP [1], firstly introduced by Brucker and Schil [2], has been intensively
studied over the past decades [3][,][4] . It has been demonstrated that the classical Job Shop Scheduling Problem is an
NP-hard problem. However, the FJSP builds upon the JSP by assuming that each operation can be allocated to
one or multiple available machines, making the FJSP even more intricate. So far, most of the existing methods to
solve FJSP have assumed a static scheduling environment with known job-shop scheduling information, that is,
relying on a deterministic pre-scheduling scheme in the scheduling process. In complex and diverse scheduling
systems, it is inevitable to consider various dynamic events [5][,][6] such as emergency insertion of job, processing
machine malfunction, changes in processing times, uncertain AGV transportation times, and so on. These
dynamic events lead to the pre-scheduling scheme in a static environment not being able to proceed as planned,
resulting in a significant decrease in production efficiency. Therefore, researching scheduling methods for the
DFJSP to dynamically handle uncertain events [7] is of paramount significance. The flexible job-shop dynamic
scheduling problem can optimize multiple performance indicators at the same time, and consider the selection
of processing machines and AGV transportation time when solving the processing sequence problem of the
working procedure, which is more close to the actual production workshop environment, so it has been widely
concerned by many scholars.

Dynamic scheduling, an attractive research field for both academia and industry, has been extensively studied
over the past few decades as an effective solution for workshop dynamic scheduling problems. Among the most
widely applied approaches are scheduling rules, meta-heuristic algorithms [8], and learning-based methods.
Scheduling rules can react immediately to dynamic events, achieving optimal time efficiency. However, they
cannot attain globally optimal results. Meta-heuristic algorithms decompose dynamic scheduling problems
into a series of static sub-problems, which are then solved using intelligent optimization algorithms such as
Genetic Algorithms (GA) [9] and Particle Swarm Optimization (PSO) [10][,][11] . Wen et al. [12] proposed an adjustment
method based on job classification to improve the stability of rescheduling for dynamic scheduling problems
with machine failures, using Genetic Algorithms and Variable Neighborhood Search algorithms to solve them.
Although these methods can yield high-quality solutions, they require substantial time to re-search for new


College of Mechanical and Energy Engineering, Beijing University of Technology, Beijing 100124, China. [] email:
liuhb@bjut.edu.cn


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)



scheduling schemes, making it challenging to meet the real-time scheduling requirements of DFJSP. Learningbased methods can be categorized into supervised learning and p Reinforcement Learnin (RL) according to the
learning paradigm. The former requires the optimal or near-optimal solutions of the problem as data labels for
the training dataset. However, obtaining data labels is often computationally complex. In contrast, the latter is
not constrained by data labels. Furthermore, Deep Reinforcement Learning (DRL) [13] has advantages such as
rapid solution speed and strong generalization capability, making it a research hotspot for solving DFJSP.

The rescheduling process of FJSP can be modeled as a Markov Decision Process (MDP) [14], where the decisionmaker sequentially determines the correct actions, i.e., assigning feasible actions based on the production state
at different rescheduling points to optimize predefined long-term goals. In recent years, DRL has become an
effective approach to handling MDPs. DRL can learn optimal behavior policies in a model-free manner through
trial and error, leading to many successful applications in practical dynamic scheduling problems [15][,][16] . Despite
significant breakthroughs in DRL-based research, some current DRL methods still have deficiencies in state
representation and action space definition. In terms of state representation, Palombarini et al. [17] directly used
a complete scheduling Gantt chart as the state, which, while comprehensive, resulted in high computational
complexity and low algorithm efficiency. Liu et al. [18] employed a state representation method similar to image
RGB channels, where the three channels corresponded to the processing time matrix, a Boolean matrix indicating
whether the job was assigned to a machine, and a Boolean matrix indicating whether the job was completed.
These three matrices were used as inputs to a CNN. However, this method’s state feature extraction was singular
and did not adequately consider the state characteristics of processing machines.

In the definition of the action space, current research can be broadly divided into direct and indirect
definitions. Direct definition involves selecting a job from the queue to process [19], which neither guarantees local
optimality nor achieves global optimality. Indirect definition typically involves using reinforcement learning to
optimize the parameters of other algorithms to indirectly optimize the scheduling, such as selecting appropriate
weights to construct composite rules. Zhang et al. [20] proposed eight composite scheduling rules as actions.
However, these composite actions are combinations based on original basic actions and cannot guarantee local
optimal solutions. Lei et al. [21] proposed a multi-action DRL framework for learning two sub-policy models to
solve the DFJSP problem, using these two sub-policy models to select operation nodes and machine nodes at
each decision step. Although this strategy can effectively improve solution quality, it increases computational
complexity.

The production workshop often has uncertain disturbance factors that hinder the manufacturing process.
Nelson et al. [22] were among the earliest researchers to investigate this type of problem. They categorize
disturbances in production scheduling into sudden malfunction, changes in delivery times, and transform
dynamic processes into static scheduling intervals, known as rolling window theory. By obtaining the optimal
solution for each static interval, the overall optimal solution was synthesized, but they only analyzed and studied
the case of a single objective. Zhang et al. [23] studied dynamic scheduling issues related to delays in processing
resources and urgent orders, but they did not consider flexible job shop scheduling. Luo et al. [24] focused solely
on dynamic flexible job shop scheduling problems involving new job insertions, which is challenging to apply
in complex workshops.

To address the aforementioned deficiencies in state representation and action space definition in current
research, which lead to low model execution efficiency and suboptimal global solutions, as well as insufficient
study on dynamic events, the main research contents of this paper are as follows: 1) To improve the applicability
of the model in actual production workshops, a dynamic scheduling model for flexible job shops is constructed
by considering disturbance factors such as new job insertions, machine breakdown, changes in processing times,
as well as AGV transportation time. 2) To enhance the solution quality and consider the overall dynamic changes
in the workshop, the MR algorithm is designed to globally rank the utilization status of processing machines.
Furthermore, seven efficient and precise action rules along with a reward function are devised based on the MR
values. 3) In the reinforcement learning neural network model, the use of grouped connected neural networks,
compared to fully connected neural networks, reduces the number of weights and biases in the proposed D3QN
neural network, thereby saving computational time and resources.


**Related work**
**Flexible job shop scheduling based on deep reinforcement learning**
The agent based on RL makes extensive exploration and learning through the interaction with the environment,
and finally realizes the adaptive decision at each decision point. Due to its strong self-learning ability, RL is widely
used in DFJSP. Research based on RL has made significant breakthroughs. Shiue et al. [25] proposed a real-time
scheduling mechanism based on RL, where the Q-learning algorithm is used to train an RL agent to select the
most appropriate rule from multiple scheduling rules. Chen et al. [26] introduced a rule-driven approach for multiobjective dynamic scheduling, which uses Q-learning to learn the most suitable weights from a given discrete
value space and construct composite rules. In classical RL-based dynamic scheduling methods, the Q-learning
algorithm selects the most appropriate scheduling rule for each state by building a Q-table, which shows the
estimated maximum value of the Q-function for all discrete state-action pairs. However, real production systems
operate in a complex environment with high-dimensional continuous states, which cannot be listed in a Q-table.

To address this issue, DRL, which combines deep learning and reinforcement learning, has garnered
widespread attention. DQN, one of the most popular DRL methods, directly maps the continuous states of each
action to the maximum value of the Q-function. With the deep Q-learning algorithm, DQN can be trained to
select the most appropriate operation at each decision point, and it is now widely applied to real-time dynamic
scheduling. Luo et al. [3] trained an intelligent agent using DQN, Lin et al. [27] proposed an extended DQN method
called MDQN, and Han et al. [28] proposed a DRL scheduling framework based on disjunctive graphs. However,
DQN faces overestimation issues when calculating the target Q-values, leading to unstable training. Wu et




[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)



al. [29] proposed a multi-objective DFJSP and reduced the sum of tardiness and makespan by using a scheduling
rule based on DDQN. Lei et al. [21] formulated FJSP as an MDP and proposed a multi-pointer graph network
architecture and a multi-neighborhood policy optimization algorithm to learn scheduling policies that can
solve FJSP of different sizes. Song et al. [30] introduced a heterogeneous GNN architecture to capture the complex
relationships between operations and machines and proposed a new DRL method to learn high-quality PDRs
in an end-to-end manner. However, the stability of DFJSP was not considered in the aforementioned studies.

In DFJSP, scheduling stability is crucial for the stable operation of the production process. However, the
presence of uncertain events increases the complexity and dynamism of the environment, making it challenging
for traditional scheduling algorithms. DRL can automatically extract and adjust the nonlinear high-dimensional
features of the scheduling problem by learning and interacting with the environment. Since the action space
of DFJSP is discrete, most studies adopt DQN to optimize the scheduling process. However, DQN faces an
overestimation issue when calculating target Q-values, leading to unstable training. In contrast to DQN, D3QN
introduces two Q-networks: one for selecting actions and the other for estimating Q-values. D3QN reduces the
overestimation of Q-values in traditional DQN and improves the stability of learning.


**Dynamic scheduling**
Uncertain and unexpected events, such as machine breakdowns, new job insertions, and changes in processing
time, often occur during shop floor operations, leading to uncontrollable production processes. Dynamic
scheduling is a method of dynamically adjusting the original schedule based on real-time production status after
unexpected disruptions, which better meets the demands of complex manufacturing environments in actual
production.

There are two key issues in dynamic scheduling: “when” to reschedule and “how” to reschedule. Research on
“when” to reschedule has generally stabilized into three typical methods: event-driven rescheduling [31], periodic
rescheduling [32], and hybrid rescheduling based on events and cycles [33] . Among them, hybrid-driven scheduling
combines the advantages of both methods and is more suitable for dynamic scheduling in complex shop floor
environments.

As for “how” to reschedule, most research focuses on improving the computational efficiency of algorithms in
solving dynamic scheduling problems. For example, Park et al. [34] proposed a combined scheme based on genetic
programming integration to address dynamic job shop scheduling. However, when metaheuristic algorithms
are applied to large-scale order scheduling, the computation time is often lengthy, making it difficult to quickly
respond to new scheduling tasks. To achieve high responsiveness in dynamic scheduling, Zhang et al. [35] explored
multi-agent-based solutions, which can make decisions quickly but often result in lower-quality solutions. In
recent years, the rapid development of artificial intelligence algorithms, such as deep reinforcement learning, has
provided new ideas for improving both the response speed and solution quality in dynamic scheduling.


**Development strategy**
In this section, the proposed MachineRank algorithm and AGV scheduling are introduced. To our knowledge,
there has been no previous work focused on the algorithm research of machine ranking in material scheduling.


**MachineRank algorithm**
When selecting scheduling actions, it is necessary to evaluate the status of the processing machine and select the
best machine as the processing machine for the job to be processed. At present, most research methods directly
use machine processing time and idle time to calculate utilization rate, with few factors considered, and cannot
globally evaluate the utilization of processing machines.

Based on the particularity of flexible job shop scheduling (an operation can be processed on different machines
with different processing times), this paper proposes a real-time ranking algorithm for the comprehensive
utilization rate of processing machines based on considering multiple factors. Construct a graph structure of
processing machines and processing job information, where processing machines are represented as graph
nodes and job transitions between machines are represented as graph edges.

As shown in Figure 1(a), there are 4 processing machines, _M_ 1 to _M_ 4, and 5 jobs, each containing unif[3,5]
operations. Each edge represents the operation transfer information between the processing machines for
all jobs. For example, the edge _M_ 1-> _M_ 2 includes the information that after the first operation of job _J_ 2 is
completed on _M_ 1, the job is transferred to _M_ 2 via an AGV for the second operation (processing time _t_ 1).
Additionally, consecutive operations of the same job may be processed on the same machine. For instance, after
the 4th operation of job _J_ 4 is completed on _M_ 3, the 5th operation continues on the same machine.

The algorithm defines two types of chains in the graph structure:



(1) In-chain: Operations where a job is transferred from other machines to the current machine are defined as

in-chains.
(2) Out-chain: Operations where a job is transferred from the current machine to other machines are defined

as out-chains.As shown in Figure 1(a), for machine _M_ 1, the edges _M_ 1-> _M_ 2, _M_ 1-> _M_ 3, and _M_ 1-> _M_ 4 are
out-chains, while the edges _M_ 2-> _M_ 1, _M_ 3-> _M_ 1, and _M_ 4-> _M_ 1 are in-chains.



The idea of MachineRank is that the overall utilization rate of a processing machine is equal to the influence of all
its in-chains (number of job operations and processing time) combined with its own influence (total processing
time, total idle time, etc.). When designing the calculation model, the following factors need to be considered:


1. When calculating the overall utilization rate of a processing machine, factors such as the number of jobs

processed, total processing time, and idle time need to be considered.


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)


**Fig. 1** . Machine rank model and network model.



2. Initialize the total _MR_ value of all machines to 1, meaning the initial _MR_ value of each machine is 1/m.
3. Considering dynamic scheduling in flexible job shops, each operation can be processed on multiple ma
chines, and two consecutive operations can be processed on the same machine.
4. Real-time data update. During model training, the selection and execution of each action were performed to

update the graph network data and calculate the latest _MR_ value.In summary, the calculation model for the
_MR_ value of the _n_ th iteration is designed as shown in Equation 1.

_MR_ ( _n_ ) = _a ∗_ _M_ _∗_ _MR_ ( _n −_ 1) + (1 _−_ _a_ ) _∗_ _MV_ (1)



The parameter _a_ represents the weight of the influence of all in-chains of the processing machine, while 1 _−_ _a_
represents the weight of the machine’s own influence. The matrix _MR_ ( _n −_ 1) represents the _MR_ values of all
processing machines after _n −_ 1 iterations. The matrix _MV_ calculates the utilization rate of each processing
machine based on its own influence, which is the ratio of each machine’s processing time to the total processing
time and total idle time. The specific calculation is shown in Equation 2.



_m_ 1_ _pro_ _ _time_
_total_ _ _pro_ _ _time_ ~~+~~ _total_ _ _idle_ _ _time_
_m_ 2_ _pr_ _ _time_
_total_ _ _pro_ _ _time_ ~~+~~ _total_ _ _idle_ _ _time_
_..._
_..._
_mn_ _ _pr_ _ _time_
_total_ _ _pro_ _ _time_ ~~+~~ _total_ _ _idle_ _ _time_






_, MV_ =
















_MR_ (0) =









1
_m_ 1
_m_
_..._
_..._
1
_m_



(2)



Where, _mn_ _ _pro_ _ _time_ is the processing time of the _n_ th processing machine, _total_ _ _pro_ _ _time_ is the total
processing time, and _total_ _ _idle_ _ _time_ is the total delay time.


The matrix _M_ contains the out-chain and in-chain information of the nodes in the directed graph, that is, the
information for each directed edge, considering the total number of transferred jobs and the processing time for
each edge. The specific calculation is shown in Equation 3.













 (3)



_M_ =



_V_ (1 _,_ 1) _V_ (2 _,_ 1) _..._ _V_ ( _m,_ 1)
_V_ (1 _,_ 2) _V_ (2 _,_ 2) _..._ _V_ ( _m,_ 2)
_..._
_..._
_V_ (1 _, m_ ) _V_ (2 _, m_ ) _..._ _V_ ( _m, m_ )



In the calculation equation for matrix _M_, _V_ ( _i_, _j_ ) represents the information from the out-chain node _i_ to the inchain node _j_, including the number of transferred jobs, processing time, etc. The calculation formula for _V_ ( _i_, _j_ )
is shown below.



_d_ _Job_ _ _Num_ ( _i, j_ )
_V_ ( _i, j_ ) = ~~∑~~ _N ∗_



_Job_ _ _Num_ ( _i, j_ )
_N ∗_ + [(1] ~~∑~~ _[ −]_ _N_ _[d]_ [)] _[ ∗]_ _[Pro]_ [_] _[Time]_ [(] _[i, j]_ [)]

_n_ =1 _[Job]_ [_] _[Num]_ [(] _[i, n]_ [)] _n_ =1 _[Pro]_ [_] _[Time]_ [(] _[i, n]_ [)]



_N_ (4)

_n_ =1 _[Pro]_ [_] _[Time]_ [(] _[i, n]_ [)]



where, _Job_ _ _Num_ ( _i, j_ ) represents the total number of jobs transferred from node _i_ to node _j_ ; _Pro_ _ _Time_ ( _i, j_ )
represents the total processing time of the total jobs transferred from node _i_ to node _j_ on node _j_ ; _N_ represents the
number of degrees of node _i_ . _d_ is the weight parameter. The procedure of MachineRank is give in Algorithm 1.


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)

|job transfer (edge of graph)|Number of transfer processes|Total processing time of the process|
|---|---|---|
|_M_1->_M_2|2|41|
|_M_1->_M_3|1|32|
|_M_1->_M_4|1|28|
|_M_2->_M_1|2|50|
|_M_2->_M_3|1|28|
|_M_2->_M_4|3|63|
|_M_3->_M_1|2|52|
|_M_3->_M_2|2|44|
|_M_3->_M_3|1|25|
|_M_3->_M_4|4|70|
|_M_4->_M_1|1|23|
|_M_4->_M_2|1|29|
|_M_4->_M_3|1|33|



**Table 1** . Graph edge (job transfer) information.

|node(machine|Total number of processing operations|Idle time|
|---|---|---|
|_M_1|5|48|
|_M_2|5|36|
|_M_3|4|32|
|_M_4|8|17|



**Table 2** . Node (machine) information.


**Algorithm 1** . MachineRank algorithm


Firstly, based on the number of processing machines and other initialization graph _G_ and _MR_ (0), initialize
the parameter _N_ for the number of scheduled operations (lines 1–2). Determine whether the operation has
been scheduled and completed. If it has, return to _MR_ . Otherwise, calculate the matrix _MV_ and _M_ according to
Equations 2 and 3, and use Equation 1 to calculate _MR_ (lines 3–17).

Based on the example in Figure 1(a), the following explains the process of calculating the MR value of the
machining machine at a certain moment. There are 4 machines and 5 processing jobs, with each job having
unif[3,5] processing operations. The specific data of the workshop’s processing status at a certain time is shown
in Tables 1 and 2.


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)


Based on the relevant data of each node machine in Tables 1 and 2, assuming this
is the first iteration to calculate the MR value, the constants _a_ = 0.8 and _d_ = 0.5. Using
Equation 2 and Equation 3, the state matrices _MR_ (0), _M_, and _MV_ are calculated as follows.





_MR_ (0)=



1
~~4~~
1
~~4~~
1
~~4~~
1
~~4~~



=







50+52+23

~~518+133~~
41+44+29

~~518+133~~
32+28+25+33

~~518+133~~
28+63+70

~~518+133~~














0 _._ 25
0 _._ 25
0 _._ 25
0 _._ 25











 _, M_ =





























 _, MV_ =



=



0 _._ 1920
0 _._ 1751
0 _._ 1813
0 _._ 2473



0 0 _._ 3440 0 _._ 2472 0 _._ 3020
0 _._ 4530 0 0 _._ 2263 0 _._ 3372
0 _._ 2834 0 _._ 3067 0 _._ 1210 0 _._ 3608
0 _._ 2636 0 _._ 3493 0 _._ 4054 0







Substitute the above parameters into Equation 1 to obtain the MR value. _MR_ (1) =



0 _._ 2170
 0 _._ 2383



0 _._ 2383
0 _._ 2506
0 _._ 2531



. At this



moment, the _MR_ value of processing machine _M_ 1 is 0.2170, _M_ 2 is 0.2383, _M_ 3 is 0.2506, and _M_ 4 is 0.2531. Based
on the _MR_ values, the job with the smallest _MR_ value, _M_ 1, is selected, and the pending processing operations
are assigned to this machine.


**AGV scheduling**
This paper is based on the characteristics of the current industrial flexible job shop, and the configuration of each
machining machine corresponds to the AGV. At the same time, it is assumed that each processing machine has
two cache areas: the cache area waiting for job processing (IJB) and the completed job cache area (OJB). The task
of AGV is to transport the job in the cache area to be processed on this machine to the cache area to be processed
on the next processing machine.

This paper studies the AGV moving on a fixed orbit, considering that the AGV running speed _V_ is uniform
regardless of whether it carries the job or not. Therefore, considering the distance _D_ ( _i, i_ + 1) between the
buffer _OJBi_ and _IJBi_ +1, the job transport time or the empty transport time can be calculated by the equation
_ti,j_ = _D_ ( _i, i_ + 1) _/V_ .

The transfer of a job requires two steps of AGV, the first step is to move the job _Ji_ in the completion area of
the machine _Mi_ to the waiting cache area of the processing machine _Mi_ +1 in the next step, and the second step
is the AGV to the completion area of the machine _Mi_ . Its running time is:

_T_ = _ti,j_ + _tj,i_ (5)


This is denoted _T_ when considering the transport time of the AGV. Because the job _Ji_ can be processed when
the AGV completes the first step and the processing machine _Mi_ +1 is idle, the interval time between the two
processes of the job _Ji_ is as follows:

_Ts_ = _ti,j_ + _d_ (6)


where _d_ is the waiting time from the job arriving at machine _Mi_ +1 to the start of processing.


In this paper, it is assumed that all machines are deployed sequentially from number 1 in a runway-style
workshop, and the distance between two adjacent processing machines is equal. The speed _v_ of all AGVs is 2 _m_ / _s_,
and the distance between two processing machines is 6 _m_ .


**Problem formulation**
As an extension of classical job shop scheduling, the FJSP expands the machine flexibility of each operation
and is more complex than classical job shop scheduling. So FJSP is a strong NP-hard problem. The FJSP can be
described as:

(1)  _n_ jobs _J_ = _Ji_, 1 _≤_ _i ≤_ _n_ are indexed by i.
(2)  _m_ machines _M_ = _Mk_, 1 _≤_ _k_ _≤_ _m_ are indexed by k.
(3) Each job _Ji_ comprises a sequence of operations _O_ ={ _Oij_ }, _j_ is operation number within job _Ji_ .
(4) Each operation _Oij_ needs to be assigned to a machine in the candidate processing machine set for process
ing.
(5) The processing time of operation _Oij_ on machine _k_ is _Ci,j,k_ .To simplify the problem at hand, several pre


defined constraints should be satisfied as follows.



(1) Each job _Ji_ is independent and has a fixed processing time _ti,j,k_ .
(2) The order of priority between operations that must be performed, for example, _Oi,_ 2 must be processed after

_Oi,_ 1 has completed.
(3) All jobs _Ji_ and machines _Mk_ are available at 0 time.
(4) Each machine can handle only one operation at any one time.
(5) Each operation _Oi,j_ has at least one machinable machine.
(6) The operation being processed cannot be interrupted until the processing is completed.
(7) The difference between machine installation time and transit time operation is negligible.The notations

used for problem formulation are listed below table 3.



**New job insertions**
The DFJSP with new job insertions considered in this paper can be defined as follows. There are _n_ successively
arriving jobs _J_ = _J_ 1 _, J_ 2 _, ..., Jn_ to be processed on _m_ machines _M_ = _M_ 1 _, M_ 2 _, ..., Mm_ . Each job _Ji_ consists of


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)

|Parameter|Describe|Parameter|Describe|
|---|---|---|---|
|_n_|Total number of jobs|_m_|Total number of machines|
|_Ji_ and_Ai_|Te _i_th job and the arrival time of job_Ji_|_Di_|Te due date of job_Ji_|
|_ni_|Total number of operations belonging to job_Ji_|_Mi,j_|Te available machine set for operation_Oi,j_|
|_ti,j,k_|Te processing time of operation_Oi,j_ on machine_Mk_|_Mk_|Te _k_th machine|
|_Oi,j_<br>|Te _j_th operation of job_Ji_|_Ci,j,k_|Te completion time of_Oij_ on_Mk_|
|_M_0(_S_)|Te maximum completion time of original schedule|_Si,j,k_|Te start time of_Oij_ on_Mk_|



**Table 3** . Parameters.


_ni_ operations where _Oi,j_ is the _j_ th operation of job _Ji_ . Each operation _Oi,j_ can be processed on any machine _Mk_
selected from a compatible machine set _Mi,j_ ( _Mi,j_ _⊆_ _M_ ). The processing time of operation _Oi,j_ on machine
_Mk_ is denoted by _ti,j,k_ . The arrival time and due date of a job _Ji_ is _Ai_ and _Di_, respectively. _Ci,j_ represents the
actual completion time of operation _Oi,j_ . The objective is to minimize the total tardiness of all jobs.



**Machine breakdown**
Machine breakdown is the most common dynamic event in the actual production of a job shop. In the event of
machine breakdown, the machine will not be able to work the job until the machine is repaired. Therefore, it is
assumed that the simulation of a machine breakdown must include several important factors: the breakdown
machine, the time at which the breakdown begins, and the duration of the breakdown.

The algorithm proposed in this paper operates such that upon machine breakdown, it immediately halts
operations, and subsequently, the machine transitions to an unselectable maintenance state. The interrupted job
can be rescheduled for processing by other machines. In addition to the jobs currently being processed by each
machine, the remaining jobs undergo rescheduling to generate a new scheduling plan.

The probability of machine breakdown, denoted as _Mfr_, is jointly determined by the machine’s operating
rate and the accumulated operating time.



∑ _m_



∑ _n_



_i_ =1 _[t][i,j,k][ ·][ a][ijk]_ + _g_ 2 _Y_ (7)
_z_ max _∗_



_Mfr_ = _g_ 1 _∗_



_j_ =1



where, _g_ 1(0 _< g_ 1 _<_ 1) represents the weight of the machine operating rate, which is the proportion of machine
working time to the total scheduling time. _g_ 2(0 _< g_ 2 _<_ 1) represents the weight of the machine’s service life,
which in this paper is determined based on the actual service life of each machine, the value of _Y_ ranges from
unif[1, 10]. If the _j_ -th operati of the _i_ -th job is processed on machine k, then _aijk_ is 1; otherwise, it is 0. _z_ max is
the total processing time.


The starting time of the machine breakdown is generated by a uniform distribution as specified in Equation 8.

_Tk_ = [ _α_ 1 _M_ 0( _S_ ) _, α_ 2 _M_ 0( _S_ )] (8)


where, _α_ 1 and _α_ 2 are coefficients between intervals [0,1]. When _α_ 1 or _α_ 2 is between 0 and 0.5, the breakdown
is guaranteed to occur in the first half of the schedule, and when _α_ 1 or _α_ 2 is between 0.5 and 1, the breakdown
is guaranteed to occur in the second half of the schedule. For each machine _Mk_, this paper randomly selects a
machine breakdown time Ts within the range unif[55,75].


**Job processing time change**
During the production process, due to factors such as varying levels of worker proficiency in operating equipment
or equipment issues, it is not always possible to complete processing tasks according to the specified processing
time, resulting in early or delayed completion. This paper introduces the Due Date Tightness (DDT) parameter,
which serves as an indicator of the urgency of a job’s processing. For example, when a job _Ji_ arrives at a certain
time point _At_, and it contains _n_ processes, its processing deadline Di can be expressed as Equation 9.



∑ _n_

_Di_ = _At_ + ( ( _t_ [¯] _i,j_ + _t_ [¯] _agv_ )) _∗_ _DDT_ (9)

_j_ =1



The sum of the arrival time of the job and the average processing time of each process on the available machines
can be represented as the job’s processing deadline. The initial DDT is set to 1.


During the processing of jobs in the workshop, due to the fact that processing time fluctuations are influenced
by various external factors, this study assumes that the occurrence of processing time changes follows a Poisson
distribution, with a DDT range of [0.5, 1.5]. After processing time changes, other jobs are rescheduled based on
the latest estimated processing times and completion times.


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)



**Decision variables**



{
1 _,_ if _Oi,j_ is assigned on machine _Mk_ ;

_Ci,j_ : the completion time of operation _Oi,j_ . _Xi,j,k_ = 0 _,_ otherwise _._ _Magv,i_ =

{ {
1 _,_ if machine _i_ area _AGV_ is available; 1 _,_ If machine _m_ is available at time _t_



_Ci,j_ : the completion time of operation _Oi,j_ . _Xi,j,k_ =



0 _,_ otherwise _._ _Mk,t_ =



{
1 _,_ If machine _m_ is available at time _t_



0 _,_ otherwise _._ _Xi,j,k_



determines which machine an operation is assigned on, _Magv,i_ indicates the AGV configuration state of the
machine. _Mk,t_ represents whether machine _k_ is working properly at time _t_ .

Based on the above the problem definition, constraints, assumptions and notations, the mathematical model
can be established as follows.



∑ _n_



_Minimize_ (
_{_



_i_ =1



_max{Ci −_ _di,_ 0 _}_ ) _/n}._ (10)



Subject to:



∑



(1)  _k∈Mi,j_ _[X][i,j,k]_ [= 1] _[,]_ _∀i, j_ ;

(2)  _C_ ∑ _i,j,k_ _> Si,j,k,_ _∀i, j, k_ ;
(3)  _k_ _M_ _[M][agv,k]_ _[>]_ [ 0][;]



(1) 



_C_ ∑



(3)  _k∈M_ _[M][agv,k]_ _[>]_ [ 0][;]

(4) ( _Ci,_ 1 _−_ _ti,_ 1 _,k −_ _Ai_ ) _Xi,_ 1 _,k_ _≥_ 0 _,_ _∀i, k_ ;
(5) ( _Ci,j_ _−_ _ti,j,k −_ _Ci,j−_ 1) _Xi,j,k_ _≥_ 0 _,_ _∀i, j, k_ .where, Equation 10 is total tardiness of all jobs. Objective
(1) suggested that each operation can be assigned on only one machine. Objective (2) indicates that the
completion time of each job is after the start time. Objective (3) indicates that AGVs are available in the
processing workshop at every moment. Objective (4) makes sure that a job can only be processed after its
arrival time. Precedence constraint is ensured in Objective (5).



**Modeling framework**
**Define state characteristics**
In most scheduling methods based on DRL, status characteristics are defined as some indicators of production
status, that is, the number of machines/jobs/operations in the workshop, the remaining processing time of
unfinished jobs, the current workload/queue length of each machine, etc. However, in practical applications,
the number of machines/jobs/op-erations is infinite and may be very large. If these indicators are directly used
as state characteristics, the input of D3QN may have significant changes. This may result in poor performance
and universality of the D3QN model without training. Therefore, this article defines 8 state features based on
processing machines and operations, each of which takes a value within the range of [0, 1] as the input of DQN.
By limiting all state characteristics in [0, 1], D3QN can easily be extended to different untrained production
environments.

Before introducing the state features, some symbols should be defined in advance. First, let _CTk_ ( _t_ ) be
defined as the completion time of the last operation assigned to machine _Mk_ at the rescheduling point _t_ . _OPi_ ( _t_ )
is defined as the number of operations completed by job _Ji_ at the current time _t_ . The utilization rate of machine
_Mk_ at time _t_, _Uk_ ( _t_ ), is defined as equation 11.



∑ _OPj_ =1 _i_ ( _t_ ) _ti,j,kXi,j,k_

_CTk_ ( _t_ )



(11)



_Uk_ ( _t_ ) =



∑ _n_

_i_ =1



The completion rate of job _Ji_ at time _t_, denoted as _CRJi_ ( _t_ ), is defined as _CRJi_ ( _t_ ) = _[OP]_ _n_ _[i]_ [(] _[t]_ [)]



_n_ _[i]_ _i_ _[t]_ .



**Feature 1** : Average utilization rate of machines _Uave_ (t) can be calculated as _Uave_ ( _t_ ) =



∑ _nim_



_k_ =1 _[U][k]_ [(] _[t]_ [)]



: Average utilization rate of machines _Uave_ (t) can be calculated as _Uave_ ( _t_ ) = _k_ =1 _m_ _[k]_ ;

**Feature** √ **2** : T ~~∑~~ e standard deviation of machine utilization rate _Ustd_ ( _t_ ) can be calculated as



_Ustd_ ( _t_ ) =



√ ~~∑~~ _m_



_m_

_k_ =1 [(] _[U][k]_ [(] _[t]_ [)] _[−][U][ave]_ [(] _[t]_ [))][2]



_m_ _[ave]_ ;



**Feature 3** : Maximum difference value in machine utilization rate _Umax_ ( _t_ ) can be calculated as _Umax_ ( _t_ ) =
MAx( _Uk_ ( _t_ )) MIN( _Uk_ ( _t_ )); ∑ _n_

_−_ _i_ =1 _[CRJ][i]_ [(] _[t]_ [)]



**Feature 4** : Average completion rate of jobs _CRJave_ ( _t_ ) can be calculated as _CRJave_ ( _t_ ) =



∑ _n_



_i_ =1 _[CRJ][i]_ [(] _[t]_ [)]



**Feature 4** : Average completion rate of jobs _CRJave_ ( _t_ ) can be calculated as _CRJave_ ( _t_ ) = _i_ =1 _n_ _[i]_ ;

**Feature 5** : √T ~~∑~~ e standard deviation of job completion rate _CRJstd_ ( _t_ ) can be calculated as



_CRJstd_ ( _t_ ) =



√T ~~∑~~ _n_



_n_

_i_ =1 [(] _[CRJ][i]_ [(] _[t]_ [)] _[−][CRJ][ave]_ [(] _[t]_ [))][2]



_n_ _[ave]_ ;



**Feature 6** : Average completion rate of operations _CROave_ ( _t_ ) can be calculated as _CROave_ ( _t_ ) =



∑ _n_



~~∑~~ _i_ =1 _n_ _[OP][i]_ [(] _[t]_ [)]



_i_ =1 _[n][i]_



.

**Feature 7** : Estimated tardiness rate _Tarde_ ( _t_ ). First, define _T_ cur as the average completion time of the last
operation on all machines at the current time _t_ . The estimated delayed jobs can be defined as those whose
remaining processing time exceeds _T_ cur until their due time. The estimated delay rate _T_ arde( _t_ ) is defined as the




[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)



ratio of the number of estimated delayed operations _N_ tard to the total number of incomplete operations _N_ left
for all remaining jobs, calculated as _Tarde_ ( _t_ ) = _[N]_ _N_ [tard] left [;]

**Feature 8** : Actual tardiness rate _Tarda_ ( _t_ ). The actual delayed jobs are defined as those jobs that remain
incomplete before their due dates. The actual delay rate _T_ arda( _t_ ) is defined as the ratio of the number of timedelayed operations _N_ act to the total number of incomplete operations _N_ left for all remaining jobs, calculated as
_Tarda_ ( _t_ ) = _[N]_ [act] [.]




[tard]

_N_ left [;]




_[N]_ [act]

_N_ left [.]



**The proposed dispatching rules**
This paper designs a state evaluation rank MR algorithm for machining machines, which uses the MR values
of machining machines to design different action rules.The seven composite dispatching rules are described in
details in the following part. All the rules are designed for the purpose of reducing total tardiness.


_Composite dispatching rule1_
Firstly, define a reference time _Tcur_ as the average completion time of the last assigned operation on each
machine at the current decision point _t_ . Meanwhile, the set of all uncompleted jobs at _t_ is denoted by _UCjob_ ( _t_ ).
Each job _Ji_ _UCjob_ ( _t_ ) is ranked by the average idle time _n_ _[D]_ _i_ _[i][−]_ _OP_ _[T][cur]_ _i_ ~~(~~ _t_ ~~)~~ [ of its remaining operations, and then the ]
_∈_ _−_

job with the lowest average idle time is selected for the next job. At the same time, select the assignable machine
_MR_ value minimum and the AGV status for the idle to give this job.


_Composite dispatching rule2_
Firstly, define a set _Tardjob_ ( _t_ ) as the estimated set of delayed jobs, which are uncompleted jobs with deadlines
earlier than _Tcur_ . Then calculate the estimated tardiness time by selecting the currently uncompleted job, then
select the job in _Tardjob_ ( _t_ ) with the maximum estimated tardiness, and at the same time, select the machine
that can be assigned the minimum _M_ _R_ value and the AGV state to idle for this job.


_Composite dispatching rule3_
Firstly, utilize the estimated delay to determine the presence of any delayed jobs, if _Tardjob_ ( _t_ ) is not empty,
select the job in _Tardjob_ ( _t_ ) with the maximum estimated tardiness, otherwise, calculate the process completion
rate of each job, and select the job with the minimum completion rate to proceed to the next job. At the same
time, select the machine that can be assigned the minimum _MR_ value and the AGV status to idle for this job.


_Composite dispatching rule4_
Firstly, obtain the estimated set of delayed jobs _Tardjob_ ( _t_ ) as shown in Rule 3. If delayed job exists, select the
job in _Tardjob_ ( _t_ ) with the maximum estimated tardiness. Otherwise, the ratio of the relaxation time for each
unfinished job to the remaining time is calculated, and the job with the minimum ratio is selected. At the same
time, select the machine that can be assigned the minimum _MR_ value and the AGV status to idle for this job.


_Composite dispatching rule5_
Firstly, utilize the estimated delay to determine the presence of any delayed jobs, if _Tardjob_ ( _t_ ) is not empty, the
job with the maximum product of estimated tardiness and inverse completion rate is selected. If not, calculate
the product of job completion rate and relaxation time, and select the job with the minimum value. At the same
time, select the machine that can be assigned the minimum _MR_ value and the AGV status to idle for this job.


_Composite dispatching rule6_
Select the machine with the minimum _MR_ value of the assignable machine and the AGV status to be idle, then
randomly select an uncompleted job to assign the next job to that machine.


_Composite dispatching rule7_
Select the earliest available machine and the AGV is idle, and randomly select an uncompleted job operation to
assign to that machine.


**Action selection strategy**
The action selection strategy of DRL is also called the search strategy, which offers a trade-off between exploration
and exploitation. The unknown environment is explored and the acquired knowledge is utilized to guide the
choice of action by the agent. At the beginning, all Q values are zero, which means that the agent does not have
any learning experience to use, only the exploration can be performed and learned.

_ε_ -greedy is an action selection strategy that considers both exploration and exploitation, which is expressed
by equation 12 [36] . where _ε_ is called the greedy rate or the exploitation rate and _r_ 0 1 is a random value from 0 to

_−_
1. When _ε_ _r_ 0 1, the action _a_ which maximizes the expected Q value is selected, which is also called greedy
_≥_ _−_
strategy. While _ε_ _<_ _r_ 0 1, the exploration will be performed and a random action _a_ is chosen.

_−_



_n_ _[D]_ _i−_ _[i][−]_ _OP_ _[T][cur]_ _i_ ~~(~~ _t_ ~~)~~ [ of its remaining operations, and then the ]



_π_ ( _st, at_ ) =



{
max _a Q_ ( _st, a_ ) _ε_ _r_ 0 1
_≥_ _−_ (12)
_a_ ( _Randomly_ ) _ε < r_ 0 _−_ 1



**Reward function**
Due to the goal of minimizing total tardiness, the reward _rt_ of state action pairs ( _st_, _at_ ) is defined by continuously
considering the values of three key state characteristics of the current state _st_ and the next state _st_ +1, including


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)


the actual tardiness rate _Tarda_, machine _MR_ values and the estimated job tardiness rate _Tarde_ . This paper
designs a reward function that considers the actual tardiness rate, and _MR_ value. The process of calculating _rt_ is
shown in Algorithm 2.


**Algorithm 2** . Reward function


Firstly, determine the MR values and estimated delay rates of the machining machine in the current state _st_
and the next state _st_ +1. If the _MR_ value in the current state _st_ is less than or equal to the value in the state _st_ +1
and the estimated delay rate in the current state _st_ is greater than the value in the state _st_ +1, then the reward
value is increased by 1 (lines 1,2). Otherwise, if the estimated delay rate of the job in the current state _st_ is less
than or equal to the value in state _st_ +1, the reward value will be reduced by 1 (lines 3, 4). Otherwise, if the actual
delay rate of the job in the current state _st_ is greater than or equal to the value in state _st_ +1, the reward value will
be increased by 1 (lines 5,6). Otherwise, if the _MR_ value in the current state st is greater than the value in the state
_st_ +1 and the actual delay rate of the job in the current state _st_ is less than the value in the state _st_ +1, the reward
value will be reduced by 1 (lines 7,8). Otherwise, the reward value remains unchanged (line 9,10).



**Network model**
According to 8 state features, they are divided into two groups based on the dimensions of machine and job. The
state features of each group are completely connected in D3QN, but the state features of different groups are not
interconnected between the first three layers of groups. As shown in Figure 1(b).

The proposed D3QN includes one input layer, four hidden layers, and one output layer. The number of input
nodes and output nodes are the same as the number of state features and the number of actions, respectively. In
each hidden layer, use 32 nodes. In this study, D3QN was not fully connected. As shown in the figure 1(b). The 8
state features are divided into two groups, and the state features of each group are completely interconnected in
D3QN, but the state features of different groups are not interconnected in the first three hidden layers. Between
the third and fourth hidden layers, the nodes are fully connected. The “Correction Linear Unit (ReLU)” is used
as the activation function in the input layer and hidden layer, and the “pulse function” in the output layer.
Compared with fully connected neural networks, the proposed D3QN reduces the number of weights and
deviations, saving computational time and resources. In addition, state features belonging to different groups
will not be disturbed during the training process.

In this model, the online network Q performs gradient descent step training based on error
_E_ = 0 _._ 5 _∗_ [∑] _i_ _[|][A]_ =1 _[|]_ [( ˆ] _[r][i][ −]_ _[r][i]_ [)][2][, while the target network updates ] _[Q]_ [ˆ][ based on ] _[Q]_ [ˆ][ =] _[ τQ]_ [ + (1] _[ −]_ _[τ]_ [) ˆ] _[Q]_ [. To minimize ]

the error between the online network value and the target network value, the Mean Square Error [37] loss function
is used for calculation.

The loss function aims to minimize the error between the online Q value and the target Q value. The loss
function can be expressed using equation 13.



(13)




[



(



_Q_ [(]

_−_



; _θt_ _[′]_



)



_Lt_ ( _θt_ ) = E



_rt_ +1 + _γQ_



_st_ +1 _,_ argmax _aQ_ [(]



_st_ +1 _, at_ +1; _θt_



)



_st, at_ ; _θt_



]2
)



Where, _Q_ ( _st, at_ ; _θt_ ) is the online network value, which is the output of the neural network
and represents the predicted value of the neural network when the state is _s_ and the action is _a_ .



(



_st_ +1 _,_ argmax _Q_ [(]



; _θt_ _[′]_



)



)



is the target network value, which refers to the actual



_rt_ +1 + _γQ_



_st_ +1 _, at_ +1; _θt_



observed value when action _a_ is chosen. When the agent selects an action, it receives a reward, which can be
stored. We can also store the discounted rewards for every subsequent action after that. The sum of these two
values constitutes the target Q value, representing all the rewards actually obtained for a given action. Here, _γ_ is
the discount factor, indicating the decay of future rewards.


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)


**Fig. 2** . Decision process of dynamic scheduling problem.

|Parameter|Value|Parameter|Value|
|---|---|---|---|
|Total number of machines(m)|{10,20}|Learning rate|0.0001|
|Number of initial jobs at beginning (_nini_)|20|Total number new inserted jobs (_nadd_)|{30,50,80}|
|Due date tightness (_DDT_)|{1.0, 1.5}|Number of operations belonging to a job|Unif[2,10]|
|Processing time of an operation on an available machine|Unif[1,30]|Number of available machines of each operation|Unif[0,10]|
|_ε_ in the action selection policy|0.1|_τ_ in the sof target update strategy|0.01|
|Discount factor_γ_|0.95|Replay bufer size|1000|



**Table 4** . Parameter settings of different production configurations.


According to the target setting, the environment includes processing equipment, transportation AGV, and jobs
to be processed. Based on the input of the environment, the current state and reward information are obtained,
and the next state is input into the intelligent agent. Decision strategies (transportation AGV, processing jobs,
and machine information) are output through the main Q network. By executing actions, the processing jobs are
transported to the processing machine’s processing area through AGV. When a disturbance event occurs, it will
trigger the agent to execute a rescheduling. The dynamic scheduling process is shown in Figure 2.


**Results and discussion**
This paper assumes that there are several jobs initially present in the workshop, and new jobs arrive according to
a Poisson distribution, resulting in exponential distribution of inter-arrival times between two consecutive new
jobs. The experimental parameters used are shown in table 4. To ensure the effectiveness and versatility of D3QN,


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)


various sensitive parameters such as learning rate and action selection strategy _ε_ were investigated during the
training phase. The average total delay of D3QN under different production configurations was compared, and
suitable values for each parameter were ultimately determined. For instance, to determine the appropriate value
of the action selection parameter _ε_, decreased it from 0.4 to 0.1 with a step size of 0.02, ultimately selecting the
parameter value that resulted in the lowest average total delay.

This experiment was conducted on a single server model SR670, equipped with 4 GPUs and 192GB of physical
memory. It simulates a flexible job shop environment for training, consisting of 10 machines, an initial insertion
of 20 jobs, and subsequent insertion of 50 new jobs. Additionally, during the training process, it randomly
simulates scenarios involving machine malfunction and variations in job processing times. The average value of
exponential distribution between two successive new job arrivals is set to 100, while _DDT_ is set to 1.0. Further
details of the parameter settings for this training approach are available in Table 4.


**Comparison of DQN, DDQN and D3QN training results**
In order to verify the advantages of the D3QN based model proposed in this paper, this experiment will compare
the algorithm proposed in this paper, DDQN and DQN algorithms, and verify the advantages of the algorithm
by comparing the job completion rate and convergence rate. The experiment uses data to simulate a workshop
with 10 machines, an initial insertion of 20 jobs, and subsequent insertion of 50 new jobs. The average value of
exponential distribution between two successive new job arrivals is set to 100, while _DDT_ is set to 1.0, and other
parameter settings are shown in Table 4. The comparison effect is shown in Figure 3(a).

To validate the advantages of the D3QN-based model proposed in this paper, we first compare the training
results of three algorithms. The DQN algorithm performs the worst, with the lowest learning efficiency and
slow convergence speed. DDQN shows significant improvements and faster convergence but still does not
match the performance of the D3QN algorithm. The D3QN algorithm demonstrates the best convergence and
stability, benefiting from the deep double Q-network and convolutional neural network’s multidimensional
state space, which alleviates the impact of action value overestimation. The convergence results for the on-time
completion rate of workpieces for the different algorithms are shown in Figure 3(a). It is clear that all three
deep reinforcement learning algorithms reach convergence, with rates of 0.85, 0.8, and 0.7, respectively. D3QN
achieves a more stable convergence, while the other two algorithms fail to learn superior scheduling rules at
rescheduling points, resulting in lower on-time completion rates for workpieces. Experimental comparisons
demonstrate that, in the presence of multiple constraints and disturbances, the proposed D3QN-based dynamic
multi-objective intelligent scheduling algorithm learns more efficient scheduling rules.

Subsequently, to verify the efficiency of the proposed D3QN-based scheduling algorithm, we conducted a
comparison with the DDQN-based scheduling algorithm and designed the following experiments. To ensure
fairness among the different algorithms, the DDQN utilized the action and reward functions of the scheduling
algorithm proposed in this paper. The mean and standard deviation of the two optimization objectives are
shown in Table 5.

From the table 5, it can be observed that the scheduling algorithm based on D3QN demonstrates better
expectations and robustness in achieving multiple objectives compared to the DDQN-based scheduling
algorithm when selecting the optimal scheduling rules at rescheduling points.


**Fig. 3** . Data migration-fusion mechanism.


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)






|m|n<br>add|E<br>ave|DDQN(makespan)|Ours(makespan)|DDQN(trave)|Ours(trave)|
|---|---|---|---|---|---|---|
|10|30|50|6.713e+02/4.148e+01|**5.784e+02/3.571e+01**|8.110e-01/3.018e-02|**8.452e-01/2.325e-02**|
|10||75|6.954e+02/4.751e+01|**6.184e+02/3.705e+01**|8.203e-01/2.721e-02|**8.621e-01/2.388e-02**|
|10||100|7.343e+02/5.462e+01|**6.721e+02/4.165e+01**|8.017e-01/3.319e-02|**8.790e-01/2.903e-02**|
|10|50|50|9.026e+02/6.741e+01|**7.623e+02/5.253e+01**|5.134e-01/3.668e-02|**6.410e-01/1.215e-02**|
|10||75|9.831e+02/7.214e+01|**8.561e+02/5.624e+01**|5.556e-01/3.928e-02|**6.629e-01/1.355e-02**|
|10||100|1.168e+03/7.454e+01|**8.907e+02/5.511e+01**|5.724e-01/4.194e-02|**6.369e-01/1.414e-02**|
|10|80|50|1.204e+03/7.804e+01|**9.768e+02/6.051e+01**|3.055e-01/1.681e-02|**3.174e-01/9.222e-01**|
|10||75|1.357e+03/8.371e+01|**9.943e+02/6.421e+01**|3.624e-01/2.036e-02|**4.592e-01/1.604e-02**|
|10||100|1.749e+03/8.063e+01|**1.031e+03/6.401e+01**|3.446e-01/2.633e-02|**4.621e-01/2.310e-02**|
|20|30|50|2.891e+03/5.784e+01|**3.347e+02/5.315e+01**|7.155e-01/3.618e-02|**8.354e-01/2.064e-02**|
|20||75|3.514e+03/6.441e+01|**3.649e+02/5.815e+01**|7.332e-01/3.996e-02|**8.141e-01 /2.311e-02**|
|20||100|3.730e+03/7.514e+01|**3.894e+02/6.015e+01**|7.659e-01/4.924e-02|**8.934e-01/2.971e-02**|
|20|50|50|3.868e+03/5.248e+01|**3.648e+02/4.126e+01**|5.484e-01/4.012e-02|**7.356e-01/2.246e-02**|
|20||75|4.261e+03/6.351e+01|**3.781e+02/4.796e+01**|5.993e-01/4.681e-02|**7.681e-01/3.392e-02**|
|20||100|4.408e+03/7.231e+01|**3.991e+02/5.218e+01**|6.165e-01/5.247e-02|**7.802e-01/3.941e-02**|
|20|80|50|2.065e+03/6.151e+01|**5.698e+02/4.704e+01**|3.953e-01/3.621e-02|**4.632e-01/1.826e-02**|
|20||75|3.519e+03/7.415e+01|**5.825e+02/4.659e+01**|5.046e-01/4.054e-02|**5.771e-01/2.382e-02**|
|20||100|6.830e+03/8.598e+01|**8.217e+02/5.793e+01**|5.304e-01/4.872e-02|**6.439e-01/2.818e-02**|



**Table 5** . Comparison of average and standard deviation value of result after 50 runs of single combination
rules and ours.


**Fig. 4** . Gantt chart of job scheduling.


**Disturbance factors rescheduling strategy**
This paper mainly focuses on the study of disturbance factors, including machine breakdown, insertion of new
jobs, and job processing time change. When a disturbance factor occurs, a rescheduling strategy will be triggered.
This section takes machine breakdown as an example to verify the effectiveness of the strategy.

Set up 9 processing machines and 15 jobs (5 initial jobs and 10 newly inserted jobs). With the parameter
_Eave_ set to 100, DDT set to 1.0, and other parameters configured as specified in Table 4, use the trained
algorithm to screen and calculate the orders, and the initial scheduling task arrangement is shown in Figure 4.
The processing period is 19:16:46-19:21:42, which lasts for 296 _seconds_ . The machine breakdown simulation is
that _M_ 9 experiences a breakdown during the second operation of processing the sixth job (as indicated in the
figure 4).

According to the fault disturbance strategy, the processes currently being processed on other normal
processing machines will continue to be processed and completed. The processes that have not yet started
processing and the processes being processed on damaged machines need to be rescheduled to reduce the
impact of machine malfunction on the production plan. The most crucial aspect in this process is to handle the
moment of machine malfunction, the status of the jobs being processed and the machines, and to calculate and
analyze the subsequent tasks that require rescheduling as well as the manufacturing resource situation.

The statistical manufacturing resource information includes available processing machine, the allowed start
time of each processing machine, the job processes that need to continue processing, and the allowed start time
of the subsequent job processes. These information are used as constraints for rescheduling and considered
during the main Q network decision-making process. Figure 5 shows the production scheduling result after
resc-heduling. After rescheduling, the _M_ 9 processing machine, which is in a malfunction state and cannot be


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)


**Fig. 5** . Rescheduling Gantt chart.






|m|n<br>add|E<br>ave|FIFO|MRT|EDD|SPT|LPT|Ours|
|---|---|---|---|---|---|---|---|---|
|10|30|50|7.345e+02|7.897e+02|7.435e+02|6.723e+02|5.896e+02|**5.784e+02**|
|10||75|7.893e+02|8.216e+02|8.052e+02|7.415e+02|6.039e+02|**5.884e+02**|
|10||100|8.346e+02|8.408e+02|8.343e+02|7.903e+02|7.096e+02|**6.721e+02**|
|10|50|50|4.243e+02|4.135e+02|4.305e+02|3.809e+02|9.756e+02|**7.623e+02**|
|10||75|1.461e+03|1.435e+03|1.428e+03|1.152e+03|1.314e+03|**8.561e+02**|
|10||100|1.866e+03|1.607e+03|1.566e+03|1.333e+03|1.276e+03|**8.907e+02**|
|10|80|50|1.511e+03|1.461e+03|1.484e+03|1.015e+03|1.397e+03|**9.768e+02**|
|10||75|1.645e+03|1.601e+03|1.473e+03|1.408e+03|1.368e+03|**9.943e+02**|
|10||100|1.706e+03|1.691e+03|1.608e+03|1.472e+03|1.377e+03|**1.031e+03**|
|20|30|50|4.243e+02|4.135e+02|4.305e+02|3.809e+02|3.232e+02|**3.347e+02**|
|20||75|4.811e+02|4.521e+02|4.598e+02|3.984e+02|3.815e+02|**3.649e+02**|
|20||100|5.491e+02|4.775e+02|5.339e+02|4.343e+02|4.265e+02|**3.894e+02**|
|20|50|50|5.565e+02|5.647e+02|6.212e+02|4.845e+02|4.918e+02|**3.648e+02**|
|20||75|5.947e+02|5.997e+02|6.522e+02|5.597e+02|5.169e+02|**3.781e+02**|
|20||100|6.286e+02|6.059e+02|6.971e+02|5.951e+02|5.223e+02|**3.991e+02**|
|20|80|50|7.554e+02|7.371e+02|7.145e+02|7.029e+02|6.736e+02|**5.698e+02**|
|20||75|7.703e+02|7.824e+02|7.639e+02|6.812e+02|8.024e+02|**5.825e+02**|
|20||100|8.174e+02|8.575e+02|8.067e+02|7.841e+02|6.843e+02|**6.217e+02**|



**Table 6** . Comparison of average value of makespan after 50 runs of single combination rules and ours.


repaired, cannot proceed with the job processing. The entire task period after rescheduling is 19:19:54-19:24:56,
which lasts for 302 _seconds_, 6 _seconds_ later than before the malfunction. Despite the reduction in available
resources, this scheduling result meets expectations.


**Comparisons with other well-known dispatching rules**
To further demonstrate the superiority of DQN, we compared it with five other well-known scheduling rules,
including FIFO, EDD, MRT, SPT, and LPT. FIFO selects the next operation from the job that arrived first. EDD
selects the next operation from the job with the earliest due date. MRT selects the next operation from the
job with the most remaining processing time. SPT selects the next operation from the job with the shortest
processing time among the current jobs. LPT selects the next operation from the job with the longest processing
time among the current jobs. Additionally, a completely random rule, where an unprocessed operation is
randomly selected and assigned to an available machine, was also considered for comparison.

In this section, multiple sets of experimental data were generated under different parameter settings for _m_,
_nadd_, and _Eave_ . These sets simulate various task scheduling scenarios in the production process. For each set
of experimental data, the proposed scheduling algorithm was repeated 50 times, each time alongside a different
scheduling rule. Table 6 lists the average total completion time obtained from each method, with the best results
highlighted in bold.

It can be seen from the experimental results that different experimental parameter configurations will affect
the performance of our proposed scheduling algorithm and other scheduling rules. As expected, due to the
greater number of new parts arriving and the more frequent arrival times of new parts, the delay time and
completion time will also increase, and the job completion rate will decrease. On the other hand, with the


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)


increase in the number of devices in the production environment, the job malfunction time can be reduced and
the job completion rate can be improved.

Compared with a single scheduling rule, the proposed scheduling algorithm has significantly better
performance in minimizing the maximum completion time, which means that the algorithm has learned to
select an efficient scheduling strategy with appropriate scheduling rules at different rescheduling points. Under
the variable production environment, there is no single scheduling rule that can achieve the optimal scheduling
performance like our proposed scheduling algorithm. This proves that the algorithm also has good universality
in all kinds of untrained cases.


**Compare fully connected neural networks**
In this paper, in order to improve the training speed of the model, a classification neural network model is
designed in this paper. Compared with the fully connected neural network, the proposed D3QN-based
classification neural network reduces the number of weights and deviations, and saves computing time and
resources. In addition, state features belonging to different groups are not disturbed during training.

As shown in Figure 3(b), the training effect of the fully connected neural network and the classification neural
network is compared. The experiment uses data to simulate a workshop with 10 machines, an initial insertion
of 30 jobs, and subsequent insertion of 50 new jobs. The average value of exponential distribution between
two successive new job arrivals is set to 75, while _DDT_ is set to 1.5, and other parameter settings are shown in
Table 4. Through the comparison, it can be found that the convergence state can be reached faster by using the
classification neural network for training, which verifies the performance advantage of the classification network
model.


**Conclusion**
In this paper, a D3QN network is proposed to solve the dynamic scheduling problem of flexible job shop
considering new job insertion, machine breakdown, job processing time change, and AGV status. A precise
calculation of the comprehensive utilization rate of machining machines, the MachineRank algorithm,
was proposed, and seven composite scheduling rules were developed based on the MR algorithm to select
unprocessed operations and allocate them to available machines each time an operation is completed or a
new job arrives. Numerical experiments under different production environments are conducted to verify the
effectiveness and generality of the proposed D3QN. The results demonstrate that the scheduling algorithm
proposed in this paper consistently outperforms single scheduling rules significantly. By comparing the D3QNbased scheduling algorithm with the traditional DDQN algorithm in deep reinforcement learning, the paper
validates the effectiveness and robustness of the proposed scheduling algorithm in handling scenarios with
multiple constraints and disruptions.In the future work, we will consider the influence of more disturbance
factors on shop scheduling. Such as energy consumption, material handling time, etc.


**Data availability**
The datasets generated during this study are available from the corresponding author on reasonable request.


Received: 2 September 2024; Accepted: 11 November 2024



**References**
1. Ouelhadj, D. & Petrovic, S. A survey of dynamic scheduling in manufacturing systems. _Journal of Scheduling_ **12**, 417–431 (2009).
2. Brucker, P. & Schlie, R. Job-shop scheduling with multi-purpose machines. _Computing_ **45**, 369–375 (1990).
3. Luo, S. Zhang, L. & Fan, Y. Dynamic multi-objective scheduling for flexible job shop by deep reinforcement learning. _Computers_

_& Industrial Engineering_ **159** [, 107489, https://doi.org/10.1016/j.cie.2021.107489 (2021).](https://doi.org/10.1016/j.cie.2021.107489)
4. Luo, S. Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning. _Applied Soft Computing_

**91** [, 106208. https://doi.org/10.1016/j.asoc.2020.106208 (2020).](https://doi.org/10.1016/j.asoc.2020.106208)
5. LuoShu, ZhangLinxuan & FanYushun. Dynamic multi-objective scheduling for flexible job shop by deep reinforcement learning.

(2021).
6. Yan, Q., Wu, W. & Wang, H. Deep reinforcement learning for distributed flow shop scheduling with flexible maintenance. _Machines_

**10** [, https://doi.org/10.3390/machines10030210 (2022).](https://doi.org/10.3390/machines10030210)
7. Yan, Q., Wang, H. & Wu, F. Digital twin-enabled dynamic scheduling with preventive maintenance using a double-layer q-learning

algorithm. _Computers & Operations Research_ **144**, 105823 (2022).
8. Lou, P., Liu, Q., Zhou, Z., Wang, H. & Sun, S. Multi-agent-based proactive–reactive scheduling for a job shop. _International Journal_

_of Advanced Manufacturing Technology - INT J ADV MANUF TECHNOL_ **59** [, https://doi.org/10.1007/s00170-011-3482-4 (2012).](https://doi.org/10.1007/s00170-011-3482-4)
9. Kundakci, N. & Kulak, O. Hybrid genetic algorithms for minimizing makespan in dynamic job shop scheduling problem.

_Computers & Industrial Engineering_ **96**, 31–51 (2016).
10. Ning, T., Huang, M., Liang, X. & Jin, H. A novel dynamic scheduling strategy for solving flexible job-shop problems. _Journal of_

_Ambient Intelligence and Humanized Computing_ **7** [, https://doi.org/10.1007/s12652-016-0370-7 (2016).](https://doi.org/10.1007/s12652-016-0370-7)
11. Xu, Y., Zhang, M., Yang, M. & Wang, D. Hybrid quantum particle swarm optimization and variable neighborhood search for

flexible job-shop scheduling problem. _Journal of Manufacturing Systems_ **73**, 334–348 (2024).
12. Wen, X. _et al._ Dynamic scheduling method for integrated process planning and scheduling problem with machine fault. _Robotics_

_and Computer-Integrated Manufacturing_ **77**, 102334– (2022).
13. Zhang, J.-D., He, Z., Chan, W.-H. & Chow, C.-Y. Deepmag: Deep reinforcement learning with multi-agent graphs for flexible job

shop scheduling. _Knowledge-Based Systems_ **259**, 110083 (2023).
14. Waubert de Puiseau, C., Meyes, R. & Meisen, T. On reliability of reinforcement learning based production scheduling systems: A

comparative survey. _J. Intell. Manuf._ **33**, 911–927 (2022).
15. Liu, R., Piplani, R. & Toro, C. Deep reinforcement learning for dynamic scheduling of a flexible job shop. _International Journal of_

_Production Research_ **60**, 4049–4069 (2022).
16. Lee, Y. H. & Lee, S. Deep reinforcement learning based scheduling within production plan in semiconductor fabrication. _Expert_

_Syst. Appl._ **191** (2022).




[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)



17. Palombarini, J. A. & Martínez, E. C. Closed-loop rescheduling using deep reinforcement learning. _IFAC-PapersOnLine_ **52**, 231–

236 (2019).
18. Liu, C.-L., Chang, C.-C. & Tseng, C.-J. Actor-critic deep reinforcement learning for solving job shop scheduling problems. _Ieee_

_Access_ **8**, 71752–71762 (2020).
19. Ren, J., Ye, C. & Yang, F. A novel solution to jsps based on long short-term memory and policy gradient algorithm. _International_

_Journal of Simulation Modelling_ **19**, 157–168 (2020).
20. Zhang, L. et al. Deep reinforcement learning for dynamic flexible job shop scheduling problem considering variable processing

times. _Journal of Manufacturing Systems_ **71**, 257–273 (2023).
21. Lei, K. et al. A multi-action deep reinforcement learning framework for flexible job-shop scheduling problem. _Expert Systems with_

_Applications_ **205**, 117796 (2022).
22. Nelson, R. T. & Wong, C. A. H. . R. M.-L. Centralized scheduling and priority implementation heuristics for a dynamic job shop

model. _A I I E Transactions_ (1977).
23. Chaoyong, Z., Xinyu, L., Xiaojuan, W., Qiong, L. & Liang, G. Multi-objective dynamic scheduling optimization strategy based on

rolling-horizon procedure. _China Mechanical Engineering_ (2009).
24. Luo, S. Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning. _Applied Soft Computing_

**91**, 106208 (2020).
25. Shiue, Y.-R., Lee, K.-C. & Su, C.-T. Real-time scheduling for a smart factory using a reinforcement learning approach. _Computers_

_& Industrial Engineering_ **125**, 604–614 (2018).
26. Chen, X., Hao, X., Lin, H. W. & Murata, T. Rule driven multi objective dynamic scheduling by data envelopment analysis and

reinforcement learning. In _2010 IEEE International Conference on Automation and Logistics_, 396–401 (IEEE, 2010).
27. Lin, C.-C., Deng, D.-J., Chih, Y.-L. & Chiu, H.-T. Smart manufacturing scheduling with edge computing using multiclass deep q

network. _IEEE Transactions on Industrial Informatics_ **15**, 4276–4284 (2019).
28. Han, B.-A. & Yang, J.-J. Research on adaptive job shop scheduling problems based on dueling double dqn. _Ieee Access_ **8**, 186474–

186495 (2020).
29. Wu, Z., Fan, H., Sun, Y. & Peng, M. Efficient multi-objective optimization on dynamic flexible job shop scheduling using deep

reinforcement learning approach. _Processes_ **11**, 2018 (2023).
30. Song, W., Chen, X., Li, Q. & Cao, Z. Flexible job-shop scheduling via graph neural network and deep reinforcement learning. _IEEE_

_Transactions on Industrial Informatics_ **19**, 1600–1610 (2022).
31. Kundakcı, N. & Kulak, O. Hybrid genetic algorithms for minimizing makespan in dynamic job shop scheduling problem.

_Computers & Industrial Engineering_ **96**, 31–51 (2016).
32. de Araujo, S. A., Arenales, M. N. & Clark, A. R. Joint rolling-horizon scheduling of materials processing and lot-sizing with

sequence-dependent setups. _Journal of Heuristics_ **13**, 337–358 (2007).
33. Wang, Z., Zhang, J. & Yang, S. An improved particle swarm optimization algorithm for dynamic job shop scheduling problems

with random job arrivals. _Swarm and Evolutionary Computation_ **51**, 100594 (2019).
34. Park, J., Mei, Y., Nguyen, S., Chen, G. & Zhang, M. An investigation of ensemble combination schemes for genetic programming

based hyper-heuristic approaches to dynamic job shop scheduling. _Applied Soft Computing_ **63**, 72–86 (2018).
35. Zhang, Y., Zhu, H., Tang, D., Zhou, T. & Gui, Y. Dynamic job shop scheduling based on deep reinforcement learning for multi
agent manufacturing systems. _Robotics and Computer-Integrated Manufacturing_ **78**, 102412 (2022).
36. Wang, H. et al. Adaptive and large-scale service composition based on deep reinforcement learning. _Knowledge-Based Systems_ **180**,

75–90 (2019).
37. Error, M. S. Mean squared error. _MA: Springer US_ 653–653 (2010).



**Acknowledgements**
This research was supported by the National Key Research and Development Program of China under Grant
2021YFB1716200 and the Research Funds for leading Talents Program(048000514122549).


**Author contributions**
H.L. contributed to the conception, supervision, review and editing of the study. F.R. significantly contributed
to the methodology, experiment and manuscript preparation. F.R. supervised the project and contributed to the
mathematical modeling and experimental data analysis. All authors discussed the results and contributed to the
final manuscript.


**Declarations**


**Competing interests**
The authors declare no competing interests.


**Additional information**
**Correspondence** and requests for materials should be addressed to H.L.

**Reprints and permissions information** is available at www.nature.com/reprints.

**Publisher’s note** Springer Nature remains neutral with regard to jurisdictional claims in published maps and
institutional affiliations.


[www.nature.com/scientificreports/](http://www.nature.com/scientificreports)


**Open Access** This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives
4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in
any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide
a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have
permission under this licence to share adapted material derived from this article or parts of it. The images or
other third party material in this article are included in the article’s Creative Commons licence, unless indicated
otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence
and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to
[obtain permission directly from the copyright holder. To view a copy of this licence, visit ​h​t​t​p​:​/​/​c​r​e​a​t​i​v​e​c​o​m​m​o​](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[n​s​.​o​r​g​/​l​i​c​e​n​s​e​s​/​b​y​-​n​c​-​n​d​/​4​.​0​/.](http://creativecommons.org/licenses/by-nc-nd/4.0/) **​​**


© The Author(s) 2024


