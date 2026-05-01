Received May 26, 2021, accepted July 11, 2021, date of publication July 15, 2021, date of current version July 23, 2021.


_Digital Object Identifier 10.1109/ACCESS.2021.3097254_

# Deep Reinforcement Learning for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups


BOHYUNG PAENG 1,2, IN-BEOM PARK 3, AND JONGHUN PARK 1,2

1Department of Industrial Engineering, Seoul National University, Seoul 08826, South Korea
2Institute for Industrial Systems Innovation, Seoul National University, Seoul 08826, South Korea
3Department of Industrial Engineering, Sungkyunkwan University, Suwon 16419, South Korea


Corresponding author: In-Beom Park (inbeom@skku.edu)


This work was supported in part by the National Research Foundation of Korea (NRF) Grant funded by Ministry of Science and ICT
(MSIT) under Grant NRF-2015R1D1A1A01057496, and in part by the Institute of Engineering Research, Seoul National University.


**ABSTRACT** Parallel machine scheduling with sequence-dependent family setups has attracted much
attention from academia and industry due to its practical applications. In a real-world manufacturing system,
however, solving the scheduling problem becomes challenging since it is required to address urgent and
frequent changes in demand and due-dates of products. To minimize the total tardiness of the scheduling
problem, we propose a deep reinforcement learning (RL) based scheduling framework in which trained
neural networks (NNs) are able to solve unseen scheduling problems without re-training even when such
changes occur. Specifically, we propose state and action representations whose dimensions are independent
of production requirements and due-dates of jobs while accommodating family setups. At the same time,
an NN architecture with parameter sharing was utilized to improve the training efficiency. Extensive
experiments demonstrate that the proposed method outperforms the recent metaheuristics, rule-based, and
other RL-based methods in terms of total tardiness. Moreover, the computation time for obtaining a schedule
by our framework is shorter than those of the metaheuristics and other RL-based methods.


**INDEX TERMS** Deep reinforcement learning, unrelated parallel machine scheduling, sequence-dependent
family setups, total tardiness objective, deep _Q_ -network.



**I.** **INTRODUCTION**
As the competition among enterprises intensifies, production
scheduling becomes one of the essential decision-making
problems in modern manufacturing systems. Specifically,
manufacturers should fulfill production orders under the
sequence-dependent family setup time (SDFST) requirement
that occurs when two products belonging to different families are consecutively processed on a machine [1]. Furthermore, since customer demands frequently and unpredictably
change, it is required to deal with the variabilities associated
with the production requirements and due-dates of the products [2]. Accordingly, there is a challenge in developing a
scheduling method that is able to obtain high-quality schedules while accommodating the variabilities.


The associate editor coordinating the review of this manuscript and


approving it for publication was Hao Shen .



We focus on the unrelated parallel machine scheduling
problem (UPMSP) with SDFST, which has attracted a great
deal of attention in various domains such as semiconductor

[3]–[5], chemical [6], and food industries [7]. A UPMSP aims
to allocate each job to one of the machines where the processing time of a job on different machines is not related. This
scheduling problem is known to be NP-hard for minimizing
the total tardiness [8].
Metaheuristics have been successfully adopted for solving
UPMSPs with SDFST under due-date constraints [9]–[12].
Unfortunately, it is not guaranteed for them to find a
high-quality schedule for large-scale scheduling problems
within a specific time limit. As an alternative, manufacturers
have actively employed rule-based methods due to their short
computation time, and ease of implementation [13]. However, schedules obtained by the rule-based methods may not
be satisfactory since their decisions are made in a myopic
manner [14].



This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/
101390 VOLUME 9, 2021


B. Paeng _et al._ : Deep RL for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups



To overcome the drawbacks of rule-based methods, reinforcement learning (RL) approaches have been actively
investigated from decades ago [15], [16]. The purpose of RL
is to learn an adaptive policy that maximizes the expected
sum of cumulative rewards. In recent years, due to the
remarkable success in deep reinforcement learning (DRL)
that utilizes deep neural networks (DNNs) [17], several studies have shown promising results on the scheduling problems
in manufacturing systems [18]–[21]. Yet, there are still two
challenges to solve UPMSPs based on a DRL-based method
while addressing SDFST as well as the variabilities in terms
of production requirements and due-dates. First, the size of
the state space might become large when accommodating
due-dates and sequence-dependent setups in a state representation of a neural network, which leads to difficulties in
function approximation and DNN generalization [21]. Second, since learning complexity grows quickly as the numbers
of jobs and machines increase, it is intractable to re-train a
DNN whenever such variabilities occur in large-scale manufacturing systems.
To this end, we propose a DRL-based method for minimizing tardiness for UPMSP with SDFST to address the above
challenges. It is worth noting that learning DNN parameters
robust to changes in production requirements and due-dates
is the primary concern of our work. The contributions of this
paper are summarized as follows:

  - We design a novel state representation whose dimensionality is independent of production requirements and
due-dates of jobs while accommodating SDFST. Given
a state, an action is executed by periodically determining
a setup status of a machine and a family of a job.

  - To reduce the size of the network and increase the training efficiency, we proposed a DNN architecture in which
network parameters are shared among several hidden
layers [22]. In the experiments, the effectiveness of the
proposed state representation and parameter sharing was
demonstrated.

  - To validate the performance of the proposed method,
we tested our method on large-scale datasets. Experimental results showed that the proposed method outperforms recent metaheuristics, rule-based, and other
RL-based methods in terms of the total tardiness for
all datasets. Moreover, the computation time taken
by the proposed method was shorter than those of
other RL-based methods and metaheuristics considered. Finally, the robustness of the proposed method
was investigated by solving scheduling problems with
stochastic processing and setup time.
The rest of this paper is organized as follows. Section II
introduces the previous approaches for solving the UPMSP
with SDFST and the implications of studies on RL-based
scheduling methods. Section III defines the scheduling problem considered in this paper. Details of the proposed method
and its training algorithm are presented in Section IV. Performance comparisons with considered alternatives are carried
out in Section V. Finally, Section VI concludes this paper.



**II.** **LITERATURE REVIEW**
_A._ _UPMSPs WITH SDFST UNDER DUE-DATES_
_CONSTRAINTS_
UPMSPs with due-date related objectives are classical scheduling problems that have been comprehensively
researched for decades [23]. In particular, several studies
have addressed SDFST as main constraints [24]–[27]. To
reduce the number of setups in a scheduling problem, batch
scheduling heuristics were popularly adopted by forming
batches of jobs processed on a machine without a setup [28].
To minimize the weighted tardiness, two-level batch heuristics were proposed under the assumption of common
due-dates [29], and identical jobs [30]. Additionally, the batch
apparent tardiness cost with setup (BATCS) heuristic was
suggested by incorporating the processing time, slack time,
and family setup time [31]. For the total tardiness objective,
an improved simulated annealing heuristic was developed
through incorporating a batch-based repair method [9].
On the other hand, metaheuristics have been adopted for
solving UPMSPs with SDFST to minimize the total tardiness. They employed the batch formation technique to
restrict schedules with a small number of setup changes in
the solution space. The scheduling problem with primary
customer constraints was successfully solved by the iterated
greedy (IG) algorithm [10], which enhances an initial solution
through the iterative neighborhood selection and exhaustive
local search. More recently, Pinheiro _et. al._ [11] developed an
improved IG by designing six local search operations such as
the batch swap and job insertion. They showed competitive
results in solving scheduling problems with six machines
and 50 jobs. In [12], the artificial bee colony algorithm was
proposed through applying crossover operations to exchange
job sequences.


_B._ _REINFORCEMENT LEARNING ON UPMSPs_
Markov processes have been broadly adopted to solve optimal control problems in the presence of abrupt changes in
system dynamics [32]. To the recent, several Markov random processes were employed with advantages of modeling fuzzy systems, such as Markov chaotic systems [33]
and semi-Markov jump nonlinear systems [34]. For solving
scheduling problems, a manufacturing system is formulated
as Markov decision process (MDP) [35], and RL is utilized to
learn a policy of MDP. RL aims to train an agent by interacting with an environment that consists of everything outside
the agent. In particular, _Q_ -learning (QL) [36], one of the
representative model-free RL approaches, has been widely
adopted to solve scheduling problems. Given a state observed
from the environment that corresponds to a manufacturing
system, the agent makes scheduling decisions by predicting
the estimated value of an action called the _Q_ -value.
For solving UPMSPs by utilizing RL-based methods,
Zhang _et. al._ adopted QL to minimize the weighted tardiness

[37], [38]. They employed a linear basis function to approximate _Q_ -values for given state features indicating the status



VOLUME 9, 2021 101391


of all jobs and machines. After six heuristics are designed as
actions, QL-based adaptive rule selections outperformed individual rules. Although [37] dealt with sequence-dependent
setups, the agents in [37] were only validated on the same
scheduling problems as those for training. The method in [38]
was tested on various scheduling problems without considering setups.
Yuan _et. al._ [39], [40] addressed ready time constraints and
machine breakdown for minimizing the total tardiness and
number of tardy jobs, respectively. They adopted a tabular
method that stores _Q_ -values by exploring state-action pairs.
To represent the state in a limited size of the table, the values
of continuous attributes should be discretized into several
groups. For instance, in [40], the mean lateness of jobs was
classified as being greater than or less than zero. As alternative methods for representing the state, unsupervised learning
methods, such as self-organizing map [41], and k-means nearest neighbor [42], [43], have also been adopted for solving
scheduling problems.


_C._ _DEEP REINFORCEMENT LEARNING FOR SOLVING_
_SCHEDULING PROBLEMS_
Compared to the traditional RL methods, DRL aims to
approximate _Q_ -values from high-dimensional state features
through DNNs [17]. This helps an agent to learn a nonlinear policy from large and continuous state spaces. Besides,

[17] proposed a target network to enhance learning stability, and experience replay for reducing correlations among
state-action pairs. With the extensive application of DRL
to various decision-making problems such as energy supply
control [44], vehicle routing [45], and electricity market pricing [46], it has gradually attracted prominence in the field of
scheduling. Until recently, a lot of studies widened the scope
of DRL to scheduling problems in computer resource management [47], [48] and distributed system [49], [50]. Those
studies encoded the state as a 2D matrix of resources and
upcoming timesteps. Since they employed fully connected
networks, the matrix was flattened into a one-dimensional
vector.
In addition, DRL-based methods have been employed for
solving scheduling problems in manufacturing systems. In a
hybrid flow shop scheduling problem, an agent allocates jobs
from a given state that indicates whether the machine status is
idle or busy or finished [51]. Among various shop scheduling
problems, several researchers have investigated DRL-based
methods for minimizing the makespan in job shop scheduling
problems. Lin _et._ _al._ [52] represented machine status and
statistics of processing time in the state to infer dispatching
rules for each machine. In [20], the setup time was considered
by utilizing the setup history and setup status of all machines.
Recently, convolutional neural networks were employed to
address the states represented in 2D matrices indicating relationships between jobs, operations, and machines [19], [53].
On the other hand, a few DRL approaches were developed
for solving scheduling problems with due-date related objectives. In [54], due-dates of all waiting jobs were represented



B. Paeng _et al._ : DRL for Minimizing Tardiness in Parallel Machine Scheduling


in the state for maximizing throughput. To minimize the
total tardiness in a single machine scheduling problem, [55]
utilized the slack time, which refers to the difference between
the processing time and remaining time until due-dates of a
job. Washneck _et. al._ [18] accommodated setup constraints in
a semiconductor production scheduling problem for minimizing due-date deviations. Since their state included the setup
status of all machines and due-dates of all jobs, DNNs in

[18] should be trained again when solving new scheduling
problems whose number of jobs is different from those of
the training. Luo _et._ _al._ [21] focused on new job insertions
in the flexible job shop scheduling problem under ready
time constraints with the total tardiness objective. For a state
that consists of seven features, an action is determined by
selecting one of the heuristics. Yet, they did not consider setup
constraints.


**III.** **PROBLEM DESCRIPTION**
In this section, we describe UPMSP with SDFST considered
in this paper. There are _NJ_ jobs where the _j_ _[th]_ job is denoted
as _Jj_ . Each job belongs to one of _NF_ families from the set
_F_ = {1 _,_ 2 _, . . ., NF_ }. Each job can be processed by one of
any _NM_ machines where the _i_ _[th]_ machine is denoted as _Mi_ .
The processing time is denoted as _pi,j_ when the job _Jj_ is
performed on _Mi_ . A job is finished after being processed once
on one of the machines. Let _P_ ( _f_ ) be the total number of the
jobs that belong to family _f_, which indicates the production
requirements of family _f_ . As a result, the following equation
holds:


_NF_

   
_P_ ( _f_ ) = _NJ_ (1)

_f_ =1


At the beginning of the scheduling, due-dates of _Jj_, denoted
as _dj_, are given. Let _Gi_ denote the current setup status of _Mi_ .
_Gi_ is an element of _F_ and equivalent to _g_ if the job of family
_g_ can be processed on the machine without a setup change. If
a job of family _f_ is assigned on _Mi_ whose _Gi_ is _g_, the setup
time, denoted as _σf,g_, is incurred before the job is processed
on the machine.
The goal of the scheduling is to allocate each job to one of
the machines in order to minimize the total tardiness, denoted
as _TT_, which is defined as



where _cj_ is the completion time of _Jj_ . When there is no setup
time, the scheduling problem considered becomes equivalent
to the problem in [8], which is proven to be NP-hard. Finally,
the assumptions made in this paper are listed below.


  - At the beginning of the scheduling, all jobs are ready to
be processed, and all machines have been set up.

  - There is no machine breakdown.

  - After the setup change of a machine is finished for processing a job, the machine immediately starts to process
the job.



_TT_ =



_NJ_

- max(0 _, cj_ - _dj_ ) (2)


_j_ =1



101392 VOLUME 9, 2021


B. Paeng _et al._ : Deep RL for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups


**TABLE 1.** Five matrices and one vector that compose a state.


  - Machine can only process one job at a time.

  - The preemption is not allowed.

  - The moving time for each job is zero.



**IV.** **PROPOSED METHOD**
We propose a DRL-based scheduling method in this section
which is divided into four subsections. First, we describe
the MDP to solve UPMSP with SDFST constraint. Second,
the proposed parameter sharing architecture is introduced.
Finally, a training algorithm and the flowchart are described.


_A._ _MDP FORMULATION_
For employing DRL, the scheduling problem considered in
this paper is formulated as MDP. We denote the state, action,
and reward at timestep _k_ as _sk_, _ak_, and _rk_, respectively. After
the agent executes an action _ak_, the state transition takes place
from _sk_ to _sk_ +1. Then, reward _rk_ and next state _sk_ +1 are
observed. We denote the time interval from _sk_ until _sk_ +1 as
the period _k_ . In this paper, we model that the time spent in
each period, denoted as _T_, is constant. We define in detail the
action, state, reward, and state transition in the following.


1) ACTION
When an action is defined for each pair of a job and a
machine, the size of the action space grows quickly as the
numbers of jobs and machines increase. To reduce the size of
action space, we define an action as a tuple of a job family
and a machine setup status. The action set _A_ is defined as
follows.


_A_ = {( _f, g_ )| _f_ = 1 _, . . ., NF_ _, g_ = 1 _, . . ., NF_ } (3)


Then, we denote the feasible action set at _sk_ as _Ak_ ⊂ _A_ . An
action _ak_ = ( _fk_ _, gk_ ) indicates that a job of family _fk_ will be
assigned on a machine whose setup status is _gk_ during the
period _k_, where _ak_ ∈ _Ak_ . We note that _ak_ is feasible only if
there is a waiting job of family _fk_ and an idle machine whose
setup status is _gk_ in the period _k_ .


2) STATE
When adopting RL-based methods to solve scheduling problems, the state is usually represented by utilizing observations
on the current status of machines and jobs. If the dimension
of a state varies as production requirements and due-dates
of jobs change, the DNN is required to be re-trained whenever such changes occur. To solve the scheduling problems



(4)


where _Wk_ ( _f_ ), ⌈·⌉, and _Hw_ refer to the set of waiting
jobs that belong to family _f_ at _sk_, ceiling function, and a
positive integer smaller than max _j dj/T_, respectively. As
indicated by Eq. (4) and the vertical line on the top of
Fig. 1(a), **S** _w_ ( _f,_ 1) and **S** _w_ ( _f,_ 2 _Hw_ ) refer to the number of
waiting jobs for family _f_ where their _δj_ are smaller than
(1 - _Hw_ ) _T_ and larger than ( _Hw_ - 1) _T_, respectively. This
representation aims to restrict the number of columns
in **S** _w_ into 2 _Hw_ while capturing due-dates of all waiting
jobs in the matrix. The practical rationale behind Eq. (4)



**FIGURE 1.** The 2D matrix representations of _sk_ . Red, green, and blue
colors indicate family 1, 2, and 3, respectively. The numbers inside the
matrices correspond to the values in _sk_ .


without re-training DNNs even when such changes occur,
we propose a family-based state representation whose dimensionality is independent of the production requirements and
due-dates of jobs.
The proposed state consists of five 2D matrices and one
vector. Table. 1 describes them in terms of notations, names,
and dimension. To help the understanding of _sk_, we present
an example of _sk_ in a scheduling problem with three families,
as depicted in Fig. 1. Next, we define the details of _sk_ . It is
noted that the index _k_ is omitted in each of the five matrices
and vector for the sake of conciseness.

  - **S** _w_ : To accommodate the due-dates of waiting jobs,
the value of the _f_ _[th]_ row in this matrix refers to the number of waiting jobs for family _f_ where their due-dates
belong to one of periods. For separately counting each
waiting job _Jj_ with respect to their remaining time until
due-dates, denoted as _δj_ = _dj_   - _kT_, **S** _w_ ( _f, n_ ) is defined
as follows.



**S** _w_ =








|{ _Jj_ ∈ _Wk_ ( _f_ ) | _δj_ ≤ (1 − _Hw_ ) _T_ }| _n_ = 1
|{ _Jj_ ∈ _Wk_ ( _f_ ) | _δj_ _>_ ( _Hw_ - 1) _T_ }| _n_ = 2 _Hw_
|{ _Jj_ ∈ _Wk_ ( _f_ ) | - _δTj_ - = _n_ - _Hw_ }| otherwise







VOLUME 9, 2021 101393


is that jobs whose due-dates are quite far from the current
period have a smaller impact on executing an appropriate
action than the other jobs. In Fig. 1(a), **S** _w_ (1 _, Hw_ ) is
equal to 6, which indicates that there are six waiting jobs
whose family is 1 and  - _δTj_  - = 0.

- **S** _p_ : This matrix contains the number of in-progress jobs
for each family of which their remaining processing time
is included in one of periods. In-progress jobs refer to the
ones that are currently being processed on a machine.
Since the tardiness of in-progress and finished jobs are
already determined, due-dates of in-progress jobs are
not relevant to minimizing the tardiness. Meanwhile,
the remaining processing time of in-progress jobs affects
the completion time of the waiting jobs that will be
assigned after in-progress jobs are completed. Therefore, each job _Jj_ that belongs to _Pk_ ( _f_ ) is classified
according to the remaining processing time, denoted as
_ρj_, where _Pk_ ( _f_ ) is the set of in-progress jobs of the
family _f_ at _sk_ . Then, we define **S** _p_ ( _f, n_ ) as below.



**S** _p_ =




|{ _Jj_ ∈ _Pk_ ( _f_ ) | ( _n_  - 1) _T_ _< ρj_ ≤ _nT_ }| _n < Hp_
|{ _Jj_ ∈ _Pk_ ( _f_ ) | ( _n_  - 1) _T_ _< ρj_ }| _n_ = _Hp_

(5)



where _Hp_ is a positive integer less than max _j ρj/T_ . We
note that the dimension of the column in **S** _p_ is constrained to _Hp_ in a similar way to Eq. (4). For example
in Fig. 1(b), **S** _p_ (2 _,_ 1) is equal to 1, which indicates
that there exists one job satisfying the following two
conditions: the job belongs to _Pk_ (2), and its remaining
processing time is shorter than _T_ .

- **S** _s_ : To capture the family setup time for predicting
the tardiness that will be incurred after _sk_, **S** _s_ ( _f, g_ )
represents the required setup time to process a job
of family _f_ on a machine whose setup status is _g_ .
Specifically, **S** _s_ ( _f, g_ ) is defined as follow.



B. Paeng _et al._ : DRL for Minimizing Tardiness in Parallel Machine Scheduling


**Algorithm 1** State Transition by an Action


_NF_
**Input:** _ak_ = ( _fk_ _, gk_ ), _Wk_ = - _Wk_ ( _f_ [′] )

_f_ [′] =1

**Output:** State _sk_ +1, reward _rk_, _Wk_ +1
1: _M_ ←{ _Mi_ | _Gi_ = _gk_ }, _i_ = 1 _, . . ., NM_
2: _M_ ∗ ← arg min( [�] _j_ [′] _[ p][i][,][j]_ [′][), where] _[ J][j]_ [′] [∈] _[W][k]_ [(] _[f][k]_ [)]
_Mi_ ∈ _M_

3: **while** _Wk_ ̸= ∅ and min _i_ ′ _Ei_ ′ _<_ ( _k_ + 1) _T_ **do**

4: _Mi_ ← arg min _i_ ′ _Ei_ ′

5: _g_ ← _Gi_
6: **if** _M_ ∗ = _Mi_ **then**

7: _f_ ← _fk_
8: **else**

9: _f_ ← arg min _σf_ ′ _,g_, where | _Wk_ ( _f_ [′] )| _>_ 0
_f_ [′]

10: **end if**

11: _Jj_ ← arg min _dj_ ′
_Jj_ ′ ∈ _Wk_ ( _f_ )

12: Assign _Jj_ on _Mi_
13: _Ei_ ← _Ei_ + _pi,j_ + _σf,g_
14: _Wk_ ← _Wk_ \ { _Jj_ }

15: **end while**
16: Obtain transited state _sk_ +1
17: Calculate _rk_ from Eq. (8)


otherwise 0. Figs. 1(e) and (f) provide the following four
information: _k_ = 5, _r_ 4 = −1 _._ 7, _a_ 4 = (1 _,_ 2), and _s_ 5 is
not terminal.


3) REWARD
The reward proposed in this paper is motivated by [37] in
the sense that _rk_ is calculated by considering job delays
which occurred only during the period _k_ . However, since
we assumed that the time spent in each period is always
equal to _T_, the reward in [37] is redefined in this paper to
accommodate such assumption by using the following clip
function.


_λk_ ( _t_ ) = max( _kT_ _,_ min(( _k_ + 1) _T_ _, t_ )) (7)


Then, _rk_ is defined as follows.



**S** _s_ ( _f, g_ ) =




_σf,g_ ( _f, g_ ) ∈ _Ak_
(6)
_σ_ max otherwise



where _σ_ max refers to the maximum setup time. If ( _f, g_ ) ∈ _/_
_Ak_, **S** _s_ ( _f, g_ ) is set to _σ_ max to consider the worst case.

- **S** _u_, **S** _a_, and _S_ [⃗] _f_ : These are devised to incorporate the history of the job, machine, and agent status until _sk_, which
has been known to be effective for solving scheduling
problems based on DRL [20]. First, **S** _u_ ( _f,_ 1) and **S** _u_ ( _f,_ 2)
respectively refer to the amounts of processing and
setup time until _sk_ that have been spent for processing
jobs of the family _f_ . **S** _u_ ( _f,_ 3) is the number of finished
jobs whose family is _f_ . Next, **S** _a_ represents the last
action _ak_ −1. By using one-hot encoding [56], **S** _a_ (· _,_ 1)
and **S** _a_ (· _,_ 2) denote _NF_ -dimensional vectors that indicate
the family of a job and the setup status of a machine,
respectively. Finally, _S_ [⃗] _f_ consists of the three historic
features that cannot be grouped into a specific family.
_S_ ⃗ _f_ (1), _S_ ⃗ _f_ (2), and _S_ ⃗ _f_ (3) are respectively equal to _rk_ −1,
_k_, and a binary value that is set to 1 if _sk_ is terminal,



_rk_ =



_NJ_


 - max(0 _, λk_ ( _cj_ ) − _λk_ ( _dj_ )) (8)

_j_ =1



Eq. (8) states that the reward is equivalent to the negative
sum of the job tardiness clipped by Eq. (7). When computing
_rk_, we assumed that _cj_ is set to be infinite if _Jj_ is waiting until
_sk_ +1. As a result, the total sum of rewards is equal to _TT_,
which was proven in [37].


4) STATE TRANSITIONS
After executing _ak_, the state transition from _sk_ to _sk_ +1 occurs.
Algorithm 1 describes the procedure for the state transition.
Let _Ei_ be the time when the current job performed on _Mi_ will
be finished.



101394 VOLUME 9, 2021


B. Paeng _et al._ : Deep RL for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups


**FIGURE 2.** The proposed DNN architecture with parameter sharing. The red, green, blue, and brown colors represent different
families. The sky color refers to the hidden layers and output layer.


assigned on a machine whose setup status is 1. By following
lines 2 and 11 in Algorithm 1, _J_ 5 is assigned on _M_ 2, and
_σ_ 2 _,_ 1 is incurred before processing the job. Meanwhile, _J_ 3 and
_J_ 4, which respectively belong to family 1, are consecutively
allocated to _M_ 1 whose _G_ 1 is 1. At the end of the period _k_,
_sk_ +1 and _rk_ are obtained as the consequences of the state
transition. After executing _ak_ +1 = (1 _,_ 3), the job _J_ 7 of family
3 is allocated on _M_ 1 whose setup status is 1. In summary,
_J_ 3, _J_ 4, and _J_ 5 are scheduled according to _ak_ and _J_ 6 and _J_ 7
according to _ak_ +1.



**FIGURE 3.** An example of a schedule obtained from state transitions. Red,
green, and blue colors indicate family 1, 2, and 3, respectively.


In line 1, we obtain a machine set _M_ that consists of the
machines whose setup status is _gk_ . Among _M_, we select the
machine _M_ ∗ where the sum of processing time for the jobs
in _Wk_ ( _fk_ ) is the shortest (line 2). Lines 3–15 continue until
satisfying the following two conditions: there is at least one
waiting job, and the minimum of _Ei_ among all machines do
not exceed the end time of the current period _k_ . In line 4,
the machine _Mi_ whose _Ei_ is the smallest among all machines
is selected. Then, the setup status of the machine, called _g_,
is obtained (line 5). In lines 6–10, we determine the family of
a next job that will be performed on _Mi_, denoted as _f_ . If _M_ ∗ is
equal to _Mi_, _f_ is set to _fk_ by following _ak_ (line 7). Otherwise, _f_
is selected to minimize the setup time incurred on _Mi_ (Line 9).
Among waiting jobs whose family is _f_, the job _Jj_ with the
earliest due-date is selected (line 11). After _Jj_ is assigned on
_Mi_ (line 12), both _Ei_ and _Wk_ are updated (lines 13 and 14).
Finally, _sk_ +1, _Ak_ +1, and _rk_ are obtained (lines 16 and 17).


5) EXAMPLE
Fig. 3 depicts a schedule that is built during the periods _k_
and _k_ + 1. At _sk_, _M_ 1 and _M_ 2 are respectively performing _J_ 1
and _J_ 2, and the jobs _J_ 3 to _J_ 7 are waiting for being assigned.
Since _ak_ is equal to (2 _,_ 1), a job of the family 2 should be



_B._ _DNN ARCHITECTURE_
A deep _Q_ -network (DQN) was employed to estimate a _Q_ value given a state [17]. DQN takes a state _s_ as an input
and outputs _Q_ -values for all possible actions, called _Qθ_ ( _s, a_ ),
where _θ_ is the network parameters and _a_ ∈ _A_ . Fig. 2.
depicts the proposed fully-connected network architecture
with parameter sharing that has been adopted for solving single lot-sizing [57], and job scheduling problems in computing
platforms [22].
The input of DQN is equal to a matrix that is constructed
by concatenating the five matrices **S** _w_, **S** _p_, **S** _s_, **S** _u_, and **S** _a_ .
Each row vector of the input is connected to a block that
consists of several hidden layers. To reduce the network
parameter size and increase the training efficiency, the parameters for each block are set to be the same. The last hidden
layer is composed by concatenating the values in the last
layers of _NF_ blocks and _S_ [⃗] _f_ . Finally, the number of nodes
in the output layer is equal to the number of all possible
actions. The ReLU function [58] was adopted as an activation
function except for the output layer to represent negative
_Q_ -values.


_C._ _TRAINING DQN_
Algorithm 2 describes the training procedure of the proposed DQN. Given a scheduling problem for training DQN,
a scheduling process is repeated until there is no waiting job
(lines 3–16). We refer to the completion of one scheduling



VOLUME 9, 2021 101395


**FIGURE 4.** A flowchart of the proposed algorithms.


**Algorithm 2** DQN Training Procedure
**Input:** Scheduling problem
**Output:** _Q_ -network

1: **Initialization** : Set network _Qθ_ with random weight _θ_,
target network _Qθ_ ˆ with _θ_ [ˆ] = _θ_, and replay buffer _B_ to size
_NB_ .
2: **for** _e_ = 1, 2, ..., _NE_ **do**
3: _k_ ← 0

4: Initialize _Fj, dj_ of _NJ_ jobs, and status of _NM_ machines.
5: Observe _sk_ and _Wk_
6: **while** _Wk_ ̸= ∅ **do**
7: With probability _ε_ select _ak_ randomly from _Ak_
8: otherwise _ak_ ← arg max _Q_ ( _sk_ _, a_ )
_a_ ∈ _Ak_

9: Get _sk_ +1, _Ak_ +1, _rk_ _, Wk_ +1 from **Algorithm 1**
10: Store transition _(sk_ _, ak_ _, rk_ _, sk_ +1 _)_ in _B_
11: Sample _NTR_ transitions ( _su, au, ru, su_ +1) ∈ _B_
12: Calculate loss _L_ from (9)-(11)
13: Perform a gradient descent step on _L_ w.r.t. _θ_

14: _k_ ← _k_ + 1

15: **end while**
16: Synchronize _θ_ [ˆ] to _θ_ at every _NU_ episodes

17: **end for**

18: **return** _Q_ -network


process as an episode, and _e_ indicates the index of the episode
currently being performed. The scheduling processes continue until _e_ reaches the number of training episodes, denoted
as _NE_ .
At the start of an episode, _k_ is set to 0, and _NJ_ jobs
and _NM_ machines are initialized (lines 3 and 4). In line 5,
the agent initially observes the _sk_ with _Wk_ . For each timestep
_k_, the agent selects _ak_ from the _ε_ -greedy policy [36] presented
in lines 7 and 8, where _ε_ ∈ [0 _,_ 1] is a probability to select
an random action. After the state transition takes place from



B. Paeng _et al._ : DRL for Minimizing Tardiness in Parallel Machine Scheduling


**TABLE 2.** Dataset.


_sk_ to _sk_ +1 (line 9), the transition, represented as a quadruple
of state, action, reward, and next state, is stored in replay
buffer (line 10). The replay buffer and its size are denoted
as _B_ and _NB_, respectively. If the number of stored transitions
in _B_ exceeds _NB_, the oldest ones are removed. Line 11 indicates that _NTR_ transitions are sampled to train DQN, where
_NTR_ is the number of sampled transitions. Given a transition
( _su, au, ru, su_ +1), the temporal difference error, called _ηu_, are
calculated by the prediction _Qθ_ ( _s, a_ ) and target _Q_ -value [17]
as follows.


_ηu_ = _ru_ + _γ_ _a_ [′] max∈ _Au_ +1 _Qθ_ ˆ( _su_ +1 _, a_ [′] ) − _Qθ_ ( _su, au_ ) (9)


where _γ_ and _θ_ [ˆ] respectively indicate the discount factor

[36] and the parameters of a target DQN which has the
same network architecture as DQN for training. In Eq. (9),
_ru_ + _γ_ max _a_ ′ _Qθ_ ˆ( _su_ +1 _, a_ [′] ) indicates a target _Q_ -value which
is set to _ru_ when _su_ +1 is terminal. We denote the temporal
difference loss from sampled transitions as _L_ ( _θ_ ), which is
defined as follows.



1
_L_ ( _θ_ ) =
_NTR_



_NTR_

- _h_ ( _ηu_ ) (10)


_u_ =1



101396 VOLUME 9, 2021


B. Paeng _et al._ : Deep RL for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups


where _h_ is the loss function, defined as below.



1
if _η_ ≤ [1]
2 _[η]_ [2] 2



1
2 [(][|] _[η]_ [| −] [(] 2 [1]



(11)

[1] otherwise.

2 [)][2][)]



_h_ ( _η_ ) =











2 _[,]_



In Eq. (11), we adopted Huber loss [59] instead of
mean-squared error for further enhancing the stability of
DQN training [20]. By calculating _L_ ( _θ_ ), the network parameter _θ_ is updated (lines 12 and 13). Line 16 shows that the target
network parameter _θ_ [ˆ] is periodically replaced to the _θ_ [17].
Finally, the trained DQN is acquired at the end of _NE_ episodes
(line 18). Fig. 4 depicts the overall flowchart of the proposed
algorithms.
After DQN training, a test procedure is implemented to
solve the scheduling problems whose production requirements and due-dates change from those of the training problems. During the test procedure, random actions (line 8 in
Algorithm 2) are not executed any more. The rest of the procedure is identical to Algorithm 2 except the lines 10–13 and
16 that are required to train a DQN.


**V.** **EXPERIMENTS**
_A._ _DATASETS_
We prepared 8 datasets that simulated the semiconductor wafer preparation facilities in South Korea. Table. 2
presents _NM_, _NJ_, _NF_, and due-date tightness, denoted as _τ_,
for the scheduling problems in each dataset. Except for _τ_,
datasets 1 and 3 are equivalent to datasets 2 and 4, respectively. Moreover, _NM_ and _NJ_ of datasets 5–8 are 2.5 times
larger than those of datasets 1–4, respectively.
Each dataset has 330 different scheduling problems which
are divided into 300 problems for training the proposed DQN,
and the other problems for validating the trained DQN. Fig. 5
depicts the distributions of production requirements for each
family. In particular, across all datasets, production requirements were perturbed by at least 37%.
For each scheduling problem, _dj_ of _NJ_ jobs were set to be
uniformly distributed between _L_ (0 _._ 5 - _τ_ ) and _L_ (1 _._ 5 - _τ_ ),
where _L_ indicates an expected makespan adopted in several
studies [9], [11], [30]. To simulate the real-world scenario,
each machine performs a job at the beginning of the scheduling, where its remaining processing time was randomly
generated.


_B._ _EXPERIMENTAL SETTINGS_
Experiments were conducted on a Xeon E5 2.2-GHz PC
with 126-GB memory. When employing DRL-based methods, choosing appropriate values of hyperparameters plays a
crucial role in the performance of DNN. Since it is challenging to determine optimal values of hyperparameters due to
their huge search space, we implemented the random search

[60], and found the values that achieved the best performance.
For performing a gradient descent step, we adopted
RMSProp optimizer [61] where the learning rate is set to
2 _._ 5 × 10 [−][3] . In the _ε_ -greedy policy, the initial value of



**FIGURE 5.** The ranges of variability in production requirements for
8 datasets.


**FIGURE 6.** Mean tardiness results with changes in _Hw_ and _Hp_ on
dataset 1. (a) _Hw_ with _Hp_ **=** 5. (b) _Hp_ with _Hw_ **=** 6.


_ε_ is set to 0.2. The value decays linearly to zero until _e_
reaches 0 _._ 9 × _NE_ . Besides, _NB_, _NU_, _NTR_, and _γ_ is set
to 10 [5], 50, 64, and 1, respectively. Finally, we set _NE_ =
10 [5], _T_ = 32 _[p]_ [¯] [in] [datasets] [1–4,] [and] _[N][E]_ = 1 _._ 5 × 10 [4],
_T_ = 2 [1] _[p]_ [¯] [in] [datasets 5–8,] [respectively,] [where] _[p]_ [¯] [is] [the] [aver-]

age processing time over all pairs of jobs and machines.
Each shared block consists of three hidden layers where
the numbers of nodes in the first, second, and third layers
are 64, 32, and 16, respectively. For each dataset, after the
trained DQNs were stored when _e_ is equal to 0 _._ 91 × _NE_,
0 _._ 92 × _NE_, ..., 0 _._ 99 × _NE_, and _NE_, respectively, they were
used for performance comparisons.
Figs. 6(a) and (b) illustrate the mean tardiness results (in
hours) on dataset 1 by varying _Hw_ and _Hp_, respectively. In
the experiments, the mean tardiness is calculated by dividing
_TT_ to _NJ_ . We note that the values of _Hw_ and _Hp_ were chosen
in the range described in Eqs. (4) and (5), respectively. In
Fig. 6(a), _TT_ significantly decreased until _Hw_ reaches 6,
which reveals the validity of **S** _w_ . On the other hand, the performance changes were negligible when _Hw_ exceeds 6. This
implies that the advantage of classifying waiting jobs in terms
of their due-dates is diminished due to the increase in the
dimension of **S** _w_ . As shown in Fig. 6(b), the performance
improvement achieved by varying _Hp_ was less significant
than varying _Hw_ . This can be attributed to the fact that **S** _w_
contains more observations than **S** _p_ since the number of
waiting jobs is larger than those of in-progress jobs in most
periods. As a result, the rest of the experiments were carried
out with the best values of _Hw_ and _Hp_, which were 6 and 5,
respectively.



VOLUME 9, 2021 101397


B. Paeng _et al._ : DRL for Minimizing Tardiness in Parallel Machine Scheduling


**TABLE 3.** Mean tardiness results of the proposed method and the other methods. Bold marks indicate the best results among all methods for each
dataset.



_C._ _PERFORMANCE COMPARISON_


In order to show the effectiveness of the proposed method,
we compared performances of IG in [11] and two RL-based
methods, which are two-phase DQN (TPDQN) [18], and QL
method with linear basis functions (LBF-Q) [37], respectively. The parameters of IG were the same as those in [11].
Since the production scheduling in the semiconductor manufacturing systems is usually carried out on an hourly basis

[14], IG was terminated after an hour. For LBF-Q and
TPDQN, ten models were stored as described in Section V-B
for performance comparisons, respectively. The rest of the
hyperparameters were the same as those in [18], [37],
respectively.
Furthermore, we made comparisons between our method
and four rule-based methods: BATCS [31], shortest setup
time with earliest due date (SSTEDD), least slack remaining (LSR) [13], COVERT [62]. LSR and COVERT are
widely adopted to minimize due-date related objectives,
while BATCS is effective when solving scheduling problems
with SDFST. In particular, SSTEDD selects the jobs which
require the shortest setup time and decides a job with the
earliest due-date among those jobs.
Table. 3 presents the mean tardiness results (in hours) of
ours and the other methods. Among the rule-based methods,
SSTEDD outperformed the other methods in all datasets.
Meanwhile, LSR and COVERT yielded 3.2 times longer
_TT_ than the other rule-based methods in the best case and
11.3 times in the worst case. It can be said that addressing sequence-dependent setups is crucial for minimizing
_TT_ of the scheduling problems considered. It was observed
that _TT_ achieved by LBF-Q and TPDQN was longer than
SSTEDD for all datasets. This may be due to the fact that
the family setups were not accommodated in their state
and action representations. Although the performances of IG
were better than those of the rule-based and other RL-based
methods for all datasets, _TT_ achieved by IG was 13% to
55% longer than those of the proposed method. Based on
these results, the proposed method appears to be effective
for solving scheduling problems even when the production
requirements and due-dates are changed from those of the
training.



**TABLE 4.** Computation time results (in seconds) of SSTEDD, IG, LBF-Q,
TPDQN, and the proposed method.


Table. 4 presents the average computation time taken by
our method, TPDQN, LBF-Q, IG, and SSTEDD. We only
presented the results of SSTEDD whose average computation
time is the shortest among the four rule-based methods. The
computation time results of LBF-Q and TPDQN were longer
than those of the proposed method for all datasets. This might
be related to the fact that _Q_ -values are computed by the
proposed method at each period, different from LBF-Q and
TPDQN that compute _Q_ -values whenever allocating a job to a
machine. Moreover, the ratio of average computation time for
the datasets 5–8 to that for datasets 1–4 was 5.56 for LBF-Q,
6.23 for TPDQN, and 3.98 for the proposed method, respectively. This observation can be attributed to the fact that the
number of parameters for the proposed DQN is independent
of _NM_ and _NJ_, different from LBF-Q and TPDQN.
Compared to the best rule, the computation time of the
proposed method was increased by 8.8 to 12 times for the
datasets 1–4, and 6.7 to 7.9 times for the datasets 5–8, respectively. Nevertheless, the results demonstrate that the proposed
method built a schedule less than 20s for all datasets. Different from metaheuristics and rule-based methods, the proposed DRL-based method can quickly obtain a new schedule
by using the trained DQN, which suggests the viability of
the proposed method in terms of the computation time for
real-world manufacturing systems with parallel machines.
To examine the robustness of the proposed method when
both processing and setup time is stochastic, we carried out



101398 VOLUME 9, 2021


B. Paeng _et al._ : Deep RL for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups


**FIGURE 7.** Mean tardiness results of PABS, FBS-1D, and ours across all datasets.



**TABLE 5.** The mean tardiness (in hours) of SSTEDD, IG, LBF-Q, and the
proposed method on datasets 1 and 5.


additional performance comparisons. Specifically, both _pi,j_
and _σf,g_ were not known in advance and set to be uniformly
distributed between [0 _._ 8 _pi,j,_ 1 _._ 2 _pi,j_ ] and [0 _._ 8 _σf,g,_ 1 _._ 2 _σf,g_ ],
respectively. Each test scheduling problems was solved
30 times with different random seeds.
Table. 5 shows the average and standard deviation of the
mean tardiness (in hours) for SSTEDD, IG, LBF-Q, and our
method. For datasets 1 and 5, both the average and standard
deviation of _TT_ yielded by the proposed method were the
lowest among all methods. Based on these results, the proposed method seems to be robust even when processing and
setup time are stochastic.
We further analyzed the effectiveness of the proposed
state representation and parameter sharing architecture. To
solely investigate the performance improvements achieved
by the state representation and parameter sharing, we compared the proposed method with two modified baseline
methods. First, we adopted production attributes-based state
representation (PABS) proposed in [37]. Next, the proposed family-based state representation was utilized as
one-dimensional vector (FBS-1D) by flattening and concatenating **S** _w_, **S** _p_, **S** _s_, **S** _u_, **S** _a_, and _S_ [⃗] _f_ . For PABS and FBS-1D,
we utilized the fully-connected network with three hidden
layers where the number of nodes were the same as mentioned in Section. IV-B. The rest of the details are equivalent
to the proposed method. Note that PABS and FBS-1D were
the same except for the state representation.
Fig. 7 highlights the mean tardiness results (in hours) of
PABS, FBS-1D, and ours in datasets 1–8. For all datasets,
FBS-1D consistently outperformed PABS with respect to
_TT_ . Specifically, _TT_ achieved by FBS-1D was 70% lower
than those of PABS in datasets 1–4 and 34% lower for
datasets 5–8, respectively, which demonstrate the superiority
of the proposed state representation. Furthermore, compared



to FBS-1D, the performances of the proposed method were
30% better for datasets 1–4 and 47% in datasets 5–8, respectively. Based on the above observations, parameter sharing
appears to be more efficient for solving scheduling problems
with large numbers of jobs and machines.


**VI.** **CONCLUSION**
In this paper, we proposed a DRL-based method for solving
UPMSPs with SDFST constraint to minimize the total tardiness. To cope with the variabilities in production requirements and due-dates while addressing SDFST, we proposed
a novel state representation whose dimension is invariant to
such variabilities. Furthermore, we suggested the parameter
sharing architecture to learn DQN parameters effectively and
reduce the parameter size. As a result, the trained DQN was
able to quickly solve unseen scheduling problems whose production requirements and due-dates are different from those
considered in training.
To examine the performance of the trained DQN, the proposed method was compared to IG, four rule-based methods,
LBF-Q, and TPDQN. The experimental results demonstrated
that our method outperformed the existing methods in terms
of the total tardiness for all datasets. Moreover, the computation time of the proposed method was shorter than those of
IG and two RL-based methods. Through further experiments,
it was verified that both the proposed state representation
and parameter sharing architecture have contributed to the
performance improvements.
Yet, the proposed DQN poses a limitation when the number
of families changes since a re-training procedure is required.
To address such limitation, we plan to develop the state and
action whose dimensions are independent of the number of
families. Furthermore, some assumptions made in this paper
related to the ready time of jobs and machine breakdown will
be relaxed in the future work. Finally, the network architecture can be improved by utilizing other deep learning models,
such as recurrent DNNs, and convolutional neural networks.


**REFERENCES**


[1] A. Allahverdi, ‘‘The third comprehensive survey on scheduling problems
with setup times/costs,’’ _Eur._ _J._ _Oper._ _Res._, vol. 246, no. 2, pp. 345–378,
2015.



VOLUME 9, 2021 101399


[2] Y.-R. Shiue, K.-C. Lee, and C.-T. Su, ‘‘A reinforcement learning approach
to dynamic scheduling in a product-mix flexibility environment,’’ _IEEE_
_Access_, vol. 8, pp. 106542–106553, 2020.

[3] T. Yang, Y.-F. Wen, Z.-R. Hsieh, and J. Zhang, ‘‘A lean production system
design for semiconductor crystal-ingot pulling manufacturing using hybrid
taguchi method and simulation optimization,’’ _Assem._ _Autom._, vol. 40,
no. 3, pp. 433–445, Jan. 2020.

[4] W. L. Pearn, S. H. Chung, and M. H. Yang, ‘‘The wafer probing scheduling problem (WPSP),’’ _J._ _Oper._ _Res._ _Soc._, vol. 53, no. 8, pp. 864–874,
Aug. 2002.

[5] L. Mönch, J. W. Fowler, and S. J. Mason, ‘‘Semiconductor manufacturing
process description,’’ in _Production_ _Planning_ _and_ _Control_ _for_ _Semicon-_
_ductor Wafer Fabrication Facilities_ . New York, NY, USA: Springer, 2013,
pp. 11–28.

[6] L. Shen, L. Mönch, and U. Buscher, ‘‘An iterative approach for the serial
batching problem with parallel machines and job families,’’ _Ann._ _Oper._
_Res._, vol. 206, no. 1, pp. 425–448, 2013.

[7] C. A. Sáenz-Alanís, V. D. Jobish, M. A. Salazar-Aguilar, and V. Boyer,
‘‘A parallel machine batch scheduling problem in a brewing company,’’
_Int. J. Adv. Manuf. Technol._, vol. 87, nos. 1–4, pp. 65–75, 2016.

[8] J. Du and J. Y.-T. Leung, ‘‘Minimizing total tardiness on one
machine is NP-hard,’’ _Math._ _Oper._ _Res._, vol. 15, no. 3, pp. 483–495,
Aug. 1990.

[9] J.-F. Chen, ‘‘Scheduling on unrelated parallel machines with sequenceand machine-dependent setup times and due-date constraints,’’ _Int. J. Adv._
_Manuf. Technol._, vol. 44, nos. 11–12, pp. 1204–1212, Oct. 2009.

[10] S.-W. Lin, C.-C. Lu, and K.-C. Ying, ‘‘Minimization of total tardiness on
unrelated parallel machines with sequence- and machine-dependent setup
times under due date constraints,’’ _Int._ _J._ _Adv._ _Manuf._ _Technol._, vol. 53,
nos. 1–4, pp. 353–361, Mar. 2011.

[11] J. C. S. N. Pinheiro, J. E. C. Arroyo, and L. B. Fialho, ‘‘Scheduling
unrelated parallel machines with family setups and resource constraints
to minimize total tardiness,’’ in _Proc. Genetic Evol. Comput. Conf. Com-_
_panion_, Jul. 2020, pp. 1409–1417.

[12] K.-C. Ying and S.-W. Lin, ‘‘Unrelated parallel machine scheduling with
sequence-and machine-dependent setup times and due date constraints,’’
_Int. J. Innov. Comput., Inf. Control_, vol. 8, no. 5, 2012, pp. 3279–3297.

[13] Y. H. Lee, K. Bhaskaran, and M. Pinedo, ‘‘A heuristic to minimize the total
weighted tardiness with sequence-dependent setups,’’ _IIE Trans._, vol. 29,
no. 1, pp. 45–52, Jan. 1997.

[14] J. Lim, M.-J. Chae, Y. Yang, I.-B. Park, J. Lee, and J. Park, ‘‘Fast scheduling
of semiconductor manufacturing facilities using case-based reasoning,’’
_IEEE Trans. Semicond. Manuf._, vol. 29, no. 1, pp. 22–32, Feb. 2016.

[15] W. Zhang and T. G. Dietterich, ‘‘A reinforcement learning approach to jobshop scheduling,’’ in _Proc. IJCAI_, vol. 95, 1995, pp. 1114–1120.

[16] T. Gabel and M. Riedmiller, ‘‘Adaptive reactive job-shop scheduling with
reinforcement learning agents,’’ _Int. J. Inf. Technol. Intell. Comput._, vol. 24,
no. 4, pp. 14–18, 2008.

[17] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness,
M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski,
and S. Petersen, ‘‘Human-level control through deep reinforcement
learning,’’ _Nature_, vol. 518, pp. 529–533, 2015.

[18] B. Waschneck, A. Reichstaller, L. Belzner, T. Altenmuller, T. Bauernhansl,
A. Knapp, and A. Kyek, ‘‘Deep reinforcement learning for semiconductor
production scheduling,’’ in _Proc._ _29th_ _Annu._ _SEMI_ _Adv._ _Semiconductor_
_Manuf. Conf. (ASMC)_, Apr. 2018, pp. 301–306.

[19] C.-L. Liu, C.-C. Chang, and C.-J. Tseng, ‘‘Actor-critic deep reinforcement
learning for solving job shop scheduling problems,’’ _IEEE Access_, vol. 8,
pp. 71752–71762, 2020.

[20] I.-B. Park, J. Huh, J. Kim, and J. Park, ‘‘A reinforcement learning approach
to robust scheduling of semiconductor manufacturing facilities,’’ _IEEE_
_Trans. Autom. Sci. Eng._, vol. 17, no. 3, pp. 1420–1431, Jul. 2020.

[21] S. Luo, ‘‘Dynamic scheduling for flexible job shop with new job insertions
by deep reinforcement learning,’’ _Appl. Soft Comput._, vol. 91, Jun. 2020,
Art. no. 106208.

[22] D. Zhang, D. Dai, Y. He, F. S. Bao, and B. Xie, ‘‘RLScheduler: An automated HPC batch job scheduler using reinforcement learning,’’ 2019,
_arXiv:1910.08925_ . [Online]. Available: http://arxiv.org/abs/1910.08925

[23] A. Kramer and A. Subramanian, ‘‘A unified heuristic and an annotated
bibliography for a large class of earliness–tardiness scheduling problems,’’
_J. Scheduling_, vol. 22, no. 1, pp. 21–57, Feb. 2019.

[24] M. A. Bozorgirad and R. Logendran, ‘‘Sequence-dependent group scheduling problem on unrelated-parallel machines,’’ _Expert Syst. Appl._, vol. 39,
no. 10, pp. 9021–9030, 2012.



B. Paeng _et al._ : DRL for Minimizing Tardiness in Parallel Machine Scheduling


[25] O. Shahvari and R. Logendran, ‘‘An enhanced tabu search algorithm to
minimize a bi-criteria objective in batching and scheduling problems on
unrelated-parallel machines with desired lower bounds on batch sizes,’’
_Comput. Oper. Res._, vol. 77, pp. 154–176, Jan. 2017.

[26] A. Ekici, M. Elyasi, O. Ö. Özener, and M. B. Sarıkaya, ‘‘An application of
unrelated parallel machine scheduling with sequence-dependent setups at
vestel electronics,’’ _Comput. Oper. Res._, vol. 111, pp. 130–140, Nov. 2019.

[27] J. R. Zeidi and S. MohammadHosseini, ‘‘Scheduling unrelated parallel
machines with sequence-dependent setup times,’’ _Int. J. Adv. Manuf. Tech-_
_nol._, vol. 81, nos. 9–12, pp. 1487–1496, 2015.

[28] C. N. Potts and M. Y. Kovalyov, ‘‘Scheduling with batching: A review,’’
_Eur. J. Oper. Res._, vol. 120, no. 2, pp. 228–249, 2000.

[29] R. H. Suriyaarachchi and A. Wirth, ‘‘Earliness/tardiness scheduling with a
common due date and family setups,’’ _Comput. Ind. Eng._, vol. 47, nos. 2–3,
pp. 275–288, Nov. 2004.

[30] D.-W. Kim, D.-G. Na, and F. F. Chen, ‘‘Unrelated parallel machine
scheduling with setup times and a total weighted tardiness objective,’’
_Robot. Comput.-Integr. Manuf._, vol. 19, nos. 1–2, pp. 173–181, Feb. 2003.

[31] S. J. Mason, J. W. Fowler, and W. M. Carlyle, ‘‘A modified shifting
bottleneck heuristic for minimizing total weighted tardiness in complex
job shops,’’ _J. Scheduling_, vol. 5, no. 3, pp. 247–262, 2002.

[32] O. L. V. Costa and M. D. Fragoso, ‘‘Discrete-time LQ-optimal control
problems for infinite Markov jump parameter systems,’’ _IEEE_ _Trans._
_Autom. Control_, vol. 40, no. 12, pp. 2076–2088, Dec. 1995.

[33] J. Wang, ‘‘ _H_ ∞ synchronization for fuzzy Markov jump chaotic systems
with piecewise-constant transition probabilities subject to PDT switching rule,’’ _IEEE_ _Trans._ _Fuzzy_ _Syst._, early access, Jul. 29, 2020, doi:
[10.1109/TFUZZ.2020.3012761.](http://dx.doi.org/10.1109/TFUZZ.2020.3012761)

[34] H. Shen, M. Dai, Y. Luo, J. Cao, and M. Chadli, ‘‘Fault-tolerant fuzzy control for semi-Markov jump nonlinear systems subject to incomplete SMK
and actuator failures,’’ _IEEE Trans. Fuzzy Syst._, early access, Jul. 24, 2020,
[doi: 10.1109/TFUZZ.2020.3011760.](http://dx.doi.org/10.1109/TFUZZ.2020.3011760)

[35] M. L. Puterman, _Markov Decision Processes: Discrete Stochastic Dynamic_
_Programming_ . Hoboken, NJ, USA: Wiley, 2014.

[36] R. S. Sutton and A. G. Barto, _Reinforcement_ _Learning:_ _An_ _Introduction_,
Cambridge, MA, USA: MIT Press, 2018.

[37] Z. Zhang, L. Zheng, and M. X. Weng, ‘‘Dynamic parallel machine scheduling with mean weighted tardiness objective by Q-learning,’’ _Int._ _J._ _Adv._
_Manuf. Technol._, vol. 34, nos. 9–10, pp. 968–980, Oct. 2007.

[38] Z. Zhang, L. Zheng, N. Li, W. Wang, S. Zhong, and K. Hu, ‘‘Minimizing mean weighted tardiness in unrelated parallel machine scheduling with reinforcement learning,’’ _Comput._ _Oper._ _Res._, vol. 39, no. 7,
pp. 1315–1324, Jul. 2012.

[39] B. Yuan, L. Wang, and Z. Jiang, ‘‘Dynamic parallel machine scheduling
using the learning agent,’’ in _Proc. IEEE Int. Conf. Ind. Eng. Eng. Manage._,
Dec. 2013, pp. 1565–1569.

[40] B. Yuan, Z. Jiang, and L. Wang, ‘‘Dynamic parallel machine scheduling
with random breakdowns using the learning agent,’’ _Int. J. Services Oper._
_Informat._, vol. 8, no. 2, pp. 94–103, 2016.

[41] Y.-R. Shiue, K.-C. Lee, and C.-T. Su, ‘‘Real-time scheduling for a smart
factory using a reinforcement learning approach,’’ _Comput._ _Ind._ _Eng._,
vol. 125, pp. 604–614, Nov. 2018.

[42] J. Shahrabi, M. A. Adibi, and M. Mahootchi, ‘‘A reinforcement learning
approach to parameter estimation in dynamic job shop scheduling,’’ _Com-_
_put. Ind. Eng._, vol. 110, pp. 75–82, Aug. 2017.

[43] Y.-F. Wang, ‘‘Adaptive job shop scheduling strategy based on weighted
Q-learning algorithm,’’ _J._ _Intell._ _Manuf._, vol. 31, no. 2, pp. 417–432,
Feb. 2020.

[44] Y. Cheng, J. Peng, X. Gu, F. Jiang, H. Li, W. Liu, and Z. Huang, ‘‘Optimal
energy management of energy internet: A distributed actor-critic reinforcement learning method,’’ in _Proc._ _Amer._ _Control_ _Conf._ _(ACC)_, Jul. 2020,
pp. 521–526.

[45] M. Nazari, ‘‘Reinforcement learning for solving the vehicle routing
problem,’’ in _Proc._ _Adv._ _Neural_ _Inf._ _Process._ _Syst._, vol. 31, Feb. 2018,
pp. 9839–9849.

[46] Y. Ye, D. Qiu, J. Li, and G. Strbac, ‘‘Multi-period and multi-spatial
equilibrium analysis in imperfect electricity markets: A novel multiagent deep reinforcement learning approach,’’ _IEEE_ _Access_, vol. 7,
pp. 130515–130529, 2019.

[47] H. Mao, M. Alizadeh, I. Menache, and S. Kandula, ‘‘Resource management with deep reinforcement learning,’’ in _Proc. 15th ACM Workshop Hot_
_Topics Netw._, Nov. 2016, pp. 50–56.



101400 VOLUME 9, 2021


B. Paeng _et al._ : Deep RL for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups




[48] Y. Bao, Y. Peng, and C. Wu, ‘‘Deep learning-based job placement in
distributed machine learning clusters,’’ in _Proc._ _IEEE_ _Conf._ _Comput._
_Commun._ _(IEEE_ _INFOCOM)_, Apr. 2019, pp. 505–513, doi: [10.1109/](http://dx.doi.org/10.1109/INFOCOM.2019.8737460)
[INFOCOM.2019.8737460.](http://dx.doi.org/10.1109/INFOCOM.2019.8737460)

[49] A. Mirhoseini, ‘‘Device placement optimization with reinforcement learning,’’ in _Proc. 34th Int. Conf. Mach. Learn._, vol. 70, 2017, pp. 2430–2439.

[50] Z. Cao, H. Zhang, Y. Cao, and B. Liu, ‘‘A deep reinforcement learning
approach to multi-component job scheduling in edge computing,’’ in _Proc._
_15th Int. Conf. Mobile Ad-Hoc Sensor Netw. (MSN)_, Dec. 2019, pp. 19–24.

[51] D. Shi, ‘‘Intelligent scheduling of discrete automated production line
via deep reinforcement learning,’’ _Int._ _J._ _Prod._ _Res._, vol. 58, pp. 1–19,
Jun. 2020.

[52] C.-C. Lin, D.-J. Deng, Y.-L. Chih, and H.-T. Chiu, ‘‘Smart manufacturing
scheduling with edge computing using multiclass deep Q network,’’ _IEEE_
_Trans. Ind. Informat._, vol. 15, no. 7, pp. 4276–4284, Jul. 2019.

[53] B.-A. Han and J.-J. Yang, ‘‘Research on adaptive job shop scheduling problems based on dueling double DQN,’’ _IEEE_ _Access_, vol. 8,
pp. 186474–186495, 2020.

[54] T. E. Thomas, J. Koo, S. Chaterji, and S. Bagchi, ‘‘MINERVA: A reinforcement learning-based technique for optimal scheduling and bottleneck
detection in distributed factory operations,’’ in _Proc. 10th Int. Conf. Com-_
_mun. Syst. Netw. (COMSNETS)_, Jan. 2018, pp. 129–136.

[55] S. Zheng, C. Gupta, and S. Serita, ‘‘Manufacturing dispatching using
reinforcement and transfer learning,’’ 2019, _arXiv:1910.02035_ . [Online].
Available: http://arxiv.org/abs/1910.02035

[56] D. Harris and S. Harris, _Digital_ _Design_ _and_ _Computer_ _Architecture_ .
San Mateo, CA, USA: Morgan Kaufmann, 2010.

[57] H. Rummukainen and J. K. Nurminen, ‘‘Practical reinforcement learningexperiences in lot scheduling application,’’ _IFAC-PapersOnLine_, vol. 52,
no. 13, pp. 1415–1420, 2019.

[58] V. Nair and G. E. Hinton, ‘‘Rectified linear units improve restricted
Boltzmann machines,’’ in _Proc._ _Int._ _Conf._ _Mach._ _Learn._ _(ICML)_, 2010,
pp. 807–814.

[59] P. J. Huber, ‘‘Robust estimation of a location parameter,’’ _Ann._ _Math._
_Statist._, vol. 35, no. 1, pp. 73–101, 1964.

[60] J. Bergstra and Y. Bengio, ‘‘Random search for hyper-parameter optimization,’’ _J. Mach. Learn. Res._, vol. 13, pp. 281–305, Feb. 2012.

[61] T. Tieleman and G. Hinton, ‘‘Lecture 6.5-RMSPROP: Divide the gradient
by a running average of its recent magnitude,’’ _COURSERA, Neural Netw._
_Mach. Learn._, vol. 4, no. 2, pp. 26–31, 2012.

[62] A. P. J. Vepsalainen and T. E. Morton, ‘‘Priority rules for job shops with
weighted tardiness costs,’’ _Manage._ _Sci._, vol. 33, no. 8, pp. 1035–1047,
1987.



BOHYUNG PAENG received the B.S. degree
in electrical engineering from KAIST, South
Korea, in 2013. He is currently pursuing the
Ph.D. degree with the Information Management
Laboratory, Department of Industrial Engineering, Seoul National University, South Korea.
His current research interests include scheduling,
deep reinforcement learning, and deep learning
applications.


IN-BEOM PARK received the Ph.D. degree
in industrial engineering from Seoul National
University, Seoul, South Korea, in 2020. He is
currently a Postdoctoral Researcher in industrial
engineering with Sungkyunkwan University. His
current research interests include scheduling manufacturing systems, machine learning, and deep
reinforcement learning.


JONGHUN PARK received the Ph.D. degree in
industrial and systems engineering with a minor
in computer science from the Georgia Institute of
Technology, Atlanta, in 2000. He is currently a
Professor with the Department of Industrial Engineering, Seoul National University (SNU), South
Korea. Before joining SNU, he was an Assistant
Professor with the School of Information Sciences
and Technology, Pennsylvania State University,
University Park, and the Department of Industrial
Engineering, KAIST, Daejeon. His research interests include generative
artificial intelligence and deep learning applications.



VOLUME 9, 2021 101401


