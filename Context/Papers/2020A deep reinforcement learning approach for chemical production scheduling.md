A Deep Reinforcement Learning Approach for Chemical Production Scheduling

### Journal Pre-proof


A Deep Reinforcement Learning Approach for Chemical Production
Scheduling


Christian D. Hubbs, Can Li, Nikolaos V. Sahinidis,
Ignacio E. Grossmann, John M. Wassick


PII: S0098-1354(20)30159-9

Reference: CACE 106982


To appear in: _Computers and Chemical Engineering_


Received date: 13 February 2020
Revised date: 13 June 2020
Accepted date: 17 June 2020


Please cite this article as: Christian D. Hubbs, Can Li, Nikolaos V. Sahinidis,
Ignacio E. Grossmann, John M. Wassick, A Deep Reinforcement Learning Approach for
Chemical Production Scheduling, _Computers_ _and_ _Chemical_ _Engineering_ (2020), doi:
[https://doi.org/10.1016/j.compchemeng.2020.106982](https://doi.org/10.1016/j.compchemeng.2020.106982)


This is a PDF file of an article that has undergone enhancements after acceptance, such as the addition
of a cover page and metadata, and formatting for readability, but it is not yet the definitive version of
record. This version will undergo additional copyediting, typesetting and review before it is published
in its final form, but we are providing this version to give early visibility of the article. Please note that,
during the production process, errors may be discovered which could affect the content, and all legal
disclaimers that apply to the journal pertain.


© 2020 Published by Elsevier Ltd.


# A Deep Reinforcement Learning Approach for Chemical Production Scheduling

#### Christian D. Hubbs, [∗]

the reinforcement learning method outperforms the naive MILP approaches and is


competitive with a shrinking horizon MILP approach in terms of profitability, inventory


levels, and customer service. The speed and flexibility of the reinforcement learning


system is promising for achieving real-time optimization of a scheduling system, but


there is reason to pursue integration of data-driven deep reinforcement learning methods


and model-based mathematical optimization approaches.


_∗_ Department of Chemical Engineering, Carnegie Mellon University, Pittsburgh, PA 15123

_†_ Dow Chemical, Digital Fulfillment Center, Midland, MI 48667


1


_Keywords:_ Machine Learning, Reinforcement Learning, Optimization, Scheduling, Stochas

tic Programming

## **1 Introduction**


Industrial chemical operations in the modern day convert thousands of tons of raw material


inputs into thousands of tons of product worth tens of millions of dollars each day. Complex


questions regarding resource allocation must be asked and answered continuously [Harjunkoski


including uncertainty lead to significantly higher computational costs due to a large number


of scenarios in cases where discrete uncertainty is present. Computational costs of models


which represent uncertainty as continuous probability distributions are driven by integration


requirements [Balasubramanian and Grossmann, 2003]. Here, we will only focus on the


primary areas of research that are directly relevant to the planning and scheduling problem


we are pursuing (see Sahinidis [2004] for a fuller discussion of optimization under uncertainty).


2


Two primary methods to address planning and scheduling under uncertainty have emerged


over the years: robust optimization and stochastic optimization [Grossmann et al., 2016].


Robust optimization ensures a schedule is feasible over a given set of possible outcomes of the


uncertainty in the system [Bertsimas et al., 2011]. Lin et al. [2004] provides an example of


robust optimization for scheduling a chemical process modeled as a continuous time state-task


network (STN) with uncertainty in the processing time, demand, and raw material prices.


The stochastic optimization approach deals with uncertainty in stages whereby a decision


forth, with decisions being what to schedule next. The next state is a consequence of previous


decisions and the realization of any random variables in the system.


As illustrated in Figure 1, reinforcement learning involves an _agent_ which takes _actions_


based on observations and information – known as the _state_ - it receives from the _environment_


[Sutton and Barto, 2018]. In the case of planning and scheduling, the agent is the scheduling


algorithm and its actions are scheduling decisions (i.e. what to produce, when to produce,


3


Figure 1: Diagram of reinforcement learning system.


how much, etc.). The environment is the plant, factory, or machine that will do the processing,


and the state can be defined as inventory levels, demand, the existing schedule, or whatever


information is deemed relevant to developing a schedule. The goal of the agent is to maximize


multilayer perceptron and an RL technique - Q-learning - to learn local policies to minimize


the tardiness for a flexible job shop scheduling problem [Watkins, 1989]. Stockheim et al.


[2003] applied Q-learning to a job acceptance system whereby, the RL agent would learn a


good policy for accepting new jobs, which would then be scheduled using a deterministic


scheduling algorithm. Martinez et al. [2011] utilized a tabular Q-learning approach for a


flexible job shop scheduling problem and showed superior results for Q-learning ant colony


4


optimization and tabu search, yet under-performed relative to a genetic algorithm. Mortazavi


et al. [2015] used discretized Q-learning to develop a four-echelon supply chain simulation


consisting of a retailer, distributor, manufacturer, and supplier. The Q-learning system is


able to adapt and learn based on non-stationary demand described by a Poisson distribution


with a mean that changes deterministically on a 12-week cycle. Palombarini et al. [2018]


used the _SARSA_ ( _λ_ ) algorithm to develop logical if/else repair operators to build robustness


into schedules [Singh and Sutton, 1996].


champion, Lee Sedol [Silver et al., 2016]. The following year, these same techniques were


used to defeat the world champion Ke Jie and a new system was developed - AlphaGo Zero 

which learned without human input and easily defeated the previous AlphaGo system 100-0


[Silver et al., 2017]. A thorough overview of the developments in DRL and applications of


the technique is provided by Li [2017].


Given the success of DRL in large problems and the amenability of planning and scheduling


5


problems to MDP representations [Schneider et al., 1998], it seems natural to extend these


techniques to industrial planning and scheduling models. The literature on DRL in this area,


however, is severely lacking - perhaps not surprising given the recency of many of the DRL


accomplishments. Regardless, there is some research in this area, notably Oroojlooyjadid


et al. [2017] applied a single, deep Q-network (DQN) to each of the four stages of the beer


game (retailer, wholesaler, distributor, and manufacturer) to obtain near-optimal results.


Mao et al. [2016] considered an application of the REINFORCE algorithm to assigning jobs


respectively.

## **2 Problem Statement**


We consider a continuous, chemical manufacturing process with a single stage and single


production unit operating under stochastic demand modeled after a site owned and operated


6


by Dow Inc. [1] The goal of the scheduling algorithms is to build a production schedule for


the full planning horizon of _K_ days (product data is given in Table 1, all reported mass


quantities are given in metric tons, MT). The schedule is fixed for the first _H_ days (fixed


horizon) in accordance with operating procedures provided by the plant. This is done in


order to provide operating stability for the production facility and maintain commitments


to down-stream customers where deliveries have been confirmed. This makes the schedule


inflexible, particularly in cases where new orders are entered into the system for high-priority


may continue with product B in the next period, at which point it will register a transition


from B to B and no off-grade losses will be incurred (Table 2).


Demand is represented by an order book and is generated at the beginning of the simulation


according to a fixed statistical profile. The demand is revealed to the planning models when


the current day matches the order entry date that is associated with each order in the system.


1All dollar values, order quantities, profit margins, and so forth are shown for illustrative purposes only
and do not represent actual values of Dow Inc.


7


Table 1: Product data for simulated reactor.


Product Run Rate (MT/Day) Average Standard Margin ($/MT) Curing Time (Days)


A 218 28 1


B 237 39 1


C 259 40 1


D 246 44 1


This provides limited visibility to the models of future demand and forces it to react to new


demand forecast with actual, low-level orders. Further information on the specific strategies


employed can be found in Section 3.5.


The model is discretized in time to one day time intervals and the scheduling problem


takes place over a 90 day horizon. The planning models must operate in the presence of a


fixed planning horizon _H_, meaning the schedule cannot change for the next _H_ days out. For


example, if _H_ = 7 and a schedule is made on January 1 [st] for the 1 [st] -15 [th], then the schedule


8


Table 2: Product transition losses (MT).


Transition To


A B C D


Parameter Value Description


_H_ 7 Fixed Planning Horizon


_K_ 15 Lookahead Planning Horizon


_D_ 90 Number of Periods in Simulation


_η_ 12% Percentage of Annual Working Capital Cost


_α_ 25% Daily Late Shipment Penalty


9


shipped one day late, $500 if two days, and so forth.


The goal of the reinforcement learning and optimization methods are to maximize the


profitability of the site over the simulation period. The reward/objective function is given as:




  max _z_ =


_i_






_t_








  _VinSitn_ _η_
_n_ _−_ _i_



_i_





_βiIit_ (1)

_t_



10


where _Vin_ indicates the discounted standard margin for order _n_ and product _i_, and _Sitn_


denotes the shipped amount for order number _n_ for each time period _t_ corresponding to


product _i_ . This represents the income - although it may become a loss if there are sufficient


late penalties to pay - while the costs are related to carrying large inventory _Iit_, with _βi_


being the average variable standard margin for each product and _η_ being a fractional, capital


cost multiplier.


The approaches are compared using the total profitability and other key performance


approach. Finally, the perfect information MILP is used to give an optimistic upper-bound


on model performance for benchmarking purposes. All solutions are validated in simulation


to provide the results.


11


#### **3.1 Reinforcement Learning Model**

Machine learning is typically divided into three categories: supervised learning, unsupervised


learning, and reinforcement learning [Bishop, 2006]. Supervised learning deals with labeled


data and is used to address classification and regression problems. Unsupervised learning seeks


to find patterns in the structure of data such as with clustering algorithms. Reinforcement


learning is agent-based whereby an agent interacts with an environment in order to maximize


a reward.









(2)



**3.1.1** **Reinforcement** **Learning** **Algorithm**


We implemented a version of the Advantage Actor-Critic (A2C) algorithm [Mnih et al., 2016].


This is a policy gradient algorithm that learns a value function approximation (the critic)


and a policy function (the actor).


The critic learns to approximate the value of the current policy ( _v_ ˆ)  - according to its


12


parameterization ( _θv_ ˆ) - which is the expectation of the sum of the future discounted rewards:


                 -                  _v_ ˆ( _st, θv_ ˆ) _≈_ E _Rt_ + _γRt_ +1 + _γ_ [2] _Rt_ +2 + _..._ + _γ_ _[T]_ _Rt_ + _T_ (4)


where _st_ is the state at time _t_, _Rt_ is the reward received at time _t_ from transitioning from _st_


to _st_ +1, and _γ_ is a discount factor where 0 _< γ_ 1, which causes current rewards to be more
_≤_

valuable than future rewards.


The actor learns a stochastic policy ( _π_ ) parameterized by _θπ_ that is designed to take










  _R_ ( _st, at_ ) =


_i_








  _VinSitn_ _η_
_n_ _−_ _i_



_βiIit_ (6)

_i_



Updates are made to the parameters by calculating the temporal difference error (TD-Error).


∆= _Rt_ + _γv_ ˆ( _st_ +1 _, θv_ ˆ) _−_ _v_ ˆ( _st, θv_ ˆ) (7)


The gradient of the TD-Error is taken with respect to the parameters, which are then updated


13


using stochastic gradient ascent at the end of each episode, where an episode contains all of


the time steps from the beginning of the simulation until its termination [Sutton and Barto,


2018].


Policy gradient algorithms were first proposed by Williams [1992], and later a proof for


local optimality convergence was provided by Sutton et al. [1999]. More recently, these


algorithms provided the backbone for AlphaGo’s historic achievements [Silver et al., 2016].


The A2C algorithm deploys an actor-critic pair represented as a deep neural network to










[1] - _π_ _T_

_T_ _t_ [(∆] _[t][ −]_ _[R][t]_ [)][2]



Calculate critic loss: ( _θv_ ˆ) = _T_ [1]
_L_



Update critic: _θv_ ˆ := _θv_ ˆ + _αv_ ˆ _∇θv_ ˆ _L_ ( _θv_ ˆ)
4: **end** **for**



**3.1.2** **Action** **Selection**


The agent is modeled using a deep neural network (DNN) to represent the _policy_ _π_ - the


function which maps states to actions. The policy is stochastic and yields a probability


14


distribution over possible actions for each state. In typical DRL applications, the action


taken at time _t_ is immediately acted upon and the environment transitions to a new state,


_st_ +1. In the case of planning and scheduling, decisions must be made in advance for the


entire planning horizon without the benefit of observing the new state. There are at least


two ways to deal with this complication: the agent may sample over possible schedules for


the planning horizon, or the agent may iteratively sample over all products while taking into


account a model of the evolution of future states.


_−_


The DRL approach described above is known as a _model-free_ algorithm, meaning there is no


transition function of the system that is given to the RL agent - it must learn a good policy


through trial and error. For this reason, the model must be trained extensively by making


mistakes early on, and then learning what actions yield high rewards over time. This is done


through a sequence of Monte Carlo simulations of the scheduling environment to provide


feedback and data for the algorithm to update the network’s parameters.


15


as required in Section 3.5. Combining the state values in this way takes the relevant inputs


considered by a planner and reduces the size of the input vector. The feature engineering in


the net inventory means the network does not have to learn these relationships itself, which


did help speed training.


16


#### **3.2 Deterministic Optimization Model**

The following model is used to describe a single-stage, continuous production plant. Produc

tion decisions are discretized to daily time intervals ( _t_ ) and it takes one full day for each


product _i_ to be produced. Another full day is required for curing time so that the product


may be shipped. Thus, it takes two time intervals from when a product is scheduled and


production begins until it is available to be shipped and loaded into inventory. The production


facility has fixed run rates associated with each product ( _b_ _[max]_ _i_ ) and transition losses are


17


Table 5: Definitions of sets, indices, and variables.


Sets and Indices:


_i_, _j_ Product indices.


_t_ Index for time periods.


_n_ Index for individual orders.


Decision variables are given by _xitn_ and _yit_ where _xitn_ is a binary given to each unique


order number _n_ which is 1 when it is to be fulfilled with product _i_ at time _t_, and 0 otherwise.


_yit_ is a binary that relates the production of each product _i_ to be produced at time _t_ . _yit_ is


equal to one when product _i_ at time _t_ is scheduled and 0 otherwise. Only one product may


18


be produced at a time and an order may only be fulfilled once and only at or after the due


date, i.e. no early shipments.


The model is tasked with determining which products to produce and when in order to


maximize the profitability given in the objective function. Each order is specified by _ditn_


and has a ship-by date _tn_, and a corresponding variable standard margin _vn_ . A 25% per


day penalty is assessed on the variable standard margin for each day beyond the ship-by


date that an order is not shipped. Additionally, there is a cost to carrying inventory which


better end-state conditions, then the schedule is passed to the facility model to execute. The


simulation is stepped forward one time step, and the results are fed back into the MILP


model to generate a new schedule over the _K_ -step planning horizon. The objective function is


given in Equation 1 whereby the algorithm seeks to maximize the accrued profit by satisfying


orders with as little delay as possible, while minimizing inventory levels.


19


**3.3.1** **Mass** **Balance**




   _Iit_ = _Iit_ 1 + _pit_ 2

_−_ _−_
_−_




  _δijzijt−_ 2
_j_ _−_ _n_



_n_





_Sitn_ _i, t_ (8)
_∀_
_t≥tn_



Equation 8 states that for any product _i_ at time _t_, we calculate the mass balance of


the system as the inventory from the previous period, plus the scheduled production from


two days prior (given the two day delay for production and curing), minus any sales and


transition losses that may occur due to the production type changes.


was made at time _t −_ 1 and _j_ was produced in the subsequent time period _t_, where _j_ is an

alias of _i_ .


**3.3.3** **Shipping** **Constraints**


Each order in the system has a particular due date, _tn_, to determine when the order can be


fulfilled. The demand can only be satisfied for times where _t_ _tn_ . In other words, early
_≥_


20


satisfaction of orders is prohibited. The product quantity and orders to be fulfilled at time _t_


is given by the constraint:


_Sitn_ = _ditnnxitn_ _n, i, t_ _tn_ (11)
_∀_ _≥_


where the demand parameter _ditnn_ is satisfied when the corresponding binary variable _xitn_ = 1.


Because there may be situations where all demand cannot be satisfied, we relax the


constraint on the binary shipping variables such that they can be less than or equal to 1 for



all orders _n_ :





for this facility. The next equation ensures that the production values _pit_ do not exceed this


run-rate if the product is selected, or else is 0.


0 _pit_ _b_ _[max]_ _i_ _yit_ _i, t_ (14)
_≤_ _≤_ _∀_


Shipment quantities ( _Sitn_ ), inventory levels ( _Iit_ ), and production quantities ( _pit_ ) are


positive, continuous variables, while binaries are associated with individual order shipments


21


( _xitn_ ), production decisions ( _yit_ ), and product transitions ( _zijt_ ). These requirements are


expressed by:


_Sitn, Iit, pit_ 0 (15)
_≥_


_xitn, yit, zijt_ 0 _,_ 1 (16)
_∈{_ _}_


decisions, or recourse decisions, are taken as corrective actions at the end of the period. Stage


1 decisions usually correspond to the decisions that decision-maker needs to fix right now.


For this problem, the scheduler needs to fix the decisions for the next _H_ days from now.


Therefore, we treat the decisions corresponding to the first _H_ days from now as the first


stage decisions, and the decisions from day _H_ + 1 to day _K_ as the second stage decisions in


the rolling horizon framework.


22


Compared with the deterministic MILP model, we add index _ω_ to all the decisions


variables to distinguish the decisions corresponding to different scenarios. The objective


function seeks to maximize the the expected profit over all the scenarios:






i




  max IE(z) = _τω_

_ω_



��


n








   Vin _ω_ sitn _ω_ _η_
t _−_ i



i





_β_ iIit _ω_
t







(17)



where _τω_ is the probability of scenario _ω_ . We generate 10 different forecasts in the test case,


i.e., _τω_ = 0 _._ 1 for all _ω_ Ω.









_∀_



**3.4.2** **Transition** **Constraints**


Transition constraints are enforced by the stochastic versions of the equations found in the


deterministic model:

     
_zijtω_ = _yijtω_ _j, t, ω_ (19)
_∀_
_i_


23


     
_zijtω_ = _yit−_ 1 _ω_ _∀i, t, ω_ (20)
_j_


Each of these constraints behave according to the same dynamics, but is solved for each


scenario _ω_ .


**3.4.3** **Shipping** **Constraints**


Each scenario that is generated yields differing orders at various times over the planning








      
_yitω_ = 1 _t, ω_ (23)
_∀_
_i_


0 _pitω_ _b_ _[max]_ _i_ _yitω_ _t, ω_ (24)
_≤_ _≤_ _∀_


24


**3.4.5** **Non-Anticipativity** **Constraints**


Non-anticipativity constraints are provided as follows:


_yitω_ = _yitω_ +1 _ω, t_ _H_ (25)
_∀_ _≤_


These constraints ensure that the first stage production decision variables over the initial


fixed horizon _H_ remain consistent across all scenarios _ω_ .


The performance of the model is predicated on meeting specific orders with a given order


quantity and value. In addition, a high-level forecast is also provided to assist planning


decisions where orders are absent. Orders are entered daily, but there are often times when


there is excess capacity particularly at the beginning of each month before most orders are


entered. Because specific orders are difficult to forecast, greater accuracy can be achieved


forecasting total demand at a monthly level, albeit with a given amount of error. Thus, the


25


Table 6: Definitions of sets, indices, and variables for the stochastic optimization model.


Sets and Indices:


_i_, _j_ Product indices.


_t_ Index for time periods.


_ω_ Index for the scenarios


_n_ Index for the individual orders in each scenario.


26


                 -                  -                  _Fnet,it_ _[m]_ [= max] _Fit_ _[m]_ _[−]_ _i,n,t∈m_ _ditcn,_ 0 _∀i, t ≤_ _tc_ (28)


If the actual demand is greater than the forecasted demand, the net forecast is 0 and the


forecast is discarded because it underestimated the actual demand and provides no new


information.


27


The net forecast is then broken into individual orders and distributed over the remaining


horizon. Three basic disaggregation approaches were developed, a uniform forecast disaggre

gation (see Figure 5), a smoothed forecast disaggregation (see Figure 6), and a stochastic


forecast disaggregation (see Figure 7).


The following figures provide illustrations of how this is achieved. The top left panel


shows the actual demand that must be met as of the first day of the simulation when only a


few orders have been entered into the system. The top right panel shows how the net forecast


the month by calculating the average demand level, and evening out each of the days until


it reaches that point. Days that currently have above average actual demand receive no


additional forecasted demand, whereas days that are lacking actual demand are provided


forecasted demand.


28


Figure 5: Forecast disaggregation for a uniform forecast.


29


Figure 6: Smoothed forecast disaggregation.


30


**3.5.3** **Stochastic** **Demand** **Forecast**


The stochastic MILP requires multiple scenarios to be developed and optimized over. To


address this, we keep the actual demand constant across all scenarios and adjust the forecast


by sampling from a probability distribution over the remaining days.

#### **3.6 Shrinking Horizon Model**


To incorporate additional information from the forecast, we also implement a schedule


with a shrinking horizon model for times where the lookahead horizon is larger than the


remaining simulation time.

## **4 Example**


The methods and model discussed in this paper have been tailored to address the specific


scheduling needs of an existing site. The business constraints imposed on this problem differ


31


Figure 7: Stochastic forecast disaggregation.


32


we will further assume no forecast information is available to the model at this time. Finally,


this example will use the same objective function as the actual problem described in Equation


1.


At day 0, the model has no inventory and must begin producing to meet the demand


given in Table 9. The schedule must be developed for the next 10 days according with the


2See https://github.com/hubbs5/public drl ~~s~~ c


33


Table 7: Product data for simulated reactor.


Product Run Rate (Orders/Day) Average Standard Margin ($/Order) Curing Time (Days)


A 1 10 1


B 1 15 1


Table 8: Simulation parameters.


Parameter Value Description


_H_ 5 Fixed Planning Horizon


Moving to day 1, new orders are entered into the system, revealing that the model has


not scheduled any product to account for demand corresponding to product B. Moreover,


Table 9: Example order book available at day 0.


Document Number Document Creation Date Order Due Date Product Order Quantity Order Margin ($/Order) Forecast Flag


1 0 3 A 1 10 0


2 0 3 A 1 10 0


3 0 5 A 1 10 0


4 0 4 A 1 10 0


34


Figure 10: Predicted net inventory based on optimal schedule and order book at day 0.


35


Document Number Document Creation Date Order Due Date Product Order Quantity Order Margin ($/Order) Forecast Flag


1 0 3 A 1 10 0


2 0 3 A 1 10 0


3 0 5 A 1 10 0


4 0 4 A 1 10 0


5 1 4 B 1 15 0


6 1 5 B 1 15 0


Table 10: Example order book available at day 1.


Re-optimizing at Day 1 while respecting the previous scheduling decisions, we get a new


optimal schedule in Figure 11


Our predicted inventory is shown in Figure 12. In this case, the new orders force the


net inventory negative, indicating that losses will be accruing on the relevant orders. In


complex planning and scheduling problems, these mistakes can continue to compound over


time, leading to larger divergence between the implemented scheduling decisions and the


36


optimal.


which takes about 110 seconds to complete using a 2.9GHz Intel i7-7820HQ CPU.

## **5 Results and Discussion**


For the single-stage, industrial system, the RL model was subjected to 50,000 Monte Carlo


training episodes. Each training episode lasted for 90 simulated days and had the same


37


Figure 14: RL schedule for simplified example from day 0.


38


used in training the model above are given in Table 11.

#### **5.1 DRL and MILP Performance**


All models discussed were tasked with scheduling the reactor over an identical 90-day


period. Because of the stochastic nature of the model, we report average results as well as


direct comparisons between models while holding the given scenario constant to understand


39


Parameter Value Description













Table 11: Hyperparameter values used to train the DRL agent.


40


differences. The MILP models were all solved to a 1% optimality gap using Gurobi 8.1 using


a 2.9 GHz Intel i7-7820HQ CPU with four cores [Gurobi Optimization LLC, 2020].


Additionally, because all of the models were trained in simulation, we were able to build


a deterministic model with perfect information that could optimize the schedule over the


entire simulation horizon. This perfect information mixed-integer linear program (PIMILP)


yields an optimistic upper bound for the system, and enables more thorough comparisons of


the various approaches. This is particularly important because most DRL algorithms lack


bound.


is capable of accounting for the uncertainty in the system by training on the simulation


environment.


Examining the schedules (see Figure 17) we see that the RL agent learns a cycle from


product A to B to D to C, which yields 75 MT of lost prime production for the cycle. This


is a similar cycle as the one employed by the deterministic MILP with smoothed demand


forecast early in the simulation when demand is generally lower. The deterministic model


41


fewer delays and late penalties for a better overall reward as depicted in Figure 20. Thus,


while the product availability of the RL system is higher, so are the inventory costs, which


provide a buffer in the face of uncertainty.


Table 12 shows the different simulation times and model sizes for the different approaches


explored in this paper. The reported times are the average simulation times to complete the


90-day test simulation. The comparison is not entirely similar given that the RL model is


42


43


44


solve each scenario once, but solve 90 smaller, sub-problems as the simulation moves through


time.

#### **5.2 Robustness**


One of the well-known issues with model-free DRL is the lack of robustness if there are


changes in the environment. This is particularly relevant for a system that is intended to be


45


Deterministic MILP with Uniform Forecast 33,048 15,746 197


Deterministic MILP with Smoothed Forecast 34,356 16,332 298


Shrinking Horizon MILP 272,028 133,985 4,193


Stochastic MILP 918,069 432,913 21,913


Training Avg. Train Avg. Sim
Model

Episodes Time (s) Time (s)


RL 50,000 40,248 0.021


46


evenly over the simulation horizon to determine whether or not it performs as well (see Figure


21 for an illustration). In short, we move from a training simulation with seasonality, to a


testing environment without seasonal demand effects.


Comparing multiple scenarios under this new, uniform demand profile to the output of the


PIMILP model, we can see the optimality gap and decrease from changing the environmental


variables versus the training environment.


47


remains relatively consistent in its scheduling decisions exhibiting the same cyclic pattern as


shown in Figure 17, albeit with a shorter mean cycle time (8.3 days per cycle compared to


12.6 days). This reduction in cycle time, as well as poorer transition decisions as the agent


finds itself facing states outside of its previous experience, increases the off-grade production


by a factor of two. The loss in prime product further keeps the DRL agent from meeting


its goals yielding poorer performance over the course of the simulation as it falls farther


48


changes in the environment. These limitations can be addressed by re-training the system on


the new distribution before deployment, or perhaps through development of hybrid models


which can leverage the best of both worlds from the machine learning and MILP frameworks.


Additionally, providing a system of trained agents that are more responsive to different


regions of the state space or special situations outside of the assumed demand distribution,


may prove fruitful as well.


49


A reinforcement learning model has been developed and proposed for dynamic scheduling of a


single-stage multi-product reactor. The proposed approach provides a natural representation


for capturing the uncertainty in a system and outperforms the MILP schedulers operating


with a short receding time horizon for this reason. The incorporation of a forecast and


changing to a longer lookahead by switching to the shrinking horizon MILP model enables


50


this approach to perform better than all others.


DRL provides a viable and promising approach for chemical production scheduling. It is


often easier to incorporate uncertain elements in simulation versus in a mathematical program.


This uncertainty can be represented by the DRL agent such that, once the DRL agent is


trained, it can produce schedules online, that are superior to more computationally intensive


methods. The schedule can be generated almost instantly via a sequence of forward-passes


through a deep neural network. This makes DRL for use-cases with regular and rapid


oracle. Another possibility is using the DRL agent to restrict the search space in a stochastic


programming algorithm. The agent, once trained, could assign low probability of receiving a


high reward to certain actions in order to remove those branches and accelerate the search of


the optimization algorithm.


Future research will explore possibilities for integrating DRL and optimization methods,


examining DRL in a continuous time representation, and extending DRL to multi-stage and


51


multi-agent systems for network optimization approaches. Additional sources of uncertainty


may be considered as well such as maintenance and equipment reliability, changes in the


prime rates of transitions, price fluctuations, and so forth to more accurately mirror the


actual system in question.

## **Declaration of interests**


The authors declare that they have no known competing financial interests or personal


52


## **References**

T. A. Badgwell, J. H. Lee, and K.-h. Liu. Reinforcement Learning - Overview of Recent


Progress and Implications for Process Control. In _International_ _Symposium_ _on_ _Process_


_Systems_ _Engineering_, pages 71–85, 2018. ISBN 9780444642417.


J. Balasubramanian and I. Grossmann. Scheduling optimization under uncertainty—an


alternative approach. _Computers_ _and_ _Chemical_ _Engineering_, 27:469–490, 2003. ISSN


in mathematical programming techniques for the optimization of process systems under


uncertainty. _Computers_ _and_ _Chemical_ _Engineering_, 91:3–14, 2016. ISSN 00981354. doi:


10.1016/j.compchemeng.2016.03.002. URL `http://dx.doi.org/10.1016/j.compchemeng.`


`2016.03.002` .


D. Gupta and C. T. Maravelias. On deterministic online scheduling: Major considerations,


paradoxes and remedies. _Computers_ _and_ _Chemical_ _Engineering_, 94:312–330, 2016. ISSN


53


00981354. doi: 10.1016/j.compchemeng.2016.08.006. URL `http://dx.doi.org/10.1016/`


`j.compchemeng.2016.08.006` .


Gurobi Optimization LLC. Gurobi Optimizer Reference Manual, 2020. URL `https://www.`


`gurobi.com` .


I. Harjunkoski, R. Nystr¨om, and A. Horch. Integration of scheduling and control-Theory or


practice? _Computers_ _and_ _Chemical_ _Engineering_, 33(12):1909–1918, 2009. ISSN 00981354.


doi: 10.1016/j.compchemeng.2009.06.016.


_and_ _Chemical_ _Engineering_, 28(10):2087–2106, 2004. ISSN 00981354. doi: 10.1016/j.


compchemeng.2004.06.006.


J. H. Lee, J. Shin, and M. J. Realff. Machine learning: Overview of the recent progresses


and implications for the process systems engineering field. _Computers_ _and_ _Chemical_


_Engineering_, 114:111–121, 2018. ISSN 00981354. doi: 10.1016/j.compchemeng.2017.10.008.


URL `http://dx.doi.org/10.1016/j.compchemeng.2017.10.008` .


54


F. L. Lewis and D. Liu. _Reinforcement_ _Learning_ _and_ _Approximate_ _Dynamic_ _Programming_


_for_ _Feedback_ _Control_ . Wiley, Hoboken, New Jersey, 2013.


Y. Li. Deep Reinforcement Learning: An Overview. pages 1–70, 2017. ISSN 1701.07274. doi:


10.1007/978-3-319-56991-8 _{\_ ~~_}_~~ 32. URL `http://arxiv.org/abs/1701.07274` .


Z. Li and M. Ierapetritou. Process scheduling under uncertainty: Review and challenges.


_Computers_ _and_ _Chemical_ _Engineering_, 32(4-5):715–727, 2008. ISSN 00981354. doi: 10.


1016/j.compchemeng.2007.03.001.


1087–1092, 1953. ISSN 00219606. doi: 10.1063/1.1699114.


V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and


K. Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning. 48, 2016. ISSN


1938-7228. doi: 10.1177/0956797613514093. URL `http://arxiv.org/abs/1602.01783` .


J. E. Morinelly and B. E. Ydstie. Dual MPC with Reinforcement Learning. _IFAC-_


55


_PapersOnLine_, 49(7):266–271, 2016. ISSN 24058963. doi: 10.1016/j.ifacol.2016.07.276.


URL `http://dx.doi.org/10.1016/j.ifacol.2016.07.276` .


A. Mortazavi, A. Arshadi Khamseh, and P. Azimi. Designing of an intelligent self-adaptive


model for supply chain ordering management system. _Engineering_ _Applications_ _of_ _Artificial_


_Intelligence_, 37:207–220, 2015. ISSN 09521976. doi: 10.1016/j.engappai.2014.09.004. URL


`http://dx.doi.org/10.1016/j.engappai.2014.09.004` .


A. Oroojlooyjadid, M. Nazari, L. Snyder, and M. Tak´aˇc. A Deep Q-Network for the Beer


2017.


N. V. Sahinidis. Optimization under uncertainty: State-of-the-art and opportunities. In


_Computers_ _and_ _Chemical_ _Engineering_, volume 28, pages 971–983, 2004. ISBN 0098-1354.


doi: 10.1016/j.compchemeng.2003.09.017.


G. Sand and S. Engell. Modeling and solving real-time scheduling problems by stochastic


integer programming. _Computers_ _and_ _Chemical_ _Engineering_, 28(6-7):1087–1103, 2004.


ISSN 00981354. doi: 10.1016/j.compchemeng.2003.09.009.


56


J. G. Schneider, J. A. Boyan, and A. W. Moore. Value function based production scheduling.


In _International_ _Conference_ _on_ _Machine_ _Learning_, volume 15, pages 522–530, Madison,


Wisconsin, USA, 1998.


J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel. Trust Region Policy


Optimization. _arXiv_, 2015. ISSN 19506708. doi: 10.3917/rai.067.0031.


J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal Policy Optimiza

tion Algorithms John. _arXiv_, 2017. ISSN 09594388. doi: 10.1016/j.conb.2007.07.004. URL


_Nature_, 550(7676):354–359, 2017. ISSN 14764687. doi: 10.1038/nature24270. URL


`http://dx.doi.org/10.1038/nature24270` .


S. P. Singh and R. S. Sutton. Reinforcement learning with replacing elibility traces, 1996.


T. Stockheim, M. Schwind, and K. Wolfgang. A reinforcement learning approach for supply


chain management. _1st_ _European_ _Workshop_ _on_ _Multi-Agent_ _Systems_, 2003.


57


R. Sutton and A. Barto. _Reinforcement_ _Learning:_ _An_ _Introduction_ . MIT Press, Cam

bridge, Massachusetts, 2 edition, 2018. URL `http://incompleteideas.net/book/`


`bookdraft2017nov5.pdf` .


R. S. Sutton, D. Mcallester, S. Singh, and Y. Mansour. Policy Gradient Methods for


Reinforcement Learning with Function Approximation. _In_ _Advances_ _in_ _Neural_ _Information_


_Processing_ _Systems_ _12_, pages 1057–1063, 1999. ISSN 0047-2875. doi: 10.1.1.37.9714.


S. Sutton, Richard. Learning to Predict by the Method of Temporal Differences. _Machine_


1989.


58


