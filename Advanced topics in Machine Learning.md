
# Reinforcement Learning - Setup

## The Loop

- Agent: The learner and decision maker.
- Environment: everything outside the agent.
- Reward: a number the agent tries to maximize over time. 
- **One time step**: at time t the agent sees state ${}S_{t}{}$ and picks action ${}A_{t}{}$. It then gets reward ${}R_{t+1}{}$ and lands in ${}S_{t+1}{}$.
- Trajectory: ${}S_{0},A_{0},R_{1},S_{1},A_{1},R_{2},S_{2},\dots{}$
- OBS: The reward for acting at time t is ${}R_{t+1}{}$, not ${}R_{t}{}$.

## Markov Decision Process 
MDP: a tuple ${}(S,A,p,\gamma){}$

* **S** = The set of states
* ${}A{}$ = the set of actions
* ${}p{}$ = the dynamics (how the environment works)
* ${}\gamma \in [0,1]{}$  = the discount factor

## Return ${}G_{T}{}$.
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}$$
- Recursive form (MEMORIZE THIS, everything later builds on it):
$$G_t = R_{t+1} + \gamma G_{t+1}$$
- If ${}\gamma = 0{}$, the agent only cares about the next reward. As ${}\gamma \to 0{}$, it becomes far-sighted.  
- If the reward is a constant +1 forever, then ${}G_{t}=\frac{1}{1-\gamma}{}$

## Policy ${}\pi(a|s){}$


* ${}\pi(a  |  s){}$ is the probability of choosing action ${}a{}$ in state ${}s{}$. 

## Value functions

Answers: “How good is it to be here, if I keep following policy ?
* $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ is the value of a state
* $q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$ is the value of a state when you first take action a.

#### Definitions
* **State value**: $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$. This is the expected return starting in ${}s{}$ and following ${}\pi{}$ afterwards. 
* **Action value**: $q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$. Same idea, but you first take action ${}a{}$.
* Both can be estimated by averaging real returns from experience. That's the Monte Carlo idea.  

#### Bellman equation
Plugging ${}G_{t} = R_{t+1} + \gamma G_{t+1}{}$ into ${}v_{\pi}{}$. 
$$v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\,\big[r + \gamma v_\pi(s')\big]$$
In words: The value now is an average over my action and then over what the world does, of (reward + discounted value of where I land).

### Optimal versions
${}v_{*}(s) = \max_{\pi}{v_{\pi}(s)}{}$ and ${}q_{*}(s,a) = \max_{\pi}{q_{\pi}(s,a)}{}$

The only change in the equation is that the average over actions becomes a max.
  $$v_*(s) = \max_a \sum_{s',r} p(s',r|s,a)\big[r + \gamma v_*(s')\big]$$ $$q_*(s,a) = \sum_{s',r} p(s',r|s,a)\big[r + \gamma \max_{a'} q_*(s',a')\big]$$
![[Pasted image 20261018173509.png|598]]
*BACKUP TREES where* 
* *Open Circles are states and filled dots are actions, purple is what the agent controls. Teal is what the environment decides.*
* *Each leaf contributes to ${}r + \gamma v(s'){}$ and the tree "backs up" those values to s.

### Getting the optimal policy
* From ${}v_{*}{}$: do a one-step lookahead using ${}p{}$ and pick the best action. You need the model. 
* From ${}q_{*}{}$: just pick ${}\arg max_{a}q_{*}(s,a){}$. No model is needed. This is why later algorithms (Sarsa, Q-learning) learn q instead of v.

A model is the agents knowledge of how the environment works. Environment being ${}p(s',r|s,a):{}$ for each state and action, where can you end up, what reward you get, and with what probability.

# 2. Estimating ${}v_{\pi}{}$ with DP, MC and TD 

### 2.1 The problem: prediciton
* **Prediction** (also called policy evaluation) means estimating ${}v_{\pi}{}$ for a fixed policy ${}\pi{}$
* **Control** means finding the optimal policy. That comes in later topics.
* **Notation**:
	* ${}v_{\pi}{}$ is the true value.
	* V is our current estimate of it
	* ${}v_{k}{}$ is the DP estimate after iteration k.

## 2.2 The starting point: two ways to write ${}v_{\pi}{}$ 
$$v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] \qquad (6.3)$$ $$v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \qquad (6.4)$$
* MC (Monte Carlo) estimates (6.3). It uses a sampled return instead of the expected return.
* DP estimates (6.4). It uses its current guess instead of the unknown ${}v_{\pi}(S_{t+1}){}$
* TD estimates (6.4) for both reasons: it samples, and it uses its current guess. 

## 2.3 Dynamic Programming (RL 4)
### 2.3.1 Idea
* DP computes the expectation exactly, using a perfect model ${}p(s',r | s,a){}$
* Its limitations: it needs a **perfect model**, and its computationally expensive.

### 2.3.2 Iterative policy evaluation
$$v_{k+1}(s) = \sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)\big[r + \gamma v_k(s')\big]$$
* Start from ${}v_{0}(s) = 0{}$ (or any values), with the terminal state fixed at 0.
* ${}v_{k} \to as k \to \infty{}$
* This is the Bellman equation turned into an update rule.
* 
### 2.3.3 In-place vs synchronous
* **Synchronous**: Compute all new values from the old table ${}v_{k}{}$. The exam's value-iteration question uses this.
* **In-place**: each new value immediately overwrites the old one. 
* The algorithm box loops until the largest change $\Delta$ is smaller than a threshold $\theta$.

### 2.3.4 Properties

* Needs a model
* Bootstraps: yes (it uses estimates of other states)
* Samples: no
* Looks at all branches, but only one step deep

## 2.4 Monte Carlo

### 2.4.1 Idea
* MC learns from **experience**: sampled episodes of states, actions and rewards. 
* It averages the actual returns. No model is needed.
* It only works for **episodic** tasks, because you need the episode to end to know ${}G_{t}{}$.

### 2.4.2 First-visit vs every-visit
* **First-visit MC**: average only the returns that follow the **first** visit to *s* in each episode.
* **Every-visit MC**: average the returns after **all** visits to *s*. 
* Both converge to ${}v_{\pi}(s){}$. For first-visit, each return is an independent sample, so this follows the law of large numbers. 

### 2.4.3 Constant-$\alpha$ MC update
$$V(S_t) \leftarrow V(S_t) + \alpha\big[G_t - V(S_t)\big]$$
### 2.4.4 Properties
* Uses only the sampled path, and goes all the way to the end of the episode.
* Does **not** bootstrap: each state's estimate is independent of the others.

### 2.4.5 MC for action values
* Without a model, it's more useful to estimate ${}q_{\pi}{}$(s,a) than ${}v_{\pi}(s){}$
* **Problem**: some-state action pairs may never be visited.
* **Fix**: exploration. One option is **exploring starts**: every episode starts from a random state-action pair. More on this in the control topic. 

## 2.5 TD(0)
### 2.5.1 Idea
* TD is **model-free**, like MC.
* TD **bootstraps**, like DP. 

### 2.5.2 Update
$$V(S_t) \leftarrow V(S_t) + \alpha\big[\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)\big]$$
### 2.5.3 TD error
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
### 2.5.4 Advantages
* Compared to DP, it needs no model
* Compared to MC, it works online and step by step, with no waiting for the episode to end.
* It does converge. Which one is faster has no theoretical answer, but in practice TD usually wins. 

## 2.6 n-step TD: between TD(0) and MC
### 2.6.1 n-step return
$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1}R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n})$$
* Use *n* real rewards, then bootstrap from the estimate of the state *n* steps later.
* *n* = 1 gives TD(0): $G_{t:t+1} = R_{t+1} + \gamma V_t(S_{t+1})$.
* If $t + n \ge T$, then $G_{t:t+n} = G_t$, which is the MC return.

### 2.6.2 Update
$$V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha\big[G_{t:t+n} - V_{t+n-1}(S_t)\big]$$
* You have to wait *n* steps to collect the rewards. No updates happen during the first *n* - 1 steps of each episode. 

## 2.7 Comparing DP, MC and TD

### 2.7.1 Summary table
|                                     | DP                            | MC    | TD(0)                         |
| ----------------------------------- | ----------------------------- | ----- | ----------------------------- |
| Needs the model $p(s',r \mid s,a)$? | Yes                           | No    | No                            |
| Bootstraps?                         | Yes                           | No    | Yes                           |
| Samples experience?                 | No                            | Yes   | Yes                           |
| Waits until the episode ends?       | No                            | Yes   | No                            |
| Target                              | expected $r + \gamma v_k(s')$ | $G_t$ | $R_{t+1} + \gamma V(S_{t+1})$ |
One-liner: **TD == MC's sampling + DP's bootstrapping**.

### 2.7.2 Backup diagrams
* **DP**: all branches, one step deep.
* **MC**: one sampled path, all the way to the end.
* **TD**: one sampled path, one step
* See the three-column diagram from earlier.

### 2.7.3 Spot the paradigm from the formula
* **MC**: a sum of actual rewards to to the end, like $\sum_{k=1}^{\infty}\gamma^{k-1}R_{t+k}$.
* **DP**: an expectation over the model using $v_k$, often with a max, like $v_{k+1}(s) = \max_a \mathbb{E}[R_{t+1} + \gamma v_k(S_{t+1}) \mid \dots]$.
* **TD**: one real reward plus your own estimate, ${}R_{t+1} + \gamma V(S_{t+1}){}$

### 2.7.4 Bias vs variance of the targets

| Target                            | Bias                                            | Variance                                                |
| --------------------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| MC: $G_t$                         | **Unbiased:** on average it equals $v_\pi(S_t)$ | **High:** it depends on every random step until the end |
| TD: $R_{t+1} + \gamma V(S_{t+1})$ | **Biased:** it uses the estimate $V$            | **Low:** only one random step                           |

- n-step TD sits in between. As $n$ grows it behaves more like MC.
- Exam: "multi-step TD has a smaller bias than MC" is **False**, because MC is already unbiased (example Q35b).
- RL 9, slides 18–21 cover the bias part. The variance part isn't on the slides.

## 2.8 Batch updating

### 2.8.1 What "batch" means
* You have a finite amount of experience, for example 10 episodes. 
* Go through it repeatedly until the values converge. 
* Within one pass, add up all the increments and update once. 

### 2.8.2 Batch MC
* V(s) = the average returns observed after *s*.
* This minimizes the squared error on the **training data**.

### 2.8.3 Batch TD
* First build a model of the Markov process from the data, then compute the values that model gives. 
* If the process is Markov, this gives a lower error on **future** data. That's why batch TD beats batch MC. 

### 2.8.4 Example: "You are the Predictor"
* *V(B)* = 3/4
* **Batch MC**: *A* was seen once, and the return after it was 0. So *V(A)=0*.
* **Batch TD**: *A* always went to *B*. So *V(A) = V(B) = 3/4*.

### 2.8.5 Exam recipe
#### Batch MC:
1. For each state, list every return that followed it (add up the rewards until T).
2. Average that list.

#### Batch TD:
1. For each state, find the average immediate reward and the fraction of times it goes to each next state.
2. $V(s) =$ (average reward) $+ \gamma \sum_{s'}$ (fraction to $s'$) $\cdot V(s')$
3. Solve backwards, starting from the states just before T. 
# Topic 3: Dynamic Programming

## 3.1 What DP is
* DP is a collection of algorithms that compute **optimal policies** when you have a **perfect model** of the environment, i.e. you know $p(s',r \mid s,a)$.
* Limitations: it needs a perfect model, and it is computationally expensive.
* It assumes **finite MDP**: $\mathcal{S}$, $\mathcal{A}$ and $\mathcal{R}$ are all finite.
* Core trick: take a Bellman equation and turn it into an **update rule**.
* Once you have $v_*$ or $q_*$, the optimal policy is easy (Topic 1, 5.3.2).

## 3.2 Policy evaluation
### 3.2.1 Goal
* Compute ${} v_{\pi}{}$ for a fixed policy ${}\pi{}$ ("prediction").
* Start from the Bellman equation for ${}v_{\pi}{}$: 
$$v_\pi(s) = \sum_a \pi(a \mid s)\sum_{s',r} p(s',r \mid s,a)\big[r + \gamma v_\pi(s')\big] \qquad (4.4)$$
* With known dynamics this is ${}|S|{}$ linear equations with ${}|S|{}$ unknowns. Solvable directly, but tedious, so we iterate instead.

### 3.2.2 Iterative policy evaluation

$$v_{k+1}(s) = \sum_a \pi(a \mid s)\sum_{s',r} p(s',r \mid s,a)\big[r + \gamma v_k(s')\big]$$
* Start with ${}v_{0}(s) = 0{}$ for all ${}s{}$ (any start works; the terminal state is always 0).
* Same as the Bellman equation, but the right side uses the old estimate ${}v_{k}{}$ instead of the true ${}v_{\pi}{}$. 
* ${}v_{k} \to v_{\pi} as k \to \infty{}$.

### 3.2.3 In-place version and stopping rule
* In-place: each new value immediately overwrites the old one (one array instead of two).
* Track the biggest change in a sweep: ${}\Delta = max_{s} | v-V(S)|$
* Stop when ${}\Delta < \theta{}$ (a small threshold).

### 3.2.4 Example: 4x4 gridworld
* 14 states plus a terminal corner (shown twice), reward -1 per move, ${}\gamma = 1{}$, random policy (each action with probability 1/4).
* ${}v_{k}{}$ for the random policy: ${}k=1{}$ gives -1 everywhere;
	${}k=\infty{}$ gives values like -14, -18, -20, -22. 
* **Key observation**: the greedy policy w.r.t. ${}v_{k}{}$ is already optimal at ${}k=3{}$, long before ${}v_{k}{}$ has converged. 
* So finishing a policy evaluation is often unnecessary. This motivates **value iteration** (section 5). 

### 3.2.5 Exam: "policy evaluation for two steps"
This is iterative policy evaluation from ${}v_{0}{}$, run twice. 
With ${}\gamma = 1{}$: 
1. $v_1(s) = \mathbb{E}_\pi[r \mid s]$: the expected immediate reward in each state under $\pi$.
2. Write down $P_\pi(s' \mid s)$: the probability of moving from $s$ to $s'$ under $\pi$.
3. $v_2(s) = \mathbb{E}_\pi[r \mid s] + \sum_{s'} P_\pi(s' \mid s)\, v_1(s')$

##### Worked example (for from the exam):
States L and R, actions stay and switch, ${}\gamma = 1{}$, the policy picks each action with probability 0.5.
* In L: stay gives r = 1, switch gives r = 0. In R: any action gives r = 2.

| State | $v_1(s)$ | Next state under $\pi$ | $v_2(s)$ |
| --- | --- | --- | --- |
| L | $0.5 \cdot 1 + 0.5 \cdot 0 = 0.5$ | L: 0.5, R: 0.5 | $0.5 + 0.5 \cdot 0.5 + 0.5 \cdot 2 = 1.75$ |
| R | $2$ | L: 0.5, R: 0.5 | $2 + 0.5 \cdot 0.5 + 0.5 \cdot 2 = 3.25$ |

## 3. Policy improvement

### 3.1 Idea

* What if, in state *s*, we pick action *a* once and then follow ${}\pi{}$? Its value is:
$$q_\pi(s,a) = \sum_{s',r} p(s',r \mid s,a)\big[r + \gamma v_\pi(s')\big] \qquad (4.6)$$
* If $q_\pi(s,a) > v_\pi(s)$, choosing ${}a{}$ every time we are in ${}s{}$ should give a better policy. 

### 3.2 Policy improvement theorem
* Let ${}\pi{}$ and ${}\pi'{}$ be deterministic policies. If for all ${}s{}$
$$q_\pi(s, \pi'(s)) \ge v_\pi(s) \qquad (4.7)$$
* then ${}\pi'{}$ is as good as or better than ${}\pi{}$ in every state:
$$v_{\pi'}(s) \ge v_\pi(s) \qquad (4.8)$$

### 3.3 Proof
Start from, expand with (4.6), apply (4.7) again at the next state, and repeat.
$$  
\begin{aligned}  
v_\pi(s) &\le q_\pi(s, \pi'(s)) && \text{by (4.7)}\\  
&= \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = \pi'(s)] && \text{by (4.6)}\\  
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] && \text{first action chosen by } \pi'\\  
&\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) \mid S_t = s] && \text{(4.7) at } S_{t+1}\\  
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t = s] && \text{(4.6) again}\\  
&\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 v_\pi(S_{t+3}) \mid S_t = s]\\  
&\;\;\vdots\\  
&\le \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \mid S_t = s]\\  
&= v_{\pi'}(s)  
\end{aligned}  
$$
* In words: switching to ${}\pi'{}$ for one step is at least as good. So switching for two steps, three steps, and so on forever. "Forever" is just following ${}\pi'{}$.

### 3.4 The greedy policy satisfies the theorem

$$\pi'(s) = \arg\max_a q_\pi(s,a) = \arg\max_a \sum_{s',r} p(s',r \mid s,a)\big[r + \gamma v_\pi(s')\big]$$
* The greedy action gives $q_\pi(s,\pi'(s)) = \max_a q_\pi(s,a) \ge q_\pi(s,\pi(s)) = v_\pi(s)$, so (4.7) holds automatically. 

### 3.5 When improvement stops, you are optimal
* If ${}\pi' {}$ is no better than ${}\pi{}$, then ${}v_{\pi'} = v_{\pi}{}$, and for all s:
$$v_{\pi'}(s) = \max_a \sum_{s',r} p(s',r \mid s,a)\big[r + \gamma v_{\pi'}(s')\big]$$
* That is the Bellman optimality equation, so ${}v_{\pi'} = v_{*}{}$ and both ${}\pi{}$ and ${}\pi'{}$ are optimal. 

### 3.6 Exam (Example Q27)
* Greedy ${}\pi'{}$ w.r.t. ${}v_{\pi}:v_{\pi'}(s) \ge v_{\pi}(s){}$ always holds for all s.
* ${}v_{\pi'}(s) = v_{\pi}(s){}$ for all ${}s{}$ holds **only if** ${}\pi{}$ is optimal. 

## 4. Policy iteration 
### 4.1 Idea
$$\pi_0 \xrightarrow{E} v_{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} v_{\pi_1} \xrightarrow{I} \pi_2 \xrightarrow{E} \cdots \xrightarrow{I} \pi_* \xrightarrow{E} v_*$$
* E = policy evaluation, I = policy improvement.
* Each new policy is **strictly better** than the last, unless it is already optimal. 
* A finite MDP has finitely many policies, so this reaches ${}\pi_{*}{}$ in a **finite** number of iterations. 

### 4.2 Algorithm
1. **Initialize** ${}V(s){}$ and ${}\pi(s){}$ arbitrarily. 
2. **Policy evaluation**: repeat sweeps of
	$V(s) \leftarrow \sum_{s',r} p(s',r \mid s,\pi(s))\big[r + \gamma V(s')\big]$  
	until $\Delta < \theta$.
3. **Policy improvement**: for each ${}s{}$, set
	$\pi(s) \leftarrow \arg\max_a \sum_{s',r} p(s',r \mid s,a)\big[r + \gamma V(s')\big]$.
	If no action changed (*policy-stable*), stop: 
	$V \approx v_*$ and $\pi \approx \pi_*$. Otherwise go back to step 2.

* ${}\pi{}$ is deterministic here, so step 2 has no $\sum_a \pi(a \mid s)$, just the single action ${}\pi(s){}$. 

### 4.3 Example: Jack's car rental
* Two rental locations, at most 20 cars each. A rental earns 10 dollars; moving a car overnight costs 2 dollars. Requests and returns are Poisson. 
* Continuing MDP with ${}\gamma = 0.7{}$. State = number of cars at each location at the end of the day. Action = net number of cars moved overnight. 
* Solved with policy iteration. 

## 5. Value Iteration

### Why
* The evaluation step inside policy iteration is expensive (many sweeps).
* The gridworld showed that evaluation can be cut short without changing the greedy policy. 
* Extreme case: stop evaluation after **one sweep**. That is value iteration. 

### 5.2 Update rule
$$v_{k+1}(s) = \max_a \sum_{s',r} p(s',r \mid s,a)\big[r + \gamma v_k(s')\big] \qquad (4.10)$$
* View 1: one sweep of evaluation and one improvement step, merged into one (the max does the improving).
* View 2: the **Bellman optimality equation** turned into an update rule. 
* The only difference from iterative policy evaluation: $\sum_a \pi(a \mid s)$ is replaced by $\max_a$.

### 5.3 Algorithm
1. Initialize *V(s)* arbitrarily, except *V(terminal)* = 0 
2. Repeat sweeps of (4.10) over all states until ${}\Delta < \theta{}$.
3. Output a **deterministic** policy: $\pi(s) = \arg\max_a \sum_{s',r} p(s',r \mid s,a)\big[r + \gamma V(s')\big]$

### 5.4 Truncated policy iteration
* In between the two extremes: do a few evaluation sweeps between each improvement. 
* All these variants converge to an optimal policy for discounted finite MDPs.

### 5.5 Example: Gambler's problem
* Bet on coin flips: win the stake on heads, lose it on tails. Goal: reach 100. ${}p_{h}{}$= 0.4.
* States: capital ${}s \in \{ 1,\dots,99 \}{}$. Actions: stake ${}a \in \{ {0,1,\dots,min(s,100-s)} \}{}$.
* Reward 0 everywhere except +1 on reaching 100. Undiscounted and episodic. 
* So ${}v(s){}$ = the probability of winning from capital s. Solved with value iteration.

### 5.6 Exam recipe: value iteration by hand

For a deterministic grid with reward -1 per move and ${}\gamma{}$ = 1, (4.10) becomes:
$$v_{k+1}(s) = \max_a \big[-1 + v_k(s')\big]$$
where ${}s'{}$ is where action ${}a{}$ takes you (your own cell if you hit a wall or the edge).
1. Synchronous: compute every new value from the old table ${}v_{k}{}$ only. 
2. For each cell, try all 4 moves and take the best.
3. The terminal state stays 0.
4. Greedy policy: draw **every** action that ties for the max. 

**Worked example** (not from the exam): a corridor 1 - 2 - 3 - Z, actions left/right, start from ${}v_{0}{}$ = 0. 

| Sweep | $v(1)$ | $v(2)$ | $v(3)$ |
| ----- | ------ | ------ | ------ |
| $v_0$ | 0      | 0      | 0      |
| $v_1$ | −1     | −1     | −1     |
| $v_2$ | −2     | −2     | −1     |
| $v_3$ | −3     | −2     | −1     |
* Example: ${}v_{2}(2){}$: right gives -1 + ${}v_{1}(3) = -2{}$, left gives ${}-1+v_{1}(1) = -2{}$ , so ${}v_{2}(2)=-2{}$.
* After ${}v_{3}{}$ nothing changes. Each value = minus the number of steps to Z, and the greedy policy is "right" everywhere. 
* The correct values spread outward from the terminal state, one step per sweep. The interactive grid from earlier shows the same thing. 

![[Pasted image 20261020120118.png]]

## 6. Policy iteration vs value iteration (exam Q16)

|  | Policy iteration | Value iteration |
| --- | --- | --- |
| Evaluation per iteration | Full: many sweeps until $\Delta < \theta$ | One sweep only |
| Update used | Bellman equation for $v_\pi$, with $a = \pi(s)$ | Bellman optimality equation, with $\max_a$ |
| Explicit policy during the run | Yes, improved after each evaluation | No, read off at the end |
| Cost per iteration | High | Low |
| Number of iterations | Few (finite) | Many (stops when $\Delta < \theta$) |
| Output | Deterministic optimal policy | Deterministic optimal policy |
## 7. Asynchronous DP
* Sweeping over every state each time is expensive.
* Asynchronous DP updates states **in any order**, using whatever values are currently available. Some states may be updated several times before others are updated once. 
* It still has to keep updating **all** states to converge.

## 8. Generalized policy iteration
* GPI is two processes working together:
	* **Evaluation**: make the value function consistent with the current policy.
	* **Improvement**: make the policy greedy w.r.t. the current value function.
* How often and how fully each one run can vary. Policy iteration, value iteration and asynchronous DP are all GPI.
* When both stop changing, the Bellman optimality equation holds, so the result is optimal.
* Almost every method in the course is GPI, including MC control, Sarsa, and Q-learning. Thats why the exam asks "which line of Sarsa is policy evaluation / policy improvement?"

## 9. Efficiency of DP
* DP finds an optimal policy in time that is polynomial in the number of states and actions.
* The curse of dimensionality (the state space grows exponentially with the number of state variables) comes from the problem itself, not from DP. 
* For large state spaces, asynchronous DP and other GPI variants are preferred. 

## 10. Exam quick facts
- $v_{k+1}(s) = \max_a \mathbb{E}[R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t = s, A_t = a]$ is **DP**, specifically value iteration (Q18).
- “Value iteration does only one sweep of evaluation and updates the policy directly”: **True** (Q16).
- “Policy iteration evaluates until a stopping criterion before updating the policy”: **True** (Q16).
- “Value iteration outputs a stochastic policy”: **False**. It outputs a deterministic one (slide 24). (Q16)
- “One step of policy iteration rarely improves the policy”: **False**. Every step strictly improves the policy unless it’s already optimal (Q16).
- Greedy w.r.t. $v_\pi$ gives $v_{\pi'}(s) \ge v_\pi(s)$ for all $s$: **True** (Q27).
- To get the optimal policy with value iteration (2021 Q3b): run (4.10) until $\Delta < \theta$, then take the greedy policy w.r.t. $V$.


# Topic 4: Sarsa & Q-learning

## 4.1 The Idea
* TD control = GPI, with TD used for the evaluation step.
* Leraning ${}Q(s,a){}$ instead of ${}V(s){}$, so no model is needed to pick actions. 
* Explore with ε-greedy: with probablity ε pick a random action, otherwise pick the action with the highest Q. 

## 4.2 Sarsa

### 4.2.1 Update

$$Q(S,A) \leftarrow Q(S,A) + \alpha\big[R + \gamma Q(S',A') - Q(S,A)\big]$$

- $A'$ = the action the agent will **actually take next** (chosen ε-greedily).
- The name comes from the five things it uses: $S, A, R, S', A'$.
- If $S'$ is terminal, $Q(S',A') = 0$.

### 4.2.2 On-policy
* It learns the value of the policy it is actually following (ε-greedy).
* Behaviour policy = target policy.


### 4.2.3 Which line does what (exam Q9, Q10)
* **Policy improvement**: lines 5 and 8, where an action is chosen ε-greedily from $Q$.
* **Policy evaluation**: lines 9, where ${}Q{}$ is updated.

### 4.2.4 Why ε-greedily instead of greedy
* Every state-action pair must be visited infinitely often, or some ${}Q{}$ values never get corrected.
* The policy should still become greedy in the limit (for example ε = 1/t), so it converges to the optimal policy. 

## 4.3 Q-learning
### 4.3.1 Update
$$Q(S,A) \leftarrow Q(S,A) + \alpha\big[R + \gamma \max_a Q(S',a) - Q(S,A)\big]$$
It uses the **best** next action, no matter which action the agent actually takes next.


### 4.3.2 Off-policy
* Behavior policy (what is does): ε-greedy.
* Target policy (what it learns about: greedy). So it learns ${}q_{*}{}$ directly.
* No importance sampling is needed (slide 32)

### When it converges 
* With a Q-table, it converges to ${}q*{}$ as long as every state-action pair keeps being visited.
* Data from an ε-greedy policy: **yes**.
* Data from a uniformly random policy: **yes** (off-policy, so the behavior policy doesn't matter). 

## 4.4 Sarsa VS Q-learning: cliff walking
* Reward -1 per step. Stepping into the cliff gives -100 and sends you back to the start. ε = 0.1.
* **Q-learning** learns the optimal path right along the cliff edge. But it keeps exploring with ε-greedy, so it sometimes falls off. Result: worse reward during learning.
* Sarsa learns the value of its own ε-greedy behavior, so it "knows" it might slip. It learns the **safer path** further from the edge. Result: **better reward during learning**

## 4.5 Expected Sarsa
$$Q(S,A) \leftarrow Q(S,A) + \alpha\Big[R + \gamma \sum_a \pi(a \mid S')\,Q(S',a) - Q(S,A)\Big]$$
* Averages over all next actions instead of sampling one. 
* Removes the variance from randomly picking ${}A'{}$, at the cost of more computation per step. 

## 4.6 Maximization bias and Double Q-learning (slides 40-46)
* Taking a max over noisy estimates **overestimates** values (positive bias).
* **Double Q-learning**: keep two tables. One picks the best action, the other says how good it is: 
$$Q_1(S,A) \leftarrow Q_1(S,A) + \alpha\big[R + \gamma\, Q_2\big(S', \arg\max_a Q_1(S',a)\big) - Q_1(S,A)\big]$$
* A coin flip decides which table gets updated each step
* Twice the memory, same computation per step. 

## 4.7 Optimistic initial values
* Initialize ${}Q{}$ **high**. Every action the agent tries "disappoints" it and its ${}Q{}$ drops, so it tries the others. 
* **True**: it increases exploration, and it may improve convergence speed in practice.
* **False statements about it**: it makes the algorithm converge to a better policy. **False**: it makes ε-greedy unnecessary.
## 4.8 Summary

|  | Sarsa | Q-learning |
| --- | --- | --- |
| Target | $R + \gamma Q(S',A')$ | $R + \gamma \max_a Q(S',a)$ |
| On- or off-policy | On-policy | Off-policy |
| Learns | $q_\pi$ of its own ε-greedy policy | $q_*$ |
| Cliff walking | Safe path, better online reward | Optimal path, worse online reward |
# Topic 5: Bandits A: regret, ε-greedy, UCB1 (bandits 21)

## 5.1 The setup
* *K* arms in the set ${}A{}$. In each round ${}t=1,\dots,T{}$ you pick an arm ${}a_{t}{}$ and get a random reward from that arm's distribution ${}D_{a}{}$. Rewards are IID and in [0,1]. 
* ${}{}$$\mu(a)$= the (unknown) mean reward of ${}a{}$. 
* ${}\mu^* = max_{a}\mu(a){}$ = the best mean. a* = an optimal arm. 
* **Gap:** ${}\Delta(a) = \mu^* - \mu(a){}$, i.e. how much worse arm ${}a{}$ is than the best arm.
* **Regret (1.1)**:
$$R(T) = \mu^* \cdot T - \sum_{t=1}^{T} \mu(a_t)$$
	* What you would get by always playing the best arm, minus what your choices are worth on average.
	* Each round loses at most 1, so ${}R(T) \le T{}$ always. 
* **Goal**: sublinear regret, i.e. regret grows slower than T, so the loss per round goes to 0.

## 5.2 Non-adaptive exploration

### 5.2.1 Explore-First (Algorithm 1.1, slides 8-19)
* Try each arm ${}N{}$ times, pick the arm with the best average, then play it for the rest of the time. 
* Regret ${}O(T^{2/3}(K \log T)^{1/3}){}$ (Theorem 1.3).
* Weakness: the exploration phase wastes many rounds on bad arms.

### 5.2.2 ε-greedy (Algorithm 1.2, slides 20–25)
* Each round: with probability ${}ε_{t}{}$ explore (pick a random arm), otherwise **exploit** (pick the arm with the best average so far).
* With $\varepsilon_t = t^{-1/3}(K \log t)^{1/3}$: $\mathbb{E}[R(t)] \le O\big(t^{2/3}(K \log t)^{1/3}\big)$ (Theorem 1.4), the same as Explore-First.
* **Fixed ${}\varepsilon{}$ gives linear regret**: exploitation gets better as the estimates improve, but exploration keeps picking bad arms with the same probablity forever. Every round still costs something, so regret grows like ${}T{}$. 
* ${}\varepsilon = 0{}$ **(pure greedy) also gives linear regret:** it can lock onto a bad arm forever.

### 5.2.3 Why these are "non-adaptive"
* Their exploration schedule is fixed in advance. It does **not** depend on the rewards observed. 
* **Exam Q5:** - Exam Q5: "ε_t-greedy adapts the exploration to the observed rewards" is **False**.

## 5.3 Confidence intervals
- $n_t(a)$ = how many times arm $a$ has been played up to round $t$.
- $\bar\mu_t(a)$ = the average reward of arm $a$ so far (the exam writes $\hat\mu_t(a)$).
- **Confidence radius:** $r_t(a) = \sqrt{\dfrac{2 \log T}{n_t(a)}}$. It shrinks the more an arm is played.
- $\text{UCB}_t(a) = \bar\mu_t(a) + r_t(a)$ and $\text{LCB}_t(a) = \bar\mu_t(a) - r_t(a)$.
- The true mean $\mu(a)$ is very likely inside $[\text{LCB}_t(a), \text{UCB}_t(a)]$.

## 5.4 UCB1 (Algorithm 1.5, slides 51-59)

### 5.4.1 Algorithm
1. Try each arm once.
2. Every round after that, play the arm with the highest ${}UCB_{t}(a){}$

### 5.4.2 Why it works: optimism under uncertainty (slides 51-54)
* An arm gets a high UCV for one of two reasons:
	* It average $\bar\mu_t(a)$ is high, which is exploitation
	* Its radius ${}r_{t}(a){}$ is high because it hasn't been tried much, which is exploration 
* Exploration depends on what has been observed, so UDB1 is **adaptive**. Exam Q21: **True**.

### 5.4.3 Regret (Theorem 1.14)
* Instance-independent $\mathbb{E}[R(T)] \le O\big(\sqrt{KT \log T}\big)$ (1.9)
* Instance-dependent: $\mathbb{E}[R(T)] \le O\Big(\log T \sum_{a:\,\mu(a) < \mu^*} \dfrac{1}{\Delta(a)}\Big)$ (1.11)

## 5.6. Which regret bounds are true of UCB
* $O\big(\sqrt{KT \log T}\big)$: **True**, from (1.9).
- $O\Big(\dfrac{K \log T}{\mu(a^*) - \mu(a^{(2)})}\Big)$, where $a^{(2)}$ is the second-best arm: **True**. The second-best arm has the smallest gap, so every $\frac{1}{\Delta(a)}$ in (1.11) is at most $\frac{1}{\mu(a^*) - \mu(a^{(2)})}$, and there are fewer than $K$ terms.
- $O\Big(\sum_a \dfrac{\sqrt{\log T}}{\Delta(a)}\Big)$: **False**. $\sqrt{\log T}$ grows slower than $\log T$, and no algorithm can do better than $\log T$ (see the lower bound).

## 5.7. Lower bound (Slide 61)
- **Lai & Robbins:** no algorithm can have regret that grows slower than $\log T$: $$\lim_{T \to \infty} R(T) \ge \log T \sum_{a:\,\Delta(a) > 0} \frac{\Delta(a)}{\text{KL}(\mathcal{D}_a, \mathcal{D}_{a^*})}$$
- Hard problems are those where the arms have similar distributions.

## 5.8 Conclusion

| Algorithm                                | How it explores                             | Adaptive? | Regret                                                                   |
| ---------------------------------------- | ------------------------------------------- | --------- | ------------------------------------------------------------------------ |
| Greedy, or ε-greedy with fixed ε         | never / constant ε                          | No        | Linear, $O(T)$                                                           |
| Explore-First                            | first $N$ rounds per arm                    | No        | $O\big(T^{2/3}(K \log T)^{1/3}\big)$                                     |
| ε-greedy with decreasing $\varepsilon_t$ | random arm with probability $\varepsilon_t$ | No        | $O\big(t^{2/3}(K \log t)^{1/3}\big)$                                     |
| UCB1                                     | bonus $r_t(a)$                              | Yes       | $O\big(\sqrt{KT \log T}\big)$ and $O\big(\log T \sum_a 1/\Delta(a)\big)$ |


# Topic 6: Bandits B: Successive Elimation & Thompson Sampling

## 6.1 Successive Elimation (Algorithm 1.4, Bandits 21 slides 32-50)

#### 6.1.1 Algorithm
1. All arms start **active**.
2. Each phase: play every active arm once.
3. 3. At the end of the phase: **deactivate** every arm *a* for which some other arm *a'* has the ${}UCB_{t}(a) < \mathcal{L}CB_{t}(a').{}$
4. Repeat until we reach T.

Same intervals as UCB1: ${}UCB_{t}(a) = \bar\mu(a) + \sqrt{\frac{2\ln T}{/n(a)} }{}$  and ${}LCB_{t}(a) = \bar\mu(a) - \sqrt{\frac{2\ln T}{/n(a)}}{}$ .

### 6.1.2 Why it is fine to drop arms early
* If two intervals **dont overlap**, the arm with interval is very unlikely to be the best.
* So it can be removed permanently without losing much.

### 6.1.3 Regret
* Same as UCB1: ${}O(\sqrt{ K*T*\log T }){}$ and $O\big(\log T \sum_a 1/\Delta(a)\big)$ (Theorem 1.9).

### 6.1.4
1. **Which arms are played this phase?** All arms that are active at the start. Arms are only removed at the **end** of a phase.
2. Compute ${}\bar\mu = \frac{s}{n}{}$, the radius ${}r = \sqrt{ 2*\ln \frac{T}{n} }{}$, and UCB, and LCB for each arm.
3. After the phase, add each arms new reward and one more play, then recompute.
4. Find the **highest LCB**. Deactivate every arm whose UCB is below it.

**Shortcut:** after one more play the radius shrinks a little. So a reward of 0 pushes the UCB down, and a reward of pushes the LCB up. If two intervals were exactly touching before, this decides the answer without exact decimals.

**Worked example**: 3 active arms, n = 200 each, ${}T \approx 4{}$ so ${}r=\sqrt{ \frac{8}{200}} = 0.2{}$ 

| Arm | $s$ | $\hat\mu$ | LCB | UCB |
| --- | --- | --- | --- | --- |
| 1 | 140 | 0.7 | 0.5 | 0.9 |
| 2 | 60 | 0.3 | 0.1 | 0.5 |
| 3 | 100 | 0.5 | 0.3 | 0.7 |
* Before the phase: ${}UCB(2) = 0.5 = LCB(1){}$. Touching is not "below", so all 3 arms are still active and all 3 are played. 
* Rewards this phase: arm 1 get 1, arm 2 gets 0, arm 3 get 1. 
* After: arm 2's UCB drops below 0.5 and arm 1's LCB rises above 0.5. So UCB(2) < LCB(1) and arm 2 is deactivated. 
* Arm 3's UCB is still above every LCB, so it says.

## 6.2 Bayesian bandits 
* Same as stochastic bandits, plus one assumption: the mean vector ${}\mu{}$ is drawn from a **known prior** $\mathbb{P}$.
* **Bayesian regret:** $\text{BR}(T) = \mathbb{E}_{\mathcal{I} \sim \mathbb{P}}\big[\mathbb{E}[R(T) \mid \mathcal{I}]\big]$, i.e. the normal regret averaged over problem instances drawn from the prior.
* **Posterior** $\mathbb{P}_H$: your updated belief about $\mu$ after seeing the history $H$ (Bayes' rule).
* **Sequential update:** the posterior after round $t$ is the prior for round $t+1$.
* **Independent prior:** each arm is updated separately, so only the played arm changes.
- **Conjugate prior:** the posterior is in the same family as the prior, so it can be computed directly with a simple formula.

## 6.3 Thompson Sampling (Bandits 3, slides 16-24)

### 6.3.1 Idea
* Play each arm with the probability that it is the best arm, according to the posterior. 
* In practice (Algorithm 3.2), every round:
	* Sample a mean for every arm from its posterior
	* Play the arm with the **highest sample**.
	* Update that arm's posterior with the reward.

### 6.3.2 Beta-Bernoulli case (slides 18-22)
* Rewards are 0 or 1. Start each arm with Beta(1, 1), the uniform distribution.
* Reward 1: $\alpha \leftarrow \alpha + 1$. Reward 0: $\beta \leftarrow \beta + 1$.
* Posterior mean: ${}\frac{\alpha}{\alpha + \beta}{}$. More plays make the distribution narrower.

### 6.3.3 How it balances exploration and exploitation
* **High posterior mean** means the arm usually gives a big sample: **exploitation**.
* **High posterior variance** (a wide distribution, because the arm has been tried only a few times) means it sometimes gives a big sample: **exploration**.


### 6.5. Summary

|                  | UCB1                          | Successive Elimination           | Thompson Sampling                              |
| ---------------- | ----------------------------- | -------------------------------- | ---------------------------------------------- |
| Picks            | the arm with the highest UCB  | every active arm, in turn        | the arm with the highest posterior sample      |
| Explores through | the confidence bonus          | keeping arms until clearly worse | posterior variance                             |
| Random choices?  | No                            | No                               | Yes                                            |
| Regret           | $O\big(\sqrt{KT \log T}\big)$ | $O\big(\sqrt{KT \log T}\big)$    | $\text{BR}(T) \le O\big(\sqrt{KT \log T}\big)$ |


# Topic 7: Policy gradients  DQN (RL 13, RL 10)

## Part A: Policy Gradient methods (Lecture 7 - RL 13)

### 7.1 The idea
* Learn the policy directly: ${}\pi(a | s, \theta){}$, where ${}\theta{}$ are the policy's parameters. No value function is needed to pick actions. 
* Improve it with gradient ascent on a performance measure $J(\boldsymbol\theta)$:
$$\boldsymbol\theta_{t+1} = \boldsymbol\theta_t + \alpha \nabla J(\boldsymbol\theta_t)$$
* Three families (slide 4):
	* **Value-based**: learns a value function; the policy is implicit (e.g. ${}\varepsilon{}$-greedy)
	* **Policy-based**: learns only a policy.
	* **Actor-critic**: learns both.
* **Advantages**: can learn stochastic policies; action probabilities change smoothly; works with continuous or high-dimensional actions. 
* Disadvantages: can get stuck in local optima. 

### 7.2 Policy gradient theorem 
$$\nabla J(\boldsymbol\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s,a)\, \nabla \pi(a \mid s, \boldsymbol\theta)$$
- $\mu(s)$ = how often state $s$ is visited under $\pi$.
- Why it matters: the gradient does **not** need the derivative of the state distribution, which is unknown.

### 7.3 REINFORCE (slides 24-36)
#### 7.3.1 Update (slides 26-27)
$$\boldsymbol\theta_{t+1} = \boldsymbol\theta_t + \alpha\, G_t\, \nabla \ln \pi(A_t \mid S_t, \boldsymbol\theta_t)$$
- $\nabla \ln \pi = \dfrac{\nabla \pi}{\pi}$.
- **Times $G_t$:** move most toward actions that gave a **high return**.
- **Divided by $\pi$:** otherwise actions that are chosen often would get updated more often and win even if they aren't the best.
- The exam's algorithm has an extra $\gamma^t$ for the discounted case: $\boldsymbol\theta \leftarrow \boldsymbol\theta + \alpha \gamma^t G \nabla \ln \pi(A_t \mid S_t, \boldsymbol\theta)$.

#### 7.3.2 Derivation from the theorem (slide 25)
1. The sum over states weighted by $\mu(s)$ is an expectation over the states visited under $\pi$: $\nabla J(\boldsymbol\theta) \propto \mathbb{E}_\pi\big[\sum_a q_\pi(S_t,a) \nabla\pi(a \mid S_t,\boldsymbol\theta)\big]$
2. Multiply and divide by $\pi(a \mid S_t, \boldsymbol\theta)$. Now the sum over $a$ is an expectation over the action, so replace $a$ by the sampled action $A_t$: $= \mathbb{E}_\pi\Big[q_\pi(S_t,A_t)\, \dfrac{\nabla\pi(A_t \mid S_t,\boldsymbol\theta)}{\pi(A_t \mid S_t,\boldsymbol\theta)}\Big]$
3. Replace $q_\pi(S_t,A_t)$ by $G_t$, because $\mathbb{E}_\pi[G_t \mid S_t, A_t] = q_\pi(S_t,A_t)$: $= \mathbb{E}_\pi\Big[G_t\, \dfrac{\nabla\pi(A_t \mid S_t,\boldsymbol\theta)}{\pi(A_t \mid S_t,\boldsymbol\theta)}\Big]$
4. The quantity inside can be sampled every step, so use it as the update. That gives the REINFORCE update above

#### 7.3.3 Properties
* **On-policy**: it learns from episodes generated by its current policy.
* **Monte Carlo**: it needs the full episode to compute ${}G_{t}{}$.
* The basic version uses **no baseline**.

#### 7.3.4 REINFORCE with baseline

$$\boldsymbol\theta_{t+1} = \boldsymbol\theta_t + \alpha\big(G_t - b(S_t)\big) \nabla \ln \pi(A_t \mid S_t, \boldsymbol\theta_t)$$
* ${}b(s){}$ can be anything that doesn't depends on ${}a{}$. Natural choice: ${}\hat{v}(S_{t}, w){}$.
* The expected update stays the same, but the **variance drops**, so it learns faster.

### 7.4 One-step Actor-Critic

#### 7.4.1 Update (Slide 39)
* TD error: ${}\delta = R + \gamma \hat{v}(S', w) - \hat{v}(S, w){}$
* Critic (line 10): ${}w \leftarrow w + \alpha^w \delta \nabla \hat{v}(S,w){}$
* Actor (line 11): ${}\theta \leftarrow \theta+\alpha^\theta I \delta \nabla \ln(\pi(A|S,\theta)){}$, where ${}I = \gamma^t{}$.
* It is REINFORCE with the full return ${}G_{t}{}$ replaced by the **one-step return** ${}R + \gamma \hat{v}(S', w){}$, and with ${}\hat{v}(S,w){}$ as the **baseline**.

#### 7.4.2 Roles
* **Actor** = the policy ${}\pi((a | s, \theta)){}$. It picks actions and is improved in line 11. 
* **Critic** = the state-value function ${}\hat{v}(s,w){}$. It learns the state values of the current policy and is improved in line 10.
* The critic's **only output** is the TD error ${}\delta{}$, a single number that judges the action just taken:
	* ${}\delta > 0:{}$ the action turned out better than expected, so make it **more** likely.
	* ${}\delta < 0{}$: worse than expected, so make it **less** likely. 
* **On-policy:** the critic judges the policy the actor is currently following. 
* Fully online: it updates every step. 

#### 7.4.3. REINFORCE vs Actor-Critic
|  | REINFORCE | One-step Actor–Critic |
| --- | --- | --- |
| Learns | policy only | policy and state-value function |
| Target | full return $G_t$ (Monte Carlo) | one-step return $R + \gamma \hat v(S', \mathbf{w})$ (TD) |
| Baseline | none (basic version) | $\hat v(S, \mathbf{w})$ |
| Updates | after the episode ends | every step |
| On- or off-policy | on-policy | on-policy |
| In common | policy gradient + stochastic gradient ascent | policy gradient + stochastic gradient ascent |

## Part B: Deep Q-Network

### 7.5. Function Approximation

* Too many states for a table, approximate ${}\hat{q}(s,a,w){}$ with weights ${}w{}$, for example a neural network. 
* Semi-gradient Sarsa: ${}w \leftarrow w + \alpha[R + \gamma \hat{q}(S',A',w) - \hat{q}(S,A,w)] \nabla \hat{q}(S,A,w){}$
* "Semi-gradient": the target also depends on ${}w{}$, but we don't take its gradient (RL9, slides 20-21).

### 7.6. DQN

#### 7.6.1. What it is
* Q-learning combined with a deep convolutional network. It learned 49 Atari games from raw pixels, with the same network and settings for every game.
* Acts ${}\varepsilon{}$-greedily, with ${}\varepsilon{}$ decreasing over time. 


#### 7.6.2. Three changes to Q-learning
1. Experience replay
2. Duplicate (target) network
3. Clipping the error term to [-1, 1]

#### 7.6.3. Update
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\big[R_{t+1} + \gamma \max_a \tilde q(S_{t+1}, a, \mathbf{w}_t) - \hat q(S_t, A_t, \mathbf{w}_t)\big] \nabla \hat q(S_t, A_t, \mathbf{w}_t)$$
* ${}\hat{q}{}$ = the network being trained. $\tilde q$ = the duplicate network, kept fixed for ${}C{}$ updates. 

**How the pieces of the DQN algorithm connect:**
![[Pasted image 20261021122448.png]]

#### 7.6.4. Experience replay
* Every transition ${}(S,A,R,S'){}$ is stored in a replay memory. Training uses **random mini-batches** drawn from it.
* This is allowed because Q-learning is off-policy, so it doesn't need consecutive steps from one trajectory.
* **Three advantages:**
	1. **More data-efficient**: each experience is reused for many updates.
	2. **Lower variance**: consecutive updates aren't correlated with each other.
	3. **More stable:** the experiences used for training no longer depend on the current weights.

#### 7.6.5. Duplicate network
* **Problem:** the target is computed with the same weights that are being updated. That's like chasing a moving target, and it can cause **oscillation** or **divergence**. 
* **Fix:** every ${}C{}$ steps, copy ${}\hat{q}{}$'s weights into ${}\hat{q}{}$, then keep ${}\tilde{q}{}$ fixed. Targets are computed with these **older parameters**.

|                   | REINFORCE            | Actor–Critic                      | DQN                               |
| ----------------- | -------------------- | --------------------------------- | --------------------------------- |
| Learns            | policy               | policy and $\hat v$               | $\hat q$                          |
| On- or off-policy | on                   | on                                | off                               |
| MC or TD          | MC                   | TD                                | TD (Q-learning)                   |
| Key idea          | $G_t \nabla \ln \pi$ | TD error $\delta$ as the critique | replay memory + duplicate network |

# SUPER SECRET KNOWLEDGE!
Here you can find some tips and tricks for certain problems. Things you need to solve questions even though you might not understand the whole concept.

------------
## How do I know if an algorithm is biased or unbiased?

One question sorts all of them:

### Solution:
**Does the target contain a guess (V, Q or ${}\hat{v}{}$)?**
* **No, only real rewards** ${}\to{}$ **UNBIASED**, but with high variance
* **Yes, a guess** ${}\to{}$ **BIASED** (the guess can be wrong), but **lower variance**

---------------

