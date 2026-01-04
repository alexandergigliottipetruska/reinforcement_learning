# Temporal Difference (TD)

**Temporal Difference** algorithms involve computing estimates based on the value function at the subsequent time step. This provides an advantage over dynamic programming algorithms which require
a perfect model of the environment and may involve heavy computation costs in sweeping over the state space by using immediate experience at the next time step to compute updates, while preserving bootstrapping (computing
estimates based on existing estimates). When compared to Monte-Carlo algorithms, they do not need to wait until termination to learn. This avoids two huge problems with MC algorithms in general:
1. In very long episodes, MC learning can become very slow.
2. MC can get stuck in non-terminal states and never learn. TD avoids by learning during the episode to avoid them and update their policy without waiting for termination.

The simplest possible TD(0) update provides the basis of all the other algorithms' updates, 

$$V(s) \gets V(s) + \alpha (r + \gamma V(s') - V(s))$$

Where r is the reward at the subsequent time step, s is the current state $S_t$, and s' is the next state $S_{t+1}$. In TD algorithms, the error is related to the difference in the value functions between successive time-steps.

$$\delta_t = r + \gamma V(s') - V(s)$$

In practice, however, the algorithms use action-state instead of state values.

The following algorithms will be implemented and tested on Toy Test environments:
- **Q-learning**
- **SARSA**
- **Expected SARSA**

The aim of this project is to become acquainted with standard reinforcement learning algorithms, and, in particular, famous TD(0) algorithms such as SARSA or Q-learning. 
Moreover, core concepts in RL such as exploration vs. exploitation, discount factor affecting present value of future actions, $\epsilon$-greedy action selection, etc. are
also aimed to be understood concretely. 

