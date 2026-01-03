Q-learning is a temporal difference algorithm that computes updates by choosing the action at the next time-step that maximizes the $Q(s', a')$. It 
utilizes bootstrapping as it uses existing estimates to compute new ones. Moreover, it is off-policy, the control policy that generates behaviours
(which states to visit and update) is different from the target policy that is being evaluated and is greedy with respect to the action value function.
The update formula is the following:

$$Q(s, a) \gets Q(S, A) + \alpha (r + \gamma max_aQ(s', a') - Q(s, a))$$

For reinforcement learning algorithms, it is essential to balance exploration and exploitation. When training, at the beginning the agent should be mostly exploring to discover its environment and towards the end should be exploiting its accumulated knowledge as much as possible. To implement this, an exponential decay for $\epsilon$ was used, with a minimum set to ensure a small amount of exploration still remained at the very end.

$$min(0.01, \epsilon_0 \lambda^t)$$

Finally, the standard learning rate $$\alpha$$ was decayed based on how often a particular action was taken given a state, for all state and action combinations.
