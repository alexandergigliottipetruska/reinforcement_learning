# Deep Q Network (DQN)

Deep Q Networks (DQN) are another method that allows an agent to generalize from seen to unseen states, using a neural network as a nonlinear estimator. It can represent far more complex functions than linear estimators and was created specifically to master Atari games - of which several will be done in this section.

DQN deals with challenges that are common in reinforcement learning:
- Learning is done from a sparse, noisy, and delayed reward signal.
- Data samples are not independent - sequences of highly correlated states.
- No underlying fixed data distribution as it changes as algorithm learns new behaviours.

In order to deal with these, DQN contains 3 main innovations:
1. **Experience replay**: random selection of stored experiences which allow it learn a varied set of experiences
   - Randomization gets rid of problems with temporally correlated data.
   - By randomly sampling transitions using experience relay, DQN deals with the last two problems. 
2. Convolutional networks make sense of game screen outputs, captures image regions and spatial relationships between objects on the screen.
3. Separate target network prevents issue of constantly shifting target values to adjust network values which is a huge source of instability.
   - Instead, update target network's weights either periodically and all at once, or slowly but frequently to primary Q-network's values.

**DQN Algorithm**: Experience replay and fixed Q-targets. (adapted from Silverman's lectures)
1. Take an action $a_t$ according to $\epsilon$-greedy policy
2. Store transition $(s_t, a_t, r_{t+1}, s_{t+1})$ in replay memory D
3. Sample random mini-batch of transitions $(s, a, r, s')$ from D
4. Compute Q-learning w.r.t. old, fixed parameters $w^{-}$
5. Optimize MLE between Q-network and Q-learning targets

$$L_i(w_i) = \mathbb{E}_{s, a, r, s' \sim D_i}[(r + \gamma \max_{a'}Q(s', a'; w_i^{-}) - Q(s, a; w_i))^2]$$

- Use a variant of stochastic gradient descent
- Use old params for targets and new params for approximation.
