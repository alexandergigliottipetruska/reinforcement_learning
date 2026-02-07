# Soft-Actor Critic (SAC)
Actor Critic methods learn approximations to both policy and value functions. The Critic learns a value function which is used for bootstrapping and criticizes policy's action selections by assigning credit, while the Actor is the learned policy.

Soft-Actor Critic (SAC) is an actor-critic algorithm designed to address very high sample complexity and brittle convergence found in traditional model-free Deep RL algorithms, by basing itself on the maximum entropy RL framework.
Therefore, it aims to maximize expected reward and entropy simultaneously, succeeding while acting as randomly as possible. It is sample efficient, stable, and can be extended to complex, high dimensional data.

![Humanoid score](https://github.com/alexandergigliottipetruska/reinforcement_learning/raw/main/assets/humanoid_demo2_smaller.gif)
![Humanoid score](https://github.com/alexandergigliottipetruska/reinforcement_learning/raw/main/assets/real_ant_demo.gif)
![Humanoid score](https://github.com/alexandergigliottipetruska/reinforcement_learning/raw/main/assets/cheetah_demo1.gif)

The three main experiments with their rolling average return over 100 episodes: **Humanoid** (5202.24), **Ant** (4834.88), and **HalfCheetah** (8103.09). 

## Maximum Entropy Framework
The maximum entropy objective favors stochastic policies,

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim p_{\pi}}[r(s_t, a_t) + \alpha H(\pi (\cdot |s_t))]$$ 

And offers several advantages, such as incentivizing exploration (while giving up unpromising paths), capturing multiple modes of near-optimal behaviour (assigning equal probability mass when confronted with multiple attractive options), and improving learning speed.
Note H is the entropy parameter of the policy, measuring randomness of action selection. If maximized, it increases exploration, and if minimized, exploitation. Therefore, the agent
gets a bonus reward at each time step, proportional to entropy of the policy. Note that $\alpha$ is the tradefoff coefficient, determining the relative importance of the entropy with respect to the expected reward.

## Algorithm 
It learns by alternating optimization w.r.t. to the policy, $\pi_{\theta}$ and two Q-functions, $Q_{\phi_1}$, $Q_{\phi_2}$, either with a fixed entropy regularization coefficient or a learnt one.

It shares a few things in common with TD3:
- Regression to a single target via MSBE
- Clipped double-Q
- Shared target computed by using target Q-networks, polyak averaging their parameters during training.

While retaining some differences:
- Target includes derived entropy regularization term.
- Next-state actions used in the target comes from the current policy (not target policy).
- No explicit target policy smoothing.

Approximating the Q-function can be done using samples, 

$$Q^{\pi}(s, a) \approx r + \gamma (Q^{\pi} (s', \tilde{a}') - \alpha logp(\pi (\tilde{a}'|s')$$

Where r and s' comes from the replay buffer, D, and $\tilde{a}^{\prime} \sim \pi_{\theta} (\cdot | s')$, the current policy.

Using clipped double-Q, SAC takes the minimum between the Q-function approximators to avoid overestimation and accumulation of error when using sample approximation to compute the targets,

$$
y(r, s', d) = r + \gamma (1 - d) \left( \min_{j=1,2} Q_{\phi_{\text{targ}, j}}(s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}'|s^{\prime}) \right)
$$

and the loss function for the Q-networks is the MSBE between the estimation and the target,

$$L(\phi_i, D) = \mathbb{E}_{(s, a, r, s^{\prime}, d) \sim D} [Q_{\phi_i (s, a)} - y(r, s^{\prime}, d)]$$

Learning the policy involves maximizing the expected future return and expected future entropy. However, optimizing it requires the reparametrization trick,
drawing a sample from $\pi_{\theta}(\cdot | s)$ by computing a deterministic function of state, policy parameters, and independent noise. Using a squashed Gaussian function,

$$
\tilde{a}_{\theta}(s, \xi) = \tanh \big(\mu_{\theta}(s) + \sigma_{\theta}(s) \odot \xi \big), \quad \xi \sim \mathcal{N}(0, 1)
$$

Which allows us to write an expectation over action to one over noise, eliminating dependence on parameters. Note that standard deviation parametrization is state dependent, the output of a neural network.

$$\mathbb{E}_{a \sim {\pi_{\theta}}}[Q^{\pi}(s, a) - \alpha \log \pi_{\theta}(a | s)]$$

to become, substituting $Q^{\pi_{\theta}}$ with a function approximator, $\min_{j=1, 2} Q_{\phi_j}$, and optimizing the policy,

$$\max_{\theta}\mathbb{E}_{s \sim D, \xi \sim \mathcal{N}} [\min_{j=1, 2}Q_{\phi_j}(s, \tilde{a}_{\theta}(s, \xi)) - \alpha \log \pi_{\theta} (\tilde{a}_{\theta}(s, \xi)|s))]$$

When evaluating, at test time, remove the stochasticy by using the mean action instead of sampling from the distribution, contributes to the performance.

## Sources

Here is a list of sources used for this README.md and for learning about Soft Actor-Critic (SAC):

1. **Original SAC Paper** – *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*  
   [Link to paper](https://arxiv.org/abs/1801.01290) by Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.

2. **OpenAI Spinning Up in Deep RL** – SAC documentation and tutorial  
   [Link to Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)

3. **CleanRL SAC Implementation** – Minimal, reproducible PyTorch implementation  
   [Link to CleanRL SAC](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)




