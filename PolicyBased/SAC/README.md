# Soft-Actor Critic (SAC)
Actor Critic methods learn approximations to both policy and value functions. The Critic learns a value function which is used for bootstrapping and criticizes policy's action selections by assigning credit, while the Actor is the learned policy.

Soft-Actor Critic (SAC) is an actor-critic algorithm designed to address very high sample complexity and brittle convergence found in traditional model-free Deep RL algorithms, by basing itself on the maximum entropy RL framework. Some of the classical problems of model-free Deep RL are sample inefficency and intense hyperparameter tuning for different problem settings. On-policy methods in particular suffer from the former, requiring a huge number of gradient steps and samples per step, especially with higher dimensional tasks. On the other hand, off-policy methods suffer from stability and convergence due to a combination of nonlinear function approximation, off-policy learning, and using a separate actor network for maximization in Q-learning. 

The maximum entropy RL objective aims to maximize expected reward and expected entropy simultaneously, succeeding while acting as randomly as possible. The resulting algorithm, SAC, is sample efficient, stable, and extends to a variety of different problems and environments, even complex and high dimensional ones, while avoding intricate hyperparameter tuning. This step is essential for real-world applicability.

![Humanoid score](https://github.com/alexandergigliottipetruska/reinforcement_learning/raw/main/assets/humanoid_demo2_smaller.gif)
![Ant score](https://github.com/alexandergigliottipetruska/reinforcement_learning/raw/main/assets/new_ant_gif.gif)
![HalfCheetah score](https://github.com/alexandergigliottipetruska/reinforcement_learning/raw/main/assets/new_cheetah_gif.gif)

The three main experiments with their rolling average return over 100 episodes: **Humanoid** (5202.24), **Ant** (5814.51), and **HalfCheetah** (13571.46). The top evaluation scores (average return over 10 episodes rollout every 20k time steps) were **HalfCheetah** (13952.70), **Ant** (5814.51), and **Humanoid** (5543.15).  

## Maximum Entropy Framework
The maximum entropy objective includes both the expected sum of rewards and expected entropy. 

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim p_{\pi}}[r(s_t, a_t) + \alpha H(\pi (\cdot |s_t))]$$ 

where H is the entropy parameter of the policy, measuring the randomness of action selection, so the agent receives a bonus reward at each time step proportional to the entropy of the policy. If it is maximized, it increases exploration, while if minimized, increases exploitation. $\alpha$ is the tradefoff coefficient, determining the relative importance of the entropy with respect to the expected reward.

This objective offers many advantages compared to the standard one, favoring stochastic policies.  It incentivizes exploration while abandoning unpromising paths, assigns equal probability mass when confronted with multiple attractive options (different modes of near-optimal behaviour), improves learning speed, and is significantly more robust.

## Algorithm 
It learns by alternating optimization w.r.t. to the policy, $\pi_{\theta}$ and two Q-functions, $Q_{\phi_1}$, $Q_{\phi_2}$, either with a fixed entropy regularization coefficient or a learnt one. Comparing it to another off-policy model-free RL algorithm, TD3, it shares using MSBE for regression to a single target, clipped double-Q, and polyak averaging target Q-networks over training, while presenting significant differences. For example, the target includes a derived entropy regularization term, next-state actions used in the target come from the current policy, and there is no explicit target policy smoothing. 

Approximating the Q-function can be done using samples from a replay buffer D, 

$$Q^{\pi}(s, a) \approx r + \gamma (Q^{\pi} (s', \tilde{a}') - \alpha logp(\pi (\tilde{a}'|s')$$

Where r and s' comes from the replay buffer, D, and $\tilde{a}^{\prime} \sim \pi_{\theta} (\cdot | s')$, is an action sampled from the current policy, not the buffer.

Using clipped double-Q, SAC takes the minimum between the Q-function approximators to avoid overestimation and accumulation of error when using sample approximation to compute the targets by mitigating positive bias, significantly increasing training speed on complex tasks,

$$
y(r, s', d) = r + \gamma (1 - d) \left( \min_{j=1,2} Q_{\phi_{\text{targ}, j}}(s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}'|s^{\prime}) \right)
$$

and the loss function for the Q-networks is the MSBE between the estimation and the target,

$$L(\phi_i, D) = \mathbb{E}_{(s, a, r, s^{\prime}, d) \sim D} [Q_{\phi_i (s, a)} - y(r, s^{\prime}, d)]$$

Learning the policy involves maximizing the expected future return and expected future entropy. However, optimizing it requires the reparametrization trick, which results in a lower variance estimator. This trick draws a sample from $\pi_{\theta}(\cdot | s)$, computing a function of the state, policy params, and an independent noise, using a squashed Gaussian, 

$$
\tilde{a}_{\theta}(s, \xi) = \tanh \big(\mu_{\theta}(s) + \sigma_{\theta}(s) \odot \xi \big), \quad \xi \sim \mathcal{N}(0, 1)
$$

Which eliminates dependence on parameters by allowing us to write an expectation over action to one over noise. 

$$\mathbb{E}_{a \sim {\pi_{\theta}}}[Q^{\pi_{\theta}}(s, a) - \alpha \log \pi_{\theta}(a | s)]$$
$$
= \mathbb{E}_{\xi \sim \mathcal{N}} \Big[ Q^{\pi_{\theta}}(s, \tilde{a}_\theta(\xi; s)) - \alpha \log \pi_{\theta}(\tilde{a}_\theta(\xi; s) \mid s) \Big]
$$

with which, substituting $Q^{\pi_{\theta}}$ with a function approximator, $\min_{j=1, 2} Q_{\phi_j}$, one can obtain the policy loss and optimize it according to,

$$\max_{\theta}\mathbb{E}_{s \sim D, \xi \sim \mathcal{N}} [\min_{j=1, 2}Q_{\phi_j}(s, \tilde{a}_{\theta}(s, \xi)) - \alpha \log \pi_{\theta} (\tilde{a}_{\theta}(s, \xi)|s))]$$

Crucially, the standard deviation parametrization is state dependent, the output of a neural network, rather than an independent parameter. During evaluation, stochasticity is removed by using the mean action instead of sampling from the distribution, improving the performance.

## Entropy Autotune
Autotuning the entropy allows for the algorithm to balance the exploration and exploitation via controlling the tradeoff parameter, $\alpha$, that determines how much the expected entropy is weighted against the expected reward in the maximum entropy objective. This parameter is updated online, and requires updating the loss for the policy and value network w.r.t. to their parameters $\theta$ and the adapted tradeoff parameter, $\alpha$. 

The temperature, $\alpha$, is optimized with respect to the following loss:

$$\mathcal{L}(\alpha) = \mathbb{E}_{s \sim D,\, a \sim \pi_\phi} \Big[ -\alpha \big( \log \pi_\phi(a|s) + \mathcal{H}_{\mathrm{target}} \big) \Big]$$

This form differs from the one proposed in the SAC paper, which expresses the meta-loss in terms of the Q-function:

$$\mathcal{L}_{\text{meta}}(\alpha) = \mathbb{E}_{s_0 \sim D_0} \Big[ - Q_{\omega}(s_0, \pi^{\mathrm{det}}_{\phi_{t+1}}(\alpha)(s_0)) \Big]$$

In practice, implementations use the log probabilities of actions sampled from the current policy to tune $\alpha$ so the policy's entropy matches the target entropy. The advantage of this approach is that the temperature parameter is adjusted online during training, allowing the model to explore more when needed, and focus on exploitation once an effective policy is learned.


## Sources

Here is a list of sources used for this README.md and for learning about Soft Actor-Critic (SAC):

1. **Original SAC Paper** – *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*  
   [Link to paper](https://arxiv.org/abs/1801.01290) by Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine.

2. **OpenAI Spinning Up in Deep RL** – SAC documentation
   [Link to Spinning Up SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)

3. **CleanRL SAC Implementation** – Minimal, reproducible PyTorch implementation  
   [Link to CleanRL SAC](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)

4. **Entropy Autotune** - *Meta-SAC: Auto-tune the Entropy Temperature of Soft Actor-Critic via Metagradient*
   [Link to paper](https://arxiv.org/pdf/2007.01932) by Yufei Wang and Tianwei Ni.




