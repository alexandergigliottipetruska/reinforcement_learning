# Policy Based Methods

**Policy-based** methods select actions based on a parametrized policy, not action values, and maximize a scalar performance $J(\theta)$ with respect to policy parameters $\theta\$ via gradient ascent.

$$\theta_{t+1} = \theta_{t} + \alpha \widehat{\nabla J}(\theta)$$

In the episode case, perforamnce is the value of the starting state under the parametrized policy, while in the continuing case it is the average reward rate. There are a few big advantages of policy gradient methods over action value methods:
1) Can deal with continuous action spaces
2) Selection of actions with arbitrary possibilities
3) Allows inclusion of prior knowledge.

Using the **Policy Gradient Theorem**, the expectation of sample gradient is proportional to actual gradient of performance measure as a function of the policy parameter, 

$$\nabla J(\theta) \propto \sum_{s} \mu (s) \sum_{a} q_{\pi}(s, a) \nabla \pi (a | s, \theta)$$

The above equation is for the episodic case. $\mu(s)$ is the state visition distribution under policy $\pi$, either a Markov stationary distribution in the continuous case (time spent in state s in the long run) or visition distribution in episodic case.
A natural consequence is that the more often a state is visited, the more it counts,

$$\mu(s) = \sum_{t=0}^{\infty} \gamma^{t} P(S_t = s | \pi)$$

## Actor-Critic Methods
Actor Critic methods learn approximations to both policy and value functions. The Critic learns a value function which is used for bootstrapping and criticizes policy's action selections by assigning credit, while the Actor is the learned policy.
Moreover, a one-step AC replaces full-return with one-step return and uses a learned state value function as a baseline.

$$\theta_{t+1} = \theta_{t} + \alpha (R_{t+1} + \gamma \hat{v} (S_{t+1}, w) - \hat{v} (S_t, w)) \frac{\nabla \pi(A_t | S_t, \theta_t)}{\pi (A_t | S_t, \theta_t)}$$

States, actions, and rewards are processed as they occur and never revisited. The algorithm implemented follows the one presented in Sutton's RL book.

## Continuous Actions
Policy-based methods can deal with large action spaces, even an infinite number, by learning statistics of probability distributions. With policy parametrization, policy is probability density over real-valued scalar action.
Where mean and standard deviation are given by parametric function approximation. In this case, the probability density for a Gaussian distribution is used,

$$\pi (a | s, \theta) = \frac{1}{\sigma (s, \theta) \sqrt{2 \pi}} e^{\frac{(a - \mu (s, \theta))^2}{2\sigma (s, \theta)^2}}$$

To get $\mu (s, \theta)$ and $\sigma (s, \theta)$, the param vector must be divided into two parts, $\theta=[\theta_{\mu}, \theta_{\sigma}]$.
