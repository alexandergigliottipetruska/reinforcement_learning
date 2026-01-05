# Linear Value Function Approximation

Lookup tables of the form $Q(s, a)$ where each state s has an entry for all possible actions at that state a, does not scale to enormous state ranges or countably infinite state spaces. This is because there are too many actions/states to store in memory and it  would take too long to learn the value of each state individually. 

**Value Function Approximation** is necessary allows the agent to generalize from seen states to unseen states, eliminating the need to learn the value of every state-action pair, by learning a function $\hat{q}(s, a, w) \approx q_{\pi}(s, a)$ with parameter w. Possible differentiable function approximators include linear combinations of features and neural networks, although here the focus will be on linear ones. 

Incremental methods rely on minimizing the MSE between the true value function $q_{\pi}(s, a)$ and the approximate value $\hat{q}(s, a, w,)$ using stochastic gradient descent. True value functions are not usually not known and are provided via targets, using Sarsa the target is

$$r + \gamma max_{a'}Q(s', a')$$

And the gradient becomes

$$\Delta w = \alpha (q_{\pi}(s, a) - \hat{q}(s, a, w))\nabla_{w}\hat{q}(s, a, w) = \alpha \delta x(s, a)$$

Note gradients are not taken with respect to the targets, only the approximations. And the value function approximator is,

$$\hat{q}(s, a, w) = \sum_{j=1}^{n} x_j(s, a) w_j = x(s, a)^{\mathrm{T}} w$$

Batch methods can also be used, via least-squares, but are omitted in this section. Environments used are taken from Classic Control in gymnasium.


