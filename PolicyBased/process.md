The environment is **Pendulum** from gymnasium, a classics control environment with a continuous action space representing the torque applied to the free end of a pendulum. The state space consists of observations of the
x and y coordinates and the angular velocity ($\theta$). 

The minimum reward is 0 and the maximum is around -16.3, where the former corresponds to a completely upright pendulum, and the reward function can be described as follows (note $\theta_{dt}$ is angular velocity),

$$r = - (\theta^2 + 0.1 * \theta_{dt}^2 + 0.0001 * \text{torque}^2)$$

Given this environment, a one-step Actor Critic agent will be implemented that learns a Gaussian probability density from which to sample actions. 
