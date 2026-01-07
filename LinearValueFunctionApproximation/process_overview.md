The environments used to test the model are Cart Pole and Mountain Car. Linear epsilon decay was used as it was found that exponential decay caused the 
model to stop exploring too early and exploit with insufficient knowledge of the environment, as the rewards are very sparse. 

Radial Basis Function (RBF) was used to extract features from the observations as it provides a set of rich, high-quality features from the observation space
with which the agent can used to learn its parameters for estimation. Using bare features (just ones provided in observation space) was insufficient and resulted
in the agent learning nothing over the entire training process.

A separate linear model was trained for each action to allow each action to have its own relation to state space, where some features may be relevant to one but
unimportant for the other. 

Cart Pole takes aims to balance a pole on a cart by taking into account the pole's angle, its angular velocity, the cart's velocity, and its position.
This creates a continuous state space with discrete actions (push cart to right or left) necessitating value approximation. Rewards are +1 for every time
step taken, including terminal step, ensuring the agent learns to keep the pole upright as long as possible to maximize reward. Termination occurs if
pole angle is greater than $\pm 12^\circ$ or cart position greater than $\pm 2.4$ (center of cart reaches edge of display). Q-learning was used for providing
the target.

Mountain Car tries to reach the top of a hill by utilising a car at the bottom of valley to generate momentum by repeatedly going as far back and forth as possible with the car.
It is a continuous state space that takes into account the velocity of the car and its position along the x-axis. Moreover, it has a discrete action space
that consists of accelerating to the left, doing nothing, and accelerating to the right. To reach the hill as quickly as possible, the agent receives a reward of -1 for each time step taken to do so.
Lastly, gravity and force are simulated to determine transitions to subsequent positions and velocities, creating the effect of a hill's slope, and episode termination occurs when car's position
is greater than or equal to 0.5 (goal position on the top of the hill).
