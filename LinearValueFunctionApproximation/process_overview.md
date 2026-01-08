The environment used to test the model is Cart Pole. Linear epsilon decay was used as it was found that exponential decay caused the 
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
