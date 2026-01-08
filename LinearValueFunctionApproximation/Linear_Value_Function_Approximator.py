import gymnasium as gym
import numpy as np
import sklearn
import pickle
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler

class Estimator():
    def __init__(self, env, scaler, featurizer, learning_rate):
        self.action_space = env.action_space.n
        self.alpha = learning_rate
        self.scaler = scaler
        self.featurizer = featurizer
        self.n_features = 400
        # Initialize weights
        self.w = np.zeros((self.action_space, self.n_features))
        self.b = np.zeros(self.action_space)

    def update(self, s, a, y):
        x = self.extract_features(s, a)
        estimate = self.predict(s, a)
        self.w[a] = self.w[a] + self.alpha * (y - estimate) * x
        self.b[a] = self.b[a] + self.alpha * (y - estimate)

    def extract_features(self, obs, a):
        obs_scaled = self.scaler.transform(obs.reshape(1, -1))[0]
        features = self.featurizer.transform(obs_scaled.reshape(1, -1))
        return features[0]

    def predict(self, s, a=None):
        if a is not None:
            x = self.extract_features(s, a)
        else:
            return np.array([self.extract_features(s, a) @ self.w[a] + self.b[a] for a in range(self.action_space)])

        return x @ self.w[a] + self.b[a]

# Run the RL program
def run(episodes, render, training):
    # Create env
    env = gym.make("CartPole-v1", render_mode='human' if render else None)

    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=0.5, n_components=100)),
        ("rbf3", RBFSampler(gamma=0.25, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.1, n_components=100))
        ])
    
    featurizer.fit(scaler.transform(observation_examples))

    if training:
        # Create a new estimator
        estimator = Estimator(env, scaler, featurizer, learning_rate=0.01)
    else:
        # load a currently existing trained estimator
        f = open('LinearValueFunctionApproximation/cart_pole.pkl', 'rb')
        estimator = pickle.load(f)
        f.close()

    # Define parameters
    discount_factor_g = 0.95
    initial_epsilon = 1.0
    epsilon = initial_epsilon
    epsilon_decay_rate = 0.9995

    # Create random generator
    random_generator = np.random.default_rng()

    # Track rewards
    rewards = np.zeros(episodes)
    
    
    # Loop through episodes
    for i in range(episodes):
        truncated = False
        terminated = False
        
        # Initial state
        state = env.reset()[0] # observation space is of shape, (4,), with position and movement

        while (not terminated and not truncated):

            if training and random_generator.random() < epsilon:
                action = env.action_space.sample()
            else:
                action_values = estimator.predict(state)
                action = np.random.choice(np.flatnonzero(action_values == np.max(action_values)))
                
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Calculate target
            target = reward + discount_factor_g * np.max(estimator.predict(next_state))

            if training:
                estimator.update(state, action, target)

            state = next_state

            rewards[i] += reward

        # Linear Epsilon decay
        epsilon = max(0.01, epsilon - 1e-4)

    # Close env
    env.close()

    average_rewards = np.zeros(episodes)
        
    for i in range(episodes):
        average_rewards[i] = np.mean(rewards[max(0, i-100):i+1])

    if training:
        plt.plot(average_rewards, c='magenta')
        plt.title("Rolling Average Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.savefig('LinearValueFunctionApproximation/cart_pole.png')
        
        f = open('LinearValueFunctionApproximation/cart_pole.pkl', 'wb')
        pickle.dump(estimator, f)
        f.close()

if __name__ == "__main__":
    run(1, render=True, training=False)
