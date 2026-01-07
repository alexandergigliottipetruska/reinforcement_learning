import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
import sklearn
import numpy as np

class Estimator():
    def __init__(self, scaler, featurizer, learning_rate, action_space):
        self.scaler = scaler
        self.featurizer = featurizer
        self.alpha = learning_rate
        self.action_space = action_space
        self.n_features = 400
        self.w = np.zeros((self.action_space, self.n_features))
        self.b = np.zeros(self.action_space)

    def update(self, s, a, y):
        x = self.feature_extractor(s, a)
        estimate = self.predict(s, a)
        self.w[a] = self.w[a] + self.alpha * (y - estimate) * x
        self.b[a] = self.b[a] + self.alpha * (y - estimate)

    def feature_extractor(self, obs, a=None):
        scaled_obs = self.scaler.transform(obs.reshape(1, -1))[0]
        features  = self.featurizer.transform(scaled_obs.reshape(1, -1))
        return features[0]

    def predict(self, s, a=None):
        if a is not None:
            x = self.feature_extractor(s, a)
        else:
            return np.array([self.feature_extractor(s, a) @ self.w[a] + self.b[a] for a in range(self.action_space)])
        
        return x @ self.w[a] + self.b[a]

def run(episodes, training, render):
    env = gym.make("MountainCar-v0", render_mode='human' if render else None)

    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
    featurizer.fit(scaler.transform(observation_examples))

    if training:
        estimator = Estimator(scaler, featurizer, learning_rate=0.05, action_space=env.action_space.n)
    else:
        f = open("LinearValueFunctionApproximation/MountainCar/mountain_car.pkl", 'rb')
        estimator = pickle.load(f)
        f.close()

    # Parameters
    discount_factor_g = 0.95
    epsilon = 1.0
    epsilon_decay_rate = 1e-4

    # Generator
    random_generator = np.random.default_rng()

    # Rewards
    rewards = np.zeros(episodes)

    # Go through all the episodes
    for i in range(episodes):
        print(i)
        terminated = False
        truncated = False
        state = env.reset()[0]

        while (not terminated and not truncated):
            if random_generator.random() < epsilon:
                action = env.action_space.sample()
            else:
                action_values = estimator.predict(state)
                action = np.random.choice(np.flatnonzero((action_values == np.max(action_values))))

            next_state, reward, terminated, truncated, _ = env.step(action)

            target = reward + discount_factor_g * np.max(estimator.predict(next_state))

            if training:
                estimator.update(state, action, target)

            state = next_state

            rewards[i] = reward

        epsilon = min(0.01, epsilon - epsilon_decay_rate)

    env.close()

    average_rewards = np.zeros(episodes)
        
    for i in range(episodes):
        average_rewards[i] = np.mean(rewards[max(0, i-100):i+1])

    if training:
        plt.plot(average_rewards, c='green')
        plt.title("Rolling Average Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.savefig('LinearValueFunctionApproximation/MountainCar/mountain_car.png')
        
        f = open('LinearValueFunctionApproximation/MountainCar/mountain_car.pkl', 'wb')
        pickle.dump(estimator, f)
        f.close()

if __name__ == "__main__":
    run(10000, training=True, render=False)