# Imports
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
import numpy as np

class ExpectedSARSA:
    def __init__(self, state_space, action_space, discount_factor=0.99, learning_rate=0.1):
        self.g = discount_factor
        self.a = learning_rate
        self.Q = np.zeros((state_space, action_space))
        self.action_space = action_space

    def update(self, state, action, next_state, reward, epsilon):
        actions = np.zeros(self.action_space)
        
        next_state_action_values = self.action_values(next_state)
        greedy_actions = np.flatnonzero(next_state_action_values == np.max(next_state_action_values))

        for a in greedy_actions:
            actions[a] = (1 - epsilon) / len(greedy_actions)
        
        for a in range(len(actions)):
            if a not in greedy_actions:
                actions[a] = epsilon / (self.action_space - 1)

        self.Q[state, action] = self.Q[state, action] + self.a * (reward + self.g * np.dot(actions, self.Q[next_state, :]) - self.Q[state, action])
        
    def action_values(self, state):
        return self.Q[state, :]

def run(episodes, render=False, training=True):
    env = gym.make("Taxi-v3", render_mode='human' if render else None)

    if training:
        state_space = env.observation_space.n
        action_space = env.action_space.n

        Q = ExpectedSARSA(state_space, action_space)
    else:
        f = open('TemporalDifference/Expected_Sarsa/taxi.pkl', 'rb')
        Q = pickle.load(f)
        f.close()

    state_counter = np.zeros((env.observation_space.n, env.action_space.n))
    
    epsilon = 1.0
    initial_epsilon = epsilon
    epsilon_decay_rate = 0.9995
    random_generator = np.random.default_rng()
    beta = 0.5

    rewards = np.zeros(episodes)
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            if training and random_generator.random() < epsilon:
                action = env.action_space.sample()
            else:
                action_values = Q.action_values(state)
                action = np.random.choice(np.flatnonzero(action_values == np.max(action_values)))

            Q.a = 1/ ((1 + state_counter[state, action])**beta)
            state_counter[state, action] += 1

            next_state, reward, terminated, truncated, _ = env.step(action)

            if training:
                Q.update(state, action, next_state, reward, epsilon)

            state = next_state

            rewards[i] += reward

        epsilon = max(0.01, initial_epsilon * epsilon_decay_rate**i)
       

    env.close()

    average_rewards = np.zeros(episodes)
        
    for i in range(episodes):
        average_rewards[i] = np.mean(rewards[max(0, i-100):i+1])

    if training:
        plt.plot(average_rewards, c='blue')
        plt.title("Rolling Average Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.savefig('TemporalDifference/Expected_Sarsa/taxi.png')

        f = open('TemporalDifference/Expected_Sarsa/taxi.pkl', 'wb')
        pickle.dump(Q, f)
        f.close()

if __name__ == '__main__':
    run(1, training=False, render=True)