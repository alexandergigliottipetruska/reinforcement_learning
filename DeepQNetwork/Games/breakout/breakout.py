import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import os
import sys
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

# Get DQN dir
dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(dir)

# Now import the class
from DQN import DQN, Experience_Replay, policy_network, target_network

class Train():
    def __init__(self, maxlen=1000000, learning_rate=0.0001, network_sync=10000, mode="double"):
        # Hyperparameters
        self.alpha = learning_rate
        self.epsilon = 1.0
        self.number_steps = 1000000
        self.discount_factor_g = 0.99
        self.random_generator = np.random.default_rng()
        self.network_sync = network_sync
        self.batch_size = 32
        self.loss_fn = torch.nn.MSELoss()
        self.DQN = None
        self.experience_replay = Experience_Replay(maxlen)
        self.optimizer = None
        self.mode = mode

    def train(self, episodes, render=False):
        env = gym.make('BreakoutNoFrameskip-v4', render_mode='human' if render else None)

        ## Atari Preprocessing Wrapper
        # Converts to grayscale, resizes to 84 x 84, etc.
        env = AtariPreprocessing(
            env,
            noop_max=10, frame_skip=4, terminal_on_life_loss=False,
            screen_size=84, grayscale_obs=True, grayscale_newaxis=False
        )

        # More on params
        epsilon_start = 1.0
        epsilon_min = 0.01

        ## Frame Stack Observation
        # A single frame provides no information on velocity, direction, or temporal patterns
        # Considering the last 4 frames at once, as one observation, allows the CNN to infer all of these
        env = FrameStackObservation(
            env, stack_size=4
        )

        policy_net = policy_network(env.action_space.n)
        target_net = target_network(env.action_space.n)
        self.DQN  = DQN(policy_net, target_net)

        self.optimizer = torch.optim.RMSprop(self.DQN.policy_network.parameters(), lr=self.alpha)

        rewards = np.zeros(episodes)
        t = 0

        for i in range(episodes):
            terminated = False
            truncated = False
            state = env.reset()[0] # (4, 84, 84), downsized, grayscaled, with 4 frames in one observation due to frame-skipping
            print(i)

            while(not terminated and not truncated):
                if self.random_generator.random() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = torch.argmax(self.DQN.policy_network(torch.tensor(state, dtype=torch.float32).reshape(1, 4, 84, 84)))

                next_state, reward, terminated, truncated, _ = env.step(action)
                t += 1

                # Store a transition (s, a, r, s', terminated) in memory
                self.experience_replay.push(state, action, reward, next_state, terminated)

                # Update current state
                state = next_state

                # Accumulate reward
                rewards[i] += reward

                # Warmup, wait until buffer has enough states to begin training.
                if len(self.experience_replay) >= 50000:
                    # Get batch
                    batch = self.experience_replay.sample(self.batch_size)
                    # Optimize the policy network
                    batch_s = torch.tensor(np.stack([transition[0] for transition in batch], axis=0), dtype=torch.float32)
                    batch_a = torch.tensor(np.stack([transition[1] for transition in batch], axis=0), dtype=torch.int32)
                    batch_r = torch.tensor(np.stack([transition[2] for transition in batch], axis=0), dtype=torch.float32)
                    batch_ns = torch.tensor(np.stack([transition[3] for transition in batch], axis=0), dtype=torch.float32)
                    batch_terminated = torch.tensor(np.stack([transition[4] for transition in batch], axis=0), dtype=torch.bool)
                    
                    # Get targets
                    with torch.no_grad():
                        if self.mode == 'standard':
                            y = torch.where(batch_terminated, batch_r, batch_r + self.discount_factor_g * torch.max(self.DQN.target_network(batch_ns)))
                        elif self.mode == 'double':
                            chosen_a = torch.argmax(self.DQN.policy_network(batch_ns), dim=1)
                            y = torch.where(batch_terminated, batch_r, 
                                            batch_r + self.discount_factor_g * self.DQN.target_network(batch_ns).gather(1, chosen_a.unsqueeze(1)).squeeze(1)
                                            )
            

                    # Get predictions (approximations)
                    preds = self.DQN.policy_network(batch_s) # (32, 4)
                    preds = preds.gather(1, batch_a.unsqueeze(1)).squeeze(1) # (dim, idx)


                    # Calculate loss
                    loss = self.loss_fn(preds, y)

                    # Backpropagate the loss
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_value_(self.DQN.policy_network.parameters(), 1)

                    # Adjust learning weights
                    self.optimizer.step()
                else:
                    pass

                if t % self.network_sync == 0:
                    self.DQN.step()

                # Decay epsilon
                self.epsilon = max(0.01, epsilon_start - (epsilon_start - epsilon_min) * t / self.number_steps)

        env.close()

        average_rewards = np.zeros(episodes)
    
        for i in range(episodes):
            average_rewards[i] = np.mean(rewards[max(0, i-100):i+1])

        plt.plot(average_rewards, c='red')
        plt.title("Rolling Average Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.savefig('DeepQNetwork/breakout/breakout.png')
        
        torch.save(self.DQN.policy_network.state_dict(), 'breakout_dqn.pt')

if torch.cuda.is_available():
    episodes = 600
else:
    episodes = 50

class Test():
    def test(self, episodes, render):
        env = gym.make('ALE/Breakout-v5', render_mode='human' if render else None)
        env = AtariPreprocessing(
            env,
            noop_max=10, frame_skip=4, terminal_on_life_loss=False,
            screen_size=84, grayscale_obs=True, grayscale_newaxis=False
        )

        env = FrameStackObservation(
            env, stack_size=4
        )

        # Load learned policy
        policy_dqn = policy_network()
        policy_dqn.load_state_dict(torch.load('breakout_dqn.pt'))
        policy_dqn.eval()

        for i in range(episodes):
            terminated = False
            truncated = False
            state = env.reset()[0]

            while (not terminated and not truncated):
                # select best action
                with torch.no_grad():
                    action = policy_dqn(state).argmax().item()

                state, reward, terminated, truncated, _ = env.step(action)

        env.close()

if __name__ == "__main__":
    train = Train()
    train.train(600, False)