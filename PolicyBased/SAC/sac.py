import torch
import numpy as np
import gymnasium as gym
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import random
from torch.distributions import Normal
import os
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save directory
SAVE_DIR = "/content/drive/MyDrive/rl/sac/walker"
os.makedirs(SAVE_DIR, exist_ok=True)


## Replay Buffer
class ReplayBuffer(object):
  def __init__(self, state_dim, action_dim, size=1e6):
    super().__init__()

    self.maxlen = int(size)
    self.current_len = 0

    # Preallocated  arrays
    self.s = torch.empty((self.maxlen, state_dim), device=device)
    self.a = torch.empty((self.maxlen, action_dim), device=device)
    self.r = torch.empty((self.maxlen, 1), device=device)
    self.ns = torch.empty((self.maxlen, state_dim), device=device)
    self.t = torch.empty((self.maxlen, 1), device=device)

    # Pointer
    self.ptr = 0

  def sample(self, batch_size):
    rand_indices = torch.randint(0, self.current_len, (batch_size,), device=device)

    # Format is (batch_s, batch_a, batch_r, batch_ns, batch_t)
    return (self.s[rand_indices],
            self.a[rand_indices],
            self.r[rand_indices],
            self.ns[rand_indices],
            self.t[rand_indices])

  def push(self, transition):
    state, action, reward, next_state, terminated = transition

    # Convert
    state = torch.tensor(state, dtype=torch.float32, device=device)
    action = torch.tensor(action, dtype=torch.float32, device=device)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
    reward = torch.tensor([reward], dtype=torch.float32, device=device)
    terminated = torch.tensor([terminated], dtype=torch.float32, device=device)

    self.s[self.ptr] = state
    self.a[self.ptr] = action
    self.r[self.ptr] = reward
    self.ns[self.ptr] = next_state
    self.t[self.ptr] = terminated

    self.ptr = (self.ptr + 1) % self.maxlen # automatically resets pointer when equal to maxlen

    if self.current_len != self.maxlen:
      self.current_len += 1

  def __len__(self):
    return self.current_len
  
## Network Architecture
# Q-Functions
class Q1(nn.Module):
  def __init__(self, state_space, action_space, hidden_size=256):
    super().__init__()
    self.ffn1 = nn.Linear((np.array(np.prod(state_space) + np.prod(action_space))), hidden_size)
    self.ffn2 = nn.Linear(hidden_size, hidden_size)
    self.ffn3 = nn.Linear(hidden_size, 1)
    self.relu = nn.ReLU()

  def forward(self, state, action):
    x = torch.concat((state, action), 1)
    x = self.relu(self.ffn1(x))
    x = self.relu(self.ffn2(x))
    x = self.ffn3(x)
    return x
  
class Q2(nn.Module):
  def __init__(self, state_space, action_space, hidden_size=256):
    super().__init__()
    self.ffn1 = nn.Linear((np.array(np.prod(state_space) + np.prod(action_space))), hidden_size)
    self.ffn2 = nn.Linear(hidden_size, hidden_size)
    self.ffn3 = nn.Linear(hidden_size, 1)
    self.relu = nn.ReLU()

  def forward(self, state, action):
    x = torch.concat((state, action), 1)
    x = self.relu(self.ffn1(x))
    x = self.relu(self.ffn2(x))
    x = self.ffn3(x)
    return x
  
# Policy
log_std_min = -5
log_std_max = 2

class policy(nn.Module):
  def __init__(self, num_obvs, num_actions, hidden_size=256):
    super().__init__()
    self.ffn1 = nn.Linear(num_obvs, hidden_size)
    self.ffn2 = nn.Linear(hidden_size, hidden_size)
    self.mu = nn.Linear(hidden_size, num_actions)
    self.log_std = nn.Linear(hidden_size, num_actions)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.relu(self.ffn1(x))
    x = self.relu(self.ffn2(x))
    mu = self.mu(x)
    log_std = self.tanh(self.log_std(x))
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std * 1) # Found in Spinning Up, Yarats, and CleanRL
    return mu, log_std
  
class SAC():
  def __init__(self, discount=0.99, smoothing_coefficient=0.005, batch_size=1024, total_steps=3000000, buffer_size=1e6):
    # General Hyperparameters
    self.discount_factor = discount
    self.tau = smoothing_coefficient
    self.policy_lr = 3e-4
    self.q_lr = 3e-4
    self.alpha_lr = 3e-4
    self.batch_size = batch_size
    self.total_steps = total_steps
    # Entropy constraint
    self.buffer_size = buffer_size

    # Networks
    self.q1 = None
    self.q2 = None
    self.policy = None
    self.q1_targ = None
    self.q2_targ = None

    # Optimizers
    self.q_optim = None
    self.policy_optim = None
    self.a_optim = None

    # Replay Buffer
    self.buffer = None

    # Warmup period
    self.warmup = 10000

    # Loss
    self.q1_loss_fn = nn.MSELoss()
    self.q2_loss_fn = nn.MSELoss()

    # Tanh
    self.tanh = nn.Tanh()

    # Scaling and Action effects
    self.action_scale = None
    self.action_bias = None

    # Reward scaling
    self.reward_scaling = 1.0
    self.alpha = None

    # Autotune
    self.autotune = True

  def train(self, render=False):
    # Create environment
    env = gym.make("Ant-v5", render_mode="human" if render else None)

    # Get spaces and scale + bias
    state_space = env.observation_space.shape
    action_space = env.action_space.shape

    self.action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, device=device) # following clean RL
    self.action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, device=device)

    # Create Q-networks
    self.q1 = Q1(state_space=state_space, action_space=action_space).to(device)
    self.q2 = Q2(state_space=state_space, action_space=action_space).to(device)

    # Create Policy Network
    self.policy = policy(num_obvs=state_space[0], num_actions=action_space[0]).to(device)

    # Create Target networks
    self.q1_targ = Q1(state_space=state_space, action_space=action_space).to(device)
    self.q2_targ = Q2(state_space=state_space, action_space=action_space).to(device)

    # Copy state dicts
    self.q1_targ.load_state_dict(self.q1.state_dict())
    self.q2_targ.load_state_dict(self.q2.state_dict())

    # Create buffer
    self.buffer = ReplayBuffer(state_dim=state_space[0], action_dim=action_space[0], size=self.buffer_size)

    # Autotune
    if self.autotune: # following the implementation in CleanRL
      alpha_target = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
      log_alpha = torch.zeros(1, requires_grad=True, device=device)
      self.alpha = torch.exp(log_alpha).item()
      self.a_optim = torch.optim.Adam([log_alpha], lr=self.alpha_lr)

    # Optimizers
    self.q_optim = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.q_lr)
    self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)

    # Count time steps
    global_time_step = 0

    # Metrics
    episode_returns = []
    reward_per_episode = 0
    top_episode_return = - float('inf')
    rolling_returns = deque(maxlen=100)

    # Initial state
    state = env.reset()[0]

    # Main loop
    while global_time_step < self.total_steps:
      # Get action
      if global_time_step < self.warmup:
        action = env.action_space.sample() # sample uniformly from action space, trick mentioned in Spinning Up
      else:
        with torch.no_grad():
          mu, log_std = self.policy(torch.tensor(state, dtype=torch.float32, device=device))
          std = torch.exp(log_std)
          dist = Normal(mu, std)
          xt = dist.rsample()
          yt = self.tanh(xt)
          action = yt * self.action_scale + self.action_bias
          action = action.detach().cpu().numpy()

      # Environment step
      next_state, reward, terminated, truncated, _ = env.step(action) # next step
      done = terminated or truncated

      # Store transition in replay buffer
      transition = (state, action, reward*self.reward_scaling, next_state, done)
      self.buffer.push(transition)

      # Track episode returns
      reward_per_episode += reward

      # Next state depends on termination
      if terminated or truncated:
        state = env.reset()[0]
        episode_returns.append(reward_per_episode)
        rolling_returns.append(reward_per_episode)
        reward_per_episode = 0

      else:
        state = next_state

      if global_time_step % 5000 == 0 and len(episode_returns) != 0:
        print(f"At time step {global_time_step}:")
        print("=======================================")
        rolling_avg = sum(rolling_returns) / len(rolling_returns)
        print(f"Rolling average of last {len(rolling_returns)} episodes: {rolling_avg:.2f}")
        episode_returns = []

        if top_episode_return < rolling_avg:
          top_episode_return = rolling_avg
          print(f"New top episode returns {top_episode_return:.2f}")

      # Warmup period over
      if len(self.buffer) > self.warmup:
        # Get batches
        batch_s, batch_a, batch_r, batch_ns, batch_t = self.buffer.sample(self.batch_size)

        ## Compute targets
        with torch.no_grad():
          # Next action sampling
          mu, log_std = self.policy(batch_ns)
          std = torch.exp(log_std)
          dist = Normal(mu, std)
          batch_xt = dist.rsample()
          batch_yt = self.tanh(batch_xt)
          batch_na = batch_yt * self.action_scale + self.action_bias

          # Log probs
          log_probs_na = (dist.log_prob(batch_xt) - torch.log(self.action_scale * (1 - batch_yt.pow(2)) + 1e-6)).sum(1, keepdim=True)

          # Compute Q target networks values
          Q1_target = self.q1_targ(batch_ns, batch_na)
          Q2_target = self.q2_targ(batch_ns, batch_na)

          # Compute targets
          batch_y = batch_r + self.discount_factor * (torch.ones((self.batch_size, 1), device=device) - batch_t) * (torch.min(Q1_target, Q2_target) - self.alpha * log_probs_na)

        ## Update Q-functions
        q1_loss = self.q1_loss_fn(self.q1(batch_s, batch_a), batch_y)
        q2_loss = self.q2_loss_fn(self.q2(batch_s, batch_a), batch_y)
        q_loss = q1_loss + q2_loss

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        ## Update policy network via gradient ascent
        # Reparametrization trick via Squashed Gaussian policy
        mu, log_std = self.policy(batch_s)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        batch_xt = dist.rsample()
        batch_yt = self.tanh(batch_xt)
        batch_a_tilde = batch_yt * self.action_scale + self.action_bias # Affine transformation

        log_probs_tilde = (dist.log_prob(batch_xt) - torch.log(self.action_scale  * (1 - batch_yt.pow(2)) + 1e-6)).sum(1, keepdim=True) # trick in Spinning Up

        # Min Q-function
        min_q = torch.min(self.q1(batch_s, batch_a_tilde), self.q2(batch_s, batch_a_tilde))

        # Calculate loss
        policy_loss = - (min_q - self.alpha * log_probs_tilde).mean()

        # Backpropagate loss and update weights
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        ## Update Alpha, following CleanRL
        if self.autotune:
          with torch.no_grad():
            mu, log_std = self.policy(batch_s)
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            xt = dist.rsample()
            yt = self.tanh(xt)
            action = yt * self.action_scale + self.action_bias
            log_prob = (dist.log_prob(xt) - torch.log(self.action_scale * (1 - yt.pow(2)) + 1e-6)).sum(1, keepdim=True)

          # Calculate alpha loss
          alpha_loss = (-torch.exp(log_alpha) * (log_prob + alpha_target)).mean()

          # Update alpha
          self.a_optim.zero_grad()
          alpha_loss.backward()
          self.a_optim.step()
          self.alpha = log_alpha.exp().item()


        ## Slowly update target networks
        for parameters, target_params in zip(self.q1.parameters(), self.q1_targ.parameters()):
          q1_targ_params = (1 - self.tau) * target_params + self.tau * parameters
          target_params.data.copy_(q1_targ_params)

        for parameters, target_params in zip(self.q2.parameters(), self.q2_targ.parameters()):
          q2_targ_params = (1 - self.tau) * target_params + self.tau * parameters
          target_params.data.copy_(q2_targ_params)

      if global_time_step % 50000 == 0:
          torch.save(self.q1.state_dict(), os.path.join(SAVE_DIR, f"sac_ant_q1_{global_time_step}.pth"))
          torch.save(self.q2.state_dict(), os.path.join(SAVE_DIR, f"sac_ant_q2_{global_time_step}.pth"))
          torch.save(self.policy.state_dict(), os.path.join(SAVE_DIR, f"sac_ant_policy_{global_time_step}.pth"))


      global_time_step += 1

## Evaluation
class LiveDemo():
    def __init__(self, device):
        self.device = device

    def run(self, episodes=5, fps=60):
        env = gym.make("Ant-v5", render_mode="human")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    
        # Scale and bias should match training settings
        action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)
        action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)

        # Load the policy
        trained_policy = policy(num_obvs=state_dim, num_actions=action_dim).to(self.device)
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'ant_model.pth')

        trained_policy.load_state_dict(torch.load(model_path, map_location=self.device))
        trained_policy.eval()
    
        print(f"Running {episodes} episodes...")

        for i in range(episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            
            # For a more bird's eye view
            env.render() 
            try:
                viewer = env.unwrapped.mujoco_renderer.viewer
                viewer.cam.type = 0          
                viewer.cam.distance = 8.0      
                viewer.cam.elevation = -20      
                viewer.cam.lookat[0] = 2.0     
            except Exception:
                pass

            while not (done or truncated):
                time.sleep(1/fps) # slow it down
                
                state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    mu, _ = trained_policy(state_t) # get the mean and use it instead of sampling from a distribution
                    
                    # Apply tanh and scaling
                    action_squashed = torch.tanh(mu)
                    action = (action_squashed * action_scale + action_bias).cpu().numpy().squeeze()

                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward

            print(f"Episode {i+1} Reward: {total_reward:.2f}")

        env.close()

if __name__ == "__main__":
    ## Training
    sac = SAC()
    sac.train()
