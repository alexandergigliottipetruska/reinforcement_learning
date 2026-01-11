import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class policy_network(nn.Module):
    def __init__(self, action_space):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=4)
        self.ffn1 = nn.Linear(in_features=2592, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.ffn1(x))
        x = self.output(x)
        return x

class target_network(nn.Module):
    def __init__(self, action_space):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=4)
        self.ffn1 = nn.Linear(in_features=2592, out_features=256)
        self.output = nn.Linear(in_features=256, out_features=action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) # output torch.Size([32, 32, 9, 9])
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.ffn1(x))
        x = self.output(x)
        return x

class Experience_Replay(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN():
    def __init__(self, policy_network, target_network):
        # Set networks
        self.policy_network = policy_network
        self.target_network = target_network
        
        # Initialize
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
    def step(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    
