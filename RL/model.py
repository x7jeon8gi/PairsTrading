import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gym
from gym import spaces

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
activation = nn.ReLU()

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SimpleMLP(nn.Module):
    def __init__(self, state_size, num_actions, hidden_sizes=(256, 256, )):
        super(SimpleMLP, self).__init__()

        self.input_layer = nn.Linear(state_size + num_actions, hidden_sizes[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            hidden_layer = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            self.hidden_layers.append(hidden_layer)
        
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.activation = activation 

    def forward(self, state, action):
        
        #! state and action concatenated
        x = torch.cat((state, action), dim=-1)
        x = self.activation(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size):
        super(QNetwork, self).__init__()

        self.q1 = SimpleMLP(state_size, num_actions, hidden_size) # state action concatenated in SimpleMLP
        self.q2 = SimpleMLP(state_size, num_actions, hidden_size)
        # self.q1 = ImprovedMLP(state_size, num_actions, hidden_size)
        # self.q2 = ImprovedMLP(state_size, num_actions, hidden_size)
        self.apply(weights_init_)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)


class GaussianPolicy(nn.Module):
    def __init__(self, state_size, num_actions, hidden_sizes=(256, 256, ), action_space=None):
        super(GaussianPolicy, self).__init__()

        self.input_layer = nn.Linear(state_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            hidden_layer = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.mean_layer = nn.Linear(hidden_sizes[-1], num_actions)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], num_actions)
        self.activation = activation 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            # action_space가 list일 때
            if isinstance(action_space, list):
                if len(action_space) != 2:
                    raise ValueError("action_space 리스트는 두 개의 요소([min, max])를 가져야 합니다.")
                low, high = action_space
                low = torch.tensor(low, dtype=torch.float32)
                high = torch.tensor(high, dtype=torch.float32)
            else:
                # Gym의 Box 공간과 유사한 객체라고 가정
                low = torch.tensor(action_space.low, dtype=torch.float32).to(self.device)
                high = torch.tensor(action_space.high, dtype=torch.float32).to(self.device)
            self.action_scale = (high - low) / 2.0
            self.action_bias = (high + low) / 2.0

            print(f"Action Scale: {self.action_scale}")
            print(f"Action Bias: {self.action_bias}")

    def forward(self, state):
        x = self.activation(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)  # Clamping to stabilize training
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias # 
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def to(self, device):
        self.device = device
        return super(GaussianPolicy, self).to(device)