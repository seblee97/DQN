import torch
import numpy as np 
import torch.nn.functional as F
from torch import nn

import gym
import gym-tetris


class QNetwork(nn.module):

    def __init__(self, obs_space, num_actions):
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.num_actions = num_actions

    def forward(self, obs):



class DQN:

    def __init__(self):

