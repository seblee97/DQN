import torch
import numpy as np 
import torch.nn.functional as F
from torch import nn
import random

import gym
import gym_tetris

"""""""""""""""""""
# Tetris action space: Discrete(12) no-action, left, right, down, rotate-left, rotate-right, left+down, 
#                                   right+down, left+rotate-left, right+rotate-right, left+rotate-right, right+rotate-left
"""""""""""""""""""

config = {
    "env": gym_tetris.make('Tetris-v0'),
    "replay_buffer_size": 1000,
    "epsilon": 0.05,
    "gamma": 0.95,
    "batch_size": 30
}

def trial_env():
    env = gym_tetris.make('Tetris-v0')
    done=True
    for step in range(5000):
        if done:
            state=env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
    env.close()
    return env


class FeatureEncoder(nn.Module):

    """
    feature encoder to preprocess state observation
    """

    def __init__(self, obs_space):
        nn.Module.__init__(self)
        self.obs_space = obs_space
    
    def forward(self, obs):
        return None

class QNetwork(nn.Module):

    def __init__(self, obs_space, num_actions):
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.num_actions = num_actions

    def forward(self, obs, action):

        return None

    def compute_actions(self, obs):
        amax = np.argmax([self.forward(obs, a) for a in #])


class DQN:

    def __init__(self, config):
        self.config = config 

        self.env = self.config["env"]
        self.action_space = self.env.action_space

        self.feature_encoder = FeatureEncoder(OBSSPACE)
        self.qnetwork = QNetwork(OBSSPACE, NUMACTIONS)
        
        self.replay_buffer_size = self.config["replay_buffer_size"]
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        self.epsilon = self.config["epsilon"]
        self.gamma = self.config["gamma"]
        self.batch_size = self.config["batch_size"]

    def step_env(self, action):
        """
        query environment to collect new experience
        """
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info


    def train(self):
        state = self.env.reset()
        for i in range(500):
            if random.random() < self.epsilon:
                act = self.action_space.sample()
            else:
                raise NotImplementedError #ComputeActions
            new_state, reward, done, info = self.step_env(act)
            experience = (self.feature_encoder(state), action, reward, self.feature_encoder(new_state))
            self.replay_buffer.add(experience)
            state = new_state
            if i >= self.batch_size:
                self.optimize()
        return None


    def optimize(self):
        state, action, reward, new_state = self.replay_buffer.sample(batch_size)
        if done:
            y = reward
            print("Episode finished after {} timesteps".format(i+1)) 
            break
        else:
            y = reward + self.gamma*np.amax([self.qnetwork(OBSSPACE, NUMACTIONS) for a in ACTIONS])
        loss = LOSSFN(y, state, action)
        UPDATE_Q


class ReplayBuffer:

    def __init__(self, size):
        self._buffer = []
        self._size = size


    def add(self, experience):
        """
        method to add experiences to the replay buffer

        args:
            experience (tuple) -- (state, action, reward, next_state) 
        """
        if len(self._buffer) < self._size:
            self._buffer.append(experience)
        else:
            self._buffer.remove(random.choice(self._buffer))
            self._buffer.append(experience)

    def sample(self, batch_size):
        """
        method to sample experiences from the replay buffer

        args:
            batch_size (int) -- number of experiences to collect for batch training 

        returns"
            random experience (tuple) -- (state, action, reward, next_state) 
        """
        return random.sample(self._buffer, batch_size)
