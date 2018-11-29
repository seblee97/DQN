import torch
import numpy as np 
import torch.nn.functional as F
from torch import nn
import random
import PIL
from torchvision import transforms
from torch import optim
from collections import deque

import gym
import gym_tetris

# Tetris action space: Discrete(12) no-action, left, right, down, rotate-left, rotate-right, left+down, 
#                                   right+down, left+rotate-left, right+rotate-right, left+rotate-right, right+rotate-left

config = {
    "env": gym_tetris.make('Tetris-v0'),
    "replay_buffer_size": 1000,
    "epsilon": 0.05,
    "gamma": 0.95,
    "batch_size": 30,
    "processed_state_dims": (84, 84),
    "frame_skip": 4,
    "stack_dim": 4,
    "lr": 1e-2,
    "momentum": 0.9
}

def _array_to_image(rgb_state):
    return PIL.Image.fromarray(rgb_state)

def trial_env():
    env = gym_tetris.make('Tetris-v0')
    done=True
    for step in range(5000):
        if done:
            state=env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
    env.close()
    return env


class PreProcessor(nn.Module):

    """
    preprocess state observation: grayscale + downsample
    """

    def __init__(self, output_size=(84,84)):
        nn.Module.__init__(self)
        self.transform = transforms.Compose([
            transforms.Resize(output_size),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    
    def forward(self, state):
        pil = _array_to_image(state)
        processed_state = self.transform(pil)
        return processed_state

class QNetwork(nn.Module):

    """
    Follows architecture of original DQN atari paper
    """

    def __init__(self, obs_space=((4,84,84)), num_actions=12):
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(self.obs_space[0], out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, num_actions)


    def forward(self, obs):
        #print('obs_shape', obs.size())
        x = F.relu(self.conv1(obs))
        #print('x', x.size())
        x = F.relu(self.conv2(x))
        #print('x', x.size())
        x = x.view((x.shape[0], -1))
        #print('x', x.size())
        x = self.lin(x)
        return x

    def compute_actions(self, obs):
        actions = self.forward(obs)[0]
        act, arg = actions.detach().max(0)
        #print('actions', arg)
        return arg


class DQN:

    def __init__(self, config):
        self.config = config 

        self.epsilon = self.config["epsilon"]
        self.gamma = self.config["gamma"]
        self.batch_size = self.config["batch_size"]
        self.frame_skip = self.config["frame_skip"]
        self.stack_dim = self.config["stack_dim"]
        self.lr = self.config["lr"]

        self.env = self.config["env"]
        self.action_space = self.env.action_space

        self.processed_state_dims = self.config["processed_state_dims"]
        self.preprocessor = PreProcessor(self.processed_state_dims)
        self.qnetwork = QNetwork()
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=self.lr)
        
        self.replay_buffer_size = self.config["replay_buffer_size"]
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def train(self):
        self.env.reset()
        states = self._initialise_env()
        print(states[0].shape)
        for i in range(500):
            if random.random() < self.epsilon:
                act = self.action_space.sample()
            else:
                act = self.qnetwork.compute_actions(torch.stack(states, dim=1))
            new_states = deque([states], maxlen=self.stack_dim)
            rewards = deque([0 for _ in range(self.stack_dim)], maxlen=self.stack_dim)
            for s in range(self.frame_skip - 1): #skip
                new_state, reward, done, info = self._step_env(act)
            new_states.append(self.preprocessor(new_state))
            rewards.append(reward)
            list_states = list(states)
            list_new_states = list(new_states)
            print(list_states, list_new_states)
            experience = (torch.stack(list_states, dim=1), act, np.sum(rewards), torch.stack(list_new_states, dim=1))
            self.replay_buffer.add(experience)
            states = new_states
            if i >= self.batch_size:
                self._optimize()
    
    def _qloss(self, state, reward, action, new_state):
        qval = self.qnetwork(state)[action]
        y = reward + self.gamma*np.amax(self.q(new_state))
        loss = F.smooth_l1_loss(qval, y)
        return loss

    def _optimize(self):
        state, action, reward, new_state = self.replay_buffer.sample(batch_size) #the state & new state sampled here have been run through preprocessor and are stacked
        loss = self._qloss(reward)
        self.optimizer.step()

    def _step_env(self, action):
        """
        query environment to collect new experience
        """
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info

    def _initialise_env(self):
        """
        ensures first phi represents stack of four states
        """
        states = []
        for i in range(4):
            states.append(self.preprocessor(self._step_env(self.env.action_space.sample())[0]))
        return states

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

if __name__ == "__main__":
    dqn = DQN(config)
    dqn.train()