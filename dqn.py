import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torch import optim

import random
import PIL
import numpy as np 
from collections import deque
import os
import argparse

import gym
import gym_tetris
from gym import wrappers

from tensorboardX import SummaryWriter
writer = SummaryWriter()

# Tetris action space: Discrete(12) no-action, left, right, down, rotate-left, rotate-right, left+down, 
#                                   right+down, left+rotate-left, right+rotate-right, left+rotate-right, right+rotate-left

environments = {0: gym_tetris.make('Tetris-v0'), 1: gym.make('MsPacman-v0')}

parser = argparse.ArgumentParser(description='DQN Configuration')
parser.add_argument('-e', '--env', default=1, type=int, 
                    help='Index of environments dictionary to specify gym environment')
parser.add_argument('-bs', '--buffer_size', default=1000, type=int, 
                    help='Size of experience replay buffer')
parser.add_argument('-fe', '--final_eps', default=0.01, type=float, 
                    help='Final value of epsilon, reached after annealing process of epsilon')
parser.add_argument('-ea', '--eps_anneal', default=200000, type=int, 
                    help='Number of steps in which epsilon is linearly annealed from 1.0 to --final_eps specification')
parser.add_argument('-g', '--gamma', default=0.95, type=float, 
                    help='Discount factor (standard RL)')
parser.add_argument('-b', '--batch_size', default=32, type=int, 
                    help='Training batch size')
parser.add_argument('-sd', '--state_dims', default=84, type=int, 
                    help='Dimension of processed state. Note this argument will specify both x and y dimensions, so only offers square outputs')
parser.add_argument('-fs', '--frame_skip', default=4, type=int, 
                    help='Number of frames to skip, during which action is repeated')
parser.add_argument('-stack', '--stack_dim', default=4, type=int, 
                    help='Number of states to stack into phi - the state representations stored in the replay buffer')
parser.add_argument('--lr', default=1e-2, type=float, 
                    help='Learning rate')
parser.add_argument('-mom', '--momentum', default=0.0, type=float, 
                    help='Momentum (parameter for optimiser)')
parser.add_argument('-m', '--monitor', default=None, type=str, 
                    help='Parameter to specify desintation of gym monitor outputs. No outputs will be made if None')
parser.add_argument('--exp_id', default="test0", type=str, 
                    help='Name of experiment')
parser.add_argument('--living_rew', default=0.0, type=float, 
                    help='Reward given to agent for each step')


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
        self.lin = nn.Linear(32*9*9, self.num_actions)


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

        self.epsilon = 1
        self.final_epsilon = self.config["final_epsilon"]
        self.epsilon_annealing_duration = self.config["epsilon_annealing_duration"]
        self.epsilon_step = (self.epsilon - self.final_epsilon)/self.epsilon_annealing_duration
        self.gamma = self.config["gamma"]
        self.batch_size = self.config["batch_size"]
        self.frame_skip = self.config["frame_skip"]
        self.stack_dim = self.config["stack_dim"]
        self.lr = self.config["lr"]
        self.momentum = self.config["momentum"]
        self.living_reward = self.config["living_reward"]

        self.exp_id = self.config["exp_id"]
        self.env = self.config["env"]
        if self.config["monitor"] is not None:
            self.env = wrappers.Monitor(self.env, os.getcwd() + self.config["monitor"])
        self.action_space = self.env.action_space
        self.global_steps = 0
        self.global_reward = 0

        self.processed_state_dims = self.config["processed_state_dims"]
        self.preprocessor = PreProcessor(self.processed_state_dims)
        self.qnetwork = QNetwork(num_actions=self.env.action_space.n)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=self.lr)
        
        self.replay_buffer_size = self.config["replay_buffer_size"]
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def train(self):
        for ep in range(20000):
            ep_reward = 0
            self.env.reset()
            states, rewards = self._initialise_env()
            new_states = states
            done = False
            t = 0
            while not done:
                if random.random() < self.epsilon:
                    act = self.action_space.sample()
                else:
                    act = int(self.qnetwork.compute_actions(torch.stack(list(states), dim=1)))
                for s in range(self.frame_skip - 1): #skip
                    if not done:
                        new_state, reward, done, info = self._step_env(act)
                new_states.append(self.preprocessor(new_state))
                rewards.append(reward)
                if done:
                    print('final reward', reward)
                ep_reward += reward
                self.global_reward += reward
                experience = (torch.stack(list(states), dim=1), act, np.sum(rewards), torch.stack(list(new_states), dim=1))
                self.replay_buffer.add(experience)
                states = new_states
                t += 1
                if t >= self.batch_size:
                    if done:
                        self._optimize(done=True)
                        print("epsiode terminated with reward of {}".format(ep_reward))
                    else:
                        self._optimize(done=False)
                writer.add_scalar("data/global_reward", self.global_reward, self.global_steps) #log total reward to tb
            writer.add_scalar("data/episode_reward", ep_reward, ep+1) #log reward in this episode to tb
            writer.add_scalar("data/mean_episode_reward", self.global_reward/(ep+1), self.global_steps) #log average episode reward to tb
            writer.add_scalar("data/mean_episode_track", ep, self.global_steps)
        self.env.close()
        #writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
    
    def _qloss(self, state, reward, action, new_state, done=False):
        qvalues = torch.stack([self.qnetwork(state)[i][action[i]] for i in range(len(action))])
        if done:
            y = torch.LongTensor(reward) 
        else:
            y = torch.LongTensor(reward) + self.gamma*torch.stack([torch.max(r) for r in self.qnetwork(new_state)]).long()
        loss = F.smooth_l1_loss(qvalues, y.float())
        # print('loss', loss)
        writer.add_scalar("data/loss", loss, self.global_steps) #log loss to tb
        return loss

    def _optimize(self, done=False):
        experience_sample = self.replay_buffer.sample(self.batch_size) #returns list of experience tuples
        experience_sampleT = [list(r) for r in zip(*experience_sample)]
        states = torch.stack(experience_sampleT[0], dim=1).squeeze()
        new_states = torch.stack(experience_sampleT[3], dim=1).squeeze()
        rewards = np.array(experience_sampleT[2])
        actions = np.array(experience_sampleT[1])
        loss = self._qloss(states, rewards, actions, new_states, done=done)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _step_env(self, action):
        """
        query environment to collect new experience
        """
        state, reward, done, info = self.env.step(action)
        reward += self.living_reward
        self.global_steps += 1
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_step
        writer.add_scalar("data/epsilon", self.epsilon, self.global_steps) #log epsilon to tb
        return state, reward, done, info

    def _initialise_env(self):
        """
        ensures first phi represents stack of four states
        """
        states = []
        rewards = []
        for i in range(4):
            state, reward, a, b = self._step_env(self.env.action_space.sample())
            states.append(self.preprocessor(state))
            rewards.append(reward)
        return deque(states, maxlen=self.stack_dim), deque(rewards, maxlen=self.stack_dim)

class ReplayBuffer:

    def __init__(self, size):
        self._buffer = deque([], maxlen=size)

    def add(self, experience):
        """
        method to add experiences to the replay buffer

        args:
            experience (tuple) -- (state, action, reward, next_state) 
        """
        self._buffer.append(experience)

    def sample(self, batch_size):
        """
        method to sample experiences from the replay buffer

        args:
            batch_size (int) -- number of experiences to collect for batch training 

        returns"
            random experience (tuple) -- (state, action, reward, next_state) 
        """
        return random.sample(list(self._buffer), batch_size)

def run():
    args = parser.parse_args()
    config = {
        "env": environments[args.env],
        "replay_buffer_size": args.buffer_size,
        "final_epsilon": args.final_eps,
        "epsilon_annealing_duration": args.eps_anneal,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "processed_state_dims": (args.state_dims, args.state_dims),
        "frame_skip": args.frame_skip,
        "stack_dim": args.stack_dim,
        "lr": args.lr,
        "momentum": args.momentum,
        "monitor": args.monitor,
        "exp_id": args.exp_id,
        "living_reward": args.living_rew
    }
    dqn = DQN(config)
    dqn.train()


if __name__ == "__main__":
    run()
    
    