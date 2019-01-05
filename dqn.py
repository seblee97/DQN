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

if torch.cuda.is_available():
    print("Using the GPU")
    experiment_device = torch.device("cuda")
else:
    print("Using the CPU")
    experiment_device = torch.device("cpu")

# Tetris action space: Discrete(12) no-action, left, right, down, rotate-left, rotate-right, left+down, 
#                                   right+down, left+rotate-left, right+rotate-right, left+rotate-right, right+rotate-left

#CartPole HyperParameters: python dqn_new.py -c 500 --final_eps 0.1 --eps_anneal 10000 --replay_start_size 3000 --gamma 1 --lr 1e-4 -fs 1 -sd 1 -bs 100000 -b 20

environments = {0: gym_tetris.make('Tetris-v0'), 1: gym.make('MsPacman-v0'), 2: gym.make('Breakout-v0'), 3: gym.make('Pong-v0'), 4: gym.make('CartPole-v0')}

parser = argparse.ArgumentParser(description='DQN Configuration')
parser.add_argument('-e', '--env', default=3, type=int, 
                    help='Index of environments dictionary to specify gym environment')
parser.add_argument('-bs', '--buffer_size', default=1000000, type=int, 
                    help='Size of experience replay buffer')
parser.add_argument('-fe', '--final_eps', default=0.1, type=float, 
                    help='Final value of epsilon, reached after annealing process of epsilon')
parser.add_argument('-ea', '--eps_anneal', default=1000000, type=int, 
                    help='Number of steps in which epsilon is linearly annealed from 1.0 to --final_eps specification')
parser.add_argument('-g', '--gamma', default=0.99, type=float, 
                    help='Discount factor (standard RL)')
parser.add_argument('-b', '--batch_size', default=32, type=int, 
                    help='Training batch size')
parser.add_argument('-sd', '--state_dims', default=84, type=int, 
                    help='Dimension of processed state. Note this argument will specify both x and y dimensions, so only offers square outputs')
parser.add_argument('-fs', '--frame_skip', default=4, type=int, 
                    help='Number of frames to skip, during which action is repeated')
parser.add_argument('-stack', '--stack_dim', default=4, type=int, 
                    help='Number of states to stack into phi - the state representations stored in the replay buffer')
parser.add_argument('--lr', default=1e-3, type=float, 
                    help='Learning rate')
parser.add_argument('-mom', '--momentum', default=0.0, type=float, 
                    help='Momentum (parameter for optimiser)')
parser.add_argument('-m', '--monitor', default=None, type=str, 
                    help='Parameter to specify desintation of gym monitor outputs. No outputs will be made if None')
parser.add_argument('--exp_id', default="test0", type=str, 
                    help='Name of experiment')
parser.add_argument('--living_rew', default=0.0, type=float, 
                    help='Reward given to agent for each step')
parser.add_argument('-c', '--C', default=10000, type=int, 
                    help='Number of steps between resetting target network to qnetwork')
parser.add_argument('--target', default=True, type=bool, 
                    help='Use target network to stabilise training')
parser.add_argument('--replay_start_size', default=50000, type=int, 
                    help='Number of steps for which experiences are collected before training commences')
parser.add_argument('-n', '--num_episodes', default=20000, type=int, 
                    help='Number of episodes over which to train')


def _array_to_image(rgb_state):
    return PIL.Image.fromarray(rgb_state)

def trial_env(env_index, steps):
    """
    function to test environment
    """
    env = environments[env_index]
    done=True
    for step in range(steps):
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
            transforms.Grayscale(),
            transforms.Resize(output_size),
            transforms.ToTensor()
        ])
    
    def forward(self, state):
        pil = _array_to_image(state)
        processed_state = self.transform(pil)/255
        processed_state = processed_state.to(experiment_device)
        return processed_state

class QNetwork(nn.Module):

    """
    Architecture taken from original DQN paper https://www.nature.com/articles/nature14236
    """

    def __init__(self, obs_space=((4,84,84)), num_actions=4, symbolic=False):
        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.num_actions = num_actions
        self.symbolic = symbolic

        #Visual Layers
        self.conv1 = nn.Conv2d(self.obs_space[0], out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, self.num_actions)

        #Symbolic Layers
        self.fc1 = nn.Linear(self.obs_space[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.num_actions)

    def forward(self, obs):
        if self.symbolic:
            obs = obs.float()
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.relu(self.conv1(obs))
            x = F.relu(self.conv2(x))
            x = x.view((x.shape[0], -1))
            x = self.lin(x)
        return x

    def compute_actions(self, obs):
        actions = self.forward(obs)
        if not self.symbolic:
            actions = actions[0]
        act, arg = actions.detach()[0].max(0)
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
        self.sync_networks = self.config["C"]
        self.target = self.config["target"]
        self.replay_start = self.config["replay_start_size"]
        self.num_eps = self.config["num_episodes"]

        self.exp_id = self.config["exp_id"]
        self.env = self.config["env"]
        if self.config["monitor"] is not None:
            self.env = wrappers.Monitor(self.env, os.getcwd() + self.config["monitor"])
        self.action_space = self.env.action_space
        self.global_steps = 0
        self.global_reward = 0
        self.symbolic = self.config["symbolic"]

        self.processed_state_dims = self.config["processed_state_dims"]
        self.preprocessor = PreProcessor(self.processed_state_dims).to(experiment_device)
        self.qnetwork = QNetwork(num_actions=self.env.action_space.n, symbolic=self.symbolic).to(experiment_device)
        if self.target:
            self.qtarget = QNetwork(num_actions=self.env.action_space.n, symbolic=self.symbolic).to(experiment_device)
            self.qtarget.load_state_dict(self.qnetwork.state_dict()) #initialise target network with same weights as qnetwork
            self.update_target = False
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=self.lr)
        
        self.replay_buffer_size = self.config["replay_buffer_size"]
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def train(self):
        print('sym', self.symbolic)
        for ep in range(self.num_eps):
            self.env.reset()
            states, ep_reward = self._initialise_env()
            new_states = states.copy()
            done = False
            while not done:
                if random.random() < self.epsilon:
                    act = self.action_space.sample()
                else:
                    if self.symbolic:
                        act = int(self.qnetwork.compute_actions(torch.stack([torch.tensor(s) for s in list(states)])))
                    else:
                        preprocessed_state = torch.stack([self.preprocessor(s) for s in list(states)], dim=1)
                        act = int(self.qnetwork.compute_actions(preprocessed_state))
                r = 0
                for _f in range(self.frame_skip): #skip
                    if not done:
                        new_state, reward, done, info = self._step_env(act)
                        r += reward
                        ep_reward += reward
                        self.global_reward += reward
                if self.symbolic:
                    new_states.append(new_state)
                else:
                    new_states.append(new_state.astype('uint8'))
                experience = (states, act, np.clip(np.sum(r), -1, 1), new_states, done) #clipping rewards to between -1 and 1
                self.replay_buffer.add(experience)
                states = new_states.copy()
                if self.global_steps >= self.replay_start:
                    self._optimize()
                    if done:
                        print("epsiode terminated with reward of {}".format(ep_reward))
                writer.add_scalar("data/global_reward", self.global_reward, self.global_steps) #log total reward to tb
            writer.add_scalar("data/episode_reward", ep_reward, ep+1) #log reward in this episode to tb
            writer.add_scalar("data/mean_episode_reward", self.global_reward/(ep+1), self.global_steps) #log average episode reward to tb
            writer.add_scalar("data/mean_episode_track", ep, self.global_steps)
        self.env.close()
        writer.close()

    def _qloss(self, state, reward, action, new_state, dones):
        qnv = self.qnetwork(state)
        if self.target:
            qv = self.qtarget(new_state)
        else:
            qv = self.qnetwork(new_state)
        if self.symbolic:
            qnvalues = torch.stack([qnv[i][0][action[i]] for i in range(len(action))])
            y = (torch.FloatTensor(reward).to(experiment_device) + self.gamma*dones.float()*torch.stack([torch.max(qv[i][0]) for i in range(len(action))]).float()).detach().to(experiment_device)
        else:
            qnvalues = torch.stack([qnv[i][action[i]] for i in range(len(action))])
            y = (torch.FloatTensor(reward).to(experiment_device) + self.gamma*dones.float()*torch.stack([torch.max(qv[i]) for i in range(len(action))]).float()).detach()
        loss = F.smooth_l1_loss(qnvalues, y)
        writer.add_scalar("data/loss", loss, self.global_steps) #log loss to tb
        return loss


    def _optimize(self):
        experience_sample = self.replay_buffer.sample(self.batch_size) #returns list of experience tuples
        experience_sampleT = np.array(experience_sample).T
        if self.symbolic:
            states = torch.stack([torch.tensor(list(s)) for s in list(experience_sampleT[0])]).to(experiment_device)
            new_states = torch.stack([torch.tensor(list(s)) for s in list(experience_sampleT[3])]).to(experiment_device)
        else:
            preprocessed_individual_states = [torch.stack([self.preprocessor(s) for s in list(d)], dim=1) for d in experience_sampleT[0]]
            states = torch.stack(preprocessed_individual_states, dim=1).squeeze().to(experiment_device)
            preprocessed_new_individual_states = [torch.stack([self.preprocessor(s) for s in list(d)], dim=1) for d in experience_sampleT[3]]
            new_states = torch.stack(preprocessed_new_individual_states, dim=1).squeeze().to(experiment_device)
        rewards = np.array(experience_sampleT[2], dtype=float)
        actions = np.array(experience_sampleT[1], dtype=int)
        dones = torch.from_numpy(np.ones(len(actions), dtype=float) - np.array(experience_sampleT[4], dtype=float)).to(experiment_device)
        loss = self._qloss(states, rewards, actions, new_states, dones)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.update_target:
            print('--------------Target Network Updated', self.global_steps)
            self.qtarget.load_state_dict(self.qnetwork.state_dict()) #reset target network to q network
            self.update_target = False


    def _step_env(self, action):
        """
        query environment to collect new experience
        """
        state, reward, done, info = self.env.step(action)
        reward += self.living_reward
        self.global_steps += 1
        if self.global_steps%self.sync_networks == 0:
            self.update_target = True
        if self.epsilon > self.final_epsilon and self.global_steps > self.replay_start:
            self.epsilon -= self.epsilon_step
        writer.add_scalar("data/epsilon", self.epsilon, self.global_steps) #log epsilon to tb
        return state, reward, done, info

    def _initialise_env(self):
        """
        ensures first phi represents stack of four states
        """
        states = []
        rewards = 0
        for i in range(self.stack_dim):
            state, reward, a, b = self._step_env(self.env.action_space.sample())
            if self.symbolic:
                states.append(state)
            else:
                states.append(state.astype('uint8'))
            rewards += reward
        return deque(states, maxlen=self.stack_dim), rewards


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
    if args.replay_start_size < args.batch_size:
        raise ValueError("Initial experience gathering must exceed batch size in order for samples to be drawn from buffer during training")
    symbolic = args.env == 4
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
        "living_reward": args.living_rew,
        "sync_networks": args.C,
        "target": args.target,
        "replay_start_size": args.replay_start_size,
        "num_episodes": args.num_episodes,
        "symbolic": symbolic
    }
    dqn = DQN(config)
    dqn.train()

if __name__ == "__main__":
    run()
    