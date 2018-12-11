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

parser = argparse.ArgumentParser(description='VPG Configuration')
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
parser.add_argument('-tt', '--training_time', default=20000, type=int, 
                    help='Length of time for which to train VPG')



class PolicyNetwork(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

class ValueNetwork(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

class VPG:

    def __init__(self):
        self.policy = PolicyNetwork()
        self.value = ValueNetwork()

        self.training_time = self.config["training_time"]

    def train(self):
        for k in range(self.training_time):
            trajectories = 


class TrajBuffer:

    def __init__(self):
        


def run():
    args = parser.parse_args()
    if args.replay_start_size < args.batch_size:
        raise ValueError("Initial experience gathering must exceed batch size in order for samples to be drawn from buffer during training")
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
        "C": args.C,
        "target": args.target,
        "replay_start_size": args.replay_start_size,
        "num_episodes": args.num_episodes,
        "training_time": args.training_time
    }
    dqn = DQN(config)
    dqn.train()


if __name__ == "__main__":
    run()    