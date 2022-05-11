from collections import namedtuple, deque
import gym
import numpy as np
import random

import network_sim
from vdn_agent import Agent
from vdn_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


from vdn_agent import *

def train_vdn():
    # NN policy produces an action
    # Take a step in the env
    # Agent.step()
    # All we need to do is add the two q values together.
    # Training parameters
    num_episodes = 200
    max_timesteps_per_episode = 500
    env = gym.make('PccNs-v0')
    gamma = 0.99
    epsilon = 1
    epsolon_decay = 0.995
    # 
    # TODO: get correct state and action sizes
    state_size = 10
    action_size = 10
    agent = Agent(state_size, action_size, seed=0)

    state = env.reset()
    for episode in range(num_episodes):
        print("----------Episode %d----------" % (episode))
        for t in range(max_timesteps_per_episode):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
        




if __name__ == "__main__":
    train_vdn()