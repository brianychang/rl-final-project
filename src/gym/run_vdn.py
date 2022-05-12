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
    num_episodes = 1000
    env = gym.make('PccNs-v0')
    gamma = 0.99
    epsilon = 1
    epsolon_decay = 0.995
    # 
    # TODO: get correct state and action sizes
    state_size = 30
    action_size = 11
    agent = Agent(state_size, action_size, seed=0)

    # state = env.reset()
    for episode in range(num_episodes):
        done = False
        state = env.reset()
        # print("----------Episode %d----------" % (episode))
        while not done:
            action = agent.act(state, epsilon)
            # action[0], action[1] are the actions of the 2 agents.
            next_state, reward, done, _ = env.step(action)
            # print("rewardreward ", reward)
            agent.step(state, action, reward, next_state, done)
            state = next_state

        # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_ep%d_agent1.pth'%(episode))
        # torch.save(agent.qnetwork_local2.state_dict(), 'checkpoint_ep%d_agent2.pth'%(episode))



if __name__ == "__main__":
    print("=====================Main training loop=====================")
    train_vdn()