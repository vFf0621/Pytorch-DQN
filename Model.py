#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:18:39 2022

@author: guanfei1
"""

import torch.nn as nn
import torch
import numpy as np
import torch.optim
import random
from collections import deque
class Q(nn.Module):
    
    def __init__(self, env, hidden_dim = 200, lr=0.0005):
        super().__init__()
        action_space = env.action_space.n
        state_space = env.observation_space.shape[0]

        self.net = nn.Sequential(nn.Linear(state_space, hidden_dim),  
                                 nn.Sigmoid(), 
                                 nn.Linear(hidden_dim, action_space))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        
    def act(self, x):
        x = self.forward(x)
        x = torch.argmax(x)

        return x.cpu().detach().item()
    def forward(self, x):
        return self.net(x)
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
class Agent:
    def __init__(self, env, epsilon_start = 0.5, decay = 0.9999):
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                   else "cpu")
    
        self.env = env
        self.Q = Q(env)
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_decay = decay
        self.criterion = nn.SmoothL1Loss()
        self.reward_buffer = deque(maxlen=100)
        self.replay_buffer = deque(maxlen=1000000)
        self.Q.to(self.device)
        self.target_Q = Q(env)
        self.target_Q.to(self.device)
    def select_action(self, state):
        e = random.random()
        state = torch.tensor(state).to(self.device)
        self.epsilon *= self.epsilon_decay
        if self.epsilon < 0.01:
            self.epsilon = 0.01
        if e < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.Q.act(state)
        return action
    def forward(self, x):
        x = x.to(self.device)
        return self.Q(x)
    
    def target_forward(self, x):
        x = x.to(self.device)
        return self.target_Q(x)
    def train(self, source, target):
        loss = self.criterion(source, target)
        self.Q.update(loss)
    def get_Q(self):
        return self.Q
        
    def avg_rew(self):
        return np.mean(self.reward_buffer)
    def store_transition(self, t):
        self.replay_buffer.append(t)

    def store_episode_reward(self, r):
        self.reward_buffer.append(r)        
    def sample(self, batch_size):
        t = random.sample(self.replay_buffer, batch_size)
        actions = []
        states = []
        dones = []
        states_ = []
        rewards = []
        for i in t:
            state, action, reward, done, state_ = i
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            states_.append(state_)
        states = torch.tensor(states, dtype=torch.float32).\
        to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).\
        to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).\
            to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).\
            to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        
        return states, actions, rewards, dones, states_
    