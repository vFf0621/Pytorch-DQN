
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:08:27 2022

@author: guanfei1
"""

import gym
import numpy as np
from Model import *
import torch
batch_size = 32
gamma = 0.99
if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    agent = Agent(env)
    for i in range(100):
        s = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            s_, r, done, _ = env.step(action)
            s_ = s_.tolist()
            agent.store_transition((s, action, r, done, s_))
            s = s_ 
        s = env.reset()
    print("Replay Buffer Initialized")

    for episode in range(15000):
        done = False
        episode_reward = 0
        agent.target_Q.load_state_dict(agent.Q.state_dict())
        while not done:
            action = agent.select_action(s)
            s_, r, done, _ = env.step(action)
            s_ = s_.tolist()
            agent.store_transition((s, action, r, done, s_))
            episode_reward += r
            states, actions, rewards, dones, states_ = agent.sample(batch_size)
            target_q_values = agent.target_Q(states_)
            max_target_q = target_q_values.max(dim=1, keepdims=True)[0]
            q_values = agent.forward(states)
            action_q_values = torch.gather(input=q_values, dim=1, \
                                           index=actions)

            targets = rewards + gamma * ((1 - dones)* max_target_q)
            agent.train(action_q_values, targets)
            if episode > 14000:
                env.render()
                
            s = s_
        agent.store_episode_reward(episode_reward)

        env.reset()
        print("Rewards: ", np.mean(agent.reward_buffer))
    while True:
        s = env.reset()
        done = False
        while not done:
            action = agent.select_action(s)
            s_, _, done, _ = env.step(action)
            s = s_
            
            
    