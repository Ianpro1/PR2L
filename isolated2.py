from common import playground, models, extentions
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from PR2L import agent, experience, utilities
import os
import gym
import PR2L.playground as play
import time
'''N_OUT = 4
net = nn.Linear(1,N_OUT)

input = torch.tensor([[1.]])
max_ent = torch.full(size=(4,), fill_value=1/N_OUT)
max_ent = nn.functional.softmax(max_ent, dim=0) * nn.functional.log_softmax(max_ent, dim=0)
max_ent = -max_ent.sum()

print(max_ent)

for x in range(10):
    
    out = net(input)
    probs = nn.functional.softmax(out, dim=1)
    log_probs = nn.functional.log_softmax(out, dim=1)
    ent = (probs * log_probs).sum(dim=1).mean()
    print(probs)
    print(ent.item())
    loss = ent
    loss.backward()

    w_grad = net.weight.grad
    b_grad = net.bias.grad

    net.weight = nn.Parameter(net.weight - w_grad)
    net.bias = nn.Parameter(net.bias - b_grad)
    net.zero_grad()
    print("ent from max", max_ent - loss.absolute())'''

def obs(x, y):
    return np.random.choice([-1., 1.], size=(1,))

def rew(x, y):
    obs = x.cur_obs[0]
    act = y
    if obs == 1. and act == 0:
        return 1.0
    if obs == -1. and act == 1:
        return 1.0
    
    return -1.0

env = play.Dummy((1,), obs_func=obs, rew_func=rew)

net = models.LinearA2C((1,), 4)

_agent = agent.PolicyAgent(net)
preprocessor = agent.numpytoFloatTensor_preprossesing
exp_source = experience.ExperienceSourceV2(env, _agent, n_steps=1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

from collections import deque
avg_r = deque(maxlen=100)
for exp in exp_source:
    

    states, actions, rewards, last_states, not_dones = utilities.unpack_batch([exp])

    avg_r.append(rewards[0])
    print(np.array(avg_r).mean())
    states = preprocessor(states)
    rewards = preprocessor(rewards)
    last_states = preprocessor(last_states)
    actions = torch.LongTensor(np.array(rewards))

    tgt_q = rewards
    optimizer.zero_grad()
    act, value = net(states)

    value_loss = F.mse_loss(tgt_q.squeeze(-1), value.squeeze(-1))
    print("mse", value_loss)
    G_v = tgt_q - value.detach()

    probs = F.softmax(act, dim=1)
    log_probs = F.log_softmax(act, dim=1)
    
    policy_loss = -log_probs[0, actions] * G_v

    ent_loss = (log_probs * probs).sum()


    loss = policy_loss.mean() + value_loss + 0.01 * ent_loss

    loss.backward()

    optimizer.step()