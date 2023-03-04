from common import playground, models, extentions
import numpy as np
import torch
import torch.nn as nn
from PR2L import experience, agents, utilities
import os
import gym

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

print(2%2)


