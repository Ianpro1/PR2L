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


net = nn.Linear(4, 4)

x = torch.rand(size=(4,))

print(x)

logits = net(x)
probs = F.softmax(logits, dim=0)
log_probs = F.log_softmax(logits, dim=0)
print("initial", probs)

G_v = -0.5
policy_loss = -log_probs[2] * G_v
policy_loss.backward()

net.weight = nn.Parameter(net.weight - net.weight.grad)

logits = net(x)
probs = F.softmax(logits, dim=0)
print("final", probs)