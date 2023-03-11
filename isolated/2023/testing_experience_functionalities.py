from collections import namedtuple, deque
import timeit
from PR2L import playground, experience, agent
import torch
import torch.nn as nn
import gym
import numpy as np
net = nn.Linear(3, 3)

NUM = 100
_agent = agent.BasicAgent(net)
env = playground.Dummy((3,), done_func=playground.EpisodeLength(5))

exp_source2 = experience.ExperienceSourceV2(env, _agent,)
exp_source = experience.ExperienceSource(env, _agent,)

print("---test1---")

def test1():
    for i, x in enumerate(exp_source):
        
        print(x)
        if i > 10:
            break
test1()
print("---test2---")
def test2():
    for i, x in enumerate(exp_source2):
        print(x)
        if i > 10:
            break
test2()

'''
print(timeit.timeit(test1, number=NUM))
print(timeit.timeit(test2, number=NUM))'''
