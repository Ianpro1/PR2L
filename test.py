from common import models
import numpy as np
import torch
from PR2L import agent, experience, playground
import ptan
import timeit

N_STEPS = 1000
#performance comparison between PR2L ExperienceSource, PR2L ExperienceSourcev2 and ptan's equivalent: ExperienceSourceFirstLast
REP=1
EPISODE_LENGTH = 3000

env = [playground.Dummy((1,)) for _ in range(3)]

net = models.DenseDQN(1, 128, 4)

agent = agent.BasicAgent(net)
exp_source = experience.ExperienceSource(env, agent, N_STEPS, 0.99)
exp_sourcev2 = experience.ExperienceSourceV2(env, agent, N_STEPS, 0.99)

def test1(idx=1000, bprint=False):
    list = []
    for i, x in enumerate(exp_source):
        if bprint:
            print(x, "\n")
        list.append(x)
        if i > idx:
            break
    return list

def speedtest1():
    return timeit.timeit(test1, number=REP)

speed1 = speedtest1()

def test2(idx=1000, bprint=False):
    list = []
    for i, x in enumerate(exp_sourcev2):
        if bprint:
            print(x, "\n")
        list.append(x)
        if i > idx:
            break
    return list

def speedtest2():
    return timeit.timeit(test2, number=REP)

speed2 = speedtest2()

print(speed1, speed2)

test1(3, 1)
print("--------Test 2 --------")
test2(3, 1)





