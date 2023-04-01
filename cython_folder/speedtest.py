from common import models
import numpy as np
import torch
from PR2L import agent, experience, playground
import ptan
import timeit
from exp import expe


N_STEPS = 4
#performance comparison between PR2L ExperienceSource, PR2L ExperienceSourcev2 and ptan's equivalent: ExperienceSourceFirstLast
REP=1
EPISODE_LENGTH = 30

env = [playground.Dummy((1,)) for _ in range(3)]

net = models.DenseDQN(1, 128, 4)

ag = agent.BasicAgent(net)
exp_source = experience.ExperienceSource(env, ag, N_STEPS, 0.99, False)


exp_sourcev2 = expe(env, ag, N_STEPS, 0.99)

def test1(idx=10000, bprint=False):
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

def test2(idx=10000, bprint=False):
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

#test1(3, 1)
print("--------Test 2 --------")
#test2(3, 1)





