from common import playground, models
import numpy as np
import torch
from PR2L import agent, experience
import ptan
import timeit

N_STEPS = 1000
#performance comparison between PR2L ExperienceSource, PR2L ExperienceSourcev2 and ptan's equivalent: ExperienceSourceFirstLast
REP=1
EPISODE_LENGTH = 3000
env1 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(EPISODE_LENGTH), observation_func=playground.VaryObservation(np.float32))
env2 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(EPISODE_LENGTH), observation_func=playground.VaryObservation(np.float32))
env3 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(EPISODE_LENGTH), observation_func=playground.VaryObservation(np.float32))
env = [env1, env2, env3]

net = models.DenseDQN(1, 128, 4)

agent = agent.BasicAgent(net)
exp_source = experience.ExperienceSource(env, agent, N_STEPS, 0.99)
exp_sourcev2 = experience.ExperienceSourceV2(env, agent, N_STEPS, 0.99)

def test1(idx=10000):
    list = []
    for i, x in enumerate(exp_source):
        #print(x, "\n")
        list.append(x)
        if i > idx:
            break
    return list

def speedtest1():
    return timeit.timeit(test1, number=REP)

speed1 = speedtest1()

def test2(idx=10000):
    list = []
    for i, x in enumerate(exp_sourcev2):
        #print(x, "\n")
        list.append(x)
        if i > idx:
            break
    return list

def speedtest2():
    return timeit.timeit(test2, number=REP)

speed2 = speedtest2()

env1 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(6), observation_func=playground.VaryObservation(np.float32))
env1 = playground.OldDummyWrapper(env1)
env2 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(6), observation_func=playground.VaryObservation(np.float32))
env2 = playground.OldDummyWrapper(env2)
env3 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(6), observation_func=playground.VaryObservation(np.float32))
env3 = playground.OldDummyWrapper(env3)
env = [env1, env2, env3]

ptan_agent = ptan.agent.DQNAgent(net, action_selector=ptan.actions.ArgmaxActionSelector())
ptan_exp_source = ptan.experience.ExperienceSourceFirstLast(env, ptan_agent, 0.99, steps_count=N_STEPS)

def test3(idx=10000):
    list = []
    for i, x in enumerate(ptan_exp_source):
        #print(x, "\n")
        list.append(x)
        if i > idx:
            break
    return list

def speedtest3():
    return timeit.timeit(test3, number=REP)
#timeit results from test1->test3: 12.535710900003323 11.351821199998085 16.51639150000119
speed3 = speedtest3()


print(speed1, speed2, speed3)
'''for exp in test2(10):
    print(exp)
print('\n' * 5)
for exp in test3(10):
    print(exp)'''
 