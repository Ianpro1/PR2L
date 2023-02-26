from common import playground, models
import numpy as np
import torch
from PR2L import experience, agents
import ptan
import timeit


#performance comparison between PR2L ExperienceSource, PR2L ExperienceSourcev2 and ptan's equivalent: ExperienceSourceFirstLast


env1 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(6), observation_func=playground.VaryObservation(np.float32))
env2 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(6), observation_func=playground.VaryObservation(np.float32))
env3 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(6), observation_func=playground.VaryObservation(np.float32))
env = [env1, env2, env3]

net = models.DenseDQN(1, 128, 4)

agent = agents.BasicAgent(net)
exp_source = experience.ExperienceSource(env, agent, 3, 0.99)
exp_sourcev2 = experience.ExperienceSourcev2(env, agent, 3, 0.99)

def test1():
    list = []
    for i, x in enumerate(exp_source):
        #print(x, "\n")
        list.append(x)
        if i > 100:
            break
    return list

speed1 = timeit.timeit(test1, number=1000)

def test2():
    list = []
    for i, x in enumerate(exp_sourcev2):
        #print(x, "\n")
        list.append(x)
        if i > 100:
            break
    return list

speed2 = timeit.timeit(test2, number=1000)

env1 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(6), observation_func=playground.VaryObservation(np.float32))
env1 = playground.OldDummyWrapper(env1)
env2 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(6), observation_func=playground.VaryObservation(np.float32))
env2 = playground.OldDummyWrapper(env2)
env3 = playground.DummyEnv((1,), userWarning=False, done_func=playground.EpisodeLength(6), observation_func=playground.VaryObservation(np.float32))
env3 = playground.OldDummyWrapper(env3)
env = [env1, env2, env3]

ptan_agent = ptan.agent.DQNAgent(net, action_selector=ptan.actions.ArgmaxActionSelector())
ptan_exp_source = ptan.experience.ExperienceSourceFirstLast(env, ptan_agent, 0.99, steps_count=3)

def test3():
    list = []
    for i, x in enumerate(ptan_exp_source):
        #print(x, "\n")
        list.append(x)
        if i > 100:
            break
    return list

speed3 = timeit.timeit(test3, number=1000)
#timeit results from test1->test3: 12.535710900003323 11.351821199998085 16.51639150000119

print(speed1, speed2, speed3)
 