
from common import models
import numpy as np
import torch
from PR2L import agent, experience, playground
import ptan
import timeit
import time

class debugEnv:
    def __init__(self):
        self.timeline = list(range(3))
        self.pos = 0
        self.rew = [1]*(len(self.timeline)-1)
    
    def step(self, x):
        obs = self.timeline[self.pos]
        rew = self.rew[self.pos-1]
        if (self.pos == len(self.rew)):
            done = True
        else:
            done = False
        self.pos += 1
        return [obs], rew, done, False, ""
    
    def reset(self,):
        self.pos = 1
        return [self.timeline[0]], ""



N_STEPS = 4
#performance comparison between PR2L ExperienceSource, PR2L ExperienceSourcev2 and ptan's equivalent: ExperienceSourceFirstLast
REP=1
EPISODE_LENGTH = 4

env = [debugEnv() for _ in range(3)]

net = models.DenseDQN(1, 128, 4)

ag = agent.BasicAgent(net)
exp_source = experience.ExperienceSource(env, ag, N_STEPS, 0.99, False)


def test1(idx=10000, bprint=False):
    list = []
    for i, x in enumerate(exp_source):
        if bprint:
            print(x, "\n")
        list.append(x)
        if i > idx:
            break
    return list

class experienceIterator:
    def __init__(self, env, agent) -> None:
        self.agent = agent
        self.envs = env
        self.dbuffer = experience.SimpleDecayBuffer(len(env), N_STEPS, 0.99, False)

    def __iter__(self):
        #init
        _obs = []
        for i,e in enumerate(self.envs):
           obs1, _ = e.reset()
           _obs.append(obs1)
           self.dbuffer._add(i, obs1)
        
        #loop
        while(1):
            acts, _ = self.agent(_obs, None)
            k = acts[:]
            
            for i, e in enumerate(self.envs):
                tuple1 = e.step(k[i])

                obs, rew, done, _, _ = tuple1
                _obs[i] = obs
                self.dbuffer._add(i, obs, acts[i], rew, done)

                if (done):
                    obs1, _ = e.reset()
                    self.dbuffer._add(i, obs1)
                    _obs[i] = obs1
                
                for _ in range(len(self.dbuffer)):
                    yield self.dbuffer.pop_left()


exp_sourcev2 = experienceIterator(env, ag)

def test2(idx=100, bprint=False):
    list = []
    for i, x in enumerate(exp_sourcev2):
        if bprint:
            print(x, "\n")
        list.append(x)
        if i > idx:
            break
    return list


ti = time.time()
test1(1000000, 0)
print(time.time() - ti)
print("--------Test 2 --------")
ti = time.time()
test2(1000000, 0)
print(time.time() - ti)
