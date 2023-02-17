import common.extentions as ex
import common.models as models
import torch
import torch.nn as nn
import gym
import numpy as np
from collections import namedtuple, deque

class ActionSelector:
    def __call__(self, x):
        raise NotImplementedError


class ArgmaxSelector(ActionSelector):

    def __init__(self):
        super().__init__()

    def __call__(self,x):
        return x.argmax(dim=1)

class Agent:
    def __call__(self):
        raise NotImplementedError

def numpytotensor_preprossesing(x):
    return torch.tensor(np.array(x))

class BasicAgent(Agent):
    def __init__(self, net, device="cpu", Selector= ArgmaxSelector(), preprocessing=numpytotensor_preprossesing):
        super().__init__()
        assert isinstance(Selector, ActionSelector)
        self.selector = Selector
        self.net = net
        self.device = device
        self.preprocessing = preprocessing

    @torch.no_grad()
    def __call__(self, x):
        x = self.preprocessing(x)
        x.to(self.device)
        values = self.net(x)
        return self.selector(values)


class ExperienceSource:

    def __init__(self, env, agent, n_steps=2):
        assert isinstance(env, (gym.Env, list, tuple))
        if isinstance(env, (list, tuple)):
            self.env = env
        else: 
            self.env = [env]
        self.agent = agent
        self.n_steps = n_steps
        self.states = []
        self.actions = []
        self.rewards = []
        
    def __iter__(self):
        
        pass




device = "cpu"
env = gym.make("Breakout-v4")

class wrapobs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, obs):
        return np.moveaxis(obs, -1, 0).astype(np.float32)
env = wrapobs(env)

net = models.DualDQN((3,210,160), 4)

agent = BasicAgent(net)





obs, _ = env.reset()
print(obs.shape)
print(agent([obs]))