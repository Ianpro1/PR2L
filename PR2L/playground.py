#playground provides Dummy environments and tools for testing basic algorithms 
import gymnasium as gym
import numpy as np


class DeterministicObservations:
    """
    Once initialized, the class __call__ should always return the same observation on every given step.
    Since the observations are initially random, there are stored in a buffer of length = buffer_length.
    The second argument: input_shape, will defined the shape of every observation.
    """
    def __init__(self, buffer_length, input_shape):
        shape = [buffer_length]
        shape.extend(input_shape)
        self.buffer = np.random.uniform(-100., 100., size=shape)
        self.pos = 0
    def __call__(self, y, x):
        if x is None:
            self.pos = 0
        elif self.pos >= len(self.buffer):
            self.pos = 0

        obs = self.buffer[self.pos]
        self.pos += 1
        return obs

class DeterministicRewards:
    """
    Once initialized, the class __call__ should always return the same rewards on every given step.
    Since the rewards are initially random, there are stored in a buffer of length = buffer_length.
    """
    def __init__(self, buffer_length):
        self.buffer = np.random.choice(2, size=buffer_length, p=(0.20, 0.80))
        self.pos = 0
    def __call__(self, y, x):
        if x is None:
            self.pos = 0
        elif self.pos >= len(self.buffer):
            self.pos = 0

        rew = self.buffer[self.pos]
        self.pos += 1
        return rew

class Dummy(gym.Env):
    """
    A class environment inheriting gymnasium gym.Env.
    The class serves as dummy for testing algorithms, where you can defined custom observation functions, rewards functions and more.
    Using the default argument will result in an environment that uses DeterministicObservations() as obs_func and DeterministicRewards() as rew_func
    """
    def __init__(self, obs_shape, obs_func=None, rew_func=None, done_func=None, trunc_func=None, info_func=None):
        self.cur_obs = None
        self.shape = obs_shape
        
        if obs_func:                
            self.obs_func = obs_func
        else:
            self.obs_func = DeterministicObservations(1000, self.shape)
        
        if rew_func:
            self.rew_func = rew_func
        else:
            self.rew_func = DeterministicRewards(1000)

        if done_func:
            self.done_func = done_func
        else:
            self.done_func = lambda x, y: False

        if trunc_func:
            self.trunc_func = trunc_func
        else:
            self.trunc_func = lambda x, y: False
        
        if info_func:
            self.info_func = info_func
        else:
            self.info_func = lambda x, y: {"DummyEnv":True}

    def reset(self):
        action = None
        obs = self.obs_func(self, action)
        self.rew_func(self, action)
        self.done_func(self, action)
        self.trunc_func(self, action)
        info = self.info_func(self, action)

        self.cur_obs = (obs, info)
        return obs, info

    def step(self, action):
        obs = self.obs_func(self, action)
        rew = self.rew_func(self, action)
        done = self.done_func(self, action)
        trunc = self.trunc_func(self, action)
        info = self.info_func(self, action)

        self.cur_obs = (obs, rew, done, trunc, info)
        return obs, rew, done, trunc, info

class OldDummyWrapper(Dummy):
    """wraps the Dummy class to configure it as an older version of gym environments (removes truncatedflag from outputs, and info flag when reseting)"""
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, r, d, _, i = self.env.step(action)
        return obs, r, d, i
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs

class EpisodeLength:
    """
    Class that can be used for the dummy's done_func.
    The class will make each episode end after n steps (the environment's n+1 step will return done)
    """
    def __init__(self, length):
        self.len = length
        self.count = 0
    def __call__(self, y, x):
        if self.count > self.len -1:
            self.count = 0
            return True
        else:
            self.count +=1
            return False

#TODO environment that verifies the contents of exp_source and or buffer