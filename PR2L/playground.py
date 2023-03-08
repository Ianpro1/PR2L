#playground provides Dummy environments and tools for testing basic algorithms 
import gym
import numpy as np


class Dummy(gym.Env):
    def __init__(self, obs_shape, obs_func=None, rew_func=None, done_func=None, trunc_func=None, info_func=None):
        self.cur_obs = None
        self.shape = obs_shape
        if obs_func:
            self.obs_func = obs_func
        else:
            self.obs_func = lambda x, y: np.empty(shape=obs_shape)
        
        if rew_func:

            self.rew_func = rew_func
        else:
            self.rew_func = lambda x, y: 1.0

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
        obs = self.obs_func(self, None)
        info = self.info_func(self, None)
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
    #wraps the DummyEnv class to configure it as an older version of gym environments
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, r, d, _, i = self.env.step(action)
        return obs, r, d, i
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs
    
class EpisodeLength:
    #ends episode after n-1 steps (the nth step returns done)
    def __init__(self, length):
        self.len = length
        self.count = 0
    def __call__(self, y, x):
        if self.count > self.len -2:
            self.count = 0
            return True
        else:
            self.count +=1
            return False
        
class VaryObservation:
    #returns random pixel image observation of integer values 0->255 (in this case shape is user-defined)
    def __init__(self, dtype=np.uint8):
        self.dtype = dtype
    
    def __call__(self, y, x):
        x = np.random.randint(0, 255, size=y.shape).astype(self.dtype)
        return x

def ScaleRGBimage(y, x):
    #returns float version of RGB image with integer values from 0 to 255
    x = x.astype(np.float32) / 255.
    return x