#this file is for testing purposes with a dumb linear environment
import numpy as np
import gym

class DummyEnv(gym.Env):
    # environment that performs simple actions (user-defined)
    # configured as a gym env from ~ v0.26
    def __init__(self, observation_shape, reward_value=1.0, reward_func=None, observation_func=None, done_func=None, userWarning=True):
        self.r_value = reward_value

        if reward_func is not None:
            self.r_func = reward_func
        else:
            self.r_func = self.doNothing

        if observation_func is not None:
            self.obs_func = observation_func
        else:
            self.obs_func = self.doNothing

        if done_func is not None:
            self.d_func = done_func
        else:
            self.d_func = self.doNothing
        self.warning = userWarning
        self.last_obs = []
        self.obs = np.empty(observation_shape, dtype=np.float32)

    @staticmethod
    def doNothing(x):
        return x
    
    def reset_last(self):
        last_obs = [None] * 5
        last_obs[2] = False
        return last_obs

    def observation(self, obs):
        return self.obs_func(obs)

    def done(self, done):
        return self.d_func(done)

    def reward(self, r):
        return self.r_func(r)

    def reset(self):
        if self.warning:
            print("Dummyreset...")
        info = None
        self.last_obs = self.reset_last()
        obs = (self.observation(self.obs), info)
        self.obs = obs
        return obs

    def step(self, action):
        info = None
        #prevent undefined behavior
        if self.last_obs[2] == True:
            if self.warning:
                raise UserWarning("env.step() was called while episode was over!")
            return self.last_obs
        else:
            obs = (self.observation(self.obs), self.reward(self.r_value), self.done(False), False, info)
            self.last_obs = obs
            self.obs = obs
            return obs

class OldDummyWrapper(DummyEnv):
    #wraps the DummyEnv class to configure it as an older version of gym environments
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, r, d, i, _ = self.env.step(action)
        return obs, r, d, i
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs

#example of custom processing

class EpisodeLength:
    #ends episode after n-1 steps (the nth step returns done)
    def __init__(self, length):
        self.len = length
        self.count = 0
    def __call__(self, done):
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
    
    def __call__(self, x):
        x = np.random.randint(0, 255, size=x.shape).astype(self.dtype)
        return x

def ScaleRGBimage(x):
    #returns float version of RGB image with integer values from 0 to 255
    x = x.astype(np.float32) / 255.
    return x