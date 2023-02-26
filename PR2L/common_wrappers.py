import gym
import numpy as np
import cv2
from collections import deque
import math
class process84Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        

    def observation(self, obs):
        obs = process84Wrapper.process84(obs)
        return obs

    @staticmethod
    def process84(frame):
        if frame.size ==210*160*3 or 250*160*3:
            img = np.array(frame, dtype=np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.333 + img[:, :, 1] *0.333 + img[:,:,2] * 0.333
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        #minmaxscaling --temporary edit 
        x_t = (x_t - 70.) / 78.
        return x_t

class AutomateFireAction(gym.Wrapper):
    def __init__(self, env=None, penalize=0.0):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        #print(env.unwrapped.get_action_meanings())
        self.lives = None
        self.last = 0.0
        self.penalize = penalize
    def reset(self):
        obs, lives = self.env.reset()
        self.lives = lives["lives"]
        return obs, lives

    def step(self, action):
        obs, r, done, info, lives = self.env.step(action)
        self.lives = lives["lives"]
        if self.last > self.lives:
            self.last = self.lives
            obs, r, done, info, lives = self.env.step(1)
            if self.penalize:
                return obs, r - self.penalize, done, info, lives
            else:
                return obs, r, done, info, lives
        self.last = self.lives
        return obs, r, done, info, lives

class MaxAndSkipFireReset(gym.Wrapper):
    def __init__(self, env, frameskip=4):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self.skip = frameskip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self.skip):
            obs, reward, done, info, extra= self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        #max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return obs, total_reward, done, info, extra
        

    def reset(self):
        self._obs_buffer.clear()
        self.env.reset()
        obs, _, _, _, extra = self.env.step(1)
        self._obs_buffer.append(obs)

        return obs, extra

class BufferWrapper(gym.ObservationWrapper):
    #creates n channels for n steps
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.n_steps = n_steps

        #bad implementation
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(n_steps, 84, 84))

    def reset(self):
        obs, info = self.env.reset()
        self.buffer = []
        for _ in range(self.n_steps):
            obs, _, _, _, _ = self.env.step(0)
            self.buffer.append(obs)
        self.buffer = np.array(self.buffer)
        return self.buffer, info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ClipReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info, extra = self.env.step(action)
        if done:
            reward = -1
        return obs, reward, done, info, extra
    def reset(self):
        obs = self.env.reset()
        return obs


def WrapAtariEnv(env):
    env = AutomateFireAction(env)
    env = MaxAndSkipFireReset(env)
    env = process84Wrapper(env)
    env = BufferWrapper(env, n_steps=3)
    #env = ClipReward(env)
    return env


class LiveRenderWrapper(gym.Wrapper):
    #basic liverender wrapper although it is preferable to use funtionalObservationWrapper
    #Note: configured as an old env wrapper with 4 and 1 expected outputs from step() and reset(), respectively
    def __init__(self, env, func):
        super().__init__(env)
        self.F = func
    def step(self, action):
        obs = self.env.step(action)
        self.F(obs[0])
        return obs

    def reset(self):
        obs = self.env.reset()
        self.F(obs[0])
        return obs


class RGBtoFLOAT(gym.ObservationWrapper):
    def __init__(self, env, scaling_factor=255.):
        super().__init__(env)
        self.scaling_factor = scaling_factor
    def observation(self, observation):
        return observation / self.scaling_factor

class BetaSumBufferWrapper(gym.Wrapper):
    #make sure that image is scaled between 1 and -1 or 0 and 1
    def __init__(self, env, n_channel=5, Beta=0.5):
        super().__init__(env)
        self.buffer = []
        self.n_channel = n_channel
        self.beta = Beta
        self.factor = 1. * self.beta**(self.n_channel-1)
        for i in reversed(range(n_channel - 1)):
            self.factor += 1 * (self.beta**i)

    def __sum_img(self):
        self.buffer[-1] *= self.beta**(self.n_channel-1)
        for i in reversed(range(len(self.buffer) - 1)):
            self.buffer[-1] += self.buffer[i] * (self.beta**i)
        return (self.buffer[-1] / self.factor)

    def reset(self):
        obs, _ = self.env.reset()
        self.buffer = np.array([obs]*self.n_channel, dtype=np.float32)
        return self.__sum_img(), _
    
    def step(self, action):
        obs, rew, done, _, _ = self.env.step(action)
        self.__push(obs)
        return self.__sum_img(), rew, done, _, _
        
    def __push(self, new_obs):
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = new_obs



