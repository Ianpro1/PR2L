import gym
import numpy as np
import cv2
import gym.spaces
import collections


# custom wrappers
class functionalObservationWrapper(gym.ObservationWrapper):
    #applies any function to obs
    def __init__(self, env, func):
        super().__init__(env)
        self.F = func
    def observation(self, obs):
        return self.F(obs)

class reshapeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        obs_shape = np.array(env.observation_space.shape)
        self.observation_space = gym.spaces.Box(low=0.0, high=255., shape=obs_shape[[2,0,1]], dtype=np.float32)
        
    def observation(self, obs):
        return obs.transpose(2,0,1)

class ListWrapper(gym.ObservationWrapper):   
    #may be required for discrete environment observations    
    def observation(self, obs):
        return [obs]

class oldStepWrapper(gym.Wrapper):
    """for outdated wrappers that expects 4 values to unpack from env.step() method,
    Note: ObservationWrappers should be applied first else -> error expected 5 got 4"""
    def __init__(self, env=None, ):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        return obs, reward, done, info


class oldResetWrapper(gym.Wrapper):
    def __init__(self, env, rrecord_count=None):
        super().__init__(env)
        if rrecord_count is not None:
            self.rrecorder = collections.deque(maxlen=rrecord_count)
        else:
            self.rrecorder = None


    def reset(self):
        obs, extra = self.env.reset()
        if self.rrecorder is not None:
            self.rrecorder.append(extra)
        return obs


class oldWrapper(gym.Wrapper):
    # Note: lost information is not recorded (no recorder)
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        return obs[0]
    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        return obs, reward, done, info

class RenderWrapper(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.frames = []

    def step(self, action):
        obs, rewards, done, info, _ = self.env.step(action)
        self.frames.append(obs)
        return obs, rewards, done, info, _

    def pop_frames(self):
        frames = self.frames.copy()
        self.frames.clear()
        return np.array(frames)

#bad implementation
'''class SingleLifeWrapper(gym.Wrapper):
    #Only works for atari:Breakout
    #instead of returning fake observation, try doing firestep for the agent
    def __init__(self, env=None):
        super().__init__(env)
        self.lives = 0
        self.last = None
        self.last_obs = None
    def reset(self):
        if self.lives == 0:
            obs = self.env.reset()
            self.lives = 5
            self.last = 5
            return obs
        else:
            #this fails after 2 consecutives resets
            return (self.last_obs[0], self.last_obs[4])    
    def step(self, action):
        obs, r, done, info, lives = self.env.step(action)
        self.lives = lives["lives"]
        #print(self.lives)
        if self.last > self.lives:
            
            self.last = self.lives
            self.last_obs = obs
            return (obs, r, True, info, lives)
        self.last = self.lives
        return (obs, r, done, info, lives)'''

class AutomateFireAction(gym.Wrapper):
    def __init__(self, env=None, penalize=0.0):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self.lives = None
        self.last = 0.0
        self.penalize = penalize
    def reset(self):
        obs, lives = self.env.reset()
        self.lives = lives["lives"]
        return (obs, lives)

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
        return (obs, r, done, info, lives)

# well known wrappers (Some are incompatible with gym >= 0.26 and need fix due to outdated gym observations which used to return 4 objects)

class FireResetEnv(gym.Wrapper):
    #early wrapper slightly improve see MaxAndSkip
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs = self.env.step(1)
        if obs[2]:
            self.env.reset()
        obs = self.env.step(2)
        if obs[2]:
            self.env.reset()
        return (obs[0], {})


class MaxAndSkipEnv(gym.Wrapper):
    #slight improvement made: takes 5 positional arguments instead of 4 (for newer environment and perfomance) *is applied early
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info, extra= self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info, extra

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, extra = self.env.reset()
        self._obs_buffer.append(obs)
        return obs, extra


class ProcessFrame84(gym.ObservationWrapper):
    '''Scaling every frame down from 210×160, with three color frames, to a single-color 
84×84 image. Different approaches are possible. For example, the DeepMind 
paper describes this transformation as taking the Y-color channel from the 
YCbCr color space and then rescaling the full image to an 84×84 resolution. 
Some other researchers do grayscale transformation, cropping non-relevant 
parts of the image and then scaling down. In the Baselines repository (and in 
the following example code), the latter approach is used'''

    def __init__(self, env=None, fReshape=(84,84,1)):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.fReshape = fReshape

    def observation(self, obs):
        return ProcessFrame84.process(obs, self.fReshape)

    @staticmethod
    def process(frame, reshape):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, reshape)
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env, maxchannel_v=255.):
        super().__init__(env)
        self.maxchannel_v=maxchannel_v
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / self.maxchannel_v


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
       
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0), old_space.high.repeat(n_steps, axis=0), dtype=dtype)

        

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation[0]
        return self.buffer

