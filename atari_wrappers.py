import gym
import numpy as np
import cv2
import gym.spaces
import collections


# custom wrappers

class reshapeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        obs_shape = np.array(env.observation_space.shape)
        self.observation_space = gym.spaces.Box(low=0.0, high=255., shape=obs_shape[[2,0,1]], dtype=np.float32)

    def observation(self, obs):
        return obs.transpose(2,0,1)


class oldStepWrapper(gym.Wrapper):
    """for outdated wrappers that expects 4 values to unpack from env.step() method,
    Note: ObservationWrappers should be applied first else -> error expected 5 got 4"""
    def __init__(self, env=None, record_count=None):
        super().__init__(env)
        if record_count is not None:
            self.recorder = collections.deque(maxlen=record_count)
        else:
            self.recorder = None

    def step(self, action):
        obs, reward, done, info, extra = self.env.step(action)
        if self.recorder is not None:
            self.recorder.append(extra)
        return obs, reward, done, info
        
    def reset(self):
        obs = self.env.reset()
        if self.recorder is not None:
            self.recorder.clear()
        return obs

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


# well known wrappers (Some are incompatible with gym >= 0.26 and need fix due to outdated gym observations which used to return 4 objects)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
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
            obs, reward, done, info= self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
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
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


'''def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)'''

