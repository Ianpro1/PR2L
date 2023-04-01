import cython
from collections import namedtuple
import numpy as np
cimport numpy as np

Experience = namedtuple("Experience", ("state", "action", "reward", "next"))

cdef extern from "stdlib.h":
    void *malloc(size_t size)

class expe:
    
    def __init__(self, env, agent, n_steps, gamma):
        assert isinstance(env, (list, tuple))
        self.n_steps = n_steps
        self.gamma = gamma
        self.agent = agent
        self.env = env
        self.size = len(env) * n_steps

    def __iter__(self):
        states = [None] * self.size
        actions = [None] * self.size
        cdef np.ndarray[np.float32_t, ndim=1] rewards = np.zeros((self.size,), dtype=np.float32)
        cdef np.ndarray[np.int32_t, ndim=1] idd = np.zeros((len(self.env),), dtype=np.int32)
        cur_obs = []
        for i, env in enumerate(self.env):
            obs, _ = env.reset()
            cur_obs.append(obs)            
        
        cdef double prev = 0.0
        cdef double tot = 0.0
        cdef double cur = 0.0

        while True:   
            acts = self.agent(cur_obs)
            for i, env in enumerate(self.env):
                nextobs, reward, done, _, _ = env.step(acts[i])
                index = i * self.n_steps + idd[i]
                states[index] = cur_obs[i]             
                rewards[index] = reward
                actions[index] = acts[i]
                
                if done:
                #decay all
                    prev = 0.0
                    for n in range(idd[i]+1):
                        cur = rewards[index-n] + prev
                        prev = rewards[index-n]*self.gamma
                        exp = Experience(states[index-n], actions[index-n], cur, None)
                        yield exp
                        rewards[index-n] = 0.0
                    obs, _ = self.env[i].reset()
                    cur_obs[i] = obs
                    idd[i] = 0
                    continue
                
                cur_obs[i] = nextobs

                if idd[i] >= self.n_steps-1:
                    #decay oldest
                    tot = 0.0
                    for n in range(self.n_steps):
                        tot += rewards[i*self.n_steps + n] * self.gamma**n
                    exp = Experience(states[i*self.n_steps], actions[i*self.n_steps], tot, nextobs)
                    yield exp
                    rewards[i*self.n_steps:index-1] = rewards[i*self.n_steps+1:index]   
                else:
                    idd[i] += 1

