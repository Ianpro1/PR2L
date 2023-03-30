import array
from collections import deque
from PR2L.experience import Experience

#doesn't work but performance gains are not impressive
class exp:
    def __init__(self, env, agent, n_steps, GAMMA):
        self._len = len(env)
        self.agent = agent
        self.n_steps = n_steps
        self.env = env
        self.GAMMA = GAMMA

    def __iter__(self):
        states = [] 
        actions = []
        cur_obs = []
        init = [0.]*self.n_steps*self._len
        rewards = array.array('d', init)
        idd = [0]*self._len
        
        for e in self.env:
            states.append(deque(maxlen=self.n_steps))
            actions.append(deque(maxlen=self.n_steps))
            obs, _ = e.reset()
            cur_obs.append(obs)
        
        while(True):
            acts = self.agent(cur_obs)
            for i, env in enumerate(self.env):
                obs, rew, done, trunc, info = env.step(acts[i])
                states[i].append(cur_obs[i])
                actions[i].append(acts[i])
                rewards[idd[i] + self.n_steps * i] = rew

                if done:
                    rews = self.__decay_all(rewards[self.n_steps * i:idd[i]]) #also zero out
                    
                    for r in reversed(rews):
                        exp = Experience(states[i].popleft(), actions[i].popleft(), r, None)
                        yield exp
                    
                    rews = 0
                    obs, _ = self.env[i].reset()
                    cur_obs[i] = obs
                    idd[i] = 0
                    continue
                
                cur_obs[i] = obs


                if idd[i] >= self.n_steps - 1:
                    
                    #decay for only the oldest
                    old = self.__decay_oldest(rewards[self.n_steps * i:idd[i]]) #do a shift

                    exp = Experience(states[i].popleft(), actions[i].popleft(), old, obs)
                    yield exp
                
                if idd[i] < self.n_steps-1:
                    idd[i] += 1
    
    def __decay_all(self, arr):
        prev = 0.
        for i in reversed(range(self.n_steps)):
            arr[i] = arr[i] + prev* self.GAMMA
            prev = arr[i]
        return arr
    
    def __decay_oldest(self, arr):

        tot = 0
        for i, r in enumerate(arr):
            tot += r * self.GAMMA**i
        arr[:-1] = arr[1:]
        return tot

