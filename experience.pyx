from collections import deque, namedtuple
Experience = namedtuple("Experience", ("state", "action", "reward", "next"))

class ExperienceSource:
    #remove all deque and arrays related to done flags since last_state=None is essentially done
    def __init__(self, env, agent, n_steps=2, GAMMA=0.99, track_rewards=True):

        assert isinstance(env, (list, tuple))
                
        self.n_steps = n_steps

        if isinstance(env, (list, tuple)):
            self.env = env
            env_len = len(env)
        else: 
            self.env = [env]
            env_len = 1
        self.agent = agent
        self.env_len = env_len    
        self.track_rewards = track_rewards

        if track_rewards:
            self.tot_reward = [0.]*env_len
            self.tot_rewards = []
            self.tot_step = [0.]*env_len
            self.tot_steps = []

        self.gamma = GAMMA
        self.n_eps_done = 0

    def __iter__(self):
        _states = []
        _rewards = []
        _actions = []
        cur_obs = []
        for e in self.env:
            _states.append(deque(maxlen=self.n_steps))
            _rewards.append(deque(maxlen=self.n_steps))
            _actions.append(deque(maxlen=self.n_steps))
            obs, _ = e.reset()
            cur_obs.append(obs)

        while True:   
            actions = self.agent(cur_obs)
            for i, env in enumerate(self.env):
                nextobs, reward, done, _, _ = env.step(actions[i])
                _actions[i].append(actions[i])
                _rewards[i].append(reward)
                _states[i].append(cur_obs[i])
                if self.track_rewards:
                    self.__sum_rewards_steps(reward, done, i)
                if done:
                    print("terminated from env id: ", i)
                    #decay all
                    decayed = self.__decay_all_rewards(_rewards[i])
                    _rewards[i].clear()
                    for r_id in range(1, len(decayed)+1): #used to get proper length
                        
                        exp = Experience(_states[i].popleft(), _actions[i].popleft(), decayed[-r_id], None)
                        yield exp
                    self.n_eps_done += 1
                    obs, _ = self.env[i].reset()
                    cur_obs[i] = obs
                    continue
                
                cur_obs[i] = nextobs

                #len(deque) is O(n)! TODO -> try to place rewards in an array instead of deque
                if len(_actions[i]) == self.n_steps:
                    #decay for only the oldest
                    self.__decay_oldest_reward(_rewards[i])

                    exp = Experience(_states[i].popleft(), _actions[i].popleft(), _rewards[i].popleft(), nextobs)
                    yield exp

    def __decay_all_rewards(self, rewards):
        prev = 0.0
        res = []
        for r in reversed(rewards):
            r += prev
            res.append(r)
            prev = r * self.gamma
        return res

    def __decay_oldest_reward(self, rewards):
        tot = 0.0
        for r in reversed(rewards):
             tot = r + tot* self.gamma
        rewards[0] = tot

    def __sum_rewards_steps(self, reward, done, env_id):
        #keeps track of rewards and steps
        self.tot_step[env_id] += 1
        self.tot_reward[env_id] += reward
        if done:
            self.tot_rewards.append(self.tot_reward[env_id])
            self.tot_reward[env_id] = 0
            self.tot_steps.append(self.tot_step[env_id])
            self.tot_step[env_id] = 0
    

    def pop_rewards_steps(self):
        assert self.track_rewards is True
        res = list(zip(self.tot_rewards, self.tot_steps))
        if res:
            self.tot_rewards.clear()
            self.tot_steps.clear()
        return res  