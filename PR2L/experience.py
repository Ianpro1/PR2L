import gym
from collections import namedtuple, deque
from .agents import Agent


#Experience tuple not implemented yet
Experience = namedtuple("Experience", ("state", "action", "reward", "done"))

NextExperience = namedtuple("Experience", ("state", "action", "reward", "done", "next"))
          
class ExperienceSource:
    def __init__(self, env, agent, n_steps, GAMMA=0.99):
        assert isinstance(agent, Agent)
        assert isinstance(env, (gym.Env, list, tuple))
        if isinstance(env, (list, tuple)):
            self.env = env
            env_len = len(env)
        else: 
            self.env = [env]
            env_len = 1
        self.agent = agent
        self.env_len = env_len        
        self.tot_reward = [0.]*env_len
        self.tot_rewards = []
        self.tot_step = [0.]*env_len
        self.tot_steps = []
        self.n_steps = n_steps
        self.gamma = GAMMA

    def __iter__(self):
        _states = []
        _rewards = []
        _actions = []
        _dones = []
        _nextstates = []
        cur_obs = []
        for e in self.env:
            _states.append(deque(maxlen=self.n_steps))
            _rewards.append(deque(maxlen=self.n_steps))
            _actions.append(deque(maxlen=self.n_steps))
            _dones.append(deque(maxlen=self.n_steps))
            _nextstates.append(deque(maxlen=self.n_steps))
            obs, _ = e.reset()
            cur_obs.append(obs)

        while True:
            actions = self.agent(cur_obs)

            for i, env in enumerate(self.env):
                nextobs, reward, done, info, _ = env.step(actions[i])

                _states[i].append(cur_obs[i])
                _actions[i].append(actions[i])
                _rewards[i].append(reward)
                _dones[i].append(done)
                self.__sum_rewards_steps(reward, done, i)
                if done:
                    _nextstates[i].append(None)
                    #decay all
                    _rewards[i] = self.decay_all_rewards(_rewards[i], self.gamma)
                    for _ in range(len(_dones[i])):
                        exp = NextExperience(_states[i].popleft(), _actions[i].popleft(), _rewards[i].popleft(), _dones[i].popleft(), _nextstates[i].popleft())
                        yield exp
                    obs, _ = self.env[i].reset()
                    cur_obs[i] = obs
                    continue
                cur_obs[i] = nextobs
                _nextstates[i].append(nextobs)
                if len(_dones[i]) == self.n_steps:
                    #decay for only the oldest
                    _rewards[i] = self.decay_oldest_rewards(_rewards[i], self.gamma)
                    exp = NextExperience(_states[i].popleft(), _actions[i].popleft(), _rewards[i].popleft(), _dones[i].popleft(), _nextstates[i].popleft())
                    yield exp
    
    def decay_all_rewards(self, rewards, gamma):
        for i in range(len(rewards)):
            if i ==0:
                r1 = rewards.pop()
                r2 = 0
            else:
                r1 = rewards.pop()
                r2 = rewards[0]
            r = r1 + r2*gamma
            rewards.appendleft(r)
        return rewards

    def decay_oldest_rewards(self, rewards, gamma):
        decayed = 0.0
        for i in range(len(rewards)):
            r = rewards.pop()
            decayed *= gamma
            decayed += r
            if i == (len(rewards)):
                rewards.appendleft(decayed)
            else:
                rewards.appendleft(r)
        return rewards

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
        res = list(zip(self.tot_rewards, self.tot_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res  



class StepReplayBuffer:
    pass