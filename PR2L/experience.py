#experience contains a variety of classes to gather and store the PR2L.agent.Agent experiences
import gym
import gymnasium
from collections import namedtuple, deque
from .agent import Agent
import numpy as np
import math
Experience = namedtuple("Experience", ("state", "action", "reward", "next"))

MemorizedExperience = namedtuple("MemorizedExperience", ("state", "action", "reward", "next", "memory")) 

class SimpleReplayBuffer:
    def __init__(self, exp_source, replay_size):
        assert isinstance(replay_size, int)
        self.capacity = replay_size
        if exp_source is not None:
            self.exp_source_iter = iter(exp_source)
            self.exp_source = exp_source
        else:
            self.exp_source_iter = None
            self.exp_source = None
        
        self.buffer = []
        self.pos = 0

    def _add(self, exp):
    #should only add one sample per call
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp

        self.pos = (self.pos + 1) % self.capacity
    
    def populate(self, n_samples):
        for _ in range(n_samples):
            entry = next(self.exp_source_iter)
            self._add(entry)
    
    def populate_episodes(self, n_episodes):
        for exp in self.exp_source:
            self._add(exp)
            if self.exp_source.n_eps_done >= n_episodes:
                self.exp_source.n_eps_done = 0
                break

    def sample(self, n_samples):
        if len(self.buffer) <= n_samples:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), n_samples, replace=True)
        return [self.buffer[key] for key in keys]

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

#this addition can be made inside every agent class as internal states
#TODO: add _internal_states
#change Memorized ExperienceSource to use DequeSource but compare performence between them!
class MemorizedExperienceSource:
    """
    ExperienceSource with additional custom (User Defined) object stored in each Experience i.e. MemorizedExperience
    """
    def __init__(self, env, agent, n_steps=2, GAMMA=0.99):
        assert isinstance(agent, Agent)
        assert isinstance(env, (gym.Env,gymnasium.Env, list, tuple))
                
        self.n_steps = n_steps

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
        self.gamma = GAMMA
        self.n_eps_done = 0

    def __iter__(self):
        _states = []
        _rewards = []
        _actions = []
        _memories = []
        cur_obs = []
        
        for e in self.env:
            _states.append(deque(maxlen=self.n_steps))
            _rewards.append(deque(maxlen=self.n_steps))
            _actions.append(deque(maxlen=self.n_steps))
            _memories.append(deque(maxlen=self.n_steps))
            obs, _ = e.reset()
            cur_obs.append(obs)
        
        while True:   
            actions, memories = self.agent(cur_obs)
            
            for i, env in enumerate(self.env):
                nextobs, reward, done, _, _ = env.step(actions[i])

                _memories[i].append(memories[i])
                _actions[i].append(actions[i])
                _rewards[i].append(reward)
                _states[i].append(cur_obs[i])
                self.__sum_rewards_steps(reward, done, i)
                if done:
                    print('reset from exp')
                    #decay all
                    self.__decay_all_rewards(_rewards[i])

                    for _ in range(len(_rewards[i])): #used to get proper length
                        exp = MemorizedExperience(_states[i].popleft(), _actions[i].popleft(), _rewards[i].popleft(), None, _memories[i].popleft())
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

                    exp = MemorizedExperience(_states[i].popleft(), _actions[i].popleft(), _rewards[i].popleft(), nextobs, _memories[i].popleft())
                    yield exp
   
    def __decay_all_rewards(self, rewards):
        prev = 0.0
        for i in reversed(range(len(rewards))):
            rewards[i] += prev
            prev = rewards[i] * self.gamma

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
        res = list(zip(self.tot_rewards, self.tot_steps))
        if res:
            self.tot_rewards.clear()
            self.tot_steps.clear()
        return res

#TODO HeldExperienceSource (holds terminated environments until all are finished)
#TODO EpisodeSource (stores episode instead of experiences)
#TODO SyncExperienceSource (does not have __decay_all_rewards / yields only 1 exp per iteration over envs)

class DequeSource:
    """
    Parent class for all source class.

    Note for devs: This class allows for decay_all_rewards (used when an environment is terminated),
    decay_oldest_reward (used when the deque is full of experience), sum_rewards_steps (should be called
      upon every reset and check that track_rewards is True because this is not implemented internally),
      pop_rewards_steps (returns: rewards and steps as zipped list)
    """
    def __init__(self, ENV_NUM, GAMMA=0.99, track_rewards=True):
        self.n_envs = ENV_NUM
        if track_rewards:
            self.tot_reward = [0.]*ENV_NUM
            self.tot_rewards = []
            self.tot_step = [0.]*ENV_NUM
            self.tot_steps = []
        self.track_rewards = track_rewards
        self.gamma = GAMMA
        self.n_eps_done = 0

    def __iter__(self):
        raise NotImplementedError
    
    def decay_all_rewards(self, rewards):
        prev = 0.0
        res = []
        for r in reversed(rewards):
            r += prev
            res.append(r)
            prev = r * self.gamma
        return res

    def decay_oldest_reward(self, rewards):
        tot = 0.0
        for r in reversed(rewards):
             tot = r + tot* self.gamma
        rewards[0] = tot

    def sum_rewards_steps(self, reward, done, env_id):
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

class ExperienceSource(DequeSource):
    """
    A class that yields experiences storing states, actions, rewards and "next" (the state that follows the action).
    Rewards are each decayed like so: Q = r0 + r(1) * GAMMA^(1) + ... + r(n) * GAMMA^(n).
    Actions are sampled using the agent that was passed as argument.

    NOTE: Its a common practice in reinforcement learning to individually sample the agent's actions for each environment.
    This class achieves better perfomance by sampling all actions at once for every environment. Hence, when an environment is
    terminated before the others, the class will yield all the experiences that were stored for this environment. The user should
    therefore expect every environment to be out of sync.
    """
    def __init__(self, env, agent, n_steps=2, GAMMA=0.99, track_rewards=True):
        assert isinstance(agent, Agent)
        assert isinstance(env, (gym.Env,gymnasium.Env, list, tuple))
        
        self.n_steps = n_steps

        if isinstance(env, (list, tuple)):
            self.env = env
            ENV_NUM = len(env)
        else: 
            self.env = [env]
            ENV_NUM = 1
        self.agent = agent   
        super().__init__(ENV_NUM, GAMMA, track_rewards)

    def __iter__(self):
        _states = []
        _rewards = []
        _actions = []
        _internal_states = []
        cur_obs = []
        for e in self.env:
            _states.append(deque(maxlen=self.n_steps))
            _rewards.append(deque(maxlen=self.n_steps)) #lists are slower than deque (about 20% slower)
            _actions.append(deque(maxlen=self.n_steps))
            obs, _ = e.reset()
            cur_obs.append(obs)
            _internal_states.append(self.agent.initial_state())

        while True:   
            actions, _internal_states = self.agent(cur_obs, _internal_states)
            for i, env in enumerate(self.env):
                nextobs, reward, done, _, _ = env.step(actions[i])
                _actions[i].append(actions[i])
                _rewards[i].append(reward)
                _states[i].append(cur_obs[i])
                if self.track_rewards:
                    self.sum_rewards_steps(reward, done, i)
                if done:
                    #decay all
                    decayed = self.decay_all_rewards(_rewards[i])
                    _rewards[i].clear()
                    for r_id in range(1, len(decayed)+1):
                        exp = Experience(_states[i].popleft(), _actions[i].popleft(), decayed[-r_id], None)
                        yield exp
                    self.n_eps_done += 1
                    obs, _ = env.reset()
                    cur_obs[i] = obs
                    _internal_states[i] = None
                    continue
                
                cur_obs[i] = nextobs

                if len(_actions[i]) == self.n_steps:
                    #decay oldest
                    self.decay_oldest_reward(_rewards[i])
                    exp = Experience(_states[i].popleft(), _actions[i].popleft(), _rewards[i].popleft(), nextobs)
                    yield exp 

class ScarsedExperienceSource(DequeSource):
    """
    (See ExperienceSource for details about algorithm) 
    ScarsedExperienceSource performs random steps per environments before yielding.
    """
    def __init__(self, max_steps, env, agent, n_steps=2, GAMMA=0.99, track_rewards=True):
        assert isinstance(agent, Agent)
        assert isinstance(env, (gym.Env,gymnasium.Env, list, tuple))
        assert isinstance(max_steps, int)
        self.n_steps = n_steps
        self.max_steps = max_steps
        if isinstance(env, (list, tuple)):
            self.env = env
            ENV_NUM = len(env)
        else: 
            self.env = [env]
            ENV_NUM = 1
        self.agent = agent   
        super().__init__(ENV_NUM, GAMMA, track_rewards)

    def __iter__(self):
        _states = []
        _rewards = []
        _actions = []
        _internal_states = []
        cur_obs = []
        for e in self.env:
            _states.append(deque(maxlen=self.n_steps))
            _rewards.append(deque(maxlen=self.n_steps)) #lists are slower than deque (about 20% slower)
            _actions.append(deque(maxlen=self.n_steps))
            obs, _ = e.reset()
            cur_obs.append(obs)
            _internal_states.append(self.agent.initial_state())

        #generate steps
        eval_steps = np.random.randint(low=0, high=self.max_steps, size=len(self.env))
        #perform steps
        eval_states = {}
        beval = True
        while beval:
            actions, _internal_states = self.agent(cur_obs, _internal_states)
            for i, env in enumerate(self.env):
                if len(eval_states) == len(self.env):
                    beval = False
                    break
                if eval_steps[i] == -1:
                    continue
                elif eval_steps[i] <= 0:
                    eval_steps[i] = -1
                    eval_states[i] = _internal_states[i]
                    continue

                nextobs, reward, done, _, _ = env.step(actions[i])
                eval_steps[i] -= 1
                _actions[i].append(actions[i])
                _rewards[i].append(reward)
                _states[i].append(cur_obs[i])
                if done:
                    _rewards[i].clear()
                    obs, _ = env.reset()
                    cur_obs[i] = obs
                    _internal_states[i] = None
                cur_obs[i] = nextobs

        #reinstall lost internal_states
        for key,value in eval_states.items():
            _internal_states[key] = value

        #delete temp. variables
        del beval
        del eval_states
        del eval_steps

        #yield
        while True:   
            actions, _internal_states = self.agent(cur_obs, _internal_states)
            for i, env in enumerate(self.env):
                nextobs, reward, done, _, _ = env.step(actions[i])
                _actions[i].append(actions[i])
                _rewards[i].append(reward)
                _states[i].append(cur_obs[i])
                if self.track_rewards:
                    self.sum_rewards_steps(reward, done, i)
                if done:
                    #decay all
                    decayed = self.decay_all_rewards(_rewards[i])
                    _rewards[i].clear()
                    for r_id in range(1, len(decayed)+1):
                        exp = Experience(_states[i].popleft(), _actions[i].popleft(), decayed[-r_id], None)
                        yield exp
                    self.n_eps_done += 1
                    obs, _ = env.reset()
                    cur_obs[i] = obs
                    _internal_states[i] = None
                    continue
                
                cur_obs[i] = nextobs

                if len(_actions[i]) == self.n_steps:
                    #decay oldest
                    self.decay_oldest_reward(_rewards[i])
                    exp = Experience(_states[i].popleft(), _actions[i].popleft(), _rewards[i].popleft(), nextobs)
                    yield exp 


class SimpleDecayBuffer(DequeSource):
    """
    DecayBuffer stores experiences that are pushed to it and decays them to n_steps. 
    This flexible approach allows users to implement their own agent's step-through-environments loop,
    without thinking about storing n_steps experiences.
    *ideal for off-policy networks
    """

    def __init__(self, n_envs : int, n_steps=2, GAMMA =0.99 , track_rewards =True):
        self.n_steps = n_steps
        ENV_NUM = n_envs 
        super().__init__(ENV_NUM, GAMMA, track_rewards)
        self._states = []
        self._rewards = []
        self._actions = []
        self._cur_obs = [None] * ENV_NUM
        
        self.buffer = []

        for _ in range(self.n_envs):
            self._states.append(deque(maxlen=self.n_steps))
            self._rewards.append(deque(maxlen=self.n_steps))
            self._actions.append(deque(maxlen=self.n_steps))

    def _add(self, env_id, obs, act=None, rew=None, done=None) -> None:
        if self._cur_obs[env_id] is None:
            self._cur_obs[env_id] = obs
            return None
        assert act is not None
        assert rew is not None
        assert done is not None

        self._actions[env_id].append(act)
        self._rewards[env_id].append(rew)
        self._states[env_id].append(self._cur_obs[env_id])
        if self.track_rewards:
            self.sum_rewards_steps(rew, done, env_id)
        if done:
            #decay all
            decayed = self.decay_all_rewards(self._rewards[env_id])
            self._rewards[env_id].clear()
            for r_id in range(1, len(decayed)+1):
                exp = Experience(self._states[env_id].popleft(), self._actions[env_id].popleft(), decayed[-r_id], None)
                self.buffer.append(exp)
            self.n_eps_done += 1

            #need to behave differently on next _add()
            self._cur_obs[env_id] = None

        else:
            self._cur_obs[env_id] = obs

            if len(self._actions[env_id]) == self.n_steps:
                #decay oldest
                self.decay_oldest_reward(self._rewards[env_id])
                exp = Experience(self._states[env_id].popleft(), self._actions[env_id].popleft(), self._rewards[env_id].popleft(), obs)
                self.buffer.append(exp)
        return None
        
    def __len__(self) -> int:
        return len(self.buffer)

    def _extend(self, env_ids, observations, actions=None, rewards=None, dones=None) -> None:
        for env_id, obs, act, rew, done in list(zip(env_ids, observations, actions, rewards, dones)):
            self._add(env_id, obs, act, rew, done)
        return None

    def sample(self, n_samples, retain_samples=True) -> list:
        if len(self.buffer) <= n_samples:
            samples = self.buffer[:]
            if (retain_samples != True):
                self.buffer.clear()
            return samples
        keys = np.random.choice(len(self.buffer), n_samples, replace=True)
        samples = [self.buffer[key] for key in keys]
        if (retain_samples != True):
            for key in sorted(keys, reverse=True):
                del self.buffer[key]
        return samples

    def pop_left(self):
        return self.buffer.pop(0)
    
    def releaseBuffer(self) -> list:
        samples = self.buffer[:]
        self.buffer.clear()
        return samples


class DequeDecayBuffer(DequeSource):
    """
    DequeDecayBuffer works the same as SimpleDecayBuffer however it does not store the experiences and instead of using a list, it uses a deque.
    It should be used after _add as an iterator object to pop all experiences from the small buffer : deque(maxlen=n_steps).
    *ideal for on-policy networks

    """

    def __init__(self, n_envs : int, n_steps=2, GAMMA =0.99 , track_rewards =True, bufferLength = -1):
        self.n_steps = n_steps
        ENV_NUM = n_envs 
        super().__init__(ENV_NUM, GAMMA, track_rewards)
        self._states = []
        self._rewards = []
        self._actions = []
        self._cur_obs = [None] * ENV_NUM
        
        if (bufferLength == -1):
            self.buffer = deque(maxlen=n_steps)
        else:
            assert bufferLength > 0
            self.buffer = deque(maxlen=bufferLength)

        for _ in range(self.n_envs):
            self._states.append(deque(maxlen=self.n_steps))
            self._rewards.append(deque(maxlen=self.n_steps))
            self._actions.append(deque(maxlen=self.n_steps))

    def _add(self, env_id, obs, act=None, rew=None, done=None) -> None:
        if self._cur_obs[env_id] is None:
            self._cur_obs[env_id] = obs
            return None
        assert act is not None
        assert rew is not None
        assert done is not None

        self._actions[env_id].append(act)
        self._rewards[env_id].append(rew)
        self._states[env_id].append(self._cur_obs[env_id])
        if self.track_rewards:
            self.sum_rewards_steps(rew, done, env_id)
        if done:
            #decay all
            decayed = self.decay_all_rewards(self._rewards[env_id])
            self._rewards[env_id].clear()
            for r_id in range(1, len(decayed)+1):
                exp = Experience(self._states[env_id].popleft(), self._actions[env_id].popleft(), decayed[-r_id], None)
                self.buffer.append(exp)
            self.n_eps_done += 1

            #need to behave differently on next _add()
            self._cur_obs[env_id] = None

        else:
            self._cur_obs[env_id] = obs

            if len(self._actions[env_id]) == self.n_steps:
                #decay oldest
                self.decay_oldest_reward(self._rewards[env_id])
                exp = Experience(self._states[env_id].popleft(), self._actions[env_id].popleft(), self._rewards[env_id].popleft(), obs)
                self.buffer.append(exp)
        return None

    def __len__(self) -> int:
        return len(self.buffer)
    

    def __iter__(self):
        while(1):
            yield self.buffer.popleft
    