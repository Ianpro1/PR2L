from PR2L import playground
from collections import namedtuple, deque

Experience = namedtuple("Experience", ("state", "action", "reward", "next"))


def decay_all(arr, idd):
    prev = 0.
    for i in reversed(range(idd+1)):
        arr[i] = arr[i] + prev*GAMMA
        prev = arr[i]
        
 
def decay_oldest(arr):
    tot = 0
    for i, r in enumerate(arr):
        tot += r * GAMMA**i
    arr[:-1] = arr[1:]
    return tot

#initialization of needed variables and objects
envs = [playground.Dummy((1,), done_func=playground.EpisodeLength(10)) for _ in range(4)]
_len = len(envs)
n_steps = 4
GAMMA = 0.99
states = []
actions = []
rewards = []
cur_obs = []
for e in envs:
    obs, info = e.reset()
    states.append(deque(maxlen=n_steps))
    actions.append(deque(maxlen=n_steps))
    cur_obs.append(obs)
    rewards.append([0.]*n_steps)
idd = [0]*_len
for i in range(10):
    acts = [1,2,3,1] #use obs
    for i, env in enumerate(envs):
        nextobs, rew, done, trunc, info = env.step(acts[i])

        states[i].append(cur_obs[i])
        rewards[i][idd] = rew
        actions[i].append(acts[i])

        if done:
            
            idd[i] = 0
            pass
        
        if idd[i] == n_steps-1:
            pass
        

        #could be removed assuming all episodes are always over n_steps
        if idd[i] < n_steps-1:
            idd[i] += 1