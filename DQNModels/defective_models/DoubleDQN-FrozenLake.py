import gym
import numpy as np
import torch
import common.models as models
import ptan
import common.extentions as E
import common.atari_wrappers as atari_wrappers
from torch.utils.tensorboard import SummaryWriter
import time
import math


ENV_NAME = "FrozenLake-v1"
EPSILON_DECAY_LAST_FRAME = 15000 #150000 default
EPSILON_START = 1.0 
EPSILON_FINAL = 0.01
GAMMA = 0.99
REPLAY_SIZE = 10000
TGT_NET_SYNC = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


'''def calc_sin_epsilon(x):
    return (2.0*math.cos(x/20.)+3.)/100.'''


parameters = {
    "epsilon":EPSILON_START,
    "complete":False,
    "LEARNING_RATE":LEARNING_RATE,
    "GAMMA":GAMMA,
    "TGT_NET_SYNC":TGT_NET_SYNC,
    "EPSILON_DECAY_LAST_FRAME":EPSILON_DECAY_LAST_FRAME,
    "BATCH_SIZE":BATCH_SIZE,
    "REPLAY_SIZE":REPLAY_SIZE,
    "elapsed": 0.0
}

env = gym.make(ENV_NAME)
env = atari_wrappers.expandWrapper(env)
env = atari_wrappers.oldWrapper(env)


device = "cuda"

obs_shape = env.observation_space
action_shape = env.action_space.n
if isinstance(obs_shape, gym.spaces.Discrete):
    obs_shape = 1
    print("shape detected: Discrete")
elif isinstance(obs_shape, gym.spaces.Box):
    obs_shape=np.prod(obs_shape.shape)
    print("shape detected: Box")


net = models.DenseDQN(obs_shape, 256, action_shape).to(device)
tgt_net = ptan.agent.TargetNet(net)

preprocessor = E.ndarray_preprocessor(E.FloatTensor_preprocessor())
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

if False:
    renv = gym.make(ENV_NAME)
    
    renv = atari_wrappers.RenderWrapper(renv)
    renv = atari_wrappers.SingleLifeWrapper(renv)
    renv = atari_wrappers.FireResetEnv(renv)
    renv = atari_wrappers.MaxAndSkipEnv(renv)
    renv = atari_wrappers.ProcessFrame84(renv)
    renv = atari_wrappers.reshapeWrapper(renv)
    renv = atari_wrappers.ScaledFloatFrame(renv, 148.)
    renv = atari_wrappers.BufferWrapper(renv, 3)
    renv = atari_wrappers.oldStepWrapper(renv)
    
    r_agent = ptan.agent.DQNAgent(net, action_selector=ptan.actions.EpsilonGreedyActionSelector(epsilon=0.1), device=device, preprocessor=preprocessor)
    render_source = ptan.experience.ExperienceSource(renv, r_agent)
    common.playandsave_episode(render_source, renv)
    video = renv.pop_frames()
    print(video.shape)
    common.create_video(video, "output2.mp4")
    raise MemoryError


idx = 0
episode = 0
ts_frame = 0
ts = time.time()
start_time = time.time()
backup = E.ModelBackup(ENV_NAME, net=net, notify=True, Temp_disable=True)
backup.save(parameters=parameters)
#need to save settings and create new model folder to keep old models and new ones

writer = SummaryWriter(comment=ENV_NAME +"_--" + device)
solved = False

from collections import deque
maxlen = 100
rewards_arr = deque(maxlen=maxlen)

while True:
    idx += 1
    buffer.populate(1)

    for rewards, steps in exp_source.pop_rewards_steps():
        speed = (idx - ts_frame) / (time.time() - ts)
        ts_frame = idx
        ts = time.time()
        episode+=1
        rewards_arr.append(rewards)
        print(sum(rewards_arr))
        if len(rewards_arr) >= maxlen-1:
            print("FPS %.0f" % speed, "reward_avg: %.2f" % sum(rewards_arr)/float(maxlen) )

        #print("idx %d, steps %d, episode %d done, reward=%.3f rewards, epsilon=%.4f, FPS %.3f" %(idx,steps,episode, rewards, selector.epsilon, speed))
        writer.add_scalar("episodes", episode, idx)
        writer.add_scalar("rewards", rewards, idx)
        writer.add_scalar("epsilon", selector.epsilon, idx)
        writer.add_scalar("FPS", speed, idx)
        solved = rewards > 1000
    if solved:
        print("idx %d, steps %d, episode %d done, reward=%.3f rewards, epsilon=%.4f, FPS %.3f" %(idx,steps,episode, rewards, selector.epsilon, speed))
        print("done in %d episodes, elapsed: %.2f seconds" % (episode, time.time()-start_time))
        print('epsilon_last_frame: %.3f' %EPSILON_DECAY_LAST_FRAME)
        parameters["epsilon"] = selector.epsilon
        parameters["complete"] = True
        backup.save(parameters=parameters)
        break
    if len(buffer) < 2*BATCH_SIZE:
        continue
    batch = buffer.sample(BATCH_SIZE)
    states, actions, rewards, last_states, dones = E.unpack_batch(batch, obs_shape)
    
    # agent returns best actions for tgt

    last_states = preprocessor(last_states).to(device)
    
    with torch.no_grad(): #try to use numpy instead
        tgt_q = net(last_states)
        tgt_actions = torch.argmax(tgt_q, 1)
        tgt_actions = tgt_actions.unsqueeze(1)
        tgt_qs = tgt_net.target_model(last_states)
        tgt_q_v = tgt_qs.gather(1, tgt_actions).squeeze(1)

    tgt_q_v[dones] = 0.0
    rewards = preprocessor(rewards).to(device)
    q_v_refs = rewards + tgt_q_v * GAMMA
    optimizer.zero_grad()
    states = preprocessor(states).to(device)
    actions = torch.tensor(actions).to(device)
    q_v = net(states)
    q_v = q_v.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = torch.nn.functional.mse_loss(q_v, q_v_refs)
    loss.backward()
    optimizer.step()
            
    writer.add_scalar("loss", loss, idx)
    
    selector.epsilon = max(EPSILON_FINAL, EPSILON_START - idx / EPSILON_DECAY_LAST_FRAME)
    
    if idx % TGT_NET_SYNC ==0:
        tgt_net.sync()
    
    if idx % 50000 == 0:
        parameters["epsilon"] = selector.epsilon
        parameters["elapsed"] = time.time() - start_time
        backup.save(parameters=parameters)
    
    
"""
SCOREBOARD:

FROZENLAKE:

"""