import gym
import numpy as np
import torch
#import ptan_copy as ptan
import ptan
import common
import atari_wrappers
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from collections import namedtuple



ENV_NAME = "Breakout-v4"
EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
GAMMA = 0.99
REPLAY_SIZE = 10000
TGT_NET_SYNC = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

parameters = {
    "epsilon":EPSILON_START,
    "complete":False,
    "LEARNING_RATE":LEARNING_RATE,
    "GAMMA":GAMMA,
    "TGT_NET_SYNC":TGT_NET_SYNC,
    "EPSILON_DECAY_LAST_FRAME":EPSILON_DECAY_LAST_FRAME,
    "BATCH_SIZE":BATCH_SIZE,
    "REPLAY_SIZE":REPLAY_SIZE
}

env = gym.make(ENV_NAME)
env = atari_wrappers.ProcessFrame84(env)
env = atari_wrappers.reshapeWrapper(env) # ->prefered over reshape argument in ProcessFrame84
env = atari_wrappers.ScaledFloatFrame(env)
env = atari_wrappers.oldWrapper(env)
env = atari_wrappers.MaxAndSkipEnv(env)

PATH = "Breakout-v4.pt"
device = "cuda"
obs_shape = env.observation_space.shape


net = common.DQN(obs_shape, env.action_space.n).to(device)
print(net)
tgt_net = ptan.agent.TargetNet(net)

preprocessor = common.ndarray_preprocessor(common.FloatTensor_preprocessor())
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


if True:
    renv = gym.make(ENV_NAME)
    renv = atari_wrappers.RenderWrapper(renv)
    renv = atari_wrappers.ProcessFrame84(renv)
    renv = atari_wrappers.reshapeWrapper(renv) # ->prefered over reshape argument in ProcessFrame84
    renv = atari_wrappers.ScaledFloatFrame(renv)
    renv = atari_wrappers.oldWrapper(renv)
    renv = atari_wrappers.MaxAndSkipEnv(renv)
    net.load_state_dict(torch.load("Breakout-v4_firstrun.pt"))
    r_agent = ptan.agent.DQNAgent(net, action_selector=ptan.actions.EpsilonGreedyActionSelector(epsilon=0.1), device=device, preprocessor=preprocessor)
    render_source = ptan.experience.ExperienceSource(renv, r_agent)
    common.playandsave_episode(render_source, renv)
    video = renv.pop_frames()
    print(video.shape)
    common.create_video(video, "output.mp4")
    raise MemoryError

def unpack_batch(batch, obs_shape): # return states, actions, calculated tgt_q_v = r + tgt_net(last_state)*GAMMA
    states = []
    rewards = []
    actions = []
    last_states = []
    dones = []
    for exp in batch: # make it in 2 for loops for if statement
        states.append(exp.state)
        rewards.append(exp.reward)
        actions.append(exp.action)
        
        if exp.last_state is not None:
            dones.append(False)
            last_states.append(exp.last_state) 
        else:
            dones.append(True)
            last_states.append(np.empty(shape=obs_shape)) #might be suboptimal
   
    return states, actions, rewards, last_states, dones


idx = 0
episode = 0
ts_frame = 0
ts = time.time()

backup = common.ModelBackup(ENV_NAME, net=net, notify=True)
backup.save(parameters=parameters)
#need to save settings and create new model folder to keep old models and new ones

writer = SummaryWriter(comment=ENV_NAME +"_--" + device)

while True:
    idx += 1
    buffer.populate(1)

    for rewards, steps in exp_source.pop_rewards_steps():
        speed = (idx - ts_frame) / (time.time() - ts)
        ts_frame = idx
        ts = time.time()
        episode+=1
        #print("idx %d, steps %d, episode %d done, reward=%.3f rewards, epsilon=%.2f, FPS %.3f" %(idx,steps,episode, rewards, selector.epsilon, speed))
        writer.add_scalar("episodes", episode, idx)
        writer.add_scalar("rewards", rewards, idx)
        writer.add_scalar("epsilon", selector.epsilon, idx)
        writer.add_scalar("FPS", speed, idx)
        solved = rewards > 150
        if solved:
            print("done in %d episodes" % episode)
            parameters["epsilon"] = selector.epsilon
            parameters["complete"] = True
            backup.save(parameters=parameters)
            break
    if len(buffer) < 2*BATCH_SIZE:
        continue
    batch = buffer.sample(BATCH_SIZE)
    states, actions, rewards, last_states, dones = unpack_batch(batch, obs_shape)
    
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

    selector.epsilon = max(EPSILON_FINAL, EPSILON_START - idx/EPSILON_DECAY_LAST_FRAME)

    if idx % TGT_NET_SYNC ==0:
        tgt_net.sync()
    
    if idx % 10000 == 0:
        parameters["epsilon"] = selector.epsilon
        backup.save(parameters=parameters)
    
    


'''
#last call or during training
torch.save(net.state_dict(), PATH)

#render_function
render_source = ptan.experience.ExperienceSource(env, agent)
print(type(render_source) == ptan.experience.ExperienceSource)
video = common.playandsave_episode(render_source=render_source)
common.create_video(video, "output.mp4")'''

