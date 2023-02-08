import gym
import numpy as np
import torch
#import ptan_copy as ptan
import ptan
import common
import atari_wrappers
from torch.utils.tensorboard import SummaryWriter
import time


EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
GAMMA = 0.99
REPLAY_SIZE = 10000
TGT_NET_SYNC = 1000
BATCH_SIZE = 32
LR = 1e-4
env = gym.make("Breakout-v4")
env = atari_wrappers.reshapeWrapper(env)
env = atari_wrappers.ScaledFloatFrame(env)
env = atari_wrappers.oldWrapper(env)
env = atari_wrappers.MaxAndSkipEnv(env)


PATH = "Breakout-v4.pt"
device = "cpu"
obs_shape = (3,210,160)
net = common.DQN(obs_shape, env.action_space.n).to(device)
print(net)
tgt_net = ptan.agent.TargetNet(net)

'''agent = common.DQNAgent(net, selector = common.EpsilonGreedyActionSelector(), device=device)
exp_source = common.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)'''

preprocessor = common.ndarray_preprocessor(common.FloatTensor_preprocessor())
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)



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


while True:
    idx += 1
    

    buffer.populate(1)

    for rewards, steps in exp_source.pop_rewards_steps():
        speed = (idx - ts_frame) / (time.time() - ts)
        ts_frame = idx
        ts = time.time()
        episode+=1
        print("idx %d, steps %d, episode %d done, reward=%.3f rewards, epsilon=%.2f, FPS %.3f" %(idx,steps,episode, rewards, selector.epsilon, speed))
        solved = rewards > 150
        if solved:
            print("done in %d episodes" % episode)
            torch.save(net.state_dict(), PATH)
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
    
    if idx % 1000 == 0:
        print("model_saved")
        torch.save(net.state_dict(), PATH)
    
    


'''
#last call or during training
torch.save(net.state_dict(), PATH)

#render_function
render_source = ptan.experience.ExperienceSource(env, agent)
print(type(render_source) == ptan.experience.ExperienceSource)
video = common.playandsave_episode(render_source=render_source)
common.create_video(video, "output.mp4")'''

