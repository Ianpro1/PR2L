import gym
import numpy as np
import torch
import common.models as models
import ptan
import common.extentions as E
import common.atari_wrappers as atari_wrappers
from torch.utils.tensorboard import SummaryWriter
import time


ENV_NAME = "CartPole-v1"
EPSILON_DECAY_LAST_FRAME = 15000 #150000 default
EPSILON_START = 1.0 
EPSILON_FINAL = 0.01
GAMMA = 0.99
N_STEPS = 4
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
    "REPLAY_SIZE":REPLAY_SIZE,
    "elapsed": 0.0
}

env = gym.make(ENV_NAME)
env = atari_wrappers.oldWrapper(env)


device = "cuda"

obs_shape = env.observation_space
if isinstance(obs_shape, gym.spaces.Discrete):
    obs_shape = 1
    print("shape detected: Discrete")
elif isinstance(obs_shape, gym.spaces.Box):
    obs_shape=np.prod(obs_shape.shape)
    print("shape detected: Box")

action_shape = env.action_space.n

net = models.DenseDQN(obs_shape, 256, action_shape).to(device)
tgt_net = ptan.agent.TargetNet(net)

preprocessor = E.ndarray_preprocessor(E.FloatTensor_preprocessor())
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=N_STEPS)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


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
while True:
    idx += 1
    buffer.populate(1)

    for rewards, steps in exp_source.pop_rewards_steps():
        speed = (idx - ts_frame) / (time.time() - ts)
        ts_frame = idx
        ts = time.time()
        episode+=1
        #print("FPS %.0f" % speed, "reward: %.2f" %rewards )
        print("idx %d, steps %d, episode %d done, reward=%.3f rewards, epsilon=%.4f, FPS %.3f" %(idx,steps,episode, rewards, selector.epsilon, speed))
        writer.add_scalar("episodes", episode, idx)
        writer.add_scalar("rewards", rewards, idx)
        writer.add_scalar("epsilon", selector.epsilon, idx)
        writer.add_scalar("FPS", speed, idx)
        solved = rewards > 1000
    if solved:
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

CARTPOLE:
reward_bound = 1000

idx 55718, steps 1878, episode 486 done, reward=1878.000 rewards, epsilon=0.0100, FPS 262.401
done in 486 episodes, elapsed: 217.72 seconds
epsilon_last_frame: 15000.000

idx 29558, steps 1305, episode 403 done, reward=1305.000 rewards, epsilon=0.0100, FPS 250.172
done in 403 episodes, elapsed: 129.74 seconds
epsilon_last_frame: 15000.000

idx 64738, steps 1311, episode 825 done, reward=1311.000 rewards, epsilon=0.0100, FPS 259.964
done in 825 episodes, elapsed: 278.92 seconds
epsilon_last_frame: 1500.000

idx 74628, steps 7672, episode 642 done, reward=7672.000 rewards, epsilon=0.0100, FPS 253.813
done in 642 episodes, elapsed: 313.54 seconds
epsilon_last_frame: 25000.000

WITH TGT_SYNC_FRAMES: 500
idx 60162, steps 1224, episode 692 done, reward=1224.000 rewards, epsilon=0.0100, FPS 246.252
done in 692 episodes, elapsed: 253.88 seconds
epsilon_last_frame: 25000.000

WITH TGT_SYNC_FRAMES: 2000
epsilon_last_frame: 25000.000
idx is 100000 and agent seem stuck at local optimum
good training increase of rewards but sharp drop at 50k frames


-----N-STEPS -----

3: with 0.97 failed to converge at 60k+ frames, with 0.99 42k -> (1064 score)
4: 30k frames, 30k
5: 50k frames (final score was infinite)
4: with 0.98 -> 20k, 60k+ frames
4, with 0.97 -> 40k, 24k (good stability)
5, with 0.97 -> good stability, slow increase, lost reward at 43k almost won at 30k (~980 score)(closed training at 50k)
BEST SETTINGS:
epsilon_last_frame to 25% of expected frames = 15000 -> reaches less than 30k frames
"""