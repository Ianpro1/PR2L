import gym
import numpy as np
import torch
#import ptan_copy as ptan
import ptan
import common
import atari_wrappers
from torch.utils.tensorboard import SummaryWriter


EPSILON = 1.0
GAMMA = 0.99
REPLAY_SIZE = 2000
TGT_NET_SYNC = 10
EPS_DECAY = 0.96
BATCH_SIZE = 800

env = gym.make("Breakout-v4")
env = atari_wrappers.reshapeWrapper(env)
env = atari_wrappers.ScaledFloatFrame(env)
env = atari_wrappers.oldWrapper(env)
env = atari_wrappers.MaxAndSkipEnv(env)
env = atari_wrappers.DumbRewardWrapper(env)

PATH = "Breakout-v4.pt"
device = "cpu"

net = common.DQN((3, 210,160), env.action_space.n).to(device)
tgt_net = ptan.agent.TargetNet(net)

'''agent = common.DQNAgent(net, selector = common.EpsilonGreedyActionSelector(), device=device)
exp_source = common.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)'''

preprocessor = common.ndarray_preprocessor(common.VariableTensor_preprocessor())
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON)
agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
optimizer = torch.optim.Adam(net.parameters())


for i, exp in enumerate(exp_source):
    exp = list(exp)
    exp[0]=[]
    exp[-1] = []
    print(exp, "idx-> %d" %i)

    if i > 5:
        break

'''
#last call or during training
torch.save(net.state_dict(), PATH)

#render_function
render_source = ptan.experience.ExperienceSource(env, agent)
print(type(render_source) == ptan.experience.ExperienceSource)
video = common.playandsave_episode(render_source=render_source)
common.create_video(video, "output.mp4")'''

