import gym
import numpy as np
import torch
import ptan
import common
import atari_wrappers
import cv2


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


PATH = "Breakout-v4.pt"
device = "cuda"

net = common.DQN((3, 210,160), env.action_space.n).to(device)
tgt_net = common.TargetNet(net)

'''agent = common.DQNAgent(net, selector = common.EpsilonGreedyActionSelector(), device=device)
exp_source = common.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)'''

preprocessor = common.ndarray_preprocessor(common.VariableTensor_preprocessor())
selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON)
agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
optimizer = torch.optim.Adam(net.parameters())

#last call or during training
torch.save(net.state_dict(), PATH)

#render_function
env2 = gym.make("Breakout-v4", render_mode="rgb_array")
env2 = atari_wrappers.reshapeWrapper(env2)
env2 = atari_wrappers.ScaledFloatFrame(env2)
env2 = atari_wrappers.oldWrapper(env2)
render_env = atari_wrappers.MaxAndSkipEnv(env2)


render_source = ptan.experience.ExperienceSource(render_env, agent)


video = []
for x in render_source:

    video.append(x[0][0])
    if x[0][3]:
        print("done")
        break

video = np.array(video, dtype=np.float32).transpose(0,2,3,1)
print(video.shape)

import cv2
import numpy as np

def create_video(frames, output_filename):
    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for MP4 video
    video = cv2.VideoWriter(output_filename, fourcc, 10.0, (width, height))

    for frame in frames:
        frame = (frame * 255).astype(np.uint8)  # scale values from 0-1 to 0-255
        video.write(frame)

    video.release()

create_video(video, "output.mp4")
