from PR2L import agent, experience, utilities
from common import models, playground
import numpy as np
import time

device = "cpu"

def observation(x):
    x = np.random.randint(-255, 255, x.shape).astype(np.float32)
    return x

env = playground.DummyEnv((1,), observation_func=observation, done_func=playground.EpisodeLength(4))
net = models.DenseDQN(1,2,4)
agent = agent.BasicAgent(net, device)
exp_source = experience.ExperienceSourceV2(env, agent,n_steps=1)
buffer = experience.SimpleReplayBuffer(exp_source, 100)

for i, exp in enumerate(exp_source):
    print(exp)
    buffer._add(exp)
    if i > 10000:
        break

print("--------testing_phase---------")
for x in range(10000):
    batch = buffer.sample(1)
    if agent([batch[0].state]).item() != batch[0].action.item():
        print("Action mismatch!")
    