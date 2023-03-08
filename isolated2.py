import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import gym

env = gym.make("TennisNoFrameskip-v4")
print(env.unwrapped.get_action_meanings())