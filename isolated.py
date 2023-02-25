import gym
from gym.wrappers import atari_preprocessing



env = gym.make("BreakoutNoFrameskip-v4")

env = atari_preprocessing.AtariPreprocessing(env)

obs = env.reset()
print(obs[0].shape)
