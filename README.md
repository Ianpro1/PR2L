# PR2L (python/pytorch reinforcement learning library)
This is a repository for a Reinforcement Learning Library called PR2L

### A3C on BattleZone
This is an Asynchronus Actor Critic agent playing Atari Battlezone. PREPROCESSING: The agent uses openai's AtariPreprocessing class, thus, the agent's observation space is wrapped from the standard ATARI observation of 210 by 160 RGB pixels to the conventional 84 by 84 gray pixels. To display motion, the image was then summed with n previous frames, decayed to an arbitruary factor.

<img src="https://github.com/Ianpro1/RL-agents/blob/master/GIF/BattleZone.gif" width="400">

### RainbowDQN-variant on Freeway
This is a variant of the Rainbow DQN architecture playing Freeway
PREPROCESSING: The agent uses the same preprocessing as the above model.

<img src="https://github.com/Ianpro1/RL-agents/blob/master/GIF/Freeway.gif" width="400">

