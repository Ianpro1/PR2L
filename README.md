# PR2L (python/pytorch reinforcement learning library)
This is a repository for a Reinforcement Learning Library called PR2L (python/pytorch reinforcement learning library). Currently, PR2L has a total of 5 modules: agent, experience, playground, rendering, and utilities. Each module file includes a brief description of its usage. As this library is still in early development, there is limited documentation available. Therefore, if you are interested in trying out PR2L, I recommend taking a look at the source code and the comments provided.

### A3C on BattleZone
An Asynchronous Actor Critic agent is playing the Atari Battlezone game. The AtariPreprocessing class from openai is used to process the agent's observation space before it is played. This causes the agent's observation space to change from the typical 210 by 160 RGB pixel ATARI observation to a more traditional 84 by 84 grey pixel format along with other changes. The current frame is combined with n prior frames, each of which has had its values decremented by an arbitrary amount, to simulate motion.

<img src="https://github.com/Ianpro1/RL-agents/blob/master/GIF/BattleZone.gif" width="400">

### RainbowDQN-variant on Freeway
This is a variant of the Rainbow DQN architecture playing Freeway
PREPROCESSING: The agent uses the same preprocessing as the above model.

<img src="https://github.com/Ianpro1/RL-agents/blob/master/GIF/Freeway.gif" width="400">

