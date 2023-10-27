# PR2L

This is a repository for a small reinforcement learning library I made while reading DEEP REINFORCEMENT LEARNING HANDS-ON by Maxim Lapan (https://github.com/Shmuma). The motivation for this library comes from PTAN (the RL library used in the book) being out of date with the latest features of gym/gymnasium. Therefore, some functions here may also be found in PTAN however most have been replaced since the creation of this library, as most of the core features found in PR2L have since diverged from PTAN's core features. Therefore, expect this library to be similar to PTAN with extra features for rendering environments and aiding training. I've also left a few RL agent scripts in the repository as examples.

Currently, PR2L has a total of 6 modules: training (new), agent, experience, playground, rendering, and utilities. Each module file includes a brief description of its usage.
******
RL SCRIPTS:

### A3C on BattleZone
An Asynchronous Actor Critic agent playing the Atari Battlezone game. The AtariPreprocessing class from openai is used to process the agent's observation space before it is played. Hence, the agent's observation space is reduced from the typical 210 by 160 RGB pixel ATARI observation to a more traditional 84 by 84 grey pixel format along with other changes. To simulate motion, the current frame was combined with n prior frames, each of which has had its values decremented by an arbitrary amount.

<img src="https://github.com/Ianpro1/RL-agents/blob/master/GIF/BattleZone.gif" width="400">

******
### RainbowDQN-Variant on Freeway
This is a variant of the Rainbow DQN architecture playing Freeway
PREPROCESSING: The agent uses the same preprocessing as the above model.

<img src="https://github.com/Ianpro1/RL-agents/blob/master/GIF/Freeway.gif" width="400">

### DDPG on Tosser
A Deep Deterministic Policy Gradient agent playing Tosser: a MuJoCo task/environment by Openai.
<img src="https://github.com/Ianpro1/PR2L/blob/master/GIF/TosserCPPGIF.gif" width="600">

#### INSTALLATION
>git clone https://github.com/Ianpro1/PR2L
>
>pip install ./PR2L

_Versions and Dependencies (NOT INCLUDED IN PACKAGE): PyTorch Cuda 11.6, Python 3.10.8, gym 0.26.2, Pygame 2.1.2 and gymnasium 0.27.1_ (gymnasium is recommanded over gym)
<sup>*Most dependencies will work at different versions with the exception of gym/gymnasium (the library handles truncated flags [Not Yet], therefore older versions are not supported.)</sup>
