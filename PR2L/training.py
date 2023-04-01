#this is a new module that will include utilities for training agents

import torch
import numpy as np
from PR2L import utilities, agent


#for Actor critic based Models
def unpack_batch_A2C(batch, net, GAMMA=0.99, N_STEPS=2, device="cpu", value_index_pos=1):
    """
    Used to unpack batches of regular experiences (experience.Experience) and return the states, actions and Q_values.
    
    More specifically, this class is for Actor critic based Models
    """
    states, actions, rewards, last_states, not_dones = utilities.unpack_batch(batch)

    states = agent.float32_preprocessing(states).to(device)
    rewards = agent.float32_preprocessing(rewards).to(device)
    actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
    if last_states:
        last_states = agent.float32_preprocessing(last_states).to(device)
        tgt_q = torch.zeros_like(rewards)
        with torch.no_grad():
            next_q_v = net(last_states)[value_index_pos]

        tgt_q[not_dones] = next_q_v.squeeze(-1)
        refs_q_v = rewards + tgt_q * (GAMMA**N_STEPS)
    else:
        refs_q_v = rewards
    return states, actions, refs_q_v


def unpack_batch_DQN_tgt(batch, tgt_net, GAMMA=0.99, N_STEPS=2, device="cpu"):
    """
    Used to unpack batches of regular experiences (experience.Experience) and return the states, actions and Q_values.

    More specifically, this class is for Simple DQN models.
    - using target net for argmax step (Dev - Never tested, there might be unsqueezing to do)
    """
    states, actions, rewards, last_states, not_dones = utilities.unpack_batch(batch)

    states = agent.float32_preprocessing(states).to(device)
    rewards = agent.float32_preprocessing(rewards).to(device)
    actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
    if last_states:
        tgt_q_v = torch.zeros_like(rewards)
        last_states = agent.float32_preprocessing(last_states).to(device)
        with torch.no_grad():
            tgt_q = tgt_net.target_model(last_states)
            tgt_q = tgt_q.max(dim=1)[0]
            tgt_q_v[not_dones] = tgt_q

        refs_q_v = rewards + tgt_q_v * (GAMMA**N_STEPS)
    else:
        refs_q_v = rewards
    return states, actions, refs_q_v
    

def unpack_batch_DQN_double(batch, net, tgt_net, GAMMA=0.99, N_STEPS=2, device="cpu"):
    """
    Used to unpack batches of regular experiences (experience.Experience) and return the states, actions and Q_values.

    More specifically, this class if for Double DQN models 
    - using main network for argmax step (Dev - Never tested, there might be unsqueezing to do)
    """
    states, actions, rewards, last_states, not_dones = utilities.unpack_batch(batch)

    states = agent.float32_preprocessing(states).to(device)
    rewards = agent.float32_preprocessing(rewards).to(device)
    actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
    if last_states:
        tgt_q_v = torch.zeros_like(rewards)
        last_states = agent.float32_preprocessing(last_states).to(device)
        with torch.no_grad():
            tgt_q = net(last_states)
            argmax = tgt_q.argmax(dim=1)[0]
            tgt_q = tgt_net.target_model(last_states)
            tgt_q = tgt_q[argmax.unsqueeze(-1)]
            tgt_q_v[not_dones] = tgt_q

        refs_q_v = rewards + tgt_q_v * (GAMMA**N_STEPS)
    else:
        refs_q_v = rewards
    return states, actions, refs_q_v