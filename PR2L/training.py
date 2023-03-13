#this is a new module that will include utilities for training agents

import torch
import numpy as np
from PR2L import utilities, agent


#for Actor critic based Models
def unpack_batch_A2C(batch, net, GAMMA=0.99, N_STEPS=2, device="cpu", value_index_pos=1):
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



    