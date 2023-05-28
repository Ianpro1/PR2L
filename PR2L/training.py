#this is a new module that will include utilities for training agents

import torch
import numpy as np
from PR2L import utilities, agent


#for Actor critic based Models
def unpack_batch_A2C(batch, net, GAMMA=0.99, N_STEPS=2, device="cpu", value_index_pos=1):
    """
    Used to unpack batches of regular experiences (experience.Experience) and return the states, actions and Q_values.
    
    Returns the following: states, actions, refs_q_v

    More specifically, this class is for Actor critic based Models
    """
    states, actions, rewards, next_states, not_dones = utilities.unpack_batch(batch)

    states = agent.float32_preprocessing(states).to(device)
    rewards = agent.float32_preprocessing(rewards).to(device)
    actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
    if next_states:
        next_states = agent.float32_preprocessing(next_states).to(device)
        tgt_q = torch.zeros_like(rewards)
        with torch.no_grad():
            next_q_v = net(next_states)[value_index_pos]

        tgt_q[not_dones] = next_q_v.squeeze(-1)
        refs_q_v = rewards + tgt_q * (GAMMA**N_STEPS)
    else:
        refs_q_v = rewards
    return states, actions, refs_q_v


def unpack_batch_DQN_tgt(batch, tgt_net, GAMMA=0.99, N_STEPS=2, device="cpu"):
    """
    Used to unpack batches of regular experiences (experience.Experience) and return the states, actions and Q_values.

    Returns the following: states, actions, refs_q_v

    More specifically, this class is for Simple DQN models.
    - using target net for argmax step (Dev - Never tested, there might be unsqueezing to do)
    """
    states, actions, rewards, next_states, not_dones = utilities.unpack_batch(batch)

    states = agent.float32_preprocessing(states).to(device)
    rewards = agent.float32_preprocessing(rewards).to(device)
    actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
    if next_states:
        tgt_q_v = torch.zeros_like(rewards)
        next_states = agent.float32_preprocessing(next_states).to(device)
        with torch.no_grad():
            tgt_q = tgt_net.target_model(next_states)
            tgt_q = tgt_q.max(dim=1)[0]
            tgt_q_v[not_dones] = tgt_q

        refs_q_v = rewards + tgt_q_v * (GAMMA**N_STEPS)
    else:
        refs_q_v = rewards
    return states, actions, refs_q_v
    

def unpack_batch_DQN_double(batch, net, tgt_net, GAMMA=0.99, N_STEPS=2, device="cpu"):
    """
    Used to unpack batches of regular experiences (experience.Experience) and return the states, actions and Q_values.

    Returns the following: states, actions, refs_q_v

    More specifically, this class if for Double DQN models 
    - using main network for argmax step (Dev - Never tested, there might be unsqueezing to do)
    """
    states, actions, rewards, next_states, not_dones = utilities.unpack_batch(batch)

    states = agent.float32_preprocessing(states).to(device)
    rewards = agent.float32_preprocessing(rewards).to(device)
    actions = torch.LongTensor(np.array(actions, copy=False)).to(device)
    if next_states:
        tgt_q_v = torch.zeros_like(rewards)
        next_states = agent.float32_preprocessing(next_states).to(device)
        with torch.no_grad():
            tgt_q = net(next_states)
            argmax = tgt_q.argmax(dim=1)[0]
            tgt_q = tgt_net.target_model(next_states)
            tgt_q = tgt_q[argmax.unsqueeze(-1)]
            tgt_q_v[not_dones] = tgt_q

        refs_q_v = rewards + tgt_q_v * (GAMMA**N_STEPS)
    else:
        refs_q_v = rewards
    return states, actions, refs_q_v


def distr_projection(next_v_distr, rewards, not_dones, gamma : float, N_ATOMS : int, V_MAX : float, V_MIN : float, DELTA : float, device="cpu"):
    """
    Uses the distribution of future rewards and the current reward to project the target distribution.
    (generally this is used in cross-entropy related losses for distributions)
    """
    #handle episodes that were not terminated
    #next_v_distr is based on next_states which does not have the same shape as rewards!
    next_distr = next_v_distr.data.cpu().numpy()
    rewards = rewards.data.cpu().numpy()
    batch_size = len(next_distr) #length of next_distr for batch which doesn't match rewards!
    proj_distr = np.zeros(shape=(batch_size,N_ATOMS), dtype=np.float32)
    for atom in range(N_ATOMS):
        #project the atom from reward while respecting range
        proj_atom = np.minimum(V_MAX, np.maximum(V_MIN, rewards[not_dones] + (V_MIN + atom * DELTA)*gamma))
        #get index
        proj_id = (proj_atom - V_MIN) / DELTA
        lower = np.floor(proj_id).astype(np.int64)
        upper = np.ceil(proj_id).astype(np.int64)
        #handle rare event when atom falls directly on a boundary
        same = (lower == upper)
        proj_distr[same, lower[same]] += next_distr[same, atom]
        not_same = (lower != upper)
        proj_distr[not_same, lower[not_same]] += next_distr[not_same, atom] * (upper - proj_id)[not_same]
        proj_distr[not_same, upper[not_same]] += next_distr[not_same, atom] * (proj_id - lower)[not_same]

    #handle episodes that were terminated
    dones = (not_dones == False)    
    if dones.any():
        resized_proj_distr = np.zeros(shape=(len(rewards), N_ATOMS), dtype=np.float32)
        resized_proj_distr[not_dones] = proj_distr
        #this will handle the situation when the episode is over,
        #in this case the projected distribution will be a single or double strand representing the rewards
        proj_atom = np.minimum(V_MAX, np.maximum(V_MIN, rewards[dones]))
        proj_id = (proj_atom-V_MIN) / DELTA
        lower = np.floor(proj_id).astype(np.int64)
        upper = np.ceil(proj_id).astype(np.int64)
        same = (lower == upper)
        same_dones = dones.copy()
        same_dones[dones] = same
        if same_dones.any():
            resized_proj_distr[same_dones, lower[same]] = 1.0
        not_same = (lower != upper)
        not_same_dones = dones.copy()
        not_same_dones[dones] = not_same
        if not_same_dones.any():
            resized_proj_distr[not_same_dones, lower[not_same]] = (upper - proj_id)[not_same]
            resized_proj_distr[not_same_dones, upper[not_same]] = (proj_id - lower)[not_same]
        #overwrite proj_distr
        proj_distr = resized_proj_distr
    return torch.FloatTensor(proj_distr).to(device)