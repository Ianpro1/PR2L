

def unpack_batch(batch):
    #Note: the unpack function returns a reduce size last_state list (does not include terminated states)
    states = []
    rewards = []
    actions = []
    last_states = []
    not_dones = []
    for exp in batch:
        states.append(exp.state)
        rewards.append(exp.reward)
        actions.append(exp.action)
        
        if exp.next is not None:
            not_dones.append(True)
            last_states.append(exp.next) 
        else:
            not_dones.append(False)   
    return states, actions, rewards, last_states, not_dones