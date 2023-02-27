import torch
import datetime
import os


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


#backup
class ModelBackup:
    def __init__(self, ENV_NAME, iid, net, notify=True, env=None, render_source=None, folder="model_saves", prefix="model_"):
        assert isinstance(iid, str)
        self.env = env
        self.render_source = render_source
        self.net = net
        self.notify = notify
        self.path = os.path.join(folder, ENV_NAME, prefix+iid)
        if os.path.isdir(self.path) == False:
            os.makedirs(self.path)

    def save(self, parameters=None):
        assert isinstance(parameters, (dict, type(None)))
        date = str(datetime.datetime.now().date())
        time = datetime.datetime.now().strftime("-%H-%M")

        if parameters is not None:
            temproot = os.path.join(self.path, "parameters", date)

            if os.path.isdir(temproot) == False:
                os.makedirs(temproot)

            with open(os.path.join(temproot, "save" + time +".txt"), 'w') as f:
                f.write(str(parameters))
        
        temproot = os.path.join(self.path, "state_dicts", date)
        if os.path.isdir(temproot) == False:
            os.makedirs(temproot)

        location = os.path.join(temproot, "save" + time +".pt")
        torch.save(self.net.state_dict(), location)
        
        if self.notify:
            print("created " + location)
    
    def mkrender(self):
        pass  