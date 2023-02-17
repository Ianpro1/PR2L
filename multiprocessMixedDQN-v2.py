import gym
import numpy as np
import torch
import ptan
import common.extentions as E
import common.models as models
import common.atari_wrappers as atari_wrappers
from torch.utils.tensorboard import SummaryWriter
import time
from collections import namedtuple
import common.Rendering as Rendering
import torch.multiprocessing as tmp

EpisodeEnded = namedtuple("EpisodeEnded", ("reward", "steps"))

parameters = {
    "ENV_NAME":"PongNoFrameskip-v4",
    "complete":False,
    "LEARNING_RATE":5e-4,
    "GAMMA":0.99,
    "N_STEPS":4,
    "TGT_NET_SYNC":300,
    "BATCH_SIZE":32,
    "REPLAY_SIZE":10000,
    "BETA_START":  0.4,
    'PRIO_REPLAY_ALPHA' : 0.6,
    'BETA_FRAMES' : 5000
}

device = "cuda"
gamma = parameters['GAMMA']
beta_frames= parameters["BETA_FRAMES"]
beta = parameters["BETA_START"]
preprocessor = E.ndarray_preprocessor(E.FloatTensor_preprocessor())

def get_obs_act_n():
    env = make_env(parameters['ENV_NAME'])
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    return obs_shape, n_actions

class sendimg:
    def __init__(self, inconn, frame_skip=2):
        self.inconn = inconn
        self.frame_skip = frame_skip
        self.count = 0
    def __call__(self, img):
        self.count +=1
        if self.count % self.frame_skip ==0:
            self.count = 0
            self.inconn.send(img)
        return img

def make_env(ENV_NAME, inconn=None):
        env = gym.make(ENV_NAME)

        if inconn is not None: # or if inconn is not None
            env = atari_wrappers.functionalObservationWrapper(env, sendimg(inconn, frame_skip=16)) 
        env = atari_wrappers.AutomateFireAction(env)
        env = atari_wrappers.FireResetEnv(env)
        env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
        env = atari_wrappers.ProcessFrame84(env)
        env = atari_wrappers.reshapeWrapper(env)
        env = atari_wrappers.ScaledFloatFrame(env, 255.)
        env = atari_wrappers.BufferWrapper(env, 3)
        env = atari_wrappers.oldStepWrapper(env)
        '''if inconn is not None:
            env = atari_wrappers.LiveRenderWrapper(env, sendimg(inconn, frame_skip=4))'''
        return env

def play_func(parameters, net, exp_queue, device, inconn=None):

        env1 = make_env(parameters['ENV_NAME'], inconn=inconn)
        env2 = make_env(parameters['ENV_NAME'])
        env3 = make_env(parameters['ENV_NAME'])
        env = [env1, env2, env3]
        print(net)
        preprocessor = E.ndarray_preprocessor(E.FloatTensor_preprocessor())
        selector = ptan.actions.ArgmaxActionSelector()
        agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
        exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=parameters.get('GAMMA', 0.99), steps_count=parameters['N_STEPS'])
        
        time.sleep(1)
        Rendering.params_toDataFrame(net, path="DataFrames/parallelNetwork_params.csv")
        idz = 0
        for exp in exp_source:
            idz +=1
            exp_queue.put(exp)
            
            '''if idz % 500 == 0:
                Rendering.params_toDataFrame(net, path="DataFrames/parallelNetwork_params.csv")'''

            for rewards, steps in exp_source.pop_rewards_steps():   
                exp_queue.put(EpisodeEnded(rewards, steps))

def unpack_batchv2(batch): #this can be removed
    # returns: states, actions, calculated tgt_q_v => r + tgt_net(last_state)*GAMMA
    #used for most basic DQN
    states = []
    rewards = []
    actions = []
    last_states = []
    not_dones = []
    for exp in batch:
        states.append(exp.state)
        rewards.append(exp.reward)
        actions.append(exp.action)
        
        if exp.last_state is not None:
            not_dones.append(True)
            last_states.append(exp.last_state) 
        else:
            not_dones.append(False)   
    return states, actions, rewards, last_states, not_dones

def calc_loss(states, actions, rewards, last_states, not_dones, tgt_net, net):
    last_states = preprocessor(last_states).to(device)
    rewards = preprocessor(rewards).to(device)
    states = preprocessor(states).to(device)
    actions = torch.tensor(actions).to(device)
    #batch_w_v = torch.tensor(batch_weights).to(device)

    with torch.no_grad(): #try to use numpy instead

        #compute tgt_q values (does not include dones)
        tgt_q = tgt_net.target_model(last_states)

        #get index of best tgt_q values
        tgt_actions = torch.argmax(tgt_q, 1).unsqueeze(1)

        #best tgt_q_v gathered
        tgt_q = tgt_q.gather(1, tgt_actions).squeeze(1)

        #get back dones as 0 values
        #implementation 1
        tgt_q_v = torch.zeros_like(rewards) #used rewards to get shape
        tgt_q_v[not_dones] = tgt_q
        
        #bellman step
        q_v_refs = rewards + tgt_q_v * gamma

    optimizer.zero_grad()
    q_v = net(states)
    q_v = q_v.gather(1, actions.unsqueeze(1)).squeeze(1)
    losses = (q_v - q_v_refs.detach()).pow(2)
    #losses = batch_w_v * losses
    loss = losses.mean()
    sample_prios_v = losses + 1e-5
    return loss, sample_prios_v

class BatchGenerator:
    def __init__(self, buffer, exp_queue, initial, batch_size):
        self.buffer = buffer
        self.exp_queue = exp_queue
        self.initial = initial
        self.batch_size = batch_size
        self.rewardSteps = []

    def pop_rewards_steps(self):
        res = list(self.rewardSteps)
        self.rewardSteps.clear()
        return res
    
    def __iter__(self):
        while True:
            while self.exp_queue.qsize() > 0:
                exp = self.exp_queue.get()
                if isinstance(exp, EpisodeEnded):
                    self.rewardSteps.append((exp.reward, exp.steps))
                else:
                    self.buffer._add(exp)
            
            if len(self.buffer) < self.initial:
                continue
            yield self.buffer.sample(self.batch_size*Batch_MUL) #beta=beta


if __name__ == '__main__':

    tmp.set_start_method("spawn")
    inconn, outconn = tmp.Pipe()

    #init display
    p1 = tmp.Process(target=Rendering.init_display, args=(outconn, 320, 420, (210, 160)))
    p1.start()


    #multiprocessing for training
    
    Batch_MUL = 4
    
    obs_shape, n_actions = get_obs_act_n()
    
    net = models.NoisyDualDQN(obs_shape, n_actions).to(device)
    net.share_memory()
    
    tgt_net = ptan.agent.TargetNet(net)
    
    backup = E.ModelBackup(parameters['ENV_NAME'], net=net, notify=True)
    writer = SummaryWriter(comment=parameters['ENV_NAME'] +"_--" + device)  
    
    exp_queue = tmp.Queue(maxsize=Batch_MUL*2)

    play_proc = tmp.Process(target=play_func, args=(parameters, net, exp_queue, device, inconn)) #->inconn
    play_proc.start()
    net.apply(models.network_reset)
    time.sleep(1)
    net.apply(models.network_reset)
    #net.load_state_dict(torch.load("-v4/modelsave0_().pt"))
    tgt_net.sync()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=parameters['LEARNING_RATE'])
    idx = 0
    running = True
    buffer = ptan.experience.ExperienceReplayBuffer(None, parameters["REPLAY_SIZE"]) #try a waiting list to wait for network sync
    #buffer = ptan.experience.PrioritizedReplayBuffer(experience_source=None,buffer_size=parameters["REPLAY_SIZE"], alpha=parameters["PRIO_REPLAY_ALPHA"])
    BatchGen = BatchGenerator(buffer=buffer, exp_queue=exp_queue, initial= 2*parameters["BATCH_SIZE"], batch_size=parameters["BATCH_SIZE"])

    t1 = time.time()
    Rendering.params_toDataFrame(net, path="DataFrames/mainNetwork_params.csv")
    Rendering.params_toDataFrame(tgt_net.target_model, path="DataFrames/tgtNetwork_params.csv")
    
    solved = False
    #grads = Rendering.grads_manager(net, func="mean", path='DataFrames/mainNetwork_grads.csv')
    
    #batch, batch_idxs, batch_weights = next(iter(BatchGen))
    for batch in BatchGen:
        idx +=1

        #if idx % 500 == 0:
            #grads.write_csv()

        for rewards, steps in BatchGen.pop_rewards_steps():
            t2 = time.time() - t1
            print("idx %d, steps %d, reward=%.1f rewards, elapsed: %.1f" %(idx,steps,rewards, t2))
            print("loss %.3f" % (loss))
            writer.add_scalar("rewards", rewards, idx)
            writer.add_scalar("steps", steps, idx)

            solved = rewards > 19

        if solved:
            print("Solved!")
            parameters["complete"] = True
            backup.save(parameters=parameters)
            break          

        states, actions, rewards, last_states, not_dones = unpack_batchv2(batch)
        loss, sample_prios_v = calc_loss(states, actions, rewards, last_states, not_dones, tgt_net, net)
        loss.backward()
        optimizer.step()

        #buffer.update_priorities(batch_idxs,sample_prios_v.data.cpu().numpy())
        writer.add_scalar("loss", loss, idx)  

        if idx %5000 == 0:
            Rendering.params_toDataFrame(net, path="DataFrames/mainNetwork_params.csv")
            #Rendering.params_toDataFrame(tgt_net.target_model, path="DataFrames/tgtNetwork_params.csv")
        
        if idx % parameters['TGT_NET_SYNC'] ==0:
            print("sync...")
            tgt_net.sync()
    
        if idx % 40000 == 0:
            backup.save(parameters=parameters)
