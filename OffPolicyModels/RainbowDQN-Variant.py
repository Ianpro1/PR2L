#DQN from the multiprocessMixedDQN family found in DQNModels (equivalent to multiprocessMixedDQN-v7(noisy))

#Dual Model which uses NoisyLinear layers along with BatchNormalization

#Changes from multiprocessMixedDQNs include:
# not using Double DQN strategy, removed PriorityBuffer and using rendering from PR2L library
 
import gym
import numpy as np
import torch
import torch.nn.utils as utils
from common import common_wrappers
import common.models as models
from torch.utils.tensorboard import SummaryWriter
import time
from collections import namedtuple, deque
import torch.multiprocessing as tmp
from PR2L import agent, utilities, experience, rendering
from PR2L.training import unpack_batch
from gym.wrappers.atari_preprocessing import AtariPreprocessing

EpisodeEnded = namedtuple("EpisodeEnded", ("reward", "steps"))

parameters = {
    "ENV_NAME":"BreakoutNoFrameskip-v4",
    "complete":False,
    "LEARNING_RATE":1e-4,
    "GAMMA":0.99,
    "N_STEPS":4,
    "TGT_NET_SYNC":300,
    "BATCH_SIZE":32,
    "REPLAY_SIZE":10000,
    "Noisy": True,
    "CLIP_GRAD":0.1,
    "SOLVED":1000
}

device = "cuda" if torch.cuda.is_available() else "cpu"
preprocessor = agent.float32_preprocessing

def make_env(ENV_NAME, inconn=None, render=False): 
        if render:
            env = gym.make(ENV_NAME, render_mode="rgb_array")
            env = utilities.render_env(env)
        else:
            env = gym.make(ENV_NAME)
            if inconn is not None: 
                env = rendering.SendimgWrapper(env, inconn, frame_skip=4) 
        env = AtariPreprocessing(env)
        env = common_wrappers.RGBtoFLOAT(env)
        env = common_wrappers.BetaSumBufferWrapper(env, 3, 0.4)
        env = common_wrappers.SingleChannelWrapper(env)
        #specific to atariBreakout
        env = common_wrappers.PenalizedLossWrapper(env)
        return env

def play_func(parameters, net, exp_queue, device, inconn=None):

        env1 = make_env(parameters['ENV_NAME'], inconn=inconn)
        env2 = make_env(parameters['ENV_NAME'])
        env3 = make_env(parameters['ENV_NAME'])
        
        env = [env1, env2, env3]
        print(net)
        selector = agent.ArgmaxSelector()
        _agent = agent.BasicAgent(net, device, selector)
        exp_source = experience.ExperienceSource(env, _agent, parameters['N_STEPS'], GAMMA=parameters.get('GAMMA', 0.99))
        
        idz = 0
        for exp in exp_source:
            
            exp_queue.put(exp)
            
            '''if idz % 500 == 0:
                Rendering.params_toDataFrame(net, path="DataFrames/parallelNetwork_params.csv")'''


            for rewards, steps in exp_source.pop_rewards_steps():
                idz +=1
                print(idz)
                exp_queue.put(EpisodeEnded(rewards, steps))

def calc_loss(states, actions, rewards, last_states, not_dones, tgt_net, net):
    last_states = preprocessor(last_states).to(device)
    rewards = preprocessor(rewards).to(device)
    states = preprocessor(states).to(device)
    actions = torch.tensor(np.array(actions, copy=False)).to(device)

    #Warning: rare error thrown when len(last_states) < 1
    with torch.no_grad(): #TODO use numpy instead

        #compute tgt_q values (does not include dones)
        tgt_q = tgt_net.target_model(last_states)
        
        tgt_q = tgt_q.max(dim=1)[0]
        
        #get back dones as 0 values
        #implementation 1
        tgt_q_v = torch.zeros_like(rewards) #used rewards to get shape
        tgt_q_v[not_dones] = tgt_q
    
        #bellman step
        q_v_refs = rewards + tgt_q_v * (parameters["GAMMA"]**parameters["N_STEPS"])

    optimizer.zero_grad()
    q_v = net(states)
    q_v = q_v.gather(1, actions.unsqueeze(1)).squeeze(1)
    losses = (q_v - q_v_refs.detach()).pow(2)
    loss = losses.mean()
    return loss

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
    

    #init display
    inconn, outconn = tmp.Pipe()
    p1 = tmp.Process(target=rendering.init_display, args=(outconn, (210, 160)))
    p1.start()
    
    #multiprocessing for training
    Batch_MUL = 4
    
    #specific for atari
    obs_shape = (1, 84, 84)
    n_actions = 4
    
    net = models.NoisyDualDQN(obs_shape, n_actions).to(device)
    net.share_memory()
    
    tgt_net = agent.TargetNet(net)
    render_agent = agent.BasicAgent(net, device)
    render_env = make_env(parameters["ENV_NAME"], render=True)
    backup = utilities.ModelBackup(parameters['ENV_NAME'],"001", net, render_env=render_env, agent=render_agent)
    writer = SummaryWriter(comment=parameters['ENV_NAME'] +"_--" + device)  
    
    exp_queue = tmp.Queue(maxsize=Batch_MUL*2)

    play_proc = tmp.Process(target=play_func, args=(parameters, net, exp_queue, device, inconn)) #->inconn
    play_proc.start()
    net.apply(models.network_reset)
    time.sleep(1)
    net.apply(models.network_reset)
    net.load_state_dict(torch.load("model_saves/BreakoutNoFrameskip-v4/model_001/state_dicts/2023-03-09/save-00-37.pt"))
    tgt_net.sync()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=parameters['LEARNING_RATE'])
    idx = 0
    running = True
    buffer = experience.SimpleReplayBuffer(None, parameters["REPLAY_SIZE"]) #try a waiting list to wait for network sync
    
    BatchGen = BatchGenerator(buffer=buffer, exp_queue=exp_queue, initial= 2*parameters["BATCH_SIZE"], batch_size=parameters["BATCH_SIZE"])

    t1 = time.time()
    solved = False    
    solved = deque(maxlen=20)
    solved.append(0.)
    rew_m = 0.
    loss = 404.

    for batch in BatchGen:
        idx +=1

        for rewards, steps in BatchGen.pop_rewards_steps():
            t2 = time.time() - t1
            solved.append(rewards)
            rew_m = np.array(solved, dtype=np.float32).mean()


            print("idx %d, steps %d, mean_reward=%.1f, elapsed: %.1f" %(idx,steps,rew_m, t2))
            print("loss %.3f" % (loss))
            writer.add_scalar("rewards", rewards, idx)
            writer.add_scalar("steps", steps, idx)

            solved.append(rewards)

        if rew_m > parameters["SOLVED"]:
            print("Solved!")
            parameters["complete"] = True
            backup.save(parameters=parameters)
            backup.mkrender(fps=160.0)
            break          

        states, actions, rewards, last_states, not_dones = unpack_batch(batch)
        loss = calc_loss(states, actions, rewards, last_states, not_dones, tgt_net, net)
        loss.backward()

        #utils.clip_grad_norm_(net.parameters(), parameters["CLIP_GRAD"])
        optimizer.step()

        writer.add_scalar("loss", loss, idx)  
        
        if idx % parameters['TGT_NET_SYNC'] ==0:
            print("sync...")
            tgt_net.sync()
    
        if idx % 60000 == 0:
            backup.save(parameters=parameters)
            backup.mkrender(fps=160.0)