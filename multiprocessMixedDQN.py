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
    "ENV_NAME":"Breakout-v4",
    "complete":False,
    "LEARNING_RATE":1e-4,
    "GAMMA":0.99,
    "N_STEPS":4,
    "TGT_NET_SYNC":1000,
    "BATCH_SIZE":32,
    "REPLAY_SIZE":10000,
    "BETA_START":  0.4,
    'PRIO_REPLAY_ALPHA' : 0.6,
    'BETA_FRAMES' : 100000
}

device = "cuda"
epsilon_frames = 10000
beta_frames= parameters["BETA_FRAMES"]
beta = parameters["BETA_START"]
preprocessor = E.ndarray_preprocessor(E.FloatTensor_preprocessor())


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

def make_env(ENV_NAME, LiveRendering=False, inconn=None):
        env = gym.make(ENV_NAME)
        if LiveRendering:
            env = atari_wrappers.functionalObservationWrapper(env, sendimg(inconn, frame_skip=4))
        env = atari_wrappers.AutomateFireAction(env)
        env = atari_wrappers.FireResetEnv(env)
        env = atari_wrappers.MaxAndSkipEnv(env)
        env = atari_wrappers.ProcessFrame84(env)
        env = atari_wrappers.reshapeWrapper(env)
        env = atari_wrappers.ScaledFloatFrame(env, 148.)
        env = atari_wrappers.BufferWrapper(env, 3)
        env = atari_wrappers.oldStepWrapper(env)
        return env

def play_func(parameters, net, exp_queue, device, inconn=None):
        LiveRendering = True
        if inconn is None:
            LiveRendering = False
        env1 = make_env(parameters['ENV_NAME'], LiveRendering=LiveRendering, inconn=inconn)
        env2 = make_env(parameters['ENV_NAME'], LiveRendering=False)
        env3 = make_env(parameters['ENV_NAME'], LiveRendering=False)
        env = [env1, env2, env3]
        print(net)

        preprocessor = E.ndarray_preprocessor(E.FloatTensor_preprocessor())
        selector = ptan.actions.EpsilonGreedyActionSelector(1.0)
        agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
        exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=parameters.get('GAMMA', 0.99), steps_count=parameters['N_STEPS'])
        
        for i, exp in enumerate(exp_source):
            exp_queue.put(exp)
            if i > 5000:
                break
            for rewards, steps in exp_source.pop_rewards_steps():       
                exp_queue.put(EpisodeEnded(rewards, steps))

        selector.epsilon = 0.01
        for exp in exp_source:
            exp_queue.put(exp)
            
            for rewards, steps in exp_source.pop_rewards_steps():
                
                exp_queue.put(EpisodeEnded(rewards, steps))
            

if __name__ == '__main__':
    tmp.set_start_method("spawn")
    inconn, outconn = tmp.Pipe()

    #init display
    p1 = tmp.Process(target=Rendering.init_display, args=(outconn, 320, 420))
    p1.start()


    def calc_loss(states, actions, rewards, last_states, dones, tgt_net, net):
                last_states = preprocessor(last_states).to(device)
                
                with torch.no_grad(): #try to use numpy instead
                    tgt_q = net(last_states)
                    tgt_actions = torch.argmax(tgt_q, 1)
                    tgt_actions = tgt_actions.unsqueeze(1)
                    tgt_qs = tgt_net.target_model(last_states)
                    tgt_q_v = tgt_qs.gather(1, tgt_actions).squeeze(1)

                tgt_q_v[dones] = 0.0
                rewards = preprocessor(rewards).to(device)
                q_v_refs = rewards + tgt_q_v * parameters['GAMMA']
                optimizer.zero_grad()
                states = preprocessor(states).to(device)
                actions = torch.tensor(actions).to(device)
                q_v = net(states)
                q_v = q_v.gather(1, actions.unsqueeze(1)).squeeze(1)

                batch_w_v = torch.tensor(batch_weights).to(device)
                losses = batch_w_v *(q_v - q_v_refs) **2
                loss = losses.mean()
                sample_prios_v = losses + 1e-5
                return loss, sample_prios_v

    #multiprocessing for training
    
    Batch_MUL = 4

    def get_obs_act_n():
        env = make_env(parameters['ENV_NAME'], LiveRendering=False)
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n
        return obs_shape, n_actions
    
    obs_shape, n_actions = get_obs_act_n()

    net = models.NoisyDuelDQN(obs_shape, n_actions).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    backup = E.ModelBackup(parameters['ENV_NAME'], net=net, notify=True)
    writer = SummaryWriter(comment=parameters['ENV_NAME'] +"_--" + device)

    print(net)
    
    print(E.meanmax_weight(net))

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
                yield self.buffer.sample(self.batch_size*Batch_MUL, beta=beta)
    
    
    exp_queue = tmp.Queue(maxsize=Batch_MUL*2)

    play_proc = tmp.Process(target=play_func, args=(parameters, net, exp_queue, device, inconn))
    play_proc.start()
    
    

    optimizer = torch.optim.Adam(net.parameters(), lr=parameters['LEARNING_RATE'])
    idx = 0
    running = True
    buffer = ptan.experience.PrioritizedReplayBuffer(experience_source=None,buffer_size=parameters["REPLAY_SIZE"], alpha=parameters["PRIO_REPLAY_ALPHA"])
    BatchGen = BatchGenerator(buffer=buffer, exp_queue=exp_queue, initial= 2*parameters["BATCH_SIZE"], batch_size=parameters["BATCH_SIZE"])

    t1 = time.time()
    while running:
        idx +=1
        
        beta = min(1.0, beta + idx * (1.0 - beta) / beta_frames)

        for rewards, steps in BatchGen.pop_rewards_steps():
            t2 = time.time() - t1
            print("idx %d, steps %d, reward=%.3f rewards, elapsed: %.1f" %(idx,steps,rewards, t2))
            writer.add_scalar("rewards", rewards, idx)
            writer.add_scalar("steps", steps, idx)
            
            solved = rewards > 350
            if solved:
                print("Solved!")
                parameters["complete"] = True
                backup.save(parameters=parameters)
                running=False
                continue
        
        batch, batch_idxs, batch_weights = next(iter(BatchGen))

        states, actions, rewards, last_states, dones = E.unpack_batch(batch, obs_shape)
        loss, sample_prios_v = calc_loss(states, actions, rewards, last_states, dones, tgt_net, net)
        loss.backward()
        optimizer.step()

        buffer.update_priorities(batch_idxs,sample_prios_v.data.cpu().numpy())
        writer.add_scalar("loss", loss, idx)

        if idx % 20 == 0:
            print("loss %.3f" % (loss))
        if idx % 100 == 0:
            mean_w, max_w =E.meanmax_weight(net)
            print(mean_w, max_w)
        if idx % parameters['TGT_NET_SYNC'] ==0:
        
            tgt_net.sync()
    
        if idx % 40000 == 0:
            backup.save(parameters=parameters)
    




        