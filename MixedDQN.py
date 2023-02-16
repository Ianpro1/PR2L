import gym
import numpy as np
import torch
import ptan
import common.extentions as E
import common.models as models
import common.atari_wrappers as atari_wrappers
from torch.utils.tensorboard import SummaryWriter
import time

parameters = {
    "ENV_NAME":"PongNoFrameskip-v4",
    "complete":False,
    "LEARNING_RATE":1e-4,
    "GAMMA":0.99,
    "N_STEPS":4,
    "TGT_NET_SYNC":1000,
    "BATCH_SIZE":32,
    "REPLAY_SIZE":10000,
    "BETA_START":  0.4,
    'PRIO_REPLAY_ALPHA' : 0.6,
    'BETA_FRAMES' : 100000,
    'penalize': False
}


import multiprocessing as mp
import common.Rendering as Rendering

if __name__ == '__main__':
    inconn, outconn = mp.Pipe()
        
    class sendimg:
        def __init__(self, inconn, frame_skip=2):
            self.inconn = inconn
            self.frame_skip = frame_skip
            self.count = 0
        def __call__(self, img):
            self.count +=1
            if self.count % self.frame_skip ==0:
                self.count = 0
                inconn.send(img)
            return img

    p1 = mp.Process(target=Rendering.init_display, args=(outconn, 320, 420, (210, 160)))
    p1.start()
    
    def make_env(ENV_NAME, LiveRendering=False):
        env = gym.make(ENV_NAME)
        if LiveRendering:
            env = atari_wrappers.functionalObservationWrapper(env, sendimg(inconn, frame_skip=4))
        env = atari_wrappers.AutomateFireAction(env, parameters["penalize"])
        env = atari_wrappers.FireResetEnv(env)
        env = atari_wrappers.MaxAndSkipEnv(env)
        env = atari_wrappers.ProcessFrame84(env)
        env = atari_wrappers.reshapeWrapper(env)
        env = atari_wrappers.ScaledFloatFrame(env, 148.)
        env = atari_wrappers.BufferWrapper(env, 3)
        #env = atari_wrappers.oldWrapper(env) # this is messing up the buffer wrapper
        env = atari_wrappers.oldStepWrapper(env)
        return env

    env = make_env(parameters['ENV_NAME'], LiveRendering=True)
    env1 = make_env(parameters['ENV_NAME'], LiveRendering=False)
    env2 = make_env(parameters['ENV_NAME'], LiveRendering=False) #no sure this is optimal for an argmaxselection agent

    
    
    device = "cuda"
    preprocessor = E.ndarray_preprocessor(E.FloatTensor_preprocessor())
    obs_shape = env.observation_space.shape
    act_n = env.action_space.n
    net = models.NoisyDuelDQN(obs_shape, act_n).to(device)
    print(net)
    
    
    data = preprocessor([env.reset()])
    tgt_net = ptan.agent.TargetNet(net)

    print(env.unwrapped.get_action_meanings())

    env = [env, env1, env2]
    
    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=parameters.get('GAMMA'), steps_count=parameters['N_STEPS'])
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, parameters["REPLAY_SIZE"], alpha=parameters["PRIO_REPLAY_ALPHA"])
    optimizer = torch.optim.Adam(net.parameters(), lr=parameters['LEARNING_RATE'])

    
    idx = 0
    episode = 0
    ts_frame = 0
    ts = time.time()

    backup = E.ModelBackup(parameters['ENV_NAME'], net=net, notify=True)

    writer = SummaryWriter(comment=parameters['ENV_NAME'] +"_--" + device)
    beta_frames= parameters["BETA_FRAMES"]
    beta = parameters["BETA_START"]
    
    
    '''net.load_state_dict(torch.load("Breakout-v4/2023-02-12/modelsave31_(12-10).pt"))
    tgt_net.sync()'''

    running = True

    while running:
        idx += 1
        buffer.populate(1)
        beta = min(1.0, beta + idx * (1.0 - beta) / beta_frames)

        for rewards, steps in exp_source.pop_rewards_steps():
            speed = (idx - ts_frame) / (time.time() - ts)
            ts_frame = idx
            ts = time.time()
            episode+=1
            #print("FPS %.0f" % speed, "reward: %.2f" %rewards )
            print("idx %d, steps %d, episode %d done, reward=%.1f rewards, FPS %d" %(idx,steps,episode, rewards, speed))
            writer.add_scalar("episodes", episode, idx)
            writer.add_scalar("rewards", rewards, idx)
            writer.add_scalar("FPS", speed, idx)
            solved = rewards > 350
            if solved:
                print("done in %d episodes" % episode)
                parameters["complete"] = True
                backup.save(parameters=parameters)
                running = False
                break
        if len(buffer) < 2*parameters['BATCH_SIZE']:
            continue

        batch, batch_idxs, batch_weights  = buffer.sample(parameters['BATCH_SIZE'], beta=beta)
        states, actions, rewards, last_states, dones = E.unpack_batch(batch, obs_shape)
    
        last_states = preprocessor(last_states).to(device)
        rewards = preprocessor(rewards).to(device)
        with torch.no_grad(): #try to use numpy instead
            tgt_q = net(last_states)
            tgt_actions = torch.argmax(tgt_q, 1)
            tgt_actions = tgt_actions.unsqueeze(1)
            tgt_qs = tgt_net.target_model(last_states)
            tgt_q_v = tgt_qs.gather(1, tgt_actions).squeeze(1)
            tgt_q_v[dones] = 0.0
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
        loss.backward()
        optimizer.step()

        buffer.update_priorities(batch_idxs,sample_prios_v.data.cpu().numpy())
            
        writer.add_scalar("loss", loss, idx)
        if idx % parameters['TGT_NET_SYNC'] ==0:
            
            tgt_net.sync()
        
        if idx % 50000 == 0:
            backup.save(parameters=parameters)
    
    p1.join()

