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
    "epsilon":1.0,
    "EPSILON_START":1.0,
    "EPSILON_DECAY_LAST_FRAME":150000,
    "BATCH_SIZE":32,
    "REPLAY_SIZE":10000,
    "EPSILON_FINAL":0.01   
}


import multiprocessing as mp
import common.Rendering as Rendering

if __name__ == '__main__':
    #inconn, outconn = mp.Pipe()
        
    '''class sendimg:
        def __init__(self, inconn, frame_skip=2):
            self.inconn = inconn
            self.frame_skip = frame_skip
            self.count = 0
        def __call__(self, img):
            self.count +=1
            if self.count % self.frame_skip ==0:
                self.count = 0
                inconn.send(img)
            return img'''

    #p1 = mp.Process(target=Rendering.init_display, args=(outconn, 320, 420))
    #p1.start()
    
    def make_env(ENV_NAME, LiveRendering=False):
        env = gym.make(ENV_NAME)
        '''if LiveRendering:
            env = atari_wrappers.functionalObservationWrapper(env, sendimg(inconn, frame_skip=4))'''
        env = atari_wrappers.AutomateFireAction(env)
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

    device = "cuda"

    obs_shape = env.observation_space.shape

    net = models.NoisyDualDQN(obs_shape, env.action_space.n).to(device)
    print(net)
    tgt_net = ptan.agent.TargetNet(net)

    #make list of env
    env2 = make_env(parameters['ENV_NAME'], LiveRendering=False)
    env3 = make_env(parameters['ENV_NAME'], LiveRendering=False)
    env = [env, env2, env3]

    preprocessor = E.ndarray_preprocessor(E.FloatTensor_preprocessor())
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=parameters['EPSILON_START'])
    agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device, preprocessor=preprocessor)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=parameters.get('GAMMA'), steps_count=parameters['N_STEPS'])
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=parameters['REPLAY_SIZE'])
    optimizer = torch.optim.Adam(net.parameters(), lr=parameters['LEARNING_RATE'])


    '''net.load_state_dict(torch.load("Breakout-v4/2023-02-11/modelsave1_(03-03).pt"))
    tgt_net.sync()'''

    idx = 0
    episode = 0
    ts_frame = 0
    ts = time.time()

    backup = E.ModelBackup(parameters['ENV_NAME'], net=net, notify=True)
    #need to save settings and create new model folder to keep old models and new ones

    writer = SummaryWriter(comment=parameters['ENV_NAME'] +"_--" + device)


    from common.performance import timer

    '''timer = timer()
    timer.start()'''
    while True:
        '''if idx > 5000:
            #3 env = 82.879
            timer.stop()'''
        idx += 1
        buffer.populate(1)
        
        for rewards, steps in exp_source.pop_rewards_steps():
            speed = (idx - ts_frame) / (time.time() - ts)
            ts_frame = idx
            ts = time.time()
            episode+=1
            #print("FPS %.0f" % speed, "reward: %.2f" %rewards )
            print("idx %d, steps %d, episode %d done, reward=%.3f rewards, epsilon=%.4f, FPS %.3f" %(idx,steps,episode, rewards, selector.epsilon, speed))
            writer.add_scalar("episodes", episode, idx)
            writer.add_scalar("rewards", rewards, idx)
            writer.add_scalar("epsilon", selector.epsilon, idx)
            writer.add_scalar("FPS", speed, idx)
            solved = rewards > 350
            if solved:
                print("done in %d episodes" % episode)
                parameters["epsilon"] = selector.epsilon
                parameters["complete"] = True
                backup.save(parameters=parameters)
                break
        if len(buffer) < 2*parameters['BATCH_SIZE']:
            continue
        batch = buffer.sample(parameters['BATCH_SIZE'])
        states, actions, rewards, last_states, dones = E.unpack_batch(batch, obs_shape)
        
        # agent returns best actions for tgt

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
        loss = torch.nn.functional.mse_loss(q_v, q_v_refs)
        loss.backward()
        optimizer.step()
        
        print(loss)
        writer.add_scalar("loss", loss, idx)
        
        selector.epsilon = max(parameters['EPSILON_FINAL'], parameters['EPSILON_START'] - idx / parameters['EPSILON_DECAY_LAST_FRAME'])
        
        if idx % parameters['TGT_NET_SYNC'] ==0:
            
            tgt_net.sync()
        
        if idx % 50000 == 0:
            parameters["epsilon"] = selector.epsilon
            backup.save(parameters=parameters)
    
    p1.join()

