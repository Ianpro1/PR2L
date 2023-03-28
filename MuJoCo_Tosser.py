import mjct
import multiprocessing as mp
import PR2L
import gymnasium as gym
import numpy as np


#example script that creates 4 TosserCPP from mjct

def play():

    env = mjct.make("TosserCPP", render=True)
    obs, info = env.reset()
    idx=0
    while(True):
        running=True
        triggerzone = np.random.randint(0, 22, 1)
        while(running):
            idx += 1
            
            if idx >triggerzone:
                action = [1,-0.87]
            else:
                action = [0,0]
            obs, reward, done, trunc, info = env.step(action)
            env.render()
            if done:
                idx = 0
                print("reset_True")
                env.reset()
                running=False



if __name__ == "__main__":

    mp.set_start_method('spawn')

    processes = [mp.Process(target=play) for x in range(4)]

    for p in processes:
        p.start()

    