#playground provides Dummy environments and tools for testing basic algorithms 
try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        raise ImportError("Failed to import both gymnasium and gym")

import numpy as np
import pygame
import warnings
import random

class DeterministicObservations:
    """
    Once initialized, the class __call__ should always return the same observation on every given step.
    Since the observations are initially random, there are stored in a buffer of length = buffer_length.
    The second argument: input_shape, will defined the shape of every observation.
    """
    def __init__(self, buffer_length, input_shape):
        shape = [buffer_length]
        shape.extend(input_shape)
        self.buffer = np.random.uniform(-100., 100., size=shape)
        self.pos = 0
    def __call__(self, y, x):
        if x is None:
            self.pos = 0
        elif self.pos >= len(self.buffer):
            self.pos = 0

        obs = self.buffer[self.pos]
        self.pos += 1
        return obs

class DeterministicRewards:
    """
    Once initialized, the class __call__ should always return the same rewards on every given step.
    Since the rewards are initially random, there are stored in a buffer of length = buffer_length.
    """
    def __init__(self, buffer_length):
        self.buffer = np.random.choice(2, size=buffer_length, p=(0.20, 0.80))
        self.pos = 0
    def __call__(self, y, x):
        if x is None:
            self.pos = 0
        elif self.pos >= len(self.buffer):
            self.pos = 0

        rew = self.buffer[self.pos]
        self.pos += 1
        return rew

class Dummy(gym.Env):
    """
    A class environment inheriting gym.Env.
    The class serves as dummy for testing algorithms, where you can defined custom observation functions, rewards functions and more.
    Using the default argument will result in an environment that uses DeterministicObservations() as obs_func and DeterministicRewards() as rew_func
    """
    def __init__(self, obs_shape, obs_func=None, rew_func=None, done_func=None, trunc_func=None, info_func=None):
        self.cur_obs = None
        self.shape = obs_shape
        
        if obs_func:                
            self.obs_func = obs_func
        else:
            self.obs_func = DeterministicObservations(1000, self.shape)
        
        if rew_func:
            self.rew_func = rew_func
        else:
            self.rew_func = DeterministicRewards(1000)

        if done_func:
            self.done_func = done_func
        else:
            self.done_func = lambda x, y: False

        if trunc_func:
            self.trunc_func = trunc_func
        else:
            self.trunc_func = lambda x, y: False
        
        if info_func:
            self.info_func = info_func
        else:
            self.info_func = lambda x, y: {"DummyEnv":True}

    def reset(self):
        action = None
        obs = self.obs_func(self, action)
        self.rew_func(self, action)
        self.done_func(self, action)
        self.trunc_func(self, action)
        info = self.info_func(self, action)

        self.cur_obs = (obs, info)
        return obs, info

    def step(self, action):
        obs = self.obs_func(self, action)
        rew = self.rew_func(self, action)
        done = self.done_func(self, action)
        trunc = self.trunc_func(self, action)
        info = self.info_func(self, action)

        self.cur_obs = (obs, rew, done, trunc, info)
        return obs, rew, done, trunc, info

class OldDummyWrapper(Dummy):
    """wraps the Dummy class to configure it as an older version of gym environments (removes truncatedflag from outputs, and info flag when reseting)"""
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, r, d, _, i = self.env.step(action)
        return obs, r, d, i
    
    def reset(self):
        obs, _ = self.env.reset()
        return obs

class EpisodeLength:
    """
    Class that can be used for the dummy's done_func.
    The class will make each episode end after n steps (the environment's n+1 step will return done)
    """
    def __init__(self, length):
        self.len = length
        self.count = 0
    def __call__(self, y, x):
        if self.count > self.len -1:
            self.count = 0
            return True
        else:
            self.count +=1
            return False

class BasicController:
    """
    Basic Controller is a controller based on pygame that can be configured on the go by following the prompts.
    It lets the user play and test environments using keyboard inputs.

    Arguments: *action_shape is the fixed shape of the output that is passed into the user's *step_callback function.
    *reload_callback is a custom callback function separate from the controller that can be called using '`' (if enabled,
    it can be used to allow the user to reset the environment anytime). 

    NOTE: to create custom controllers and load them instead of configurating them during runtime, the user can utilize the
    loadController() method. It expects a list of inputs of the form: ['w', 'a', 's', 'd'] and a list of outputs of same length: 
    [(1.0, [0]), (-1.0, [0]), (1.0, [1]), (-1.0, [1])] where the first index of each output represents: the output's value and
    the second: the index which will be overwritten with the output's value in the final output passed to step_callback. 
    """
    def __init__(self, action_shape, step_callback=print, reload_callback = None, reload_key='`'):
        assert isinstance(action_shape, (list, tuple, np.ndarray))
        if isinstance(action_shape, np.ndarray) == False:
            self.action_shape = np.array(action_shape, copy=False)
        self.ndim = self.action_shape.ndim
        self.r_key = reload_key
        self.outputs = []
        self.inputs = []
        self.step_callback = step_callback
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("Controller")
        if reload_callback is not None:
            self.reload_callback = reload_callback
        else:
            self.reload_callback = lambda : print("Reload Callback Disabled!")

    def loadController(self, inputs, outputs):
        assert isinstance(inputs, (tuple, list))
        self.inputs = inputs
        assert isinstance(outputs[0][0], (float, int))
        assert isinstance(outputs[0][1], (list, tuple))
        self.outputs = outputs

    def makeController(self):
        print("\n", "*"*25)
        print("Manual Controller Setup")
        print("\n", "*"*25)
        print("\nPress ENTER to confirm your inputs or BACKSPACE to rewrite them.")
        self.__pycollect_inputs(self.inputs)
        size = len(self.inputs)
        self.__pycollect_outputs(self.ndim, self.inputs, self.outputs, size)
        print("\nSuccessfully initialized controller: ")
        print("\nInputs: ", self.inputs)
        print("\nOutputs: ", self.outputs)

    def play(self, msDelay):
        # Dictionary to store currently pressed keys and their outputs
        pressed_keys = {}

        # Create a list with the same length as outputs and values set to 0
        while True:
            pygame.time.wait(msDelay)
            action = np.zeros(shape=self.action_shape)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.unicode in self.inputs:
                        input_index = self.inputs.index(event.unicode)
                        output = self.outputs[input_index]
                        pressed_keys[event.unicode] = output[0]
                    elif event.unicode == self.r_key:
                        self.reload_callback()
                if event.type == pygame.KEYUP:
                    if event.unicode in pressed_keys:
                        del pressed_keys[event.unicode]

            # Update the output_list with the currently pressed keys and their outputs
            for key, value in pressed_keys.items():
                for i, inp in enumerate(self.inputs):
                    if key == inp:
                        action[self.outputs[i][1]] = self.outputs[i][0]

            self.step_callback(action)
    
    @staticmethod
    def __pycollect_outputs(dim, inputs, outputs, size):
        
        while True:
            print("\n", "*"*25)
            #condition
            if len(outputs) == size:
                break
            # wait for user input
            print("\nCurrent controller: ")
            print(inputs)
            print(outputs)
            out = []
            cur_val = ''
            cur_b = True
            prompt = "\nEnter the output value for your key-input: '" + inputs[len(outputs)] + "'"
            print(prompt)
            message = "\nCurrent output: "
            print(message, end="")
            while cur_b:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        keyname = pygame.key.name(event.key)
                        if keyname.isdigit() or event.unicode == '.' or event.unicode == '-':
                            cur_val += event.unicode
                            message += cur_val[-1]
                            if (len(cur_val) > 4):
                                print("\nPress ENTER to confirm or BACKSPACE to start-over")
                                print(message)
                                continue
                            print(cur_val[-1], end="", flush=True)
                        elif event.key == pygame.K_RETURN:
                            try: 
                                out.append(float(cur_val))
                            except:
                                print("\nOutput must be int or float!")
                                cur_val = ''
                                message = "\nCurrent output: "
                                print(message, end="")
                                continue
                            else:
                                cur_b = False
                        elif event.key == pygame.K_BACKSPACE:
                            print("\n<back>")
                            cur_val = ''
                            message = "\nCurrent output: "
                            print(prompt)
                            print(message, end="")
            
            cur_b = True
            indices = []
            double_backspace = False
            print("\nEnter the output's corresponding index for the final output: ")
            while cur_b:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        keyname = pygame.key.name(event.key)
                        if keyname.isdigit():
                            print("\nCurrent output: ", cur_val)
                            double_backspace = False
                            if len(indices) == dim:
                                print("\nindex is already matching with ndim! Press ENTER to continue or BACKSPACE to start-over:")
                                continue
                            indices.append(int(event.unicode))
                            print("with index: ", indices)
                        elif event.key == pygame.K_RETURN and indices:
                            double_backspace = False
                            cur_b = False
                            out.append(indices)
                        elif event.key == pygame.K_BACKSPACE:
                            print("<back>")
                            if double_backspace:
                                print("   <back>")
                                cur_b = False
                                continue
                            double_backspace = True
                            indices = []
                            print("\nCurrent output: ", cur_val)
                            print("with index: ", indices)

            if double_backspace:
                continue
            outputs.append(out)

    @staticmethod
    def __pycollect_inputs(inputs):
            print("\nEnter input values to map onto the controller: ")
            print(inputs)
            while True:
                # wait for user input
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            # user pressed Enter, break out of the loop
                            return
                        elif event.key == pygame.K_BACKSPACE:
                            print("<back>")
                            inputs.clear()
                            print(inputs)
                        elif event.key != pygame.K_RETURN:
                            # add input value to the inputs list
                            inputs.append(event.unicode)
                            print(inputs) 

class CustomControllerCallback:
    """
    Basic controller callback example.
    HOW TO USE: construct a controller instance
    using a reference or the custom_ControllerCallback object as the step_callback argument and
    a reference of custom_ControllerCallback.reload() as reload_callback argument.
    """
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def __call__(self, x):
        _, _, done, _, _ = self.env.step(x)
        self.env.render()
        if done:
            self.env.reset()
    def reload(self):
        self.env.reset()

#TODO environment that verifies the contents of exp_source and or buffer


class VectorSampler:
    """
    A class that creates vector samples which are stored in a training and testing batch.
    The user can then fetch a minibatch of samples using fetch_train or fetch_test. 
    """
    def __init__(self, _range , size : int, train_test_ratio : float):
        _range = np.array(_range, dtype=np.float32)  
        assert(train_test_ratio <= 1 and train_test_ratio >= 0)

        self.train_size = round(size * train_test_ratio)
        self.test_size = size - self.train_size
        self.train_test_ratio = train_test_ratio
        self.training_batch = None
        self.testing_batch = None

        for sub_range in _range:
            if(len(sub_range) != 2):
                wrong_dimension_w = "Parameter at position 1 with length of %d must instead have length %d" % (len(_range), 2)
                warnings.warn(wrong_dimension_w)
            
            scale = max(sub_range) - min(sub_range)
            batch = np.random.random(size) * scale + min(sub_range)
            
            indices = list(range(size))
            random.shuffle(indices)
            if self.training_batch is None:
                self.training_batch = batch[indices[:self.train_size]].reshape((-1, 1))
                self.testing_batch = batch[indices[self.train_size:]].reshape((-1, 1))     
            else:
                self.training_batch = np.concatenate((self.training_batch, batch[indices[:self.train_size]].reshape((-1, 1))), axis=1)
                self.testing_batch = np.concatenate((self.testing_batch, batch[indices[self.train_size:]].reshape((-1, 1))), axis=1)

    def fetch(self, sample_size : int, noise_scale = 0, is_test = False):
        if is_test:
            if(sample_size >= self.test_size):
                output = self.testing_batch[:]
            indices = np.random.randint(0, self.test_size-1 , sample_size)
            output = self.testing_batch[indices]
        else:
            if(sample_size >= self.train_size):
                output = self.training_batch[:]
            indices = np.random.randint(0, self.train_size-1 , sample_size)
            output = self.training_batch[indices]
        
        if noise_scale != 0:
            noise_matrix = np.random.random_sample(size=output.shape) - 0.5
            noise_matrix *= noise_scale
            output = output + noise_matrix

        return output        

    def get_sizeDict(self):
        if self.size_dict is None:
            self.size_dict = {"training_size" : self.train_size, "testing_size" : self.test_size, "batch_size" : self.test_size + self.train_size}
        return self.size_dict
    
    def __len__(self):
        size_d = self.get_sizeDict()
        return size_d["batch_size"]
