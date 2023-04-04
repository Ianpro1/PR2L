import pygame
import sys
import numpy as np
import mjct

class LinearController:
    def __init__(self, env, action_shape, ndim=1):
        assert isinstance(ndim, int)
        self.env = env
        self.ndim = ndim
        self.outputs = []
        self.inputs = []
        self.action_shape = action_shape
        self.newPyController()

    def newPyController(self):
        self.__collect_inputs(self.inputs)
        size = len(self.inputs)
        self.__collect_outputs(self.ndim, self.inputs, self.outputs, size)
        print("\nSuccessfully initialized controller: ")
        print("\nInputs: ", self.inputs)
        print("\nOutputs: ", self.outputs)
    
    def play(self, delay):
        #self.env.reset()
        pygame.init()
        screen = pygame.display.set_mode((400, 400))
        font = pygame.font.Font(None, 36)
        pygame.display.set_caption("Key press handler")

        # Dictionary to store currently pressed keys and their outputs
        pressed_keys = {}

        # Create a list with the same length as outputs and values set to 0
        while True:
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
                if event.type == pygame.KEYUP:
                    if event.unicode in pressed_keys:
                        del pressed_keys[event.unicode]

            # Update the output_list with the currently pressed keys and their outputs
            for key, value in pressed_keys.items():
                for i, inp in enumerate(self.inputs):
                    if key == inp:
                        action[self.outputs[i][1]] = self.outputs[i][0]

            """# Do something...
            pygame.time.wait(delay)
            self.env.render()
            _, _, done, _, _ = self.env.step(np.array(output_list, copy=False))
            if done:
                self.env.reset()"""
            print(action)
    
    @staticmethod
    def __collect_outputs(dim, inputs, outputs, size):
        while True:
            #condition
            if len(outputs) == size:
                break

            # wait for user input
            print(inputs)
            print(outputs)
            
            out = []
            while True:
                output_value = input("Enter an output value (or press Enter to stop): ")
                if output_value == "":
                    break
                else:
                    try:
                        float(output_value)
                    except:
                        print("The output value must be an integer or float!\n")
                        continue
                break

            if output_value == "":
                # user pressed Enter, break out of the loop
                diff = len(outputs) - size
                if diff != 0:
                    print("output_shape does not match input_shape! missing " + str(np.absolute(diff)) + " outputs...\n")
                    continue
                else: break
            else:
                # add output value to the outputs list
                value = float(output_value)
            
            out.append(value)
            shape = []

            if dim == 0:
                shape.append(0)
            else:
                for _ in range(dim):
                    print(out)
                    while True:
                        index_value = input("Enter an index integer value: ")
                        try:
                            shape.append(int(index_value))
                        except:
                            print("Not an integer...")
                            continue
                        else:
                            break
            
            out.append(shape)
            outputs.append(out)

    @staticmethod
    def __collect_inputs(inputs):
        while True:
            # wait for user input
            print(inputs)
            input_value = input("Enter an input value (or press Enter to start the loop): ")
            if input_value == "":
                # user pressed Enter, break out of the loop
                break
            else:
                # add input value to the inputs list
                inputs.append(input_value)


#env = mjct.make("TosserCPP", True, 0.02, 100)
ctrl = LinearController(None, (2,))
ctrl.play(1)