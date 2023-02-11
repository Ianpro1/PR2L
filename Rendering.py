import multiprocessing as mp
import pygame
import numpy as np
# Initialize the display using Pygame
def init_display(queue, width, height):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    while True:
        
        frame = queue.get()
        if frame is None:
            break
        screen.fill((255, 255, 255))
        frame = np.array(frame, dtype=np.float32)
        frame = np.repeat(frame, height // 84, axis=0)
        frame = np.repeat(frame, width // 84, axis=1)
        frame = (frame * 255).astype(np.uint8)
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


'''def main():
    width = 420
    height = 420
    queue = mp.Queue()
    display_process = mp.Process(target=init_display, args=(queue, width, height))
    display_process.start()

    # Generate and push random frames to the queue
    while 10:
        frame = np.random.rand(84, 84, 3).astype(np.float32)
        queue.put(frame)

    #queue.put(None)
    #display_process.join()'''
    

if __name__ == '__main__':
    width = 420
    height = 420
    queue = mp.Queue()
    display_process = mp.Process(target=init_display, args=(queue, width, height))
    display_process.start()
    import atari_agents
    queue.put(None)
    display_process.join()
