import multiprocessing as mp
import numpy as np
import time
import pygame

def init_display(conn, width, height):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    while True:
        #print("receiving...")
        frame = conn.recv()
        #print(frame.shape)
        if frame is None:
            break
        screen.fill((255, 255, 255))
        frame = np.array(frame, dtype=np.float32).transpose(1,0,2)
        frame = np.repeat(frame, height // 210, axis=0)
        frame = np.repeat(frame, width // 160, axis=1)
        frame = (frame).astype(np.uint8)
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


def init_bar(data_shape, height, bar_width, conn, pool=1):
    
    if len(data_shape) != 1:
        raise MemoryError
    pygame.init()
    dlen = data_shape[-1]
    screen = pygame.display.set_mode((dlen * bar_width, height))
    while True:
        
        data = conn.recv()
        assert isinstance(data, (np.ndarray))
        if data is None:
            break
        screen.fill((255, 255, 255))

        frame = np.full(shape=(dlen, height), fill_value=255, dtype=np.uint8)
        dmin = data.min()
        dmax = data.max()
        drange = dmax - dmin
        if drange !=0:
            scaled_v = (data - dmin )/ drange
            scaled_v = np.floor(scaled_v * height)
            scaled_v = scaled_v.astype(np.uint8)
        else:
            scaled_v = data
        
        for i, x in enumerate(scaled_v):
            frame[i][:x] = 0
        frame = np.repeat(frame, bar_width, 0)
        frame = (frame).astype(np.uint8)
        frame = np.array([frame] * 3).transpose(1,2,0)
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
                     
