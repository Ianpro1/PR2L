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


'''if __name__ == '__main__':
    running = True
    inconn, outconn = mp.Pipe()

    def sendimg(img):
        inconn.send(img)

    p1 = mp.Process(target=init_display, args=(outconn, 420, 420))
    p1.start()
    time.sleep(3)
    while range(1000):
        sendimg(np.random.randint(0, 255, size=(84,84,3), dtype=np.uint8))
        #print("sending...")
    p1.join()
    print("done")'''