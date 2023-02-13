import time
import math
class timer:
    def __init__(self):
        self.s = None

    def start(self):
        self.s=time.time()
    
    def stop(self):
        if self.s == None:
            print("timer hasn't started yet...")
            return 0
        t = time.time() - self.s
        print("time elapsed: %.3f" % t)
        raise TimeoutError


class calc_FPS:
    def __init__(self, treshold=10):
        self.delta_frames = 0
        self.last = 0
        self.t1 = 10
        self.t2 = None
        self.treshold = treshold
    
    def __call__(self, frames):
            self.delta_frames = frames - self.last
            self.t2 = time.time()
            elapsed = self.t2 - self.t1
            FPS = self.delta_frames / elapsed

            if frames % self.treshold == 0:
                self.last = frames
                self.t1 = self.t2
            FPS = math.ceil(FPS)
            return FPS

    

