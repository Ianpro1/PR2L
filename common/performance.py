import time

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

