import time

class FPScounter:
    #should not use both functions simultaneously
    #Note: reset function is never necessary when using only one function
    def __init__(self, treshold=1000):
        self.treshold = treshold
        self.count = 0
        self.t1 = None
    def step(self):
        if self.count < 1:
            self.t1 = time.time()
        self.count += 1
        if self.count > self.treshold:
            tf = time.time() - self.t1
            if tf > 0:
                fps = self.count / tf
                self.count = 0
                print(fps)
            else:
                print("Treshold too low, consider increasing it.")
   
    def __call__(self, frame):
        if self.count == 0:
            self.t1 = time.time()
            self.count = 1
        
        score = frame/self.count
        if score > self.treshold:
            t2 = time.time()
            tf = t2 - self.t1
            if tf != 0:
                fps = score / tf
                self.count += 1
                self.t1 = t2
                print(fps)
            else:
                print("Treshold too low, consider increasing it.")
    
    def reset(self):
        self.count = 0
        self.t1 = None

