#EXPERIENCE REPLAY
from collections import deque
import random

class ReplayMemory():
    #create FIFO queue - experiance replay

    def __init__(self , maxlen , seed = None):
        self.memory = deque([] , maxlen)

    def append(self , new_experience):
        self.memory.append(new_experience)

    def sample(self , sample_size):
        return random.sample(self.memory , sample_size)


    #function for current buffer size
    def __len__(self):
        return len(self.memory)
