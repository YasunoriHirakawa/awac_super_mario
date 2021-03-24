from random import sample
import torch


class ReplayMemory:
    def __init__(self, max_size):
        # deque object that we've used for 'episodic_memory' is not suitable for random sampling
        # here, we instead use a fix-size array to implement 'buffer'
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = sample(range(self.size), batch_size)
        state, action, reward, next_state, done = zip(*[self.buffer[index] for index in indices])
        cat_func = lambda x: torch.cat(x).reshape(len(x), *x[0].shape) 
        
        return cat_func(state), cat_func(action), cat_func(reward), cat_func(next_state), cat_func(done)

    def __len__(self):
        return self.size

    def reset(self):
        self.buffer = [None] * self.max_size
        self.index = 0
        self.size = 0
