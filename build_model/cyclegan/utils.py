import random
from torch.autograd import Variable
import torch
import numpy as np
from typing import List, Union


class ReplayBuffer:

    def __init__(self, max_size: int = 50):
        assert max_size > 0, "Buffer size must be greater than 0."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor) -> Variable:
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs: int, offset: int, decay_start_epoch: int):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int) -> float:
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
