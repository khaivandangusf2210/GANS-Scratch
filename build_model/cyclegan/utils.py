import random
import torch
from torch.autograd import Variable


class ReplayBuffer:
    def __init__(self, max_size: int = 50):
        if max_size <= 0:
            raise ValueError("Buffer size must be greater than 0.")
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor) -> Variable:
        data_len = len(self.data)
        to_return = []

        for element in data:
            element = element.unsqueeze(0)
            if data_len < self.max_size:
                self.data.append(element)
                to_return.append(element)
                data_len += 1
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[idx].clone())
                    self.data[idx] = element
                else:
                    to_return.append(element)

        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs: int, offset: int, decay_start_epoch: int):
        if (n_epochs - decay_start_epoch) <= 0:
            raise ValueError("Decay must start before the training session ends!")
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int) -> float:
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
