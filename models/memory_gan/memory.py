import torch.nn as nn
import torch.nn.functional as F
import torch

class memory(nn.Module):
    """DISCRIMINATIVE MEMORY NETWORK
	"""
    def __init__(self, key_dim, memory_size, choose_k, cuda):
        """
        """
        super(memory, self).__init__()
        self.beta = 1e-8
        self.key_dim = key_dim
        self.memory_size = memory_size
        self.choose_k = min(choose_k, self.memory_size)

        if cuda:
            self.memory_key = torch.zeros([self.memory_size, self.key_dim]).cuda()
            self.memory_values = (torch.ones([self.memory_size]) / self.memory_size).cuda()
            self.memory_age = torch.zeros([self.memory_size]).cuda()
            self.memory_hist = torch.FloatTensor(size=[self.memory_size]).cuda()
            self.memory_hist[:] = 1e-5
        else:
            self.memory_key = torch.zeros([self.memory_size, self.key_dim])              # K in 3.1 of paper
            torch.ones([self.memory_size]) / self.memory_size                            # v in 3.1 of paper
            self.memory_age = torch.zeros([self.memory_size])                            # a in 3.1 of paper
            self.memory_hist = torch.FloatTensor(size=[self.memory_size])                # h in 3.1 of paper
            self.memory_hist[:] = 1e-5

    def query(self, q):
        #  compute P(x|c)
        post_x_given_c = torch.matmul(self.memory_key, q)

        #  compute P(c)

        #  compute P(c|x) from eq 1 of paper

        #  compute P(y|x) = \sum_i P(c=i|x)v_i from eq 4 of paper


