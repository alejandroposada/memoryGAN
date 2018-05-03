import torch.nn as nn
import torch
import torch.nn.functional as F

from models.config import config
from models.memory_gan.mcgn import mcgn
from models.memory_gan.dmn import dmn
from models.memory_gan.memory import memory


class GAN(nn.Module):
    """GAN
	"""
    def __init__(self, dataset, is_cuda):
        """
        """
        super(GAN, self).__init__()
        config_vars = config(dataset)
        self.lamb = config_vars.lamb
        self.key_dim = config_vars.key_dim
        self.mem_size = config_vars.mem_size
        self.c_dim = config_vars.c_dim
        self.fc_dim = config_vars.fc_dim
        self.f_dim = config_vars.f_dim
        self.choose_k = config_vars.choose_k
        self.z_dim = config_vars.z_dim
        self.is_cuda = is_cuda

        self.dmn = dmn(f_dim=self.f_dim, fc_dim=self.fc_dim, key_dim=self.key_dim)
        self.mcgn = mcgn(f_dim=self.f_dim, fc_dim=self.fc_dim, z_dim=self.z_dim,
                         c_dim=self.c_dim, key_dim=self.key_dim)
        self.memory = memory(key_dim=self.key_dim, memory_size=self.mem_size, choose_k=self.choose_k, cuda=is_cuda)

    def discriminate(self, x, label):
        q = self.dmn.forward(x)  # get query vec
        qn = torch.norm(q, p=2, dim=1).detach()  # l2 normalize
        q = q.div(torch.transpose(qn.expand(256, q.size(0)), 1, 0))
        self.q = q
        post_prob = self.memory.query(q)
        self.memory.update_memory(q, label)

        return post_prob

    def generate(self, z, batch_size):
        if self.is_cuda:
            key = self.memory.sample_key(batch_size).cuda()
        else:
            key = self.memory.sample_key(batch_size)

        self.key = key
        gen_input = torch.cat((z, key), dim=1)
        fake_batch = self.mcgn.forward(gen_input)
        return fake_batch

    def Dloss(self, true_output, fake_output):
        gamma = 0.000002
        I_hat = -torch.mean(F.cosine_similarity(self.key, self.q))
        return -torch.mean(torch.log(true_output)) - torch.mean(torch.log(1 - fake_output)) + gamma*I_hat

    def Gloss(self, fake_output):
        gamma = 0.000002
        I_hat = -torch.mean(F.cosine_similarity(self.key, self.q))
        return torch.mean(torch.log(1 - fake_output)) + gamma*I_hat
