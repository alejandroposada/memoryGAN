import torch.nn as nn
import torch
from models.config import config

class gan_super(nn.Module):
    def __init__(self, dataset, is_cuda):
        """
        """
        super(gan_super, self).__init__()
        config_vars = config(dataset)
        self.lamb = config_vars.lamb
        self.key_dim = config_vars.key_dim
        self.mem_size = config_vars.mem_size
        self.z_dim = config_vars.z_dim
        self.c_dim = config_vars.c_dim
        self.fc_dim = config_vars.fc_dim
        self.f_dim = config_vars.f_dim
        self.lamb = config_vars.lamb
        self.key_dim = config_vars.key_dim
        self.mem_size = config_vars.mem_size
        self.c_dim = config_vars.c_dim
        self.fc_dim = config_vars.fc_dim
        self.f_dim = config_vars.f_dim
        self.choose_k = config_vars.choose_k
        self.z_dim = config_vars.z_dim
        self.is_cuda = is_cuda

    def discriminate(self, x, label):
        return self.dmn.forward(x)

    def generate(self, z):
        return self.mcgn.forward(z)

    def Dloss(self, true_output, fake_output):
        return -torch.mean(torch.log(true_output)) - torch.mean(torch.log(1 - fake_output))

    def Gloss(self, fake_output):
        return -torch.mean(torch.log(fake_output))