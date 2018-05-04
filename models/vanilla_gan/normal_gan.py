import torch.nn as nn
import torch

from models.config import config
from models.vanilla_gan.normal_disc import disc
from models.vanilla_gan.normal_gen import gen


class GAN(nn.Module):
    """GAN
	"""
    def __init__(self, dataset):
        """
        """
        super(GAN, self).__init__()
        config_vars = config(dataset)
        self.lamb = config_vars.lamb
        self.key_dim = config_vars.key_dim
        self.mem_size = config_vars.mem_size
        self.z_dim = config_vars.z_dim
        self.c_dim = config_vars.c_dim
        self.fc_dim = config_vars.fc_dim
        self.f_dim = config_vars.f_dim


        self.dmn = disc(f_dim=self.f_dim, fc_dim=self.fc_dim)
        self.mcgn = gen(f_dim=self.f_dim, fc_dim=self.fc_dim,
                        z_dim=self.z_dim, c_dim=self.c_dim)

    def discriminate(self, x, label):
        return self.dmn.forward(x)

    def generate(self, z, batch_size):
        return self.mcgn.forward(z)

    def Dloss(self, true_output, fake_output):
        return -torch.mean(torch.log(true_output)) - torch.mean(torch.log(1 - fake_output))

    def Gloss(self, fake_output):
        return - torch.mean(torch.log(fake_output))
