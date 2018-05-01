import torch.nn as nn

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


        self.dmn = disc(f_dim=config_vars.f_dim, fc_dim=config_vars.fc_dim)
        self.mcgn = gen(f_dim=config_vars.f_dim, fc_dim=config_vars.fc_dim,
                        z_dim=config_vars.z_dim, c_dim=config_vars.c_dim)
