import torch.nn as nn

from models.config import config
from models.dmn import dmn
from models.mcgn import mcgn


class GAN(nn.Module):
    """GAN
	"""
    def __init__(self, dataset):
        """
        """
        super(GAN, self).__init__()
        config_vars = config(dataset)
        self.dmn = dmn(f_dim=config_vars.f_dim, fc_dim=config_vars.fc_dim)
        self.mcgn = mcgn(f_dim=config_vars.f_dim, fc_dim=config_vars.fc_dim,
                         z_dim=config_vars.z_dim, c_dim=config_vars.c_dim)
