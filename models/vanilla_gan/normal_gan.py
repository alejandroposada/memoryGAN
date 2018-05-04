from models.gan_super import gan_super
from models.vanilla_gan.normal_disc import disc
from models.vanilla_gan.normal_gen import gen

#loss = torch.nn.CrossEntropyLoss()


class GAN(gan_super):
    """GAN
	"""
    def __init__(self, dataset, is_cuda):
        """
        """
        super().__init__(dataset, is_cuda)
        self.dmn = disc(f_dim=self.f_dim, fc_dim=self.fc_dim)
        self.mcgn = gen(f_dim=self.f_dim, fc_dim=self.fc_dim,
                        z_dim=self.z_dim, c_dim=self.c_dim)
