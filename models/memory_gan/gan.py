import torch
import torch.nn.functional as F

from models.memory_gan.mcgn import mcgn
from models.memory_gan.dmn import dmn
from models.memory_gan.memory import memory
from models.gan_super import gan_super
from helpers import normalize

class GAN(gan_super):
    """GAN
	"""
    def __init__(self, dataset, is_cuda, use_mcgn=True):
        """
        """
        super().__init__(dataset, is_cuda)

        self.dmn = dmn(f_dim=self.f_dim, fc_dim=self.fc_dim, key_dim=self.key_dim)
        self.mcgn = mcgn(f_dim=self.f_dim, fc_dim=self.fc_dim, z_dim=self.z_dim,
                         c_dim=self.c_dim, key_dim=self.key_dim, use_mcgn=use_mcgn)
        self.memory = memory(key_dim=self.key_dim, memory_size=self.mem_size, choose_k=self.choose_k,
                             is_cuda=is_cuda, alpha=self.alpha, num_steps=self.num_steps)
        self.use_mcgn = use_mcgn
        print(self.use_mcgn)

    def discriminate(self, x, label):
        q = self.dmn.forward(x)  # get query vec
        qn = normalize(q, 1)
        self.q = qn
        post_prob = self.memory.query(qn)
        return post_prob, qn

    def generate(self, z):
        if self.is_cuda:
            key = self.memory.sample_key(z.shape[0]).cuda()
        else:
            key = self.memory.sample_key(z.shape[0])

        self.key = key
        if self.use_mcgn:
            gen_input = torch.cat((z, key), dim=1)
        else:
            gen_input = z
        fake_batch = self.mcgn.forward(gen_input)

        return fake_batch

    def Dloss(self, true_output, fake_output):
        # I_hat = -torch.mean(F.cosine_similarity(self.key, self.q))
        if self.use_mcgn:
            I_hat = F.mse_loss(self.key, self.q)
        else:
            I_hat = 0.0
        return -torch.mean(torch.log(true_output)) - torch.mean(torch.log(1 - fake_output)) + self.lamb*I_hat

    def Gloss(self, fake_output):
        # I_hat = -torch.mean(F.cosine_similarity(self.key, self.q))
        if self.use_mcgn:
            I_hat = F.mse_loss(self.key, self.q)
        else:
            I_hat = 0.0
        return -torch.mean(torch.log(fake_output)) + self.lamb*I_hat
