import torch
import torch.nn.functional as F

from models.memory_gan.mcgn import mcgn
from models.memory_gan.dmn import dmn
from models.memory_gan.memory import memory
from models.gan_super import gan_super

class GAN(gan_super):
    """GAN
	"""
    def __init__(self, dataset, is_cuda, use_EM):
        """
        """
        super().__init__(dataset, is_cuda)

        self.dmn = dmn(f_dim=self.f_dim, fc_dim=self.fc_dim, key_dim=self.key_dim)
        self.mcgn = mcgn(f_dim=self.f_dim, fc_dim=self.fc_dim, z_dim=self.z_dim,
                         c_dim=self.c_dim, key_dim=self.key_dim)
        self.memory = memory(key_dim=self.key_dim, memory_size=self.mem_size, choose_k=self.choose_k, is_cuda=is_cuda)
        self.use_EM = use_EM

    def discriminate(self, x, label):
        q = self.dmn.forward(x)  # get query vec
        qn = torch.norm(q, p=2, dim=1).detach()  # l2 normalize
        qn2 = q.div(torch.transpose(qn.expand(q.size(1), q.size(0)), 1, 0))
        self.q = qn2
        post_prob = self.memory.query(qn2)
        if self.use_EM:
            self.memory.Roger_update_memory(qn2, label)
        else:
            self.memory.update_memory_noEM(qn2, label)
        return post_prob

    def generate(self, z):
        if self.is_cuda:
            key = self.memory.sample_key(z.shape[0]).cuda()
        else:
            key = self.memory.sample_key(z.shape[0])

        self.key = key
        gen_input = torch.cat((z, key), dim=1)
        fake_batch = self.mcgn.forward(gen_input)

        return fake_batch

    def Dloss(self, true_output, fake_output):
        I_hat = -torch.mean(F.cosine_similarity(self.key, self.q))
        return -torch.mean(torch.log(true_output)) - torch.mean(torch.log(1 - fake_output)) + self.lamb*I_hat

    def Gloss(self, fake_output):
        I_hat = -torch.mean(F.cosine_similarity(self.key, self.q))
        return -torch.mean(torch.log(fake_output)) + self.lamb*I_hat
