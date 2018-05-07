import torch
import torch.nn as nn
import torch.nn.functional as F


class dmn(nn.Module):
    """DISCRIMINATIVE MEMORY NETWORK
	"""
    def __init__(self, f_dim, fc_dim, key_dim):
        """
        """
        super(dmn, self).__init__()
        self.f_dim = f_dim
        self.fc_dim = fc_dim
        self.key_dim = key_dim

        #  discriminative network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.f_dim, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.f_dim, out_channels=2*self.f_dim, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(2*self.f_dim)
        self.conv3 = nn.Conv2d(in_channels=self.f_dim*2, out_channels=4 * self.f_dim, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(4 * self.f_dim)
        self.conv4 = nn.Conv2d(in_channels=self.f_dim*4, out_channels=self.key_dim, kernel_size=1, stride=1)

        self.weight_init(mean=0, std=0.02)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.bn1(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.bn2(self.conv3(h)), 0.2)
        q = F.leaky_relu(self.conv4(h), 0.2)
        q = q.view(q.size(0), -1)
        return q

    def weight_init(self, mean=0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
