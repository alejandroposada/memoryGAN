import torch
import torch.nn as nn
import torch.nn.functional as F


class dmn(nn.Module):
    """DISCRIMINATIVE MEMORY NETWORK
	"""
    def __init__(self, f_dim, fc_dim):
        """
        """
        super(dmn, self).__init__()
        self.f_dim = f_dim
        self.fc_dim = fc_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.f_dim, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.f_dim, out_channels=2*self.f_dim, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2*self.f_dim*25, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, 1)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.fc1(h), 0.2)
        y = self.fc2(h)
        return y
