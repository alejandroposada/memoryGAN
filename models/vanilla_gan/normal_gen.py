import torch
import torch.nn as nn
import torch.nn.functional as F


class gen(nn.Module):
    """GENERATIVE MEMORY NETWORK
	"""
    def __init__(self, f_dim, fc_dim, z_dim, c_dim):
        """network: object which takes as input r_t and x_t and returns h_t
        """
        super(gen, self).__init__()
        self.f_dim = f_dim
        self.fc_dim = fc_dim
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.fc1 = nn.Linear(z_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, f_dim*2*5*5)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.f_dim*2,
                                        out_channels=self.f_dim*2, kernel_size=4, stride=2)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.f_dim*2, out_channels=self.c_dim,
                                        kernel_size=4, stride=2)

    def forward(self, h):
        x_new = F.leaky_relu(self.fc1(h), 0.2)
        x_new = F.leaky_relu(self.fc2(x_new), 0.2)
        x_new = x_new.view(h.shape[0], self.f_dim * 2, 5, 5)
        x_new = F.leaky_relu(self.conv1(x_new, output_size=[13, 13]), 0.2)
        x_new = F.leaky_relu(self.conv2(x_new, output_size=[28, 28]), 0.2)
        return F.sigmoid(x_new)
