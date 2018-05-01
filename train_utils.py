from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dset
import torch.utils as utils
import torch
import torch.optim as optim

from models.gan import GAN

def load_dataset(dataset, batch_size):
    try:
        assert dataset in ['fashion-mnist', 'mnist', 'cifar10']
    except AssertionError:
        print('you need to use either fashion-mnist mnist or cifar10!')
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        root = './data/mnist'
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
    elif dataset == 'fashion-mnist':
        root = './data/fashion-mnist'
        train_set = dset.FashionMNIST(root=root, train=True, download=True, transform=trans)
        test_set = dset.FashionMNIST(root=root, train=False, download=True, transform=trans)
    elif dataset == 'cifar10':
        root = './data/cifar10'
        train_set = dset.CIFAR10(root=root, train=True, download=True, transform=trans)
        test_set = dset.CIFAR10(root=root, train=False, download=True, transform=trans)


    train_loader = utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_loader = utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)

    return train_loader, test_loader


def load_model(model_file, cuda):
    #return total_examples, fixed_noise, gen_losses, disc_losses, gen_loss_per_epoch, \
    #disc_loss_per_epoch, prev_epoch, gen, disc, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer
    pass

def create_new_model(dataset, cuda, learning_rate, beta_0, beta_1):
    gan = GAN(dataset)
    fixed_noise = torch.randn(9, gan.mcgn.z_dim)

    if cuda:
        gan.cuda()
        fixed_noise = fixed_noise.cuda()

    gen_losses = []
    disc_losses = []
    gen_loss_per_epoch = []
    disc_loss_per_epoch = []
    total_examples = 0
    prev_epoch = 0

    # Adam optimizer
    disc_optimizer = optim.Adam(gan.dmn.parameters(), lr=learning_rate, betas=(beta_0, beta_1), eps=1e-8)
    gen_optimizer = optim.Adam(gan.mcgn.parameters(), lr=learning_rate, betas=(beta_0, beta_1), eps=1e-8)

    return total_examples, fixed_noise, gen_losses, disc_losses, gen_loss_per_epoch, \
           disc_loss_per_epoch, prev_epoch, gan, disc_optimizer, gen_optimizer
