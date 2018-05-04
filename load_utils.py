import torch
import torch.optim as optim
import torch.utils as utils
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vanilla_gan.normal_gan import GAN
from models.memory_gan.gan import GAN as MemGAN


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


def load_model(model_file, is_cuda, learning_rate, beta_0, beta_1):
    if is_cuda:
        from_before = torch.load(model_file)
    else:
        from_before = torch.load(model_file, map_location=lambda storage, loc: storage)

    total_examples = from_before['total_examples']
    gen_losses = from_before['gen_losses']
    disc_losses = from_before['disc_losses']
    gen_loss_per_epoch = from_before['gen_loss_per_epoch']
    disc_loss_per_epoch = from_before['disc_loss_per_epoch']
    gan_state_dict = from_before['gan_state_dict']
    fixed_noise = from_before['fixed_noise']
    prev_epoch = from_before['epoch']
    dataset = from_before['train_set']
    memory = from_before['memory']
    use_EM = from_before['use_EM']
    disc_optimizer_state_dict = from_before['disc_optimizer']
    gen_optimizer_state_dict = from_before['gen_optimizer']

    # load generator and discriminator
    if memory:
        gan = MemGAN(dataset, is_cuda, use_EM)
    else:
        gan = GAN(dataset)

    if is_cuda:
        gan.cuda()

    gan.load_state_dict(gan_state_dict)

    disc_optimizer = optim.Adam(gan.dmn.parameters(), lr=learning_rate, betas=(beta_0, beta_1), eps=1e-8)
    gen_optimizer = optim.Adam(gan.mcgn.parameters(), lr=learning_rate, betas=(beta_0, beta_1), eps=1e-8)
    disc_optimizer.load_state_dict(disc_optimizer_state_dict)
    gen_optimizer.load_state_dict(gen_optimizer_state_dict)

    return total_examples, fixed_noise, gen_losses, disc_losses, gen_loss_per_epoch, \
           disc_loss_per_epoch, prev_epoch, gan, disc_optimizer, gen_optimizer, memory, use_EM


def create_new_model(dataset, is_cuda, learning_rate, beta_0, beta_1, memory, use_EM):
    if memory:
        gan = MemGAN(dataset, is_cuda, use_EM)
    else:
        gan = GAN(dataset, is_cuda)

    fixed_noise = torch.randn(9, gan.z_dim)

    if is_cuda:
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
