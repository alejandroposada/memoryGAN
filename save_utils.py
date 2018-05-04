from matplotlib import pyplot as plt
import numpy as np
import matplotlib

import torch
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")
matplotlib.use('Agg')

def save_image_sample(dataset, batch, is_cuda, total_examples, directory):

    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                   transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                        std=[1., 1., 1.]),
                                   transforms.ToPILImage()
                                   ])

    f, axarr = plt.subplots(nrows=int(np.sqrt(len(batch))), ncols=int(np.sqrt(len(batch))))
    indx = 0
    for i in range(int(np.sqrt(len(batch)))):
        for j in range(int(np.sqrt(len(batch)))):
            if is_cuda:
                if dataset == 'cifar10':
                    axarr[i, j].imshow(invTrans(batch[indx].cpu()))
                else:  # b&w
                    axarr[i, j].imshow(np.asarray(invTrans(batch[indx].cpu())), cmap='gray')
                indx += 1
            else:
                if dataset == 'cifar10':
                    axarr[i, j].imshow(invTrans(batch[indx]))
                else:  # b&w
                    axarr[i, j].imshow(np.asarray(invTrans(batch[indx])), cmap='gray')
                indx += 1

            # Turn off tick labels
            axarr[i, j].axis('off')

    f.tight_layout()
    f.savefig(directory+'/gen_images_after_{}_examples'.format(total_examples))


def save_checkpoint(total_examples, gan, gen_losses, disc_losses, disc_loss_per_epoch, gen_loss_per_epoch,
                    fixed_noise, epoch, directory, disc_optimizer, gen_optimizer, train_set, memory):
    basename = directory+"/example-{}".format(total_examples)
    model_fname = basename + ".model"
    state = {
        'total_examples': total_examples,
        'gan_state_dict': gan.state_dict(),
        'gen_losses': gen_losses,
        'disc_losses': disc_losses,
        'disc_loss_per_epoch': disc_loss_per_epoch,
        'gen_loss_per_epoch': gen_loss_per_epoch,
        'fixed_noise': fixed_noise,
        'epoch': epoch,
        'disc_optimizer': disc_optimizer.state_dict(),
        'gen_optimizer': gen_optimizer.state_dict(),
        'train_set': train_set,
        'memory': memory
    }
    torch.save(state, model_fname)


def compute_model_stats(model):
    num_weights = 0
    for params in model.parameters():
        num_weights += params.numel()
    print('there are {} parameters'.format(num_weights))

    for params in model.parameters():
        print('avg weight value: {:.3f}'.format(params.mean().data[0]))


def save_learning_curve(gen_losses, disc_losses, total_examples, directory):
    plt.figure()
    #plt.title('GAN Learning Curves')
    plt.plot(gen_losses, color='red', label='Generator')
    plt.plot(disc_losses, color='blue', label='Discriminator')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(directory+'/learn_curves_after_{}_examples'.format(total_examples))


def save_learning_curve_epoch(gen_losses, disc_losses, total_epochs, directory):
    plt.figure()
    #plt.title('GAN Learning Curves')
    plt.plot(np.arange(len(gen_losses)) + 1, gen_losses, color='red', label='Generator')
    plt.plot(np.arange(len(disc_losses)) + 1, disc_losses, color='blue', label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(directory+'/learn_curves_after_{}_epochs'.format(total_epochs))


def save_all(total_examples, fixed_noise, gan, disc_loss_per_epoch, gen_loss_per_epoch, gen_losses,
             disc_losses, epoch, checkpoint_dir, is_cuda, gen_images_dir, train_summaries_dir,
             disc_optimizer, gen_optimizer, train_set, memory):

    save_checkpoint(total_examples=total_examples, fixed_noise=fixed_noise, gan=gan,
                    disc_loss_per_epoch=disc_loss_per_epoch,
                    gen_loss_per_epoch=gen_loss_per_epoch,
                    gen_losses=gen_losses, disc_losses=disc_losses, epoch=epoch, directory=checkpoint_dir,
                    disc_optimizer=disc_optimizer, gen_optimizer=gen_optimizer,
                    train_set=train_set, memory=memory)
    print("Checkpoint saved!")

    # sample images for inspection
    save_image_sample(batch=gan.generate(fixed_noise),
                      is_cuda=is_cuda, total_examples=total_examples, directory=gen_images_dir, dataset=train_set)
    print("Saved images!")

    # save learning curves for inspection
    save_learning_curve(gen_losses=gen_losses, disc_losses=disc_losses, total_examples=total_examples,
                        directory=train_summaries_dir)
    print("Saved learning curves!")

def save_verbose(h_per, k_per, a_per, v_per):
    pass