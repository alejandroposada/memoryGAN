from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
import torch
import time


def show_image(images, dataset, is_cuda):
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                   transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                        std=[1., 1., 1.]),
                                   transforms.ToPILImage()
                                   ])
    batch = images[:9]  # restrict size to 3x3 grid

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


def timer(fun, **kwargs):
    start = time.time()
    fun(**kwargs)
    end = time.time()
    print('time elapsed: {:.3f}'.format(end - start))


def get_grads(model):
    return [p.grad for p in model.parameters()]

def normalize(matrix, dim):
    return matrix.div(matrix.norm(dim=dim, keepdim=True))
    #return torch.transpose(torch.transpose(matrix, 1, 0)/matrix.norm(dim=1), 1, 0)