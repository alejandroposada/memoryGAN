# stdlib imports
import argparse
import os

# thirdparty imports
import torch
import torch.nn as nn
import numpy as np

# from project
from load_utils import load_dataset, create_new_model, load_model
from save_utils import save_learning_curve_epoch, save_all
from train_utils import train_step


def main():
    #  load datasets
    train_loader, test_loader = load_dataset(args.train_set, args.batch_size)

    # initialize model
    if args.model_file:
        try:
            total_examples, fixed_noise, gen_losses, disc_losses, gen_loss_per_epoch, \
            disc_loss_per_epoch, prev_epoch, gan, disc_optimizer, gen_optimizer, memory \
                = load_model(args.model_file, args.cuda, args.learning_rate, args.beta_0, args.beta_1)
            print('model loaded successfully! resuming from step {}'.format(prev_epoch))
            args.memory = memory   # prevents any contradictions during loading
        except:
            print('could not load model! creating new model...')
            args.model_file = None

    if not args.model_file:
        print('creating new model...')
        total_examples, fixed_noise, gen_losses, disc_losses, gen_loss_per_epoch, disc_loss_per_epoch, \
        prev_epoch, gan, disc_optimizer, gen_optimizer \
            = create_new_model(args.train_set, args.cuda, args.learning_rate, args.beta_0, args.beta_1, args.memory)

    # results save folder
    gen_images_dir = 'results/generated_images'
    train_summaries_dir = 'results/training_summaries'
    checkpoint_dir = 'results/checkpoints'
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir(gen_images_dir):
        os.mkdir(gen_images_dir)
    if not os.path.isdir(train_summaries_dir):
        os.mkdir(train_summaries_dir)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    np.random.seed(args.seed)  # reset training seed to ensure that batches remain the same between runs!

    try:
        for epoch in range(prev_epoch, args.n_epochs):

            disc_losses_epoch = []
            gen_losses_epoch = []
            for idx, (true_batch, _) in enumerate(train_loader):
                disc_train_loss, gen_train_loss, disc_true_accuracy, disc_fake_accuracy \
                        = train_step(gan=gan, batch_size=args.batch_size, label_smoothing=args.label_smoothing,
                                     is_cuda=args.cuda, true_batch=true_batch, grad_clip=args.grad_clip,
                                     disc_optimizer=disc_optimizer, gen_optimizer=gen_optimizer)

                if (total_examples != 0) and (total_examples % args.display_result_every == 0):
                    print('epoch {}: step {}/{} disc true acc: {:.4f} disc fake acc: {:.4f} '
                          'disc loss: {:.4f}, gen loss: {:.4f}'
                          .format(epoch+1, idx+1, len(train_loader), disc_true_accuracy, disc_fake_accuracy,
                                  disc_train_loss.item(), gen_train_loss.item()))

                # Checkpoint model
                total_examples += args.batch_size
                if (total_examples != 0) and (total_examples % args.checkpoint_interval == 0):

                    disc_losses.extend(disc_losses_epoch)
                    gen_losses.extend(gen_losses_epoch)
                    save_all(total_examples=total_examples, fixed_noise=fixed_noise, gan=gan,
                             disc_loss_per_epoch=disc_loss_per_epoch, gen_loss_per_epoch=gen_loss_per_epoch,
                             gen_losses=gen_losses, disc_losses=disc_losses, epoch=epoch,
                             checkpoint_dir=checkpoint_dir, is_cuda=args.cuda,
                             gen_images_dir=gen_images_dir, train_summaries_dir=train_summaries_dir,
                             disc_optimizer=disc_optimizer, gen_optimizer=gen_optimizer,
                             train_set=args.train_set, memory=args.memory)

                #  Collect information per epoch
                disc_losses_epoch.append(disc_train_loss.item())
                gen_losses_epoch.append(gen_train_loss.item())

            disc_loss_per_epoch.append(np.average(disc_losses_epoch))
            gen_loss_per_epoch.append(np.average(gen_losses_epoch))

            # Save epoch learning curve
            save_learning_curve_epoch(gen_losses=gen_loss_per_epoch, disc_losses=disc_loss_per_epoch,
                                      total_epochs=epoch+1, directory=train_summaries_dir)
            print("Saved learning curves!")

            print('epoch {}/{} disc loss: {:.4f}, gen loss: {:.4f}'
                  .format(epoch+1, args.n_epochs, np.array(disc_losses_epoch).mean(), np.array(gen_losses_epoch).mean()))

            disc_losses.extend(disc_losses_epoch)
            gen_losses.extend(gen_losses_epoch)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save_all(total_examples=total_examples, fixed_noise=fixed_noise, gan=gan,
                 disc_loss_per_epoch=disc_loss_per_epoch, gen_loss_per_epoch=gen_loss_per_epoch,
                 gen_losses=gen_losses, disc_losses=disc_losses, epoch=epoch,
                 checkpoint_dir=checkpoint_dir, is_cuda=args.cuda, gen_images_dir=gen_images_dir,
                 train_summaries_dir=train_summaries_dir, disc_optimizer=disc_optimizer, gen_optimizer=gen_optimizer,
                 train_set=args.train_set, memory=args.memory)


if __name__ == '__main__':

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, default='fashion-mnist')
    argparser.add_argument('--learning_rate', type=float, default=0.0002)
    argparser.add_argument('--n_epochs', type=int, default=30)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--beta_0', type=float, default=0.5)
    argparser.add_argument('--beta_1', type=float, default=0.999)
    argparser.add_argument('--model_file', type=str, default=None)
    #  --model_file results/checkpoints/example-28928.model
    argparser.add_argument('--cuda', action='store_true', default=True)
    argparser.add_argument('--display_result_every', type=int, default=640)   # 640
    argparser.add_argument('--checkpoint_interval', type=int, default=32000)  # 32000
    argparser.add_argument('--seed', type=int, default=1024)
    argparser.add_argument('--label_smoothing', action='store_true', default=False)
    argparser.add_argument('--memory', action='store_true', default=False)  # use memory?
    argparser.add_argument('--grad_clip', type=int, default=10)
    args = argparser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    main()
