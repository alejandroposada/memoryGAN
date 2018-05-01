# stdlib imports
import argparse
import os

# thirdparty imports
import torch
import torch.nn as nn
import numpy as np

# from project
from train_utils import load_dataset, load_model, create_new_model
from save_utils import save_learning_curve_epoch, save_all


def main():
    #  load datasets
    train_loader, test_loader = load_dataset(args.train_set, args.batch_size)

    # initialize model
    if args.model_file:
        try:
            total_examples, fixed_noise, gen_losses, disc_losses, gen_loss_per_epoch, \
            disc_loss_per_epoch, prev_epoch, gan, disc_optimizer, gen_optimizer \
                = load_model(args.model_file, args.cuda)
            print('model loaded successfully!')
        except:
            print('could not load model! creating new model...')
            args.model_file = None

    if not args.model_file:
        print('creating new model...')
        total_examples, fixed_noise, gen_losses, disc_losses, gen_loss_per_epoch, \
        disc_loss_per_epoch, prev_epoch, gan, disc_optimizer, gen_optimizer \
            = create_new_model(args.train_set, args.cuda, args.learning_rate, args.beta_0, args.beta_1)

    # Binary Cross Entropy loss
    BCE_loss = nn.BCEWithLogitsLoss()

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
                gan.dmn.zero_grad()

                #  hack 6 of https://github.com/soumith/ganhacks
                if args.label_smoothing:
                    true_target = torch.FloatTensor(args.batch_size).uniform_(0.7, 1.2)
                else:
                    true_target = torch.ones(args.batch_size)

                #  Sample  minibatch  of examples from data generating distribution
                if args.cuda:
                    true_batch = true_batch.cuda()
                    true_target = true_target.cuda()

                #  train discriminator on true data
                true_disc_result = gan.dmn.forward(true_batch)
                disc_train_loss_true = BCE_loss(true_disc_result.squeeze(), true_target)
                disc_train_loss_true.backward()
                torch.nn.utils.clip_grad_norm_(gan.dmn.parameters(), args.grad_clip)

                #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
                if args.label_smoothing:
                    fake_target = torch.FloatTensor(args.batch_size).uniform_(0, 0.3)
                else:
                    fake_target = torch.zeros(args.batch_size)

                if args.cuda:
                    z = torch.randn(args.batch_size, gan.mcgn.z_dim).cuda()
                    fake_target = fake_target.cuda()
                else:
                    z = torch.randn(args.batch_size, gan.mcgn.z_dim)

                #  train discriminator on fake data
                fake_batch = gan.mcgn.forward(z.view(-1, gan.mcgn.z_dim, 1, 1))
                fake_disc_result = gan.dmn.forward(fake_batch.detach())  # detach so gradients not computed for generator
                disc_train_loss_false = BCE_loss(fake_disc_result.squeeze(), fake_target)
                disc_train_loss_false.backward()
                torch.nn.utils.clip_grad_norm_(gan.dmn.parameters(), args.grad_clip)
                disc_optimizer.step()

                #  compute performance statistics
                disc_train_loss = disc_train_loss_true + disc_train_loss_false
                disc_losses_epoch.append(disc_train_loss.item())


                disc_fake_accuracy = 1 - torch.sum(fake_disc_result>0).item()/args.batch_size
                disc_true_accuracy = torch.sum(true_disc_result>0).item()/args.batch_size

                #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
                if args.label_smoothing:
                    true_target = torch.FloatTensor(args.batch_size).uniform_(0.7, 1.2)
                else:
                    true_target = torch.ones(args.batch_size)

                if args.cuda:
                    z = torch.randn(args.batch_size, gan.mcgn.z_dim).cuda()
                    true_target = true_target.cuda()
                else:
                    z = torch.rand(args.batch_size, gan.mcgn.z_dim)

                # train generator
                gan.mcgn.zero_grad()
                fake_batch = gan.mcgn.forward(z.view(-1, gan.mcgn.z_dim, 1, 1))
                disc_result = gan.dmn.forward(fake_batch)
                gen_train_loss = BCE_loss(disc_result.squeeze(), true_target)

                gen_train_loss.backward()
                torch.nn.utils.clip_grad_norm_(gan.mcgn.parameters(), args.grad_clip)
                gen_optimizer.step()
                gen_losses_epoch.append(gen_train_loss.item())

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
                             checkpoint_dir=checkpoint_dir, cuda=args.cuda,
                             gen_images_dir=gen_images_dir, train_summaries_dir=train_summaries_dir,
                             gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer, train_set=args.train_set)

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
                 checkpoint_dir=checkpoint_dir, cuda=args.cuda,
                 gen_images_dir=gen_images_dir, train_summaries_dir=train_summaries_dir,
                 gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer, train_set=args.train_set)


if __name__ == '__main__':

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, default='fashion-mnist')
    argparser.add_argument('--learning_rate', type=float, default=0.0002)
    argparser.add_argument('--n_epochs', type=int, default=30)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--beta_0', type=float, default=0.5)
    argparser.add_argument('--beta_1', type=float, default=0.999)
    argparser.add_argument('--model_file', type=str, default=None)
    argparser.add_argument('--cuda', action='store_true', default=True)
    argparser.add_argument('--display_result_every', type=int, default=640)   # 640
    argparser.add_argument('--checkpoint_interval', type=int, default=32000)  # 32000
    argparser.add_argument('--seed', type=int, default=1024)
    argparser.add_argument('--label_smoothing', action='store_true', default=True)
    argparser.add_argument('--grad_clip', type=int, default=10)
    args = argparser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    main()