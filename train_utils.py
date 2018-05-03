import torch


def train_step(gan, batch_size, label_smoothing, cuda, true_batch, loss,
               grad_clip, disc_optimizer, gen_optimizer, memory):
    gan.dmn.zero_grad()

    #  hack 6 of https://github.com/soumith/ganhacks
    if label_smoothing:
        true_target = torch.FloatTensor(batch_size).uniform_(0.7, 1.2)
    else:
        true_target = torch.ones(batch_size)

    # Sample  minibatch  of examples from data generating distribution
    if cuda:
        true_batch = true_batch.cuda()
        true_target = true_target.cuda()

    # train discriminator on true data
    true_disc_result = gan.discriminate(true_batch, true_target)

    disc_train_loss_true = loss(true_disc_result.squeeze(), true_target)
    disc_train_loss_true.backward()
    torch.nn.utils.clip_grad_norm_(gan.dmn.parameters(), grad_clip)

    #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
    if label_smoothing:
        fake_target = torch.FloatTensor(batch_size).uniform_(0, 0.3)
    else:
        fake_target = torch.zeros(batch_size)

    if cuda:
        z = torch.randn(batch_size, gan.z_dim).cuda()
        key = memory.sample_key(batch_size).cuda()
        fake_target = fake_target.cuda()
    else:
        z = torch.randn(batch_size, gan.z_dim)
        key = memory.sample_key(batch_size)

    # train discriminator on fake data
    gen_input = torch.cat((z, key), dim=1)
    fake_batch = gan.generate(gen_input)
    fake_disc_result = gan.discriminate(fake_batch.detach(), fake_target)  # detach so gradients not computed for generator
    disc_train_loss_false = loss(fake_disc_result.squeeze(), fake_target)
    disc_train_loss_false.backward()
    torch.nn.utils.clip_grad_norm_(gan.dmn.parameters(), grad_clip)
    disc_optimizer.step()  # set for discriminator only

    #  compute performance statistics
    disc_train_loss = disc_train_loss_true + disc_train_loss_false

    disc_fake_accuracy = 1 - torch.sum(fake_disc_result > 0).item() / batch_size
    disc_true_accuracy = torch.sum(true_disc_result > 0).item() / batch_size

    #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
    if label_smoothing:
        true_target = torch.FloatTensor(batch_size).uniform_(0.7, 1.2)
    else:
        true_target = torch.ones(batch_size)

    if cuda:
        z = torch.randn(batch_size, gan.mcgn.z_dim).cuda()
        key = memory.sample_key(batch_size).cuda()
        true_target = true_target.cuda()
    else:
        z = torch.randn(batch_size, gan.mcgn.z_dim)
        key = memory.sample_key(batch_size)

    # train generator
    gan.mcgn.zero_grad()
    gen_input = torch.cat((z, key), dim=1)
    fake_batch = gan.generate(gen_input)
    disc_result = gan.discriminate(fake_batch, fake_target)
    gen_train_loss = loss(disc_result.squeeze(), true_target)

    gen_train_loss.backward()
    torch.nn.utils.clip_grad_norm_(gan.mcgn.parameters(), grad_clip)
    gen_optimizer.step()  # set for generator only

    return disc_train_loss, gen_train_loss, disc_true_accuracy, disc_fake_accuracy
