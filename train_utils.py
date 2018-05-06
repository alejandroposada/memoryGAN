import torch


def train_step(gan, batch_size, is_cuda, true_batch, grad_clip, disc_optimizer, gen_optimizer, memory, use_EM):
    gan.dmn.zero_grad()

    true_target = torch.ones(batch_size)

    # Sample  minibatch  of examples from data generating distribution
    if is_cuda:
        true_batch = true_batch.cuda()
        true_target = true_target.cuda()

    # train discriminator on true data
    true_disc_result, q_real = gan.discriminate(true_batch, true_target)

    #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
    fake_target = torch.zeros(batch_size)

    if is_cuda:
        z = torch.randn(batch_size, gan.z_dim).cuda()
        fake_target = fake_target.cuda()
    else:
        z = torch.randn(batch_size, gan.z_dim)

    # train discriminator on fake data
    fake_batch = gan.generate(z)
    fake_disc_result, q_fake = gan.discriminate(fake_batch.detach(), fake_target)  # gradients not computed for generator

    # Calculate Dloss
    disc_train_loss = gan.Dloss(true_disc_result, fake_disc_result)
    disc_train_loss.backward()
    # print(gan.dmn.conv4.bias.grad)

    # Update memory
    if memory:
        if use_EM:
            gan.memory.update_memory(q_real.detach(), true_target.detach())
            gan.memory.update_memory(q_fake.detach(), fake_target.detach())
        else:
            gan.memory.update_memory_noEM(q_real.detach(), true_target.detach())
            gan.memory.update_memory_noEM(q_fake.detach(), fake_target.detach())

    torch.nn.utils.clip_grad_norm_(gan.dmn.parameters(), grad_clip)
    disc_optimizer.step()  # set for discriminator only

    disc_fake_accuracy = 1 - torch.sum(fake_disc_result > 0.5).item() / batch_size
    disc_true_accuracy = torch.sum(true_disc_result > 0.5).item() / batch_size

    #  Sample minibatch of m noise samples from noise prior p_g(z) and transform
    fake_target = torch.ones(batch_size)

    if is_cuda:
        z = torch.randn(batch_size, gan.z_dim).cuda()
        fake_target = fake_target.cuda()
    else:
        z = torch.randn(batch_size, gan.z_dim)

    # train generator
    gan.mcgn.zero_grad()
    fake_batch = gan.generate(z)

    disc_result, _ = gan.discriminate(fake_batch, fake_target)
    gen_train_loss = gan.Gloss(disc_result)

    gen_train_loss.backward()
    torch.nn.utils.clip_grad_norm_(gan.mcgn.parameters(), grad_clip)
    gen_optimizer.step()  # set for generator only

    if memory:
        if use_EM:
            gan.memory.update_memory(q_fake.detach(), fake_target.detach())
        else:
            gan.memory.update_memory_noEM(q_fake.detach(), fake_target.detach())

    return disc_train_loss, gen_train_loss, disc_true_accuracy, disc_fake_accuracy
