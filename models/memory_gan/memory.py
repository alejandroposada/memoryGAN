import torch.nn as nn
import torch.nn.functional as F
import torch


class memory(nn.Module):
    """DISCRIMINATIVE MEMORY NETWORK """
    def __init__(self, key_dim, memory_size, choose_k, is_cuda):
        """
        """
        super(memory, self).__init__()
        self.beta = 1e-8
        self.epsilon = 1e-3
        self.key_dim = key_dim
        self.memory_size = memory_size
        self.choose_k = min(choose_k, self.memory_size)

        if is_cuda:
            self.memory_key = torch.zeros([self.memory_size, self.key_dim]).cuda()  # K in 3.1 of paper
            self.memory_values = torch.ones([self.memory_size]).cuda()
            self.memory_values[self.memory_size // 2:] = 0                          # v in 3.1 of paper
            self.memory_age = torch.zeros([self.memory_size]).cuda()                # a in 3.1 of paper
            self.memory_hist = torch.FloatTensor(size=[self.memory_size]).cuda()    # h in 3.1 of paper
            self.memory_hist[:] = 1e-5
        else:
            self.memory_key = torch.zeros([self.memory_size, self.key_dim])              # K in 3.1 of paper
            self.memory_values = torch.ones([self.memory_size])
            self.memory_values[self.memory_size // 2:] = 0                               # v in 3.1 of paper
            self.memory_age = torch.zeros([self.memory_size])                            # a in 3.1 of paper
            self.memory_hist = torch.FloatTensor(size=[self.memory_size])                # h in 3.1 of paper
            self.memory_hist[:] = 1e-5

        #self.memory_key.detach()
        #self.memory_values.detach()
        #self.memory_age.detach()
        #self.memory_hist.detach()

    def query(self, q):
        #  compute P(x|c)
        similarities = torch.matmul(q, torch.transpose(self.memory_key, 1, 0))
        p_x_given_c_unnorm = torch.exp(similarities)

        #  compute P(c)  # Roger: technically, you don't need to divide by the sum here (last equality in eq(3))
        p_c = ((self.memory_hist+self.beta)/torch.sum(self.memory_hist+self.beta))

        #  compute P(c|x) from eq 1 of paper
        p_c_given_x = (p_x_given_c_unnorm * p_c)

        #  take only top k
        p_c_given_x_approx, idxs = torch.topk(p_c_given_x, k=self.choose_k)

        #  compute P(y|x) = \sum_i P(c=i|x)v_i from eq 4 of paper
        p_y_given_x_unnorm = (self.memory_values[idxs] * p_c_given_x_approx).sum(1)
        p_y_given_x = (p_y_given_x_unnorm / p_c_given_x_approx.sum(1))

        #  clip values
        p_y_given_x[p_y_given_x < self.epsilon] = self.epsilon
        p_y_given_x[p_y_given_x > 1-self.epsilon] = 1-self.epsilon

        return p_y_given_x

    #  EM update of memory, 3.1.2
    def update_memory(self, q, label, n_steps=1):
        #  compute P(x|c, v_c=y)
        similarities = torch.matmul(q, torch.transpose(self.memory_key, 1, 0))
        p_x_given_c_unnorm = torch.exp(similarities)

        #  compute P(c)
        p_c = ((self.memory_hist+self.beta)/torch.sum(self.memory_hist+self.beta))

        #  compute P(c|x, v_c=y) from eq 1 of paper
        p_c_given_x = (p_x_given_c_unnorm * p_c)

        p_v_c_eq_y = torch.ger(label, self.memory_values) + torch.ger(1-label, 1-self.memory_values)
        p_c_given_x_v = (p_c_given_x*p_v_c_eq_y)
        p_c_given_x_v_approx, idx = torch.topk(p_c_given_x_v, k=self.choose_k)

        #  check if S contains correct label
        s_with_correct_label =\
            torch.eq(self.memory_values[idx], torch.transpose(label.expand(self.choose_k, len(label)), 0, 1))

        for i, l in enumerate(s_with_correct_label):
            if not l.any():
                #  find oldest memory slot and copy information onto it
                oldest_idx = torch.argmax(self.memory_age)
                self.memory_key[oldest_idx] = q[i]
                self.memory_hist[oldest_idx] = self.memory_hist.mean()
                self.memory_age[oldest_idx] = 0
                self.memory_values[oldest_idx] = label[i]
            else:
                idx_to_change = idx[i, l == 1]
                gamma = 0
                alpha = 0.5
                h_hat = alpha * self.memory_hist[idx_to_change]
                k_hat = self.memory_key[idx_to_change]
                for _ in range(n_steps):
                    #  E Step
                    similarities = torch.matmul(q[i], torch.transpose(k_hat, 1, 0))
                    p_x_given_c_unnorm = torch.exp(similarities).detach()
                    p_c = ((h_hat + self.beta) / torch.sum(h_hat + self.beta))
                    p_c_given_x = (p_x_given_c_unnorm * p_c).detach()
                    next_gamma = (p_c_given_x / p_c_given_x.sum(0))

                    #  M step
                    h_hat += next_gamma - gamma
                    k_hat += torch.transpose(((next_gamma - gamma) / h_hat).
                                             expand(self.key_dim, len(h_hat)), 0, 1)*(q[i] - k_hat)
                    gamma = next_gamma

                k_hat = (k_hat/torch.norm(k_hat, p=2, dim=0))  # l2 normalize
                self.memory_key[idx_to_change] = k_hat
                self.memory_hist[idx_to_change] = h_hat

    def update_memory_noEM(self, q, label):
        pass

    def Roger_update_memory(self, q, label, n_steps=1):
        # Goal: compute P(x|c, v_c=y)
        # Start with finding indices where mem_value=label
        indices = ((self.memory_values == label[0]) == 1).nonzero().squeeze()  # this assumes all labels are the same.
        reduced_memory = self.memory_key[indices]  # memory with only the correct classes

        # compute similarity between batch of q and my mem_keys with correct class
        similarities = torch.matmul(q, torch.t(reduced_memory))
        p_x_given_c_unnorm = torch.exp(similarities)

        #  compute P(c)
        p_c_unnorm = self.memory_hist[indices] + self.beta

        #  compute P(c|x, v_c=y) from eq 1 of paper
        p_c_given_x_v = (p_x_given_c_unnorm * p_c_unnorm)
        _, idxs = torch.topk(p_c_given_x_v, k=self.choose_k)

        for i, l in enumerate(idxs):
            if list(indices.size())[0] == 0:  # I don't think this ever happens...
                #  find oldest memory slot and copy information onto it
                oldest_idx = torch.argmax(self.memory_age)
                self.memory_key[oldest_idx] = q[i]
                self.memory_hist[oldest_idx] = self.memory_hist.mean()
                self.memory_age[oldest_idx] = 0
                self.memory_values[oldest_idx] = label[i]
            else:
                gamma = 0
                alpha = 0.5
                h_hat = alpha * self.memory_hist[idxs[i]]
                k_hat = reduced_memory[idxs[i]]
                for _ in range(n_steps):
                    #  E Step
                    similarities = torch.matmul(q[i], torch.t(k_hat))
                    p_x_given_c_unnorm = torch.exp(similarities).detach()
                    p_c = ((h_hat + self.beta) / torch.sum(h_hat + self.beta))
                    p_c_given_x = (p_x_given_c_unnorm * p_c)/(torch.matmul(p_x_given_c_unnorm, p_c))
                    next_gamma = p_c_given_x

                    #  M step
                    h_hat += next_gamma - gamma
                    k_hat += torch.transpose(((next_gamma - gamma) / h_hat).
                                             expand(self.key_dim, len(h_hat)), 0, 1)*(q[i] - k_hat)
                    gamma = next_gamma
                    k_hat = (k_hat/torch.norm(k_hat, p=2, dim=0))  # l2 normalize

                self.memory_key[idxs[i]] = k_hat
                self.memory_hist[idxs[i]] = h_hat
        self.memory_age[:] += 1  # I guess...

    def sample_key(self, batch_size):
        real_hist = self.memory_hist * self.memory_values
        probs = real_hist / real_hist.sum(0)
        distrib = torch.distributions.Categorical(probs)
        sampled_idxs = distrib.sample(torch.Size([batch_size]))
        sample_keys = self.memory_key[sampled_idxs]
        return sample_keys