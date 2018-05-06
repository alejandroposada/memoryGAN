import torch.nn as nn
import torch
from helpers import normalize


class memory(nn.Module):
    """DISCRIMINATIVE MEMORY NETWORK """

    def __init__(self, key_dim, memory_size, choose_k, is_cuda, alpha, num_steps):
        """
        """
        super(memory, self).__init__()
        self.beta = 1e-8
        self.epsilon = 1e-3
        self.key_dim = key_dim
        self.memory_size = memory_size
        self.choose_k = min(choose_k, self.memory_size)
        self.alpha = alpha
        self.num_steps = num_steps

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

    def sample_key(self, batch_size):
        real_hist = self.memory_hist * self.memory_values
        probs = real_hist / real_hist.sum(0)
        distrib = torch.distributions.Categorical(probs)
        sampled_idxs = distrib.sample(torch.Size([batch_size]))
        sample_keys = self.memory_key[sampled_idxs]
        return sample_keys

    def query(self, q):
        result, _, __ = self.get_result(q)
        return result

    def update_memory(self, q, label):
        '''
        result, joint_, vals_ = self.get_result(q, alpha)
        reset_mask = self.get_reset_mask(label, joint_, vals_)

        k_idxs = self.get_hint_pool_idxs(q, label=label)

        # Usual EM update
        gather memory_keys, values, hists from k_idxs
        '''
        result, joint_, vals, = self.get_result(q)
        reset_mask = self.get_reset_mask(label, vals)
        k_idxs = self.obtain_topk(q, label=label, label_bool=True)

        # EM
        gamma = 0.0  # same as initial posterior
        h_hat = self.alpha * self.memory_hist[k_idxs]  # 64x128
        k_hat = self.memory_key[k_idxs]  # 64x128x256
        red_val = self.memory_values[k_idxs]  # 64x128, to be used in updates I guess

        for _ in range(self.num_steps):
            #  E Step
            similarities = torch.matmul(q.view(q.size(0), 1, self.key_dim), torch.transpose(k_hat, 1, 2)).squeeze(1)  # 64x128
            likelihood = torch.exp(similarities - 1.)  # 64x128
            prior = h_hat + self.beta  # 64x128
            joint = likelihood * prior  # 64x128
            next_gamma = torch.div(joint, joint.sum(1).view(q.size(0), 1))  # 64x128
            h_hat += (next_gamma - gamma).sum(1).unsqueeze(1).repeat(1, self.choose_k)  # 64x128 (may need to change the axis of sum)
            upd_ratio = (next_gamma - gamma) / h_hat  # 64x128

            #  M step
            k_hat = k_hat * (1. - upd_ratio.unsqueeze(2).expand(-1, -1, self.key_dim))
            k_hat += upd_ratio.unsqueeze(2).expand(-1, -1, self.key_dim) * q.unsqueeze(1).repeat(1, self.choose_k, 1)
            k_hat = normalize(k_hat)

            gamma = next_gamma

    def get_result(self, q):
        '''compute posterior conditioned over top_k keys from before
        input q
        1) get top_k from get_hint(q)
        2) gather vals, keys, hist of these indices
        3) compute sim using only these keys
        4) compute likelihood = exp(sim - 1), prior, joint = likelihood*prior
        5) post = joint / joint.sum(axis=1) (would be size 64 but keep_dims so 64x128)
        6) result = (post*vals).sum(axis=1) (this time its 64)
        7) return result (64), joint (64x128), vals (64, 128)
        '''
        k_idxs = self.obtain_topk(q)
        red_mem_keys = self.memory_key[k_idxs]  # 64x128x256
        red_mem_hist = self.memory_hist[k_idxs] * self.alpha  # 64x128, mul by alpha for some reason
        red_mem_vals = self.memory_values[k_idxs]  # 64x128
        # Please check the following matmul: 64x128x256 * 64x256 --> 64x128
        similarities = torch.matmul(q.view(q.size(0), 1, self.key_dim),
                                    torch.transpose(red_mem_keys, 1, 2)).squeeze(1)  # 64x128
        likelihood = torch.exp(similarities - 1.)  # 64x128
        prior = red_mem_hist + self.beta  # 64x128
        joint = likelihood * prior  # 64x128
        posterior = torch.div(joint, joint.sum(1).view(q.size(0), 1))  # 64x128
        result = (posterior * red_mem_vals).sum(1)
        return result, joint, red_mem_vals

    def obtain_topk(self, q, label=None, label_bool=False):
        '''compute top k matching indices over FULL keys
        input q, label
        1) compute similarity between q and K, get something of dim 64x4096 (batch size=64)
        2) if label != None, sim = sim - 2*is_wrong, where is_wrong = |label-self.memory_values| broadcasted to 64x4096
        3) likelihood = exp(sim - 1), prior = self.memory_hist + beta
        4) top_k = topk(prior*likelihood) (keep top k=128 indices)
        5) return the indices 64x128
        '''
        similarities = torch.matmul(q, torch.t(self.memory_key))  # size= 64x4096
        if label_bool:
            is_wrong = torch.abs(label.unsqueeze(1).repeat(1, self.memory_size) -
                                 self.memory_values.unsqueeze(0).repeat(q.size(0), 1))  # 64x4096
            # ignored the minimum statement since it looks like it doesn't really do anything.
            similarities = similarities - 2 * is_wrong
        likelihood = torch.exp(similarities - 1.)  # 64x4096
        prior = self.memory_hist.unsqueeze(0).repeat(q.size(0), 1) + self.beta
        p_c_given_x_est = likelihood * prior
        _, k_idxs = torch.topk(p_c_given_x_est, k=self.choose_k)
        return k_idxs

    def get_reset_mask(self, label, val):
        '''get index of nearest correct answer, check if it is
        1) teacher_hints = |label-val| broadcasted to 64x128
        2) teacher_hints = 1 - min(1, teacher_hints)
        3) take top teacher hint per (top_hints has dim 64)
        4) reset_mask == top_hints == 0 (64)
        '''
        teacher_hints = 1.0 - torch.abs(label.unsqueeze(1).repeat(1, val.size(1)) - val)  # 64x128
        sliced_hints = teacher_hints[:, 0]
        reset_mask = torch.eq(torch.zeros_like(sliced_hints), sliced_hints)
        return reset_mask  # 64


"""
OLD CODE FOR REFERENCE
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
    def update_memory(self, q, label):

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
                h_hat = self.alpha * self.memory_hist[idx_to_change]
                k_hat = self.memory_key[idx_to_change]
                for _ in range(self.num_steps):
                    #  E Step
                    similarities = torch.matmul(q[i], torch.transpose(k_hat, 1, 0))
                    p_x_given_c_unnorm = torch.exp(similarities)
                    p_c = ((h_hat + self.beta) / torch.sum(h_hat + self.beta))
                    p_c_given_x = (p_x_given_c_unnorm * p_c)
                    next_gamma = (p_c_given_x / p_c_given_x.sum(0))

                    #  M step
                    h_hat += next_gamma - gamma
                    k_hat += torch.transpose(((next_gamma - gamma) / h_hat).
                                             expand(self.key_dim, len(h_hat)), 0, 1)*(q[i] - k_hat)
                    gamma = next_gamma

                k_hat = normalize(k_hat)

                self.memory_key[idx_to_change] = k_hat
                self.memory_hist[idx_to_change] = h_hat
                self.memory_age += 1e-5  # testing

    def update_memory_noEM(self, q, label):
        # compute similarity between batch of q and my mem_keys
        similarities = torch.matmul(q, torch.transpose(self.memory_key, 1, 0))
        nearest_neighbours, idx = torch.topk(similarities, k=1)
        for i in idx:
            if self.memory_values[i] == label[i]:
                self.memory_key[i] = (q[i] + self.memory_key[i])/(q[i] + self.memory_key[i]).norm(2)
                self.memory_age += 1
                self.memory_age[i] = 0
            else:
                oldest_idx = torch.argmax(self.memory_age)
                self.memory_key[oldest_idx] = q[i]
                self.memory_key[oldest_idx] = 0
                self.memory_values[oldest_idx] = label[i]

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
        _, unnorm_idxs = torch.topk(p_c_given_x_v, k=self.choose_k)
        idxs = unnorm_idxs
        for i in range(idxs.size(0)):
            for j in range(idxs.size(1)):
                idxs[i, j] = indices[unnorm_idxs[i, j]].tolist()

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
                k_hat = self.memory_key[idxs[i]]
                for _ in range(n_steps):
                    #  E Step
                    similarities = torch.matmul(q[i], torch.t(k_hat))
                    p_x_given_c_unnorm = torch.exp(similarities).detach()
                    p_c = (h_hat + self.beta)  # / torch.sum(h_hat + self.beta))
                    joint = p_x_given_c_unnorm * p_c
                    p_c_given_x = joint / joint.sum(0)
                    # p_c_given_x = (p_x_given_c_unnorm * p_c)/(torch.matmul(p_x_given_c_unnorm, p_c))
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
"""