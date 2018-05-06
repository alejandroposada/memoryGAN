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
        self.is_cuda = is_cuda

        if self.is_cuda:
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
        #  clip values
        result[result < self.epsilon] = self.epsilon
        result[result > 1 - self.epsilon] = 1 - self.epsilon
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

        k_idxs = self.obtain_topk(q, label=label)
        
        #  tile for multiple update
        rep_reset_mask = reset_mask.reshape([len(label), 1]).repeat(1, self.choose_k).reshape([-1])
        rep_oldest_idxs = self.get_oldest_idxs(len(label)).repeat([1, self.choose_k]).reshape([-1])
        rep_q = q.unsqueeze(1).repeat([1, self.choose_k, 1]).reshape([-1, self.key_dim])
        rep_label = label.unsqueeze(1).repeat([1, self.choose_k]).reshape([-1])
        #print('labels {}'.format(label.sum().item()))

        # EM
        gamma = 0.0  # same as initial posterior
        h_hat = self.alpha * self.memory_hist[k_idxs]  # 64x128
        k_hat = self.memory_key[k_idxs]  # 64x128x256
        v_hat = self.memory_values[k_idxs]  # 64x128, to be used in updates I guess

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
            k_hat = normalize(k_hat, 1)

            gamma = next_gamma

        upd_idxs = k_idxs.reshape([-1]).masked_scatter(rep_reset_mask, rep_oldest_idxs)
        upd_keys = k_hat.reshape([-1, self.key_dim]).masked_scatter(
            rep_reset_mask.unsqueeze(1).repeat([1, self.key_dim]), rep_q)
        upd_vals = v_hat.reshape([-1]).masked_scatter(rep_reset_mask, rep_label)
        upd_hists = torch.ones_like(rep_label)*self.memory_hist.mean()
        upd_hists.masked_scatter_(rep_reset_mask, h_hat.reshape([-1]))
        print(rep_reset_mask.sum(0))

        self.memory_age += 1
        if self.is_cuda:
            self.memory_age.put_(upd_idxs, torch.zeros([len(label) * self.choose_k]).cuda())
            self.memory_key.index_copy_(0, upd_idxs, upd_keys.cuda())
            self.memory_values.index_copy_(0, upd_idxs, upd_vals.cuda())
        else:
            self.memory_age.put_(upd_idxs, torch.zeros([len(label) * self.choose_k]))
            self.memory_key[upd_idxs] = upd_keys
            self.memory_values.index_copy_(0, upd_idxs, upd_vals)

        self.memory_hist.put_(upd_idxs, upd_hists)

        #print('update values: {}'.format(upd_vals.sum().item()))

        del upd_idxs, upd_keys, upd_vals, upd_hists, rep_reset_mask, rep_oldest_idxs, rep_q, rep_label
        if self.is_cuda:
            torch.cuda.empty_cache()


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

    def obtain_topk(self, q, label=None):
        '''compute top k matching indices over FULL keys
        input q, label
        1) compute similarity between q and K, get something of dim 64x4096 (batch size=64)
        2) if label != None, sim = sim - 2*is_wrong, where is_wrong = |label-self.memory_values| broadcasted to 64x4096
        3) likelihood = exp(sim - 1), prior = self.memory_hist + beta
        4) top_k = topk(prior*likelihood) (keep top k=128 indices)
        5) return the indices 64x128
        '''
        similarities = torch.matmul(q, torch.t(self.memory_key))  # size= 64x4096

        if label is not None:
            is_wrong = torch.abs(label.unsqueeze(1).repeat(1, self.memory_size) -
                                 self.memory_values.unsqueeze(0).repeat(q.size(0), 1))  # 64x4096
            is_wrong = torch.clamp(is_wrong, max=1.0, min=0)
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
        teacher_hints = torch.abs(label.unsqueeze(1).repeat(1, val.size(1)) - val)  # 64x128
        teacher_hints = 1.0 - torch.clamp(teacher_hints, max=1.0, min=0)
        sliced_hints = teacher_hints[:, 0]
        reset_mask = torch.eq(torch.zeros_like(sliced_hints), sliced_hints)
        return reset_mask  # 64

    def get_oldest_idxs(self, batch_size):
        _, oldest_idxs = torch.topk(self.memory_age, k=batch_size, sorted=False, largest=True)
        return oldest_idxs.reshape([batch_size, 1])

    def get_info_for_logging(self):
        return self.memory_hist.clone(), self.memory_key.clone(), self.memory_age.clone(), self.memory_values.clone()

    def update_memory_noEM(self, q, label):
        result, joint_, vals, = self.get_result(q)
        reset_mask = self.get_reset_mask(label, vals)

        k_idxs = self.obtain_topk(q, label=label)

        #  tile for multiple update
        rep_reset_mask = reset_mask.reshape([len(label), 1]).repeat(1, self.choose_k).reshape([-1])
        rep_oldest_idxs = self.get_oldest_idxs(len(label)).repeat([1, self.choose_k]).reshape([-1])
        rep_q = q.unsqueeze(1).repeat([1, self.choose_k, 1]).reshape([-1, self.key_dim])
        rep_label = label.unsqueeze(1).repeat([1, self.choose_k]).reshape([-1])

        h_hat = self.memory_hist[k_idxs]  # 64x128
        k_hat = self.memory_key[k_idxs]  # 64x128x256
        v_hat = self.memory_values[k_idxs]  # 64x128, to be used in updates I guess

        k_hat = normalize(k_hat + q.unsqueeze(1), 1)

        upd_idxs = k_idxs.reshape([-1]).masked_scatter(rep_reset_mask, rep_oldest_idxs)
        upd_keys = k_hat.reshape([-1, self.key_dim]).masked_scatter(
            rep_reset_mask.unsqueeze(1).repeat([1, self.key_dim]), rep_q)
        upd_vals = v_hat.reshape([-1]).masked_scatter(rep_reset_mask, rep_label)
        upd_hists = torch.ones_like(rep_label)*self.memory_hist.mean()
        upd_hists.masked_scatter_(rep_reset_mask, h_hat.reshape([-1]))

        self.memory_age += 1
        if self.is_cuda:
            self.memory_age.put_(upd_idxs, torch.zeros([len(label) * self.choose_k]).cuda())
            self.memory_key.index_copy_(0, upd_idxs, upd_keys.cuda())
            self.memory_values.index_copy_(0, upd_idxs, upd_vals.cuda())
        else:
            self.memory_age.put_(upd_idxs, torch.zeros([len(label) * self.choose_k]))
            self.memory_key[upd_idxs] = upd_keys
            self.memory_values.index_copy_(0, upd_idxs, upd_vals)

        self.memory_hist.put_(upd_idxs, upd_hists)

        #print('update values: {}'.format(upd_vals.sum().item()))

        del upd_idxs, upd_keys, upd_vals, upd_hists, rep_reset_mask, rep_oldest_idxs, rep_q, rep_label
        if self.is_cuda:
            torch.cuda.empty_cache()

        '''
        # compute similarity between batch of q and mem_keys
        similarities = torch.matmul(q, torch.transpose(self.memory_key, 1, 0))
        nearest_neighbours, idx = torch.topk(similarities, k=1)
        for i in idx:
            if self.memory_values[i] == label[i]:
                self.memory_key[i] = (q[i] + self.memory_key[i]) / (q[i] + self.memory_key[i]).norm(2)
                self.memory_age += 1
                self.memory_age[i] = 0
            else:
                oldest_idx = torch.argmax(self.memory_age)
                self.memory_key[oldest_idx] = q[i]
                self.memory_key[oldest_idx] = 0
                self.memory_values[oldest_idx] = label[i]
        pass
        '''