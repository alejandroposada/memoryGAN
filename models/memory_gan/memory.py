import torch.nn as nn
import torch.nn.functional as F
import torch

class memory(nn.Module):
    """DISCRIMINATIVE MEMORY NETWORK
	"""
    def __init__(self, key_dim, memory_size, choose_k, cuda):
        """
        """
        super(memory, self).__init__()
        self.beta = 1e-8
        self.epsilon = 1e-3
        self.key_dim = key_dim
        self.memory_size = memory_size
        self.choose_k = min(choose_k, self.memory_size)

        if cuda:
            self.memory_key = torch.zeros([self.memory_size, self.key_dim]).cuda()
            self.memory_values = torch.ones([self.memory_size]).cuda()
            self.memory_values[self.memory_size // 2:] = 0
            self.memory_age = torch.zeros([self.memory_size]).cuda()
            self.memory_hist = torch.FloatTensor(size=[self.memory_size]).cuda()
            self.memory_hist[:] = 1e-5
        else:
            self.memory_key = torch.zeros([self.memory_size, self.key_dim])              # K in 3.1 of paper
            self.memory_values = torch.ones([self.memory_size])
            self.memory_values[self.memory_size // 2:] = 0                               # v in 3.1 of paper
            self.memory_age = torch.zeros([self.memory_size])                            # a in 3.1 of paper
            self.memory_hist = torch.FloatTensor(size=[self.memory_size])                # h in 3.1 of paper
            self.memory_hist[:] = 1e-5

    def query(self, q):
        #  compute P(x|c)
        similarities = torch.matmul(q, torch.transpose(self.memory_key, 1, 0)).detach()
        p_x_given_c_unnorm = torch.exp(similarities).detach()

        #  compute P(c)
        p_c = ((self.memory_hist+self.beta)/torch.sum(self.memory_hist+self.beta)).detach()

        #  compute P(c|x) from eq 1 of paper
        p_c_given_x = (p_x_given_c_unnorm*p_c).detach()

        #  take only top k
        p_c_given_x_approx, idxs = torch.topk(p_c_given_x, k=self.choose_k)

        #  compute P(y|x) = \sum_i P(c=i|x)v_i from eq 4 of paper
        p_c_given_x_approx = (p_c_given_x_approx / p_c_given_x_approx.sum(1)).detach()
        p_y_given_x = (self.memory_values[idxs]*p_c_given_x_approx).sum(1).detach()

        #  clip values
        p_y_given_x[p_y_given_x < self.epsilon] = self.epsilon
        p_y_given_x[p_y_given_x > 1-self.epsilon] = 1-self.epsilon

        return p_y_given_x

    #  EM update of memory, 3.1.2
    def update_memory(self, q, label, n_steps=1):
        #  compute P(x|c, v_c=y)
        similarities = torch.matmul(q, torch.transpose(self.memory_key, 1, 0)).detach()
        p_x_given_c_unnorm = torch.exp(similarities).detach()

        #  compute P(c)
        p_c = ((self.memory_hist+self.beta)/torch.sum(self.memory_hist+self.beta)).detach()

        #  compute P(c|x, v_c=y) from eq 1 of paper
        p_c_given_x = (p_x_given_c_unnorm*p_c).detach()

        p_v_c_eq_y = torch.ger(label, self.memory_values) + torch.ger(1-label, 1-self.memory_values)
        p_c_given_x_v = (p_c_given_x*p_v_c_eq_y).detach()
        p_c_given_x_v_approx, idx = torch.topk(p_c_given_x_v, k=self.choose_k)

        #  check if S contains correct label
        s_with_correct_label =\
            torch.eq(self.memory_values[idx], torch.transpose(label.expand(len(label),len(label)), 0, 1))

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
                    similarities = torch.matmul(q[i], torch.transpose(k_hat, 1, 0)).detach()
                    p_x_given_c_unnorm = torch.exp(similarities).detach()
                    p_c = ((h_hat + self.beta) / torch.sum(h_hat + self.beta)).detach()
                    p_c_given_x = (p_x_given_c_unnorm * p_c).detach()
                    next_gamma = (p_c_given_x / p_c_given_x.sum(0)).detach()

                    #  M step
                    h_hat += next_gamma - gamma
                    k_hat += torch.transpose(((next_gamma - gamma) / h_hat).expand(self.key_dim, len(h_hat)),0,1)*(q[i] - k_hat)
                    gamma = next_gamma

                k_hat = torch.norm(k_hat, p=2, dim=0).detach()  # l2 normalize
                self.memory_key[idx_to_change] = k_hat
                self.memory_hist[idx_to_change] = h_hat




            #p_c_given_x_y_approx = []

            #idxs = []
            #for i, l in enumerate(label):
            #    probs, idx = torch.topk(p_c_given_x[i, self.memory_values == l], k=self.choose_k)
            #    p_c_given_x_y_approx.append(probs)
            #    idxs.append(idx)

            #p_c_given_x_y_approx = torch.stack(p_c_given_x_y_approx, 0)
            #idxs = torch.stack(idxs, 0)

            #p_c_given_x_y = p_c_given_x[:, self.memory_values == 1]

            #  take only top k
            #p_c_given_x_y_approx, idxs = torch.topk(p_c_given_x_y, k=self.choose_k)

            #  compute gamma = P(c|x, v_c=y)


            # M step

            #  check if S_y contains the correct label y










