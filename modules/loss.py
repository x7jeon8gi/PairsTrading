
import torch
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn

"""
https://github.com/Yunfan-Li/Contrastive-Clustering/blob/main/modules/contrastive_loss.py
"""


class InstanceLoss(nn.Module):
    
    LARGE_NUMBER = 1e4
    
    def __init__(self, tau=0.5, multiplier=2):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        
    def forward(self, z,):
        n = z.shape[0] # batch size
        assert n % self.multiplier ==0
        
        z = z / np.sqrt(self.tau)
        
        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] =- self.LARGE_NUMBER # diagonal
        
        logprob = F.log_softmax(logits, dim=1)
        
        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        return loss

class ClusterLoss(torch.nn.Module):
    LARGE_NUMBER = 1e4

    def __init__(self, tau=1.0, multiplier=2):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier

    def forward(self, c, get_map=False):
        n = c.shape[0]
        assert n % self.multiplier == 0
        half_n = n // 2  # Assuming c = torch.cat([c_i, c_j])

        # Split c into c_i and c_j
        c_i = c[:half_n]
        c_j = c[half_n:]

        c = F.normalize(c, p=2, dim=1) / np.sqrt(self.tau)

        logits = c @ c.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # Entropy loss for c_i
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

        # Entropy loss for c_j
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()

        # Combined entropy loss
        ne_loss = ne_i + ne_j

        # Original clustering loss
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        original_loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        # Combined loss
        final_loss = original_loss + ne_loss

        return final_loss