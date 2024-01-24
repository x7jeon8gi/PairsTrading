import torch
import numpy as np

# NOT USED FOR CONTRASTIVE LEARNING

def shuffle_tensor(tensor, device=None, ratio=0.5):
    batch_size, length, dim = tensor.shape
    shuffle_len = int(length * ratio)
    indices = torch.randperm(length)[:shuffle_len]
    # print(indices)
    
    shuffled_tensor = tensor.clone()
    for i in range(batch_size):
        shuffled_tensor[i, indices] = tensor[i, indices[torch.randperm(shuffle_len)]]

    # stacked_tensor = torch.cat([tensor, shuffled_tensor], dim=0)
    labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)], dim=0)
    
    if isinstance(device, torch.device):
        labels = labels.to(device)
    else:
        pass

    return shuffled_tensor, labels


def shuffle_tensor_one(tensor, ratio=0.5):
    length, dim = tensor.shape
    shuffle_len = int(length * ratio)
    indices = torch.randperm(length)[:shuffle_len]
    # print(indices)
    
    shuffled_tensor = tensor.clone()
    shuffled_tensor[indices] = tensor[indices[torch.randperm(shuffle_len)]]

    # stacked_tensor = torch.cat([tensor, shuffled_tensor], dim=0)
    labels = torch.cat([torch.zeros(1), torch.ones(1)], dim=0)

    return shuffled_tensor, labels