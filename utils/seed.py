import torch
import numpy as np
import random
import os

def seed_everything(seed):

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    os.environ["PYTHONHASHSEED"] = str(seed)
