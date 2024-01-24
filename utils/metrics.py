

from typing import Union, Tuple, Optional, List
import numpy as np
import torch
from sklearn import metrics

def cal_mse(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    ) -> Union[float, torch.Tensor]:
    
    
    assert type(predictions) == type(targets), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )
    
    lib = np if isinstance(predictions, np.ndarray) else torch
    
    return lib.mean(lib.square(predictions - targets))