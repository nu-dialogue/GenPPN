import os
import torch

def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        local_rank = os.environ.get('LOCAL_RANK', 0)
        device = torch.device('cuda', int(local_rank))
    return device