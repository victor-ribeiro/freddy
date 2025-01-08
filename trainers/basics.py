import torch
from enum import Enum


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class DEVICE(Enum):
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
