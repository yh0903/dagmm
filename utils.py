import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.tensor(x, requires_grad=requires_grad)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)