import types
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# This module was created based on https://github.com/xxxnell/how-do-vits-work

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, p, **kwargs):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)
        return x

    def extra_repr(self):
        return "p=%s" % repr(self.p)