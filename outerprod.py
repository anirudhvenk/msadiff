import torch

from math import pi, sqrt
from pathlib import Path
from functools import partial, wraps
from collections import namedtuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from torch.nn import (
    Module,
    ModuleList,
    Linear,
    Sequential,
)

from typing import List, Literal, Tuple, NamedTuple, Dict, Callable
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

from tqdm import tqdm

from importlib.metadata import version

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

LinearNoBias = partial(Linear, bias = False)


class OuterProductMean(Module):
    """ Algorithm 9 """

    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 128,
        dim_hidden = 32,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim_msa)
        self.to_hidden = LinearNoBias(dim_msa, dim_hidden * 2)
        self.to_pairwise_repr = nn.Linear(dim_hidden ** 2, dim_pairwise)

    def forward(
        self,
        msa
    ):

        msa = self.norm(msa)

        # line 2

        a, b = self.to_hidden(msa).chunk(2, dim = -1)
        print(a.shape)
        print(b.shape)

        # maybe masked mean for outer product

        num_msa = msa.shape[1]
        outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')
        outer_product_mean = outer_product / num_msa
        print(outer_product_mean.shape)

        # flatten

        # outer_product_mean = rearrange(outer_product_mean, '... d e -> ... (d e)')
        # print(outer_product_mean.shape)

        # masking for pairwise repr


outerprod = OuterProductMean()
msa = torch.randn(1,8,5,64)

outerprod(msa)