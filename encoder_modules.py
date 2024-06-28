import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

LinearNoBias = partial(nn.Linear, bias=None)

class SwiGLU(nn.Module):
    def forward(
        self,
        x
    ):
        x, gates = x.chunk(2, dim = -1)
        return F.silu(gates) * x

class Transition(nn.Module):
    def __init__(
        self,
        dim,
        expansion_factor=4
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            LinearNoBias(dim, dim_inner * 2),
            SwiGLU(),
            LinearNoBias(dim_inner, dim)
        )

    def forward(
        self,
        x
    ):
        return self.ff(x)

class OuterProductMean(nn.Module):
    def __init__(
        self,
        dim_msa,
        dim_seq,
        dim_hidden,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim_msa)
        self.to_hidden = LinearNoBias(dim_msa, dim_hidden * 2)
        self.to_single_repr = nn.Linear(dim_hidden ** 2, dim_seq)


    def forward(
        self,
        msa,
        mask=None
    ):
        msa = self.norm(msa)
        a, b = self.to_hidden(msa).chunk(2, dim=-1)
        
        num_alignments = msa.shape[1]
        outer_product = einsum(a, b, "b s l d, b s l e -> b l d e")
        outer_product_mean = outer_product / num_alignments
        outer_product_mean = rearrange(outer_product_mean, "... d e -> ... (d e)")
        
        if mask is not None:
            outer_product_mean = outer_product_mean * mask.unsqueeze(-1)
        
        single_repr = self.to_single_repr(outer_product_mean)
        return single_repr
    
class PairWeightedAveraging(nn.Module):
    def __init__(
        self,
        dim_msa,
        dim_seq,
        dim_head,
        heads,
        dropout,
        dropout_type="row"
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.msa_to_values_and_gates = nn.Sequential(
            nn.LayerNorm(dim_msa),
            LinearNoBias(dim_msa, dim_inner * 2),
            Rearrange("b s n (gv h d) -> gv b h s n d", gv=2, h=heads)
        )

        self.pairwise_repr_to_attn = nn.Sequential(
            nn.LayerNorm(dim_seq),
            LinearNoBias(dim_seq, heads),
            Rearrange("b l h -> b h l")
        )

        self.to_out = nn.Sequential(
            Rearrange("b h s n d -> b s n (h d)"),
            LinearNoBias(dim_inner, dim_msa),
            Dropout(dropout, dropout_type)
        )

    def forward(
        self,
        msa,
        single_repr,
        mask=None
    ):
        values, gates = self.msa_to_values_and_gates(msa)
        gates = gates.sigmoid()

        b = self.pairwise_repr_to_attn(single_repr)
        if mask is not None:
            mask = mask.unsqueeze(1)
            b = b.masked_fill(~mask, -torch.finfo(b.dtype).max)
            
        weights = b.softmax(dim = -1)
        
        out = einsum(weights, values, 'b h l, b h s l d -> b h s l d')
        out = out * gates

        return self.to_out(out)
    
class Dropout(nn.Module):
    def __init__(
        self,
        prob,
        dropout_type
    ):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type

    def forward(
        self,
        x
    ):
        if self.dropout_type in {"row", "col"}:
            assert x.ndim == 4, "tensor must be 4 dimensions for row / col structured dropout"
        else:
            return self.dropout(x)

        if self.dropout_type == "row":
            batch, row, _, _ = x.shape
            ones_shape = (batch, row, 1, 1)

        elif self.dropout_type == "col":
            batch, _, col, _ = x.shape
            ones_shape = (batch, 1, col, 1)

        ones = x.new_ones(ones_shape)
        dropped = self.dropout(ones)
        return x * dropped
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, batch_size, num_nodes):
        x = self.pos_table[:, :num_nodes].clone().detach()
        x = x.expand(batch_size, -1, -1)
        return x