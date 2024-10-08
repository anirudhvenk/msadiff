import torch
import torch.nn as nn
import einx
import torch.nn.functional as F

from einops import einsum, rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
from functools import partial
from encoder_utils import Attention

LinearNoBias = partial(nn.Linear, bias = False)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def default(v, d):
    return v if exists(v) else d

def exists(v):
    return v is not None

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

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
        *,
        dim,
        expansion_factor = 4
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.ff = nn.Sequential(
            LinearNoBias(dim, dim_inner * 2),
            SwiGLU(),
            LinearNoBias(dim_inner, dim)
        )

    def forward(
        self,
        x
    ):

        return self.ff(x)

class PreLayerNorm(nn.Module):
    def __init__(
        self,
        fn,
        *,
        dim,
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        **kwargs
    ):
        x = self.norm(x)
        return self.fn(x, **kwargs)
    
class OuterProductMean(nn.Module):
    """ Algorithm 9 """

    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 128,
        dim_hidden = 64,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim_msa)
        self.to_hidden = LinearNoBias(dim_msa, dim_hidden * 2)
        self.to_pairwise_repr = nn.Linear(dim_hidden ** 2, dim_pairwise)

    def forward(
        self,
        msa,
        *,
        mask = None,
        msa_mask = None
    ):

        msa = self.norm(msa)

        # line 2

        a, b = self.to_hidden(msa).chunk(2, dim = -1)

        # maybe masked mean for outer product

        if exists(msa_mask):
            a = einx.multiply('b s i d, b s -> b s i d', a, msa_mask.float())
            b = einx.multiply('b s j e, b s -> b s j e', b, msa_mask.float())

            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')

            num_msa = reduce(msa_mask.float(), '... s -> ...', 'sum')

            outer_product_mean = einx.divide('b i j d e, b', outer_product, num_msa.clamp(min = self.eps))
        else:
            num_msa = msa.shape[1]
            outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e')
            outer_product_mean = outer_product / num_msa

        # flatten

        outer_product_mean = rearrange(outer_product_mean, '... d e -> ... (d e)')

        # masking for pairwise repr

        if exists(mask):
            mask = einx.logical_and('b i , b j -> b i j 1', mask, mask)
            outer_product_mean = outer_product_mean * mask

        pairwise_repr = self.to_pairwise_repr(outer_product_mean)
        return pairwise_repr
    
class MSAPairWeightedAveraging(nn.Module):
    """ Algorithm 10 """

    def __init__(
        self,
        *,
        dim_msa = 64,
        dim_pairwise = 128,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        dropout_type = None
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.msa_to_values_and_gates = nn.Sequential(
            nn.LayerNorm(dim_msa),
            LinearNoBias(dim_msa, dim_inner * 2),
            Rearrange('b s n (gv h d) -> gv b h s n d', gv = 2, h = heads)
        )

        self.pairwise_repr_to_attn = nn.Sequential(
            nn.LayerNorm(dim_pairwise),
            LinearNoBias(dim_pairwise, heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            Rearrange('b h s n d -> b s n (h d)'),
            LinearNoBias(dim_inner, dim_msa),
            Dropout(dropout, dropout_type = dropout_type)
        )

    def forward(
        self,
        *,
        msa,
        pairwise_repr,
        mask = None,
    ):

        values, gates = self.msa_to_values_and_gates(msa)
        gates = gates.sigmoid()

        # line 3

        b = self.pairwise_repr_to_attn(pairwise_repr)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            b = b.masked_fill(~mask, max_neg_value(b))

        # line 5

        weights = b.softmax(dim = -1)

        # line 6

        out = einsum(weights, values, 'b h i j, b h s j d -> b h s i d')

        out = out * gates

        # combine heads

        return self.to_out(out)
    

class MSAModule(nn.Module):
    """ Algorithm 8 """

    def __init__(
        self,
        *,
        dim_single = 384,
        dim_pairwise = 128,
        depth = 4,
        dim_msa = 64,
        dim_msa_input = None,
        outer_product_mean_dim_hidden = 32,
        msa_pwa_dropout_row_prob = 0.15,
        msa_pwa_heads = 8,
        msa_pwa_dim_head = 8,
        pairwise_block_kwargs: dict = dict(),
        max_num_msa = None,
        layerscale_output: bool = False
    ):
        super().__init__()

        self.max_num_msa = default(max_num_msa, float('inf'))  # cap the number of MSAs, will do sample without replacement if exceeds

        self.msa_init_proj = LinearNoBias(dim_msa_input, dim_msa) if exists(dim_msa_input) else nn.Identity()
        
        self.single_to_msa_feats = LinearNoBias(dim_single, dim_msa)

        layers = nn.ModuleList([])

        for _ in range(depth):

            msa_pre_ln = partial(PreLayerNorm, dim = dim_msa)

            outer_product_mean = OuterProductMean(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                dim_hidden = outer_product_mean_dim_hidden
            )

            msa_pair_weighted_avg = MSAPairWeightedAveraging(
                dim_msa = dim_msa,
                dim_pairwise = dim_pairwise,
                heads = msa_pwa_heads,
                dim_head = msa_pwa_dim_head,
                dropout = msa_pwa_dropout_row_prob,
                dropout_type = 'row'
            )

            msa_transition = Transition(dim = dim_msa)

            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )

            layers.append(nn.ModuleList([
                outer_product_mean,
                msa_pair_weighted_avg,
                msa_pre_ln(msa_transition),
                pairwise_block
            ]))

        self.layers = layers

        self.layerscale_output = 1.

    def forward(
        self,
        *,
        single_repr,
        pairwise_repr,
        msa,
        mask = None,
        msa_mask = None,
    ):
        batch, num_msa, device = *msa.shape[:2], msa.device
        msa = self.msa_init_proj(msa)
        

        single_msa_feats = self.single_to_msa_feats(single_repr)

        msa = rearrange(single_msa_feats, 'b n d -> b 1 n d') + msa

        for idx, (
            outer_product_mean,
            msa_pair_weighted_avg,
            msa_transition,
            pairwise_block
        ) in enumerate(self.layers):

            # communication between msa and pairwise rep

            pairwise_repr = outer_product_mean(msa, mask = mask, msa_mask = msa_mask) + pairwise_repr

            msa = msa_pair_weighted_avg(msa = msa, pairwise_repr = pairwise_repr, mask = mask) + msa
            msa = msa_transition(msa) + msa            

            # pairwise block

            pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)

        return msa, pairwise_repr * self.layerscale_output
    
class PairwiseBlock(nn.Module):
    def __init__(
        self,
        *,
        dim_pairwise = 128,
        tri_mult_dim_hidden = None,
        tri_attn_dim_head = 32,
        tri_attn_heads = 4,
        dropout_row_prob = 0.25,
        dropout_col_prob = 0.25,
    ):
        super().__init__()

        pre_ln = partial(PreLayerNorm, dim = dim_pairwise)

        tri_mult_kwargs = dict(
            dim = dim_pairwise,
            dim_hidden = tri_mult_dim_hidden
        )

        tri_attn_kwargs = dict(
            dim = dim_pairwise,
            heads = tri_attn_heads,
            dim_head = tri_attn_dim_head
        )

        self.tri_mult_outgoing = pre_ln(TriangleMultiplication(mix = 'outgoing', dropout = dropout_row_prob, dropout_type = 'row', **tri_mult_kwargs))
        self.tri_mult_incoming = pre_ln(TriangleMultiplication(mix = 'incoming', dropout = dropout_row_prob, dropout_type = 'row', **tri_mult_kwargs))
        self.tri_attn_starting = pre_ln(TriangleAttention(node_type = 'starting', dropout = dropout_row_prob, dropout_type = 'row', **tri_attn_kwargs))
        self.tri_attn_ending = pre_ln(TriangleAttention(node_type = 'ending', dropout = dropout_col_prob, dropout_type = 'col', **tri_attn_kwargs))
        self.pairwise_transition = pre_ln(Transition(dim = dim_pairwise))

    def forward(
        self,
        *,
        pairwise_repr,
        mask = None
    ):
        pairwise_repr = self.tri_mult_outgoing(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_mult_incoming(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_attn_starting(pairwise_repr, mask = mask) + pairwise_repr
        pairwise_repr = self.tri_attn_ending(pairwise_repr, mask = mask) + pairwise_repr

        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr
    
class TriangleMultiplication(nn.Module):

    def __init__(
        self,
        *,
        dim,
        dim_hidden = None,
        mix = 'incoming',
        dropout = 0.,
        dropout_type = None
    ):
        super().__init__()

        dim_hidden = default(dim_hidden, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_right_proj = nn.Sequential(
            LinearNoBias(dim, dim_hidden * 4),
            nn.GLU(dim = -1)
        )

        self.out_gate = LinearNoBias(dim, dim_hidden)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'incoming':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(dim_hidden)

        self.to_out = nn.Sequential(
            LinearNoBias(dim_hidden, dim),
            Dropout(dropout, dropout_type = dropout_type)
        )

    def forward(
        self,
        x,
        mask = None
    ):

        if exists(mask):
            mask = einx.logical_and('b i, b j -> b i j 1', mask, mask)

        x = self.norm(x)

        left, right = self.left_right_proj(x).chunk(2, dim = -1)

        if exists(mask):
            left = left * mask
            right = right * mask

        out = einsum(left, right, self.mix_einsum_eq)

        out = self.to_out_norm(out)

        out_gate = self.out_gate(x).sigmoid()
        out = out * out_gate

        return self.to_out(out)
    
class TriangleAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        node_type,
        dropout = 0.,
        dropout_type = None,
        **attn_kwargs
    ):
        super().__init__()
        self.need_transpose = node_type == 'ending'

        self.attn = Attention(dim = dim, heads = heads, **attn_kwargs)

        self.dropout = Dropout(dropout, dropout_type = dropout_type)

        self.to_attn_bias = nn.Sequential(
            LinearNoBias(dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    def forward(
        self,
        pairwise_repr,
        mask = None,
        **kwargs
    ):

        if self.need_transpose:
            pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b j i d')

        attn_bias = self.to_attn_bias(pairwise_repr)

        batch_repeat = pairwise_repr.shape[1]
        attn_bias = repeat(attn_bias, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b repeat) ...', repeat = batch_repeat)

        pairwise_repr, packed_shape = pack_one(pairwise_repr, '* n d')

        out = self.attn(
            pairwise_repr,
            mask = mask,
            attn_bias = attn_bias,
            **kwargs
        )

        out = unpack_one(out, packed_shape, '* n d')

        if self.need_transpose:
            out = rearrange(out, 'b j i d -> b i j d')

        return self.dropout(out)
    
class Dropout(nn.Module):
    def __init__(
        self,
        prob: float,
        *,
        dropout_type = None
    ):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type

    def forward(
        self,
        t
    ):

        if self.dropout_type in {'row', 'col'}:
            assert t.ndim == 4, 'tensor must be 4 dimensions for row / col structured dropout'

        if not exists(self.dropout_type):
            return self.dropout(t)

        if self.dropout_type == 'row':
            batch, row, _, _ = t.shape
            ones_shape = (batch, row, 1, 1)

        elif self.dropout_type == 'col':
            batch, _, col, _ = t.shape
            ones_shape = (batch, 1, col, 1)

        ones = t.new_ones(ones_shape)
        dropped = self.dropout(ones)
        return t * dropped