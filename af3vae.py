import torch
import torch.nn as nn

from functools import partial
from einops import rearrange, repeat, reduce, einsum, pack, unpack

torch.manual_seed(42)

LinearNoBias = partial(nn.Linear, bias=False)

class OuterProductMean(nn.Module):
    """ Algorithm 9 """

    def __init__(
        self,
        *,
        dim_msa = 5,
        dim_pairwise = 128,
        dim_hidden = 5,
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
        mask=None,
        msa_mask=None
    ):

        msa = self.norm(msa)
        
        # print(self.to_hidden(msa))

        # a, b = self.to_hidden(msa).chunk(2, dim = -1)
        
        # print(a.shape)
        # print(b.shape)
        
        
        aggregated_msa = msa.mean(dim=1, keepdim=True)  # You can also use sum or max

        # Apply the linear transformation
        hidden = self.to_hidden(aggregated_msa)

        # Broadcast the result back to match the original dimensions of msa
        hidden = hidden.expand(-1, msa.size(1), -1)

        # Split into two parts
        a, b = hidden.chunk(2, dim=-1)

        outer_product = einsum(a, b, 'b s i d, b s j e -> b i j d e s')
        print(outer_product)

        # # maybe masked mean for outer product

        # if exists(msa_mask):
        #     outer_product = einx.multiply('b i j d e s, b s -> b i j d e s', outer_product, msa_mask.float())

        #     num = reduce(outer_product, '... s -> ...', 'sum')
        #     den = reduce(msa_mask.float(), '... s -> ...', 'sum')

        #     outer_product_mean = einx.divide('b i j d e, b', num, den.clamp(min = self.eps))
        # else:
        #     outer_product_mean = reduce(outer_product, '... s -> ...', 'mean')

        # # flatten

        # outer_product_mean = rearrange(outer_product_mean, '... d e -> ... (d e)')

        # # masking for pairwise repr

        # if exists(mask):
        #     mask = einx.logical_and('b i , b j -> b i j 1', mask, mask)
        #     outer_product_mean = outer_product_mean * mask

        # pairwise_repr = self.to_pairwise_repr(outer_product_mean)
        # return pairwise_repr
        
outer_product_mean = OuterProductMean()
msa = torch.randn((1,3,3,5))
print("Original")
with torch.no_grad():
    outer_product_mean(msa)

print("Permuted")
msa[:, [0, 1]] = msa[:, [1, 0]]
with torch.no_grad():
    outer_product_mean(msa)
