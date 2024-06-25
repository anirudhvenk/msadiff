import math
import torch
import torch.nn as nn

from typing import Optional
from esm.modules import FeedForwardNetwork, NormalizedResidualBlock

class MSACrossAttention(nn.Module):
    """Computes cross-attention of 2D input with another query sequence."""

    def __init__(
        self,
        query_dim,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens_per_msa: int = 2 ** 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(query_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, k):
        num_rows = k.size(0)
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(
        self,
        x,
        query,
        cross_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                query,
                x[start : start + max_rows],
                scaling,
                cross_attn_padding_mask=cross_attn_padding_mask[:, start : start + max_rows]
                if cross_attn_padding_mask is not None
                else None,
            )
            attns += attn_weights
        attn_probs = attns.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(x[start : start + max_rows], attn_probs)
            outputs.append(output)

        output = torch.cat(outputs, 0)
        return output, attn_probs

    def compute_attention_weights(
        self,
        x,
        query,
        scaling: float,
        cross_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(query).view(num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k *= scaling

        attn_weights = torch.einsum("jnhd,rinhd->hnij", q, k)

        if cross_attn_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                cross_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2),
                -10000,
            )

        return attn_weights

    def compute_attention_update(
        self,
        x,
        attn_probs,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum("hnij,rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
        self,
        x,
        query,
        cross_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if (num_rows * num_cols > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            return self._batched_forward(x, query, cross_attn_padding_mask)
        else:
            scaling = self.align_scaling(x)
            attn_weights = self.compute_attention_weights(
                x, query, scaling, cross_attn_padding_mask
            )
            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            output = self.compute_attention_update(x, attn_probs)
            return output, attn_probs


class MSATransformerLayer(nn.Module):
    """Implements an Axial MSA Transformer block."""

    def __init__(
        self,
        query_dim: int = 320,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2**14,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        self.msa_cross_attention = MSACrossAttention(
            query_dim,
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        self.feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
            activation_dropout=activation_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
        )

        self.msa_cross_attention = self.build_residual(self.msa_cross_attention)
        self.feed_forward_layer = self.build_residual(self.feed_forward_layer)

    def build_residual(self, layer: nn.Module):        
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
            self.dropout_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        query: torch.Tensor,
        cross_attn_padding_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        x, attn_weights = self.msa_cross_attention(
            x=x,
            query=query,
            cross_attn_padding_mask=cross_attn_padding_mask
        )
        
        x = self.feed_forward_layer(x)
        if need_head_weights:
            return x, attn_weights
        else:
            return x
