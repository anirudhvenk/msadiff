# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

class RowCrossAttention(nn.Module):
    """Compute self-attention over rows of a 2D input."""

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
        self.attn_shape = "hnj"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(query_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(
        self,
        query,
        key,
        value,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = key.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(key)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                query,
                key[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, start : start + max_rows]
                if self_attn_padding_mask is not None
                else None,
            )
            attns += attn_weights
        attn_probs = attns.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(value[start : start + max_rows], attn_probs)
            outputs.append(output)

        output = torch.cat(outputs, 0)
        return output, attn_probs

    def compute_attention_weights(
        self,
        query,
        key,
        scaling: float,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = key.size()
        
        q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        if self_attn_padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - self_attn_padding_mask.permute(0, 2, 1).unsqueeze(2).unsqueeze(3).to(q)

        attn_weights = torch.einsum(f"nhd,rjnhd->{self.attn_shape}", q, k)

        if self_attn_mask is not None:
            raise NotImplementedError
            # Mask Size: [B x R x C], Weights Size: [H x B x C x C]

        if self_attn_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2),
                -10000,
            )

        return attn_weights

    def compute_attention_update(
        self,
        value,
        attn_probs,
    ):
        num_rows, num_cols, batch_size, embed_dim = value.size()
        v = self.v_proj(value).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"{self.attn_shape},rjnhd->rjnhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
        self,
        query,
        key,
        value,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = key.size()
        if (num_rows * num_cols > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            return self._batched_forward(query, key, value, self_attn_mask, self_attn_padding_mask)
        else:
            scaling = self.align_scaling(query)
            attn_weights = self.compute_attention_weights(
                query, key, scaling, self_attn_mask, self_attn_padding_mask
            )
            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            output = self.compute_attention_update(value, attn_probs)
            return output, attn_probs
        
class ColumnCrossAttention(nn.Module):
    """Compute self-attention over columns of a 2D input."""

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

    def _batched_forward(
        self,
        query,
        key,
        value,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = key.size()
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        outputs = []
        attns = []
        for start in range(0, num_cols, max_cols):
            output, attn = self(
                query,
                key[:, start : start + max_cols],
                value[:, start : start + max_cols],
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, :, start : start + max_cols]
                if self_attn_padding_mask is not None
                else None,
            )
            outputs.append(output)
            attns.append(attn)
        output = torch.cat(outputs, 1)
        attns = torch.cat(attns, 1)
        return output, attns

    def compute_attention_update(
        self,
        query,
        key,
        value,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = key.size()
        if num_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with padding
            attn_probs = torch.ones(
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device=key.device,
                dtype=key.dtype,
            )
            output = self.out_proj(self.v_proj(key))
        else:
            q = self.q_proj(query).view(batch_size, self.num_heads, self.head_dim)
            k = self.k_proj(key).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            v = self.v_proj(value).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            q *= self.scaling

            attn_weights = torch.einsum("nhd,jcnhd->hnj", q, k)

            if self_attn_mask is not None:
                raise NotImplementedError
            if self_attn_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    self_attn_padding_mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(3),
                    -10000,
                )

            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            context = torch.einsum("hnj,jcnhd->jcnhd", attn_probs, v)
            context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
            output = self.out_proj(context)
        return output, attn_probs

    def forward(
        self,
        query,
        key,
        value,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = key.size()
        # if False and num_rows * num_cols > 2 ** 14 and not torch.is_grad_enabled():
        if (num_rows * num_cols) > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(
                query,
                key,
                value,
                self_attn_mask,
                self_attn_padding_mask,
            )
        else:
            return self.compute_attention_update(query, key, value, self_attn_mask, self_attn_padding_mask)
