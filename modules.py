import torch
import torch.nn as nn
import torch.nn.functional as F

from msa_encoder import MSADataset
from torch.utils.data import DataLoader

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

msa_datset = MSADataset("./data")
train_dataloader = DataLoader(msa_datset, batch_size=1, shuffle=False)
sample = next(iter(train_dataloader)).cuda()
batch_size, num_rows, num_cols, embed_dim = sample.size()
# sample = sample.view(num_rows, num_cols, batch_size, embed_dim)

embed_tokens = nn.Embedding(
    512, 768, padding_idx=1
)

emb_pos = torch.nn.Embedding(512, 768)
position_ids = torch.arange(512).expand(1, 128, -1)[:,:,:num_cols]

# print(position_ids.shape)

print(emb_pos(position_ids).shape)
# print(sample.ndims)

# print(embed_tokens(sample).shape)

# embed_positions = LearnedPositionalEmbedding(num_embeddings=512, embedding_dim=320, padding_idx=1).cuda()
# print(embed_positions(sample).shape)

