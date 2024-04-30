import torch
import torch.nn as nn
import math
from typing import Optional
from modules import AxialTransformerLayer
from esm.modules import ESM1bLayerNorm
from preprocess import MSADataset
from torch.utils.data import Dataset, DataLoader

class MSATransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.dropout_module = nn.Dropout(self.args["dropout"])
        self.axial_transformer_layer = AxialTransformerLayer(
            self.args["seq_embed_dim"],
            self.args["embed_dim"],
            self.args["ffn_embed_dim"],
            self.args["attention_heads"],
            self.args["dropout"],
            self.args["attention_dropout"],
            self.args["activation_dropout"],
            self.args["max_tokens"]
        )
        
        self.emb_layer_norm_before = ESM1bLayerNorm(self.args["embed_dim"])
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args["embed_dim"])
        
    def forward(self, seq, msa):
        msa = self.emb_layer_norm_before(msa)
        msa = self.dropout_module(msa)
        
        # B x R x C x D -> R x C x B x D
        msa = msa.permute(1, 2, 0, 3)
        seq = seq.permute(1, 0, 2)
        
        msa = self.axial_transformer_layer(x=msa, query=seq)
        msa = self.emb_layer_norm_after(msa)
        msa = msa.permute(2, 0, 1, 3)
             
        return msa

TransformerBlock = MSATransformerBlock 
   
class TransformerEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_hidden_layers = 4
        self.hidden_size = args["embed_dim"]
        self.input_blocks = torch.nn.ModuleList(
            [TransformerBlock(args) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.output_blocks = torch.nn.ModuleList(
            [TransformerBlock(args) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.time_layers = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)]
        )
        self.self_cond_layers = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(0, self.num_hidden_layers)]
        )

    def forward(
            self,
            x: torch.Tensor,
            query: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            emb_t=None,
            x_0_self_cond=None,
    ):
        x_input_list = []

        for i, block in enumerate(self.input_blocks):
            x_input_list.append(x)
            x = x + self.time_layers[i](emb_t) + self.self_cond_layers[i](x_0_self_cond)
            x = block(
                seq=query,
                msa=x,
                # attention_mask=attention_mask
            )

        for i, block in enumerate(self.output_blocks):
            ind = i + self.num_hidden_layers // 2
            x = x + x_input_list.pop() + self.time_layers[ind](emb_t) + self.self_cond_layers[ind](x_0_self_cond)
            x = block(
                seq=query,
                msa=x,
                # attention_mask=attention_mask
            )

        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MSAScoreEstimatorEMB(nn.Module):
    def __init__(self, args):
        super(MSAScoreEstimatorEMB, self).__init__()

        hidden_layer_dim = args["embed_dim"]
        self._hidden_layer_dim = hidden_layer_dim
        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_layer_dim * 2, hidden_layer_dim)
        )

        self.encoder = TransformerEncoder(args)

        self._max_position_embeddings = args["max_position_embeddings"]
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, args["num_rows"], -1)))
        self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self._hidden_layer_dim)

    def get_extended_attention_mask(self, attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
            self,
            x_t: torch.Tensor,
            query: torch.Tensor,
            time_t: Optional[torch.Tensor] = None,
            attention_mask=None,
            x_0_self_cond=None,
    ):
        assert time_t is not None

        emb_t = timestep_embedding(time_t, self._hidden_layer_dim)
        hidden_t = self.time_emb(emb_t)
        hidden_t = hidden_t[:, None, :].unsqueeze(1)

        seq_length = query.size(1)
        position_ids = self.position_ids[:, :, :seq_length]
        emb_pos = self.position_embeddings(position_ids)

        emb_x = x_t
        hidden_state = emb_x + emb_pos

        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(
                attention_mask=attention_mask,
                dtype=hidden_state.dtype
            )

        output = self.encoder(
            x=hidden_state,
            query=query,
            attention_mask=attention_mask,
            emb_t=hidden_t,
            x_0_self_cond=x_0_self_cond,
        )

        return output
