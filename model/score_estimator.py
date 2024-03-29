import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput, \
    apply_chunking_to_forward




class BertBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )
        attention_output = self_attention_outputs[0]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            attention_output = cross_attention_outputs[0]

        outputs = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


TransformerBlock = BertBlock


class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_hidden_layers = 12
        self.hidden_size = config.hidden_size
        self.input_blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
        )
        self.output_blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(0, self.num_hidden_layers // 2)]
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
            attention_mask: Optional[torch.FloatTensor] = None,
            emb_t=None,
            x_0_self_cond=None,
    ):
        x_input_list = []

        for i, block in enumerate(self.input_blocks):
            x_input_list.append(x)
            x = x + self.time_layers[i](emb_t) + self.self_cond_layers[i](x_0_self_cond)
            x = block(
                hidden_states=x,
                attention_mask=attention_mask
            )

        for i, block in enumerate(self.output_blocks):
            ind = i + self.num_hidden_layers // 2
            x = x + x_input_list.pop() + self.time_layers[ind](emb_t) + self.self_cond_layers[ind](x_0_self_cond)
            x = block(
                hidden_states=x,
                attention_mask=attention_mask
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


class ScoreEstimatorEMB(nn.Module):
    def __init__(self, input_size, config):
        super(ScoreEstimatorEMB, self).__init__()

        self.input_size = input_size
        hidden_layer_dim = config.hidden_size
        self._hidden_layer_dim = hidden_layer_dim
        self.time_emb = torch.nn.Sequential(
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_layer_dim * 2, hidden_layer_dim)
        )

        self.encoder = TransformerEncoder(config)

        self._max_position_embeddings = config.max_position_embeddings
        self.register_buffer("position_ids", torch.arange(self._max_position_embeddings).expand((1, -1)))
        self.position_embeddings = torch.nn.Embedding(self._max_position_embeddings, self._hidden_layer_dim)

    def get_extended_attention_mask(self, attention_mask, dtype):
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
            self,
            x_t: torch.Tensor,
            time_t: Optional[torch.Tensor] = None,
            attention_mask=None,
            x_0_self_cond=None,
    ):
        assert time_t is not None

        emb_t = timestep_embedding(time_t, self._hidden_layer_dim)
        hidden_t = self.time_emb(emb_t)
        hidden_t = hidden_t[:, None, :]

        seq_length = x_t.size(1)
        position_ids = self.position_ids[:, : seq_length]
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
            attention_mask=attention_mask,
            emb_t=hidden_t,
            x_0_self_cond=x_0_self_cond,
        )

        return output
