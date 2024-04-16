import torch
import math
from encoders import ESM2EncoderModel
from msa_encoder import MSADataset


hidden_layer_dim = 320

time_emb = torch.nn.Sequential(
    torch.nn.Linear(hidden_layer_dim, hidden_layer_dim * 2),
    torch.nn.SiLU(),
    torch.nn.Linear(hidden_layer_dim * 2, hidden_layer_dim)
)

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


# time_t = torch.tensor([0.6,0.5,0.5])
# emb_t = timestep_embedding(time_t, 320)
# hidden_t = time_emb(emb_t)
# hidden_t = hidden_t[:, None, :]

# print(hidden_t.shape)


encoder = ESM2EncoderModel()
outputs, tokenized = encoder.batch_encode(["AGRY"])