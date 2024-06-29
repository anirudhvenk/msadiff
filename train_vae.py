import torch
import esm
import torch.nn.functional as F

from model import MSAVAE, MSAEncoder, MSADecoder
from loss import Critic
from torch.utils.data import DataLoader
from data import read_msa, greedy_select, MSADataset
from config import create_config
from tqdm import tqdm

config = create_config()

torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)

_, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()

seq_encoder, seq_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
seq_encoder.to(config.device)
seq_batch_converter = seq_alphabet.get_batch_converter()

def collate_fn(data):
    seqs, msas = zip(*data)
    _, _, msa_tokens = msa_batch_converter(msas)
    _, _, seq_tokens = seq_batch_converter(seqs)
    seq_batch_lens = (seq_tokens != seq_alphabet.padding_idx).sum(1)
    
    with torch.no_grad():
        raw_seq_embeddings = seq_encoder(seq_tokens, repr_layers=[6])["representations"][6]
    
    seq_embeddings = []
    mask = torch.ones(seq_tokens.size(0), config.data.max_sequence_len, dtype=torch.bool)
    for i, tokens_len in enumerate(seq_batch_lens):
        seq_repr = raw_seq_embeddings[i, 1 : tokens_len - 1]
        mask[i, : seq_repr.size(0)] = False
        seq_repr = F.pad(
            seq_repr, 
            (0, 0, 0, config.data.max_sequence_len - seq_repr.size(0)), 
            "constant", 
            seq_alphabet.padding_idx
        )
        seq_embeddings.append(seq_repr)
    seq_embeddings = torch.stack(seq_embeddings)
    
    msa_tokens = msa_tokens[:,:,1:]
    msa_tokens = F.pad(
        msa_tokens,
        (0, config.data.max_sequence_len - msa_tokens.size(-1)),
        "constant",
        msa_alphabet.padding_idx
    )
    
    return seq_embeddings, msa_tokens, mask

train_dataset = MSADataset(config)
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config.data.batch_size)

msavae = MSAVAE(config).to(config.device)
msaencoder = MSAEncoder(config)
msadecoder = MSADecoder(config)
loss = Critic(config)
optimizer = torch.optim.Adam(msavae.parameters())

print("encoder: ", sum(p.numel() for p in msaencoder.parameters() if p.requires_grad))
print("decoder: ", sum(p.numel() for p in msadecoder.parameters() if p.requires_grad))

for e in range(20):
    for seq_embeddings, msa_tokens, mask in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        pred_msa, perm, mu, logvar = msavae(seq_embeddings, msa_tokens, mask)
        loss_dict = loss(msa_tokens, pred_msa, mask, perm, mu, logvar)
        
        loss_dict["loss"].backward()
        optimizer.step()
        
        print(loss_dict)