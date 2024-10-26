import torch
import esm
import os
import torch.nn.functional as F

from config import create_config
from data import read_msa, greedy_select, pad_msa_sequence
from tqdm import tqdm

config = create_config()
config.device = "cuda:0"

_, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()

seq_encoder, seq_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
seq_encoder.to("cuda:0")
seq_batch_converter = seq_alphabet.get_batch_converter()            
        
for dirname in os.listdir(config.data.train_dataset_path)[-1:]:
    alignment_cluster = os.path.join(config.data.train_dataset_path, dirname)
    for filename in tqdm(os.listdir(alignment_cluster)[:80000]):
        f = os.path.join(os.path.join(alignment_cluster, filename), "uniclust30.a3m")
        msa = read_msa(f)
        if (len(msa[0][1]) <= config.data.max_sequence_len and len(msa) >= config.data.msa_depth + 2):
            single_seq_embeddings, pairwise_seq_embeddings, msa_tokens, mask = pad_msa_sequence(
                config, 
                msa[0], 
                msa[2:],
                (seq_encoder, seq_alphabet, seq_batch_converter, msa_alphabet, msa_batch_converter)
            )
            if not os.path.exists(f"databases/train/{dirname}/{filename}"):
                os.makedirs(f"databases/train/{dirname}/{filename}")
            
            torch.save(single_seq_embeddings.squeeze(0).cpu().to(torch.bfloat16), f"databases/train/{dirname}/{filename}/single_repr.pt")
            torch.save(pairwise_seq_embeddings.squeeze(0).cpu().to(torch.bfloat16), f"databases/train/{dirname}/{filename}/pairwise_repr.pt")
            torch.save(msa_tokens.squeeze(0).cpu().to(torch.bfloat16), f"databases/train/{dirname}/{filename}/msa_tokens.pt")
            torch.save(mask.squeeze(0).cpu().to(torch.bfloat16), f"databases/train/{dirname}/{filename}/mask.pt")
