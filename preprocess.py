import torch
import esm
import os
import torch.nn.functional as F

from config import create_config
from data import read_msa, greedy_select
from tqdm import tqdm

config = create_config()

_, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()

seq_encoder, seq_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
seq_encoder.to("cuda:0")
seq_batch_converter = seq_alphabet.get_batch_converter()

for filename in tqdm(os.listdir(config.data.train_dataset_path)[:100]):
    f = os.path.join(os.path.join(config.data.train_dataset_path, filename), "uniclust30.a3m")
    msa = read_msa(f)
    if (len(msa[0][1]) <= config.data.max_sequence_len and len(msa) >= 34):
        msas = [msa[2:]]
        seqs = [msa[0]]
        
        filtered_msas = []
        for msa in msas:
            filtered_msas.append(greedy_select(msa, 32))
        
        _, _, msa_tokens = msa_batch_converter(filtered_msas)
        _, _, seq_tokens = seq_batch_converter(seqs)
        seq_batch_lens = (seq_tokens != seq_alphabet.padding_idx).sum(1)
        
        with torch.no_grad():
            raw_seq_embeddings = seq_encoder(seq_tokens.to("cuda:0"), repr_layers=[6], return_contacts=True)
            raw_pairwise_embeddings = raw_seq_embeddings["attentions"]
            raw_seq_embeddings = raw_seq_embeddings["representations"][6]
            
        # Symmetrize
        batch_size, layers, heads, seqlen, _ = raw_pairwise_embeddings.size()
        raw_pairwise_embeddings = raw_pairwise_embeddings.view(batch_size, layers * heads, seqlen, seqlen)
        raw_pairwise_embeddings = raw_pairwise_embeddings + raw_pairwise_embeddings.transpose(-1, -2)
                
        pairwise_seq_embeddings = []
        single_seq_embeddings = []
        mask = torch.zeros(seq_tokens.size(0), config.data.max_sequence_len, dtype=torch.bool).to("cuda:0")
        for i, tokens_len in enumerate(seq_batch_lens):
            pairwise_seq_repr = raw_pairwise_embeddings[i, :, 1 : tokens_len - 1, 1 : tokens_len - 1]
            pairwise_seq_repr = pairwise_seq_repr.permute(1,2,0)
            single_seq_repr = raw_seq_embeddings[i, 1 : tokens_len - 1]
            
            mask[i, : single_seq_repr.size(0)] = True
            pairwise_seq_repr = F.pad(
                pairwise_seq_repr, 
                (0, 0,
                    0, config.data.max_sequence_len - pairwise_seq_repr.size(0), 
                    0, config.data.max_sequence_len - pairwise_seq_repr.size(0)), 
                "constant", 
                seq_alphabet.padding_idx
            )
            
            single_seq_repr = F.pad(
                single_seq_repr, 
                (0, 0, 0, config.data.max_sequence_len - single_seq_repr.size(0)), 
                "constant", 
                seq_alphabet.padding_idx
            )
            
            pairwise_seq_embeddings.append(pairwise_seq_repr)
            single_seq_embeddings.append(single_seq_repr)
        
        pairwise_seq_embeddings = torch.stack(pairwise_seq_embeddings)
        single_seq_embeddings = torch.stack(single_seq_embeddings)
        
        msa_tokens = msa_tokens[:,:,1:].to("cuda:0")
        msa_tokens = F.pad(
            msa_tokens,
            (0, config.data.max_sequence_len - msa_tokens.size(-1)),
            "constant",
            msa_alphabet.padding_idx
        )
        
        if not os.path.exists(f"databases/dummy/{filename}"):
            os.mkdir(f"databases/dummy/{filename}")
        
        torch.save(single_seq_embeddings.squeeze(0).to("cpu"), f"databases/dummy/{filename}/single_repr.pt")
        torch.save(pairwise_seq_embeddings.squeeze(0).to("cpu"), f"databases/dummy/{filename}/pairwise_repr.pt")
        torch.save(msa_tokens.squeeze(0).to("cpu"), f"databases/dummy/{filename}/msa_tokens.pt")
        torch.save(mask.squeeze(0).to("cpu"), f"databases/dummy/{filename}/mask.pt")
        # print(single_seq_embeddings.device)
        # print(pairwise_seq_embeddings.device)
        # print(msa_tokens.device)
        # print(mask.device)

            
        
            
            
    