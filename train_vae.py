import torch
<<<<<<< HEAD
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
from typing import Optional

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
encoder = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D", add_cross_attention=False, is_decoder=False)

sequences = ["ATGY"]


tokenized = tokenizer(
    sequences, 
    return_attention_mask=True, 
    return_tensors="pt", 
    truncation=True, 
    padding=True, 
    max_length=256
)

with torch.no_grad():
    embeddings = encoder(**tokenized).last_hidden_state

print(embeddings.shape)
=======
import esm
import torch.nn.functional as F

from model import MSAVAE, MSAEncoder
from data import read_msa, greedy_select
from config import create_config
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM

config = create_config()
msavae = MSAVAE(config)

torch.manual_seed(42)

seq_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
seq_encoder = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D", add_cross_attention=False, is_decoder=False)

msa_transformer, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer_batch_converter = msa_transformer_alphabet.get_batch_converter()
print(len(msa_transformer_alphabet.all_toks))

msa = read_msa("databases/data/a3m/1vgj_1_A.a3m")
msa = greedy_select(msa, num_seqs=config.data.msa_depth)
_, _, msa_tokens = msa_transformer_batch_converter(msa)
msa_tokens = msa_tokens[:,:,1:]
msa_tokens = F.pad(msa_tokens, (0, 256-msa_tokens.size(-1)), "constant", config.data.padding_idx)
perm_indices = torch.randperm(msa_tokens.size(1))
permuted_msa_tokens = msa_tokens[:, perm_indices, :]

sequences = ["ATGY"]
seq_tokens = seq_tokenizer(
    sequences,
    return_attention_mask=True, 
    return_tensors="pt", 
    truncation=True,
    padding="max_length", 
    max_length=256+2
)
with torch.no_grad():
    encoded_seq = seq_encoder(**seq_tokens).last_hidden_state
encoded_seq = encoded_seq[:,1:-1,:]
permuted_encoded_seq = encoded_seq.clone()

mask = seq_tokens["attention_mask"]
indices_to_remove = [0, 5]
rm_mask = torch.ones(mask.shape[1], dtype=torch.bool)
rm_mask[indices_to_remove] = False
mask = mask[:, rm_mask].bool()

msaencoder = MSAEncoder(config)
msavae(encoded_seq, msa_tokens, mask)
msavae(permuted_encoded_seq, permuted_msa_tokens, mask)
>>>>>>> f699727a7385af273b541e0138e1dd45cace2598
