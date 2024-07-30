import torch
import esm
import os
import numpy as np
import string
import pickle
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from typing import List, Tuple
from tqdm import tqdm
from scipy.spatial.distance import cdist
from config import create_config
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
from esm.data import BatchConverter

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)
    
def read_msa(filename: str) -> List[Tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

# Select sequences from the MSA to maximize the hamming distance
# Alternatively, can use hhfilter 
def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa
    
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)

    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

class MSADataset(Dataset):
    def __init__(self, config, test=False):
        self.seqs = []
        self.msas = []
        
        if not test:
            for filename in tqdm(os.listdir(config.data.train_dataset_path)):
                f = os.path.join(os.path.join(config.data.train_dataset_path, filename), "uniclust30.a3m")
                msa = read_msa(f)
                if (len(msa[0][1]) <= config.data.max_sequence_len and len(msa) >= config.data.msa_depth + 1):
                    self.msas.append(msa[1:])
                    self.seqs.append(msa[0])
        else:
            for filename in tqdm(os.listdir(config.data.test_dataset_path)):
                msa = read_msa(os.path.join(config.data.test_dataset_path, filename))
                if (len(msa[0][1]) <= config.data.max_sequence_len and len(msa) >= config.data.msa_depth + 1):
                    self.msas.append(msa[1:])
                    self.seqs.append(msa[0])
        
    def __len__(self):
        return len(self.msas)
    
    def __getitem__(self, idx):
        return self.seqs[idx], self.msas[idx]
    
_, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()

seq_encoder, seq_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
seq_encoder.to("cuda:0")
seq_batch_converter = seq_alphabet.get_batch_converter()
    
def pad_msa_sequence(config, seqs, msas):
    filtered_msas = []
    for msa in msas:
        filtered_msas.append(greedy_select(msa, config.data.msa_depth))
    
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
    
    # return torch.randn_like(single_seq_embeddings), torch.randn_like(pairwise_seq_embeddings), msa_tokens, mask
    return single_seq_embeddings, pairwise_seq_embeddings , msa_tokens, mask