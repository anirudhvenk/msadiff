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

def pad_msa_sequence(config, seq, msa, esm_modules):
    seq_encoder, seq_alphabet, seq_batch_converter, msa_alphabet, msa_batch_converter = esm_modules
    filtered_msa = greedy_select(msa, config.data.msa_depth)
    
    _, _, msa_tokens = msa_batch_converter([filtered_msa])
    _, _, seq_tokens = seq_batch_converter([seq])
    seq_batch_lens = (seq_tokens != seq_alphabet.padding_idx).sum(1)
    
    with torch.no_grad():
        raw_seq_embeddings = seq_encoder(seq_tokens.to(config.device), repr_layers=[6], return_contacts=True)
        raw_pairwise_embeddings = raw_seq_embeddings["attentions"]
        raw_seq_embeddings = raw_seq_embeddings["representations"][6]
        
    # Symmetrize
    batch_size, layers, heads, seqlen, _ = raw_pairwise_embeddings.size()
    raw_pairwise_embeddings = raw_pairwise_embeddings.view(batch_size, layers * heads, seqlen, seqlen)
    raw_pairwise_embeddings = raw_pairwise_embeddings + raw_pairwise_embeddings.transpose(-1, -2)
            
    pairwise_seq_embeddings = []
    single_seq_embeddings = []
    mask = torch.zeros(seq_tokens.size(0), config.data.max_sequence_len, dtype=torch.bool).to(config.device)
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
    
    msa_tokens = msa_tokens[:,:,1:].to(config.device)
    msa_tokens = F.pad(
        msa_tokens,
        (0, config.data.max_sequence_len - msa_tokens.size(-1)),
        "constant",
        msa_alphabet.padding_idx
    )
    
    return single_seq_embeddings, pairwise_seq_embeddings , msa_tokens, mask

class MSADataset(Dataset):
    def __init__(self, config, test=False):
        _, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        msa_batch_converter = msa_alphabet.get_batch_converter()

        seq_encoder, seq_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        seq_encoder.to(config.device)
        seq_batch_converter = seq_alphabet.get_batch_converter()        
        
        self.single_seq_reprs = []
        self.pairwise_seq_reprs = []
        self.msa_tokens = []
        self.masks = []
        
        if test:
            for filename in tqdm(os.listdir(config.data.test_dataset_path)):
                msa = read_msa(os.path.join(config.data.test_dataset_path, filename))
                if (len(msa[0][1]) <= config.data.max_sequence_len and len(msa) >= config.data.msa_depth + 1):
                    single_seq_embeddings, pairwise_seq_embeddings, msa_tokens, mask = pad_msa_sequence(
                        config, 
                        msa[0], 
                        msa[1:],
                        (seq_encoder, seq_alphabet, seq_batch_converter, msa_alphabet, msa_batch_converter)
                    )
                    self.single_seq_reprs.append(single_seq_embeddings.squeeze(0).cpu().to(torch.bfloat16))
                    self.pairwise_seq_reprs.append(pairwise_seq_embeddings.squeeze(0).cpu().to(torch.bfloat16))
                    self.msa_tokens.append(msa_tokens.squeeze(0).cpu().to(torch.bfloat16))
                    self.masks.append(mask.squeeze(0).cpu().to(torch.bfloat16))
        else:
            for dirname in os.listdir(config.data.train_dataset_path)[:1]:
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
                        self.single_seq_reprs.append(single_seq_embeddings.squeeze(0).cpu().to(torch.bfloat16))
                        self.pairwise_seq_reprs.append(pairwise_seq_embeddings.squeeze(0).cpu().to(torch.bfloat16))
                        self.msa_tokens.append(msa_tokens.squeeze(0).cpu().to(torch.bfloat16))
                        self.masks.append(mask.squeeze(0).cpu().to(torch.bfloat16))
                
    def __len__(self):
        return len(self.msa_tokens)
    
    def __getitem__(self, idx):
        return self.single_seq_reprs[idx], self.pairwise_seq_reprs[idx], self.msa_tokens[idx], self.masks[idx]
