import torch
import esm
import os
import numpy as np
import string
import pickle
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from typing import List, Tuple
from tqdm import tqdm
from scipy.spatial.distance import cdist
from config import create_config
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
from esm.data import BatchConverter

config = create_config()

_, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()

_, seq_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
seq_batch_converter = seq_alphabet.get_batch_converter()

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

def collate_fn(data):
    seqs, msas = zip(*data)
    msas_filtered = [greedy_select(msa, num_seqs=config.data.num_rows) for msa in msas]
    _, _, msa_tokens = msa_batch_converter(msas_filtered)
    _, _, seq_tokens = seq_batch_converter(seqs)
    
    return seq_tokens, msa_tokens

def encode(seq_tokens, msa_tokens, msa_encoder, seq_encoder, device, decoding=False):
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            msa_embeddings = msa_encoder(
                msa_tokens.to(device), 
                repr_layers=[12]
            )["representations"][12]
            seq_embeddings = seq_encoder(
                seq_tokens.to(device), 
                repr_layers=[6]
            )["representations"][6][:,1:,:]
            
    msa_mean = torch.mean(msa_embeddings, dim=-1).unsqueeze(-1)
    msa_std = torch.std(msa_embeddings, dim=-1).unsqueeze(-1)
    msa_embeddings_normalized = (msa_embeddings - msa_mean) / msa_std
    
    if decoding:
        return msa_embeddings_normalized, msa_mean, msa_std
    
    return seq_embeddings, msa_embeddings_normalized

# def decode(msa_embeddings, decoder):
    # with torch.no_grad():
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            


class MSADataset(Dataset):
    def __init__(self, data, num_alignments):
        self.seqs = []
        self.msas = []
        
        for alignment in tqdm(os.listdir(data)[:num_alignments]):
            for alignment_data in os.listdir(os.path.join(data, alignment)):
                msa = read_msa(os.path.join(os.path.join(data, alignment), alignment_data))
                if (len(msa[0][1]) <= 256 and len(msa) >= config.data.num_rows):
                    self.msas.append(greedy_select(msa, num_seqs=config.data.num_rows))
                    self.seqs.append(msa[0])
        
    def __len__(self):
        return len(self.msas)
    
    def __getitem__(self, idx):
        return self.seqs[idx], self.msas[idx]
