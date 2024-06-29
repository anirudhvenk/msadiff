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

class MSADataset(Dataset):
    def __init__(self, config):
        self.seqs = []
        self.msas = []
        
        for filename in tqdm(os.listdir(config.data.train_dataset_path)[:10]):
            msa = read_msa(os.path.join(config.data.train_dataset_path, filename))
            if (len(msa[0][1]) <= config.data.max_sequence_len):
                msa_filtered = greedy_select(msa, config.data.msa_depth+1)
                self.msas.append(msa_filtered[1:])
                self.seqs.append(msa[0])
        
    def __len__(self):
        return len(self.msas)
    
    def __getitem__(self, idx):
        return self.seqs[idx], self.msas[idx]