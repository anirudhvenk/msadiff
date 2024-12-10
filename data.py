import torch
import os
import string
import pickle
import torch.nn.functional as F
import random
import numpy as np
import functools
from config import create_config

from scipy.spatial.distance import cdist
from torch.utils.data import IterableDataset, Dataset, DataLoader, DistributedSampler
from Bio import SeqIO
from typing import List, Tuple
from tqdm import tqdm
from torch.utils.data import get_worker_info
from torch.distributed import get_rank, get_world_size

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-"

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
    def __init__(self, base_path, max_msa_depth):
        self.base_path = base_path
        self.max_msa_depth = max_msa_depth
        self.msa_paths = os.listdir(base_path)        

    def __len__(self):
        return len(self.msa_paths)
    
    def __getitem__(self, idx):
        msa_data = read_msa(os.path.join(
            self.base_path, os.path.join(self.msa_paths[idx], "non_pairing.a3m"))
        )[1:]
        msa_data = random.choices(msa_data, k=self.max_msa_depth)
        msa_emb = torch.load(
            os.path.join(self.base_path, os.path.join(self.msa_paths[idx], "msa_embedding.pt")),
            map_location="cpu"
        )
        msa_emb = msa_emb.repeat(self.max_msa_depth, 1, 1, 1)

        sel_seqs = [torch.tensor([ALPHABET.index(r) for r in s]) for _, s in msa_data]
        sel_seqs = F.one_hot(torch.stack(sel_seqs), num_classes=len(ALPHABET))
        
        return sel_seqs, msa_emb