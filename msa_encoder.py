import torch
import esm
import os
import numpy as np
import string

from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from typing import List, Tuple
from tqdm import tqdm
from scipy.spatial.distance import cdist

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

class MSAEncoder():
    def __init__(self):
        self.encoder, self.alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.encoder.cuda()
        self.encoder.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        
    def batch_encode(self, sequences):
        _, _, tokens = self.batch_converter(sequences)
        
        with torch.no_grad():
            embeddings = self.encoder(tokens.cuda(), repr_layers=[12])["representations"][12]
        
        return embeddings

# data_dir = "./data"
# encoder = MSAEncoder()
# all_inputs = []
# self.msa_list = []

# add normalization later
# for msa_file in tqdm(os.listdir(data_dir)):
#     msa = read_msa(os.path.join(data_dir, msa_file))
#     inputs = greedy_select(msa, num_seqs=128)
#     all_inputs.append(inputs)
    
# print(encoder.batch_encode(all_inputs).shape)

  
class MSADataset(Dataset):
    def __init__(self, data_dir):
        encoder = MSAEncoder()
        self.msa_list = []

        # add normalization later
        for msa_file in tqdm(os.listdir(data_dir)):
            msa = read_msa(os.path.join(data_dir, msa_file))
            inputs = greedy_select(msa, num_seqs=128)
            embeddings = encoder.batch_encode(inputs)
            self.msa_list.append(embeddings[0])
        
    def __len__(self):
        return len(self.msa_list)
    
    def __getitem__(self, idx):
        return self.msa_list[idx]
    
# msa_datset = MSADataset("./data")
# train_dataloader = DataLoader(msa_datset, batch_size=1, shuffle=True)
# print(next(iter(train_dataloader)))
