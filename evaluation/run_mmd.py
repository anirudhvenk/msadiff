import logging
import os
from Bio import SeqIO
import torch
from transformers import T5Tokenizer, T5EncoderModel
import re
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance_matrix


def read_fasta(fasta_file):
    records = SeqIO.parse(fasta_file, "fasta")
    seqs = []
    for i, record in enumerate(records):
        sequence = str(record.seq)
        seqs.append(sequence)
    return seqs

def create_t5_embeds(encoder, tokenizer, raw_seq_list, device, batch_size=256, max_len=256):
    seq_list = []
    len_seqs = []
    for seq in raw_seq_list:
        init_seq = " ".join(list(re.sub(r"[UZOB]", "X", seq)))
        seq_list.append(init_seq)
        len_seqs.append(len(seq))
    inputs = tokenizer(seq_list, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    
    embeddings = np.zeros((len(seq_list), 1024))
    for i in range(0, inputs.input_ids.shape[0], batch_size):
        batch = {key: inputs[key][i:i+batch_size, :].to(device) for key in inputs.keys()}
        with torch.no_grad():
            batch_embeddings = encoder(**batch).last_hidden_state.cpu().numpy()
        batch_embeddings = np.mean(batch_embeddings, axis=1)
        embeddings[i:i+batch_size, :] = batch_embeddings
    return embeddings


def load_plm(device):
    transformer_link = "Rostlab/prot_t5_xl_uniref50"
    # # Load the pretrained ProtT5 model and tokenizer
    encoder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', torch_dtype=torch.float16).to(device)
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, torch_dtype=torch.float16)

    encoder.eval()
    return tokenizer, encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Frechet distance between two sets of sequences")
    parser.add_argument("input_fasta")
    parser.add_argument("output_csv")
    parser.add_argument("exp_name")
    parser.add_argument("--cuda", default=0, help="Cuda device id", type=int)
    parser.add_argument("--batch_size", default=512, help="Batch size", type=int)
    parser.add_argument("--max_len", default=254, help="Max length of sequences", type=int)
    args = parser.parse_args()
    # logger = setup_logger(args.out + ".log")
    input_file_1 = args.input_fasta

    if 'afdb' in args.exp_name:
        dataset_path = 'input_files/dataset_afdb_2.fasta'
        # dist_col = 'dataset_afdb'
        print(f'calcultae MMD with AFDB')
    else:
        dataset_path = 'input_files/dataset_swissprot_2.fasta'
        # dist_col = 'dataset_swissprot'
        print(f'calcultae MMD with swissprot')

    input_file_2 = dataset_path
    seq_list_1 = read_fasta(input_file_1)
    seq_list_2 = read_fasta(input_file_2)

    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    tokenizer, encoder = load_plm(device)
    print(f"Model is loaded successfully.")
    
    print('calculate embedings')
    print(f"Calculating embeddings...")
    embeddings_1 = create_t5_embeds(
        encoder, 
        tokenizer, 
        seq_list_1, 
        device, 
        batch_size = args.batch_size,
        max_len = args.max_len,
        )
    embeddings_2 = create_t5_embeds(
        encoder, 
        tokenizer, 
        seq_list_2, 
        device, 
        batch_size = args.batch_size,
        max_len = args.max_len,
        )

    print('calculate kernels')
    SIGMA = 1
    inner_data_dist = distance_matrix(embeddings_2, embeddings_2)
    inner_data_gaus = np.exp(-inner_data_dist/(2*SIGMA)).mean()
    
    inner_gen_dist = distance_matrix(embeddings_1, embeddings_1)
    inner_gen_gaus = np.exp(-inner_gen_dist/(2*SIGMA)).mean()

    outer_dist = distance_matrix(embeddings_1, embeddings_2)
    outer_gauss = np.exp(-outer_dist/(2*SIGMA)).mean()

    mmd  = inner_data_gaus + inner_gen_gaus - 2*outer_gauss
    print(args.exp_name, 'Result MMD:', mmd)

    if os.path.exists(args.output_csv):
        res_df = pd.read_csv(args.output_csv)
    else:
        res_df = pd.DataFrame({"file": [], "MMD_mean_prott5": []})
    
    res_df = res_df.set_index('file')
    res_df.loc[args.exp_name, "MMD_mean_prott5"] = mmd
    res_df = res_df.reset_index()

    res_df.to_csv(args.output_csv, index = False)