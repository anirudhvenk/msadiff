import logging
import os
import json
from Bio import SeqIO
import torch
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
import re
from scipy import linalg
import argparse
import pandas as pd

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(os.getcwd(), log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def read_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    seqs = []
    for i, record in enumerate(data):
        sequence = record["protein"]
        seqs.append(sequence)
    return seqs

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

def calculate_activation_statistics(batch):

    act = batch # (B, Dim) (2048, 1024)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid_for_a_pair(batch1, batch2):

    m1, s1 = calculate_activation_statistics(batch1)
    m2, s2 = calculate_activation_statistics(batch2)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Frechet distance between two sets of sequences")
    parser.add_argument("input_fasta_dir")
    parser.add_argument("output_csv")
    parser.add_argument("exp_name")
    # parser.add_argument("--out", default="out", help="Prefix for output names")
    parser.add_argument("--cuda", default=0, help="Cuda device id", type=int)
    parser.add_argument("--batch_size", default=512, help="Batch size", type=int)
    parser.add_argument("--max_len", default=256, help="Max length of sequences", type=int)
    args = parser.parse_args()
    # logger = setup_logger(args.out + ".log")
    input_file_1 = args.input_fasta_dir

    if 'afdb' in args.exp_name:
        dataset_path = 'input_files/dataset_afdb_2.fasta'
        print(f'calcultae FID with AFDB')
    else:
        dataset_path = 'input_files/dataset_swissprot_2.fasta'
        print(f'calcultae FID with swissprot')

    input_file_2 = dataset_path
    seq_list_1 = read_fasta(input_file_1)
    seq_list_2 = read_fasta(input_file_2)

    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16)
    encoder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', torch_dtype=torch.float16)
    encoder = encoder.eval().to(device)
    print(f"Model is loaded successfully.")
    print(f"Calculating embeddings...")
    embeddings_1 = create_t5_embeds(encoder, tokenizer, seq_list_1, device)
    embeddings_2 = create_t5_embeds(encoder, tokenizer, seq_list_2, device)

    print(f"Calculating FIDs...")
    fid_value = calculate_fid_for_a_pair(embeddings_1, embeddings_2)


    print(f"Frechet distance: {fid_value}")

    if os.path.exists(args.output_csv):
        res_df = pd.read_csv(args.output_csv)
    else:
        res_df = pd.DataFrame({"file": [], "FID": []})
    
    res_df = res_df.set_index('file')
    res_df.loc[args.exp_name, "FID"] = fid_value
    res_df = res_df.reset_index()

    res_df.to_csv(args.output_csv, index = False)

    
