
import logging
import os
import json
import torch
from Bio import SeqIO
import torch
import numpy as np
import re
from scipy import linalg
from transformers import T5Tokenizer, T5EncoderModel



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

def calculate_fid_for_files(input_file_1, input_file_2):
    if input_file_1.endswith(".json"):
        seq_list_1 = read_json(input_file_1)
    elif input_file_1.endswith(".fasta") or input_file_1.endswith(".fa"):
        seq_list_1 = read_fasta(input_file_1)
    else:
        raise ValueError("Input file must be JSON or FASTA")
    if input_file_2.endswith(".json"):
        seq_list_2 = read_json(input_file_2)
    elif input_file_2.endswith(".fasta") or input_file_2.endswith(".fa"):
        seq_list_2 = read_fasta(input_file_2)
    else:
        raise ValueError("Input file must be JSON or FASTA")

    device = torch.device('cuda:0')
    print(f'Using device: {device}')

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", torch_dtype=torch.float16)
    encoder = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', torch_dtype=torch.float16)
    encoder = encoder.eval().to(device)
    embeddings_1 = create_t5_embeds(encoder, tokenizer, seq_list_1, device)
    embeddings_2 = create_t5_embeds(encoder, tokenizer, seq_list_2, device)

    fid_value = calculate_fid_for_a_pair(embeddings_1, embeddings_2)
    return fid_value
    