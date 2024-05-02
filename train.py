import torch
from model import MSAScoreEstimatorEMB
from preprocess import MSADataset, greedy_select
from torch.utils.data import DataLoader
from diffusion_utils.diffusion_dynamic_sde import create_sde, create_solver
from config import create_config
from typing import Dict
import torch.distributed as dist
import os
import esm
import numpy as np
import pickle as pkl
from tqdm import tqdm

device = "cuda:1"

msa_encoder, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()
msa_encoder.to(device)
msa_encoder.eval()

seq_encoder, seq_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
seq_batch_converter = seq_alphabet.get_batch_converter()
seq_encoder.to(device)
seq_encoder.eval()

def collate_fn(data):
    seqs, msas = zip(*data)
    msas_filtered = [greedy_select(msa, num_seqs=128) for msa in msas]
    _, _, msa_tokens = msa_batch_converter(msas_filtered)
    _, _, seq_tokens = seq_batch_converter(seqs)
    
    with torch.no_grad():
        msa_embeddings = msa_encoder(msa_tokens.to(device), repr_layers=[12])["representations"][12]
        seq_embeddings = seq_encoder(seq_tokens.to(device), repr_layers=[6])["representations"][6]
    
    return seq_embeddings, msa_embeddings

def sample_time(batch_size: int, eps: float = 1e-5):
    return torch.FloatTensor(batch_size).uniform_() * (sde.T - eps) + eps

def calc_score(model, x_t, query, t, mask=None, x_0_self_cond=None) -> Dict[str, torch.Tensor]:
    params = sde.marginal_params_tensor(x_t, t)
    x_0 = model(x_t=x_t, query=query, time_t=t, attention_mask=mask, x_0_self_cond=x_0_self_cond)
    eps_theta = (x_t - params["alpha"] * x_0) / params["std"]
    score = -eps_theta / params["std"]

    return {
        "score": score,
        "x_0": x_0,
        "eps_theta": eps_theta
    }
    
def mse_loss(inputs, targets, mask):
    if mask is None:
        mask = torch.ones(
            (targets.shape[0], targets.shape[1], targets.shape[2]),
            device=device,
            requires_grad=False,
            dtype=torch.int64,
        )
    losses = torch.mean(torch.square(inputs - targets), dim=-1)
    losses = losses * mask
    loss = torch.sum(losses) / torch.sum(mask)
    return loss

args = {
    "seq_embed_dim": 320,
    "embed_dim": 768,
    "ffn_embed_dim": 3072,
    "attention_heads": 12,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.1,
    "num_rows": 128,
    "max_tokens": 2 ** 14,
    "max_position_embeddings": 1024
}

train_dataset = MSADataset("./databases/msa_transformer/data/a3m")
train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)

config = create_config()
sde = create_sde(config=config, score_fn=calc_score)
diff_eq_solver = create_solver(config, sde, ode_sampling=config.sde.ode_sampling)
score_estimator = MSAScoreEstimatorEMB(args).to(device)

optimizer = torch.optim.AdamW(
    score_estimator.parameters(),
    lr=config.optim.lr,
    weight_decay=config.optim.weight_decay,
    betas=(config.optim.beta_1, config.optim.beta_2),
    eps=config.optim.eps
)

for epoch in range(10):
    total_loss_x_0 = 0
    total_loss_eps = 0
    total_loss_score = 0
    
    for seq, msa in tqdm(train_loader):
        optimizer.zero_grad()
        
        # Noising
        clean_x = msa
        batch_size = clean_x.size(0)
        t = sample_time(batch_size).to(device)
        marg_forward = sde.marginal_forward(clean_x, t)
        x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']

        # self-cond estimate
        x_0_self_cond = torch.zeros_like(x_t, dtype=x_t.dtype)
        x_0_self_cond = score_estimator(
            x_t=x_t,
            query=seq,
            time_t=t,
            attention_mask=None,
            x_0_self_cond=x_0_self_cond
        )

        # model prediction
        scores = calc_score(
            score_estimator,
            x_t,
            seq,
            t,
            mask=None,
            x_0_self_cond=x_0_self_cond,
        )

        # MSE losses
        x_0, eps_theta, score = scores["x_0"], scores['eps_theta'], scores["score"]

        loss_x_0 = mse_loss(clean_x, x_0, mask=None)
        loss_eps = mse_loss(noise, eps_theta, mask=None)
        loss_score = mse_loss(score_clean, score, mask=None)
        
        loss_x_0.backward()
        optimizer.step()

        total_loss_x_0 += loss_x_0.item()
        total_loss_eps += loss_eps.item()
        total_loss_score += loss_score.item()
        
        torch.cuda.empty_cache()
        
    print("loss_x_0", total_loss_x_0/len(train_loader))
    print("loss_eps", total_loss_eps/len(train_loader))
    print("loss_score", total_loss_score/len(train_loader))
