import torch
from msa_score_estimator import MSAScoreEstimatorEMB
from msa_encoder import MSADataset
from torch.utils.data import DataLoader
from model.score_estimator import ScoreEstimatorEMB
from diffusion_utils.diffusion_dynamic_sde import create_sde, create_solver
from config import create_config
from typing import Dict
from utils.setup_ddp import setup_ddp
import torch.distributed as dist
import os
import numpy as np

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
            device="cpu",
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
    "max_position_embeddings": 512
}

config = create_config()
# config.local_rank = setup_ddp()
# config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
# config.device = f"cuda:{dist.get_rank()}"
# config.project_name = 'proteins'


dataset = MSADataset("./data")
dataloader = DataLoader(dataset, 1, False)

msa = next(iter(dataloader)).cuda()
seq = torch.rand((1,314,320)).cuda()
t = torch.tensor([10]).cuda()

score_estimator = MSAScoreEstimatorEMB(args).to(device="cuda:8")
model_parameters = filter(lambda p: p.requires_grad, score_estimator.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

score_estimator_2 = ScoreEstimatorEMB(input_size=320, config=config.bert_config)
model_parameters_2 = filter(lambda p: p.requires_grad, score_estimator_2.parameters())
params_2 = sum([np.prod(p.size()) for p in model_parameters_2])

score_estimator(x_t=msa, query=seq, time_t=t, x_0_self_cond=msa)

# sde = create_sde(config=config, score_fn=calc_score)
# diff_eq_solver = create_solver(config, sde, ode_sampling=config.sde.ode_sampling)

# # Noizing
# clean_x = msa
# batch_size = clean_x.size(0)
# t = sample_time(batch_size)
# marg_forward = sde.marginal_forward(clean_x, t)
# x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']

# # self-cond estimate
# x_0_self_cond = torch.zeros_like(x_t, dtype=x_t.dtype)
# x_0_self_cond = score_estimator(
#     x_t=x_t,
#     query=seq,
#     time_t=t,
#     attention_mask=None,
#     x_0_self_cond=x_0_self_cond
# )

# # model prediction
# scores = calc_score(
#     score_estimator,
#     x_t,
#     seq,
#     t,
#     mask=None,
#     x_0_self_cond=x_0_self_cond,
# )

# # MSE losses
# x_0, eps_theta, score = scores["x_0"], scores['eps_theta'], scores["score"]

# print(x_0.shape, clean_x.shape)
# print(eps_theta.shape, noise.shape)
# print(score.shape, score.shape)

# loss_x_0 = mse_loss(clean_x, x_0, mask=None)
# loss_eps = mse_loss(noise, eps_theta, mask=None)
# loss_score = mse_loss(score_clean, score, mask=None)

# print("loss_x_0", loss_x_0)
# print("loss_eps", loss_eps)
# print("loss_score", loss_score)
