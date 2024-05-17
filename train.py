from model import MSAScoreEstimatorEMB
from data import MSADataset, collate_fn, encode
from torch.utils.data import DataLoader
from diffusion_utils.diffusion_dynamic_sde import create_sde, create_solver
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from config import create_config
from typing import Dict
from random import random
import torch.multiprocessing as mp
import torch
import os
import esm
import numpy as np
import pickle as pkl
from tqdm import tqdm
from torch.cuda.amp import GradScaler

class DiffusionRunner:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        train_dataset = MSADataset(self.config.data.train_dataset_path)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.data.batch_size, 
            collate_fn=collate_fn,
            pin_memory=False
        )

        self.sde = create_sde(config=self.config, score_fn=self.calc_score)
        self.diff_eq_solver = create_solver(self.config, self.sde, ode_sampling=self.config.sde.ode_sampling)
        self.score_estimator = MSAScoreEstimatorEMB(self.config).to(config.device)
        
        self.optimizer = torch.optim.AdamW(
            self.score_estimator.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps
        )

        self.grad_scaler = GradScaler()

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.FloatTensor(batch_size).uniform_() * (self.sde.T - eps) + eps

    def calc_score(self, model, x_t, query, t, cross_attn_padding_mask=None, x_0_self_cond=None) -> Dict[str, torch.Tensor]:
        params = self.sde.marginal_params_tensor(x_t, t)
        x_0 = model(
            x_t=x_t, 
            query=query, 
            time_t=t, 
            cross_attn_padding_mask=cross_attn_padding_mask, 
            x_0_self_cond=x_0_self_cond
        )
        eps_theta = (x_t - params["alpha"] * x_0) / params["std"]
        score = -eps_theta / params["std"]

        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta
        }
    
    def mse_loss(self, inputs, targets, mask):
        if mask is None:
            mask = torch.ones(
                (targets.shape[0], targets.shape[1], targets.shape[2]),
                device=self.device,
                requires_grad=False,
                dtype=torch.int64,
            )
        losses = torch.mean(torch.square(inputs - targets), dim=-1)
        losses = losses * mask
        loss = torch.sum(losses) / torch.sum(mask)
        return loss

    def calc_loss(self, seq, msa):
        # Noising
        clean_x = msa
        batch_size = clean_x.size(0)
        t = self.sample_time(batch_size).to(self.device)
        marg_forward = self.sde.marginal_forward(clean_x, t)
        x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']

        # self-cond estimate
        x_0_self_cond = torch.zeros_like(x_t, dtype=x_t.dtype)
        if random() < 0.5:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x_0_self_cond = self.score_estimator(
                    x_t=x_t,
                    query=seq,
                    time_t=t,
                    cross_attn_padding_mask=None,
                    x_0_self_cond=x_0_self_cond
                ).detach()  

        # model prediction
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scores = self.calc_score(
                self.score_estimator,
                x_t=x_t,
                query=seq,
                t=t,
                cross_attn_padding_mask=None,
                x_0_self_cond=x_0_self_cond
            )

        # MSE losses
        x_0, eps_theta, score = scores["x_0"], scores['eps_theta'], scores["score"]

        loss_x_0 = self.mse_loss(clean_x, x_0, mask=None)
        loss_eps = self.mse_loss(noise, eps_theta, mask=None)
        loss_score = self.mse_loss(score_clean, score, mask=None)
        
        return loss_x_0, loss_eps, loss_score

    def optimizer_step(self, loss):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = torch.sqrt(sum([torch.sum((t.grad).to(config.device) ** 2) for t in self.score_estimator.parameters()]))
        torch.nn.utils.clip_grad_norm_(
            self.score_estimator.parameters(),
            max_norm=self.config.optim.grad_clip_norm
        )
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        
        scale = self.grad_scaler._scale.item()
        max_scale = 2 ** 30
        min_scale = 1
        scale = np.clip(scale, min_scale, max_scale)
        self.grad_scaler.update(new_scale=scale)

    def train(self, epochs):
        print("Num params: ", sum(p.numel() for p in self.score_estimator.parameters()))

        for epoch in range(epochs):
            all_ts = []
            total_loss_x_0 = 0
            total_loss_eps = 0
            total_loss_score = 0
            
            all_ts_val = []
            val_loss_x_0 = 0
            val_loss_eps = 0
            val_loss_score = 0
            
            for seq_tokens, msa_tokens in tqdm(self.train_loader):
                seq_embeddings, msa_embeddings = encode(seq_tokens, msa_tokens)
                
                loss_x_0, loss_eps, loss_score = self.calc_loss(seq_embeddings.to(config.device), msa_embeddings.to(config.device))
                self.optimizer_step(loss_x_0)
                
                torch.cuda.empty_cache()

if __name__ == "__main__":
    config = create_config()
    
    runner = DiffusionRunner(config)
    runner.train(10)