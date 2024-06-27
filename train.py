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
import torch.distributed as dist
import torch
import os
import esm
import numpy as np
import pickle as pkl
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torchsummary import summary
from torch.nn import functional as F

class DiffusionRunner:
    def __init__(self, config):
        self.config = config
        self.device = config.local_rank
        
        self.train_dataset = MSADataset(self.config.data.train_dataset_path, 80000)
        self.test_dataset = MSADataset(self.config.data.test_dataset_path, 10)
        print(len(self.train_dataset))
        print(len(self.test_dataset))
        
        self.msa_encoder, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_encoder.to(self.device)
        self.msa_encoder.eval()

        self.seq_encoder, seq_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.seq_encoder.to(self.device)
        self.seq_encoder.eval()
        
        sampler_train = torch.utils.data.DistributedSampler(
            self.train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
        
        sampler_test = torch.utils.data.DistributedSampler(
            self.test_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            sampler=sampler_train,
            batch_size=self.config.data.batch_size, 
            collate_fn=collate_fn,
            pin_memory=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            sampler=sampler_test,
            batch_size=self.config.data.batch_size, 
            collate_fn=collate_fn,
            pin_memory=False
        )

        self.sde = create_sde(config=self.config, score_fn=self.calc_score)
        self.diff_eq_solver = create_solver(self.config, self.sde, ode_sampling=self.config.sde.ode_sampling)
        self.score_estimator = MSAScoreEstimatorEMB(self.config).to(self.device)
        self.score_estimator = DDP(self.score_estimator, broadcast_buffers=False)
        
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

    def calc_score(self, model, x_t, t, mask=None, x_0_self_cond=None) -> Dict[str, torch.Tensor]:
        params = self.sde.marginal_params_tensor(x_t, t)
        x_0 = model(
            x_t=x_t, 
            query=torch.rand((1,120,320)), 
            time_t=t, 
            cross_attn_padding_mask=None, 
            x_0_self_cond=x_0_self_cond
        )
        eps_theta = (x_t[:,1:,:,:] - params["alpha"] * x_0[:,1:,:,:]) / params["std"]
        score = -eps_theta / params["std"]

        return {
            "score": score,
            "x_0": x_0[:,1:,:,:],
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
        clean_x = msa[:,1:,:,:]
        batch_size = clean_x.size(0)
        t = self.sample_time(batch_size).to(self.device)
        marg_forward = self.sde.marginal_forward(clean_x, t)
        x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']
        x_t = torch.cat((msa[:,0].unsqueeze(1), x_t), dim=1)

        # self-cond estimate
        x_0_self_cond = torch.zeros_like(x_t, dtype=x_t.dtype)
        x_0_self_cond = torch.cat((msa[:,0].unsqueeze(1), x_0_self_cond[:,1:,:,:]), dim=1)
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
                t=t,
                mask=None,
                x_0_self_cond=x_0_self_cond
            )

        # MSE losses
        x_0, eps_theta, score = scores["x_0"], scores['eps_theta'], scores["score"]

        # ignore the original sequence from the MSA
        loss_x_0 = self.mse_loss(clean_x, x_0, mask=None)
        loss_eps = self.mse_loss(noise, eps_theta, mask=None)
        loss_score = self.mse_loss(score_clean, score, mask=None)
        
        # if (t.item() < 0.1 and self.device == 0):
        #     print("t: ", t.item())  
        #     print("loss_score: ", loss_score.item())
        #     print("loss_eps: ", loss_eps.item())
        #     print("loss_x_0: ", loss_x_0.item())
            
        
        return loss_x_0, loss_eps, loss_score

    def optimizer_step(self, loss):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = torch.sqrt(sum([torch.sum((t.grad).to(self.device) ** 2) for t in self.score_estimator.parameters()]))
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
                seq_embeddings, msa_embeddings = encode(seq_tokens, msa_tokens, self.msa_encoder, self.seq_encoder, self.device)
                
                loss_x_0, loss_eps, loss_score = self.calc_loss(seq_embeddings, msa_embeddings)
                self.optimizer_step(loss_x_0)
                
                total_loss_x_0 += loss_x_0.detach()
                total_loss_eps += loss_eps.detach()
                total_loss_score += loss_score.detach()
                  
            if (self.device == 0):
                print(f"{self.device}, loss_x_0: ", (total_loss_x_0/len(self.train_loader)).item())
                print(f"{self.device}, loss_eps: ", (total_loss_eps/len(self.train_loader)).item())
                print(f"{self.device}, loss_score: ", (total_loss_score/len(self.train_loader)).item())
                
                for seq_tokens, msa_tokens in self.test_loader:
                    msa_embeddings, msa_mean, msa_std = encode(
                        seq_tokens, 
                        msa_tokens, 
                        self.msa_encoder,
                        self.seq_encoder, 
                        self.device, 
                        decoding=True
                    )
                    
                    pred_embeddings = self.pred_embeddings(msa_embeddings[:,0].unsqueeze(1), 1)
                    pred_embeddings = pred_embeddings * msa_std + msa_mean
                    
                    with torch.no_grad():
                        x = self.msa_encoder.lm_head(pred_embeddings)
                    
                    print(F.cross_entropy(x.view(-1, x.shape[-1]).cpu(), msa_tokens.view(-1)))
                
    @torch.no_grad()
    def pred_embeddings(
            self, seq, batch_size: int,
            attention_mask=None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.config.data.num_rows,
            seq.shape[2],
            self.config.model.embed_dim
        )

        with torch.no_grad():
            x = self.sde.prior_sampling(shape).to(self.device)
            x = torch.cat((seq, x[:,1:,:,:]), dim=1)
            x_0_self_cond = torch.zeros_like(x, dtype=x.dtype)
            x_0_self_cond = torch.cat((seq, x_0_self_cond[:,1:,:,:]), dim=1)
            eps_t = 0.01
            timesteps = torch.linspace(self.sde.T, eps_t, self.sde.N, device=self.device)
            for i in tqdm(range(self.sde.N)):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                output = self.diff_eq_solver.step(
                    model=self.score_estimator,
                    x_t=x, t=vec_t,
                    mask=attention_mask,
                    x_0_self_cond=x_0_self_cond,
                )
                x, x_mean = output["x"], output["x_mean"]
                x_0_self_cond = output["x_0"]

                x = torch.cat((seq, x), dim=1)
                x_mean = torch.cat((seq, x_mean), dim=1)
                x_0_self_cond = torch.cat((seq, x_0_self_cond), dim=1)
            pred_embeddings = x_mean

        return pred_embeddings
                
def setup():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    return rank

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config = create_config()
    config.local_rank = setup()
    
    runner = DiffusionRunner(config)
    runner.train(100)