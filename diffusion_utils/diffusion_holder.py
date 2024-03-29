import os
import json
import torch
import wandb
import numpy as np
import torch.distributed as dist
from copy import deepcopy
from ml_collections import ConfigDict
from random import random
from typing import Optional, Union, Dict
from tqdm import tqdm
from tqdm.auto import trange
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torch.cuda.amp import GradScaler
from timm.scheduler.cosine_lr import CosineLRScheduler
from typing import List, Dict, Union, Tuple


from model.score_estimator import ScoreEstimatorEMB
from model.ema_model import ExponentialMovingAverage
from diffusion_utils.diffusion_dynamic_sde import create_sde, create_solver
from encoders import EncNormalizer, ESM2EncoderModel
from utils import dict_to_cuda, reduce_tensor, masked_mean, masked_std, make_mask_wo_SEP_CLS, set_seed, gather_texts, load_fasta_file
from evaluation import calculate_fid_for_files
from diffusion_utils import LengthSampler


class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False,
            latent_mode: str = "embeddings"
    ):
        self.config = config
        self.latent_mode = latent_mode
        self.eval = eval
        self.use_self_cond = config.use_self_cond

        self.checkpoints_folder = config.training.checkpoints_folder

        self.enc_normalizer = EncNormalizer(
            enc_mean_path=self.config.data.enc_mean,
            enc_std_path=self.config.data.enc_std,
        ).cuda()
        self.encoder_decoder = ESM2EncoderModel(
            config.model.hg_name,
            device=self.config.device,
            enc_normalizer=self.enc_normalizer,
            decoder_path=config.decoder_path,
            max_seq_len=config.data.max_sequence_len,
        )

        self.optimizer = None
        self.scheduler = None
        self.step = 0

        self.score_estimator = ScoreEstimatorEMB(
            input_size=self.config.model.hidden_size,
            config=config.bert_config
        ).cuda().train()
        self.ddp_score_estimator = self.score_estimator
        if self.config.ddp:
            self.ddp_score_estimator = torch.nn.parallel.DistributedDataParallel(
                self.score_estimator,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
            )
        self.total_number_params = sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)
        self.config.model.total_number_params = self.total_number_params
        self.device = next(self.score_estimator.parameters()).device

        self.sde = create_sde(config=config, score_fn=self.calc_score)
        self.diff_eq_solver = create_solver(config, self.sde, ode_sampling=config.sde.ode_sampling)

        if eval:
            self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), config.model.ema_rate)
            self.restore_parameters(self.device)
            self.switch_to_ema()
            self.score_estimator.eval()

        self.train_dataset = None
        self.valid_dataset = None
        self.length_sampler = LengthSampler(path=self.config.data.test_dataset_path, max_len=self.config.data.max_sequence_len - 2)
        
        if self.config.ddp and dist.get_rank() == 0 and not eval:
            wandb.init(
                project=self.config.project_name,
                name=self.config.checkpoints_prefix,
                config=dict(self.config),
                mode="online"
            )
        

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.checkpoints_folder
        prefix = ''
        if self.config.checkpoints_prefix:
            prefix = self.config.checkpoints_prefix
        ema_ckpt = torch.load(checkpoints_folder + '/' + prefix + '.pth')["ema"]
        self.ema.load_state_dict(ema_ckpt)

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        optimizer = torch.optim.AdamW(
            self.score_estimator.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer

    def set_scheduler(self) -> None:
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config.training.training_iters,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def set_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()

    def set_train_data_generator(self) -> None:
        if self.train_dataset is None:
            self.train_dataset = load_fasta_file(self.config.data.train_dataset_path)
        print("Train dataset length:", len(self.train_dataset))

        if self.config.ddp:
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        else:
            sampler_train = None

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler_train,
            batch_size=self.config.training.batch_size_per_gpu,
            num_workers=15,
            pin_memory=False,
        )

    def set_valid_data_generator(self) -> None:
        if self.valid_dataset is None:
            self.valid_dataset = load_fasta_file(self.config.data.test_dataset_path)
        print("Valid dataset length:", len(self.valid_dataset))

        if self.config.ddp:
            sampler_valid = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                shuffle=False
            )
        else:
            sampler_valid = None

        self.valid_loader = DataLoader(
            self.valid_dataset,
            sampler=sampler_valid,
            batch_size=self.config.validation.batch_size // dist.get_world_size(),
            num_workers=15,
            pin_memory=False,
        )

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()

        self.grad_scaler.unscale_(self.optimizer)

        grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters()]))

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.score_estimator.parameters(),
                max_norm=self.grad_clip_norm
            )

        clipped_grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters()]))

        if dist.get_rank() == 0:
            self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # Custom strategy
        scale = self.grad_scaler._scale.item()
        max_scale = 2 ** 30
        min_scale = 1
        scale = np.clip(scale, min_scale, max_scale)
        self.grad_scaler.update(new_scale=scale)

        self.ema.update(self.score_estimator.parameters())
        self.scheduler.step_update(self.step)
        return grad_norm, clipped_grad_norm

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.cuda.FloatTensor(batch_size).uniform_() * (self.sde.T - eps) + eps

    def bert_acc(self, targets, outputs, mask):
        if mask is None:
            mask = torch.ones(
                (targets.shape[0], targets.shape[1]),
                device=f"cuda:{dist.get_rank()}",
                requires_grad=False,
                dtype=torch.int64,
            )
        pred_tokens = outputs.argmax(dim=-1)

        mask = deepcopy(mask)
        mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
        mask[:, 0] = 0
        return torch.sum(mask * (targets == pred_tokens)) / torch.sum(mask)

    def mse_loss(self, inputs, targets, mask):
        if mask is None:
            mask = torch.ones(
                (targets.shape[0], targets.shape[1]),
                device=f"cuda:{dist.get_rank()}",
                requires_grad=False,
                dtype=torch.int64,
            )
        losses = torch.mean(torch.square(inputs - targets), dim=-1)
        losses = losses * mask
        loss = torch.sum(losses) / torch.sum(mask)
        return loss

    def recon_loss(self, inputs, outputs, mask):
        if mask is None:
            mask = torch.ones(
                (inputs.shape[0], inputs.shape[1]),
                device=f"cuda:{dist.get_rank()}",
                requires_grad=False,
                dtype=torch.int64,
            )
        losses = cross_entropy(
            input=inputs.reshape(-1, inputs.shape[-1]),
            target=outputs.reshape(-1),
            reduce=False,
        )
        losses = losses * mask.reshape(-1)
        loss = torch.sum(losses) / torch.sum(mask)
        return loss

    def get_stat(self, z, mask):
        if mask is None:
            mask = torch.ones(
                (z.shape[0], z.shape[1]),
                device=f"cuda:{dist.get_rank()}",
                requires_grad=False,
                dtype=torch.int64,
            )
        mask_SEP_CLS = make_mask_wo_SEP_CLS(mask)
        mean = masked_mean(z, mask_SEP_CLS)
        std = masked_std(z, mask_SEP_CLS)
        norm = torch.sum(torch.norm(z, dim=2) * mask_SEP_CLS) / torch.sum(mask_SEP_CLS)
        return torch.mean(mean), torch.mean(std), norm
    

    def calc_score(self, model, x_t, t, mask=None, x_0_self_cond=None) -> Dict[str, torch.Tensor]:
        params = self.sde.marginal_params_tensor(x_t, t)
        x_0 = model(x_t=x_t, time_t=t, attention_mask=mask, x_0_self_cond=x_0_self_cond)
        eps_theta = (x_t - params["alpha"] * x_0) / params["std"]
        score = -eps_theta / params["std"]

        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta
        }


    def calc_loss(
            self,
            clean_x,
            X=None,
            eps: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        mask = X["attention_mask"]

        # Noizing
        batch_size = clean_x.size(0)

        t = self.sample_time(batch_size, eps=eps)
        marg_forward = self.sde.marginal_forward(clean_x, t)
        x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']

        # self-cond estimate
        x_0_self_cond = torch.zeros_like(x_t, dtype=x_t.dtype)
        if self.use_self_cond and random() < 0.5:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                x_0_self_cond = self.ddp_score_estimator(
                    x_t=x_t, time_t=t,
                    attention_mask=mask,
                    x_0_self_cond=x_0_self_cond
                ).detach()

        # model prediction
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            scores = self.calc_score(
                self.ddp_score_estimator,
                x_t, t,
                mask=mask,
                x_0_self_cond=x_0_self_cond,
            )

        # MSE losses
        x_0, eps_theta, score = scores["x_0"], scores['eps_theta'], scores["score"]

        loss_x_0 = self.mse_loss(clean_x, x_0, mask)
        loss_eps = self.mse_loss(noise, eps_theta, mask)
        loss_score = self.mse_loss(score_clean, score, mask)

        # Statistics
        if self.config.model.loss == "L_x_0":
            loss = loss_x_0
        elif self.config.model.loss == "L_eps":
            loss = loss_eps
        elif self.config.model.loss == "L_score":
            loss = loss_score

        loss_dict = {
            'loss': loss,
            'total_loss': loss,
            'loss_eps': loss_eps,
            'loss_x_0': loss_x_0,
            'loss_score': loss_score,
            }

        clean_x_mean, clean_x_std, clean_x_norm = self.get_stat(clean_x, mask)
        x_0_mean, x_0_std, x_0_norm = self.get_stat(x_0, mask)
        stat_dict = {
            "clean_x_mean": clean_x_mean,
            "clean_x_std": clean_x_std,
            "clean_x_norm": clean_x_norm,
            "x_0_mean": x_0_mean,
            "x_0_std": x_0_std,
            "x_0_norm": x_0_norm,
        }
        return loss_dict, stat_dict

    def train(
            self,
            project_name: str = 'bert_diffusion',
            experiment_name: str = 'bert_emb'
    ) -> None:
        self.step = 0
        self.set_optimizer()
        self.set_scheduler()
        self.set_grad_scaler()
        self.set_valid_data_generator()
        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.model.ema_rate)

        if self.config.refresh.true:
            self.refresh_checkpoint()
            self.estimation(suffix=f"masked-sc-Euler")
            self.validate()

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        self.train_range_iter = iter(self.train_range)

        while True:
            self.set_train_data_generator()
            self.ddp_score_estimator.train()
            self.train_epoch()

            if self.step >= self.config.training.training_iters:
                break

        self.score_estimator.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()

    def train_epoch(self):
        for _, X in enumerate(self.train_loader):
            if self.step >= self.config.training.training_iters:
                return
            _ = next(self.train_range_iter)

            loss_dict, stat_dict = self.train_step(X)

            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

            if self.step % self.config.training.eval_freq == 0:
                self.estimation(suffix=f"masked-sc-Euler")
                self.validate()

            self.train_range.set_description(
                f"loss_eps: {loss_dict['loss_eps'].item():0.4f}, "
                f"loss_x_0: {loss_dict['loss_x_0'].item():0.4f}, "
                f"grad_norm: {stat_dict['grad_norm'].item():0.4f}, "
            )

    def train_step(self, X):
        self.step += 1
        X = dict_to_cuda(X)
        with torch.no_grad():
            clean_X, tokenized_X = self.encoder_decoder.batch_encode(X)
        loss_dict, stat_dict = self.calc_loss(clean_x=clean_X, X=tokenized_X)

        stat_dict["grad_norm"], stat_dict["clipped_grad_norm"] = self.optimizer_step(loss_dict['total_loss'])

        if dist.get_rank() == 0:
            if self.step % 10 == 0:
                stat_dict["weight_norm"] = torch.sqrt(
                    sum([torch.sum(t.data ** 2) for t in self.score_estimator.parameters()]))

                for k, v in loss_dict.items():
                    self.log_metric(k, 'train', v.item())

                for k, v in stat_dict.items():
                    self.log_metric(k, 'train', v.item())

        return loss_dict, stat_dict

    def validate(self) -> None:
        prev_mode = self.ddp_score_estimator.training

        self.ddp_score_estimator.eval()
        self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        with torch.no_grad():
            for X in self.valid_loader:
                X = dict_to_cuda(X)
                clean_X, tokenized_X = self.encoder_decoder.batch_encode(X)

                loss_dict, _ = self.calc_loss(clean_x=clean_X, X=tokenized_X)
                for k, v in loss_dict.items():
                    if k in valid_loss:
                        valid_loss[k] += v.item() * clean_X.size(0)
                    else:
                        valid_loss[k] = torch.Tensor([v.item() * clean_X.size(0)])
                valid_count += clean_X.size(0)

        valid_count = reduce_tensor(valid_count.cuda())
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].cuda())

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        if dist.get_rank() == 0:
            for k, v in valid_loss.items():
                self.log_metric(k, 'valid_loader', v)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train(prev_mode)

    def save_checkpoint(self, last: bool = False) -> None:
        if dist.get_rank() == 0:
            if not os.path.exists(self.checkpoints_folder):
                os.makedirs(self.checkpoints_folder)

            prefix = ''
            if self.config.checkpoints_prefix:
                prefix = self.config.checkpoints_prefix + '_'
            if last:
                prefix = prefix + 'last_'
            else:
                prefix = prefix + str(self.step) + '_'

            torch.save(
                {   
                    "model": self.score_estimator.state_dict(),
                    "decoder": self.decoder.state_dict(),
                    "ema": self.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "step": self.step,
                },
                os.path.join(self.checkpoints_folder, prefix + ".pth")
            )
            print(f"Save model to: {os.path.join(self.checkpoints_folder, prefix + f'model.pth')}")

    def refresh_checkpoint(self):
        if not self.config.refresh.true:
            return
        load = torch.load(f'{self.config.refresh.prefix}', map_location="cpu")

        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.model.ema_rate)
        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.switch_to_ema()

        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.step = load["step"]
        print(f"Checkpoint refreshed {self.config.refresh.prefix}")

    def generate_text(self, batch_size):
        lens = self.length_sampler.sample(batch_size)
        attention_mask = torch.zeros((batch_size, self.config.data.max_sequence_len))
        for i in range(batch_size):
            for j in range(lens[i]):
                attention_mask[i, j] = 1

        attention_mask = attention_mask.cuda()

        with torch.no_grad():
            pred_embeddings = self.pred_embeddings(batch_size, attention_mask)
            output = self.pred_logits(pred_embeddings, attention_mask)
        return output

    def pred_logits(self, pred_embeddings, attention_mask):
        output = self.encoder_decoder.batch_decode(pred_embeddings, attention_mask=attention_mask)
        return output

    @torch.no_grad()
    def pred_embeddings(
            self, batch_size: int,
            attention_mask=None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            self.config.model.hidden_size
        )

        with torch.no_grad():
            x = self.sde.prior_sampling(shape).to(self.device)
            x_0_self_cond = torch.zeros_like(x, dtype=x.dtype)
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
            pred_embeddings = x_mean

        return pred_embeddings

    @torch.no_grad()
    def estimation(self, suffix="") -> None:
        self.score_estimator.eval()
        self.switch_to_ema()
        
        num_texts = int(self.config.validation.num_gen_texts / dist.get_world_size())
        if dist.get_rank() < self.config.validation.num_gen_texts % dist.get_world_size():
            num_texts += 1

        seed = self.config.seed + dist.get_rank()
        set_seed(seed)
        output = self.generate_text(batch_size=num_texts)

        result = [{"protein": p} for p in output]
        if self.config.ddp:
            result = gather_texts(result)

        if not self.config.ddp or dist.get_rank() == 0:
            texts_path = "./generated_seqs"
            os.makedirs(texts_path, exist_ok=True)

            file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{self.config.sde.N}-{len(result)}-{suffix}.json"
            json.dump(result, open(file_name, "w"), indent=4)
            print(file_name)

            fid_value = calculate_fid_for_files(self.config.data.test_dataset_path, file_name)
            print(f"FID: {fid_value:0.5f}")

        if not eval and self.config.ddp and dist.get_rank() == 0:
            self.log_metric(metric_name="FID", loader_name="", value=fid_value)

        self.switch_back_from_ema()
        self.score_estimator.train()