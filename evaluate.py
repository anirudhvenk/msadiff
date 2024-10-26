import torch
import torch.nn as nn
import esm
import torch.nn.functional as F
import os
import torch.distributed as dist
import wandb
import pytorch_lightning as pl

from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity
from functools import partial
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from model import MSAVAE, MSAEncoder, MSADecoder, Permuter
from torch.utils.data import DataLoader
from data import read_msa, greedy_select, MSADataset, PreprocessedMSADataset
from config import create_config
from tqdm import tqdm
from datetime import datetime
from loss import Criterion
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

_, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()

config = create_config()
config.device = torch.cuda.current_device()
loss = Criterion(config)


# save_path = "checkpoints/20241015-202450/best-checkpoint-epoch=01-val_loss=2.96.ckpt"
# output_path = "lightning_model_recent.pt"
# convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
model = MSAVAE.load_from_checkpoint("lightning_model_recent.pt", config=config).cuda()

# train_dataset = PreprocessedMSADataset("databases/train")
test_dataset = MSADataset(config, test=True)
# train_loader = DataLoader(
#     train_dataset, 
#     batch_size=2,
#     num_workers=12
# )
test_loader = DataLoader(
    test_dataset, 
    batch_size=1,
    num_workers=12
)

with torch.no_grad():
    for single_repr, pairwise_repr, msa, mask in tqdm(test_loader):
        mask = mask.cuda().bool()
        pred_msa, perm, mu, logvar = model(single_repr.cuda().float(), pairwise_repr.cuda().float(), msa.cuda().float(), mask.cuda().bool())
        
        loss_dict =  loss(msa.cuda().to(torch.long), pred_msa, mask.cuda().to(torch.bool), perm, mu, logvar)
        print(loss_dict["ppl"])

# mask_expanded = mask.unsqueeze(1)
# mask_expanded = mask_expanded.expand(-1, config.data.msa_depth, -1).cuda()
# pred_msa = pred_msa[mask_expanded]
# msa_tokens = msa.cuda().int()
# size = msa_tokens[mask_expanded].shape[-1]
# msa_tokens = msa_tokens[mask_expanded].view(1, 32, size//32)

# probs = F.softmax(pred_msa, dim=-1)
# torch.set_printoptions(threshold=10_000)
# print(torch.max(probs, dim=-1).values.view(1, 32, size//32))
# sampled_indices = torch.multinomial(probs, num_samples=1).view(1, 32, size//32)

# def indices_to_tokens(indices, alphabet):
#     return [''.join([alphabet.get_tok(idx.item()) for idx in sequence]) for sequence in indices]

# sampled_tokens = indices_to_tokens(sampled_indices[0], msa_alphabet)
# true_tokens = indices_to_tokens(msa_tokens[0], msa_alphabet)
# for token in true_tokens:
#     print(token)
    
# print("##################################################################")
    
# for token in sampled_tokens:
#     print(token)