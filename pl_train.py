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
from data import read_msa, greedy_select, MSADataset
from config import create_config
from tqdm import tqdm
from datetime import datetime

if __name__ == "__main__":
    config = create_config()
    config.device = torch.cuda.current_device()
    model = MSAVAE(config)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    wandb_logger = WandbLogger(
        project="msadiff",
        name=f"{timestamp}",
        config=config.to_dict()
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{timestamp}",
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    
    train_dataset = MSADataset(config)
    test_dataset = MSADataset(config, test=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,
        num_workers=12
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        num_workers=12
    )

    trainer = pl.Trainer(
        devices=3,
        accumulate_grad_batches=4,
        accelerator="auto",
        gradient_clip_val=1.0,
        strategy="deepspeed_stage_3",
        max_epochs=100,
        precision="bf16-mixed",
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
