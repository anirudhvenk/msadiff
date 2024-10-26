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
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        # save_top_k=1,
        # mode="min",
    )
    
    train_dataset = PreprocessedMSADataset("databases/train")
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
    
    deepspeed_config = {
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": config.optim.learning_rate
            }
        },
        "zero_optimization": {
            "stage": 3
        },
        "gradient_clipping": config.optim.grad_clip_norm
    }

    trainer = pl.Trainer(
        devices=3,
        # accumulate_grad_batches=4,
        accelerator="gpu",
        # gradient_clip_val=1.0,
        strategy=DeepSpeedStrategy(config=deepspeed_config),
        max_epochs=config.training.epochs,
        precision=16,
        # gradient_clip_algorithm="norm",
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
