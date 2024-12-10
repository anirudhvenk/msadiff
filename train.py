import torch
import torch.optim as optim
import pytorch_lightning as pl
import os

from data import MSADataset
from model import MSADiTModule
from torch.utils.data import DataLoader
from config import create_config
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

config = create_config()

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
wandb_logger = WandbLogger(
    project="msadiff",
    name=f"{timestamp}",
    config=config.to_dict()
)

train_dataset = MSADataset(
    base_path=config.data.train_dataset_path,
    max_msa_depth=config.data.max_msa_depth
)

val_dataset = MSADataset(
    base_path=config.data.val_dataset_path,
    max_msa_depth=config.data.max_msa_depth
)

train_loader = DataLoader(train_dataset, batch_size=None, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=None, num_workers=4)

model = MSADiTModule(config)

os.makedirs(f"weights/{timestamp}", exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=f"weights/{timestamp}",
    filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=50,
    gradient_clip_val=1.0,
    precision="bf16-mixed",
    logger=wandb_logger,
    # strategy="deepspeed_stage_2",
    callbacks=[checkpoint_callback]
)

trainer.fit(
    model=model, 
    train_dataloaders=train_loader, 
    val_dataloaders=val_loader
)