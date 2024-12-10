import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

from ema import ExponentialMovingAverage
# from evaluate import run_folding_eval
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dit.rotary import Rotary
from dit.transformer import TimestepEmbedder, EmbeddingLayer, DDiTBlock, DDitFinalLayer
from categorical import (
    sample_simplex,
    CategoricalFlow,
    SimplexCategoricalFlow,
    SphereCategoricalFlow,
    LinearCategoricalFlow
)
from torch.distributed import get_rank, get_world_size

class MSADiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.seq_proj = nn.Linear(config.data.vocab_size, config.model.hidden_dim//2)
        self.msa_proj = nn.Linear(config.model.msa_embedding_dim, config.model.hidden_dim//2)

        self.timestep_emb = TimestepEmbedder(config.model.conditioning_dim)
        self.rotary_emb = Rotary(config.model.hidden_dim // config.model.num_heads)
        self.blocks = nn.ModuleList([
            DDiTBlock(
                config.model.hidden_dim,
                config.model.num_heads,
                config.model.conditioning_dim, 
                dropout=config.model.dropout
            ) for _ in range(config.model.depth)
        ])
        self.output_layer = DDitFinalLayer(
            config.model.hidden_dim, 
            config.data.vocab_size, 
            config.model.conditioning_dim
        )

    def forward(self, pt, timestep, *cond_args):
        # print(get_rank(), pt.shape)
        msa_embedding = cond_args[0]
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Mean pooling on MSA embedding
            msa_embedding = self.msa_proj(msa_embedding)
            msa_embedding = torch.mean(msa_embedding, dim=1)
            
            pt = self.seq_proj(pt)
            x = torch.cat((pt, msa_embedding), dim=-1)
            c = F.silu(self.timestep_emb(timestep))
            rotary_cos_sin = self.rotary_emb(x)
        
        
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c)
            x = self.output_layer(x, c)
        
        return x
    
class MSADiTModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.model = SphereCategoricalFlow(
            encoder=MSADiT(config),
            data_dims=None,
            n_class=config.data.vocab_size,
        )
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), 
            decay=config.training.ema_decay
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.ema.update(self.model.parameters())

    def on_train_epoch_start(self):
        self.model.train()
        self.ema.to(self.device)
        
    def training_step(self, batch, batch_idx):
        loss = self.model.get_loss(*batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(*batch)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss
    
    def on_validation_epoch_start(self):
        self.model.eval()
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())

    def on_validation_epoch_end(self):
        self.ema.restore(self.model.parameters())
        self.model.train()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            factor=self.config.training.factor, 
            patience=self.config.training.patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }