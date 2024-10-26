import torch
import torch.nn as nn
import pytorch_lightning as pl

from decoder_modules import AxialTransformerLayer, LearnedPositionalEmbedding, RobertaLMHead
from encoder_utils import PositionalEncoding
from pytorch_lightning.callbacks import ModelCheckpoint
from msa_module import MSAModule
from loss import Criterion
from torch.profiler import profile, ProfilerActivity

class MSAVAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.encoder = MSAEncoder(config)
        self.decoder = MSADecoder(config)
        self.permuter = Permuter(config)
        
        self.loss = Criterion(config)
        
    def training_step(self, batch, batch_idx):
        single_repr, pairwise_repr, msa_tokens, mask = batch
        z, mu, logvar, msa = self.encoder(
            single_repr,
            pairwise_repr,
            msa_tokens.unsqueeze(-1),
            mask.to(torch.bool)
        )
        perm = self.permuter(torch.mean(msa, dim=-1))
        pred_msa = self.decoder(z, perm, mask.to(torch.bool))

        loss_dict = self.loss(msa_tokens.to(torch.long), pred_msa, mask.to(torch.bool), perm, mu, logvar)

        for loss_name, loss_value in loss_dict.items():
            self.log(
                f"train_{loss_name}",
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

        return loss_dict

    def validation_step(self, batch, batch_idx):
        single_repr, pairwise_repr, msa_tokens, mask = batch
        z, mu, logvar, msa = self.encoder(
            single_repr,
            pairwise_repr,
            msa_tokens.unsqueeze(-1),
            mask.to(torch.bool)
        )
        perm = self.permuter(torch.mean(msa, dim=-1))
        pred_msa = self.decoder(z, perm, mask.to(torch.bool))

        loss_dict = self.loss(msa_tokens.to(torch.long), pred_msa, mask.to(torch.bool), perm, mu, logvar)

        for loss_name, loss_value in loss_dict.items():
            self.log(
                f"val_{loss_name}",
                loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

        return loss_dict
    
    def configure_optimizers(self):
        return None
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    #     return optimizer
        
    def forward(self, single_repr, pairwise_repr, msa, mask=None):
        z, mu, logvar, msa = self.encoder(single_repr, pairwise_repr, msa.unsqueeze(-1), mask.to(torch.bool))
        perm = self.permuter(torch.mean(msa, dim=-1))
        msa = self.decoder(z, perm, mask)
        
        return msa, perm, mu, logvar

class MSAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.msa_module = MSAModule(
            dim_single=config.model.seq_single_dim,
            dim_pairwise=config.model.seq_pairwise_dim,
            depth=config.model.encoder_depth,
            dim_msa=config.model.encoder_msa_dim,
            dim_msa_input=1,
            max_num_msa=config.data.msa_depth,
            outer_product_mean_dim_hidden=config.model.encoder_outer_prod_mean_hidden,
            msa_pwa_heads=config.model.encoder_pair_weighted_avg_heads,
            msa_pwa_dim_head=config.model.encoder_pair_weighted_avg_hidden,
        )
            
        self.encode_z = nn.Linear(config.model.seq_pairwise_dim, 2 * config.model.seq_pairwise_dim)
        self.layer_norm = nn.LayerNorm(config.model.seq_pairwise_dim)
    
    def forward(
        self,
        single_repr,
        pairwise_repr,
        msa,
        mask=None
    ):
        msa, seq = self.msa_module(
            single_repr=single_repr, 
            pairwise_repr=pairwise_repr, 
            msa=msa, 
            mask=mask,
        )
            
        # Mean pooling
        
        seq = self.layer_norm(torch.mean(seq, dim=2))
        
        z = self.encode_z(seq)
        mu = z[:,:,:120]
        logvar = z[:,:,120:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
                
        return z, mu, logvar, msa

class MSADecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.max_sequence_len = config.data.max_sequence_len
        self.msa_depth = config.data.msa_depth
        self.alphabet_size = config.data.alphabet_size
        
        self.positional_embedding = PositionalEncoding(config.model.decoder_pos_emb_dim)
        self.decode_z = nn.Linear(
            config.model.seq_pairwise_dim+config.model.decoder_pos_emb_dim, 
            config.model.decoder_msa_dim
        )
        
        self.register_buffer("position_ids", torch.arange(self.max_sequence_len).expand((1,self.msa_depth,-1)).clone())
        self.position_embeddings = torch.nn.Embedding(self.max_sequence_len, config.model.decoder_msa_dim)
        
        self.before_norm = nn.LayerNorm(config.model.decoder_msa_dim)
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    config.model.decoder_msa_dim,
                    config.model.decoder_ffn_hidden,
                    config.model.decoder_num_heads,
                    config.model.decoder_dropout,
                    config.model.decoder_activation_dropout,
                    max_tokens_per_msa=2**14,
                )
                for _ in range(config.model.decoder_depth)
            ]
        )
        self.after_norm = nn.LayerNorm(config.model.decoder_msa_dim)
        self.roberta_lm_head = RobertaLMHead(config)
        
    def init_message_matrix(self, z, perm):
        batch_size = z.size(0)

        x = z.unsqueeze(1).expand(-1, self.msa_depth, -1, -1)

        pos_emb = self.positional_embedding(batch_size, self.msa_depth)
        pos_emb = torch.matmul(perm, pos_emb).unsqueeze(2).repeat_interleave(self.max_sequence_len, 2)

        x = z.unsqueeze(1).expand(-1, self.msa_depth, -1, -1)
        x = torch.cat((x, pos_emb), dim=-1)
        x = self.decode_z(x)
        
        return x
    
    def forward(self, z, perm, mask=None):
        x = self.init_message_matrix(z, perm)
        x += self.position_embeddings(self.position_ids)
        
        x = self.before_norm(x)
        x = x.permute(1, 2, 0, 3)
        
        for layer in self.layers:
            x = layer(
                x,
                self_attn_padding_mask=mask
            )
        
        x = self.after_norm(x)
        x = x.permute(2, 0, 1, 3)
        x = self.roberta_lm_head(x)
                
        return x

class Permuter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scoring_fc = nn.Linear(config.data.max_sequence_len, 1)

    def score(self, x, mask=None):
        scores = self.scoring_fc(x)
        return scores

    def soft_sort(self, scores, hard, tau):
        scores_sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - scores_sorted).abs().neg() / tau
        perm = pairwise_diff.softmax(-1)
        if hard:
            perm_ = torch.zeros_like(perm, device=perm.device)
            perm_.scatter_(-1, perm.topk(1, -1)[1], value=1)
            perm = (perm_ - perm).detach() + perm
        return perm

    def mask_perm(self, perm, mask):
        batch_size, num_nodes = mask.size(0), mask.size(1)
        eye = torch.eye(num_nodes, num_nodes).unsqueeze(0).expand(batch_size, -1, -1).type_as(perm)
        mask = mask.unsqueeze(-1).expand(-1, -1, num_nodes)
        perm = torch.where(mask, perm, eye)
        return perm

    def forward(self, node_features, mask=None, hard=False, tau=1.0):
        # add noise to break symmetry
        node_features = node_features + torch.randn_like(node_features) * 0.05
        scores = self.score(node_features, mask)
        perm = self.soft_sort(scores, hard, tau)
        perm = perm.transpose(2, 1)
        return perm