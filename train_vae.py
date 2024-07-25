import torch
import torch.nn as nn
import esm
import torch.nn.functional as F
import os
import torch.distributed as dist
import wandb

from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from model import MSAVAE, MSAEncoder, MSADecoder
from loss import Criterion
from torch.utils.data import DataLoader
from data import read_msa, greedy_select, MSADataset
from config import create_config
from tqdm import tqdm

def setup():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    return rank

class VAERunner():
    def __init__(self, config):
        self.config = config

        # if dist.get_rank() == 0:
        #     wandb.init(project="msadiff", config=config.to_dict())

        _, self.msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_batch_converter = self.msa_alphabet.get_batch_converter()

        self.seq_encoder, self.seq_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.seq_encoder.to(config.device)
        self.seq_batch_converter = self.seq_alphabet.get_batch_converter()
        
        self.model = MSAVAE(config).to(config.device)
        self.model = DDP(self.model, broadcast_buffers=False)
        
        encoder = MSAEncoder(config)
        decoder = MSADecoder(config)
        
        if dist.get_rank() == 0:
            print("Encoder params: ", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
            print("Decoder params: ", sum(p.numel() for p in decoder.parameters() if p.requires_grad))
            
        del encoder
        del decoder
        
        self.loss = Criterion(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)
        self.grad_scaler = GradScaler()
        
        self.train_dataset = MSADataset(config)
        sampler_train = torch.utils.data.DistributedSampler(
            self.train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
        self.train_loader = DataLoader(
            self.train_dataset, 
            sampler=sampler_train,
            batch_size=self.config.data.batch_size, 
            collate_fn=self.collate_fn,
            pin_memory=False
        )
        
    def collate_fn(self, data):
        seqs, msas = zip(*data)
        filtered_msas = []
        for msa in msas:
            filtered_msas.append(greedy_select(msa, config.data.msa_depth))
        
        _, _, msa_tokens = self.msa_batch_converter(filtered_msas)
        _, _, seq_tokens = self.seq_batch_converter(seqs)
        seq_batch_lens = (seq_tokens != self.seq_alphabet.padding_idx).sum(1)
        
        with torch.no_grad():
            raw_seq_embeddings = self.seq_encoder(seq_tokens.to(self.config.device), repr_layers=[6], return_contacts=True)
            raw_pairwise_embeddings = raw_seq_embeddings["attentions"]
            raw_seq_embeddings = raw_seq_embeddings["representations"][6]
            
        # Symmetrize
        batch_size, layers, heads, seqlen, _ = raw_pairwise_embeddings.size()
        raw_pairwise_embeddings = raw_pairwise_embeddings.view(batch_size, layers * heads, seqlen, seqlen)
        raw_pairwise_embeddings = raw_pairwise_embeddings + raw_pairwise_embeddings.transpose(-1, -2)
                
        pairwise_seq_embeddings = []
        single_seq_embeddings = []
        mask = torch.ones(seq_tokens.size(0), self.config.data.max_sequence_len, dtype=torch.bool).to(self.config.device)
        for i, tokens_len in enumerate(seq_batch_lens):
            pairwise_seq_repr = raw_pairwise_embeddings[i, :, 1 : tokens_len - 1, 1 : tokens_len - 1]
            pairwise_seq_repr = pairwise_seq_repr.permute(1,2,0)
            single_seq_repr = raw_seq_embeddings[i, 1 : tokens_len - 1]
            
            mask[i, : single_seq_repr.size(0)] = False
            pairwise_seq_repr = F.pad(
                pairwise_seq_repr, 
                (0, 0,
                 0, self.config.data.max_sequence_len - pairwise_seq_repr.size(0), 
                 0, self.config.data.max_sequence_len - pairwise_seq_repr.size(0)), 
                "constant", 
                self.seq_alphabet.padding_idx
            )
            
            single_seq_repr = F.pad(
                single_seq_repr, 
                (0, 0, 0, self.config.data.max_sequence_len - single_seq_repr.size(0)), 
                "constant", 
                self.seq_alphabet.padding_idx
            )
            
            pairwise_seq_embeddings.append(pairwise_seq_repr)
            single_seq_embeddings.append(single_seq_repr)
        
        pairwise_seq_embeddings = torch.stack(pairwise_seq_embeddings)
        single_seq_embeddings = torch.stack(single_seq_embeddings)
        
        msa_tokens = msa_tokens[:,:,1:].to(self.config.device)
        msa_tokens = F.pad(
            msa_tokens,
            (0, self.config.data.max_sequence_len - msa_tokens.size(-1)),
            "constant",
            self.msa_alphabet.padding_idx
        )
        
        return single_seq_embeddings, pairwise_seq_embeddings, msa_tokens, ~mask
    
    def train(self):
        for e in range(self.config.training.epochs):
            total_loss = 0
            recon_loss = 0
            perm_loss = 0
            kld_loss = 0
            ppl = 0
            
            for single_seq_embeddings, pairwise_seq_embeddings, msa_tokens, mask in tqdm(self.train_loader):                
                self.optimizer.zero_grad()

                pred_msa, perm, mu, logvar = self.model(single_seq_embeddings, pairwise_seq_embeddings, msa_tokens.unsqueeze(-1).float(), mask)
                loss_dict = self.loss(msa_tokens, pred_msa, mask, perm, mu, logvar)
                loss_dict["loss"].backward()
                
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    0.5
                )
                self.optimizer.step()

                total_loss += loss_dict["loss"].detach()
                recon_loss += loss_dict["recon_loss"].detach()
                perm_loss += loss_dict["perm_loss"].detach()
                kld_loss += loss_dict["kld_loss"].detach()
                ppl += loss_dict["ppl"].detach()
                
            if dist.get_rank() == 0:
                print({
                            "epoch": e,
                            "total_loss": total_loss.item()/len(self.train_loader),
                            "recon_loss": recon_loss.item()/len(self.train_loader),
                            "perm_loss": perm_loss.item()/len(self.train_loader),
                            "kld_loss": kld_loss.item()/len(self.train_loader),
                            "perplexity": ppl.item()/len(self.train_loader)
                        })                
            # if dist.get_rank() == 0:
            #     wandb.log({
            #         "epoch": e,
            #         "total_loss": total_loss.item()/len(self.train_loader),
            #         "recon_loss": recon_loss.item()/len(self.train_loader),
            #         "perm_loss": perm_loss.item()/len(self.train_loader),
            #         "kld_loss": kld_loss.item()/len(self.train_loader),
            #         "perplexity": ppl.item()/len(self.train_loader)
            #     })
        
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config = create_config()
    config.device = setup()
    
    runner = VAERunner(config)
    runner.train()
    
    # if dist.get_rank() == 0:
    #     wandb.finish()