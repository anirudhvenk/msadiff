import torch
import torch.nn as nn

from encoder_modules import OuterProductMean, PairWeightedAveraging, Transition, PositionalEncoding
from decoder_modules import AxialTransformerLayer, LearnedPositionalEmbedding

class MSAVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.encoder = MSAEncoder(config)
        self.decoder = MSADecoder(config)
        self.permuter = Permuter(config)
        
    def forward(self, seq, msa, mask=None):
        z, mu, logvar, msa = self.encoder(seq, msa, mask)
        perm = self.permuter(msa.flatten(-2))
        msa = self.decoder(z, perm, mask)
        
        return msa, perm, mu, logvar

class MSAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed_tokens = nn.Embedding(
            config.data.alphabet_size, config.model.encoder_msa_dim, config.data.padding_idx
        )
        
        self.layers = nn.ModuleList([])
        for _ in range(config.model.encoder_depth):
            outer_product_mean = OuterProductMean(
                dim_msa=config.model.encoder_msa_dim,
                dim_seq=config.model.seq_dim,
                dim_hidden=config.model.encoder_outer_prod_mean_hidden
            )
            
            pair_weighted_averaging = PairWeightedAveraging(
                dim_msa=config.model.encoder_msa_dim,
                dim_seq=config.model.seq_dim,
                dim_head=config.model.encoder_pair_weighted_avg_hidden,
                heads=config.model.encoder_pair_weighted_avg_heads,
                dropout=config.model.encoder_dropout,
                dropout_type="row"
            )
            
            msa_transition = Transition(dim=config.model.encoder_msa_dim)
            seq_transition = Transition(dim=config.model.seq_dim)
            
            self.layers.append(nn.ModuleList([
                outer_product_mean,
                pair_weighted_averaging,
                msa_transition,
                seq_transition
            ]))
            
        self.encode_z = nn.Linear(config.model.seq_dim, 2 * config.model.seq_dim)
    
    def forward(
        self,
        seq,
        msa,
        mask=None
    ):
        # TODO sampling (maybe?)
        msa = self.embed_tokens(msa)
        
        for (
            outer_product_mean,
            pair_weighted_averaging,
            msa_transition,
            seq_transition
        ) in self.layers:
            seq = outer_product_mean(msa, mask) + seq
            msa = pair_weighted_averaging(msa, seq, mask) + msa
            msa = msa_transition(msa) + msa
            seq = seq_transition(seq) + seq
            
        z = self.encode_z(torch.relu(seq))
        mu = z[:,:,:seq.size(-1)]
        logvar = z[:,:,seq.size(-1):]
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
            config.model.seq_dim+config.model.decoder_pos_emb_dim, 
            config.model.decoder_msa_dim
        )

        self.embed_positions = LearnedPositionalEmbedding(
            config.model.decoder_max_pos,
            config.model.decoder_msa_dim,
            config.data.padding_idx
        )
        
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
        self.to_logits = nn.Linear(config.model.decoder_msa_dim, config.data.alphabet_size)
        
    def init_message_matrix(self, z, perm):
        batch_size = z.size(0)

        pos_emb = self.positional_embedding(batch_size, self.msa_depth)
        pos_emb = torch.matmul(perm, pos_emb).unsqueeze(2).repeat_interleave(self.max_sequence_len, 2)

        x = z.unsqueeze(1).expand(-1, self.msa_depth, -1, -1)
        x = torch.cat((x, pos_emb), dim=-1)
        x = self.decode_z(x)
        
        return x
    
    def forward(self, z, perm, mask=None):
        x = self.init_message_matrix(z, perm)
        x = self.before_norm(x)
        x = x.permute(1, 2, 0, 3)
                
        for layer in self.layers:
            x = layer(
                x,
                self_attn_padding_mask=mask
            )
        
        x = self.after_norm(x)
        x = x.permute(2, 0, 1, 3)
        x = self.to_logits(nn.GELU()(x))
        
        return x

class Permuter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scoring_fc = nn.Linear(config.data.max_sequence_len*config.model.encoder_msa_dim, 1)

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