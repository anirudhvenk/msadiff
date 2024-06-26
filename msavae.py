import torch
import torch.nn as nn

from vae_modules import OuterProductMean, PairWeightedAveraging, Transition, PositionalEncoding

class MSAEncoder(nn.Module):
    def __init__(
        self,
        depth=4,
        msa_dim=64,
        seq_dim=320,
        outer_product_mean_hidden_dim=32,
        pair_weighted_average_hidden_dim=32,
        pair_weighted_average_heads=8,
        dropout=0.,
        expansion_factor=4
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            outer_product_mean = OuterProductMean(
                dim_msa=msa_dim,
                dim_seq=seq_dim,
                dim_hidden=outer_product_mean_hidden_dim
            )
            
            pair_weighted_averaging = PairWeightedAveraging(
                dim_msa=msa_dim,
                dim_seq=seq_dim,
                dim_head=pair_weighted_average_hidden_dim,
                heads=pair_weighted_average_heads,
                dropout=dropout,
                dropout_type="row"
            )
            
            msa_transition = Transition(
                dim=msa_dim,
                expansion_factor=expansion_factor
            )
            
            seq_transition = Transition(
                dim=seq_dim,
                expansion_factor=expansion_factor
            )
            
            self.layers.append(nn.ModuleList([
                outer_product_mean,
                pair_weighted_averaging,
                msa_transition,
                seq_transition
            ]))
            
        self.encode_w = nn.Linear(seq_dim, 2 * seq_dim)
    
    def forward(
        self,
        seq,
        msa,
        mask=None
    ):
        # skipping sampling and initial projection
        
        for (
            outer_product_mean,
            pair_weighted_averaging,
            msa_transition,
            seq_transition
        ) in self.layers:
            seq += outer_product_mean(msa, mask)
            msa += pair_weighted_averaging(msa, seq, mask)
            msa += msa_transition(msa)
            seq += seq_transition(seq)
            
        z = self.encode_w(torch.relu(seq))
        mu = z[:,:,:seq.size(-1)]
        logvar = z[:,:,seq.size(-1):]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return msa, z, mu, logvar
    
class Permuter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scoring_fc = nn.Linear(input_dim, 1)

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

class MSADecoder(nn.Module):
    def __init__(
        self,
        seq_dim=320,
        msa_dim=64,
        decode_pos_emb_dim=64
    ):
        super().__init__()
        
        self.decode_w = nn.Linear(seq_dim, msa_dim)
        self.positional_embedding = PositionalEncoding(decode_pos_emb_dim)
        
    def init_message_matrix(self, msa_emb, perm, msa_depth):
        batch_size = msa_emb.size(0)

        pos_emb = self.positional_embedding(batch_size, msa_depth)
        if perm is not None:
            pos_emb = torch.matmul(perm, pos_emb)
            pos_emb_combined = torch.cat(
                (
                    pos_emb.unsqueeze(2).repeat(1, 1, msa_depth, 1),
                    pos_emb.unsqueeze(1).repeat_interleave(msa_depth, dim=1)
                ),
                dim=-1
            )

        x = msa_emb.unsqueeze(1).unsqueeze(1).expand(-1, msa_depth, msa_depth, -1)
        x = torch.cat((x, pos_emb_combined), dim=-1)
        x = self.layer_norm(self.dropout(self.fc_in(x)))
        return x
        

    
    
    
    
    
    
    
    
    
    
    
    
torch.manual_seed(42)    

msa_encoder = MSAEncoder(
    msa_dim=4,
    seq_dim=2,
    outer_product_mean_hidden_dim=32,
    pair_weighted_average_hidden_dim=32,
    pair_weighted_average_heads=8
)
permuter = Permuter(8192)

seq = torch.randn((1,3,2))
msa = torch.randn((1,3,3,4))

padded_msa = nn.functional.pad(msa, (0,0,0,2))
padded_seq = nn.functional.pad(seq, (0,0,0,2))
padding_mask = torch.ones((1,5),dtype=torch.bool)
padding_mask[...,3:] = False

linear1 = nn.Linear(20, 1)
linear2 = nn.Linear(12, 1)

nn.init.xavier_uniform_(linear1.weight)
nn.init.zeros_(linear1.bias)

with torch.no_grad():
    linear2.weight.copy_(linear1.weight[:, :12])
    linear2.bias.copy_(linear1.bias)

msa, z, mu, logvar = msa_encoder(padded_seq, padded_msa, padding_mask)
print(z)
# print(m.flatten(-2))
# print(linear1(m.flatten(-2)))

# m,s = msa_encoder(seq, msa)
# print(linear2(m.flatten(-2)))
