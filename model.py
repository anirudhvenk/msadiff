import torch
import torch.nn as nn

from encoder_modules import OuterProductMean, PairWeightedAveraging, Transition, PositionalEncoding
from decoder_modules import AxialTransformerLayer, LearnedPositionalEmbedding

class MSAEncoder(nn.Module):
    def __init__(
        self,
        encoder_depth=4,
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
        for _ in range(encoder_depth):
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
        # TODO sampling (maybe?) and initial projection
        # also have to add positional embeddings to the sequence
        
        msa = self.msa_init_proj(msa)
        
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
        decoder_pos_emb_dim=2,
        decoder_emb_dim=64,
        msa_depth=3,
        max_positions=512,
        decoder_depth=4,
        ffn_embed_dim=3072,
        num_heads=1,
        dropout=0.,
        attention_dropout=0.,
        activation_dropout=0.,
        max_tokens=2**14
    ):
        super().__init__()
        
        self.msa_depth = msa_depth
        self.positional_embedding = PositionalEncoding(decoder_pos_emb_dim)
        self.decode_w = nn.Linear(seq_dim+decoder_pos_emb_dim, decoder_emb_dim)
        
        self.embed_positions = LearnedPositionalEmbedding(
            max_positions,
            decoder_emb_dim,
            1
        )
        
        self.before_norm = nn.LayerNorm(decoder_emb_dim)
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    decoder_emb_dim,
                    ffn_embed_dim,
                    num_heads,
                    dropout,
                    attention_dropout,
                    activation_dropout,
                    max_tokens,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.after_norm = nn.LayerNorm(decoder_emb_dim)
        
        # TODO
        # self.lm_head = RobertaLMHead(
        #     embed_dim=self.args.embed_dim,
        #     output_dim=self.alphabet_size,
        #     weight=self.embed_tokens.weight,
        # )
        
    def init_message_matrix(self, z, perm):
        batch_size = z.size(0)

        pos_emb = self.positional_embedding(batch_size, self.msa_depth)
        pos_emb = torch.matmul(perm, pos_emb).unsqueeze(2).repeat_interleave(5, 2)

        x = z.unsqueeze(1).expand(-1, self.msa_depth, -1, -1)
        x = torch.cat((x, pos_emb), dim=-1)
        x = self.decode_w(x)
        
        return x
    
    def forward(self, z, perm, mask):
        x = self.init_message_matrix(z, perm, mask)
        x = self.before_norm(x)
        x = x.permute(1, 2, 0, 3)
                
        for layer in self.layers:
            x = layer(
                x,
                self_attn_padding_mask=mask
            )
            
        x = self.after_norm(x)
        
        return x
    
# torch.manual_seed(42)    

# msa_encoder = MSAEncoder(
#     msa_dim=4,
#     seq_dim=2,
#     outer_product_mean_hidden_dim=32,
#     pair_weighted_average_hidden_dim=32,
#     pair_weighted_average_heads=8
# )
# permuter = Permuter(5*4)
# decoder = MSADecoder(
#     seq_dim=2,
#     decoder_emb_dim=4,
#     decoder_pos_emb_dim=2
# )


# seq = torch.randn((1,3,2))
# msa = torch.randn((1,3,3,4))

# padded_msa = nn.functional.pad(msa, (0,0,0,2))
# padded_seq = nn.functional.pad(seq, (0,0,0,2))
# padding_mask = torch.ones((1,5),dtype=torch.bool)
# padding_mask[...,3:] = False
# print(padding_mask.shape)
# # attn_padding_mask = padding_mask.unsqueeze(0).repeat_interleave(3, 1)

# msa, z, mu, logvar = msa_encoder(padded_seq, padded_msa, padding_mask)
# perm = permuter(msa.flatten(-2))
# print(decoder(z, perm, padding_mask))

# # print(z)
# # print(m.flatten(-2))
# # print(linear1(m.flatten(-2)))

# # m,s = msa_encoder(seq, msa)
# # print(linear2(m.flatten(-2)))
