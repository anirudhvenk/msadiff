import torch
from alphafold3_pytorch import MSAModule
# from msa_module import MSAModule
from torch.profiler import profile, ProfilerActivity
from config import create_config

config = create_config()

model = MSAModule(
    dim_additional_msa_feats=0,
    dim_single=config.model.seq_single_dim,
    dim_pairwise=config.model.seq_pairwise_dim,
    depth=config.model.encoder_depth,
    dim_msa=config.model.encoder_msa_dim,
    dim_msa_input=1,
    max_num_msa=config.data.msa_depth,
    outer_product_mean_dim_hidden=config.model.encoder_outer_prod_mean_hidden,
    msa_pwa_heads=config.model.encoder_pair_weighted_avg_heads,
    msa_pwa_dim_head=config.model.encoder_pair_weighted_avg_hidden,
).cuda()

single = torch.randn((1,100,320)).cuda()
pairwise = torch.randn((1,100,100,120)).cuda()
msa = torch.randn((1,32,100,1)).cuda()
mask = torch.ones(1,100, dtype=torch.bool).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        profile_memory=True, record_shapes=True) as prof:
    model(
        single_repr=single, 
        pairwise_repr=pairwise, 
        msa=msa,
        mask=mask
    )
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
