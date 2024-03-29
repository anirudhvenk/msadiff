import os
import torch.distributed as dist

from diffusion_utils.diffusion_holder import DiffusionRunner
from utils.util import set_seed
from config import create_config
from utils.setup_ddp import setup_ddp

if __name__ == '__main__':
    config = create_config()
    config.checkpoints_prefix = "DiMA-AFDB"

    config.local_rank = setup_ddp()
    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
    config.device = f"cuda:{dist.get_rank()}"
    config.project_name = 'proteins'

    seed = config.seed
    set_seed(seed)
    if dist.get_rank() == 0:
        print(config)

    diffusion = DiffusionRunner(config, latent_mode=config.model.embeddings_type)

    seed = config.seed + dist.get_rank()
    set_seed(seed)
    diffusion.train(
        project_name=config.project_name,
        experiment_name=config.checkpoints_prefix
    )
