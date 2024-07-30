import ml_collections

def create_config():
    config = ml_collections.ConfigDict()
    
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 0.3

    training = config.training = ml_collections.ConfigDict()
    training.epochs = 200
    
    loss = config.loss = ml_collections.ConfigDict()
    loss.kld_loss_scale = 0.001
    loss.perm_loss_scale = 0.5

    validation = config.validation = ml_collections.ConfigDict()

    model = config.model = ml_collections.ConfigDict()
    model.seq_single_dim = 320
    model.seq_pairwise_dim = 120
    
    model.encoder_depth = 4
    model.encoder_msa_dim = 512
    model.encoder_outer_prod_mean_hidden = 192
    model.encoder_pair_weighted_avg_hidden = 192
    model.encoder_pair_weighted_avg_heads = 8
    model.encoder_dropout = 0.1
    
    model.decoder_depth = 8
    model.decoder_pos_emb_dim = 128
    model.decoder_msa_dim = 512
    model.decoder_max_pos = 512
    model.decoder_ffn_hidden = 2048
    model.decoder_num_heads = 8
    model.decoder_dropout = 0.0
    model.decoder_attn_dropout = 0.0
    model.decoder_activation_dropout = 0.0
    
    data = config.data = ml_collections.ConfigDict()
    data.alphabet_size = 33
    data.padding_idx = 1
    data.msa_depth = 32
    data.batch_size = 1
    data.grad_accum_steps = 8
    data.max_sequence_len = 256
    data.train_dataset_path = "./databases/openfold/scratch/alignments_1"
    data.test_dataset_path = "./databases/data/a3m"
    data.save_path = "./weights"

    return config