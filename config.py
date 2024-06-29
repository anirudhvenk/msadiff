import ml_collections

def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()

    training = config.training = ml_collections.ConfigDict()
    
    loss = config.loss = ml_collections.ConfigDict()

    validation = config.validation = ml_collections.ConfigDict()

    model = config.model = ml_collections.ConfigDict()
    model.seq_dim = 320
    
    model.encoder_depth = 4
    model.encoder_msa_dim = 64
    model.encoder_outer_prod_mean_hidden = 32
    model.encoder_pair_weighted_avg_hidden = 32
    model.encoder_pair_weighted_avg_heads = 8
    model.encoder_dropout = 0.
    
    model.decoder_depth = 4
    model.decoder_pos_emb_dim = 64
    model.decoder_msa_dim = 64
    model.decoder_max_pos = 512
    model.decoder_ffn_hidden = 1024
    model.decoder_num_heads = 8
    model.decoder_dropout = 0.
    model.decoder_attn_dropout = 0.
    model.decoder_activation_dropout = 0.
    
    data = config.data = ml_collections.ConfigDict()
    data.alphabet_size = 33
    data.padding_idx = 1
    data.msa_depth = 32
    data.batch_size = 1
    data.max_sequence_len = 4
    data.train_dataset_path = "./databases/openfold/openfold_a3m/scratch/alignments"
    data.test_dataset_path = "./databases/msa_transformer/data/a3m"

    return config
