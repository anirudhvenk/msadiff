import ml_collections

def create_config():
    config = ml_collections.ConfigDict()

    model = config.model = ml_collections.ConfigDict()
    model.msa_embedding_dim = 128 # constant from AF3
    model.conditioning_dim = 128
    model.hidden_dim = 768
    model.num_heads = 12
    model.dropout = 0.1
    model.depth = 12

    training = config.training = ml_collections.ConfigDict()
    training.ema_decay = 0.9999
    training.max_epochs = 500
    training.lr = 1e-5
    training.weight_decay = 1e-2
    training.beta1 = 0.9
    training.beta2 = 0.999
    training.factor = 0.8
    training.patience = 5
    
    data = config.data = ml_collections.ConfigDict()
    data.batch_size = 32
    data.max_msa_depth = 32
    data.vocab_size = 27
    data.train_dataset_path = "databases/evodiff/train"
    data.val_dataset_path = "databases/evodiff/val"
    data.test_dataset_path = "databases/evodiff/test"

    return config
