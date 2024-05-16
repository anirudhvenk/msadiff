import ml_collections

def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5_000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 0.
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 1_000_000
    training.checkpoint_freq = 50_000
    training.eval_freq = 50_000
    training.batch_size = 512

    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints/'
    config.checkpoints_prefix = ''

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    refresh.prefix = ""

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 512
    validation.validation_iters = int(10_000 / validation.batch_size)
    validation.num_gen_texts = 2048

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 100
    sde.coef_d = 10.
    sde.ode_sampling = False
    sde.scheduler = "sd"

    model = config.model = ml_collections.ConfigDict()
    model.prediction = "x_0"
    model.ema_rate = 0.9999
    model.dropout = 0.1
    model.seq_embed_dim = 320
    model.embed_dim = 768
    model.ffn_embed_dim = 3072
    model.attention_heads = 8
    model.attention_dropout = 0.1
    model.activation_dropout = 0.1
    model.max_tokens = 2 ** 14
    model.num_hidden_layers = 12
    model.max_position_embeddings = 512

    data = config.data = ml_collections.ConfigDict()
    data.num_rows = 64
    data.batch_size = 1
    data.max_sequence_len = 256
    data.train_dataset_path = "./databases/openfold/openfold_a3m/scratch/alignments"
    data.test_dataset_path = "./databases/msa_transformer/data/a3m"
    
    config.seed = 0
    config.ddp = True
    config.use_self_cond = True
    config.device = "cuda:9"

    return config
