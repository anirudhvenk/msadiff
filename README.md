# Training

### 0. Environment creation
```
conda create -n dif_env python=3.7.16
conda activate dif_env
pip install -r requirements.txt

```

### 1. Data preparation

Specify dataset name `data.dataset`, train and test paths `data.train_dataset_path`, `data.test_dataset_path` in `config.py`.

### 2. Compute statistics

To train the diffusion model, it is necessary to calculate the component average and variance of the embeddings of the text.

```
python -m utils.get_statistics
```

### 3. Decoder training

The proposed architecture allows us to use the decoder of ESM-2 pre-trained simultaneously with the encoder on masked language modeling objectives.
However, we found that additional training of the decoder results in a more precise reconstruction of amino acid sequences from the latents $x$ during inference. 
The decoder architecture comprises two linear layers and an activation function.

```
python train_decoder.py
```

### 4. Diffusion model training

Multi-gpu training launch: 

```
torchrun --nproc_per_node=1 --master_port=31345  train_diffusion.py
```





