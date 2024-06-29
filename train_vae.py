import torch
from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
from typing import Optional

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
encoder = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D", add_cross_attention=False, is_decoder=False)

sequences = ["ATGY"]


tokenized = tokenizer(
    sequences, 
    return_attention_mask=True, 
    return_tensors="pt", 
    truncation=True, 
    padding=True, 
    max_length=256
)

with torch.no_grad():
    embeddings = encoder(**tokenized).last_hidden_state

print(embeddings.shape)