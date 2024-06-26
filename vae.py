import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, -1), 0)
        # print(Q_.shape)
        K_ = torch.cat(K.split(dim_split, -1), 0)
        # print(K_.shape)
        V_ = torch.cat(V.split(dim_split, -1), 0)
        # print(V_.shape)
        # print(K_.shape)
        # print(K_.transpose(2,3).shape)
        # print(Q_.shape)
        print(Q_.bmm(K_.transpose(2,3)))

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), -1)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), -1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class RowMAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(RowMAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        batch_size, num_rows, num_cols, embed_dim = K.size()
        batch_size, q_rows, q_cols, _ = Q.size()
        q = self.fc_q(Q).view(q_rows, q_cols, batch_size, self.num_heads, embed_dim // self.num_heads)
        k = self.fc_k(K).view(num_rows, num_cols, batch_size, self.num_heads, embed_dim // self.num_heads)
        attn_weights = torch.einsum(f"rinhd,rjnhd->hnij", q, k)
        attn_probs = attn_weights.softmax(-1)
        
        v = self.fc_v(K).view(num_rows, num_cols, batch_size, self.num_heads, embed_dim // self.num_heads)
        context = torch.einsum(f"hnij,rjnhd->inhd", attn_probs, v)
        context = context.contiguous().view(num_cols, batch_size, embed_dim)
        output = self.fc_o(context)
        return output.view(batch_size, num_cols, embed_dim)
    
class ColMAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(ColMAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        batch_size, num_rows, num_cols, embed_dim = K.size()
        batch_size, q_rows, q_cols, _ = Q.size()
        q = self.fc_q(Q).view(q_rows, q_cols, batch_size, self.num_heads, embed_dim // self.num_heads)
        k = self.fc_k(K).view(num_rows, num_cols, batch_size, self.num_heads, embed_dim // self.num_heads)
        attn_weights = torch.einsum(f"icnhd,jcnhd->hcnij", q, k)
        attn_probs = attn_weights.softmax(-1)
        
        v = self.fc_v(K).view(num_rows, num_cols, batch_size, self.num_heads, embed_dim // self.num_heads)
        context = torch.einsum(f"hcnij,jcnhd->icnhd", attn_probs, v)
        context = context.contiguous().view(num_cols, batch_size, embed_dim)
        output = self.fc_o(context)
        return output.view(batch_size, num_cols, embed_dim)    
    

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, 4, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = RowMAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1, 1), X)
    
# row_mab = RowMAB(4, 4, 4, 2)
pma = PMA(4, 2, 1)
x = torch.randn(1, 5, 4, 4)
print("Original tensor:")
print(pma(x))

x[:, [1, 3], :] = x[:, [3, 1], :]
print("\nTensor after swapping the first and second rows of the second dimension:")
print(pma(x))