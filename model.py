import torch 
import torch.nn as nn
import math 

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # map an id to the same vector every time
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # math.sqrt(self.d_model) is mentioned in the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len , 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # shape (seq_len , 1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, Seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # returns only up to sequence len
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)

        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        # epsilon is used for numerical stability for do not get very big numbers or very small numbers
        # also used to avoid dividing by zero
        self.eps = eps
        # parameter is used to make number learnable
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied 
        self.bais = nn.Parameter(torch.zeros(1)) # added
    
    def forward(self, x):

        mean = x.mean(dim= -1, keepdim=True)
        std = x.std(dim= -1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bais

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout:int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1, B1
        self.linear2 = nn.Linear(d_ff, d_model) # W2, B2
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # (Batch_size, seq_len, D_model) >-linear1-> (Batch_size, Seq_len, d_ff) >-Linear2-> (Batch_size, Seq_len, D_model)
        net = torch.relu(self.linear1(x))
        net = self.dropout(net)
        net = self.linear2(net)
        return net

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        # we need to make sure that d_model is divisabily by h
        assert d_model % h == 0 , "D_Model is not divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) # WQ
        self.w_k = nn.Linear(d_model, d_model) # WK
        self.w_v = nn.Linear(d_model, d_model) # WV

        self.w_output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    # static method means that you can call this method without create an object from this class
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # @ means matrix multiplications
        # Transpose last two dim seq_len, dk --> dk, seq_len
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim= -1) # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # output for next layer (attention_scores @ value)
        # attention_scores for the visualization
        return (attention_scores @ value) , attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) ## (Batch_size, seq_len, D_model)  ---> (Batch_size, seq_len, D_model) 
        key = self.w_k(k) ## (Batch_size, seq_len, D_model)  ---> (Batch_size, seq_len, D_model)
        value = self.w_v(v) ## (Batch_size, seq_len, D_model)  ---> (Batch_size, seq_len, D_model)

        # we use transpose because we want h to be in second dim not third

        # (Batch_size, seq_len, D_model) --Split--> (Batch_size, seq_len, h, dk)
        # --Transpose--> (Batch_size, h, seq_len, dk )
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key,
                                                                     value, mask,
                                                                     self.dropout)