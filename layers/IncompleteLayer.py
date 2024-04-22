import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.nn.utils import weight_norm
import math
from einops import rearrange, repeat

from layers.Transformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Embed import TokenEmbedding, TemporalEmbedding, PositionalEmbedding, TimeFeatureEmbedding
from layers.Repair_Blocks import FullAttentionWithMask

class TokenEmbeddingWithMask(nn.Module):
    def __init__(self, seq_len, c_in, d_model, n_heads, d_ff = None,dropout=0.1):
        super(TokenEmbeddingWithMask, self).__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, c_in, d_model))
        
        self.mask_filter = EncoderLayer(
            AttentionLayer(
                FullAttentionWithMask(attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
            d_model,
            d_ff,
            dropout=dropout,
            activation="gelu"
        )
    '''
    params:
        x: [B, T, D]
        mask: [B, T, D]
    '''
    def forward(self, x, mask):
        B, T, D = x.shape
        x = x.unsqueeze(dim=3)
        x = self.feature_embedding(x)
        x = x + self.pos_embedding
        
        x = rearrange(x, 'b t d d_model -> (b d) t d_model')
        mask = rearrange(mask, 'b t d -> (b d) t')
        
        x, _ = self.mask_filter(x, mask) # x: [(b d) t d_model]
        x = rearrange(x, '(b d) t d_model -> b t (d d_model)', b = B)
        # print(x.shape)
        return x
        
        
class DataEmbeddingWithMask(nn.Module):
    def __init__(self, seq_len, c_in, d_model, n_heads, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbeddingWithMask, self).__init__()

        self.value_embedding = TokenEmbeddingWithMask(seq_len, c_in, d_model, n_heads, dropout=dropout)
        self.position_embedding = PositionalEmbedding(d_model=d_model*c_in)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model*c_in, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model*c_in, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, x_mask):
        if x_mark is None:
            x = self.value_embedding(x, x_mask) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x, x_mask) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)