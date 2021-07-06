#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : graphormer.py
# @Date    : 2021/07/06
# @Time    : 13:07:17


import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_model), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            Tensor: [batch_size, seq_len, d_model]
        """
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.to_q, self.to_k, self.to_v, self.proj = [nn.Linear(d_model, d_model) for _ in range(4)]
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, seq_len, d_model]
            k: [batch_size, seq_len, d_model]
            v: [batch_size, seq_len, d_model]
            mask (BoolTensor): [seq_len, seq_len] or [batch, seq_len, seq_len]. Defaults to None.
        Returns:
            Tensor: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = q.size(0), q.size(1)
        #========== calculate multi-head attention qkv ==========#
        # map to qkv
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        # [batch_size, seq_len, n_heads, d_k] -> [n_heads, batch_size, seq_len, d_k]
        q = q.view(batch_size, seq_len, self.n_heads, -1).permute(2, 0, 1, 3)
        # [batch_size, seq_len, n_heads, d_k] -> [n_heads, batch_size, seq_len, d_k]
        k = k.view(batch_size, seq_len, self.n_heads, -1).permute(2, 0, 1, 3)
        # [batch_size, seq_len, n_heads, d_v] -> [n_heads, batch_size, seq_len, d_v]
        v = v.view(batch_size, seq_len, self.n_heads, -1).permute(2, 0, 1, 3)

        #========== calculate multi-head attention tensor ==========#
        # [n_heads, batch_size, seq_len, d_k] @ [n_heads, batch_size, d_k, seq_len] / \sqrt{d_k}
        scaled_dot_prod = q @ k.transpose(-1, -2) * k.size(-1)**-.5  # -> [n_heads, batch_size, seq_len, seq_len]
        if mask is not None:
            scaled_dot_prod += torch.where(mask, .0, -1e12)
        attn = torch.softmax(scaled_dot_prod, dim=-1)  # -> [n_heads, batch_size, seq_len, seq_len]
        if self.dropout is not None:
            attn = self.dropout(attn)

        #========== calculate multi-head attention output ==========#
        # [n_heads, batch_size, seq_len, seq_len] @ [n_heads, batch_size, seq_len, d_v]
        v = attn @ v  # -> [n_heads, batch_size, seq_len, d_v]
        # [batch_size, seq_len, n_heads, d_v] -> [batch_size, seq_len, d_model]
        v = v.permute(1, 2, 0, 3).reshape(batch_size, seq_len, -1)
        return self.proj(v)  # -> [batch_size, seq_len, d_model]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=.1):
        super(PositionwiseFeedForward, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            Tensor: [batch_size, seq_len, d_model]
        """
        return self.seq(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn_hidden, n_attn_heads=8, dropout=.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_attn_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_hidden=d_ffn_hidden, dropout=dropout)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, self_attn_mask):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            self_attn_mask (BoolTensor): [seq_len, seq_len]
        Returns:
            Tensor: [batch_size, seq_len, d_model]
        """
        res = x  # residual
        x = self.attn(x, x, x, mask=self_attn_mask)  # self attention
        x = self.norm1(x + res)  # add & norm
        x = self.dropout1(x)  # dropout
        res = x  # residual
        x = self.ffn(x)  # feed forward
        x = self.norm2(x + res)  # add & norm
        x = self.dropout2(x)  # dropout
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, d_ffn_hidden, n_layers=6, n_attn_heads=8, dropout=.1):
        super().__init__()
        self.embed = Embeddings(vocab_size=vocab_size, d_model=d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_ffn_hidden=d_ffn_hidden, n_attn_heads=n_attn_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, source):
        """
        Args:
            source (LongTensor): [batch_size, seq_len]
        Returns:
            Tensor: [batch_size, seq_len, d_model]
        """
        self_attn_mask = pad_mask(source)
        x = self.embed(source)
        for layer in self.layers:
            x = layer(x, self_attn_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self, source_vocab_size, target_vocab_size,
        max_len=5000, d_model=512, d_ffn_hidden=2048, n_attn_heads=8, n_layers=6, dropout=.1
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size=source_vocab_size, d_model=d_model, max_len=max_len, d_ffn_hidden=d_ffn_hidden,
                               n_layers=n_layers, n_attn_heads=n_attn_heads, dropout=dropout)
        self.generator = Generator(target_vocab_size, d_model)

    def forward(self, source):
        #========== encode ==========#
        return self.encoder(source)  # -> [batch_size, seq_len, d_model]

# %%
