#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : vit.py
# @Date    : 2021/07/06
# @Time    : 14:07:08


import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(n_features), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(n_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Sequential):
    def __init__(self, n_features, d_hidden, dropout=0.1):
        super(FeedForward, self).__init__(
            nn.Linear(n_features, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_features),
            nn.Dropout(dropout)
        )


class SelfMultiHeadAttention(nn.Module):
    def __init__(self, n_features, d_attn, n_heads=8, bias=False, dropout=.1):
        super(SelfMultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.to_qkv = nn.Linear(n_features, 3 * n_heads * d_attn, bias=bias)
        self.proj = nn.Linear(n_heads * d_attn, n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_patches = x.size(0), x.size(1)
        # [batch_size, n_patches, 3 × n_heads × d_attn] -> [batch_size, n_patches, 3, n_heads, d_attn] ->
        # [batch_size, n_heads, 3, n_patches, d_attn] -> 3 × [batch_size, n_heads, n_patches, d_attn]
        q, k, v = self.to_qkv(x).reshape(batch_size, n_patches, 3, self.n_heads, -1).transpose(1, 3).unbind(2)
        # [batch_size, n_heads, n_patches, d_attn] @ [batch_size, n_heads, d_attn, n_patches] / \sqrt{d_attn}
        scaled_dot_prod = q @ k.transpose(-1, -2) * k.size(-1)**-.5  # -> [ batch_size, n_heads, n_patches, n_patches]
        attn = torch.softmax(scaled_dot_prod, dim=-1)  # -> [batch_size, n_heads, n_patches, n_patches]
        # [batch_size, n_heads, n_patches, n_patches] @ [batch_size, n_heads, n_patches, d_attn]
        v = attn @ v  # -> [batch_size, n_heads, n_patches, d_attn]
        v = v.transpose(1, 2).reshape(batch_size, n_patches, -1)  # -> [batch_size, n_patches, n_heads × d_attn]
        return self.dropout(self.proj(v))  # -> [ batch_size, n_patches, n_features]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_attn, d_ffn, n_heads=8, dropout=.1):
        super(EncoderLayer, self).__init__()
        self.attn = SelfMultiHeadAttention(d_model, d_attn, n_heads, dropout=dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(d_model, d_ffn, dropout=dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        res = x  # residual
        x = self.attn(x)  # self attention
        x = self.norm1(x + res)  # add & norm
        x = self.dropout1(x)  # dropout
        res = x  # residual
        x = self.ffn(x)  # feed forward
        x = self.norm2(x + res)  # add & norm
        x = self.dropout2(x)  # dropout
        return x


class Encoder(nn.Sequential):
    def __init__(self, d_model, d_attn, d_ffn, n_heads=8, n_layers=6, dropout=.1):
        super(Encoder, self).__init__(*[
            EncoderLayer(d_model, d_attn, d_ffn, n_heads, dropout=dropout) for _ in range(n_layers)
        ])


class PositionEmbed(nn.Module):
    def __init__(self, d_model, n_patches):
        super(PositionEmbed, self).__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.empty(d_model), requires_grad=True)
        self.embedding = nn.Parameter(torch.empty(n_patches + 1, d_model), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, 1, self.d_model)  # -> [batch_size, 1, d_model]
        x = torch.cat([cls_token, x], dim=1)  # -> [batch_size, n_patches + 1, d_model]
        x += self.embedding
        return x  # -> [batch_size, n_patches + 1, d_model]


class PatchEmbed(nn.Module):
    def __init__(self, d_model, n_channels, n_patches, patch_h, patch_w):
        super(PatchEmbed, self).__init__()
        self.n_patches, self.patch_h, self.patch_w = n_patches, patch_h, patch_w
        self.proj = nn.Linear(self.patch_h * self.patch_w * n_channels, d_model)

    def forward(self, x):
        batch_size, h, w, _ = x.size()
        patches = x.reshape(batch_size, h // self.patch_h, self. patch_h, w // self.patch_w, self.patch_w, -1)
        return self.proj(patches.transpose(2, 3).reshape(batch_size, self.n_patches, -1))


class MLPHead(nn.Sequential):
    def __init__(self, d_model, n_classes):
        super(MLPHead, self).__init__(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )


class ViT(nn.Module):
    def __init__(
        self, image_size, patch_size, n_channels, n_classes,
        d_model=64, d_attn=64, d_ffn=2048, n_heads=8, n_layers=6, pool='cls', dropout=.1
    ):
        super(ViT, self).__init__()
        patch_h, patch_w = patch_size
        assert image_size[0] % patch_h == 0 and image_size[1] % patch_w == 0
        assert pool in ['cls', 'mean']
        self.pool = pool
        n_patches = (image_size[0] // patch_h) * (image_size[1] // patch_w)
        self.to_patches = PatchEmbed(d_model, n_channels, n_patches, patch_h, patch_w)
        self.embed = PositionEmbed(d_model, n_patches)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(d_model, d_attn, d_ffn, n_heads, n_layers, dropout=dropout)
        self.mlp_head = MLPHead(d_model, n_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): [batch_size, height, width, channels]
        Returns:
            Tensor:  [batch_size, n_classes]
        """
        # to patches
        x = self.to_patches(x)  # -> [batch_size, n_patches + 1, d_model]
        # embed
        x = self.embed(x)  # -> [batch_size, n_patches + 1, d_model]
        # encode
        x = self.encoder(x)  # -> [batch_size, n_patches + 1, d_model]
        # pool
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # -> [batch_size, d_model]
        # output
        return self.mlp_head(x)  # -> [batch_size, n_classes]


if __name__ == '__main__':
    B, H, W, C, h, w = 32, 100, 100, 3, 10, 10
    vit = ViT(image_size=[H, W], patch_size=[h, w], n_channels=3, n_classes=5)
    x = torch.randn(B, H, W, C)
    y = vit(x)
    print(y.shape)
