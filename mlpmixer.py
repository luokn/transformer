#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : mixer.py
# @Date    : 2021/07/09
# @Time    : 17:10:42


from functools import partial

import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn


class Residual(nn.Module):
    def __init__(self, net):
        super(Residual, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net(x) + x


class FeedForward(nn.Sequential):
    def __init__(self, dim, exp_factor, dense="cov", dropout=.1):
        Dense = partial(nn.Conv1d, kernel_size=1) if dense == "cov" else nn.Linear
        super(FeedForward, self).__init__(
            Dense(dim, dim * exp_factor), nn.GELU(), nn.Dropout(dropout),
            Dense(dim * exp_factor, dim), nn.Dropout(dropout)
        )


class Encoder(nn.Sequential):
    def __init__(self, dim, n_patches, exp_factor, depth, dropout=.1):
        super(Encoder, self).__init__(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    FeedForward(n_patches, exp_factor, dense='cov', dropout=dropout),
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(dim),
                    FeedForward(dim, exp_factor, dense='linear', dropout=dropout)
                ))
            ) for _ in range(depth)
        ])


class MLPMixer(nn.Sequential):
    def __init__(
        self, image_size, patch_size, n_channels, n_classes,
        dim=512, exp_factor=4, depth=12, dropout=0.1
    ):
        patch_h, patch_w = patch_size
        n_patches = (image_size[0] // patch_h) * (image_size[1] // patch_w)
        super(MLPMixer, self).__init__(
            Rearrange('b (r h) (s w) c -> b (r s) (h w c)', h=patch_h, w=patch_w),
            nn.Linear(patch_h * patch_w * n_channels, dim),
            Encoder(dim, n_patches, exp_factor, depth, dropout=dropout),
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', reduction='mean'),
            nn.Linear(dim, n_classes),
        )


if __name__ == "__main__":
    mixer = MLPMixer(image_size=[100, 100], patch_size=[10, 10], n_channels=3, n_classes=5)
    x = torch.randn(1, 100, 100, 3)
    print(mixer(x).shape)
