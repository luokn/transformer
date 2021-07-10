#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : mixer.py
# @Date    : 2021/07/09
# @Time    : 17:10:42


from functools import partial
import torch
from torch import nn


class Lambda(nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self, net):
        super(Residual, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net(x) + x


class ToPatches(nn.Module):
    def __init__(self, out_features, n_channels, n_patches, patch_h, patch_w):
        super(ToPatches, self).__init__()
        self.n_patches, self.h, self.w = n_patches, patch_h, patch_w
        self.proj = nn.Linear(self.h * self.w * n_channels, out_features)

    def forward(self, x):
        batch_size, h, w, _ = x.size()
        patches = x.reshape(batch_size, h // self.h, self. h, w // self.w, self.w, -1)
        return self.proj(patches.transpose(2, 3).reshape(batch_size, self.n_patches, -1))


class FeedForward(nn.Sequential):
    def __init__(self, n_features, exp_factor, dropout, dense="cov"):
        Dense = partial(nn.Conv1d, kernel_size=1) if dense == "cov" else nn.Linear
        super(FeedForward, self).__init__(
            Dense(n_features, n_features * exp_factor), nn.GELU(), nn.Dropout(dropout),
            Dense(n_features * exp_factor, n_features), nn.Dropout(dropout)
        )


class Encoder(nn.Sequential):
    def __init__(self, d_model, n_patches, exp_factor, depth, dropout):
        super(Encoder, self).__init__(*[
            nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(d_model),
                    FeedForward(n_patches, exp_factor, dropout, dense='cov')
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(d_model),
                    FeedForward(d_model, exp_factor, dropout, dense='linear')
                ))
            ) for _ in range(depth)
        ])


class MLPMixer(nn.Sequential):
    def __init__(
        self, image_size, patch_size, n_channels, n_classes,
        d_model=512, exp_factor=4, depth=12, dropout=0.1
    ):
        patch_h, patch_w = patch_size
        assert image_size[0] % patch_h == 0 and image_size[1] % patch_w == 0
        n_patches = (image_size[0] // patch_h) * (image_size[1] // patch_w)
        super(MLPMixer, self).__init__(
            ToPatches(d_model, n_channels, n_patches, patch_h, patch_w),
            Encoder(d_model, n_patches, exp_factor, depth, dropout=dropout),
            nn.LayerNorm(d_model),
            Lambda(lambda x: torch.mean(x, dim=1)),
            nn.Linear(d_model, n_classes),
        )


if __name__ == "__main__":
    mixer = MLPMixer(image_size=[100, 100], patch_size=[10, 10], n_channels=3, n_classes=5)
    x = torch.randn(1, 100, 100, 3)
    print(mixer(x).shape)
