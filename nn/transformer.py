#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : transformer.py
# @Date    : 2021/06/28
# @Time    : 17:05:19


import torch
from torch import nn


def pad_mask(q, k=None) -> torch.BoolTensor:
    """
    Args:
        q: [batch_size, seq_len]
        k: [batch_size, seq_len]. Defaults to None.
    Returns:
        BoolTensor: [batch_size, seq_len, seq_len]
    """
    q_mask = q.bool().unsqueeze(2)  # -> [batch_size, seq_len, 1]
    k_mask = k.bool().unsqueeze(1) if k is not None else q_mask.transpose(-1, -2)  # -> [batch_size, 1, seq_len]
    return q_mask & k_mask  # -> [batch_size, seq_len, seq_len]


def subsequence_mask(x) -> torch.BoolTensor:
    """
    Args:
        x: [batch_size, seq_len]
    Returns:
        BoolTensor: [batch_size, seq_len, seq_len]
    """
    seq_len = x.size(1)
    return torch.ones(seq_len, seq_len).tril().bool()


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            Tensor: [batch_size, seq_len, d_model]
        """
        return self.embed(x) * self.d_model**.5


class PostionalEncoding(nn.Module):
    r"""
    PE(2i)     = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
    PE(2i + 1) = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
    """

    def __init__(self, d_model, max_len):
        super(PostionalEncoding, self).__init__()
        # [max_len, d_model]
        self.encoding = nn.Parameter(torch.empty(max_len, d_model), requires_grad=False)
        # postion
        pos = torch.arange(0, max_len, 1.0).unsqueeze(-1)
        exp = torch.arange(0, d_model, 2.0) / d_model
        # encoding
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** exp))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** exp))

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            Tensor: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return self.encoding[:seq_len]


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout=.1):
        super(Embeddings, self).__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.encode = PostionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            Tensor: [batch_size, seq_len, d_model]
        """
        return self.dropout(self.embed(x) + self.encode(x))


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
        out = self.proj(v)  # -> [batch_size, seq_len, d_model]
        return self.dropout(out) if self.dropout else out


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
    def __init__(self, d_model, d_ffn, n_heads=8, dropout=.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_hidden=d_ffn, dropout=dropout)
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
    def __init__(self, vocab_size, d_model, max_len, d_ffn, n_layers=6, n_heads=8, dropout=.1):
        super().__init__()
        self.embed = Embeddings(vocab_size=vocab_size, d_model=d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_ffn=d_ffn, n_heads=n_heads, dropout=dropout)
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


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads=8, dropout=.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.attn2 = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_hidden=d_ffn, dropout=dropout)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, self_attn_mask, cross_attn_mask):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            memory: [batch_size, seq_len, d_model], output of encoder
            self_attn_mask (BoolTensor): [seq_len, seq_len]
            cross_attn_mask (BoolTensor): [batch_size, 1, seq_len, seq_len]
        Returns:
            Tensor: [batch_size, seq_len, d_model]
        """
        res = x  # residual
        x = self.attn1(x, x, x, mask=self_attn_mask)  # self attention
        x = self.norm1(x + res)  # add & norm
        x = self.dropout1(x)  # dropout
        res = x  # residual
        x = self.attn2(x, memory, memory, mask=cross_attn_mask)  # decoder-encoder attention
        x = self.norm2(x + res)  # add & norm
        x = self.dropout2(x)  # dropout
        res = x  # residual
        x = self.ffn(x)  # feed forward
        x = self.norm3(x + res)  # add & norm
        x = self.dropout3(x)  # dropout
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, d_ffn, n_layers=6, n_heads=8, dropout=.1):
        super().__init__()
        self.embed = Embeddings(vocab_size=vocab_size, d_model=d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_ffn=d_ffn, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, source, target, memory):
        """
        Args:
            source (LongTensor): [batch_size, seq_len]
            target (LongTensor): [batch_size, seq_len]
            memory: [batch_size, seq_len, d_model], output of encoder,
        Returns:
            [batch_size, seq_len, d_model]
        """
        # calculate mask
        self_attn_mask = pad_mask(target) & subsequence_mask(target)
        cross_attn_mask = pad_mask(target, source)
        # calcuate output
        x = self.embed(target)
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask, cross_attn_mask)
        return x


class Generator(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x).log_softmax(dim=-1)  # [batch_size, seq_len, vocab_size]


class Transformer(nn.Module):
    def __init__(
        self, source_vocab_size, target_vocab_size,
        max_len=5000, d_model=512, d_ffn=2048, n_heads=8, n_layers=6, dropout=.1
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size=source_vocab_size, d_model=d_model, max_len=max_len, d_ffn=d_ffn,
                               n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.decoder = Decoder(vocab_size=target_vocab_size, d_model=d_model, max_len=max_len, d_ffn=d_ffn,
                               n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.generator = Generator(target_vocab_size, d_model)

    def forward(self, source, target):
        """
        Args:
            source (LongTensor): [batch_size, seq_len]
            target (LongTensor): [batch_size, seq_len]
        Returns:
            Tensor: [batch_size, seq_len, target_vocab_size]
        """
        #========== encode ==========#
        encoding = self.encoder(source)  # -> [batch_size, seq_len, d_model]
        #========== decode ==========#
        decoding = self.decoder(source, target, encoding)  # -> [batch_size, seq_len, d_model]
        #========== generate ==========#
        output = self.generator(decoding)  # -> [batch_size, seq_len, target_vocab_size]
        return output


if __name__ == "__main__":
    sentences = [
        ['i love you <space>', '<bos> ich liebe dich', 'ich liebe dich <eos>'],
        ['you love me <space>', '<bos> du liebst mich', 'du liebst mich <eos>']
    ]
    source_vocab = ['<space>', 'i', 'love', 'you', 'me']
    target_vocab = ['<space>', '<bos>', '<eos>', 'ich', 'liebe', 'dich', 'du', 'liebst', 'mich']
    source_vocab_dict = {word: i for i, word in enumerate(source_vocab)}
    target_vocab_dict = {word: i for i, word in enumerate(target_vocab)}
    encoder_inputs = torch.tensor([[source_vocab_dict[word] for word in strs[0].split(' ')] for strs in sentences])
    decoder_inputs = torch.tensor([[target_vocab_dict[word] for word in strs[1].split(' ')] for strs in sentences])
    decoder_outputs = torch.tensor([[target_vocab_dict[word] for word in strs[2].split(' ')] for strs in sentences])

    print(source_vocab_dict)
    print(target_vocab_dict)
    print(encoder_inputs)
    print(decoder_inputs)
    print(decoder_outputs)

    transformer = Transformer(source_vocab_size=len(source_vocab), target_vocab_size=len(target_vocab), max_len=4)
    output = transformer(encoder_inputs, decoder_inputs)
    print(output.shape)
