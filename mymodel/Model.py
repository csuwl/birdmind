from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class MLA(nn.Module):
    def __init__(self):
        super(MLA, self).__init__()


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()


class Block(nn.Module):
    """
    主层堆叠
    """

    def __init__(self, layer_id: int):
        super().__init__()
        self.attn = MLA()


class RMSNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNormLayer, self).__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, [self.dim], self.weight, self.eps)


class Model(torch.nn.Module):
    def __init__(self, vocab_size=4096 * 4, embedding_dim=2048, block_size=10, ):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = torch.nn.ModuleList()
        for i in range(block_size):
            self.blocks.append(Block(i))
        self.rms_norm_layer = RMSNormLayer(embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """

        :param tokens:
        :return:
        """
        input_vector = self.embedding(tokens)
        mask = torch.full((seqlen, seqlen), float("-inf")).triu_(1)
        for block in self.blocks:
            output = block(input_vector, start_pos, mask)
        output = self.rms_norm_layer(output)[:, -1]
        logits = self.linear(output)
        return logits


if __name__ == "__main__":
    embedding = nn.Embedding(2048 * 4, 128)
    vocab_size = 20
    x = torch.randint(0, vocab_size, (2, 128))
    print(x)
    embed = embedding(x)
    print(embed)
    rms = RMSNormLayer(128)
    res = rms(embed)
    print(res)

    seqlen = 10
    mask = torch.full((seqlen, seqlen), float("-inf")).triu_(1)
    print(mask)
