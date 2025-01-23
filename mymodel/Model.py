from dataclasses import dataclass
from typing import Literal, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass
class ModelArgs:

    vocab_size: int = 102400
    embedding_dim: int = 2048
    block_size: int = 10

    # MHA
    num_heads: int = 8
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    qk_dim: int = 128
    v_dim: int = 128

    # moe
    moe_inter_dim = 1408
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.


class MHA(nn.Module):
    """
    注意力层
    """

    def __init__(self, args: ModelArgs):
        super(MHA, self).__init__()
        self.dim = args.embedding_dim
        self.n_head = args.num_heads
        self.qk_dim = args.qk_dim
        self.v_dim = args.v_dim

        self.wq = nn.Linear(self.dim, self.qk_dim * self.n_head)
        self.wk = nn.Linear(self.dim, self.qk_dim * self.n_head)
        self.wv = nn.Linear(self.dim, self.v_dim * self.n_head)
        self.wo = nn.Linear(self.v_dim * self.n_head, self.dim)

    def forward(self, x: torch.Tensor, start_pos: int, mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, _ = x.size()
        end_pos = start_pos + sequence_len
        q = self.wq(x)
        k = self.wk(x)

        q = q.view(batch_size, sequence_len, self.n_head, self.qk_dim)
        k = k.view(batch_size, sequence_len, self.n_head, self.qk_dim)

        #         batch,seq_len,head,qk_dim
        score = torch.einsum('bshk,bShk->bhsS', q, k)
        score = score / self.qk_dim ** 0.5
        #  batch,head,seq_len,seq_len
        score = score.softmax(dim=-1, dtype=x.dtype)

        v = self.wv(x)
        # batch,seq_len,head,v_dim
        v = v.view(batch_size, sequence_len, self.n_head, self.v_dim)

        # v * score
        out = torch.einsum('bhsS,bshv->bShv', score, v)
        # batch,seq_len,dim
        out = self.wo(out.flatten(2))
        return out


class Gate(nn.Module):
    """
    门控结构
    """

    def __init__(self, args: ModelArgs):
        self.dim = args.embedding_dim
        self.top_k = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.top_k_groups = args.n_expert_groups
        self.score_func = args.score_func
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.embedding_dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        #         -1,dim
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias

        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights *= self.route_scale
        return weights.type_as(x), indices


class MLP(torch.nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(dim, out_dim)
        self.w2 = nn.Linear(out_dim, dim)
        self.w3 = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE:
    """
    混合专家模型
    """

    def __init__(self, args: ModelArgs):
        super(MoE, self).__init__()
        self.dim = args.embedding_dim
        self.gate = Gate(args)

        self.shared_experts = MLP(args.embedding_dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, _ = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        torch.bincount(indices.flatten(), minlength=args.n_expert_groups).tolist()

        z = self.shared_experts(x)

#         todo


class Block(nn.Module):
    """
    主层堆叠
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        # attention
        self.attn = MHA(layer_id, args)
        self.moe = MoE(args)
        self.attn_norm = RMSNormLayer(args.embedding_dim)
        self.ffn_norm = RMSNormLayer(args.embedding_dim)

    def forward(self, x: torch.Tensor, start_pos: int, mask: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        # batch,seq_len,dim
        x = x + self.attn(self.attn_norm(x), start_pos, mask)
        # batch, seq_len, dim
        x = x + self.moe(self.ffn_norm(x))
        return x


class RMSNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNormLayer, self).__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, [self.dim], self.weight, self.eps)


class Model(torch.nn.Module):
    """
    主要模型类
    """

    def __init__(self, args: ModelArgs):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.blocks = torch.nn.ModuleList()
        for i in range(args.block_size):
            self.blocks.append(Block(i, args))
        self.rms_norm_layer = RMSNormLayer(args.embedding_dim)
        self.linear = nn.Linear(args.embedding_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        主model
        :param start_pos:
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
    args = ModelArgs()
    routed_weight = torch.empty(args.n_routed_experts, args.embedding_dim)
    print(routed_weight)

    test_input = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
                                  , [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]])
    print(test_input)
    print(test_input.flatten(1))

    score = torch.einsum('bsk,bak->bsa', test_input, test_input)

    test_input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    score = torch.einsum("is_j,ks_j->ik", test_input, test_input)
    print(score)

    print(test_input.shape)
    test_input = test_input[:, -1]
    print(test_input.shape)
    print(test_input)

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
