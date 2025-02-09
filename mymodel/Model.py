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
    max_seq_len = 4096 * 4
    max_batch_size = 8
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

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.embedding_dim
        self.n_head = args.num_heads
        self.qk_dim = args.qk_dim
        self.v_dim = args.v_dim

        self.wq = nn.Linear(self.dim, self.qk_dim * self.n_head)
        self.wk = nn.Linear(self.dim, self.qk_dim * self.n_head)
        self.wv = nn.Linear(self.dim, self.v_dim * self.n_head)
        self.wo = nn.Linear(self.v_dim * self.n_head, self.dim)

        self.register_buffer("k_cache",
                             torch.zeros(args.max_batch_size, args.max_seq_len, self.n_head, self.qk_dim),
                             persistent=False)
        self.register_buffer("v_cache",
                             torch.zeros(args.max_batch_size, args.max_seq_len, self.n_head, self.v_dim),
                             persistent=False)

    def get_position_embedding(self, x: torch.Tensor, head_num: int) -> torch.Tensor:
        """
         x : last 2 dimension is same
         return position shape  (head, seq_len, seq_len)
        """

        seq_len = x.shape[-1]
        position = torch.zeros(head_num, seq_len, seq_len, dtype=x.dtype, device=x.device)
        for head in range(head_num):
            for i in range(seq_len):
                for j in range(seq_len):
                    if i < j:
                        continue
                    position[head, i, j] = torch.tensor(- (i - j) * 2 ** (-(head + 1)),dtype=torch.float64)
        return position

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
        # add position_embedding
        score += self.get_position_embedding(score, self.n_head)

        #  batch,head,seq_len,seq_len
        if mask is not None:
            score += mask
        score = score.softmax(dim=-1, dtype=torch.float64).type_as(x)

        v = self.wv(x)
        # batch,seq_len,head,v_dim
        v = v.view(batch_size, sequence_len, self.n_head, self.v_dim)

        # v * score
        out = torch.einsum('bhsS,bshv->bShv', score, v)
        # batch,seq_len,dim
        out = self.wo(out.flatten(2))
        return out


class Gate(torch.nn.Module):
    """
    门控结构
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.embedding_dim
        self.top_k = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.top_k_groups = args.n_expert_groups
        self.score_func = args.score_func
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.embedding_dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts))
        self.route_scale = args.route_scale

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        #        x: -1,dim
        scores = F.linear(x, self.weight)
        # -1, n_routed_experts
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias

        indices = torch.topk(scores, self.top_k, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights *= self.route_scale
        return weights.type_as(x), indices


class MLP(torch.nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, out_dim)
        self.w2 = nn.Linear(out_dim, dim)
        self.w3 = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Expert(nn.Module):
    """
    专家
    """

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, out_dim)
        self.w2 = nn.Linear(out_dim, dim)
        self.w3 = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    混合专家模型
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.embedding_dim
        self.gate = Gate(args)
        self.n_expert_groups = args.n_expert_groups
        self.experts = nn.ModuleList(
            [Expert(args.embedding_dim, args.moe_inter_dim) for _ in range(args.n_routed_experts)])
        self.shared_experts = MLP(args.embedding_dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, _ = x.size()
        shape = x.shape
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_expert_groups).tolist()
        # -1,dim
        y = torch.zeros_like(x)

        for i in range(len(counts)):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)

        return (y + z).view(shape)


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
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, [self.dim], self.weight,None, self.eps)


class Model(torch.nn.Module):
    """
    主要模型类
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.blocks = torch.nn.ModuleList()
        for i in range(args.block_size):
            self.blocks.append(Block(i, args))
        self.rms_norm_layer = RMSNormLayer(args.embedding_dim)
        self.linear = nn.Linear(args.embedding_dim, args.vocab_size)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        主model
        :param start_pos:
        :param tokens:
        :return:
        """
        seqlen = tokens.size(1)
        output = self.embedding(tokens)
        mask = torch.full((seqlen, seqlen), float("-inf")).triu_(1)
        for block in self.blocks:
            output = block(output, start_pos, mask)
        output = self.rms_norm_layer(output)
        # batch_size,seq_len,vocab_size
        logits = self.linear(output)
        return logits

    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        主model
        :param start_pos:
        :param tokens:
        :return:
        """
        seq_len=tokens.size(-1)
        input_vector = self.embedding(tokens)
        mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)
        for block in self.blocks:
            output = block(input_vector, start_pos, mask)
        output = self.rms_norm_layer(output)[:, -1]
        # batch_size,seq_len,1
        logits = self.linear(output)
        return logits


if __name__ == "__main__":
    head_num = 8
    seq_len = 20


    att = torch.nn.MultiheadAttention(10, 10)
    mask = torch.full((5, 5), float("-inf")).triu_(1)
    score = torch.randn((5,5))
    print(mask)
    print(score)
    score +=mask
    print(score)
    score = score.softmax(-1,dtype=torch.float64)
    print(score)


