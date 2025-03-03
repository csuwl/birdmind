from dataclasses import dataclass
from typing import Literal, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F



@dataclass
class ModelArgs:
    vocab_size: int = 6400
    embedding_dim: int = 512
    block_size: int = 8

    # MHA
    max_seq_len = 4096 * 4
    max_batch_size = 8
    num_heads: int = 8
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    qk_dim: int = 128
    v_dim: int = 128

    # moe
    moe_inter_dim = 512
    n_routed_experts: int = 4
    n_shared_experts: int = 2
    n_activated_experts: int = 2
    n_expert_groups: int = 4
    score_func: Literal["softmax", "sigmoid"] = "softmax"


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

        if torch.isnan(position).any():
            raise ValueError("position value is nan")
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
        self.score_func = args.score_func
        self.weight = nn.Parameter(torch.randn(args.n_routed_experts, args.embedding_dim))
        self.bias = nn.Parameter(torch.randn(args.n_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        bsz, seq_len, h = x.shape
        hidden_states = x.view(-1, h)
        if torch.isnan(hidden_states).any():
            raise ValueError("scores value is nan")

        logits = F.linear(hidden_states, self.weight, self.bias)
        if self.score_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            scores = logits.sigmoid()

        if torch.isnan(scores).any():
            raise ValueError("scores value is nan")

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if torch.isnan(topk_weight).any():
            raise ValueError("topk_weight value is nan")

        if self.top_k > 1 :
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training :
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            # 序列级辅助损失
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_groups, device=hidden_states.device)
            ce.scatter_add_(1, topk_idx_for_aux_loss,
                            torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                seq_len * aux_topk / self.n_groups)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


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
        self.n_activated_experts=args.n_activated_experts
        self.experts = nn.ModuleList(
            [Expert(args.embedding_dim, args.moe_inter_dim) for _ in range(args.n_routed_experts)])
        self.shared_experts = MLP(args.embedding_dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.n_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_expert_groups is not None:
            y = y + self.shared_experts(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.n_activated_experts
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache



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
        if torch.isnan(mask).any():
            raise Exception("nan 错误")
        for block in self.blocks:
            output = block(output, start_pos, mask)
        output = self.rms_norm_layer(output)
        # batch_size,seq_len,vocab_size
        logits = self.linear(output)
        aux_loss = sum(l.moe.aux_loss for l in self.blocks)
        return logits, aux_loss

    @torch.inference_mode()
    def generate(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
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
        # batch_size,1，vocab_size
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


