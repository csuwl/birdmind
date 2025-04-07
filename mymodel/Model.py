from dataclasses import dataclass
from typing import Literal, Any,Optional,List,Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import AutoTokenizer,PretrainedConfig,PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import os


@dataclass
class ModelArgs(PretrainedConfig):
    model_type = "birdmind"
    
    def __init__(self, *,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 vocab_size: int = 6400,
                 embedding_dim: int = 512,
                 block_size: int = 9, 
                 max_seq_len: int = 4096,
                 num_heads: int = 8,
                 qk_dim: int = 128,
                 v_dim: int = 128,
                 moe_inter_dim: int = 512,
                 n_expert_groups: int = 6,
                 n_shared_experts: int = 2,
                 n_activated_experts: int = 2,
                 score_func: Literal["softmax", "sigmoid"] = "softmax",
                 train:bool=False,
                 **kwargs):
        
        self.device: Any = device
        self.vocab_size: int = vocab_size 
        self.embedding_dim: int = embedding_dim
        self.block_size: int = block_size

        # MHA
        self.max_seq_len = max_seq_len
        self.num_heads: int = num_heads
        self.qk_dim: int = qk_dim
        self.v_dim: int = v_dim

        # moe
        self.moe_inter_dim = moe_inter_dim
        self.n_expert_groups: int = n_expert_groups
        self.n_shared_experts: int = n_shared_experts
        self.n_activated_experts: int = n_activated_experts
        self.score_func: Literal["softmax", "sigmoid"] = score_func
        self.train = train
        super().__init__(**kwargs)


class MHA(nn.Module):
    """
    注意力层
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.mha_id = layer_id
        self.dim = args.embedding_dim
        self.n_head = args.num_heads
        self.qk_dim = args.qk_dim
        self.v_dim = args.v_dim

        self.wq = nn.Linear(self.dim, self.qk_dim * self.n_head)
        self.wk = nn.Linear(self.dim, self.qk_dim * self.n_head)
        self.wv = nn.Linear(self.dim, self.v_dim * self.n_head)
        self.wo = nn.Linear(self.v_dim * self.n_head, self.dim)
    

    def forward(self, x: torch.Tensor, start_pos: int, mask: torch.Tensor,pos_embedding:torch.Tensor,past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False) -> torch.Tensor:
        batch_size, sequence_len, _ = x.size()

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # batch,seq_len,head,v_dim
        q = q.view(batch_size, sequence_len, self.n_head, self.qk_dim)
        k = k.view(batch_size, sequence_len, self.n_head, self.qk_dim)
        v = v.view(batch_size, sequence_len, self.n_head, self.v_dim)
                
        # kv_cache实现
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        #         batch,seq_len,head,qk_dim
        score = torch.einsum('bshk,bShk->bhsS', q, k)
        score = score / self.qk_dim ** 0.5
        
        score += pos_embedding

        #  batch,head,seq_len,seq_len
        if mask is not None:
            score += mask[:sequence_len,:sequence_len]
        score = score.softmax(dim=-1).type_as(x)


        # v * score
        out = torch.einsum('bhsS,bShv->bshv', score, v)
        # batch,seq_len,dim
        out = self.wo(out.flatten(2))
        return out, past_kv


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
        self.weight = nn.Parameter(torch.randn(args.n_expert_groups, args.embedding_dim))
        self.bias = nn.Parameter(torch.randn(args.n_expert_groups))

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        bsz, seq_len, h = x.shape
        hidden_states = x.view(-1, h)

        logits = F.linear(hidden_states, self.weight, self.bias)
        if self.score_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            scores = logits.sigmoid()

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)


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
            [Expert(args.embedding_dim, args.moe_inter_dim) for _ in range(args.n_expert_groups)])
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
            y = torch.randn_like(x, dtype=torch.float16)
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
        self.block_id = layer_id
        self.attn = MHA(layer_id, args)
        self.moe = MoE(args)
        self.attn_norm = RMSNormLayer(args.embedding_dim)
        self.ffn_norm = RMSNormLayer(args.embedding_dim)

    def forward(self, x: torch.Tensor, start_pos: int, mask: torch.Tensor, pos_embedding: torch.Tensor, past_key_value = None, use_cache: bool = False,) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        # batch,seq_len,dim
        h_att, past_kv = self.attn(self.attn_norm(x), start_pos, mask, pos_embedding, past_key_value=past_key_value, use_cache=use_cache)
        x = x + h_att
        # batch, seq_len, dim
        x = x + self.moe(self.ffn_norm(x))
        return x , past_kv


class RMSNormLayer(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class Model(PreTrainedModel):
    config_class = ModelArgs
    """
    主要模型类
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        if args.train:
            self.train()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.blocks = torch.nn.ModuleList()
        for i in range(args.block_size):
            self.blocks.append(Block(i, args))
        self.rms_norm_layer = RMSNormLayer(args.embedding_dim)
        self.linear = nn.Linear(args.embedding_dim, args.vocab_size)
        print("初始化position embedding")
        if self.training:
            self.register_buffer("alibi",self.get_position_embedding(args.max_seq_len,args.num_heads,torch.device('cuda')),persistent=False)
        else:
            self.register_buffer("alibi",self.get_position_embedding(args.max_seq_len,args.num_heads,torch.device('cpu')),persistent=False)
        print("结束初始化position embedding")
        self.register_buffer("mask",torch.full((args.max_seq_len, args.max_seq_len), float("-inf"),device=args.device, requires_grad=False).triu_(1),persistent=False)
        
        self.OUT = CausalLMOutputWithPast()
        

    def forward(self, input_ids: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False, start_pos: int = 0, **args) -> torch.Tensor:
        """
        主model
        :param start_pos:
        :param tokens:
        :return:
        """
        # if past_key_values is None or len(past_key_values) == 0 or past_key_values[0] is None:
        #     pos_cis = self.alibi[:,start_pos:start_pos + input_ids.size(1),start_pos:start_pos+input_ids.size(1)]
        # else:
        #     k_len = past_key_values[0][0].size(1)
        #     pos_cis = self.alibi[:,start_pos:start_pos + input_ids.size(1),start_pos:k_len]
        sequence_len = input_ids.shape[1]
        if self.training:
            pos_cis = self.alibi[:,start_pos:start_pos+sequence_len,:start_pos+sequence_len]
        else:
            pos_cis = self.alibi[:,start_pos:start_pos+sequence_len,:start_pos+sequence_len].to(input_ids.device)

        past_key_values = past_key_values or [None] * len(self.blocks)
        output = self.embedding(input_ids)
        past_kvs = []
        for i,block in enumerate(self.blocks):
            output, past_kv = block(output, start_pos, self.mask, pos_cis ,past_key_value = past_key_values[i], use_cache = use_cache)
            past_kvs.append(past_kv)
        output = self.rms_norm_layer(output)
        # batch_size,seq_len,vocab_size
        logits = self.linear(output)
        aux_loss = sum(l.moe.aux_loss for l in self.blocks)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
    

    def get_position_embedding(self, seq_len: int, head_num: int, device) -> torch.Tensor:
        """
         x : last 2 dimension is same
         return position shape  (head, seq_len, seq_len)
        """

        position = torch.zeros(head_num, seq_len, seq_len, device=device,requires_grad=False)
        for head in range(head_num):
            for i in range(seq_len):
                for j in range(seq_len):
                    if i < j:
                        continue
                    position[head, i, j] = torch.tensor(- (i - j) * 2 ** (-(head + 1)),device=device,requires_grad=False)
        return position


    # 参考 minimind
    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break





    @torch.inference_mode()
    def generate_my(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        主model
        :param start_pos:
        :param tokens:
        :return:
        """
        tokens.to(self.mask.device)

        output = self.embedding(tokens)
       
        for block in self.blocks:
            output, past_kv = block(output, start_pos, self.mask ,self.alibi)
        output = self.rms_norm_layer(output)
        # batch_size,seq_len，vocab_size
        logits = self.linear(output)
        logits = logits[:, -1, :]
        return logits
    
    @staticmethod
    def init_model(args: ModelArgs,load_path:str = "./model.pth" ):
        tokenizer = AutoTokenizer.from_pretrained('./minimind_tokenizer')
        
        model = Model(args)
        if os.path.exists(load_path):
            model.load_state_dict(torch.load(load_path))
        print(model)
        model.to(args.device)
        
        print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        
        return tokenizer, model


if __name__ == "__main__":
    pass




