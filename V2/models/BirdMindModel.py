from dataclasses import dataclass
from typing import Literal, Any,Optional,List,Tuple,Union,Callable,Unpack


import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import AutoTokenizer,PretrainedConfig,PreTrainedModel,GenerationMixin
from transformers.cache_utils import DynamicCache,Cache,StaticCache,SlidingWindowCache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import os
from transformers.utils import is_torch_flex_attn_available

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations.flex_attention import make_flex_block_causal_mask


@dataclass
class BirdMindConfig(PretrainedConfig):
    model_type = "birdmind"
    
    def __init__(self, *,
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 vocab_size: int = 151936,
                 embedding_dim: int = 512,
                 block_size: int = 12, 
                 max_seq_len: int = 4096,
                 num_heads: int = 8,
                 use_moe: bool = True,
                 qk_dim: int = 128,
                 v_dim: int = 128,
                 moe_inter_dim: int = 512,
                 n_expert_groups: int = 6,
                 n_shared_experts: int = 2,
                 n_activated_experts: int = 2,
                 score_func: Literal["softmax", "sigmoid"] = "softmax",
                 use_flash_attention:bool=False,
                 router_aux_loss_coef=0.001,
                 use_sliding_window = False,
                 **kwargs):
        
        self.device: str = device
        self.vocab_size: int = vocab_size 
        self.embedding_dim: int = embedding_dim
        self.block_size: int = block_size
        self.use_moe:bool = use_moe
        self.use_flash_attention = use_flash_attention

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
        self.router_aux_loss_coef = router_aux_loss_coef
        self.use_sliding_window = use_sliding_window
        super().__init__(**kwargs)


def get_alibi_bias(num_heads: int, position_ids: torch.Tensor) -> torch.Tensor:
        """
        根据动态 position_ids 生成 ALiBi (Attention with Linear Biases) 位置偏置
        
        参数:
            num_heads (int): 注意力头数量
            position_ids (torch.Tensor): 位置ID张量, 形状为 [batch_size, seq_len]
        
        返回:
            torch.Tensor: ALiBi 偏置矩阵, 形状为 [batch_size, num_heads, seq_len, seq_len]
        """
        # 1. 生成每个注意力头对应的斜率 (遵循原始ALiBi的几何级数公式)
        if num_heads == 1:
            slopes = torch.tensor([1.0])
        else:
            base = 2**(-8/(num_heads-1))  # 优化计算：避免在循环中重复计算幂
            slopes = torch.pow(base, torch.arange(num_heads))
        
        # 2. 将斜率移动到与 position_ids 相同的设备
        slopes = slopes.to(position_ids.device)
        
        # 3. 计算相对位置距离矩阵 (优化点：使用广播机制避免显式展开)
        # position_ids: [batch_size, seq_len]
        relative_pos = position_ids[:, None, :] - position_ids[:, :, None]
        # relative_pos: [batch_size, seq_len, seq_len]
        
        # 4. 计算ALiBi偏置矩阵 (优化点：使用爱因斯坦求和高效计算)
        # slopes: [num_heads] -> [1, num_heads, 1, 1]
        # relative_pos: [batch_size, 1, seq_len, seq_len]
        alibi = -torch.abs(relative_pos).unsqueeze(1) * slopes.view(1, num_heads, 1, 1)
        
        # 5. 应用因果掩码 (仅保留下三角部分)
        # 创建因果掩码 [seq_len, seq_len]
        seq_len = position_ids.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=position_ids.device))
        alibi = alibi * causal_mask
    
        # 6. 禁用梯度并返回
        return alibi.detach()

class MHA(nn.Module):
    """
    注意力层
    """

    def __init__(self, layer_id: int, args: BirdMindConfig):
        super().__init__()
        self.mha_id = layer_id
        self.dim = args.embedding_dim
        self.n_head = args.num_heads
        self.qk_dim = args.qk_dim
        self.v_dim = args.v_dim
        self.use_flash_attention = args.use_flash_attention

        self.wq = nn.Linear(self.dim, self.qk_dim * self.n_head,False)
        self.wk = nn.Linear(self.dim, self.qk_dim * self.n_head,False)
        self.wv = nn.Linear(self.dim, self.v_dim * self.n_head,False)
        self.wo = nn.Linear(self.v_dim * self.n_head, self.dim,False)
    

    def forward(self, x: torch.Tensor, causal_mask, position_ids, pos_cis_bias, past_key_values:Cache, cache_position) -> torch.Tensor:
        batch_size, sequence_len, _ = x.size()

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # batch,seq_len,head,v_dim
        q = q.view(batch_size, sequence_len, self.n_head, self.qk_dim)
        k = k.view(batch_size, sequence_len, self.n_head, self.qk_dim)
        v = v.view(batch_size, sequence_len, self.n_head, self.v_dim)

        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # kv_cache实现
        if past_key_values is not None:
            k,v = past_key_values.update(k,v,self.mha_id)

        
        #         batch,seq_len,head,qk_dim
        score = torch.einsum('bhsk,bhSk->bhsS', q, k)
        score = score / self.qk_dim ** 0.5
        
        score += pos_cis_bias

        #  batch,head,seq_len,seq_len
        if causal_mask is not None:
            score += causal_mask[:,:,:sequence_len,:sequence_len]
        score = score.softmax(dim=-1).type_as(x)

        # v * score
        out = torch.einsum('bhsS,bhSv->bshv', score, v)
        # batch,seq_len,dim
        out = self.wo(out.flatten(2))
        return out, score


class Gate(torch.nn.Module):
    """
    门控结构
    """

    def __init__(self, args: BirdMindConfig):
        super().__init__()
        self.dim = args.embedding_dim
        self.top_k = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.score_func = args.score_func
        self.weight = nn.Parameter(torch.randn(args.n_expert_groups, args.embedding_dim))

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        bsz, seq_len, h = x.shape
        hidden_states = x.view(-1, h)

        logits = F.linear(hidden_states, self.weight, None)
        if self.score_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            scores = logits.sigmoid()

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)


        if self.top_k > 1 :
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        return topk_idx, topk_weight, logits


class MLP(torch.nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, out_dim,False)
        self.w2 = nn.Linear(out_dim, dim,False)
        self.w3 = nn.Linear(dim, out_dim,False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Expert(nn.Module):
    """
    专家
    """

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, out_dim,False)
        self.w2 = nn.Linear(out_dim, dim,False)
        self.w3 = nn.Linear(dim, out_dim,False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    混合专家模型
    """

    def __init__(self, args: BirdMindConfig):
        super().__init__()
        self.config = args
        self.dim = args.embedding_dim
        self.gate = Gate(args)
        self.n_expert_groups = args.n_expert_groups
        self.n_activated_experts=args.n_activated_experts
        self.experts = nn.ModuleList(
            [Expert(args.embedding_dim, args.moe_inter_dim) for _ in range(args.n_expert_groups)])
        self.shared_experts = MLP(args.embedding_dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        bsz, seq_len, hidden_dim = x.shape
        # 使用门控机制选择专家
        # (batch * sequence_length, n_experts) , (batch * sequence_length, n_experts)
        topk_idx, topk_weight, router_logits = self.gate(x)
        x = x.view(-1, x.shape[-1])
        topk_weight = topk_weight.to(x.dtype)

        final_hidden_states = torch.zeros(
            (bsz * seq_len, hidden_dim), dtype=x.dtype, device=x.device
        )
        expert_mask = torch.nn.functional.one_hot(topk_idx, num_classes=self.config.n_expert_groups).permute(2, 1, 0)

        for expert_idx in range(self.n_expert_groups):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `topk_weight` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * topk_weight[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(bsz, seq_len, hidden_dim)

        if self.n_expert_groups is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(identity)
        return final_hidden_states, router_logits




class FeedForward(nn.Module):
    def __init__(self, config: BirdMindConfig):
        super().__init__()
        hidden_dim = 4 * config.embedding_dim

        self.w1 = nn.Linear(config.embedding_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.embedding_dim, bias=False)
        self.w3 = nn.Linear(config.embedding_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """
    主层堆叠
    """

    def __init__(self, layer_id: int, args: BirdMindConfig):
        super().__init__()
        # attention
        self.block_id = layer_id
        self.attn = MHA(layer_id, args)
        self.feed_forward = FeedForward(args) if not args.use_moe else MoE(args)
        self.attn_norm = RMSNormLayer(args.embedding_dim)
        self.ffn_norm = RMSNormLayer(args.embedding_dim)

    def forward(self, x, causal_mask, position_ids, pos_cis_bias ,past_key_values, cache_position) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        # batch,seq_len,dim
        h_att, self_attn_weights = self.attn(self.attn_norm(x), causal_mask, position_ids, pos_cis_bias, past_key_values, cache_position)
        x = x + h_att
        # batch, seq_len, dim
        out = self.feed_forward(self.ffn_norm(x))
        if isinstance(out, tuple):
            out_states, router_logits = out
        else:
            router_logits = None

        x = x + out_states
        return x , router_logits , self_attn_weights


class RMSNormLayer(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts



class BirdMindModel(PreTrainedModel,GenerationMixin):
    config_class = BirdMindConfig
    """
    主要模型类
    """

    def __init__(self, args: BirdMindConfig):
        super().__init__(args)
        
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, args.pad_token_id)
        self.blocks = torch.nn.ModuleList()
        for i in range(args.block_size):
            self.blocks.append(Block(i, args))
        self.rms_norm_layer = RMSNormLayer(args.embedding_dim)
        self.linear = nn.Linear(args.embedding_dim, args.vocab_size, False)
        # print("初始化position embedding")
        # if self.training and torch.cuda.is_available():
        #     device = torch.device("cuda")
        # else:
        #     device = torch.device("cpu")

        # self.register_buffer("alibi",get_alibi_bias(args.num_heads,torch.arange(0,args.max_seq_len)).to(device),persistent=False)
        
        # print("结束初始化position embedding")
        # self.register_buffer("mask",torch.full((args.max_seq_len, args.max_seq_len), float("-inf"),device=args.device, requires_grad=False).triu_(1),persistent=False)
        self.config = args

        self.post_init()
    

    
    

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        """
        注意力矩阵调整
        """
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3Moe. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        config: BirdMindConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen3MoeConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
            )
            text_config = config.get_text_config()
            if getattr(text_config, "use_sliding_window", True) and text_config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=cache_position.device) <= (
                        cache_position.reshape(-1, 1) - text_config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Cache = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs) -> MoeCausalLMOutputWithPast:
        """
        主model
        :param start_pos:
        :param tokens:
        :return:
        """
        if past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # 位置偏置 1,num_head,seq_len,seq_len
        pos_cis = get_alibi_bias(self.config.num_heads, position_ids)

        output = inputs_embeds

        all_hidden_states = ()
        all_self_attns = ()
        all_router_logits = ()

        all_hidden_states +=(inputs_embeds,)
        for i,block in enumerate(self.blocks):
            output,router_logits,self_attn_weights, = block(output, causal_mask, position_ids, pos_cis ,past_key_values,cache_position)
            all_hidden_states += (output,)
            all_self_attns += (self_attn_weights,)
            all_router_logits += (router_logits,)

        # batch_size,seq_len,embedding_dim
        hidden_state = self.rms_norm_layer(output)
        
        # 切片保留
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # batch_size,seq_len,vocab_size
        logits = self.linear(hidden_state[:, slice_indices, :])
        
        
        
        loss = None
        aux_loss = None
        if labels is not None:
            aux_loss = load_balancing_loss_func(
                all_router_logits,
                self.config.n_expert_groups,
                self.config.n_activated_experts,
                attention_mask,
            )
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
            loss += self.config.router_aux_loss_coef * aux_loss.to(loss.device)
    
        return MoeCausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            aux_loss=aux_loss,
            past_key_values = past_key_values,
            hidden_states = all_hidden_states,
            attentions = all_self_attns,
            router_logits = all_router_logits,
        )
    


    
    
    # @staticmethod
    # def init_model(birdMindConfig: BirdMindConfig,load_path:str = "./model.pth" ):
    #     tokenizer = AutoTokenizer.from_pretrained('./birdmind_tokenizer')
        
    #     model = BirdMindModel(birdMindConfig)
    #     if os.path.exists(load_path):
    #         model.load_state_dict(torch.load(load_path))
    #     print(model)
    #     model.to(birdMindConfig.device)
        
    #     print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        
    #     return tokenizer, model


if __name__ == "__main__":
    pass




