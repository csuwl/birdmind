import torch
from transformers import AutoTokenizer,PreTrainedTokenizer,AutoModelForCausalLM,GenerationMixin,PreTrainedModel
import torch
import torch.nn as nn
import math
import os
import sys
sys.path.append('./V2/models')
from BirdMindModel import BirdMindModel
from BirdMindModel import BirdMindConfig


def get_alibi_bias(num_heads: int, position_ids: torch.Tensor) -> torch.Tensor:
        """
        position_ids: [batch_size, seq_len]
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

if __name__ == "__main__":
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained("./V2/models", trust_remote_code=True)
    # text = tokenizer.apply_chat_template([{'role': 'user', 'content': '你好吗，你能干什么？'}, {'role': 'assistant', 'content': '我好的，你好吗'}], tokenize=False,add_generation_prompt=False)
    # print(text)
    print(tokenizer.pad)
    inputs = tokenizer(["你好,大哥"], return_tensors="pt",padding='max_length', truncation=True, max_length=2048)
    print(inputs)
