from transformers import AutoTokenizer,PreTrainedTokenizer
import torch
import torch.nn as nn
import math


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
    print(f"alibi: {alibi}")
    
    # 5. 应用因果掩码 (仅保留下三角部分)
    # 创建因果掩码 [seq_len, seq_len]
    seq_len = position_ids.size(1)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=position_ids.device))
    alibi = alibi * causal_mask
    
    # 6. 禁用梯度并返回
    return alibi.detach()



if __name__ == "__main__":
    print("max: ",torch.max(torch.tensor([[0,5,6], [2,4,8]])))
    position_alibi = get_alibi_bias(3,torch.tensor([[0,5,6], [2,4,8]]))
    print(position_alibi)
    all_hidden_states = ()
    all_hidden_states += (torch.tensor([[0,5,6], [2,4,8]]),)
    all_hidden_states += (torch.tensor([[0,5,6], [2,4,8]]),)
    print(all_hidden_states)
    print(torch.nn.functional.one_hot(torch.tensor([[0, 1], [2, 5]]), num_classes=10))