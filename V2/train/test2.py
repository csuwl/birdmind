import torch


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
    alibi = get_alibi_bias(8, torch.tensor([[0,1,2,3,4,5,6]]))
    print(alibi)
    print("------------------------------------------------")
    base = 2**(-8/(2-1))  # 优化计算：避免在循环中重复计算幂
    alibi_slop = torch.pow(base, torch.arange(2))
    
    alibi_temp = get_alibi_bias(2, torch.tensor([[0,1,2,3,4,5]]))
    add_bias = -alibi_slop.view(1, -1, 1, 1) *  torch.arange(5+1,0,-1).view(1,1,-1)
    print(add_bias)
    x = torch.cat([alibi_temp,add_bias], dim = -2)
    print(x)
    lie = torch.tensor([0]).expand(1,2,x.shape[-2],1)
    print(lie)
    x = torch.cat([x,lie],dim=-1)
    print(x)