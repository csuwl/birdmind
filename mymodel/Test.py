import torch


def get_position_embedding(x: torch.Tensor, head_num: int) -> torch.Tensor:
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
                    position[head, i, j] = torch.tensor(- (i - j) * 2 ** (-(head + 1)))
        return position


if __name__ == "__main__":


    a = torch.tensor([1,2,3])
    b = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print(b+a)

    
