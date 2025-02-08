import torch


def get_position_embedding(x: torch.Tensor, head_num: int) -> torch.Tensor:
    """
     x : last 2 dimension is same
    """

    seq_len = x.shape[-1]
    position = torch.empty(head_num, seq_len, seq_len, dtype=x.dtype, device=x.device)
    for head in range(head_num):
        for i in range(seq_len):
            for j in range(seq_len):
                if i < j:
                    continue
                position[head, i, j] = - (i - j) * 2 ** (-(head+1))
    return position


if __name__ == "__main__":

    mask = torch.full((5, 5), float("-inf")).triu_(1)
    print(mask)
    position = get_position_embedding(torch.randn(5, 5), 3)
    print(position)

    scores = torch.randn(3, 2, 5, 5)
    print(scores)
    scores += mask.unsqueeze(-1)

    print("scores: ", scores)
    top_k = 2
    indices = torch.topk(scores, top_k, dim=-1)[1]
    print("indices:", indices)
    weights = scores.gather(1, indices)
    print("weights:", weights)

    counts = torch.bincount(indices.flatten(), minlength=10).tolist()
    print("counts", counts)
    idx, top = torch.where(indices == torch.tensor(2))
    print("idx:", idx, " ", "top:", top)
    print(weights[[1]])
    print(weights[idx, top])
    print(weights[idx, top, None])
