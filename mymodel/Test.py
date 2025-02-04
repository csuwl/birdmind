import torch

if __name__ == "__main__":
    mask = torch.full((5, 5), float("-inf")).triu_(1)
    print(mask)
    print(mask.unsqueeze(1))

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
