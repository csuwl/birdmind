import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Model import ModelArgs, Model

from dataprocess.PretrainDataSet import PretrainDataset

"""
训练
"""


def train(batch_size:int ,model: Model, train_loader: DataLoader, args: ModelArgs, epoch_num: int = 50):
    model.train()
    torch.set_default_dtype(torch.float16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        for batch_idx, data in enumerate(train_loader):
            x, y, loss_mask = data
            seq_len = x.shape[1]
            out, aux_loss = model.forward(x, 0)
            out = out.view(batch_size * seq_len, args.vocab_size)
            y = y.view(batch_size * seq_len)
            loss = torch.nn.functional.cross_entropy(out, y)

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += aux_loss * 0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f'batch_idx[{batch_idx}] loss: {loss.item():.4f}')
            if batch_idx % 100 == 0:
                torch.save(model.state_dict(), "./model.pth")



if __name__ == '__main__':
    
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("use cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("use cpu")

    args = ModelArgs(vocab_size=6400, embedding_dim=512)
    tokenizer, model = Model.init_model(args)
    
    train_data = PretrainDataset("../pretrain_hq.jsonl", tokenizer)

    batch_size = 32
    dataLoader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    train(batch_size, model, dataLoader, args)
