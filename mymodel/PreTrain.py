import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Model import ModelArgs, Model

from dataprocess.PretrainDataSet import PretrainDataset

"""
训练
"""


def train(model: Model, train_loader: DataLoader, args: ModelArgs):
    model.train()
    torch.set_default_dtype(torch.float16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for epoch in range(50):
        print("epoch:", epoch)
        for batch_idx, data in enumerate(train_loader):
            x, y, loss_mask = data
            seq_len = x.shape[1]
            out, aux_loss = model.forward(x, 0)
            out = out.view(batch_size * seq_len, vocab_size)
            y = y.view(batch_size * seq_len,vocab_size)
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


    tokenizer = AutoTokenizer.from_pretrained('./minimind_tokenizer')

    batch_size = 128
    vocab_size = 6400
    max_seq_len = 512
    embedding_dim = 512

    # train_data = MyTrainData(seq_len, tokenizer)
    train_data = PretrainDataset("../pretrain_hq.jsonl", tokenizer)

    dataLoader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    args = ModelArgs(vocab_size=vocab_size, embedding_dim=embedding_dim)
    model = Model(args)
    if os.path.exists("./model.pth"):
        model.load_state_dict(torch.load("./model.pth"))
    print(model)

    train(model, dataLoader, args)
