import torch
from torch.utils.data import DataLoader

from mymodel.Model import ModelArgs, Model
import tiktoken

from mymodel.MyTrainData import MyTrainData

if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    batch_size = 4
    seq_len = 20
    vocab_size = tokenizer.n_vocab

    train_data = MyTrainData(seq_len, tokenizer)
    dataLoader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    args = ModelArgs(vocab_size=vocab_size, embedding_dim=64)
    model = Model(args)
    print(model)
    total_loss = 0

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for epoch in range(50):
        print("epoch:", epoch)
        for batch_idx, data in enumerate(dataLoader):
            x, y = data
            out = model.forward(x, 0)
            out = out.view(batch_size * seq_len, vocab_size)
            y = y.view(batch_size * seq_len)
            loss = torch.nn.functional.cross_entropy(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f'batch_idx[{batch_idx}] loss: {loss.item():.4f}')
            if batch_idx % 100 == 0:
                torch.save(model.state_dict(), "./model.pth")
