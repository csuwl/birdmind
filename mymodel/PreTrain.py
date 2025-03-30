import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Model import ModelArgs, Model

from dataprocess.PretrainDataSet import PretrainDataset

"""
训练
"""


def train(batch_size:int ,model: Model, train_loader: DataLoader, args: ModelArgs, epoch_num: int = 2):
    model.train()
    model.to(args.device)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            x, y, loss_mask = data
            x.to(args.device)
            y.to(args.device)
            loss_mask.to(args.device)
            
            
            seq_len = x.shape[1]
<<<<<<< HEAD
            out, aux_loss = model.forward(x, 0)
            token_id_out = out.argmax(2)

            print(tokenizer.decode(token_id_out[0].tolist()))

=======
            
            res = model.forward(x,0)
            out, aux_loss = res.logits, res.aux_loss
            
>>>>>>> 19a770ae802e2e5f80bba5657cf702c9f9ae7138
            out = out.view(batch_size * seq_len, args.vocab_size)
            y = y.view(batch_size * seq_len)
            loss = torch.nn.functional.cross_entropy(out, y)
            
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            print('loss: ',loss)
            print('auxloss: ',aux_loss)
            loss += aux_loss * 0.01
            print('总loss: ',loss)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f'batch_idx[{batch_idx}] loss: {loss.item():.4f}')
            if batch_idx % 100 == 0:
                torch.save(model.state_dict(), "./model.pth")



if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("use cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("use cpu")

    args = ModelArgs(device = device, vocab_size=6400, embedding_dim=512)
    tokenizer, model = Model.init_model(args)
    
    train_data = PretrainDataset("../pretrain_hq.jsonl", tokenizer)

    batch_size = 32
    dataLoader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    train(batch_size, model, dataLoader, args)
