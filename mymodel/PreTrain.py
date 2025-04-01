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
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        for batch_idx, data in enumerate(train_loader):
            
            x, y, loss_mask = data
            x = x.to(args.device)
            y = y.to(args.device)
            loss_mask = loss_mask.to(args.device)
            
            
            seq_len = x.shape[1]
            res = model.forward(x,0)
            out, aux_loss = res.logits, res.aux_loss
            token_id_out = out.argmax(2)

            print(tokenizer.decode(token_id_out[0].tolist()))

            out = out.view(batch_size * seq_len, args.vocab_size)
            y = y.view(batch_size * seq_len)
            loss = torch.nn.functional.cross_entropy(out, y)
            
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += aux_loss * 0.005
            print(loss)
            # 梯度累计
            loss = loss / 8
            loss.backward()


            if (batch_idx + 1) % 8 == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                print("梯度更新")

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
    dataLoader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,num_workers=1)

    train(batch_size, model, dataLoader, args)
