import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from BirdMindModel import BirdMindConfig, BirdMindModel

from dataprocess.PretrainDataSet import PretrainDataset
from dataprocess.Pretrain2048DataSet import Pretrain2048Dataset
from contextlib import nullcontext

"""
训练
"""


def train(model: BirdMindModel, train_loader: DataLoader, args: BirdMindConfig, epoch_num: int = 2, accmulation:int = 150):
    model.train()
    ctx = torch.amp.autocast('cuda') if args.device == "cuda" else torch.amp.autocast('cpu')
    scaler = torch.amp.GradScaler('cuda') if args.device == "cuda" else torch.amp.GradScaler('cpu')
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        for batch_idx, data in enumerate(train_loader):
            
            x, y, loss_mask = data
            x = x.to(args.device)
            y = y.to(args.device)
            loss_mask = loss_mask.to(args.device)
            
            with ctx:
                res = model.forward(x)
                out, aux_loss = res.logits, res.aux_loss

                loss = loss_fct(out.view(-1, out.size(-1)), y.view(-1))
                loss = loss.view(y.size())

                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += aux_loss * 0.1
                # 梯度累计
                loss = loss / accmulation

            scaler.scale(loss).backward()


            if (batch_idx + 1) % accmulation == 0:
                token_id_out = out.argmax(2)
                print(tokenizer.decode(token_id_out[0].tolist()))
                print("auxloss:",aux_loss)
                print("总loss:",loss)

                scaler.unscale_(optimizer)
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)  # 替代 optimizer.step()
                scaler.update()  # 调整缩放因子，准备下一轮
                print("梯度更新")

            if (batch_idx+1) % (10*accmulation) == 0:
                print(f'batch_idx[{batch_idx}] loss: {loss.item():.4f}')
                torch.save(model.state_dict(), "./model_10000.pth")
            # if batch_idx % (10*accmulation) == 0:
    torch.save(model.state_dict(), "./model_10000.pth")



if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cpu")
    
    if torch.cuda.is_available():
        print("use cuda")
        print(torch.__version__)
        print(torch.version.cuda)
    else:
        print("use cpu")

    args = BirdMindConfig(device = device, vocab_size=10000, embedding_dim=512,block_size=16,train=True)
    tokenizer, model = BirdMindModel.init_model(args,"./model_10000.pth")
    
    # train_data = PretrainDataset("../pretrain_hq.jsonl", tokenizer)
    train_data = Pretrain2048Dataset("./dataset/sft_2048.jsonl",tokenizer)

    batch_size = 2
    dataLoader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,num_workers=1)

    train(model, dataLoader, args)
