import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.BirdMindModel import BirdMindConfig, BirdMindModel

from dataprocess.DistillR1DataSet import DistillR1Dataset
from dataprocess.DistillR1NoThinkDataSet import DistillR1NoThinkDataSet
from dataprocess.DistillThinkDataSet import DistillThinkDataSet
from contextlib import nullcontext

"""
训练
"""


def train(model: BirdMindModel, train_loader: DataLoader, args: BirdMindConfig, epoch_num: int = 2, accmulation:int = 12):
    model.train()
    ctx = torch.amp.autocast('cuda') if args.device == "cuda" else torch.amp.autocast('cpu')
    scaler = torch.amp.GradScaler('cuda') if args.device == "cuda" else torch.amp.GradScaler('cpu')
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    # 思考标签占位符
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids

    for epoch in range(epoch_num):
        optimizer.zero_grad(set_to_none=True)
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

                # 在 sp_ids 对应的位置增加额外的惩罚
                sp_ids = torch.isin(y.view(-1),
                                torch.tensor(start_of_think_ids + end_of_think_ids).to(args.device))

                loss_mask = loss_mask.view(-1)
                loss_mask_sum = loss_mask.sum()
                loss_mask[sp_ids] = 2
                loss_mask = loss_mask.view(y.size())
                loss = (loss * loss_mask).sum() / loss_mask_sum

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
                optimizer.zero_grad(set_to_none=True)
                print("梯度更新")

            if (batch_idx + 1) % (10*accmulation) == 0:
                print(f'batch_idx[{batch_idx}] loss: {loss.item():.4f}')
                torch.save(model.state_dict(), "./sft_r1_model_10000_nomoe.pth")



if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    if torch.cuda.is_available():
        print("use cuda")
        print(torch.__version__)
        print(torch.version.cuda)
    else:
        print("use cpu")

    args = BirdMindConfig(device = device, vocab_size=10000,block_size=16, embedding_dim=512,train=True)
    tokenizer, model = BirdMindModel.init_model(args,"./sft_r1_model_10000_nomoe.pth")
    
    train_data = DistillR1Dataset("./dataset/distill_r1_110k.jsonl", tokenizer)
    # train_data = DistillR1NoThinkDataSet(tokenizer)
    # train_data = DistillThinkDataSet(tokenizer)

    batch_size = 2
    dataLoader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,num_workers=1)

    train(model, dataLoader, args, accmulation=120)
