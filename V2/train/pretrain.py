import torch
from transformers import AutoTokenizer,PreTrainedTokenizer,AutoModelForCausalLM,GenerationMixin,PreTrainedModel,AutoModel
import torch
import torch.nn as nn
import math
import os
import sys
from torch.utils.data import DataLoader
sys.path.append('./')
from V2.dataprocess.PretraindeepctrlDataSet import PretraindeepctrlDataSet
from transformers import AutoConfig, AutoModel
sys.path.append('./V2/models')
from BirdMindModel import BirdMindModel
from BirdMindModel import BirdMindConfig
from tqdm import tqdm


def train(model:GenerationMixin , train_loader: DataLoader, epoch_num: int = 2, accmulation:int = 300):
    model.train()
    model.to('cuda')
    ctx = torch.amp.autocast('cuda') 
    scaler = torch.amp.GradScaler('cuda') 
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(epoch_num):
        optimizer.zero_grad(set_to_none=True)
        print("epoch:", epoch)
        for batch_idx, data in tqdm(enumerate(train_loader),total = len(train_loader)):
            
            x, y, loss_mask = data
            x = x.to('cuda')
            y = y.to('cuda')
            loss_mask = loss_mask.to('cuda')
            with ctx:
                res = model.generate(inputs = x,labels = y,attention_mask = loss_mask)
                logits, loss = res.logits, res.loss
                # 梯度累计
                loss = loss / accmulation

            scaler.scale(loss).backward()


            if (batch_idx + 1) % accmulation == 0:
                token_id_out = logits.argmax(2)
                print(tokenizer.decode(token_id_out[0].tolist()))
                print("总loss:",loss)

                scaler.unscale_(optimizer)
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)  # 替代 optimizer.step()
                scaler.update()  # 调整缩放因子，准备下一轮
                optimizer.zero_grad(set_to_none=True)
                print("梯度更新")

            if (batch_idx+1) % (5*accmulation) == 0:
                print(f'batch_idx[{batch_idx}] loss: {loss.item():.4f}')
                model.save("./models")
            # if batch_idx % (10*accmulation) == 0:
    model.save("./models")

if __name__ == "__main__":
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained("./V2/models", trust_remote_code=True,padding_side='left')

    config = BirdMindConfig.from_pretrained("./V2/models")
    model = BirdMindModel.from_pretrained("./V2/models",config=config)

    dataset = PretraindeepctrlDataSet(jsonl_path = './V2/dataset/sft_data_zh.jsonl',tokenizer = tokenizer)
    
    batch_size = 2
    dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=1)

    train(model, dataLoader)