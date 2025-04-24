import json

from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer
from json import JSONDecodeError


class PretraindeepctrlDataSet(Dataset):
    def __init__(self, jsonl_path, tokenizer:AutoTokenizer, max_length=2048):
        super().__init__()
        self.tokenizer:AutoTokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer('<s>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>', add_special_tokens=False).input_ids
        self.jsonl_path = jsonl_path

        line_count = self.count_lines(jsonl_path)
        print("总行数:",line_count)
        self.samples = self.load_data(self.jsonl_path, int(line_count/2))
        self.total_len = len(self.samples)
        print("加载行数:",self.total_len)

    def load_data(self, jsonl_path,line_count):
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for _ in range(line_count+1):
                next(f)
            lines = f.readlines()

        for line in lines:
            try:
                data = json.loads(line.strip())
            except JSONDecodeError as e:
                print("error:",line.strip())
                continue
            samples.append(data)
        
        return samples    

    def __len__(self):
        return self.total_len

    def count_lines(self,file_path):
        count = 0
        with open(file_path, 'r',encoding='utf-8') as file:
            for line in file:
                count += 1
        return count
    



    def _create_chat_prompt(self, sample):
        messages = ''
        for history_list in sample['history']:
            for temp_history in history_list:
                messages += temp_history   
        messages += sample['input']
        messages += sample['output']
        return messages
          

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample)
        encoding = self.tokenizer(prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')

        # 生成动态损失掩码
        # loss_mask = self._generate_loss_mask(input_ids)
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask

if __name__=="__main__":
    pass