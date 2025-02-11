import json

import tiktoken
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer: tiktoken, max_length=512):
        super().__init__()
        self.tokenizer: tiktoken = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        text = sample['text']
        encoding = self.tokenizer.encode(text)
        length = len(encoding)
        if length > self.max_length:
            encoding = encoding[:self.max_length]
        elif length < self.max_length:
            encoding = encoding + [self.tokenizer.eot_token] *(self.max_length-length)
        # encoding = self.tokenizer(
        #     text,
        #     max_length=self.max_length,
        #     padding='max_length',
        #     truncation=True,
        #     return_tensors='pt'
        # )
        input_ids = torch.tensor(encoding).squeeze()
        loss_mask = torch.empty_like(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask
