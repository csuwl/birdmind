import json

from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer


class Pretrain2048Dataset(Dataset):
    def __init__(self, jsonl_path, tokenizer:AutoTokenizer, max_length=2048):
        super().__init__()
        self.tokenizer:AutoTokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<s>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = ''
        for i, turn in enumerate(conversations):
            messages += turn['content']
        return messages
          

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
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