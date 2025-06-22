import json

from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_dataset



class DistillR1NoThinkDataSet(Dataset):
    def __init__(self, tokenizer, max_length=2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("yys/OpenOrca-Chinese")['train']
        self.bos_id = tokenizer('<s>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

  
    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        if conversations.get("system_prompt") is not None and len(conversations.get("system_prompt"))>0:
            messages.append({"role": "system", "content": conversations['system_prompt']})
        messages.append({"role": "user", "content": conversations['question']})
        messages.append({"role": "assistant", "content":"<think>\n</think>\n"+conversations['response']})
        # for i, turn in enumerate(conversations):
        #     role = 'user' if i % 2 == 0 else 'assistant'
        #     messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enalble_thinking=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask