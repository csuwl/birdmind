import json

from torch.utils.data import Dataset, DataLoader
import torch


class GrpoDataSet(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
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
                if line_num <700:
                    continue
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        messages.append({"role": "user", "content": conversations['input']})
       
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )



    def __getitem__(self, index):
        sample = self.samples[index]
        return {'prompt':self._create_chat_prompt(sample),'reason':sample['reasoning_content'],'answer':sample['content']}
       