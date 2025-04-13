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
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        messages.append({"role": "system", "content": "你是一个人工智能助手birdmind，被设计来回答用户的问题。回答问题前先对问题进行思考，思考内容用<think></think>标签进行包裹，然后再给出答案，答案用<answer></answer>包裹。"})
        messages.append({"role": "user", "content": conversations['input']})
        # messages.append({"role": "assistant", "content": "<think>\n"+conversations['reasoning_content']+"\n</think>\n"+"<answer>\n"+conversations['content']+"\n</answer>"})
        # for i, turn in enumerate(conversations):
        #     role = 'user' if i % 2 == 0 else 'assistant'
        #     messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
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
        return {'prompt':self._create_chat_prompt(sample),'reason':sample['reasoning_content'],'answer':sample['content']}
       