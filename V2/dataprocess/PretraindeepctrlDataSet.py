import json

from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer,PreTrainedTokenizer
from json import JSONDecodeError


class PretraindeepctrlDataSet(Dataset):
    def __init__(self, jsonl_path, tokenizer:PreTrainedTokenizer, max_length=2048):
        super().__init__()
        self.tokenizer:PreTrainedTokenizer = tokenizer
        self.max_length = max_length
        self.jsonl_path = jsonl_path

        # line_count = self.count_lines(jsonl_path)
        # print("总行数:",line_count)
        # self.samples = self.load_data(self.jsonl_path, line_count - int(line_count/3))
        # print("加载行数:",self.total_len)
        self.samples = self.test_load_data(self.jsonl_path,10)

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
    
    def test_load_data(self, jsonl_path,line_count):
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for _ in range(line_count):
                line = f.readline()
                data = json.loads(line.strip())
                samples.append(data)
        
        return samples    

    def __len__(self):
        return len(self.samples)

    def count_lines(self,file_path):
        count = 0
        with open(file_path, 'r',encoding='utf-8') as file:
            for line in file:
                count += 1
        return count
    

          

    def __getitem__(self, index):
        sample = self.samples[index]
        instruction = sample.get('instruction')
        input = sample.get('input')
        output = sample.get('output')
        history = sample.get('history')
        hitstory_text = ""
        for item_list in history:
            for item in item_list:
                hitstory_text +=item
        text = hitstory_text + instruction + input + output
        inputs = self.tokenizer(text, return_tensors="pt",padding='max_length', padding_side='left',truncation=True, max_length=2048)

        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]

        return input_ids, input_ids, attention_mask

if __name__=="__main__":
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained("./V2/models", trust_remote_code=True)
    PretraindeepctrlDataSet("./V2/dataset/sft_data_zh.jsonl",tokenizer)
    