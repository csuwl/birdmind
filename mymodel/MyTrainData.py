import random

import torch
from torch.utils.data import Dataset
from transformers.integrations import tiktoken


class MyTrainData(Dataset):
    def __init__(self, seq_len: int, tokenizer: tiktoken):
        super().__init__()
        self.seq_len = seq_len
        with open("../novel.txt", "rt") as file:
            text = file.read()

        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.length = len(self.tokens)

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        start_index = random.randint(0, self.length - self.seq_len - 2)
        sentence = self.tokens[start_index: (start_index + self.seq_len)]
        label = self.tokens[start_index + 1: (start_index + 1 + self.seq_len)]
        return sentence, label
