import torch
import dill
import os

os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModel
from models.BirdMindModel import BirdMindConfig,BirdMindModel




def get_position_embedding(seq_len: int, head_num: int, device) -> torch.Tensor:
        """
         x : last 2 dimension is same
         return position shape  (head, seq_len, seq_len)
        """

        position = torch.zeros(head_num, seq_len, seq_len, device=device,requires_grad=False)
        for head in range(head_num):
            for i in range(seq_len):
                for j in range(seq_len):
                    if i < j:
                        continue
                    position[head, i, j] = torch.tensor(- (i - j) * 2 ** (-(head + 1)),device=device,requires_grad=False)
        return position


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained('./birdmind_tokenizer')
    messages = []
    messages.append({"role": "user", "content": "你好撒"})
    messages.append({"role": "assistant", "content": "的撒饱和哈然后公司"})
    x= tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    print(x)

    
