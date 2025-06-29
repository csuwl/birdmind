from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.tokenization_utils import PreTrainedTokenizer
import torch
import torch.nn as nn
import math
import os
import sys
sys.path.append('./V2/models')
from BirdMindModel import BirdMindModel
from BirdMindModel import BirdMindConfig
from transformers import AutoConfig, AutoModel


if __name__ == "__main__":
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained("./V2/models", trust_remote_code=True,padding_side='left')

    config = BirdMindConfig(pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,)
    model = BirdMindModel(config)
    model.save_pretrained("./V2/models")
    config.save_pretrained("./V2/models")

    text = tokenizer.apply_chat_template([[{'role': 'user', 'content': '你好吗，你能干什么？'}, {'role': 'assistant', 'content': '我好的，你好吗'}],
    [{'role': 'system', 'content': 'xxxxxxxxxxxxxxsoandgaoeiwgw lafld;f dsl;f jdsl;afjiewoangdakfoewnagengiaojdig;jdfiog;iwergndjgl;djgaeowiagn;reagjeoing'}]], tokenize=False, add_generation_prompt=True)
    print(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, padding_side='left',truncation=True)
    print(inputs)
    out = model.generate(**inputs,max_length=100,tokenizer=tokenizer)
    print(tokenizer.batch_decode(out, skip_special_tokens=False))

    from transformers import AutoConfig, AutoModel

    # 注册配置
    AutoConfig.register("birdmind", BirdMindConfig)

    # 注册模型
    AutoModel.register(BirdMindConfig, BirdMindModel)
    
    model = AutoModel.from_pretrained("./V2/models")