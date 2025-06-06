import torch
import dill
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModel
from models.BirdMindModel import BirdMindConfig,BirdMindModel



if __name__ == "__main__":
    
    birdMindConfig = BirdMindConfig(vocab_size=10000, embedding_dim=512,block_size=16)
    tokenizer, model = BirdMindModel.init_model(birdMindConfig,'./sft_r1_model_10000_nomoe.pth')

    BirdMindConfig.register_for_auto_class()
    BirdMindModel.register_for_auto_class("AutoModelForCausalLM")

    birdMindConfig.save_pretrained("./transformers_model/")
    model.save_pretrained("./transformers_model/")
    tokenizer.save_pretrained("./transformers_model/")

    