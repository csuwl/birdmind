import torch
import torch.nn as nn
from Model import Model,ModelArgs
from transformers import AutoTokenizer

class RewardModel(nn.module):
    def __init__(self,baseModel:Model,**args):
        super().__init__(args)
        self.baseModel = baseModel
        self.rewardHead= nn.Linear(baseModel.linear.out_features,1)

    def forward(self,input_ids:torch.tensor):
        # batch_size,seq_len,embedding_dim
        hidden_states = self.baseModel.forward(input_ids).hidden_states
        # batch_size,1
        reward = self.rewardHead(hidden_states[:-1:])
        return reward
    
    @staticmethod
    def init_model(args: ModelArgs,sft_load_path:str = "./sft_model.pth",reward_load_path:str=None):
        tokenizer = AutoTokenizer.from_pretrained('./birdmind_tokenizer')
        
        model = Model.init_model(args,sft_load_path)
        rewarModel = RewardModel(model)
        if reward_load_path is not None:
            rewarModel.load_state_dict(torch.load(reward_load_path))
        print(rewarModel)
        rewarModel.to(args.device)
        
        print(f'LLM总参数量：{sum(p.numel() for p in rewarModel.parameters() if p.requires_grad) / 1e6:.3f} 百万')
        return tokenizer,rewarModel