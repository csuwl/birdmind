import torch
import dill
import os

os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModel
from models.BirdMindModel import BirdMindConfig,BirdMindModel
from trl import GRPOConfig,GRPOTrainer
from dataprocess.GrpoDataSet import GrpoDataSet


def rewardFunction(prompts, completions,**reward_kwargs):
    print("prompts:",prompts)
    print("completions:",completions)
    print("**reward_kwargs:",reward_kwargs)
    return 0.0
    




if __name__=="__main__":
    model = AutoModelForCausalLM.from_pretrained("./transformers_model/",trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("./transformers_model/")
    model_inputs = tokenizer(["你好，很高兴认识你"], return_tensors="pt").to("cuda")
    model.train()
    model.to("cuda")

    generated_ids = model.generate(**model_inputs, max_length=30)
    x = tokenizer.batch_decode(generated_ids)[0]
    print(x)
    
    grpoDataSet = GrpoDataSet("./dataset/distill_r1_110k.jsonl",tokenizer)

    grpoConfig = GRPOConfig(learning_rate=5e-6,
               adam_beta1=0.9,
               adam_beta2=0.99,
               weight_decay=0.1,
               warmup_ratio=0.1,
               lr_scheduler_type='cosine',
               logging_steps=1,
               bf16=True,
               per_device_train_batch_size=4,
               gradient_accumulation_steps=10,
               num_generations=4,
               max_prompt_length=1024,
               max_completion_length=2048,
               num_train_epochs=1,
               save_steps=100,
               max_grad_norm=0.1,
               report_to='tensorboard',
               overwrite_output_dir=True,
               output_dir='./output/model_grpo')
    
    grpoTrainer = GRPOTrainer(
        model = model,
        args = grpoConfig,
        processing_class = tokenizer,
        train_dataset = grpoDataSet,
        reward_funcs=[rewardFunction]
    )

    grpoTrainer.train()