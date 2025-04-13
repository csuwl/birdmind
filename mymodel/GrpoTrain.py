import torch
import dill
import os

os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoModel
from models.BirdMindModel import BirdMindConfig,BirdMindModel
from trl import GRPOConfig,GRPOTrainer
from dataprocess.GrpoDataSet import GrpoDataSet
from openai import OpenAI


client = OpenAI(api_key=os.environ["deepseek_key"],base_url="https://api.deepseek.com")
completion = client.chat.completions.create(model="deepseek-reasoner",
                                            messages=[{"role":"system","content":"你是一个奖励函数，擅长从用户给你的几个AI回答打分,回答只需包含分数，分数之间用,分隔"},
                                                  {"role":"user","content":"问题是1+1等于几? 回答1是:等于2 回答2是:<think>嗯，1+1应该等于2</think><answer>等于2</answer>"}])
print(completion.choices[0].message)

def rewardFunction(prompts, completions,**reward_kwargs):
    print("prompts:",prompts)
    print("completions:",completions)
    # print("**reward_kwargs:",reward_kwargs)
    completion = client.chat.completions.create(model="deepseek-reasoner",
                                                temperature=0.2,
                                                messages=[{"role":"system","content":"你是一个奖励函数，擅长从用户给你的几个AI回答打分,回答只需包含分数，分数之间用,分隔"},
                                                  {"role":"user","content":"问题是:{prompts[0]} 回答:{completions}"}])
    scores = completion.choices[0].message.content.split(",")
    scores = [float(x) for x in scores]
    print("scores:",scores)
    return scores
    




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
               logging_steps=50,
               bf16=True,
               per_device_train_batch_size=4,
               gradient_accumulation_steps=10,
               num_generations=2,
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