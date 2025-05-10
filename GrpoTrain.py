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
import time
import traceback


client = OpenAI(api_key=os.environ["deepseek_key"],base_url="https://api.deepseek.com")


# def rewardFunction(prompts, completions,**reward_kwargs):
#     print("prompts:",prompts)
#     print("completions:",completions)
#     while True:
#         try:
#             # print("**reward_kwargs:",reward_kwargs)
#             completion = client.chat.completions.create(model="deepseek-reasoner",
#                                                         temperature=0.2,
#                                                         messages=[{"role":"system","content":"你是一个奖励函数，擅长对AI回答打分，你要根据输入的问题和两个回答仔细判断，并对两个回答打分,分数在0-10之间。评分标准：回答中答案正确5分，有思考过程5分，既有思考过程答案又正确10分，其它情况的则看哪个回答好哪个分数就高。回答只需包含分数，分数之间用,分隔"},
#                                                         {"role":"user","content":"问题是:\n{prompts[0]} \n 回答1:\n{completions[0]} \n 回答2:\n{completions[1]}"}])
#             scores = completion.choices[0].message.content.split(",")
#             scores = [float(x) for x in scores]
#             print("scores:",scores)
#             return scores
#         except Exception as e:
#             print("error:",e)
#             time.sleep(5)
#             print("retry")
#             continue

model_name = "Qwen/Qwen3-1.7B"

# load the tokenizer and the model
reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
reward_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)        


def rewardFunction(prompts, completions,**reward_kwargs):
    print("prompts:",prompts)
    print("completions:",completions)
    while True:
        try:
           
            # prepare the model input
            messages = [{"role":"system","content":"你是一个奖励函数，擅长对AI回答打分，你要根据输入的问题和两个回答仔细判断，并对两个回答打分,分数在0-10之间。评分标准：回答中答案正确5分，有思考过程5分，既有思考过程答案又正确10分，其它情况的则看哪个回答好哪个分数就高。回答只需包含分数，分数之间用,分隔"},
                                                        {"role":"user","content":f"问题是:\n{prompts[0]} \n 回答1:\n{completions[0]} \n 回答2:\n{completions[1]}"}]
            text = reward_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = reward_tokenizer([text], return_tensors="pt").to(reward_model.device)

            # conduct text completion
            generated_ids = reward_model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

            # parsing thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = reward_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = reward_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            print("thinking content:", thinking_content)
            print("content:", content)
            
            scores = content.split(",")
            scores = [float(x) for x in scores]
            if len(scores) != 2:
                raise ValueError("scores length is not 2")
            print("scores:",scores)
            return scores
        except Exception as e:
            print("error:",e)
            time.sleep(5)
            print("retry")
            traceback.print_exc()
            continue



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
               save_safetensors=True,
               save_strategy='steps',
               per_device_train_batch_size=2,
               gradient_accumulation_steps=4,
               num_generations=2,
               max_prompt_length=1024,
               max_completion_length=2048,
               num_train_epochs=1,
               save_steps=2,
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