# 介绍
birdmind麻雀虽小，五脏俱全，模型总参数量1.4B,此项目完全从头开始编写，参考deepseek-v3，minimind项目。采用16层block，block包含transformer、moe、MHA层，alibi作为位置偏置。
预训练完毕，模型下载地址: https://huggingface.co/csuwl/birdmind


# 快速开始
下载模型后使用transformers直接加载运行。
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载本地模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    "./transformers_model/",
    trust_remote_code=True  # 允许执行远程代码（自定义模型需要）
)
tokenizer = AutoTokenizer.from_pretrained("./transformers_model/")

# 将模型转移到 GPU
model.to("cuda")

# 编码输入并生成文本
inputs = tokenizer(["你好，很高兴认识你"], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=30)

# 解码并打印结果
print(tokenizer.batch_decode(outputs)[0])
```
# 训练过程
首先使用公开数据进行预训练2个epoch,计划对其进行稍加sft,然后使用grpo策略进行强化学习。

# 数据集 
预训练数据minimind_dataset的sft_2048.jsonl公开数据集。
蒸馏数据distill_r1_110k.jsonl
grpo策略自我提升使用数据distill_r1_110k.jsonl

