# 介绍
birdmind麻雀虽小，五脏俱全，模型总参数量1.4B,此项目完全从头开始编写，参考deepseek-v3，minimind项目。采用16层block，block包含transformer、moe、MHA层，alibi作为位置偏置。
预训练完毕，模型下载地址: https://huggingface.co/csuwl/birdmind

# 数据集 
预训练数据minimind_dataset的sft_2048.jsonl公开数据集。
