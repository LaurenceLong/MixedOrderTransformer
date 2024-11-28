import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import model_name
from mixed_tokenizer import load_tokenizer

# 加载训练后的模型和 tokenizer
# output_dir = model_name  # 在训练时的 output_dir 中保存的模型
output_dir = "./pythia_pretrain/checkpoint-2325"  # 在训练时的 output_dir 中保存的模型
model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = load_tokenizer(output_dir)


def generate_text(input_text):
    input_ids = tokenizer.encode(input_text)

    # 将 input_ids 转换为 PyTorch 张量，并添加批量维度
    input_ids = torch.tensor([input_ids], dtype=torch.long)  # 添加 batch 维度，形状为 [1, sequence_length]

    # 设置生成参数
    max_length = 128
    generated_outputs = model.generate(
        input_ids=input_ids,  # 现在是 PyTorch 张量
        max_length=max_length,
        pad_token_id=0
    )

    # 解码生成的输出
    print("generated_outputs:", generated_outputs[0])
    decoded_outputs = tokenizer.decode(generated_outputs[0])
    print("result_text:", decoded_outputs)


with open("/home/laurence/work/MixedOrderTransformer/data/data_math.txt.jsonl") as fd:
    data = [json.loads(line) for line in fd.readlines()]

    for i in range(10):
        generate_text(data[i]["prompt"])
