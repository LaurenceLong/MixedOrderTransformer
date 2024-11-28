import json
import os
from typing import Dict

import torch
from datasets import Dataset

from data.gen_math4 import generate_training_text
from mixed_tokenizer import MixedTokenizer


def pad_sequence(sequence, max_length, pad_token_id):
    """
    对序列进行 padding，使其长度达到 max_length。
    """
    return sequence + [pad_token_id] * (max_length - len(sequence))


def create_math_dataset(
        file_path: str,
        tokenizer,
        size: int = 10 ** 5,
        reverse_ratio: float = 0.25,
        mean: float = 4,
        std: float = 1.0,
        min_len: int = 2,
        block_size: int = 512  # 每个训练样本的最大长度
) -> Dataset:
    """
    生成正序和逆序混合的数学任务数据集，使用连续文本流方式。

    Args:
        file_path: 数据集的存储路径
        tokenizer: 分词器
        size: 数据集大小
        reverse_ratio: 逆序比例
        mean: 逆序长度均值
        std: 逆序长度标准差
        min_len: 逆序最小长度
        block_size: 每个训练样本的长度

    Returns:
        Dataset: HuggingFace dataset对象
    """
    # 加载或生成原始文本数据
    if os.path.exists(file_path):
        print(f"加载已有数据集文件：{file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f.readlines()]
    else:
        print(f"生成新的数据集并保存到：{file_path}")
        sft, data = generate_training_text(file_path, size)

    # 将所有文本编码并连接成一个长序列
    all_tokens = []
    for text in data:
        encoded = tokenizer.mixed_order_encode(
            text, reverse_ratio=reverse_ratio, mean=mean, std=std, min_len=min_len
        )
        all_tokens.extend(encoded)
        # 可以在每个样本之间添加特殊分隔符
        all_tokens.append(tokenizer.eos_token_id)  # 如果需要的话

    # 将长序列切分成固定大小的块
    result_dataset = []
    for i in range(0, len(all_tokens) - block_size + 1, block_size):
        chunk = all_tokens[i:i + block_size]
        if len(chunk) == block_size:  # 只使用完整的块
            input_ids = torch.tensor(chunk, dtype=torch.long)
            result_dataset.append({
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": input_ids.clone()
            })

    return Dataset.from_list(result_dataset)


# 数据集：构造更复杂的数学任务数据
def create_sft_dataset(
        file_path,
        tokenizer: MixedTokenizer,
        size,
        block_size: int = 512  # 每个训练样本的最大长度
):
    """
    创建SFT数据集，使用连续文本流方式。

    Args:
        file_path: 数据集文件路径
        tokenizer: 分词器
        size: 数据集大小
        block_size: 每个训练样本的长度

    Returns:
        Dataset: HuggingFace dataset对象
    """
    # 加载或生成数据
    if os.path.exists(f"{file_path}.jsonl"):
        print(f"加载已有数据集文件：{file_path}.jsonl")
        with open(f"{file_path}.jsonl", "r", encoding="utf-8") as f:
            sft = [line.strip() for line in f.readlines()]
    else:
        print(f"生成新的数据集并保存到：{file_path}")
        sft, data = generate_training_text(file_path, size)

    # 将所有样本连接成一个长序列
    all_tokens = []
    for jsonl_str in sft:
        # 编码提示和回答
        jsonl = json.loads(jsonl_str)
        prompt_tokens = tokenizer.encode(jsonl["prompt"])
        answer_tokens = tokenizer.reverse_order_encode(jsonl["answer"])

        # 添加到总序列中
        all_tokens.extend(prompt_tokens)
        all_tokens.extend(answer_tokens)

        all_tokens.append(tokenizer.eos_token_id)

    # 将长序列切分成固定大小的块
    result_dataset = []
    for i in range(0, len(all_tokens) - block_size + 1, block_size):
        chunk = all_tokens[i:i + block_size]
        if len(chunk) == block_size:  # 只使用完整的块
            input_ids = torch.tensor(chunk, dtype=torch.long)
            result_dataset.append({
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": input_ids.clone()
            })

    return Dataset.from_list(result_dataset)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # 加载 Pythia 的原生 tokenizer
    origin_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tokenizer = MixedTokenizer(origin_tokenizer)

    # 数据集文件路径
    dataset_file = "math_dataset.txt"

    # 生成或加载数据集
    train_dataset = create_math_dataset(
        file_path=dataset_file,
        tokenizer=tokenizer,
        size=10000,  # 数据集大小
        reverse_ratio=0.25,  # 逆序比例
        mean=4,  # 逆序长度均值
        std=1.0,  # 逆序长度标准差
        min_len=2  # 逆序最小长度
    )

    # 查看数据集样本
    print(train_dataset[0])
