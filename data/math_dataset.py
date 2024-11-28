import json
import os

from datasets import Dataset

from data.gen_math4 import generate_training_text
from mixed_tokenizer import MixedTokenizer


def pad_sequence(sequence, max_length, pad_token_id):
    """
    对序列进行 padding，使其长度达到 max_length。
    """
    return sequence + [pad_token_id] * (max_length - len(sequence))


def create_math_dataset(file_path, tokenizer, size=10 ** 5, reverse_ratio=0.25, mean=4, std=1.0, min_len=2):
    """
    生成正序和逆序混合的数学任务数据集，并保存到本地文件。
    如果文件已存在，则直接从文件加载数据。

    :param file_path: 数据集的存储路径
    :param tokenizer: 分词器（使用 MixedTokenizer）
    :param size: 数据集大小
    :param reverse_ratio: 逆序比例
    :param mean: 逆序长度均值
    :param std: 逆序长度标准差
    :param min_len: 逆序最小长度
    :return: Dataset 对象
    """
    if os.path.exists(file_path):
        print(f"加载已有数据集文件：{file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f.readlines()]
    else:
        print(f"生成新的数据集并保存到：{file_path}")
        sft, data = generate_training_text(file_path, size)

    # 将数据编码成 Dataset 格式
    max_length = 0
    dataset = []
    for text in data:
        encoded = tokenizer.mixed_order_encode(
            text, reverse_ratio=reverse_ratio, add_special_tokens=True, mean=mean, std=std, min_len=min_len
        )
        max_length = max(max_length, len(encoded))
        dataset.append(encoded)

    # 对所有样本进行 padding
    pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    padded_dataset = [
        {
            "input_ids": pad_sequence(encoded, max_length, pad_token_id),
            "labels": pad_sequence(encoded, max_length, pad_token_id)
        }
        for encoded in dataset
    ]

    return Dataset.from_list(padded_dataset)


# 数据集：构造更复杂的数学任务数据
def create_sft_dataset(file_path, tokenizer: MixedTokenizer, size):
    if os.path.exists(f"{file_path}.jsonl"):
        print(f"加载已有数据集文件：{file_path}.jsonl")
        with open(f"{file_path}.jsonl", "r", encoding="utf-8") as f:
            sft = [json.loads(line.strip()) for line in f.readlines()]
    else:
        print(f"生成新的数据集并保存到：{file_path}")
        sft, data = generate_training_text(file_path, size)

    max_length = 0
    dataset = []
    for jsonl in sft:
        encoded = tokenizer.encode(jsonl["prompt"]) + tokenizer.reverse_order_encode(jsonl["answer"])
        max_length = max(max_length, len(encoded))
        dataset.append(encoded)
    # 对所有样本进行 padding
    pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    padded_dataset = [
        {
            "input_ids": pad_sequence(encoded, max_length, pad_token_id),
            "labels": pad_sequence(encoded, max_length, pad_token_id)
        }
        for encoded in dataset
    ]

    return Dataset.from_list(padded_dataset)


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
