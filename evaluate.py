import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.math_dataset import create_sft_dataset
from mixed_tokenizer import MixedTokenizer

# 加载训练后的模型和 tokenizer
output_dir = "pythia_pretrain_70m/checkpoint-3750"  # 在训练时的 output_dir 中保存的模型
model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = MixedTokenizer(AutoTokenizer.from_pretrained(output_dir))

# 准备验证集
data_dir = os.path.join(os.getcwd(), "data")
validation_dataset_file = os.path.join(data_dir, "data_math_valid.txt")

# `create_sft_dataset` 用于加载验证集
eval_dataset = create_sft_dataset(
    file_path=validation_dataset_file,
    tokenizer=tokenizer,
    size=10 ** 4 // 4,  # 数据集大小
)

# 创建 DataLoader 以支持批量评估
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)


# 定义评估指标：Perplexity
def calculate_perplexity(eval_loss):
    """
    根据交叉熵损失计算困惑度 Perplexity。
    Perplexity = exp(eval_loss)
    """
    return torch.exp(torch.tensor(eval_loss))


def shift_labels(input_ids, pad_token_id):
    """
    将 labels 右移一位，同时在开头填充忽略标记 (-100)。
    Args:
        input_ids: 模型的输入序列。
        pad_token_id: 用于 padding 的 token ID。
    Returns:
        labels: 右移后的标签序列。
    """
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]  # 将 input_ids 的 [1:] 部分复制到 labels 的 [: -1] 部分
    labels[:, -1] = pad_token_id  # 最后一个位置填充 pad_token_id 或 -100（忽略标记）
    return labels


# 定义评估函数
def evaluate_model(model, tokenizer, eval_dataloader, max_length=128, device="cuda"):
    """
    使用验证集评估模型性能，包括困惑度和生成结果的质量。
    """
    model.eval()
    model.to(device)
    eval_loss = 0
    num_batches = 0
    predictions = []
    references = []

    # 遍历验证集 DataLoader
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # 将 batch 中的张量列表转换为二维张量
        input_ids = torch.stack(batch["input_ids"]).to(device)  # 将张量列表堆叠成二维张量
        labels = torch.stack(batch["labels"]).to(device)        # 同样处理 labels

        with torch.no_grad():
            # 计算损失
            outputs = model(input_ids=input_ids, labels=labels)
            eval_loss += outputs.loss.item()
            num_batches += 1

            # 生成文本
            generated_outputs = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id
            )

            # 解码生成结果和参考答案
            for gen_output, label in zip(generated_outputs, labels):
                # 生成结果
                predictions.append(
                    tokenizer.decode(gen_output).strip()
                )
                # 参考答案
                references.append(
                    tokenizer.decode(label[label != -100]).strip()
                )

    # 计算总评估损失
    eval_loss = eval_loss / num_batches
    perplexity = calculate_perplexity(eval_loss)

    return eval_loss, perplexity, predictions, references


# 运行评估
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_loss, perplexity, predictions, references = evaluate_model(
    model, tokenizer, eval_dataloader, device=device
)

# 打印评估结果
print(f"Evaluation Loss: {eval_loss}")
print(f"Perplexity: {perplexity}")

# 对比生成结果与参考答案
print("\nSample Predictions:")
for pred, ref in zip(predictions[:5], references[:5]):
    print(f"Predicted: {pred}")
    print(f"Reference: {ref}\n")
