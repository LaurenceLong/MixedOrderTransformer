import os

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer

from data.math_dataset import create_math_dataset
from mixed_tokenizer import MixedTokenizer  # 引入你设计的tokenizer

# 初始化模型和 tokenizer
model_name = "EleutherAI/pythia-70m"  # 选择 Pythia 预训练模型
model = AutoModelForCausalLM.from_pretrained(model_name)

origin_tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = MixedTokenizer(origin_tokenizer)

# 更新模型的嵌入层
if tokenizer.resize_operated:
    model.resize_token_embeddings(tokenizer.vocab_size)

# 生成或加载数据集
data_dir = os.path.join(os.getcwd(), "data")
train_dataset_file = os.path.join(data_dir, "data_math.txt")
validation_dataset_file = os.path.join(data_dir, "data_math_valid.txt")
train_dataset = create_math_dataset(
    file_path=train_dataset_file,
    tokenizer=tokenizer,
    size=10**4,  # 数据集大小
    reverse_ratio=0.25,  # 逆序比例
    mean=4,  # 逆序长度均值
    std=1.0,  # 逆序长度标准差
    min_len=2  # 逆序最小长度
)
eval_dataset = create_math_dataset(
    file_path=validation_dataset_file,
    tokenizer=tokenizer,
    size=10**4//4,  # 数据集大小
    reverse_ratio=0.25,  # 逆序比例
    mean=4,  # 逆序长度均值
    std=1.0,  # 逆序长度标准差
    min_len=2  # 逆序最小长度
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./pythia_pretrain",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_total_limit=1,
    fp16=True,  # 使用混合精度
)

# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,  # 使用自定义 tokenizer
)

# 开始训练
trainer.train()
