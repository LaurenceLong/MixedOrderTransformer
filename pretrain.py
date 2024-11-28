import os

from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    AutoTokenizer
)

from config import model_name
from data.math_dataset import create_math_dataset, create_sft_dataset
from mixed_tokenizer import MixedTokenizer

# 初始化模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
origin_tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = MixedTokenizer(origin_tokenizer)

# 添加padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 更新模型的嵌入层
if tokenizer.resize_operated:
    model.resize_token_embeddings(tokenizer.vocab_size)

# 生成或加载数据集
data_dir = os.path.join(os.getcwd(), "data")
train_dataset_file = os.path.join(data_dir, "data_math.txt")
validation_dataset_file = os.path.join(data_dir, "data_math_valid.txt")

size = 10 ** 6
# train_dataset = create_math_dataset(
#     file_path=train_dataset_file,
#     tokenizer=tokenizer,
#     size=size,
# )
#
# eval_dataset = create_math_dataset(
#     file_path=validation_dataset_file,
#     tokenizer=tokenizer,
#     size=size // 4,
# )
train_dataset = create_sft_dataset(
    file_path=train_dataset_file,
    tokenizer=tokenizer,
    size=size,
)

eval_dataset = create_sft_dataset(
    file_path=validation_dataset_file,
    tokenizer=tokenizer,
    size=size // 4,
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./pythia_pretrain",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,  # 添加评估batch_size
    save_total_limit=1,
    fp16=True,
    # 添加以下参数
    logging_steps=100,
    eval_steps=500,
    warmup_steps=500,
    weight_decay=0.01,
    gradient_accumulation_steps=4,  # 梯度累积
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()
