import os

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer

from config import model_name
from data.math_dataset import create_sft_dataset
from mixed_tokenizer import MixedTokenizer  # 引入你设计的tokenizer

model = AutoModelForCausalLM.from_pretrained("./pythia_pretrain")

origin_tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer = MixedTokenizer(origin_tokenizer)

data_dir = os.path.join(os.getcwd(), "data")
sft_dataset_file = os.path.join(data_dir, "data_math_sft.txt")

# 生成微调数据
sft_dataset = create_sft_dataset(sft_dataset_file, tokenizer, 5000)

# 更新训练参数
sft_training_args = TrainingArguments(
    output_dir="./pythia_sft",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_total_limit=1,
    fp16=True,
)

# 定义微调 Trainer
sft_trainer = Trainer(
    model=model,
    args=sft_training_args,
    train_dataset=sft_dataset,
    tokenizer=tokenizer,
)

# 开始微调
sft_trainer.train()
