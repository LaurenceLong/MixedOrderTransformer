import random
import json

# operations = ['+', '-', '*', '/']
operations = ['+', '-']
max_num = 100000


def generate_arithmetic_problem():
    operation = random.choice(operations)

    if operation in ['+', '-', '*']:
        num1 = random.randint(0, max_num)
        num2 = random.randint(0, max_num)
    else:  # 除法
        num2 = random.randint(1, max_num)  # 限制除数范围，避免结果过于复杂
        num1 = num2 * random.randint(1, max_num)  # 确保能整除

    question = f"{num1} {operation} {num2} = "

    if operation == '+':
        answer = num1 + num2
    elif operation == '-':
        answer = num1 - num2
    elif operation == '*':
        answer = num1 * num2
    else:
        answer = num1 / num2  # 整数除法

    text = f"{question}{answer}"
    return {"prompt": question, "answer": str(answer)}, text


def generate_training_data(num_problems):
    sft = []
    texts = []
    for _ in range(num_problems):
        prompt_answer, text = generate_arithmetic_problem()
        sft.append(json.dumps(prompt_answer, ensure_ascii=False) + "\n")
        texts.append(text + "\n")
    return sft, texts


def generate_training_text(file_path, count):
    # 生成问题的训练数据
    json_data, text_data = generate_training_data(count)

    # 将数据保存到文件
    with open(f"{file_path}.jsonl", "w", encoding="utf-8") as f1, open(file_path, "w",
                                                                       encoding="utf-8") as f2:
        f1.writelines(json_data)
        f2.writelines(text_data)
    print(f"训练数据已生成并保存到 {file_path}.jsonl & {file_path}")
    return json_data, text_data
