import os
import json
from datasets import load_dataset, load_from_disk, get_dataset_config_names

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


DATASET_NAME = "TIGER-Lab/MMLU-Pro"
JSONL_PATH = "mmlu_pro_test.jsonl"

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

dataset = load_dataset(DATASET_NAME)["test"]
all_tasks = []
for item in dataset:
    num_choices = len(item["options"])
    choices_letter = [ALPHABET[i] for i in range(num_choices)]
    task = {
        "question": item["question"],
        "answer": item["answer"],
        "subject": item["category"],
    }
    for letter, choice in zip(choices_letter, item["options"]):
        task[letter] = choice
    all_tasks.append(task)

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for task in all_tasks:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")
