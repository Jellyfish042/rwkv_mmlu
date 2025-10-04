import os
import json
from datasets import load_dataset, load_from_disk, get_dataset_config_names

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


DATASET_NAME = "math-ai/aime25"
JSONL_PATH = "aime2025.jsonl"

dataset = load_dataset(DATASET_NAME, "default")["test"]
all_tasks = []
for item in dataset:
    question = item["problem"]
    answer = item["answer"]
    answer = str(int(answer))  # check if the answer is a integer
    all_tasks.append({"question": question, "answer": answer, "subject": "aime25"})

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for task in all_tasks:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")
