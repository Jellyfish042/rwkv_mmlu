import os
import json
from datasets import load_dataset, load_from_disk, get_dataset_config_names

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


DATASET_NAME = "HuggingFaceH4/MATH-500"
JSONL_PATH = "math500_test.jsonl"

dataset = load_dataset(DATASET_NAME, "default")["test"]
all_tasks = []
for item in dataset:
    question = item["problem"]
    answer = item["answer"]
    subject = item["subject"]
    all_tasks.append({"question": question, "answer": answer, "subject": subject})

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for task in all_tasks:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")
