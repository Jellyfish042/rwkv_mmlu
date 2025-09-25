import os
import json
from datasets import load_dataset, load_from_disk, get_dataset_config_names

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


DATASET_NAME = "cais/mmlu"
JSONL_PATH = "mmlu_test.jsonl"

dataset = load_dataset(DATASET_NAME, "all")["test"]
all_tasks = []
for item in dataset:
    all_tasks.append(
        {
            "question": item["question"],
            "A": item["choices"][0],
            "B": item["choices"][1],
            "C": item["choices"][2],
            "D": item["choices"][3],
            "answer": 'ABCD'[item["answer"]],
            "subject": item["subject"],
        }
    )

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for task in all_tasks:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")
