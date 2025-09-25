import os
import json
from datasets import load_dataset, load_from_disk, get_dataset_config_names

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


DATASET_NAME = "ceval/ceval-exam"
JSONL_PATH = "ceval_exam_test.jsonl"


all_tasks = []
for subset_name in get_dataset_config_names(DATASET_NAME):
    print(f"Processing {subset_name}...")
    dataset = load_dataset(DATASET_NAME, subset_name)
    # dataset.save_to_disk(f"{ceval-exam}/{subset_name}")
    # dataset = load_from_disk(f"{ceval-exam}/{subset_name}")

    for item in dataset["test"]:
        all_tasks.append(
            {
                "question": item["question"],
                "A": item["A"],
                "B": item["B"],
                "C": item["C"],
                "D": item["D"],
                "answer": item["answer"],
                "subject": subset_name,
            }
        )

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for task in all_tasks:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")
