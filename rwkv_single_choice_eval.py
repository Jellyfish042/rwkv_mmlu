########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os
import json
import datetime

import torch
from torch.nn import functional as F
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

########################################################################################################
# MODEL
MODEL_PATH = "../models/RWKV-x070-World-2.9B-v3-20250211-ctx4096"

os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

model = RWKV(model=MODEL_PATH, strategy="cuda fp16")
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
tokenizer = pipeline.tokenizer

########################################################################################################
# PROMPT TEMPLATE
# English (MMLU MMLU-Pro etc.)
TEMPLATE = """User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
<CHOICES>

Assistant: The answer is"""

# for Chinese benchmarks (ceval-exam etc.)
# TEMPLATE = """User: <Q>
# <CHOICES>

# Assistant: 正确答案是"""

TARGET_TOKEN_FORMAT = " <LETTER>"  # for example, "<LETTER>" -> "A", " <LETTER>" -> " A"

########################################################################################################
# DATASET
# format example: {"question": "xxx", "A": "xxx", "B": "xxx", "C": "xxx", "D": "xxx", "answer": "A", "subject": "xxx"}
DATASET_PATH = "prepare_data/mmlu_pro_test.jsonl"
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

########################################################################################################
# RUN EVALUATION
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
max_choices = max(len(set(sample.keys()) & set(ALPHABET)) for sample in dataset)
print(f"The maximum number of choices is {max_choices} (A - {ALPHABET[max_choices - 1]})")

correct = 0
total = 0
pbar = tqdm(total=len(dataset))

choices_token = [tokenizer.encode(TARGET_TOKEN_FORMAT.replace("<LETTER>", x)) for x in ALPHABET[:max_choices]]
assert all([len(x) == 1 for x in choices_token])
choices_token = [x[0] for x in choices_token]
print(f"Choices token: {choices_token}")

score_by_subject = {}
for idx, sample in enumerate(dataset):
    question = sample["question"]
    subject = sample["subject"]
    gt = ALPHABET.index(sample["answer"])

    num_choices = len(set(sample.keys()) & set(ALPHABET))
    choices_str = "\n".join([f"{ALPHABET[i]}. {sample[ALPHABET[i]]}" for i in range(num_choices)])
    all_prefix = TEMPLATE.replace("<SUBJECT>", subject.replace("_", " ")).replace("<Q>", question).replace("<CHOICES>", choices_str)

    if idx == 0:
        print(f"Format example:")
        print("-" * 100)
        print(all_prefix)
        print("-" * 100)
        format_example = all_prefix

    all_prefix_ids = [0] + tokenizer.encode(all_prefix.strip())  # RWKV padding
    logits, _ = model.forward(all_prefix_ids, None, full_output=False)

    log_prob = F.log_softmax(logits, dim=-1)
    target_prob = log_prob[choices_token[:num_choices]]
    if subject not in score_by_subject:
        score_by_subject[subject] = {"correct": 0, "total": 0}
    if torch.argmax(target_prob).item() == gt:
        correct += 1
        score_by_subject[subject]["correct"] += 1
    total += 1
    score_by_subject[subject]["total"] += 1
    pbar.set_description(f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}")
    pbar.update(1)
pbar.close()

########################################################################################################
# Save results
for subject in score_by_subject:
    score_by_subject[subject]["accuracy"] = score_by_subject[subject]["correct"] / score_by_subject[subject]["total"]
now = datetime.datetime.now()
file_name = f'logs/results_{now.strftime("%Y%m%d%H%M%S")}.json'
with open(file_name, "w") as f:
    json.dump(
        {
            "model": MODEL_PATH,
            "dataset": DATASET_PATH,
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "template": TEMPLATE,
            "example": format_example,
            "score_by_subject": score_by_subject,
        },
        f,
        indent=4,
    )
print(f"Results saved to {file_name}")
