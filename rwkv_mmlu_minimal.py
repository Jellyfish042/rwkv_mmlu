########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os, json, datetime, random
from tqdm import tqdm
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.nn import functional as F
from datasets import load_dataset, load_from_disk

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

########################################################################################################
# MODEL
MODEL_NAME = "RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth"

print(f"Loading model - {MODEL_NAME}")
model = RWKV(model=MODEL_NAME, strategy="cuda fp16", verbose=False)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
########################################################################################################
# MMLU DATASET
# mmlu = load_dataset("cais/mmlu", 'all')

# mmlu_test = mmlu['test']
# mmlu_dev = mmlu['dev']

# mmlu_test.save_to_disk('mmlu_test_dataset')
# mmlu_dev.save_to_disk('mmlu_dev_dataset')

mmlu_test = load_from_disk("mmlu_test_dataset")
mmlu_dev = load_from_disk("mmlu_dev_dataset")
########################################################################################################
# PROMPT TEMPLATE
# lm_eval baseline - 42.8%
# TEMPLATE = '''
# User: The following are multiple choice questions (with answers) about <SUBJECT>.
# <Q>
# A. <|A|>
# B. <|B|>
# C. <|C|>
# D. <|D|>
# Answer:'''

# better format for RWKV - 46.7%
TEMPLATE = """
User: <Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is"""

# prompt template pro max - 47.7%
# TEMPLATE = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
# <Q>
# A. <|A|>
# B. <|B|>
# C. <|C|>
# D. <|D|>

# Assistant: The answer is'''

# choices
CHOICES = [" A", " B", " C", " D"]

########################################################################################################
# SET RANDOM SEED
SHUFFLE = False
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
########################################################################################################
# RUN EVALUATION
correct = 0
total = 0
pbar = tqdm(total=len(mmlu_test))

choices_token = [pipeline.tokenizer.encode(x) for x in CHOICES]
assert all([len(x) == 1 for x in choices_token]), "Choices are not single token, use rwkv_mmlu.py instead"
choices_token = [x[0] for x in choices_token]

for idx, sample in enumerate(mmlu_test):
    question = sample["question"]
    choices = sample["choices"]
    subject = sample["subject"]
    gt = sample["answer"]

    if SHUFFLE and not any(["Both" in x for x in choices]):  # exclude choices like "Both A and B"
        original_gt_text = choices[gt]
        np.random.shuffle(choices)
        gt = choices.index(original_gt_text)

    all_prefix = (
        TEMPLATE.replace("<Q>", question)
        .replace("<|A|>", choices[0])
        .replace("<|B|>", choices[1])
        .replace("<|C|>", choices[2])
        .replace("<|D|>", choices[3])
        .replace("<SUBJECT>", subject.replace("_", " "))
    )

    if idx == 0:
        print(f"Format example:")
        print("-" * 100)
        print(all_prefix)
        print("-" * 100)
        format_example = all_prefix

    all_prefix_ids = pipeline.tokenizer.encode(all_prefix)

    logits, _ = model.forward(all_prefix_ids, None, full_output=False)
    neg_log_prob = F.log_softmax(logits, dim=-1)
    target_prob = neg_log_prob[choices_token]
    if torch.argmax(target_prob).item() == gt:
        correct += 1
    total += 1
    pbar.set_description(f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}")
    pbar.update(1)
pbar.close()

# Save results
now = datetime.datetime.now()
file_name = f'logs/mmlu_test_results_{now.strftime("%Y%m%d%H%M%S")}.json'
with open(file_name, "w") as f:
    json.dump(
        {
            "model": MODEL_NAME,
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "template": TEMPLATE,
            "example": format_example,
            "shuffle": SHUFFLE,
            "seed": SEED,
        },
        f,
        indent=4,
    )
print(f"Results saved to {file_name}")
