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

########################################################################################################
# MODEL
MODEL_NAME = "rwkv-x070-1b5-world-v3-80%trained-20250120-ctx4k"
MODEL_TYPE = "rwkv7"  # rwkv6, rwkv7, hf
# MODEL_NAME = "Qwen/Qwen2.5-0.5B"
# MODEL_TYPE = "hf"

print(f"Loading model - {MODEL_NAME}")
if "rwkv" in MODEL_TYPE:
    if MODEL_TYPE == "rwkv7":
        os.environ["RWKV_V7_ON"] = "1"
    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    model = RWKV(model=MODEL_NAME, strategy="cuda fp16")
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    tokenizer = pipeline.tokenizer
elif MODEL_TYPE == "hf":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
########################################################################################################
# PROMPT TEMPLATE
# lm_eval baseline - 42.8%
# TEMPLATE = '''
# The following are multiple choice questions (with answers) about <SUBJECT>.
# <Q>
# A. <|A|>
# B. <|B|>
# C. <|C|>
# D. <|D|>
# Answer:'''

# better format for RWKV - 46.7%
# TEMPLATE = """
# User: <Q>
# A. <|A|>
# B. <|B|>
# C. <|C|>
# D. <|D|>

# Assistant: The answer is"""

# prompt template pro max - 47.9%
TEMPLATE = """User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is"""

# choices
CHOICES = [" A", " B", " C", " D"]
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

if "rwkv" in MODEL_TYPE:
    choices_token = [tokenizer.encode(x) for x in CHOICES]
elif MODEL_TYPE == "hf":
    choices_token = tokenizer(CHOICES)["input_ids"]
assert all([len(x) == 1 for x in choices_token])
choices_token = [x[0] for x in choices_token]

score_by_subject = {}
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

    if "rwkv" in MODEL_TYPE:
        all_prefix_ids = [0] + tokenizer.encode(all_prefix.strip())  # RWKV padding
        logits, _ = model.forward(all_prefix_ids, None, full_output=False)
    elif MODEL_TYPE == "hf":
        all_prefix_ids = tokenizer.encode(all_prefix.strip(), return_tensors="pt")
        all_prefix_ids = all_prefix_ids.to(model.device)
        logits = model.forward(all_prefix_ids).logits[0, -1, :]

    log_prob = F.log_softmax(logits, dim=-1)
    target_prob = log_prob[choices_token]
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

# Save results
for subject in score_by_subject:
    score_by_subject[subject]["accuracy"] = score_by_subject[subject]["correct"] / score_by_subject[subject]["total"]
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
            "score_by_subject": score_by_subject,
        },
        f,
        indent=4,
    )
print(f"Results saved to {file_name}")
