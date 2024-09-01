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
from rwkv.utils import PIPELINE, PIPELINE_ARGS

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
# PRE_TEMPLATE = '\nThe following are multiple choice questions (with answers) about <SUBJECT>.'
# QUESTION_TEMPLATE = "\n<Q>"
# CHOICE_TEMPLATE = "\nA. <|A|>\nB. <|B|>\nC. <|C|>\nD. <|D|>"
# ANSWER_TEMPLATE = "\nAnswer:"

# better format for RWKV - 46.7%
PRE_TEMPLATE = "\nUser: "
QUESTION_TEMPLATE = "<Q>"
CHOICE_TEMPLATE = "\nA. <|A|>\nB. <|B|>\nC. <|C|>\nD. <|D|>"
ANSWER_TEMPLATE = "\n\nAssistant: The answer is"

# Professional format - 47.7%
# PRE_TEMPLATE = "User: You are a very talented expert in <SUBJECT>. Answer this question:\n" # Correct: 6696 - Total: 14042 - Accuracy: 0.47686
# QUESTION_TEMPLATE = "<Q>"
# CHOICE_TEMPLATE = "\nA. <|A|>\nB. <|B|>\nC. <|C|>\nD. <|D|>"
# ANSWER_TEMPLATE = "\n\nAssistant: The answer is"

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
assert all([len(x) == 1 for x in choices_token]), "Choices are not single token"
choices_token = [x[0] for x in choices_token]

for idx, sample in enumerate(mmlu_test):
    question = sample["question"]
    choices = sample["choices"]
    subject = sample["subject"]
    gt = sample["answer"]

    if SHUFFLE and not any(["Both" in x for x in choices]):
        original_gt_text = choices[gt]
        np.random.shuffle(choices)
        gt = choices.index(original_gt_text)

    pre_prompt = PRE_TEMPLATE.replace("<SUBJECT>", subject.replace("_", " "))
    question_prompt = QUESTION_TEMPLATE.replace("<Q>", question)
    choices_prompt = (
        CHOICE_TEMPLATE.replace("<|A|>", choices[0])
        .replace("<|B|>", choices[1])
        .replace("<|C|>", choices[2])
        .replace("<|D|>", choices[3])
    )
    answer_prompt = ANSWER_TEMPLATE
    # print('\n\n' + '=' * 100)
    # print(f'Question: {question}\nChoices: {choices}\nGT: {gt}')

    all_prefix = pre_prompt + question_prompt + choices_prompt + answer_prompt

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
    pbar.set_description(
        f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}"
    )
    pbar.update(1)
pbar.close()

print(f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}")

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
            "PRE_TEMPLATE": PRE_TEMPLATE,
            "QUESTION_TEMPLATE": QUESTION_TEMPLATE,
            "CHOICE_TEMPLATE": CHOICE_TEMPLATE,
            "ANSWER_TEMPLATE": ANSWER_TEMPLATE,
            "example": format_example,
            "shuffle": SHUFFLE,
            "seed": SEED,
        },
        f,
        indent=4,
    )
