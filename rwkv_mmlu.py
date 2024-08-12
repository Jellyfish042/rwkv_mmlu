########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import os, json, datetime, copy
from tqdm import tqdm
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)

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


def is_specific_subject(example, subject):
    return example["subject"] == subject


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

# other templates
FEW_SHOT_TEMPLATE = "\nUser: <Q><CHOICES>\n\nAssistant: The answer is <A>\n"

GEN_INT_TEMPLATE = "\n\nAssistant: Let's think step by step.\n"
GEN_ANS_ADD = "\nTherefore, the answer is:"

# choices
CHOICES = [" A", " B", " C", " D"]
# CHOICES = ['A', 'B', 'C', 'D']

########################################################################################################
# GENERATION ARGS
args = PIPELINE_ARGS(
    temperature=1.0,
    top_p=0.73,
    top_k=10,  # top_k = 0 then ignore
    alpha_frequency=0.25,
    alpha_presence=0.25,
    alpha_decay=0.996,  # gradually decay the penalty
    token_ban=[0],  # ban the generation of some tokens
    token_stop=[261],  # stop generation whenever you see any token here
    chunk_len=256,
)  # split input into chunks to save VRAM (shorter -> slower)

########################################################################################################
# RUN EVALUATION
USE_COT = False
USE_FEW_SHOT = False  # 5-shot

correct = 0
correct_norm = 0
total = 0
pbar = tqdm(total=len(mmlu_test))

choices_token = [pipeline.tokenizer.encode(x) for x in CHOICES]
use_fast_mode = all([len(x) == 1 for x in choices_token])
if use_fast_mode:
    choices_token = [x[0] for x in choices_token]
    print("all choices are single token, use fast mode")
else:
    print("some choices are not single token, use slow mode")

for idx, sample in enumerate(mmlu_test):
    question = sample["question"]
    choices = sample["choices"]
    subject = sample["subject"]
    gt = sample["answer"]

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

    if USE_FEW_SHOT:
        few_shot_prompt = ""
        filtered_dataset = mmlu_dev.filter(
            is_specific_subject, fn_kwargs={"subject": subject}
        )
        for sample in filtered_dataset:
            few_shot_choices_prompt = (
                CHOICE_TEMPLATE.replace("<|A|>", sample["choices"][0])
                .replace("<|B|>", sample["choices"][1])
                .replace("<|C|>", sample["choices"][2])
                .replace("<|D|>", sample["choices"][3])
            )
            few_shot_prompt += (
                FEW_SHOT_TEMPLATE.replace("<Q>", sample["question"])
                .replace("<CHOICES>", few_shot_choices_prompt)
                .replace("<A>", ["A", "B", "C", "D"][sample["answer"]])
            )
            # print(sample)
        # print(few_shot_prompt)
    else:
        few_shot_prompt = ""

    if USE_COT:
        gen_intermediate_prompt = (
            pre_prompt + question_prompt + choices_prompt + GEN_INT_TEMPLATE
        )
        thoughts = pipeline.generate(
            gen_intermediate_prompt, token_count=512, args=args
        )
        all_prefix = gen_intermediate_prompt + thoughts + GEN_ANS_ADD
    else:
        all_prefix = (
            few_shot_prompt
            + pre_prompt
            + question_prompt
            + choices_prompt
            + answer_prompt
        )

    if idx == 0:
        print(f"Format example:")
        print("-" * 100)
        print(all_prefix)
        print("-" * 100)
        format_example = all_prefix

    all_prefix_ids = pipeline.tokenizer.encode(all_prefix)
    prefix_ids_length = len(all_prefix_ids)

    if use_fast_mode:
        logits, _ = model.forward(all_prefix_ids, None, full_output=False)
        neg_log_prob = F.log_softmax(logits, dim=-1)
        target_prob = neg_log_prob[choices_token]
        if torch.argmax(target_prob).item() == gt:
            correct += 1
            correct_norm += 1
        total += 1
        pbar.set_description(
            f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f} - Accuracy Norm: {correct_norm / total:.5f}"
        )
        pbar.update(1)
    else:
        prefix_ids_pre = all_prefix_ids[:-1]
        prefix_ids_last = all_prefix_ids[-1:]
        _, state_cache = model.forward(prefix_ids_pre, None)

        temp_log_prob_list = []
        temp_log_prob_norm_list = []
        # for answer in choices:
        for answer in CHOICES:
            answer_ids = pipeline.tokenizer.encode(answer)
            input_ids = prefix_ids_last + answer_ids[:-1]
            # print(f'Length All Prefix: {len(all_prefix_ids)} - Length Answer: {len(answer_ids)} - Total: {len(input_ids)}')
            logits, _ = model.forward(
                input_ids, copy.deepcopy(state_cache), full_output=True
            )
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            assert len(answer_ids) == logits.shape[0]
            neg_log_prob = F.log_softmax(logits, dim=-1)
            target_prob = neg_log_prob[range(len(answer_ids)), answer_ids]
            target_prob_sum = target_prob.sum()
            target_prob_sum_norm = target_prob.sum() / len(
                answer
            )  # normalize by the length of the character
            temp_log_prob_list.append(target_prob_sum)
            temp_log_prob_norm_list.append(target_prob_sum_norm)
            # print(f'Target Prob Sum: {target_prob_sum}')
            # print(f'Answer Section: {answer_section.shape}')
        # print(torch.argmax(torch.tensor(temp_log_prob_list)).item(), gt)
        if torch.argmax(torch.tensor(temp_log_prob_list)).item() == gt:
            correct += 1
        if torch.argmax(torch.tensor(temp_log_prob_norm_list)).item() == gt:
            correct_norm += 1
        total += 1
        pbar.set_description(
            f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f} - Accuracy Norm: {correct_norm / total:.5f}"
        )
        pbar.update(1)
pbar.close()

print(
    f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f} - Accuracy Norm: {correct_norm / total:.5f}"
)

# Save results
now = datetime.datetime.now()
file_name = f'logs/mmlu_test_results_{now.strftime("%Y%m%d%H%M%S")}.json'
with open(file_name, "w") as f:
    json.dump(
        {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "accuracy_norm": correct_norm / total,
            "PRE_TEMPLATE": PRE_TEMPLATE,
            "QUESTION_TEMPLATE": QUESTION_TEMPLATE,
            "CHOICE_TEMPLATE": CHOICE_TEMPLATE,
            "ANSWER_TEMPLATE": ANSWER_TEMPLATE,
            "GEN_INT_TEMPLATE": GEN_INT_TEMPLATE,
            "GEN_ANS_ADD": GEN_ANS_ADD,
            "USE_COT": USE_COT,
            "example": format_example,
        },
        f,
        indent=4,
    )
