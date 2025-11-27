# RWKV Evaluation

## Expected Evaluation Results

**Model:** `rwkv7-g1a-2.9b-20250924-ctx4096`

| Dataset    | Single-Token Eval | CoT Eval | QA Eval |
|:----------:|:------------------:|:---------:|:---------:|
| Template   | User: You are a very talented expert in \<SUBJECT\>. Answer this question:<br>\<Q\>\<CHOICES\><br><br>Assistant: The answer is | User: \<Q\><br><br>Assistant: <think | User: \<Q\><br><br>Assistant: |
| MMLU       | 61.2% | ~66% | - |
| MMLU-Pro   | 31.5% | ~42% | - |
| Ceval-exam | 48.9% | ~51% | - |
| GSM8K      | - | ~77% | ~75% |
| MATH500    | - | ~50% (with LLM Judge) | ~44% (with LLM Judge) |

<!-- **Model:** `rwkv7-g0a2preview773-7.2b-20251003-ctx4096`

| Dataset    | Single-Token Eval | CoT Eval | QA Eval |
|:----------:|:------------------:|:---------:|:---------:|
| Template   | User: You are a very talented expert in \<SUBJECT\>. Answer this question:<br>\<Q\>\<CHOICES\><br><br>Assistant: The answer is | User: \<Q\><br><br>Assistant: <think | User: \<Q\><br><br>Assistant: |
| MMLU       | 64.3% | - | - |
| MMLU-Pro   | 34.8% | ~51% | - |
| Ceval-exam | 53.2% | - | - |
| GSM8K      | - | ~84% | - |
| MATH500    | - | ~63% (with LLM Judge) | - | -->

**Model:** `rwkv7-g0a2-7.2b-20251005-ctx4096`

| Dataset    | Single-Token Eval | CoT Eval | QA Eval |
|:----------:|:------------------:|:---------:|:---------:|
| Template   | User: You are a very talented expert in \<SUBJECT\>. Answer this question:<br>\<Q\>\<CHOICES\><br><br>Assistant: The answer is | User: \<Q\><br><br>Assistant: <think | User: \<Q\><br><br>Assistant: |
| MMLU       | 64.4% | - | - |
| MMLU-Pro   | 35.2% | ~50% | - |
| Ceval-exam | 52.8% | - | - |
| GSM8K      | - | ~84% | - |
| MATH500    | - | ~61% (with LLM Judge) | - |

> **Note:**  
> Due to the inherent randomness in CoT evaluation, results may fluctuate. It is recommended to run the evaluation multiple times (with different seeds) and average the results for a more reliable metric.
> Performance can be further improved by further adjusting the sampling parameters amd prompt template.

## Non-CoT Single-Choice Evaluation (Single-Token Eval)

#### For each question, feed it to the model and compare the probabilities assigned to each answer option for the next token. If the correct answer has the highest probability, it is considered correct. This evaluation approach is suitable for datasets like MMLU, MMLU-Pro, and Ceval-exam.

1.  **Prepare Data**

    Data should be in a JSONL file, with each line formatted as follows:
    ```json
    {"question": "xxx", "A": "xxx", "B": "xxx", "C": "xxx", "D": "xxx", "answer": "A", "subject": "xxx"}
    ```
    > **Note:** `MMLU`, `MMLU-Pro`, and `Ceval-exam` datasets are already prepared in the `prepare_data/` directory.

2.  **Edit Settings**

    In `rwkv_single_choice_eval.py`, set the following paths:
    - `MODEL_PATH`
    - `DATASET_PATH`

3.  **Run Script**
    ```bash
    python rwkv_single_choice_eval.py
    ```

## CoT Single-Choice Evaluation

#### After presenting the question to the model, it first generates a reasoning process. The script then appends "Therefore, the answer is" to the end of the reasoning and checks the probability distribution of the next token. If the correct answer token has the highest probability among all options, the answer is considered correct. This evaluation method is suitable for datasets such as MMLU, MMLU-Pro, and Ceval-exam.

1.  **Prepare Data**

    Data should be in a JSONL file, with each line formatted as follows:
    ```json
    {"question": "xxx", "A": "xxx", "B": "xxx", "C": "xxx", "D": "xxx", "answer": "A", "subject": "xxx"}
    ```
    > **Note:** `MMLU`, `MMLU-Pro`, and `Ceval-exam` datasets are already prepared in the `prepare_data/` directory.

2.  **Edit Settings**

    In `rwkv_single_choice_eval_cot.py`, set the following paths:
    - `MODEL_PATH`
    - `DATASET_PATH`

3.  **Run Script**
    ```bash
    python rwkv_single_choice_eval_cot.py
    ```

## General CoT Evaluation

#### After presenting the question to the model, it first generates a reasoning process. The script then appends "Therefore, the answer is \(\\boxed{" to the end of the reasoning, and after the closing boxed bracket, checks whether the content inside the boxed matches the correct answer exactly. If it matches, the answer is considered correct. This evaluation method is suitable for datasets like GSM8K.

1.  **Prepare Data**

    Data should be in a JSONL file, with each line formatted as follows:
    ```json
    {"question": "xxx", "answer": "xxx", "subject": "xxx"}
    ```
    > **Note:** `GSM8K` dataset is already prepared in the `prepare_data/` directory.

2.  **Edit Settings**

    In `rwkv_general_eval_cot.py`, set the following paths:
    - `MODEL_PATH`
    - `DATASET_PATH`

3.  **Run Script**
    ```bash
    python rwkv_general_eval_cot.py
    ```

## General CoT Evaluation with LLM Judge

#### After presenting the question to the model, it first generates a reasoning process. The script then appends "Therefore, the answer is \(\\boxed{" to the end of the reasoning, and after the boxed section is closed, an LLM is used to judge whether the content inside the boxed section is equivalent to the correct answer. If they are equivalent, the answer is considered correct. This evaluation method is suitable for datasets such as MATH500 and AIME2025.

1.  **Prepare Data**

    Data should be in a JSONL file, with each line formatted as follows:
    ```json
    {"question": "xxx", "answer": "xxx", "subject": "xxx"}
    ```
    > **Note:** `Math500`, and `AIME2025` datasets are already prepared in the `prepare_data/` directory.

2.  **Edit Settings**

    In `rwkv_general_eval_cot_llm_judge.py`, set the following paths:
    - `MODEL_PATH`
    - `DATASET_PATH`
    - `OPENAI_API_KEY` (in `.env` file)
    - `API_BASE` (if you want to use a custom API base, in `.env` file)
    - `JUDGE_MODEL` (the model to use for judging, in `.env` file, default is `gpt-4.1`)

3.  **Run Script**
    ```bash
    python rwkv_general_eval_cot_llm_judge.py
    ```

## Instruction Following Evaluation(IFEval)
1. **Generate Responses**
    ```bash
    python if_eval_gen.py
    ```
2. **Evaluate Responses**
    ```bash
    python3 -m instruction_following_eval.evaluation_main \
      --input_data=./instruction_following_eval/data/input_data.jsonl \
      --input_response_data=./instruction_following_eval/data/xxx.jsonl \
      --output_dir=./instruction_following_eval/data/
    ```

## Human-Eval
1. **Install Dependencies**
    ```bash
    $ git clone https://github.com/openai/human-eval
    $ pip install -e human-eval
    ```
2. **Generate Responses**
    ```bash
    python human_eval_generation.py
    ```
3. **Evaluate Responses**
    ```bash
    evaluate_functional_correctness human_eval_results/xxx.jsonl
    ```