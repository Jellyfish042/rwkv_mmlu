# RWKV Single Choice Evaluation

`rwkv_single_choice_eval.py` is a script for evaluating the single-choice question accuracy of RWKV models.

## Usage

1. Edit `rwkv_single_choice_eval.py` to set:
   - `MODEL_PATH`
   - `DATASET_PATH`
2. Run:
   ```bash
   python rwkv_single_choice_eval.py
   ```

## Datasets

MMLU, MMLU-Pro, and Ceval-exam datasets are already prepared in `prepare_data/` folder.  
To evaluate on other datasets, format the dataset as a JSONL file with one example per line, and update the dataset path.

Each line should have the following JSON format:
```json
{"question": "xxx", "A": "xxx", "B": "xxx", "C": "xxx", "D": "xxx", "E": "xxx", "answer": "A", "subject": "xxx"}
```