import math
from collections import defaultdict

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer


def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False


def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in iter(dataset):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)


def test():
    # https://huggingface.co/learn/nlp-course/en/chapter7/6#gathering-the-data

    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
    raw_datasets = DatasetDict(
        {
            "train": ds_train,  # .shuffle().select(range(50000)),
            "valid": ds_valid,  # .shuffle().select(range(500))
        }
    )

    raw_datasets


if __name__ == "__main__":
    test()
