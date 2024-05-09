from typing import Dict
from typing import List

from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
import torch


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def clean_data(
    path: Path,
    field_names: List[str],
) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.fillna("", inplace=True)
    for field_name in field_names:
        data[field_name] = data[field_name].apply(lambda x: x[len('["'): x.find('"]')])
    return data


def get_tokenization_length(
    text: str,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> int:
    tokenized_outputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
    )
    return len(tokenized_outputs["input_ids"])


def tokenization(
    record: Dict,
    tokenizer: AutoTokenizer,
    max_length: int,
    prompt_field: str,
    resp_a_field: str,
    resp_b_field: str,
    target_field: str,
) -> Dict:
    prompt: str = record[prompt_field]
    resp_a: str = record[resp_a_field]
    resp_b: str = record[resp_b_field]
    resp: str = f"{resp_a}{resp_b}"
    target_value = record[target_field] if target_field in record else None

    processed_record = tokenizer(
        prompt,
        resp,
        max_length=max_length,
        truncation=True,
    )
    if target_value is not None:
        processed_record["label"] = target_value

    return processed_record
