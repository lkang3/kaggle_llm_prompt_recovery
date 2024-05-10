from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def clean_data(
    path: Path,
    field_names: List[str],
) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.fillna("", inplace=True)
    for field_name in field_names:
        data[field_name] = data[field_name].apply(lambda x: x[len('["') : x.find('"]')])
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
        processed_record[target_field] = target_value

    return processed_record


def tokenization_separate(
    record: Dict,
    tokenizer: AutoTokenizer,
    max_token_length: int,
    resp_a_field: str,
    max_resp_a_token_length: int,
    resp_b_field: str,
    max_resp_b_token_length: int,
    target_field: str,
) -> Dict:
    if max_resp_a_token_length + max_resp_b_token_length + 1 > max_token_length:
        raise ValueError()

    resp_a: str = record[resp_a_field]
    resp_b: str = record[resp_b_field]
    target_value = record[target_field] if target_field in record else None

    input_ids_key = "input_ids"
    token_type_ids_key = "token_type_ids"
    attention_mask_key = "attention_mask"
    sep_token = tokenizer(tokenizer.sep_token, add_special_tokens=False)
    pad_token = tokenizer(tokenizer.pad_token, add_special_tokens=False)

    resp_a = tokenizer(
        resp_a,
        max_length=max_resp_a_token_length,
        truncation=True,
    )
    resp_b = tokenizer(
        resp_b,
        max_length=max_resp_b_token_length,
        truncation=True,
    )

    output_dtype = np.int32
    token_input_ids = np.concatenate(
        (
            np.array(resp_a[input_ids_key], dtype=output_dtype),
            np.array(sep_token[input_ids_key]),
            np.array(resp_b[input_ids_key], dtype=output_dtype),
        )
    )
    if len(token_input_ids) < max_token_length:
        tmp = np.zeros(max_token_length, dtype=output_dtype)
        tmp.fill(pad_token[input_ids_key][0])
        tmp[: len(token_input_ids)] = token_input_ids
        token_input_ids = tmp
    token_type_ids = np.concatenate(
        (
            np.array(resp_a[token_type_ids_key], dtype=output_dtype),
            np.array(sep_token[token_type_ids_key]),
            np.array(resp_b[token_type_ids_key], dtype=output_dtype),
        )
    )
    if len(token_type_ids) < max_token_length:
        tmp = np.zeros(max_token_length, dtype=output_dtype)
        tmp.fill(pad_token[input_ids_key][0])
        tmp[: len(token_type_ids)] = token_type_ids
        token_type_ids = tmp
    attention_mask = np.concatenate(
        (
            np.array(resp_a[attention_mask_key], dtype=output_dtype),
            np.array(sep_token[attention_mask_key]),
            np.array(resp_b[attention_mask_key], dtype=output_dtype),
        )
    )
    if len(attention_mask) < max_token_length:
        tmp = np.zeros(max_token_length, dtype=output_dtype)
        tmp.fill(pad_token[input_ids_key][0])
        tmp[: len(attention_mask)] = attention_mask
        attention_mask = tmp

    outputs = {
        input_ids_key: token_input_ids,
        token_type_ids_key: token_type_ids,
        attention_mask_key: attention_mask,
    }
    if target_value is not None:
        outputs[target_field] = target_value
    return outputs


class Collator(DataCollatorWithPadding):
    device = get_device()

    def __call__(self, features):
        batch = super().__call__(features)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch
