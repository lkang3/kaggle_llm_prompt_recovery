from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

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


@torch.no_grad()
def get_deberta_hidden_states(
    texts: List[str],
    llm_model: object,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> torch.tensor:
    """
    IN: (num_total, )
    OUT: (num_total, num_tokens (=max_length), embedding_size)

    :param texts:
    :param llm_model:
    :param tokenizer:
    :param max_length:
    :return:
    """

    embedding_texts = []
    for text in texts:
        tokenized_text = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        hidden_states = llm_model(**tokenized_text).last_hidden_state
        embedding_texts.append(hidden_states)

    return torch.vstack(embedding_texts)


def run_embedding_feature_engineering(
    data: pd.DataFrame,
    prompt_field: str,
    max_prompt_token_length: int,
    resp_a_field: str,
    max_resp_a_token_length: int,
    resp_b_field: str,
    max_resp_b_token_length: int,
    llm_model: object,
    tokenizer: AutoTokenizer,
) -> np.ndarray:
    prompt_texts = data[prompt_field].tolist()
    resp_a_texts = data[resp_a_field].tolist()
    resp_b_texts = data[resp_b_field].tolist()

    prompt_hidden_states = get_deberta_hidden_states(
        prompt_texts,
        llm_model,
        tokenizer,
        max_prompt_token_length,
    )
    resp_a_hidden_states = get_deberta_hidden_states(
        resp_a_texts,
        llm_model,
        tokenizer,
        max_resp_a_token_length,
    )
    resp_b_hidden_states = get_deberta_hidden_states(
        resp_b_texts,
        llm_model,
        tokenizer,
        max_resp_b_token_length,
    )
    agg_axis = 1
    resp_diff = (
        resp_a_hidden_states.mean(axis=agg_axis) - resp_b_hidden_states.mean(axis=agg_axis)
    )
    resp_diff = resp_diff.reshape(resp_diff.shape[0], 1, resp_diff.shape[1])

    all_embeddings = (
        prompt_hidden_states, resp_a_hidden_states, resp_b_hidden_states, resp_diff
    )
    all_embeddings = torch.concatenate(all_embeddings, axis=1)

    return all_embeddings.reshape(len(all_embeddings), -1).numpy()


def simple_tokenization(
    batch_text_records: List,
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device,
    return_tensors: Optional[str] = "pt",
) -> List:
    return tokenizer(
        batch_text_records,
        max_length=max_length,
        truncation=True,
        return_tensors=return_tensors,
    ).to(device)


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
    prompt_field: str,
    max_prompt_token_length: int,
    resp_a_field: str,
    max_resp_a_token_length: int,
    resp_b_field: str,
    max_resp_b_token_length: int,
    target_field: str,
) -> Dict:
    if (
        max_prompt_token_length + max_resp_a_token_length + max_resp_b_token_length
    ) > max_token_length:
        raise ValueError()

    prompt: str = record[prompt_field]
    resp_a: str = record[resp_a_field]
    resp_b: str = record[resp_b_field]
    target_value = record[target_field] if target_field in record else None

    input_ids_key = "input_ids"
    token_type_ids_key = "token_type_ids"
    attention_mask_key = "attention_mask"
    pad_token = tokenizer(tokenizer.pad_token, add_special_tokens=False)

    prompt = tokenizer(
        prompt,
        max_length=max_prompt_token_length,
        truncation=True,
    )
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
            np.array(prompt[input_ids_key], dtype=output_dtype),
            np.array(resp_a[input_ids_key], dtype=output_dtype),
            np.array(resp_b[input_ids_key], dtype=output_dtype),
        )
    )
    if len(token_input_ids) < max_token_length:
        tmp = np.zeros(max_token_length, dtype=output_dtype)
        tmp.fill(pad_token[input_ids_key][0])
        tmp[: len(token_input_ids)] = token_input_ids
        token_input_ids = tmp

    prompt_token_type_ids = np.array(prompt[token_type_ids_key], dtype=output_dtype)
    prompt_token_type_ids.fill(0)
    resp_a_token_type_ids = np.array(resp_a[token_type_ids_key], dtype=output_dtype)
    resp_a_token_type_ids.fill(1)
    resp_b_token_type_ids = np.array(resp_b[token_type_ids_key], dtype=output_dtype)
    resp_b_token_type_ids.fill(2)
    token_type_ids = np.concatenate(
        (
            prompt_token_type_ids,
            resp_a_token_type_ids,
            resp_b_token_type_ids,
        )
    )
    if len(token_type_ids) < max_token_length:
        tmp = np.zeros(max_token_length, dtype=output_dtype)
        tmp.fill(pad_token[input_ids_key][0])
        tmp[: len(token_type_ids)] = token_type_ids
        token_type_ids = tmp
    attention_mask = np.concatenate(
        (
            np.array(prompt[attention_mask_key], dtype=output_dtype),
            np.array(resp_a[attention_mask_key], dtype=output_dtype),
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
