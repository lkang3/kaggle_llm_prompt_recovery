import math
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd


def add_target(
    data: pd.DataFrame,
    resp_a_win_field: str,
    resp_b_win_field: str,
    resp_tie_field: str,
    added_target_field: str
) -> pd.DataFrame:
    data[added_target_field] = data[
        [resp_a_win_field, resp_b_win_field, resp_tie_field]
    ].apply(lambda x: x.values.tolist().index(1), axis=1)
    return data


def add_model_type_as_target(
    data: pd.DataFrame,
    model_type_fields: List[str],
) -> pd.DataFrame:
    model_types = np.concatenate(
        [
            np.unique(data[model_type_field])
            for model_type_field in model_type_fields
        ]
    )
    model_types = np.unique(model_types)
    model_type_encoding_map = {
        model_types[i]: i
        for i in range(len(model_types))
    }
    for model_type_field in model_type_fields:
        data[f"{model_type_field}_code"] = (
            data[model_type_field].map(lambda x: model_type_encoding_map[x])
        )

    return data


def create_model_type_train_data(
    data: pd.DataFrame,
    output_prompt_field_name: str,
    output_response_field_name: str,
    output_model_type_field_name: str,
) -> Tuple[pd.DataFrame, int]:
    swapped_data = data.apply(swap_responses, axis=1)
    data = pd.concat((data, swapped_data), ignore_index=True)
    model_type_fields = ["model_a", "model_b"]
    model_types = np.concatenate(
        [data[model_type_field] for model_type_field in model_type_fields]
    )
    model_types = np.unique(model_types)
    model_type_encoding_map = {
        model_types[i]: i
        for i in range(len(model_types))
    }
    data[output_model_type_field_name] = data["model_a"].map(lambda x: model_type_encoding_map[x])
    output_fields = ["prompt", "response_a", output_model_type_field_name]
    outputs = data.loc[:, output_fields]
    outputs = outputs.rename(
        columns={
            "prompt": output_prompt_field_name,
            "response_a": output_response_field_name,
        }
    )
    return outputs, len(model_types)


def get_max_train_steps(num_of_samples: int, train_batch_size: int) -> int:
    return math.ceil(num_of_samples / train_batch_size)


def swap_responses(row: pd.Series) -> pd.Series:
    if row["winner_tie"] == 0:
        tmp = row["response_a"]
        row["response_a"] = row["response_b"]
        row["response_b"] = tmp
        tmp = row["model_a"]
        row["model_a"] = row["model_b"]
        row["model_b"] = tmp
        if row["winner_model_a"] == 1:
            row["winner_model_a"] = 0
            row["winner_model_b"] = 1
        else:
            row["winner_model_a"] = 1
            row["winner_model_b"] = 0

    return row