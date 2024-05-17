import math

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
