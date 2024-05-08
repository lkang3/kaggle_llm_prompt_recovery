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
