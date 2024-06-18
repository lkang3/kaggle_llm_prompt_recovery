import pickle
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import simple_preprocess
from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.utils import time_it


def has_none(vals) -> int:
    return int(any(val is None for val in vals))


def str_length(vals) -> int:
    length = 0
    for val in vals:
        if isinstance(val, str):
            length += len(val)
    return length


class LengthFeatureEmbeddingLMSYSFlow:

    def __init__(self, config: Dict) -> None:
        self._config = config

    @property
    def config(self) -> Dict:
        return self._config

    def fit(self, data: pd.DataFrame) -> "DetertaEmbeddingFlow":
        return self

    @time_it
    def fit_and_inference(self, data: pd.DataFrame) -> ModelData:
        self.fit(data)
        return self.inference(data)

    @time_it
    def inference(self, data: pd.DataFrame) -> ModelData:
        resp_a_field = self.config["data"]["resp_a"]
        resp_b_field = self.config["data"]["resp_b"]
        resp_a = data[resp_a_field]
        resp_b = data[resp_b_field]
        resp_a_len = resp_a.apply(len).values
        resp_b_len = resp_b.apply(len).values
        resp_len_diff = resp_a_len - resp_b_len
        resp_len_mean = (resp_a_len + resp_b_len) / 2
        resp_len_diff_mean_ratio = resp_len_diff / resp_len_mean

        resp_a_len = resp_a_len.reshape((-1, 1))
        resp_b_len = resp_b_len.reshape((-1, 1))
        resp_len_diff = resp_len_diff.reshape((-1, 1))
        resp_len_mean = resp_len_mean.reshape((-1, 1))
        resp_len_diff_mean_ratio = resp_len_diff_mean_ratio.reshape((-1, 1))

        embeddings = np.concatenate(
            [resp_a_len, resp_b_len, resp_len_diff, resp_len_mean, resp_len_diff_mean_ratio],
            axis=1,
        )
        col_names = [
            "resp_a_len",
            "resp_b_len",
            "resp_a_b_len_diff",
            "resp_a_b_len_mean",
            "resp_a_b_len_ratio",
        ]

        return ModelData(
            x=embeddings,
            data_types=[DataType.NUM] * embeddings.shape[1],
            col_names=col_names,
        )
