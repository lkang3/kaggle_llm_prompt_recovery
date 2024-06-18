import pickle
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.utils import time_it


def get_column_indices(col_name_prefix: str, col_names: List[str]) -> List[int]:
    return [
        col_idx
        for col_idx, col_name in enumerate(col_names)
        if col_name.find(col_name_prefix) == 0
    ]


class EmbeddingDiffFLow:
    def __init__(self, config: Dict) -> None:
        self._config = config
        self.col_pairs = self.config["col_pairs"]

    @property
    def config(self) -> Dict:
        return self._config

    def save(self) -> None:
        with open(self.config['pipeline_output_path'], "wb") as output_file:
            pickle.dump(self, output_file)

    @time_it
    def fit(self, data: ModelData) -> "DetertaEmbeddingFlow":
        return self

    @time_it
    def fit_and_inference(self, data: ModelData) -> ModelData:
        self.fit(data)
        return self.inference(data)

    @time_it
    def inference(self, data: ModelData) -> ModelData:
        all_embeddings = []
        col_names = []
        for diff_col_names in self.col_pairs:
            col_one_prefix = diff_col_names[0]
            col_two_prefix = diff_col_names[1]
            col_one_indices = get_column_indices(col_one_prefix, data.col_names)
            col_two_indices = get_column_indices(col_two_prefix, data.col_names)
            col_one = data.x[:, col_one_indices]
            col_two = data.x[:, col_two_indices]
            col_diff = col_one - col_two
            all_embeddings.append(col_diff)
            col_names.extend(
                [f"{col_one_prefix}_{col_two_prefix}_DIFF_{i}" for i in range(col_diff.shape[1])]
            )

        all_embeddings_concatenated = np.concatenate(all_embeddings, axis=1)
        return ModelData(
            x=all_embeddings_concatenated,
            data_types=[DataType.NUM] * all_embeddings_concatenated.shape[1],
            col_names=col_names,
        )


class EmbeddingCosineSimilarityFlow:
    def __init__(self, config: Dict) -> None:
        self._config = config
        self.col_pairs = self.config["col_pairs"]

    @property
    def config(self) -> Dict:
        return self._config

    def save(self) -> None:
        with open(self.config['pipeline_output_path'], "wb") as output_file:
            pickle.dump(self, output_file)

    @staticmethod
    def get_cosine_similarities_of_two_embeddings(
        embedding_one: np.ndarray, embedding_two: np.ndarray
    ) -> np.ndarray:
        if embedding_one.shape != embedding_two.shape:
            raise ValueError(
                f"embedding_one shape: {embedding_one.shape} != embedding_two shape: {embedding_two.shape}"
            )

        size = len(embedding_one)
        similarities = np.zeros(len(embedding_one))
        for i in range(size):
            similarities[i] = cosine_similarity(
                embedding_one[i].reshape(1, -1), embedding_two[i].reshape(1, -1)
            )

        return similarities

    @time_it
    def fit(self, data: ModelData) -> "DetertaEmbeddingFlow":
        return self

    @time_it
    def fit_and_inference(self, data: ModelData) -> ModelData:
        self.fit(data)
        return self.inference(data)

    @time_it
    def inference(self, data: ModelData) -> ModelData:
        all_embeddings = []
        col_names = []
        for diff_col_names in self.col_pairs:
            col_one_prefix = diff_col_names[0]
            col_two_prefix = diff_col_names[1]
            col_one_indices = get_column_indices(col_one_prefix, data.col_names)
            col_two_indices = get_column_indices(col_two_prefix, data.col_names)
            col_one = data.x[:, col_one_indices]
            col_two = data.x[:, col_two_indices]
            col_cosine_similarity = self.get_cosine_similarities_of_two_embeddings(col_one, col_two)
            all_embeddings.append(col_cosine_similarity)
            col_names.extend(
                [
                    f"{col_one_prefix}_{col_two_prefix}_CS_{i}"
                    for i in range(col_cosine_similarity.shape[1])
                ]
            )

        all_embeddings_concatenated = np.concatenate(all_embeddings, axis=1)
        return ModelData(
            x=all_embeddings_concatenated,
            data_types=[DataType.NUM] * all_embeddings_concatenated.shape[1],
            col_names=col_names,
        )
