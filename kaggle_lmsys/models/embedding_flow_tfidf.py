import pickle
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import simple_preprocess
from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.utils import time_it


def get_sentence_embeddings(
    sentence: List[str], model: object, aggregator: Callable,
) -> np.ndarray:
    # (num_tokens, embedding_size)
    embeddings = np.array(
        [
            model.word_vec(word) if word in model.key_to_index else model.word_vec("unknown")
            for word in sentence
        ]
    )
    if not len(embeddings):
        embeddings = np.array([model.word_vec("unknown")])
    # (embedding_size,)
    return aggregator(embeddings, axis=0)


class TFIDFLMSYSFlow:
    def __init__(self, config: Dict) -> None:
        self._config = config
        model_params = config["tfidf"]["params"]
        if "ngram_range" in model_params:
            model_params["ngram_range"] = tuple(model_params["ngram_range"])
        self.model = TfidfVectorizer(**model_params)
        self.col_names = self.config["data"]

    @property
    def config(self) -> Dict:
        return self._config

    def save(self) -> None:
        with open(self.config['pipeline_output_path'], "wb") as output_file:
            pickle.dump(self, output_file)

    def _get_train_data(self, data: pd.DataFrame) -> np.ndarray:
        return data.apply(
            lambda row: " ".join([row[col_name] for col_name in self.col_names]),
            axis=1
        ).values

    @time_it
    def fit(self, data: ModelData) -> "DetertaEmbeddingFlow":
        self.model.fit(self._get_train_data(data))
        self.save()
        return self

    @time_it
    def fit_and_inference(self, data: ModelData) -> ModelData:
        self.fit(data)
        return self.inference(data)

    @time_it
    def inference(self, data: ModelData) -> ModelData:
        all_embeddings = []
        col_names = []
        for col_name in self.col_names:
            embeddings = self.model.transform(data[col_name].values)
            embeddings = embeddings.toarray()
            all_embeddings.append(embeddings)
            col_names.extend([f"{col_name}_tfidf_{i}" for i in range(embeddings.shape[1])])

        embeddings = np.concatenate(all_embeddings, axis=1)

        return ModelData(
            x=embeddings,
            data_types=[DataType.NUM] * embeddings.shape[1],
            col_names=col_names,
        )
