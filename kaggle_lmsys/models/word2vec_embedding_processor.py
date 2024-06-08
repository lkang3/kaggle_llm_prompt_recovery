from typing import Callable
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import simple_preprocess
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


class W2VEmbeddingProcessor:
    def __init__(self, config: Dict) -> None:
        self._config = config
        self.embedding_aggregators = {
            "max": np.nanmax,
            "min": np.nanmin,
            "mean": np.nanmean,
        }

    @property
    def config(self) -> Dict:
        return self._config

    def _load(self) -> object:
        return KeyedVectors.load(self.config["model_path"])

    def _save(self, model: object) -> None:
        model.save(self.config["model_path"])

    def fit(self, data: pd.DataFrame) -> "DetertaEmbeddingFlow":
        # model = downloader.load(self.config["model_name"])
        # self._save(model)
        return self

    @time_it
    def fit_and_inference(self, data: pd.DataFrame) -> np.ndarray:
        self.fit(data)
        return self.inference(data)

    @time_it
    def inference(self, data: pd.DataFrame) -> np.ndarray:
        model = self._load()
        prompt_field = self.config["data"]["prompt"]
        resp_a_field = self.config["data"]["resp_a"]
        resp_b_field = self.config["data"]["resp_b"]
        data = data.loc[:, [prompt_field, resp_a_field, resp_b_field]].map(simple_preprocess)

        aggregator_func = self.embedding_aggregators[self.config["embedding_aggregator"]]
        prompt_embeddings = data[prompt_field].apply(
            get_sentence_embeddings,
            model=model,
            aggregator=aggregator_func,
        ).values
        resp_a_embeddings = data[resp_a_field].apply(
            get_sentence_embeddings,
            model=model,
            aggregator=aggregator_func,
        ).values
        resp_b_embeddings = data[resp_b_field].apply(
            get_sentence_embeddings,
            model=model,
            aggregator=aggregator_func,
        ).values

        prompt_embeddings = np.vstack(prompt_embeddings.flatten())
        resp_a_embeddings = np.vstack(resp_a_embeddings.flatten())
        resp_b_embeddings = np.vstack(resp_b_embeddings.flatten())
        resp_diff_embeddings = resp_a_embeddings - resp_b_embeddings

        all_embeddings = (
            prompt_embeddings,
            resp_a_embeddings,
            resp_b_embeddings,
            resp_diff_embeddings,
        )
        all_embeddings = np.concatenate(all_embeddings, axis=1)
        return all_embeddings
