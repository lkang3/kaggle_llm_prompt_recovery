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


def get_sentence_embeddings(
    sentence: List[str],
    model: object,
) -> Dict[str, np.ndarray]:
    # (num_tokens, embedding_size)
    embeddings = np.array(
        [
            (
                model.word_vec(word)
                if word in model.key_to_index
                else model.word_vec("unknown")
            )
            for word in sentence.values[0]
        ]
    )
    if not len(embeddings):
        embeddings = np.array([model.word_vec("unknown")])
    # (embedding_size,)
    return {
        "mean": np.nanmean(embeddings, axis=0),
        "min": np.nanmin(embeddings, axis=0),
        "max": np.nanmax(embeddings, axis=0),
    }


class W2VEmbeddingBasicFlow:
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

    def _save_self(self) -> None:
        with open(self.config["pipeline_output_path"], "wb") as output_file:
            pickle.dump(self, output_file)

    def _save(self, model: object) -> None:
        model.save(self.config["model_path"])

    def fit(self, data: ModelData) -> "DetertaEmbeddingFlow":
        # model = downloader.load(self.config["model_name"])
        # self._save(model)
        self._save_self()
        return self

    @time_it
    def fit_and_inference(self, data: ModelData) -> np.ndarray:
        self.fit(data)
        return self.inference(data)

    @time_it
    def inference(self, data: ModelData) -> np.ndarray:
        model = self._load()
        data = pd.DataFrame(data=data.x, columns=data.col_names)
        data = data.map(simple_preprocess)

        aggregator_func = self.embedding_aggregators[
            self.config["embedding_aggregator"]
        ]
        all_embeddings = []
        for col_name in data.columns:
            prompt_embeddings = (
                data[col_name]
                .apply(
                    get_sentence_embeddings,
                    model=model,
                    aggregator=aggregator_func,
                )
                .values
            )

            all_embeddings.append(np.vstack(prompt_embeddings.flatten()))

        all_embeddings = np.concatenate(all_embeddings, axis=1)
        return all_embeddings


class W2VEmbeddingLMSYSFlow:
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
        return self

    @time_it
    def fit_and_inference(self, data: pd.DataFrame) -> ModelData:
        self.fit(data)
        return self.inference(data)

    @time_it
    def inference(self, data: pd.DataFrame) -> ModelData:
        model = self._load()
        prompt_field = self.config["data"]["prompt"]
        resp_a_field = self.config["data"]["resp_a"]
        resp_b_field = self.config["data"]["resp_b"]
        data = data.loc[:, [prompt_field, resp_a_field, resp_b_field]].map(
            simple_preprocess
        )

        aggregator_func = self.embedding_aggregators[
            self.config["embedding_aggregator"]
        ]
        resp_a_embeddings = data.loc[:, [resp_a_field]].apply(
            get_sentence_embeddings,
            model=model,
            axis=1,
            result_type="expand",
        )
        resp_b_embeddings = data.loc[:, [resp_b_field]].apply(
            get_sentence_embeddings,
            model=model,
            axis=1,
            result_type="expand",
        )
        resp_a_embeddings_mean = np.vstack(resp_a_embeddings["mean"].values.flatten())
        resp_b_embeddings_mean = np.vstack(resp_b_embeddings["mean"].values.flatten())

        resp_diff_embeddings = resp_a_embeddings_mean - resp_b_embeddings_mean
        all_embeddings = (
            resp_a_embeddings_mean,
            resp_b_embeddings_mean,
            resp_diff_embeddings,
        )
        all_embeddings = np.concatenate(all_embeddings, axis=1)
        col_names = []
        col_names.extend([f"resp_a_w2v_mean_{i}" for i in range(resp_a_embeddings_mean.shape[1])])
        col_names.extend([f"resp_b_w2v_mean_{i}" for i in range(resp_b_embeddings_mean.shape[1])])
        col_names.extend([f"resp_a_b_w2v_mean_diff_{i}" for i in range(resp_diff_embeddings.shape[1])])

        return ModelData(
            x=all_embeddings,
            data_types=[DataType.NUM] * all_embeddings.shape[1],
            col_names=col_names,
        )
