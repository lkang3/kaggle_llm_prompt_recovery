import pickle
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CosineSimilarity
from transformers import AutoModel
from transformers import AutoTokenizer

from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.utils import get_device
from kaggle_lmsys.utils import time_it


def get_cosine_similarities_of_two_embeddings(
    embedding_one: np.ndarray, embedding_two: np.ndarray
) -> np.ndarray:
    if embedding_one.shape != embedding_two.shape:
        raise ValueError(
            f"embedding_one shape: {embedding_one.shape} != embedding_two shape: {embedding_two.shape}"
        )

    size = len(embedding_one)
    similarities = np.zeros(embedding_one.size)
    for i in range(size):
        similarities[i] = cosine_similarity(
            embedding_one[i].reshape(1, -1), embedding_two[i].reshape(1, -1)
        )

    return similarities


@torch.no_grad()
def extract_deterta_embeddings(
    batch_size: int,
    texts: List[str],
    llm_model: object,
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device,
    agg_operators: List[str],
) -> Dict[str, torch.tensor]:
    """
    IN: (num_total, )
    OUT: (num_total, embedding_size)

    :param batch_size:
    :param texts:
    :param llm_model:
    :param tokenizer:
    :param max_length:
    :param device:
    :param agg_operators:
    :return:
    """

    first_hidden_states = []
    mean_hidden_states = []
    max_hidden_states = []
    min_hidden_states = []
    size_of_texts = len(texts)
    for i in range(0, size_of_texts, batch_size):
        text = texts[i : min(size_of_texts, i + batch_size)]
        tokenized_text = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokenized_text.to(device)
        last_hidden_states = llm_model(**tokenized_text).last_hidden_state
        if "first" in agg_operators:
            first_hidden_states.append(last_hidden_states[:, 0, :])
        agg_axis = 1
        if "mean" in agg_operators:
            mean_hidden_states.append(last_hidden_states.mean(axis=agg_axis))
        if "max" in agg_operators:
            max_hidden_states.append(last_hidden_states.max(axis=agg_axis))
        if "min" in agg_operators:
            min_hidden_states.append(last_hidden_states.min(axis=agg_axis))

    embedding_texts = defaultdict(list)
    if first_hidden_states:
        embedding_texts["first"] = torch.vstack(first_hidden_states)
    if mean_hidden_states:
        embedding_texts["mean"] = torch.vstack(mean_hidden_states)
    if max_hidden_states:
        embedding_texts["max"] = torch.vstack(max_hidden_states)
    if min_hidden_states:
        embedding_texts["min"] = torch.vstack(min_hidden_states)
    return embedding_texts


@time_it
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
    device: torch.device,
    batch_size: int = 10,
) -> Tuple[List[str], np.ndarray]:
    prompt_texts = data[prompt_field].tolist()
    resp_a_texts = data[resp_a_field].tolist()
    resp_b_texts = data[resp_b_field].tolist()
    agg_key = "mean"
    agg_operators = [agg_key]

    prompt_embeddings = extract_deterta_embeddings(
        batch_size,
        prompt_texts,
        llm_model,
        tokenizer,
        max_prompt_token_length,
        device,
        agg_operators,
    )
    resp_a_embeddings = extract_deterta_embeddings(
        batch_size,
        resp_a_texts,
        llm_model,
        tokenizer,
        max_resp_a_token_length,
        device,
        agg_operators,
    )
    resp_b_embeddings = extract_deterta_embeddings(
        batch_size,
        resp_b_texts,
        llm_model,
        tokenizer,
        max_resp_b_token_length,
        device,
        agg_operators,
    )

    resp_diff_embeddings = resp_a_embeddings[agg_key] - resp_b_embeddings[agg_key]
    cs = CosineSimilarity(dim=1)
    resp_a_and_prompt_cosine_similarity = cs(
        resp_a_embeddings[agg_key], prompt_embeddings[agg_key]
    ).unsqueeze(-1)
    resp_b_and_prompt_cosine_similarity = cs(
        resp_b_embeddings[agg_key], prompt_embeddings[agg_key]
    ).unsqueeze(-1)

    all_embeddings = (
        resp_diff_embeddings,
        resp_a_and_prompt_cosine_similarity,
        resp_b_and_prompt_cosine_similarity,
    )
    col_names = []
    col_names.extend([f"resp_a_b_diff_{i}" for i in range(resp_diff_embeddings.shape[1])])
    col_names.extend([f"resp_a_b_cs_{i}" for i in range(resp_a_and_prompt_cosine_similarity.shape[1])])
    col_names.extend([f"resp_a_b_cs_{i}" for i in range(resp_b_and_prompt_cosine_similarity.shape[1])])
    all_embeddings = torch.concatenate(all_embeddings, axis=1)
    return col_names, all_embeddings.detach().cpu().numpy()


@time_it
def get_embeddings(
    data: pd.DataFrame,
    max_token_lengths: List[int],
    llm_model: object,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 10,
) -> np.ndarray:
    agg_operators = ["mean"]

    all_embeddings = []
    for col_name, max_token_length in zip(data.columns, max_token_lengths):
        embeddings = extract_deterta_embeddings(
            batch_size,
            data[col_name].tolist(),
            llm_model,
            tokenizer,
            max_token_length,
            device,
            agg_operators,
        )
        all_embeddings.append(embeddings["mean"])

    all_embeddings = torch.concatenate(all_embeddings, axis=1)
    return all_embeddings.detach().cpu().numpy()


class DetertaEmbeddingBasicFlow:
    def __init__(self, config: Dict) -> None:
        self._config = config
        self.device = get_device()
        self.embedding_aggregators = {
            "max": np.nanmax,
            "min": np.nanmin,
            "mean": np.nanmean,
        }

    @property
    def config(self) -> Dict:
        return self._config

    def _setup_tokenizer(self) -> object:
        return AutoTokenizer.from_pretrained(self.config["tokenizer_name"])

    def _setup_model(self) -> object:
        llm_model = AutoModel.from_pretrained(self.config["model_name"])
        llm_model.to(self.device)
        return llm_model

    def _save(self, tokenizer: object, model: object) -> None:
        tokenizer.save_pretrained(self.config["tokenizer_output_path"])
        model.save_pretrained(self.config["model_output_path"])
        with open(self.config["pipeline_output_path"], "wb") as output_file:
            pickle.dump(self, output_file)

    def _load_tokenizer(self) -> object:
        return AutoTokenizer.from_pretrained(self.config["tokenizer_output_path"])

    def _load_model(self) -> object:
        model = AutoModel.from_pretrained(self.config["model_output_path"])
        model.to(self.device)
        return model

    @time_it
    def fit(self, data: ModelData) -> "DetertaEmbeddingLMSYSFlow":
        tokenizer = self._setup_tokenizer()
        model = self._setup_model()
        self._save(tokenizer, model)
        return self

    def fit_and_inference(self, data: ModelData) -> np.ndarray:
        self.fit(data)
        return self.inference(data)

    @time_it
    def inference(self, data: ModelData) -> np.ndarray:
        tokenizer = self._load_tokenizer()
        model = self._load_model()

        data = pd.DataFrame(data=data.x, columns=data.col_names)
        return get_embeddings(
            data=data,
            max_token_lengths=[self.config["max_token_length"]] * data.shape[1],
            llm_model=model,
            tokenizer=tokenizer,
            device=self.device,
        )


class DetertaEmbeddingLMSYSFlow:
    def __init__(self, config: Dict) -> None:
        self._config = config
        self.device = get_device()

    @property
    def config(self) -> Dict:
        return self._config

    def _setup_tokenizer(self) -> object:
        return AutoTokenizer.from_pretrained(self.config["tokenizer_name"])

    def _setup_model(self) -> object:
        llm_model = AutoModel.from_pretrained(self.config["model_name"])
        llm_model.to(self.device)
        return llm_model

    def _save_tokenizer(self, tokenizer: object) -> None:
        tokenizer.save_pretrained(self.config["tokenizer_output_path"])

    def _save_model(self, model: object) -> None:
        model.save_pretrained(self.config["model_output_path"])

    def _load_tokenizer(self) -> object:
        return AutoTokenizer.from_pretrained(self.config["tokenizer_output_path"])

    def _load_model(self) -> object:
        model = AutoModel.from_pretrained(self.config["model_output_path"])
        model.to(self.device)
        return model

    @time_it
    def fit(self, data: pd.DataFrame) -> "DetertaEmbeddingLMSYSFlow":
        tokenizer = self._setup_tokenizer()
        model = self._setup_model()
        self._save_tokenizer(tokenizer)
        self._save_model(model)
        return self

    def fit_and_inference(self, data: pd.DataFrame) -> ModelData:
        self.fit(data)
        return self.inference(data)

    @time_it
    def inference(self, data: pd.DataFrame) -> ModelData:
        tokenizer = self._load_tokenizer()
        model = self._load_model()

        col_names, embeddings = run_embedding_feature_engineering(
            data=data,
            prompt_field=self.config["data"]["prompt"],
            max_prompt_token_length=self.config["max_prompt_token_length"],
            resp_a_field=self.config["data"]["resp_a"],
            max_resp_a_token_length=self.config["max_resp_a_token_length"],
            resp_b_field=self.config["data"]["resp_b"],
            max_resp_b_token_length=self.config["max_resp_b_token_length"],
            llm_model=model,
            tokenizer=tokenizer,
            device=self.device,
        )

        return ModelData(
            x=embeddings,
            col_names=col_names,
            data_types=[DataType.NUM] * embeddings.shape[1],
        )
