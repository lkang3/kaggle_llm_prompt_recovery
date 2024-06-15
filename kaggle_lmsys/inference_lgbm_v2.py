import logging
import pickle
import sys
import yaml
from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd
import torch

from kaggle_lmsys.utils import clean_data
from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.models.embedding_flow_deterta import DetertaEmbeddingLMSYSFlow
from kaggle_lmsys.models.embedding_flow_word2vec import W2VEmbeddingLMSYSFlow
from kaggle_lmsys.models.embedding_flow_tfidf import TFIDFLMSYSFlow
from kaggle_lmsys.models.embedding_flow_length import LengthFeatureEmbeddingLMSYSFlow

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


SEED = 123
random_state = np.random.RandomState(SEED)
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)
numpy_gen = np.random.Generator(np.random.PCG64(seed=SEED))


hf_token: str = "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"


@click.command()
@click.option("--for_test", type=bool, default=False, required=True)
@click.option("--config_path", type=str, required=True)
def go(
    for_test: bool,
    config_path: str,
) -> None:
    # config stuff
    with open(Path(config_path), "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]
    pipeline_embedding_w2v_config = config["pipeline_embedding_w2v"]
    pipeline_embedding_w2v_config.update(
        {
            "data": {
                "prompt": data_config["prompt"],
                "resp_a": data_config["resp_a"],
                "resp_b": data_config["resp_b"],
            },
        },
    )
    pipeline_embedding_deterba_config = config["pipeline_embedding_deberta"]
    pipeline_embedding_deterba_config.update(
        {
            "data": {
                "prompt": data_config["prompt"],
                "resp_a": data_config["resp_a"],
                "resp_b": data_config["resp_b"],
            },
        },
    )
    pipeline_basic_embedding_tfidf = config["pipeline_basic_embedding_tfidf"]
    pipeline_embedding_length_feature = config["pipeline_embedding_length_feature"]
    classifier_lgbm_pipeline_config = config["classifier_lgbm_pipeline"]

    # data loading
    input_data_path = Path(data_config["test_data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(input_data_path, input_fields)
    if for_test:
        data = data.iloc[:100, :]

    model_inputs: List[np.ndarray] = []

    # length feature embeddings
    length_feature_embedding_flow = LengthFeatureEmbeddingLMSYSFlow(
        pipeline_embedding_length_feature
    )
    length_feature_embeddings = length_feature_embedding_flow.fit_and_inference(data)
    model_inputs.append(length_feature_embeddings)

    # deberta tokenizer embeddings inference
    deterta_embedding_flow = DetertaEmbeddingLMSYSFlow(pipeline_embedding_deterba_config)
    deberta_embeddings = deterta_embedding_flow.inference(data)
    model_inputs.append(deberta_embeddings)

    # classifier fit
    classifier = pickle.load(open(classifier_lgbm_pipeline_config["output_path"], "rb"))
    model_data_ndarray = np.concatenate(model_inputs, axis=1)
    model_data_types = [DataType.NUM] * sum(model_input.shape[1] for model_input in model_inputs)
    model_data = ModelData(
        data_types=np.array(model_data_types),
        x=model_data_ndarray,
    )
    preds = classifier.predict_proba(model_data)
    outputs = pd.DataFrame(
        {
            "id": data["id"],
            "winner_model_a": preds[:, 0],
            "winner_model_b": preds[:, 1],
            "winner_tie": preds[:, 2],
        }
    )
    outputs.to_csv("/kaggle/working/submission.csv", index=False)


if __name__ == "__main__":
    # ! python /kaggle/working/inference_lgbm_v2.py --for_test=False --config_path=
    go()
