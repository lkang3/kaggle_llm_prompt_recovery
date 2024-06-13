import copy
import logging
import sys
import yaml
from pathlib import Path
from typing import List

import click
import numpy as np
import torch
from huggingface_hub import login as hf_login

from train_utils import add_target
from kaggle_lmsys.utils import clean_data
from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.models.embedding_flow_deterta import DetertaEmbeddingLMSYSFlow
from kaggle_lmsys.models.embedding_flow_tfidf import TFIDFLMSYSFlow
from kaggle_lmsys.models.embedding_flow_word2vec import W2VEmbeddingLMSYSFlow
from kaggle_lmsys.models.embedding_flow_length import LengthFeatureEmbeddingLMSYSFlow
from kaggle_lmsys.models.classifier_lgbm_pipeline import LGBMClassifierPipeline
from kaggle_lmsys.models.classifier_lgbm_pipeline import LGBMClassifierCVBlendingPipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


SEED = 123
random_state = np.random.RandomState(SEED)
torch.manual_seed(0)
torch_gen = torch.Generator()
torch_gen.manual_seed(0)
numpy_gen = np.random.Generator(np.random.PCG64(seed=SEED))


@click.command()
@click.option("--hf_token", type=str, required=False)
@click.option("--for_test", type=bool, default=False, required=True)
@click.option("--for_test_pct", type=float, default=1.0, required=True)
@click.option("--config_path", type=str, required=True)
def go(
    hf_token: str,
    for_test: bool,
    for_test_pct: float,
    config_path: str,
) -> None:
    # login
    hf_login(hf_token)

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
    input_data_path = Path(data_config["train_data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(input_data_path, input_fields)
    if for_test:
        num_samples = int(for_test_pct * len(data))
        data = data.iloc[:num_samples, :]
    target_fields = [
        data_config["resp_a_win"],
        data_config["resp_b_win"],
        data_config["resp_tie"],
    ]
    add_target_field = data_config["added_target_field"]
    data = add_target(data, *target_fields, add_target_field)

    model_inputs: List[np.ndarray] = []

    # w2v embeddings fit/inference
    w2v_embedding_flow = W2VEmbeddingLMSYSFlow(pipeline_embedding_w2v_config)
    w2v_embeddings = w2v_embedding_flow.fit_and_inference(data)
    model_inputs.append(w2v_embeddings)

    # tfidf embeddings fit/inference
    tfidf_embedding_flow = TFIDFLMSYSFlow(pipeline_basic_embedding_tfidf)
    tfidf_embeddings = tfidf_embedding_flow.fit_and_inference(data)
    model_inputs.append(tfidf_embeddings)

    # length feature embeddings
    length_feature_embedding_flow = LengthFeatureEmbeddingLMSYSFlow(
        pipeline_embedding_length_feature
    )
    length_feature_embeddings = length_feature_embedding_flow.fit_and_inference(data)
    model_inputs.append(length_feature_embeddings)

    # deberta tokenizer embeddings fit/inference
    deberta_embedding_flow = DetertaEmbeddingLMSYSFlow(pipeline_embedding_deterba_config)
    deberta_embeddings = deberta_embedding_flow.fit_and_inference(data)
    model_inputs.append(deberta_embeddings)

    # classifier fit
    classifier = LGBMClassifierCVBlendingPipeline(classifier_lgbm_pipeline_config)
    model_data_ndarray = np.concatenate(model_inputs, axis=1)
    model_data_types = [DataType.NUM] * sum(model_input.shape[1] for model_input in model_inputs)
    model_data = ModelData(
        data_types=np.array(model_data_types),
        x=model_data_ndarray,
        y=data[add_target_field].values,
    )
    classifier.fit(model_data)
    classifier.save()


if __name__ == "__main__":
    # ! python /kaggle/working/go_lgbm_v2.py --for_test=False --for_test_pct=1.0 --config_path=
    go()
