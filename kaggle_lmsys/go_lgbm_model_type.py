import copy
import logging
import sys
import yaml
from pathlib import Path
from typing import List
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import click
import numpy as np
import pandas as pd
import torch
from huggingface_hub import login as hf_login

from kaggle_lmsys.train_utils import create_model_type_train_data
from kaggle_lmsys.utils import clean_data
from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.models.embedding_flow_word2vec import W2VEmbeddingBasicFlow
from kaggle_lmsys.models.embedding_flow_length import LengthFeatureEmbeddingBasicFlow
from kaggle_lmsys.models.embedding_flow_deterta import DetertaEmbeddingBasicFlow
from kaggle_lmsys.models.classifier_lgbm_pipeline import LGBMClassifierPipeline
from kaggle_lmsys.models.classifier_lgbm_pipeline import LGBMClassifierCVBlendingPipeline
from kaggle_lmsys.models.utils import merge_model_data

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
    pipeline_embedding_w2v_config = config["pipeline_basic_embedding_w2v"]
    pipeline_embedding_deterba_config = config["pipeline_basic_embedding_deberta"]
    pipeline_embedding_length_feature = config["pipeline_basic_embedding_length_feature"]
    classifier_lgbm_pipeline_config = config["model_type_classifier_lgbm_pipeline"]

    # data loading
    input_data_path = Path(data_config["train_data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(input_data_path, input_fields)
    data["model_a"] = data["model_a"].apply(lambda row: row.split("-")[0])
    data["model_b"] = data["model_b"].apply(lambda row: row.split("-")[0])
    if for_test:
        num_samples = int(for_test_pct * len(data))
        data = data.iloc[:num_samples, :]
    data, num_of_targets = create_model_type_train_data(
        data,
        output_prompt_field_name="prompt",
        output_response_field_name="response",
        output_model_type_field_name="labels",
    )
    classifier_lgbm_pipeline_config["lgbm"]["params"]["num_class"] = num_of_targets

    model_inputs: List[np.ndarray] = []
    # length feature embeddings
    length_feature_embedding_flow = LengthFeatureEmbeddingBasicFlow(
        pipeline_embedding_length_feature
    )
    length_feature_embeddings = length_feature_embedding_flow.fit_and_inference(data)
    model_inputs.append(length_feature_embeddings)
    # w2v embeddings fit/inference
    embedding_flow = W2VEmbeddingBasicFlow(pipeline_embedding_w2v_config)
    w2v_embeddings = embedding_flow.fit_and_inference(data)
    model_inputs.append(w2v_embeddings)
    # deberta tokenizer embeddings fit/inference
    embedding_flow = DetertaEmbeddingBasicFlow(pipeline_embedding_deterba_config)
    deberta_embeddings = embedding_flow.fit_and_inference(data)
    model_inputs.append(deberta_embeddings)

    # classifier fit
    classifier = LGBMClassifierCVBlendingPipeline(classifier_lgbm_pipeline_config)
    model_data = merge_model_data(model_inputs)
    model_data.y = data["labels"].values
    classifier.fit(model_data)
    classifier.save()

    for est_idx, feature_importances in classifier.get_feature_importance().items():
        feature_importance = pd.DataFrame(
            data=feature_importances, index=model_data.col_names, columns=["feature_importance"]
        )
        feature_importance.to_csv(
            f"/kaggle/working/model_output/fi_{est_idx}.csv", index_label="key"
        )


if __name__ == "__main__":
    # ! python /kaggle/working/go_lgbm_model_type.py --for_test=False --config_path=
    go()
