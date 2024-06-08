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
import torch
from huggingface_hub import login as hf_login

from kaggle_lmsys.train_utils import create_model_type_train_data
from kaggle_lmsys.utils import clean_data
from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.models.embedding_flow_word2vec import W2VEmbeddingBasicFlow
from kaggle_lmsys.models.embedding_flow_deterta import DetertaEmbeddingBasicFlow
from kaggle_lmsys.models.classifier_lgbm_pipeline import LGBMClassifierPipeline

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
@click.option("--config_path", type=str, required=True)
def go(
    for_test: bool,
    config_path: str,
    hf_token: str,
) -> None:
    # login
    hf_login(hf_token)

    # config stuff
    with open(Path(config_path), "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    data_config = config["data"]
    pipeline_embedding_w2v_config = config["pipeline_basic_embedding_w2v"]
    pipeline_embedding_deterba_config = config["pipeline_basic_embedding_deberta"]
    classifier_lgbm_pipeline_config = config["classifier_lgbm_pipeline"]

    # data loading
    input_data_path = Path(data_config["train_data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(input_data_path, input_fields)
    data["model_a"] = data["model_a"].apply(lambda row: row.split("-")[0])
    data["model_b"] = data["model_b"].apply(lambda row: row.split("-")[0])
    if for_test:
        data = data.iloc[:10000, :]
    data, num_of_targets = create_model_type_train_data(
        data,
        output_prompt_field_name="prompt",
        output_response_field_name="response",
        output_model_type_field_name="labels",
    )

    model_inputs_x: List[np.ndarray] = []
    # w2v embeddings fit/inference
    embedding_flow = W2VEmbeddingBasicFlow(pipeline_embedding_w2v_config)
    input_data = ModelData(
        x=data["response"].values,
        data_types=[DataType.TXT],
        col_names=["response"],
    )
    w2v_embeddings = embedding_flow.fit_and_inference(input_data)
    model_inputs_x.append(w2v_embeddings)
    # deberta tokenizer embeddings fit/inference
    embedding_flow = DetertaEmbeddingBasicFlow(pipeline_embedding_deterba_config)
    deberta_embeddings = embedding_flow.fit_and_inference(input_data)
    model_inputs_x.append(deberta_embeddings)

    # training setup
    model_inputs_x = np.concatenate(model_inputs_x, axis=1)
    model_inputs_y = data["labels"].values
    training_config = {"cv": 3}
    kfolds = StratifiedKFold(training_config["cv"], random_state=SEED, shuffle=True)
    for train_indices, test_indices in kfolds.split(np.ones(len(data)), model_inputs_y):
        train_x = model_inputs_x[train_indices, :]
        train_y = model_inputs_y[train_indices]
        test_x = model_inputs_x[test_indices, :]
        test_y = model_inputs_y[test_indices]

        classifier = LGBMClassifierPipeline(classifier_lgbm_pipeline_config)
        model_data_types = np.array([DataType.NUM] * train_x.shape[1])
        train_model_data = ModelData(
            data_types=model_data_types,
            x=train_x,
            y=train_y,
        )
        classifier.fit(train_model_data)
        classifier.save()

        test_model_data = ModelData(
            data_types=model_data_types,
            x=test_x,
        )
        pred_test_y = classifier.predict_proba(test_model_data)
        print(f">>>>>>>>>>>>>>>>>>> {pred_test_y.shape}")
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {log_loss(test_y, pred_test_y)}")


if __name__ == "__main__":
    # ! python /kaggle/working/go_lgbm_model_type.py --for_test=False --config_path=
    go()
