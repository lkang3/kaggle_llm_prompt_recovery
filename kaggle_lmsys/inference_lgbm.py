import copy
import logging
import sys
import pickle
import pandas as pd
from pathlib import Path

import click
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModel

from kaggle_lmsys.utils import clean_data
from kaggle_lmsys.utils import get_device
from kaggle_lmsys.utils import run_embedding_feature_engineering
from kaggle_lmsys.models.enum import DataType
from kaggle_lmsys.models.entities import ModelData
from kaggle_lmsys.models.model_type_classifier import predict_model_types


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
@click.option("--for_test", type=bool, default=False, required=True)
def go(for_test: bool) -> None:
    device = get_device()

    inference_config = {
        "data_path": "/kaggle/input/lmsys-chatbot-arena/test.csv",
        "tokenizer_path": "/kaggle/input/lmsys-go-lgbm-model-outputs/model_output/",
        "llm_model_path": "/kaggle/input/lmsys-go-lgbm-model-outputs/model_output/",
        "output_path": "/kaggle/working/submission.csv",
        "model_path": "/kaggle/input/lmsys-go-lgbm-model-outputs/lgbm_model.pkl",
        "model_type_tokenizer_path": "/kaggle/input/lmsys-go-deberta-model-type-model-outputs/model_type_classifier_output",
        "model_type_model_path": "/kaggle/input/lmsys-go-deberta-model-type-model-outputs/model_type_classifier_output",
    }
    data_config = {
        "prompt": "prompt",
        "resp_a": "response_a",
        "resp_b": "response_b",
        "resp_a_win": "winner_model_a",
        "resp_b_win": "winner_model_b",
        "resp_tie": "winner_tie",
        "added_target_field": "labels",
    }

    data_path = Path(inference_config["data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(data_path, input_fields)
    if for_test:
        data = data.iloc[:50, :]

    pred_model_types_args = {
        "tokenizer_path": inference_config["model_type_tokenizer_path"],
        "model_path": inference_config["model_type_model_path"],
        "data": data,
        "prompt_field_name": data_config["prompt"],
        "tokenization_max_length": 512,
    }
    model_types_args = copy.deepcopy(pred_model_types_args)
    model_types_args.update({"resp_field_name": data_config["resp_a"]})
    model_a_types = predict_model_types(**model_types_args)
    model_types_args = copy.deepcopy(pred_model_types_args)
    model_types_args.update({"resp_field_name": data_config["resp_b"]})
    model_b_types = predict_model_types(**model_types_args)
    data["model_a_type"] = model_a_types
    data["model_b_type"] = model_b_types

    tokenizer = AutoTokenizer.from_pretrained(inference_config["tokenizer_path"])
    llm_model = AutoModel.from_pretrained(inference_config["llm_model_path"])
    llm_model.to(device)
    data_embeddings = run_embedding_feature_engineering(
        data=data,
        tokenizer=tokenizer,
        prompt_field=data_config["prompt"],
        resp_a_field=data_config["resp_a"],
        resp_b_field=data_config["resp_b"],
        max_prompt_token_length=64,
        max_resp_a_token_length=255,
        max_resp_b_token_length=255,
        llm_model=llm_model,
        device=device,
        batch_size=10,
    )
    model = pickle.load(open(inference_config["model_path"], "rb"))
    model_data_ndarray = np.concatenate(
        (data_embeddings, data["model_a_type"].values.reshape(-1, 1), data["model_b_type"].values.reshape(-1, 1)),
        axis=1,
    )
    model_data_types = [DataType.NUM] * data_embeddings.shape[1]
    model_data_types.extend([DataType.CAT] * 2)
    model_data = ModelData(
        data_types=np.array(model_data_types),
        x=model_data_ndarray,
    )
    preds = model.predict_proba(model_data)
    outputs = pd.DataFrame(
        {
            "id": data["id"],
            "winner_model_a": preds[:, 0],
            "winner_model_b": preds[:, 1],
            "winner_tie": preds[:, 2],
        }
    )
    outputs.to_csv(inference_config["output_path"], index=False)


if __name__ == "__main__":
    # ! python /kaggle/working/inference_lgbm.py --for_test=False
    go()
