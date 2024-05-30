import copy
import logging
import sys
import pickle
from pathlib import Path

import click
import numpy as np
import torch
from huggingface_hub import login as hf_login
from transformers import AutoTokenizer
from transformers import AutoModel

from train_utils import add_target
from kaggle_lmsys.utils import clean_data
from kaggle_lmsys.utils import get_device
from kaggle_lmsys.utils import run_embedding_feature_engineering
from kaggle_lmsys.models.lgbm_classifier import LGBMBasedClassifier
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


hf_token: str = "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"


@click.command()
@click.option("--for_test", type=bool, default=False, required=True)
def go(for_test: bool) -> None:
    device = get_device()

    data_config = {
        "train_data_path": "/kaggle/input/lmsys-chatbot-arena/train.csv",
        "test_data_path": "/kaggle/input/lmsys-chatbot-arena/test.csv",
        "prompt": "prompt",
        "resp_a": "response_a",
        "resp_b": "response_b",
        "resp_a_win": "winner_model_a",
        "resp_b_win": "winner_model_b",
        "resp_tie": "winner_tie",
        "added_target_field": "labels",
    }
    llm_model_config = {
        "model_name": "microsoft/deberta-base",
        "max_length": 256,
        "train_batch_size": 4,
        "output_path": "model_output",
    }
    model_config = {
        "train_pct": 0.8,
        "eval_pct": 0.5,
        "cv": 4,
        "num_epoch": 1,
        "output_path": "/kaggle/working/lgbm_model.pkl",
        "learning_rate": 0.03,
        "num_labels": 3,
        "stopping_rounds": 50,
        "model_type_tokenizer_path": "/kaggle/input/lmsys-go-deberta-model-type-model-outputs/model_type_classifier_output",
        "model_type_model_path": "/kaggle/input/lmsys-go-deberta-model-type-model-outputs/model_type_classifier_output",
    }

    data_path = Path(data_config["train_data_path"])
    input_fields = [data_config["prompt"], data_config["resp_a"], data_config["resp_b"]]
    data = clean_data(data_path, input_fields)
    if for_test:
        data = data.iloc[:100, :]
    target_fields = [
        data_config["resp_a_win"],
        data_config["resp_b_win"],
        data_config["resp_tie"],
    ]
    add_target_field = data_config["added_target_field"]
    data = add_target(data, *target_fields, add_target_field)

    # add predicted model types
    pred_model_types_args = {
        "tokenizer_path": model_config["model_type_tokenizer_path"],
        "model_path": model_config["model_type_model_path"],
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

    tokenizer = AutoTokenizer.from_pretrained(llm_model_config["model_name"], token=hf_token)
    llm_model = AutoModel.from_pretrained(llm_model_config["model_name"], token=hf_token)
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

    # Create the model
    model_params = {
        'n_estimators': 1000,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multiclass',
        'num_class': model_config["num_labels"],
        'metric': 'multi_logloss',
        'random_state': SEED,
        'learning_rate': model_config["learning_rate"],
    }
    model = LGBMBasedClassifier(
        estimator_params=model_params,
        eval_pct=model_config["eval_pct"],
        seed=SEED,
    )
    model_data_ndarray = np.concatenate(
        (data_embeddings, data["model_a_type"].values.reshape(-1, 1), data["model_b_type"].values.reshape(-1, 1)),
        axis=1,
    )
    model_data_types = [DataType.NUM] * data_embeddings.shape[1]
    model_data_types.extend([DataType.CAT] * 2)
    model_data = ModelData(
        data_types=np.array(model_data_types),
        x=model_data_ndarray,
        y=data[add_target_field].values,
    )
    model.fit(model_data)
    pickle.dump(model, open(model_config['output_path'], "wb"))
    tokenizer.save_pretrained(llm_model_config["output_path"])
    llm_model.save_pretrained(llm_model_config["output_path"])


if __name__ == "__main__":
    # ! python /kaggle/working/go_lgbm.py --for_test=False
    hf_login(hf_token)
    go()
